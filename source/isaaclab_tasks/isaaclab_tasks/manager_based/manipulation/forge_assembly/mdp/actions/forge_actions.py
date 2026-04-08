# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forge assembly action term: asset-relative Jacobian transpose control with EMA smoothing.

Replicates the direct/forge control chain within the manager-based framework:
- 7D action space: [pos_xyz, rot_xyz, success_pred]
- Asset-relative action frame (targets relative to hole position)
- EMA smoothing on actions
- Dead zone randomization
- Jacobian transpose control with nullspace optimization
- Gain randomization
- Position/rotation clipping with thresholds
"""

from __future__ import annotations

import math

import torch
import numpy as np

import isaacsim.core.utils.torch as torch_utils
from isaaclab.utils.math import axis_angle_from_quat

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm


class ForgeAssemblyAction(ActionTerm):
    """Forge assembly action term using asset-relative Jacobian transpose control.

    Action space is 7D:
        [0:3] - Position delta relative to hole (scaled by pos_action_bounds)
        [3:6] - Rotation (only yaw is used, roll/pitch zeroed)
        [6]   - Success prediction (rescaled from [-1,1] to [0,1])

    The action processing chain (matching direct forge):
    1. Scale actions to allowed range
    2. Compute absolute target pose from hole position + action offset
    3. Clip position/rotation targets by randomized thresholds
    4. EMA-smooth the target with previous step
    5. Compute Jacobian transpose torque with dead zone
    6. Apply nullspace optimization towards default joint config
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._robot: Articulation = self._asset

        # Resolve joint IDs from regex patterns
        self._joint_ids, self._joint_names = self._robot.find_joints(self.cfg.joint_names)
        self._num_arm_joints = len(self._joint_ids)

        # Resolve gripper joint IDs if configured
        if self.cfg.gripper_joint_names is not None:
            self._gripper_joint_ids, self._gripper_joint_names = self._robot.find_joints(
                self.cfg.gripper_joint_names
            )
            self._has_gripper = len(self._gripper_joint_ids) > 0
        else:
            self._gripper_joint_ids = []
            self._gripper_joint_names = []
            self._has_gripper = False

        # Resolve EE body index
        body_ids, _ = self._robot.find_bodies(self.cfg.body_name)
        self._body_idx = body_ids[0]

        # Action dimensions
        self._raw_actions = torch.zeros(self.num_envs, 7, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 7, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 7, device=self.device)

        # EMA state
        self._ema_factor = torch.zeros(self.num_envs, 1, device=self.device)
        self._ema_smoothed_actions = torch.zeros(self.num_envs, 7, device=self.device)

        # Default controller parameters (from config)
        self._default_gains = torch.tensor(
            self.cfg.default_task_prop_gains, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self._default_pos_threshold = torch.tensor(
            self.cfg.pos_action_threshold, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self._default_rot_threshold = torch.tensor(
            self.cfg.rot_action_threshold, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self._default_dead_zone = torch.tensor(
            self.cfg.default_dead_zone, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        # Randomized parameters (synced from event terms on reset)
        self._task_prop_gains = self._default_gains.clone()
        self._task_deriv_gains = torch.zeros(self.num_envs, 6, device=self.device)
        self._pos_threshold = self._default_pos_threshold.clone()
        self._rot_threshold = self._default_rot_threshold.clone()
        self._dead_zone = torch.zeros(self.num_envs, 6, device=self.device)
        self._contact_penalty_threshold = torch.ones(self.num_envs, device=self.device) * 7.5

        # Default DOF positions for nullspace
        self._default_dof_pos = torch.tensor(
            self.cfg.default_dof_pos_tensor, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        # Controller constants
        self._kp_null = cfg.kp_null
        self._kd_null = cfg.kd_null
        self._torque_limit = cfg.torque_limit

        # Store for observation/reward access
        self.delta_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.delta_yaw = torch.zeros(self.num_envs, 1, device=self.device)

        # Flip quaternions for observation noise
        self._flip_quats = torch.ones(self.num_envs, device=self.device)

        # Peg is a free rigid body (not teleported).
        # In direct forge, the peg is placed at EE + offset on reset, then
        # held by gripper friction. We track peg for observation/reward access only.

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Store raw actions and apply EMA smoothing.

        Direct forge: alpha * raw + (1-alpha) * prev
        """
        self._raw_actions[:] = actions
        self._ema_smoothed_actions = (
            self._ema_factor * self._raw_actions
            + (1.0 - self._ema_factor) * self._prev_actions
        )
        self._processed_actions[:] = self._ema_smoothed_actions

    def apply_actions(self):
        """Apply asset-relative actions through Jacobian transpose control.

        Replicates forge_env._apply_action() and factory_control.compute_dof_torque().
        """
        # Get current EE state
        ee_pos = self._robot.data.body_pos_w[:, self._body_idx]
        ee_quat = self._robot.data.body_quat_w[:, self._body_idx]

        # Get hole position (asset-relative frame)
        hole_pos = self._get_hole_pos()

        # Get joint positions and velocities
        dof_pos = self._robot.data.joint_pos[:, self._joint_ids]
        dof_vel = self._robot.data.joint_vel[:, self._joint_ids]
        ee_linvel = self._robot.data.body_lin_vel_w[:, self._body_idx]
        ee_angvel = self._robot.data.body_ang_vel_w[:, self._body_idx]

        # Step 0: Scale actions to allowed range
        pos_actions = self._processed_actions[:, 0:3] * torch.tensor(
            self.cfg.pos_action_bounds, device=self.device
        )
        rot_actions = self._processed_actions[:, 3:6].clone()

        # Step 1: Compute desired target in world frame
        ctrl_target_pos = hole_pos + pos_actions

        # Step 1b: Enforce rotation constraints (only yaw)
        rot_actions[:, 0:2] = 0.0
        # Map yaw from [-1,1] to joint limit range [-180, 90] degrees
        yaw_action = rot_actions[:, 2]
        ctrl_yaw = np.deg2rad(-180.0) + np.deg2rad(270.0) * (yaw_action + 1.0) / 2.0
        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0],
            pitch=rot_actions[:, 1],
            yaw=ctrl_yaw,
        )
        # Apply bolt-to-EE rotation offset (180 deg around x)
        rot_180 = torch.tensor([np.pi, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180[:, 0], pitch=rot_180[:, 1], yaw=rot_180[:, 2]
        )
        ctrl_target_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        # Step 2: Clip targets by thresholds
        # 2a: Position clipping
        self.delta_pos = ctrl_target_pos - ee_pos
        pos_error_clipped = torch.clip(self.delta_pos, -self._pos_threshold, self._pos_threshold)
        ctrl_target_pos_clipped = ee_pos + pos_error_clipped

        # 2b: Rotation clipping via Euler angles
        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(ee_quat)
        des_roll, des_pitch, des_yaw = torch_utils.get_euler_xyz(ctrl_target_quat)

        # Wrap yaw to avoid joint limit
        curr_yaw = self._wrap_yaw(curr_yaw)
        des_yaw = self._wrap_yaw(des_yaw)

        self.delta_yaw = (des_yaw - curr_yaw).unsqueeze(-1)
        clipped_yaw = torch.clip(
            self.delta_yaw.squeeze(-1), -self._rot_threshold[:, 2], self._rot_threshold[:, 2]
        )

        # Clip roll and pitch
        des_roll = torch.where(des_roll < 0.0, des_roll + 2 * math.pi, des_roll)
        delta_roll = des_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self._rot_threshold[:, 0], self._rot_threshold[:, 0])

        curr_pitch = torch.where(curr_pitch > math.pi, curr_pitch - 2 * math.pi, curr_pitch)
        des_pitch = torch.where(des_pitch > math.pi, des_pitch - 2 * math.pi, des_pitch)
        delta_pitch = des_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self._rot_threshold[:, 1], self._rot_threshold[:, 1])

        ctrl_target_quat_clipped = torch_utils.quat_from_euler_xyz(
            roll=curr_roll + clipped_roll,
            pitch=curr_pitch + clipped_pitch,
            yaw=curr_yaw + clipped_yaw,
        )

        # Step 3: Compute pose error for Jacobian transpose
        pos_error = ctrl_target_pos_clipped - ee_pos

        # Shortest path quaternion error
        quat_dot = (ctrl_target_quat_clipped * ee_quat).sum(dim=1, keepdim=True)
        ctrl_target_quat_clipped = torch.where(
            quat_dot.expand(-1, 4) >= 0, ctrl_target_quat_clipped, -ctrl_target_quat_clipped
        )
        ee_quat_norm = torch_utils.quat_mul(ee_quat, torch_utils.quat_conjugate(ee_quat))[:, 0]
        ee_quat_inv = torch_utils.quat_conjugate(ee_quat) / ee_quat_norm.unsqueeze(-1)
        quat_error = torch_utils.quat_mul(ctrl_target_quat_clipped, ee_quat_inv)
        axis_angle_error = axis_angle_from_quat(quat_error)

        delta_pose = torch.cat((pos_error, axis_angle_error), dim=1)  # (N, 6)

        # Step 4: Apply task-space PD gains
        task_wrench = torch.zeros_like(delta_pose)
        task_wrench[:, 0:3] = (
            self._task_prop_gains[:, 0:3] * delta_pose[:, 0:3]
            + self._task_deriv_gains[:, 0:3] * (-ee_linvel)
        )
        task_wrench[:, 3:6] = (
            self._task_prop_gains[:, 3:6] * delta_pose[:, 3:6]
            + self._task_deriv_gains[:, 3:6] * (-ee_angvel)
        )

        # Step 5: Apply dead zone
        task_wrench = torch.where(
            task_wrench.abs() < self._dead_zone,
            torch.zeros_like(task_wrench),
            task_wrench.sign() * (task_wrench.abs() - self._dead_zone),
        )

        # Step 6: Map to joint space via Jacobian transpose
        ee_jacobian = self._get_ee_jacobian()
        jacobian_T = ee_jacobian.transpose(1, 2)
        dof_torque = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

        # Step 7: Nullspace optimization
        arm_mass_matrix = self._get_mass_matrix_subset()
        arm_mass_inv = torch.inverse(arm_mass_matrix)
        mass_matrix_task = torch.inverse(ee_jacobian @ arm_mass_inv @ jacobian_T)
        j_eef_inv = mass_matrix_task @ ee_jacobian @ arm_mass_inv

        dist_to_default = self._default_dof_pos - dof_pos
        dist_to_default = (dist_to_default + math.pi) % (2 * math.pi) - math.pi

        u_null = self._kd_null * (-dof_vel) + self._kp_null * dist_to_default
        u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
        torque_null = (
            torch.eye(self._num_arm_joints, device=self.device).unsqueeze(0)
            - jacobian_T @ j_eef_inv
        ) @ u_null
        dof_torque += torque_null.squeeze(-1)

        # Clamp torques
        dof_torque = torch.clamp(dof_torque, -self._torque_limit, self._torque_limit)

        # Apply to robot
        self._robot.set_joint_effort_target(dof_torque, joint_ids=self._joint_ids)

        # Apply gripper target position if configured (matching direct forge ctrl_target_gripper_dof_pos=0.0)
        if self._has_gripper:
            self._robot.set_joint_position_target(
                self.cfg.gripper_dof_pos_target, joint_ids=self._gripper_joint_ids
            )

        # --- Peg tracking removed ---
        # In direct forge, the peg is a free rigid body held by gripper friction.
        # It is NOT teleported to EE every step. Physics handles the grip.
        # The peg is placed at EE + offset on reset (via reset_peg_to_ee event)
        # and then the gripper closes around it for 0.25s to establish friction.

        # Store prev actions
        self._prev_actions[:] = self._processed_actions

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset action state and sync controller parameters from event randomization.

        Event terms (randomize_task_gains, randomize_pos_threshold, etc.) set
        env._task_prop_gains, env._pos_threshold, etc. We sync those to the
        action term's internal state here.
        """
        # Handle slice (from action_manager when env_ids=None)
        if isinstance(env_ids, slice) or env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        n = len(env_ids)

        # Sync randomized parameters from event terms (set on env by events)
        self._sync_env_params(env_ids)

        # Reset EMA factor from env (set by randomize_ema_factor event)
        if hasattr(self._env, "_ema_factor"):
            self._ema_factor[env_ids] = self._env._ema_factor[env_ids]
        else:
            ema_low, ema_high = self.cfg.ema_factor_range
            self._ema_factor[env_ids] = ema_low + torch.rand((n, 1), device=self.device) * (ema_high - ema_low)

        # Reset actions to current EE state relative to hole tip (matching direct forge line 276-281)
        # Direct forge: fixed_pos_action_frame = fixed_pos_obs_frame + init_fixed_pos_obs_noise
        # where fixed_pos_obs_frame = hole_tip_pos (hole root + height)
        hole_pos = self._get_hole_pos()
        hole_tip_pos = hole_pos.clone()
        hole_tip_pos[:, 2] += 0.025  # Peg/hole height (matches direct forge factory_hole_8mm.usd)

        # Add observation noise for robustness (direct forge: init_fixed_pos_obs_noise ~ N(0, 0.001))
        init_fixed_pos_obs_noise = torch.randn((n, 3), device=self.device) * 0.001
        fixed_pos_action_frame = hole_tip_pos[env_ids] + init_fixed_pos_obs_noise

        ee_pos = self._robot.data.body_pos_w[env_ids, self._body_idx]
        pos_actions = ee_pos - fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.pos_action_bounds, device=self.device)
        self._raw_actions[env_ids, 0:3] = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self._prev_actions[env_ids, 0:3] = self._raw_actions[env_ids, 0:3]
        self._ema_smoothed_actions[env_ids] = 0.0

        # Reset yaw action to current EE yaw relative to bolt
        unrot_180 = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(n, 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(
            roll=unrot_180[:, 0], pitch=unrot_180[:, 1], yaw=unrot_180[:, 2]
        )
        ee_quat = self._robot.data.body_quat_w[env_ids, self._body_idx]
        fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, ee_quat)
        fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
        )
        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self._raw_actions[env_ids, 5] = yaw_action
        self._prev_actions[env_ids, 5] = yaw_action

        # Reset success pred
        self._raw_actions[env_ids, 6] = -1.0
        self._prev_actions[env_ids, 6] = -1.0

        # Randomize flip quaternions
        self._flip_quats[env_ids] = 1.0
        rand_flips = torch.rand(n, device=self.device) > 0.5
        self._flip_quats[env_ids[rand_flips]] = -1.0

        # Reset force sensor smoothing buffer to match direct forge
        # Direct forge zeroes force_sensor_world_smooth in _reset_idx (line 331)
        # This prevents contact penalty from firing at initialization
        if hasattr(self._env, "_force_smooth_norm"):
            self._env._force_smooth_norm[env_ids] = 0.0

    def _sync_env_params(self, env_ids: torch.Tensor):
        """Sync randomized parameters from event terms on env to action term."""
        # Task gains
        if hasattr(self._env, "_task_prop_gains"):
            self._task_prop_gains[env_ids] = self._env._task_prop_gains[env_ids]
            self._task_deriv_gains[env_ids] = 2.0 * torch.sqrt(self._task_prop_gains[env_ids])
        # Position threshold
        if hasattr(self._env, "_pos_threshold"):
            self._pos_threshold[env_ids] = self._env._pos_threshold[env_ids]
        # Rotation threshold
        if hasattr(self._env, "_rot_threshold"):
            self._rot_threshold[env_ids] = self._env._rot_threshold[env_ids]
        # Dead zone
        if hasattr(self._env, "_dead_zone"):
            self._dead_zone[env_ids] = self._env._dead_zone[env_ids]
        # Contact penalty threshold
        if hasattr(self._env, "_contact_penalty_threshold"):
            self._contact_penalty_threshold[env_ids] = self._env._contact_penalty_threshold[env_ids].squeeze(-1)

    def _get_ee_jacobian(self) -> torch.Tensor:
        """Get Jacobian averaged from left/right finger bodies (matching direct forge).

        Direct forge computes:
          self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
          self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
          self.fingertip_midpoint_jacobian = (left + right) * 0.5
        """
        # jacobians shape: (num_envs, num_bodies, 6, num_dofs)
        jacobians = self._robot.root_physx_view.get_jacobians()

        # Try to get finger body indices for averaged Jacobian (direct forge pattern)
        try:
            left_finger_idx = self._robot.body_names.index("panda_leftfinger")
            right_finger_idx = self._robot.body_names.index("panda_rightfinger")

            # Extract finger Jacobians (note: direct forge uses idx - 1)
            left_jacobian = jacobians[:, left_finger_idx - 1, :, 0:7]
            right_jacobian = jacobians[:, right_finger_idx - 1, :, 0:7]

            # Average left and right finger Jacobians (direct forge pattern)
            return (left_jacobian + right_jacobian) * 0.5
        except ValueError:
            # Fallback: use configured body name if finger names not found
            jacobian = jacobians[:, self._body_idx, :, 0:7]
            return jacobian

    def _get_mass_matrix_subset(self) -> torch.Tensor:
        """Get mass matrix for arm joints only from PhysX.

        Matches direct forge's arm_mass_matrix = root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7].
        """
        # Get full generalized mass matrix from PhysX
        # Shape: (num_envs, num_dofs, num_dofs)
        mass_matrices = self._robot.root_physx_view.get_generalized_mass_matrices()

        # Extract arm joint submatrix (first 7 joints for Panda)
        # Direct forge uses [:, 0:7, 0:7] for 7-DOF arm
        if len(self._joint_ids) == 7 and self._joint_ids[0] == 0 and self._joint_ids[-1] == 6:
            # Direct slice for contiguous [0, 1, 2, 3, 4, 5, 6]
            return mass_matrices[:, 0:7, 0:7]
        else:
            # Fallback: index_select for non-contiguous joint_ids
            joint_indices = torch.tensor(self._joint_ids, device=self.device)
            batch_size = mass_matrices.shape[0]
            mass_subset = mass_matrices[:, joint_indices, :][:, :, joint_indices]
            return mass_subset

    def _get_hole_pos(self) -> torch.Tensor:
        """Get hole position from scene."""
        hole: Articulation = self._env.scene["hole"]
        return hole.data.root_pos_w

    @staticmethod
    def _wrap_yaw(yaw: torch.Tensor) -> torch.Tensor:
        """Wrap yaw to [-125, 235] degrees range to avoid joint limit."""
        yaw = yaw.clone()
        deg125 = np.deg2rad(125.0)
        deg235 = np.deg2rad(235.0)
        yaw = torch.where(yaw < -deg125, yaw + 2 * math.pi, yaw)
        yaw = torch.where(yaw > deg235, yaw - 2 * math.pi, yaw)
        return yaw

    @staticmethod
    def _get_random_prop_gains(default_values: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Randomize gains matching direct forge's get_random_prop_gains.

        Creates multiplicative noise: multiplier = 1 + rand * noise_level
        Then randomly inverts: multiplier = 1/multiplier with 50% probability.
        """
        n = default_values.shape[0]
        d = default_values.shape[1]
        device = default_values.device

        noise = torch.rand((n, d), dtype=torch.float32, device=device)
        noise_diag = torch.diag(torch.tensor([noise_level] * d, dtype=torch.float32, device=device))
        c_param_noise = noise @ noise_diag
        multiplier = 1.0 + c_param_noise

        decrease_flag = torch.rand((n, d), dtype=torch.float32, device=device) > 0.5
        multiplier = torch.where(decrease_flag, 1.0 / multiplier, multiplier)

        return default_values * multiplier
