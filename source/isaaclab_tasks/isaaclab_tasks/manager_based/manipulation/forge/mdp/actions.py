# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ForgeImpedanceAction: Impedance control action term for FORGE tasks.

Ports the direct version's impedance controller into a manager-based ActionTerm.
Action space is 7D:
    [0:3] - position delta relative to fixed asset (scaled by pos_action_bounds)
    [3:6] - rotation delta (only yaw used, roll/pitch zeroed)
    [6]   - success prediction (stored but not used for control)

The action processing pipeline per step:
    1. EMA smoothing with randomized alpha
    2. Scale actions by bounds, compute target EE pose relative to fixed asset
    3. Clip to randomized thresholds
    4. Impedance control: F = Kp * err - Kd * vel
    5. Dead zone filtering
    6. J^T * F + null-space torque -> clamp to +/-100Nm
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat


class ForgeImpedanceAction(ActionTerm):
    """Impedance control action term matching direct FORGE behavior."""

    cfg: ForgeImpedanceActionCfg

    def __init__(self, cfg: ForgeImpedanceActionCfg, env):
        super().__init__(cfg, env)
        self._asset: Articulation = self._env.scene[cfg.asset_name]

        # Resolve joint IDs for the arm
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=False
        )
        self._num_arm_joints = len(self._joint_ids)

        # Body indices for EE tracking and Jacobian
        self._fingertip_body_idx = self._asset.body_names.index(cfg.body_name)
        self._left_finger_body_idx = self._asset.body_names.index(cfg.finger_body_names[0])
        self._right_finger_body_idx = self._asset.body_names.index(cfg.finger_body_names[1])

        # Force sensor body (if available)
        self._force_sensor_body_idx = None
        if "force_sensor" in self._asset.body_names:
            self._force_sensor_body_idx = self._asset.body_names.index("force_sensor")
        else:
            # Fallback to hand body
            self._force_sensor_body_idx = self._asset.body_names.index(cfg.body_name)

        # Fixed asset reference
        self._fixed_asset: Articulation = self._env.scene[cfg.fixed_asset_name]

        # ---- Action dimensions: 7D ----
        self._action_dim = 7
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # ---- Default impedance gains and thresholds ----
        default_gains = torch.tensor(cfg.default_task_prop_gains, device=self.device)
        self.default_gains = default_gains.unsqueeze(0).repeat(self.num_envs, 1)

        default_pos_threshold = torch.tensor(cfg.pos_action_threshold, device=self.device)
        self.default_pos_threshold = default_pos_threshold.unsqueeze(0).repeat(self.num_envs, 1)

        default_rot_threshold = torch.tensor(cfg.rot_action_threshold, device=self.device)
        self.default_rot_threshold = default_rot_threshold.unsqueeze(0).repeat(self.num_envs, 1)

        default_dead_zone = torch.tensor(cfg.default_dead_zone, device=self.device)
        self.default_dead_zone = default_dead_zone.unsqueeze(0).repeat(self.num_envs, 1)

        # Current randomized values (initialized to defaults)
        self.task_prop_gains = self.default_gains.clone()
        self.task_deriv_gains = self._get_deriv_gains(self.task_prop_gains)
        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()
        self.dead_zone_thresholds = torch.zeros((self.num_envs, 6), device=self.device)
        self.ema_factor = torch.full((self.num_envs, 1), cfg.ema_factor_default, device=self.device)

        # Action bounds
        self.pos_action_bounds = torch.tensor(cfg.pos_action_bounds, device=self.device)
        self.rot_action_bounds = torch.tensor(cfg.rot_action_bounds, device=self.device)

        # Null-space parameters
        self.default_dof_pos = torch.tensor(cfg.default_dof_pos_tensor, device=self.device)
        self.kp_null = cfg.kp_null
        self.kd_null = cfg.kd_null

        # Previous actions for EMA and action rate penalty
        self.prev_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)

        # ---- Force/torque sensor ----
        self.force_sensor_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_sensor_world_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.ft_smoothing_factor = cfg.ft_smoothing_factor
        self.noisy_force = torch.zeros((self.num_envs, 3), device=self.device)

        # Contact penalty thresholds (randomized by events)
        self.contact_penalty_thresholds = torch.full(
            (self.num_envs,),
            (cfg.contact_threshold_range[0] + cfg.contact_threshold_range[1]) / 2.0,
            device=self.device,
        )

        # ---- Fixed asset observation frame (set during reset) ----
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # ---- Noisy fingertip state (for observations) ----
        self.noisy_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.noisy_fingertip_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.prev_noisy_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_noisy_fingertip_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        self.ee_linvel_fd = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_angvel_fd = torch.zeros((self.num_envs, 3), device=self.device)

        # Flip quaternions for observation noise
        self.flip_quats = torch.ones((self.num_envs,), device=self.device)

        # Noise levels from config
        self.fingertip_pos_noise = cfg.fingertip_pos_noise
        self.fingertip_rot_noise_deg = cfg.fingertip_rot_noise_deg
        self.ft_force_noise = cfg.ft_force_noise

        # Store delta_pos and delta_yaw for reward computation
        self.delta_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.delta_yaw = torch.zeros((self.num_envs,), device=self.device)

        # Track if intermediate values have been computed this env step
        self._intermediate_computed_this_step = False

    # ---- Abstract property implementations ----

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # ---- Core methods ----

    def process_actions(self, actions: torch.Tensor):
        """Apply EMA smoothing to raw actions."""
        self._raw_actions[:] = actions
        self._processed_actions = (
            self.ema_factor * actions + (1.0 - self.ema_factor) * self.prev_actions
        )
        self._intermediate_computed_this_step = False

    def apply_actions(self):
        """Compute impedance torques and apply to robot.

        Called each simulation step (decimation times per env step).
        """
        # Compute intermediate values (noisy state, forces, Jacobian, etc.)
        self._compute_intermediate_values()
        self._intermediate_computed_this_step = True

        # Update prev_actions AFTER EMA computation (for action rate penalty)
        self.prev_actions = self._processed_actions.clone()

        # Step (0): Scale actions to allowed range
        pos_actions = self._processed_actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(self.pos_action_bounds)

        rot_actions = self._processed_actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(self.rot_action_bounds)

        # Step (1): Compute desired pose targets in EE frame
        # (1.a) Position - action frame is the top of the bolt (noisy estimate)
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions

        # (1.b) Enforce rotation action constraints
        rot_actions[:, 0:2] = 0.0  # Zero roll and pitch

        # Map yaw to joint limit range [-180, 90] degrees
        rot_actions[:, 2] = (
            np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0
        )

        # (1.c) Get desired orientation target
        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )

        rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )
        ctrl_target_fingertip_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        # Step (2): Clip targets if they are too far from current EE pose
        # (2.a): Clip position targets
        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self._fingertip_midpoint_pos
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_fingertip_midpoint_pos = self._fingertip_midpoint_pos + pos_error_clipped

        # (2.b) Clip orientation targets
        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self._fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(
            ctrl_target_fingertip_preclipped_quat
        )
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        # (2.b.ii) Correct yaw direction to avoid joint limit
        curr_yaw = _wrap_yaw(curr_yaw)
        desired_yaw = _wrap_yaw(desired_yaw)

        # (2.b.iii) Clip yaw
        self.delta_yaw = desired_yaw - curr_yaw
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        # (2.b.iv) Clip roll and pitch
        desired_roll = torch.where(
            desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll
        )
        desired_pitch = torch.where(
            desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch
        )
        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll

        curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
        desired_pitch = torch.where(
            desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch
        )
        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
        desired_xyz[:, 1] = curr_pitch + clipped_pitch

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
        )

        # Step (3): Compute impedance torques
        dof_torque = self._compute_dof_torque(
            ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat,
        )

        # Apply torques to arm joints
        full_torque = torch.zeros((self.num_envs, self._asset.num_joints), device=self.device)
        full_torque[:, self._joint_ids] = dof_torque[:, :self._num_arm_joints]
        self._asset.set_joint_effort_target(full_torque)

        # Set gripper position target (closed), matching direct: set_joint_position_target with gripper=0.0
        gripper_target = torch.zeros((self.num_envs, self._asset.num_joints), device=self.device)
        self._asset.set_joint_position_target(gripper_target)

    def _compute_intermediate_values(self):
        """Compute noisy fingertip state, forces, Jacobian, mass matrix.

        Matches ForgeEnv._compute_intermediate_values() from direct version.
        """
        dt = self._env.step_dt

        # Get raw fingertip state
        self._fingertip_midpoint_pos = (
            self._asset.data.body_pos_w[:, self._fingertip_body_idx] - self._env.scene.env_origins
        )
        self._fingertip_midpoint_quat = self._asset.data.body_quat_w[:, self._fingertip_body_idx]
        self._fingertip_midpoint_linvel = self._asset.data.body_lin_vel_w[:, self._fingertip_body_idx]
        self._fingertip_midpoint_angvel = self._asset.data.body_ang_vel_w[:, self._fingertip_body_idx]

        # Jacobian (average of left/right finger Jacobians)
        jacobians = self._asset.root_physx_view.get_jacobians()
        left_jac = jacobians[:, self._left_finger_body_idx - 1, 0:6, 0:7]
        right_jac = jacobians[:, self._right_finger_body_idx - 1, 0:6, 0:7]
        self._fingertip_jacobian = (left_jac + right_jac) * 0.5

        # Mass matrix (no regularization, matching direct version)
        self._arm_mass_matrix = self._asset.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]

        # ---- Add noise to fingertip observations ----
        # Position noise
        pos_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        pos_noise = pos_noise * torch.tensor(
            [self.fingertip_pos_noise] * 3, dtype=torch.float32, device=self.device
        )
        self.noisy_fingertip_pos = self._fingertip_midpoint_pos + pos_noise

        # Rotation noise
        rot_noise_axis = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        rot_noise_axis /= torch.linalg.norm(rot_noise_axis, dim=1, keepdim=True)
        rot_noise_angle = (
            torch.randn((self.num_envs,), dtype=torch.float32, device=self.device)
            * np.deg2rad(self.fingertip_rot_noise_deg)
        )
        self.noisy_fingertip_quat = torch_utils.quat_mul(
            self._fingertip_midpoint_quat,
            torch_utils.quat_from_angle_axis(rot_noise_angle, rot_noise_axis),
        )
        self.noisy_fingertip_quat[:, [0, 3]] = 0.0
        self.noisy_fingertip_quat = self.noisy_fingertip_quat * self.flip_quats.unsqueeze(-1)

        # Finite-difference velocities from noisy positions
        self.ee_linvel_fd = (self.noisy_fingertip_pos - self.prev_noisy_fingertip_pos) / dt
        self.prev_noisy_fingertip_pos = self.noisy_fingertip_pos.clone()

        rot_diff_quat = torch_utils.quat_mul(
            self.noisy_fingertip_quat,
            torch_utils.quat_conjugate(self.prev_noisy_fingertip_quat),
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.ee_angvel_fd[:, 0:2] = 0.0
        self.prev_noisy_fingertip_quat = self.noisy_fingertip_quat.clone()

        # ---- Force sensor ----
        force_sensor_world = self._asset.root_physx_view.get_link_incoming_joint_force()[
            :, self._force_sensor_body_idx
        ]
        alpha = self.ft_smoothing_factor
        self.force_sensor_world_smooth = (
            alpha * force_sensor_world + (1.0 - alpha) * self.force_sensor_world_smooth
        )

        # Transform to fixed asset frame (identity transform for now, matching direct)
        identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        self.force_sensor_smooth[:, :3] = torch_utils.quat_apply(
            identity_quat, self.force_sensor_world_smooth[:, 0:3]
        )
        self.force_sensor_smooth[:, 3:6] = torch_utils.quat_apply(
            identity_quat, self.force_sensor_world_smooth[:, 3:6]
        )

        # Noisy force for observations
        force_noise = torch.randn((self.num_envs, 3), device=self.device) * self.ft_force_noise
        self.noisy_force = self.force_sensor_smooth[:, 0:3] + force_noise

    def _compute_dof_torque(self, target_pos, target_quat):
        """Compute impedance torques matching factory_control.compute_dof_torque()."""
        # Pose error
        pos_error, axis_angle_error = self._get_pose_error(
            self._fingertip_midpoint_pos,
            self._fingertip_midpoint_quat,
            target_pos,
            target_quat,
        )
        delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

        # Task wrench: F = Kp * err - Kd * vel
        task_wrench = torch.zeros_like(delta_fingertip_pose)
        task_wrench[:, 0:3] = (
            self.task_prop_gains[:, 0:3] * delta_fingertip_pose[:, 0:3]
            + self.task_deriv_gains[:, 0:3] * (0.0 - self._fingertip_midpoint_linvel)
        )
        task_wrench[:, 3:6] = (
            self.task_prop_gains[:, 3:6] * delta_fingertip_pose[:, 3:6]
            + self.task_deriv_gains[:, 3:6] * (0.0 - self._fingertip_midpoint_angvel)
        )

        # Dead zone
        if self.dead_zone_thresholds is not None:
            task_wrench = torch.where(
                task_wrench.abs() < self.dead_zone_thresholds,
                torch.zeros_like(task_wrench),
                task_wrench.sign() * (task_wrench.abs() - self.dead_zone_thresholds),
            )

        # tau = J^T * F
        jacobian_T = self._fingertip_jacobian.transpose(1, 2)
        dof_torque = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

        # Null-space control
        arm_mass_matrix_inv = torch.inverse(self._arm_mass_matrix)
        arm_mass_matrix_task = torch.inverse(
            self._fingertip_jacobian @ arm_mass_matrix_inv @ jacobian_T
        )
        j_eef_inv = arm_mass_matrix_task @ self._fingertip_jacobian @ arm_mass_matrix_inv

        default_dof_pos_tensor = self.default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1)
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]

        distance_to_default = default_dof_pos_tensor - joint_pos
        distance_to_default = (distance_to_default + np.pi) % (2 * np.pi) - np.pi

        u_null = self.kd_null * (-joint_vel) + self.kp_null * distance_to_default
        u_null = self._arm_mass_matrix @ u_null.unsqueeze(-1)
        torque_null = (
            torch.eye(7, device=self.device).unsqueeze(0) - jacobian_T @ j_eef_inv
        ) @ u_null
        dof_torque += torque_null.squeeze(-1)

        # Clamp
        dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)
        return dof_torque

    def _get_pose_error(self, current_pos, current_quat, target_pos, target_quat):
        """Compute task-space pose error (position + axis-angle)."""
        pos_error = target_pos - current_pos

        # Shortest path quaternion
        quat_dot = (target_quat * current_quat).sum(dim=1, keepdim=True)
        target_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0, target_quat, -target_quat
        )

        # Quaternion error
        quat_norm = torch_utils.quat_mul(
            current_quat, torch_utils.quat_conjugate(current_quat)
        )[:, 0]
        quat_inv = torch_utils.quat_conjugate(current_quat) / quat_norm.unsqueeze(-1)
        quat_error = torch_utils.quat_mul(target_quat, quat_inv)

        axis_angle_error = axis_angle_from_quat(quat_error)
        return pos_error, axis_angle_error

    def _get_deriv_gains(self, prop_gains, rot_deriv_scale=1.0):
        """Compute derivative gains: kd = 2 * sqrt(kp)."""
        deriv_gains = 2.0 * torch.sqrt(prop_gains)
        deriv_gains[:, 3:6] /= rot_deriv_scale
        return deriv_gains

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset action history and compute initial actions from current EE pose.

        Matches direct ForgeEnv._reset_idx(): computes initial pos/yaw actions
        from the current fingertip pose relative to the fixed asset, so that
        EMA smoothing starts from the correct state instead of zero.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            self.force_sensor_world_smooth.zero_()
            self.force_sensor_smooth.zero_()
            self.ee_linvel_fd.zero_()
            self.ee_angvel_fd.zero_()
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self.prev_actions.zero_()
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            self.force_sensor_world_smooth[env_ids] = 0.0
            self.force_sensor_smooth[env_ids] = 0.0
            self.ee_linvel_fd[env_ids] = 0.0
            self.ee_angvel_fd[env_ids] = 0.0
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0

        # Compute initial actions from current EE pose, matching direct version.
        # This must happen AFTER the scene is updated with the new state, so we
        # read from the asset data directly (which was set during env reset).
        ee_pos = (
            self._asset.data.body_pos_w[:, self._fingertip_body_idx] - self._env.scene.env_origins
        )
        ee_quat = self._asset.data.body_quat_w[:, self._fingertip_body_idx]

        # Position: (ee_pos - fixed_pos_frame) / bounds, matching direct.
        fixed_pos_frame = self.fixed_pos_obs_frame[env_ids] + self.init_fixed_pos_obs_noise[env_ids]
        pos_actions = ee_pos[env_ids] - fixed_pos_frame
        pos_actions = pos_actions @ torch.diag(1.0 / self.pos_action_bounds)

        # Yaw: relative yaw from EE to bolt frame, matching direct.
        unrot_180 = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(unrot_180[:, 0], unrot_180[:, 1], unrot_180[:, 2])
        quat_rel_bolt = torch_utils.quat_mul(unrot_quat, ee_quat[env_ids])
        yaw_bolt = torch_utils.get_euler_xyz(quat_rel_bolt)[-1]
        yaw_bolt = torch.where(yaw_bolt > np.pi / 2, yaw_bolt - 2 * np.pi, yaw_bolt)
        yaw_bolt = torch.where(yaw_bolt < -np.pi, yaw_bolt + 2 * np.pi, yaw_bolt)
        yaw_action = (yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0

        # Set initial actions for the reset envs.
        self.prev_actions[env_ids, 0:3] = pos_actions
        self.prev_actions[env_ids, 5] = yaw_action
        self.prev_actions[env_ids, 6] = -1.0
        self._processed_actions[env_ids] = self.prev_actions[env_ids].clone()


def _wrap_yaw(angle):
    """Wrap yaw to [-125, 235] degrees range."""
    return torch.where(angle > np.deg2rad(235), angle - 2 * np.pi, angle)


def get_random_prop_gains(default_values, noise_levels, num_envs, device):
    """Randomize controller gains (matching forge_utils.get_random_prop_gains)."""
    c_param_noise = torch.rand((num_envs, default_values.shape[1]), dtype=torch.float32, device=device)
    c_param_noise = c_param_noise @ torch.diag(torch.tensor(noise_levels, dtype=torch.float32, device=device))
    c_param_multiplier = 1.0 + c_param_noise
    decrease_param_flag = torch.rand((num_envs, default_values.shape[1]), dtype=torch.float32, device=device) > 0.5
    c_param_multiplier = torch.where(decrease_param_flag, 1.0 / c_param_multiplier, c_param_multiplier)
    return default_values * c_param_multiplier


@configclass
class ForgeImpedanceActionCfg(ActionTermCfg):
    """Configuration for ForgeImpedanceAction."""

    class_type: type = ForgeImpedanceAction

    # Joint names for the arm
    joint_names: list[str] = MISSING

    # Body name for end-effector tracking
    body_name: str = MISSING

    # Finger body names for Jacobian averaging
    finger_body_names: list[str] = MISSING

    # Fixed asset name in scene
    fixed_asset_name: str = "fixed_asset"

    # Default impedance gains [lin_x, lin_y, lin_z, rot_x, rot_y, rot_z]
    default_task_prop_gains: list[float] = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]

    # Action bounds
    pos_action_bounds: list[float] = [0.05, 0.05, 0.05]
    rot_action_bounds: list[float] = [1.0, 1.0, 1.0]

    # Action thresholds (clipping)
    pos_action_threshold: list[float] = [0.02, 0.02, 0.02]
    rot_action_threshold: list[float] = [0.097, 0.097, 0.097]

    # Default joint positions for null-space
    default_dof_pos_tensor: list[float] = [
        -1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754,
    ]

    # Null-space gains
    kp_null: float = 10.0
    kd_null: float = 6.3246

    # Dead zone defaults
    default_dead_zone: list[float] = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]

    # EMA factor
    ema_factor_default: float = 0.2
    ema_factor_range: list[float] = [0.025, 0.1]

    # Gain randomization
    task_prop_gains_noise_level: list[float] = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]
    pos_threshold_noise_level: list[float] = [0.25, 0.25, 0.25]
    rot_threshold_noise_level: list[float] = [0.29, 0.29, 0.29]

    # Force/torque smoothing
    ft_smoothing_factor: float = 0.25

    # Contact threshold range
    contact_threshold_range: list[float] = [5.0, 10.0]

    # Observation noise levels
    fingertip_pos_noise: float = 0.00025
    fingertip_rot_noise_deg: float = 0.1
    ft_force_noise: float = 1.0

    # Fixed asset geometry (task-specific, set in task configs)
    # These define the local offset from the fixed asset root to the observation frame (e.g., bolt tip)
    fixed_asset_height: float = 0.0
    fixed_asset_base_height: float = 0.0
    fixed_asset_offset_x: float = 0.0
    # Noise on fixed asset position observation, matching direct cfg.obs_rand.fixed_asset_pos
    fixed_asset_pos_noise: list[float] = [0.0, 0.0, 0.0]
