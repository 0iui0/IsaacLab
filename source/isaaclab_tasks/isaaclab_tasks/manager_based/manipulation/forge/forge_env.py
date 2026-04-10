# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom FORGE environment with initialization logic matching direct version.

The key difference from standard ManagerBasedRLEnv is the initialization sequence:
1. Randomize fixed asset pose
2. Position robot hand above hole using IK
3. Place held asset in gripper
4. Close gripper

This matches the direct version's randomize_initial_state() method.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import axis_angle_from_quat

if TYPE_CHECKING:
    from .forge_env_cfg import ForgeEnvCfg

# Logger for this module
logger = logging.getLogger(__name__)


class ForgeEnv(ManagerBasedRLEnv):
    """Custom FORGE environment with initialization matching direct version.

    This environment adds initialization logic to position the robot and held asset
    correctly at the start of each episode, matching the direct version's behavior.
    """

    cfg: ForgeEnvCfg

    def __init__(self, cfg: ForgeEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Get references to assets
        self._robot = self.scene["robot"]
        self._fixed_asset = self.scene["fixed_asset"]
        self._held_asset = self.scene["held_asset"]

        # Get the action term for IK control
        self._action_term = None
        for name, term in self.action_manager._terms.items():
            if hasattr(term, "_fingertip_body_idx"):
                self._action_term = term
                break

        # Get initialization config (task-specific or default)
        self._init_cfg = getattr(cfg, "init_cfg", None)
        if self._init_cfg is None:
            # Create default init config
            self._init_cfg = type('InitCfg', (), {
                'hand_init_pos': [0.0, 0.0, 0.047],
                'hand_init_pos_noise': [0.02, 0.02, 0.01],
                'hand_init_orn': [3.14159, 0.0, 0.0],
                'hand_init_orn_noise': [0.0, 0.0, 0.785],
                'held_asset_pos_noise': [0.003, 0.0, 0.003],
                'held_asset_height': 0.05,
                'franka_fingerpad_length': 0.017608,
            })()

        # Initialize buffers for held asset relative pose
        self._held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._held_asset_relative_quat = torch.zeros((self.num_envs, 4), device=self.device)

    def _reset_idx(self, env_ids):
        """Reset environments with initialization matching direct version.

        This extends the base reset to include:
        1. Standard reset events (gain randomization, etc.)
        2. Custom initialization (IK positioning, held asset placement, gripper closing)
        """
        # Call parent reset (applies reset events including fixed asset randomization)
        super()._reset_idx(env_ids)

        # Now perform custom initialization that requires multiple sim steps
        self._initialize_robot_and_held_asset(env_ids)

    def _initialize_robot_and_held_asset(self, env_ids):
        """Initialize robot and held asset to match direct version.

        This method:
        1. Disables gravity for stable initialization
        2. Positions robot hand above fixed asset using IK
        3. Places held asset in gripper
        4. Closes gripper
        5. Restores gravity
        """
        if self._action_term is None:
            logger.warning("[ForgeEnv] Action term not found, skipping initialization")
            return

        logger.info(f"[ForgeEnv] Initializing {len(env_ids)} environments...")
        logger.info(f"[ForgeEnv] Fixed asset pos: {self._fixed_asset.data.root_pos_w[env_ids[0]]}")
        logger.info(f"[ForgeEnv] Robot initial joint pos: {self._robot.data.joint_pos[env_ids[0], :7]}")

        # Disable gravity during initialization (matching direct version)
        try:
            import carb
            physics_sim_view = self.sim.physics_sim_view
            physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))
        except Exception as e:
            logger.warning(f"[ForgeEnv] Could not disable gravity: {e}")

        # Step 0: Reset robot to default pose for IK convergence (matching direct version)
        self._set_robot_to_default_pose(env_ids)

        # Step 1: Position robot hand above fixed asset using IK
        self._position_robot_with_ik(env_ids)

        # Step 2: Place held asset in gripper
        self._place_held_asset_in_gripper(env_ids)

        # Step 3: Close gripper
        self._close_gripper(env_ids)

        # Restore gravity after initialization
        try:
            import carb
            physics_sim_view = self.sim.physics_sim_view
            physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
        except Exception as e:
            logger.warning(f"[ForgeEnv] Could not restore gravity: {e}")

        logger.info(f"[ForgeEnv] Held asset final pos: {self._held_asset.data.root_pos_w[env_ids[0]]}")
        logger.info("[ForgeEnv] Initialization complete")

        # Reset action history after initialization
        self._action_term.reset(env_ids)

    def _set_robot_to_default_pose(self, env_ids):
        """Reset robot to default joint position for IK convergence.

        Matches direct version's _set_franka_to_default_pose().
        """
        # Default joint positions for IK starting pose
        default_joints = [0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0]

        # Gripper width = held_asset_diameter / 2 * 1.25
        held_diameter = 0.007986  # Peg 8mm diameter
        gripper_width = held_diameter / 2 * 1.25

        # Set joint positions
        joint_pos = self._robot.data.joint_pos[env_ids].clone()
        joint_pos[:, :7] = torch.tensor(default_joints, device=self.device)
        joint_pos[:, 7:] = gripper_width  # Open gripper

        # Zero velocity
        joint_vel = torch.zeros_like(joint_pos)

        # Write to simulation
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()

        # Step simulation to update state
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

        logger.info(f"[ForgeEnv] Reset robot to default pose: {joint_pos[0, :7]}")

    def _position_robot_with_ik(self, env_ids):
        """Position robot hand above fixed asset using DLS IK.

        Matches direct version's set_pos_inverse_kinematics().
        """
        # Get fixed asset position (already randomized by event)
        fixed_pos = self._fixed_asset.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        fixed_quat = self._fixed_asset.data.root_quat_w[env_ids]

        # Compute target position above fixed asset (bolt tip)
        task_cfg = self.cfg.actions.arm_action
        height = getattr(task_cfg, "fixed_asset_height", 0.025)
        base_height = getattr(task_cfg, "fixed_asset_base_height", 0.0)
        offset_x = getattr(task_cfg, "fixed_asset_offset_x", 0.0)

        # Target is above the bolt/hole tip
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
        tip_offset_local = torch.zeros((len(env_ids), 3), device=self.device)
        tip_offset_local[:, 2] = height + base_height
        tip_offset_local[:, 0] = offset_x

        _, tip_pos = torch_utils.tf_combine(fixed_quat, fixed_pos, identity_quat, tip_offset_local)

        # Add offset for hand position above tip
        hand_init_z = self._init_cfg.hand_init_pos[2]
        target_pos = tip_pos.clone()
        target_pos[:, 2] += hand_init_z

        # Add noise to target position
        hand_init_pos_noise = torch.tensor(self._init_cfg.hand_init_pos_noise, device=self.device)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        pos_noise = 2 * (rand_sample - 0.5) @ torch.diag(hand_init_pos_noise)
        target_pos += pos_noise

        # Target orientation (facing down)
        hand_init_euler = torch.tensor(self._init_cfg.hand_init_orn, device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
        hand_init_orn_noise = torch.tensor(self._init_cfg.hand_init_orn_noise, device=self.device)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        orn_noise = 2 * (rand_sample - 0.5) @ torch.diag(hand_init_orn_noise)
        hand_init_euler += orn_noise
        target_quat = torch_utils.quat_from_euler_xyz(
            roll=hand_init_euler[:, 0], pitch=hand_init_euler[:, 1], yaw=hand_init_euler[:, 2]
        )

        # Perform IK iterations
        ik_time = 0.0
        ik_duration = 0.25
        dt = self.physics_dt

        logger.info(f"[ForgeEnv] IK target pos: {target_pos[0]}")
        logger.info(f"[ForgeEnv] Initial fingertip pos: {self._robot.data.body_pos_w[env_ids[0], self._action_term._fingertip_body_idx]}")

        while ik_time < ik_duration:
            # Get current fingertip state
            fingertip_pos = self._robot.data.body_pos_w[env_ids, self._action_term._fingertip_body_idx] - self.scene.env_origins[env_ids]
            fingertip_quat = self._robot.data.body_quat_w[env_ids, self._action_term._fingertip_body_idx]

            # Compute error
            pos_error = target_pos - fingertip_pos

            # Quaternion error
            quat_dot = (target_quat * fingertip_quat).sum(dim=1, keepdim=True)
            target_quat_adj = torch.where(quat_dot.expand(-1, 4) >= 0, target_quat, -target_quat)
            quat_norm = torch_utils.quat_mul(fingertip_quat, torch_utils.quat_conjugate(fingertip_quat))[:, 0]
            quat_inv = torch_utils.quat_conjugate(fingertip_quat) / quat_norm.unsqueeze(-1)
            quat_error = torch_utils.quat_mul(target_quat_adj, quat_inv)

            # Convert to axis-angle using isaaclab's implementation
            axis_angle_error = axis_angle_from_quat(quat_error)

            delta_pose = torch.cat((pos_error, axis_angle_error), dim=1)

            # Get Jacobian
            jacobians = self._robot.root_physx_view.get_jacobians()
            left_jac = jacobians[env_ids, self._action_term._left_finger_body_idx - 1, 0:6, 0:7]
            right_jac = jacobians[env_ids, self._action_term._right_finger_body_idx - 1, 0:6, 0:7]
            jacobian = (left_jac + right_jac) * 0.5

            # DLS IK
            lambda_val = 0.1
            jacobian_T = jacobian.transpose(1, 2)
            lambda_matrix = (lambda_val ** 2) * torch.eye(6, device=self.device).unsqueeze(0).repeat(len(env_ids), 1, 1)
            delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

            # Update joint positions
            joint_pos = self._robot.data.joint_pos[env_ids, :7].clone()
            joint_pos += delta_dof_pos

            # Write to simulation
            joint_pos_full = self._robot.data.joint_pos[env_ids].clone()
            joint_pos_full[:, :7] = joint_pos
            joint_vel = torch.zeros_like(joint_pos_full)
            self._robot.write_joint_state_to_sim(joint_pos_full, joint_vel, env_ids=env_ids)

            # Step simulation
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            ik_time += dt

        # Log final IK position
        final_fingertip_pos = self._robot.data.body_pos_w[env_ids[0], self._action_term._fingertip_body_idx]
        logger.info(f"[ForgeEnv] IK complete. Final fingertip pos: {final_fingertip_pos}")
        logger.info(f"[ForgeEnv] Final joint pos: {self._robot.data.joint_pos[env_ids[0], :7]}")

    def _place_held_asset_in_gripper(self, env_ids):
        """Place held asset in gripper.

        Matches direct version's held asset placement in randomize_initial_state().
        """
        # Get current fingertip state
        fingertip_body_idx = self._action_term._fingertip_body_idx
        fingertip_pos = self._robot.data.body_pos_w[env_ids, fingertip_body_idx] - self.scene.env_origins[env_ids]
        fingertip_quat = self._robot.data.body_quat_w[env_ids, fingertip_body_idx]

        # Debug logging
        logger.info(f"[ForgeEnv] Place held asset - fingertip_pos[0]: {fingertip_pos[0]}")
        logger.info(f"[ForgeEnv] Place held asset - fingertip_quat[0]: {fingertip_quat[0]}")

        # Flip gripper z orientation
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=fingertip_quat, t1=fingertip_pos, q2=flip_z_quat, t2=torch.zeros((len(env_ids), 3), device=self.device)
        )

        logger.info(f"[ForgeEnv] Place held asset - fingertip_flipped_pos[0]: {fingertip_flipped_pos[0]}")
        logger.info(f"[ForgeEnv] Place held asset - fingertip_flipped_quat[0]: {fingertip_flipped_quat[0]}")

        # Get held asset relative pose
        held_relative_pos, held_relative_quat = self._get_held_asset_relative_pose(env_ids)
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(held_relative_quat, held_relative_pos)

        logger.info(f"[ForgeEnv] Place held asset - held_relative_pos[0]: {held_relative_pos[0]}")
        logger.info(f"[ForgeEnv] Place held asset - asset_in_hand_pos[0]: {asset_in_hand_pos[0]}")

        translated_held_quat, translated_held_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )

        logger.info(f"[ForgeEnv] Place held asset - translated_held_pos[0] before noise: {translated_held_pos[0]}")

        # Add held asset noise
        held_asset_pos_noise_level = torch.tensor(self._init_cfg.held_asset_pos_noise, device=self.device)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        held_asset_pos_noise = 2 * (rand_sample - 0.5) @ torch.diag(held_asset_pos_noise_level)

        translated_held_pos += held_asset_pos_noise

        logger.info(f"[ForgeEnv] Place held asset - translated_held_pos[0] after noise: {translated_held_pos[0]}")

        # Write held asset state
        held_state = self._held_asset.data.root_state_w[env_ids].clone()
        held_state[:, 0:3] = translated_held_pos + self.scene.env_origins[env_ids]
        held_state[:, 3:7] = translated_held_quat
        held_state[:, 7:] = 0.0  # Zero velocity

        logger.info(f"[ForgeEnv] Place held asset - held_state[0, 0:3] (world): {held_state[0, 0:3]}")
        logger.info(f"[ForgeEnv] Place held asset - env_origins[env_ids[0]]: {self.scene.env_origins[env_ids[0]]}")

        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset(env_ids=env_ids)  # Reset internal state, matching direct version

        # Step simulation
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

        # Log held asset position after step
        logger.info(f"[ForgeEnv] Place held asset - held asset pos after step: {self._held_asset.data.root_pos_w[env_ids[0]]}")

    def _close_gripper(self, env_ids):
        """Close gripper to grasp held asset using impedance control.

        Matches direct version's gripper closing logic:
        1. Set reset gains (stiffer)
        2. Apply impedance control to hold arm in place
        3. Let gripper close via position target
        4. Restore default gains after
        """
        grasp_time = 0.0
        grasp_duration = 0.25
        dt = self.physics_dt

        # Step once before closing to settle the held asset
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

        logger.info(f"[ForgeEnv] Close gripper - held asset pos before loop: {self._held_asset.data.root_pos_w[env_ids[0]]}")

        step_count = 0
        while grasp_time < grasp_duration:
            # Apply impedance control to hold arm in place while gripper closes
            self._action_term.apply_initialization_control(env_ids)

            # Step simulation
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            grasp_time += dt
            step_count += 1

            # Log every few steps
            if step_count % 10 == 0:
                logger.info(f"[ForgeEnv] Close gripper step {step_count} - held asset pos: {self._held_asset.data.root_pos_w[env_ids[0]]}")

        logger.info(f"[ForgeEnv] Close gripper complete - held asset pos: {self._held_asset.data.root_pos_w[env_ids[0]]}")

        # Restore default gains after initialization
        self._action_term.restore_default_gains(env_ids)

    def _get_held_asset_relative_pose(self, env_ids):
        """Get default relative pose between held asset and fingertip.

        Matches direct version's get_handheld_asset_relative_pose().
        """
        # Relative position: held asset base is below fingertip
        held_asset_relative_pos = torch.zeros((len(env_ids), 3), device=self.device)
        held_asset_relative_pos[:, 2] = self._init_cfg.held_asset_height - self._init_cfg.franka_fingerpad_length

        # Relative orientation: identity
        held_asset_relative_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)

        return held_asset_relative_pos, held_asset_relative_quat

    def _quat_to_axis_angle(self, quat):
        """Convert quaternion to axis-angle representation."""
        w = quat[:, 0]
        xyz = quat[:, 1:4]

        # Compute angle
        sin_half = torch.norm(xyz, dim=1)
        angle = 2 * torch.atan2(sin_half, w)

        # Compute axis
        sin_half_safe = torch.where(sin_half > 1e-8, sin_half, torch.ones_like(sin_half))
        axis = xyz / sin_half_safe.unsqueeze(-1)

        # Handle small angles
        angle_safe = torch.where(angle > 1e-8, angle, torch.ones_like(angle))
        axis_angle = axis * angle_safe.unsqueeze(-1)

        # For very small angles, return zero
        axis_angle = torch.where(angle.unsqueeze(-1).expand(-1, 3) < 1e-8, torch.zeros_like(axis_angle), axis_angle)

        return axis_angle