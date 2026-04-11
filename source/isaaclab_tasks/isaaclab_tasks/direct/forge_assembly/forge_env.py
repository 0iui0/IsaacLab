# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forge Assembly: robot-agnostic direct RL environment.

Supports Franka (gripper-based) and UR10/CR5 (fixed-peg) robots through
the RobotProfile abstraction.
"""

import math

import carb
import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat

from . import forge_control, forge_utils
from .forge_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, ForgeEnvCfg
from .robot_profiles import RobotProfile


class ForgeEnv(DirectRLEnv):
    cfg: ForgeEnvCfg

    def __init__(self, cfg: ForgeEnvCfg, render_mode: str | None = None, **kwargs):
        # Resolve profile-driven dimensions before parent init.
        self.profile: RobotProfile = cfg.robot_profile
        self.num_arm_joints = self.profile.num_arm_joints
        self.arm_slice = self.profile.arm_joint_ids
        self.gripper_slice = self.profile.gripper_joint_ids
        self.cfg_task = cfg.task

        super().__init__(cfg, render_mode, **kwargs)

        forge_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()

        # Success prediction (FORGE-specific).
        self.success_pred_scale = 0.0
        self.first_pred_success_tx = {}
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh] = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        # Force sensor - body index will be set in _init_tensors after scene setup
        self.force_sensor_body_idx = None
        self.force_sensor_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_sensor_world_smooth = torch.zeros((self.num_envs, 6), device=self.device)

        # Dynamics randomization defaults.
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_dead_zone = torch.tensor(self.cfg.ctrl.default_dead_zone, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()

    # -----------------------------------------------------------------------
    # Scene setup
    # -----------------------------------------------------------------------

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.profile.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)

        # For fixed-peg robots, spawn the peg as a separate Articulation (held_asset)
        # so that contact forces are properly reported through its physics simulation.
        # The peg pose is updated every frame to track the EE link.
        if self.profile.grasp_type == "gripper":
            self._held_asset = Articulation(self.cfg_task.held_asset)
        elif self.profile.grasp_type == "fixed_peg" and self.profile.peg_offset_from_ee is not None:
            self._held_asset = Articulation(self._make_peg_articulation_cfg())
        else:
            self._held_asset = None

        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        if self._held_asset is not None:
            self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _make_peg_articulation_cfg(self):
        """Create an ArticulationCfg for the fixed peg as a standalone rigid body.

        Uses the factory peg USD (which has ArticulationRootAPI) so Isaac Lab
        can resolve it as a proper Articulation. The peg pose is updated every
        frame to track the robot's EE link.
        """
        peg_usd_path = self.profile.peg_usd_path
        if peg_usd_path is None:
            raise ValueError(
                "Fixed-peg robots require peg_usd_path in RobotProfile "
                "(needs USD with ArticulationRootAPI for contact force reporting)"
            )

        peg_prim_path = "/World/envs/env_.*/PegAsset"

        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=peg_usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        )

        return ArticulationCfg(
            prim_path=peg_prim_path,
            spawn=spawn_cfg,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        if self._held_asset is not None:
            forge_utils.set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction, self.scene.num_envs)
        forge_utils.set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction, self.scene.num_envs)
        forge_utils.set_friction(self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

    def _init_tensors(self):
        """Initialize tensors once."""
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ema_factor = self.cfg.ctrl.ema_factor
        self.dead_zone_thresholds = None

        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Body indices.
        self.ee_body_idx = self._robot.body_names.index(self.profile.ee_body_name)
        if self.profile.has_gripper and self.profile.left_finger_body_name is not None:
            self.left_finger_body_idx = self._robot.body_names.index(self.profile.left_finger_body_name)
            self.right_finger_body_idx = self._robot.body_names.index(self.profile.right_finger_body_name)

        # Force sensor body index - for fixed-peg robots, contact forces come from
        # the held_asset (peg articulation) via get_net_contact_forces(), not from
        # the robot articulation's joint forces.
        if self.profile.grasp_type == "fixed_peg":
            self.force_sensor_body_idx = None  # Uses held_asset contact forces instead
        elif self.profile.force_sensor_body_name is not None:
            self.force_sensor_body_idx = self._robot.body_names.index(self.profile.force_sensor_body_name)
        else:
            self.force_sensor_body_idx = None

        # Finite-differencing.
        self.last_update_timestamp = 0.0
        self.prev_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_ee_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.prev_joint_pos = torch.zeros((self.num_envs, self.num_arm_joints), device=self.device)

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    # -----------------------------------------------------------------------
    # Intermediate value computation
    # -----------------------------------------------------------------------

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.ee_pos = self._robot.data.body_pos_w[:, self.ee_body_idx] - self.scene.env_origins
        self.ee_quat = self._robot.data.body_quat_w[:, self.ee_body_idx]
        self.ee_linvel = self._robot.data.body_lin_vel_w[:, self.ee_body_idx]
        self.ee_angvel = self._robot.data.body_ang_vel_w[:, self.ee_body_idx]

        if self._held_asset is not None:
            if self.profile.grasp_type == "fixed_peg":
                # Fixed-peg: teleport peg to track EE pose every frame.
                # This ensures the peg is always at peg_offset_from_ee from the EE link.
                ee_pos_w = self._robot.data.body_pos_w[:, self.ee_body_idx]
                ee_quat_w = self._robot.data.body_quat_w[:, self.ee_body_idx]
                peg_offset = torch.tensor(self.profile.peg_offset_from_ee, device=self.device).unsqueeze(0).expand(
                    self.num_envs, -1
                )
                identity_quat = (
                    torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
                )
                _, peg_pos_w = torch_utils.tf_combine(ee_quat_w, ee_pos_w, identity_quat, peg_offset)

                # Write peg pose to simulation (position + quaternion).
                peg_root_state = torch.zeros((self.num_envs, 13), device=self.device)
                peg_root_state[:, 0:3] = peg_pos_w
                peg_root_state[:, 3:7] = ee_quat_w
                self._held_asset.write_root_pose_to_sim(peg_root_state[:, 0:7])
                self._held_asset.write_root_velocity_to_sim(peg_root_state[:, 7:])

            self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
            self.held_quat = self._held_asset.data.root_quat_w
        else:
            # Should not reach here — held_asset is always created for valid configs.
            self.held_pos = self.ee_pos.clone()
            self.held_quat = self.ee_quat.clone()

        jacobians = self._robot.root_physx_view.get_jacobians()
        mass_matrices = self._robot.root_physx_view.get_generalized_mass_matrices()

        # Compute EE Jacobian.
        if self.profile.has_gripper:
            left_jac = jacobians[:, self.left_finger_body_idx - 1, 0:6, :]
            right_jac = jacobians[:, self.right_finger_body_idx - 1, 0:6, :]
            self.ee_jacobian = (left_jac[:, :, self.arm_slice] + right_jac[:, :, self.arm_slice]) * 0.5
        else:
            self.ee_jacobian = jacobians[:, self.ee_body_idx - 1, 0:6, self.arm_slice]

        self.arm_mass_matrix = mass_matrices[:, self.arm_slice, self.arm_slice]
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Finite-differencing for velocities.
        self.ee_linvel_fd = (self.ee_pos - self.prev_ee_pos) / dt
        self.prev_ee_pos = self.ee_pos.clone()

        rot_diff_quat = torch_utils.quat_mul(self.ee_quat, torch_utils.quat_conjugate(self.prev_ee_quat))
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_ee_quat = self.ee_quat.clone()

        joint_diff = self.joint_pos[:, self.arm_slice] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, self.arm_slice].clone()

        self.last_update_timestamp = self._robot._data._sim_timestamp

        # --- FORGE-specific: noise + force sensing ---
        self._compute_forge_noise(dt)

    def _compute_forge_noise(self, dt):
        """Add noise to observations for force sensing (FORGE-specific)."""
        pos_noise_level = self.cfg.obs_rand.fingertip_pos
        rot_noise_level_deg = self.cfg.obs_rand.fingertip_rot_deg

        fingertip_pos_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        fingertip_pos_noise = fingertip_pos_noise @ torch.diag(
            torch.tensor([pos_noise_level] * 3, dtype=torch.float32, device=self.device)
        )
        self.noisy_ee_pos = self.ee_pos + fingertip_pos_noise

        rot_noise_axis = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        rot_noise_axis /= torch.linalg.norm(rot_noise_axis, dim=1, keepdim=True)
        rot_noise_angle = torch.randn((self.num_envs,), dtype=torch.float32, device=self.device) * np.deg2rad(
            rot_noise_level_deg
        )
        self.noisy_ee_quat = torch_utils.quat_mul(
            self.ee_quat, torch_utils.quat_from_angle_axis(rot_noise_angle, rot_noise_axis)
        )
        self.noisy_ee_quat[:, [0, 3]] = 0.0
        self.noisy_ee_quat = self.noisy_ee_quat * self.flip_quats.unsqueeze(-1)

        # Repeat finite differencing with noisy values.
        self.ee_linvel_fd = (self.noisy_ee_pos - self.prev_ee_pos) / dt
        self.prev_ee_pos = self.noisy_ee_pos.clone()

        rot_diff_quat = torch_utils.quat_mul(self.noisy_ee_quat, torch_utils.quat_conjugate(self.prev_ee_quat))
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.ee_angvel_fd[:, 0:2] = 0.0
        self.prev_ee_quat = self.noisy_ee_quat.clone()

        # Force sensing.
        if self.profile.grasp_type == "fixed_peg" and self._held_asset is not None:
            # Fixed-peg: use incoming joint force on the held_asset (peg articulation).
            # For a single-body articulation with a floating base, link index 0
            # captures all external forces including contact.
            self.force_sensor_world = self._held_asset.root_physx_view.get_link_incoming_joint_force()[:, 0]
            alpha = self.cfg.ft_smoothing_factor
            self.force_sensor_world_smooth = alpha * self.force_sensor_world + (1 - alpha) * self.force_sensor_world_smooth

            self.force_sensor_smooth = torch.zeros_like(self.force_sensor_world)
            identity_quat = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            )
            self.force_sensor_smooth[:, :3], self.force_sensor_smooth[:, 3:6] = forge_utils.change_FT_frame(
                self.force_sensor_world_smooth[:, 0:3],
                self.force_sensor_world_smooth[:, 3:6],
                (identity_quat, torch.zeros((self.num_envs, 3), device=self.device)),
                (identity_quat, self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise),
            )

            force_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
            force_noise *= self.cfg.obs_rand.ft_force
            self.noisy_force = self.force_sensor_smooth[:, 0:3] + force_noise
        elif self.force_sensor_body_idx is not None:
            self.force_sensor_world = self._robot.root_physx_view.get_link_incoming_joint_force()[
                :, self.force_sensor_body_idx
            ]
            alpha = self.cfg.ft_smoothing_factor
            self.force_sensor_world_smooth = alpha * self.force_sensor_world + (1 - alpha) * self.force_sensor_world_smooth

            self.force_sensor_smooth = torch.zeros_like(self.force_sensor_world)
            identity_quat = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            )
            self.force_sensor_smooth[:, :3], self.force_sensor_smooth[:, 3:6] = forge_utils.change_FT_frame(
                self.force_sensor_world_smooth[:, 0:3],
                self.force_sensor_world_smooth[:, 3:6],
                (identity_quat, torch.zeros((self.num_envs, 3), device=self.device)),
                (identity_quat, self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise),
            )

            force_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
            force_noise *= self.cfg.obs_rand.ft_force
            self.noisy_force = self.force_sensor_smooth[:, 0:3] + force_noise
        else:
            self.noisy_force = torch.zeros((self.num_envs, 3), device=self.device)

    # -----------------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------------

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        obs_dict, state_dict = self._get_obs_state_dict()

        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        obs_dict.update(
            {
                "fingertip_pos": self.noisy_ee_pos,
                "fingertip_pos_rel_fixed": self.noisy_ee_pos - noisy_fixed_pos,
                "fingertip_quat": self.noisy_ee_quat,
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "ft_force": self.noisy_force,
                "prev_actions": prev_actions,
            }
        )

        state_dict.update(
            {
                "ema_factor": self.ema_factor,
                "ft_force": self.force_sensor_smooth[:, 0:3] if (self.force_sensor_body_idx is not None or self.profile.grasp_type == "fixed_peg") else torch.zeros((self.num_envs, 3), device=self.device),
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "prev_actions": prev_actions,
            }
        )

        obs_tensors = forge_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = forge_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _get_obs_state_dict(self):
        """Populate dictionaries for the policy and critic."""
        prev_actions = self.actions.clone()

        obs_dict = {
            "fingertip_pos": self.ee_pos,
            "fingertip_pos_rel_fixed": self.ee_pos - (self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise),
            "fingertip_quat": self.ee_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.ee_pos,
            "fingertip_pos_rel_fixed": self.ee_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": self.ee_quat,
            "ee_linvel": self.ee_linvel,
            "ee_angvel": self.ee_angvel,
            "joint_pos": self.joint_pos[:, self.arm_slice],
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
        }
        return obs_dict, state_dict

    # -----------------------------------------------------------------------
    # Action application
    # -----------------------------------------------------------------------

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.actions = self.ema_factor * action.clone().to(self.device) + (1 - self.ema_factor) * self.actions

    def _apply_action(self):
        """FORGE actions: targets relative to the fixed asset."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))

        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_ee_preclipped_pos = fixed_pos_action_frame + pos_actions

        rot_actions[:, 0:2] = 0.0
        rot_actions[:, 2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0

        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )

        rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )

        ctrl_target_ee_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        # Clip position targets.
        self.delta_pos = ctrl_target_ee_preclipped_pos - self.ee_pos
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_ee_pos = self.ee_pos + pos_error_clipped

        # Clip orientation targets.
        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.ee_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_ee_preclipped_quat)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        curr_yaw = forge_utils.wrap_yaw(curr_yaw)
        desired_yaw = forge_utils.wrap_yaw(desired_yaw)

        self.delta_yaw = desired_yaw - curr_yaw
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch)

        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll

        curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch)

        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
        desired_xyz[:, 1] = curr_pitch + clipped_pitch

        ctrl_target_ee_quat = torch_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_ee_pos=ctrl_target_ee_pos,
            ctrl_target_ee_quat=ctrl_target_ee_quat,
            ctrl_target_gripper_dof_pos=0.0 if self.profile.has_gripper else None,
        )

    # -----------------------------------------------------------------------
    # Control
    # -----------------------------------------------------------------------

    def generate_ctrl_signals(self, ctrl_target_ee_pos, ctrl_target_ee_quat, ctrl_target_gripper_dof_pos):
        """Compute joint torques via task-space impedance control."""
        self.joint_torque, self.applied_wrench = forge_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            ee_pos=self.ee_pos,
            ee_quat=self.ee_quat,
            ee_linvel=self.ee_linvel,
            ee_angvel=self.ee_angvel,
            jacobian=self.ee_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_ee_pos=ctrl_target_ee_pos,
            ctrl_target_ee_quat=ctrl_target_ee_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
            num_arm_joints=self.num_arm_joints,
            null_space_default_pos=self.profile.null_space_default_pos,
            dead_zone_thresholds=self.dead_zone_thresholds,
        )

        if self.gripper_slice is not None and ctrl_target_gripper_dof_pos is not None:
            self.ctrl_target_joint_pos[:, self.gripper_slice] = ctrl_target_gripper_dof_pos
            self.joint_torque[:, self.gripper_slice] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def close_gripper_in_place(self):
        """Keep gripper in current position as gripper closes."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)

        pos_actions = actions[:, 0:3] * self.pos_threshold
        ctrl_target_ee_pos = self.ee_pos + pos_actions

        rot_actions = actions[:, 3:6]
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_ee_quat = torch_utils.quat_mul(rot_actions_quat, self.ee_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_ee_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_ee_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_ee_pos=ctrl_target_ee_pos,
            ctrl_target_ee_quat=ctrl_target_ee_quat,
            ctrl_target_gripper_dof_pos=0.0 if self.profile.has_gripper else None,
        )

    # -----------------------------------------------------------------------
    # Rewards
    # -----------------------------------------------------------------------

    def _get_rewards(self):
        """FORGE reward includes contact penalty and success prediction."""
        rew_buf = self._get_base_rewards()

        rew_dict, rew_scales = {}, {}
        pos_error = torch.norm(self.delta_pos, p=2, dim=-1) / self.cfg.ctrl.pos_action_threshold[0]
        rot_error = torch.abs(self.delta_yaw) / self.cfg.ctrl.rot_action_threshold[0]

        if self.force_sensor_body_idx is not None or self.profile.grasp_type == "fixed_peg":
            contact_force = torch.norm(self.force_sensor_smooth[:, 0:3], p=2, dim=-1, keepdim=False)
        else:
            contact_force = torch.zeros(self.num_envs, device=self.device)
        contact_penalty = torch.nn.functional.relu(contact_force - self.contact_penalty_thresholds)

        check_rot = self.cfg_task.name == "nut_thread"
        true_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )
        policy_success_pred = (self.actions[:, 6] + 1) / 2
        success_pred_error = (true_successes.float() - policy_success_pred).abs()

        if true_successes.float().mean() >= self.cfg_task.delay_until_ratio:
            self.success_pred_scale = 1.0

        rew_dict = {
            "action_penalty_asset": pos_error + rot_error,
            "contact_penalty": contact_penalty,
            "success_pred_error": success_pred_error,
        }
        rew_scales = {
            "action_penalty_asset": -self.cfg_task.action_penalty_asset_scale,
            "contact_penalty": -self.cfg_task.contact_penalty_scale,
            "success_pred_error": -self.success_pred_scale,
        }
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self._log_forge_metrics(rew_dict, policy_success_pred)
        return rew_buf

    def _get_base_rewards(self):
        """Compute base keypoint rewards."""
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        rew_dict, rew_scales = self._get_keypoint_rew_dict(curr_successes)

        rew_buf = torch.zeros_like(rew_dict["kp_coarse"])
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self.prev_actions = self.actions.clone()
        self._log_factory_metrics(rew_dict, curr_successes)
        return rew_buf

    def _get_keypoint_rew_dict(self, curr_successes):
        """Compute keypoint-based reward terms."""
        held_base_pos, held_base_quat = forge_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = forge_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )

        keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        keypoints_fixed = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        offsets = forge_utils.get_keypoint_offsets(self.cfg_task.num_keypoints, self.device)
        keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        for idx, keypoint_offset in enumerate(keypoint_offsets):
            keypoints_held[:, idx] = torch_utils.tf_combine(
                held_base_quat,
                held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
            keypoints_fixed[:, idx] = torch_utils.tf_combine(
                target_held_base_quat,
                target_held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
        keypoint_dist = torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)

        a0, b0 = self.cfg_task.keypoint_coef_baseline
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        a2, b2 = self.cfg_task.keypoint_coef_fine

        action_penalty_ee = torch.norm(self.actions, p=2)
        action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        curr_engaged = self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold, check_rot=False)

        rew_dict = {
            "kp_baseline": forge_utils.squashing_fn(keypoint_dist, a0, b0),
            "kp_coarse": forge_utils.squashing_fn(keypoint_dist, a1, b1),
            "kp_fine": forge_utils.squashing_fn(keypoint_dist, a2, b2),
            "action_penalty_ee": action_penalty_ee,
            "action_grad_penalty": action_grad_penalty,
            "curr_engaged": curr_engaged.float(),
            "curr_success": curr_successes.float(),
        }
        rew_scales = {
            "kp_baseline": 1.0,
            "kp_coarse": 1.0,
            "kp_fine": 1.0,
            "action_penalty_ee": -self.cfg_task.action_penalty_ee_scale,
            "action_grad_penalty": -self.cfg_task.action_grad_penalty_scale,
            "curr_engaged": 1.0,
            "curr_success": 1.0,
        }
        return rew_dict, rew_scales

    # -----------------------------------------------------------------------
    # Dones
    # -----------------------------------------------------------------------

    def _get_dones(self):
        """Check which environments are terminated."""
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    # -----------------------------------------------------------------------
    # Success check
    # -----------------------------------------------------------------------

    def _get_curr_successes(self, success_threshold, check_rot=False):
        """Get success mask at current timestep."""
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        held_base_pos, held_base_quat = forge_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = forge_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )

        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))

        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name in ("peg_insert", "gear_mesh"):
            height_threshold = fixed_cfg.height * success_threshold
        elif self.cfg_task.name == "nut_thread":
            height_threshold = fixed_cfg.thread_pitch * success_threshold
        else:
            raise NotImplementedError("Task not implemented")

        is_close_or_below = torch.where(
            z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        if check_rot:
            _, _, curr_yaw = torch_utils.get_euler_xyz(self.ee_quat)
            curr_yaw = forge_utils.wrap_yaw(curr_yaw)
            is_rotated = curr_yaw < self.cfg_task.ee_success_yaw
            curr_successes = torch.logical_and(curr_successes, is_rotated)

        return curr_successes

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def _reset_idx(self, env_ids):
        """Perform full reset for specified environments."""
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_robot_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

        # FORGE-specific randomization.
        self._reset_forge_randomization(env_ids)

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0
        self.ep_success_times[env_ids] = 0
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh][env_ids] = 0

    def _reset_forge_randomization(self, env_ids):
        """FORGE-specific reset randomization."""
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        pos_actions = self.ee_pos - fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

        unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )
        ee_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.ee_quat)
        ee_yaw_bolt = torch_utils.get_euler_xyz(ee_quat_rel_bolt)[-1]
        ee_yaw_bolt = torch.where(ee_yaw_bolt > torch.pi / 2, ee_yaw_bolt - 2 * torch.pi, ee_yaw_bolt)
        ee_yaw_bolt = torch.where(ee_yaw_bolt < -torch.pi, ee_yaw_bolt + 2 * torch.pi, ee_yaw_bolt)

        yaw_action = (ee_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action
        self.actions[:, 6] = self.prev_actions[:, 6] = -1.0

        # EMA randomization.
        ema_rand = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device)
        ema_lower, ema_upper = self.cfg.ctrl.ema_factor_range
        self.ema_factor = ema_lower + ema_rand * (ema_upper - ema_lower)

        prop_gains = self.default_gains.clone()
        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()
        prop_gains = forge_utils.get_random_prop_gains(
            prop_gains, self.cfg.ctrl.task_prop_gains_noise_level, self.num_envs, self.device
        )
        self.pos_threshold = forge_utils.get_random_prop_gains(
            self.pos_threshold, self.cfg.ctrl.pos_threshold_noise_level, self.num_envs, self.device
        )
        self.rot_threshold = forge_utils.get_random_prop_gains(
            self.rot_threshold, self.cfg.ctrl.rot_threshold_noise_level, self.num_envs, self.device
        )
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = forge_utils.get_deriv_gains(prop_gains)

        contact_rand = torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
        contact_lower, contact_upper = self.cfg_task.contact_penalty_threshold_range
        self.contact_penalty_thresholds = contact_lower + contact_rand * (contact_upper - contact_lower)

        self.dead_zone_thresholds = (
            torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device) * self.default_dead_zone
        )

        self.force_sensor_world_smooth[:, :] = 0.0

        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        rand_flips = torch.rand(self.num_envs) > 0.5
        self.flip_quats[rand_flips] = -1.0

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        if self._held_asset is not None:
            held_state = self._held_asset.data.default_root_state.clone()[env_ids]
            held_state[:, 0:3] += self.scene.env_origins[env_ids]
            held_state[:, 7:] = 0.0
            self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
            self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
            self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def _set_robot_to_default_pose(self, joints, env_ids):
        """Return robot to its default joint position."""
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self.arm_slice] = torch.tensor(joints, device=self.device)[None, :]
        if self.gripper_slice is not None:
            gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25 if self.profile.has_gripper else 0.0
            joint_pos[:, self.gripper_slice] = gripper_width
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)
        self.step_sim_no_action()

    # -----------------------------------------------------------------------
    # IK and state randomization
    # -----------------------------------------------------------------------

    def set_pos_inverse_kinematics(self, ctrl_target_ee_pos, ctrl_target_ee_quat, env_ids):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.25:
            pos_error, axis_angle_error = forge_control.get_pose_error(
                ee_pos=self.ee_pos[env_ids],
                ee_quat=self.ee_quat[env_ids],
                ctrl_target_ee_pos=ctrl_target_ee_pos[env_ids],
                ctrl_target_ee_quat=ctrl_target_ee_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            delta_dof_pos = forge_control.get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.ee_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, self.arm_slice] += delta_dof_pos[:, : self.num_arm_joints]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids])

            self.ctrl_target_joint_pos[env_ids, self.arm_slice] = self.joint_pos[env_ids, self.arm_slice]
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self):
        """Get default relative pose between held asset and EE."""
        if self.profile.grasp_type == "fixed_peg":
            # Peg is rigidly attached; relative pose is the peg offset from EE link.
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            held_asset_relative_pos[:, 2] = self.profile.peg_offset_from_ee[2]
            held_asset_relative_quat = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            )
            return held_asset_relative_pos, held_asset_relative_quat

        if self.cfg_task.name == "peg_insert":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.profile.fingerpad_length
        elif self.cfg_task.name == "gear_mesh":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            gear_base_offset = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset
            held_asset_relative_pos[:, 0] += gear_base_offset[0]
            held_asset_relative_pos[:, 2] += gear_base_offset[2]
            held_asset_relative_pos[:, 2] += self.cfg_task.held_asset_cfg.height / 2.0 * 1.1
        elif self.cfg_task.name == "nut_thread":
            held_asset_relative_pos = forge_utils.get_held_base_pos_local(
                self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
            )
        else:
            raise NotImplementedError("Task not implemented")

        held_asset_relative_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        if self.cfg_task.name == "nut_thread":
            initial_rot_deg = self.cfg_task.held_asset_rot_init
            rot_yaw_euler = torch.tensor([0.0, 0.0, initial_rot_deg * np.pi / 180.0], device=self.device).repeat(
                self.num_envs, 1
            )
            held_asset_relative_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_yaw_euler[:, 0], pitch=rot_yaw_euler[:, 1], yaw=rot_yaw_euler[:, 2]
            )

        return held_asset_relative_pos, held_asset_relative_quat

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform episode-level randomization."""
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1) Randomize fixed asset pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]

        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

        # Compute observation frame.
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # (2) Move EE to randomized location above fixed asset.
        bad_envs = env_ids.clone()
        ik_attempt = 0
        max_ik_attempts = 10
        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

        while bad_envs.shape[0] > 0 and ik_attempt < max_ik_attempts:
            n_bad = bad_envs.shape[0]

            above_fixed_pos = fixed_tip_pos.clone()
            above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            above_fixed_pos[bad_envs] += above_fixed_pos_rand

            hand_down_euler = (
                torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            )
            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            )

            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_ee_pos=above_fixed_pos,
                ctrl_target_ee_quat=hand_down_quat,
                env_ids=bad_envs,
            )
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
            angle_error = torch.norm(aa_error, dim=1) > 1e-3
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            if bad_envs.shape[0] == 0:
                break

            self._set_robot_to_default_pose(
                joints=self.profile.reset_arm_joint_pos, env_ids=bad_envs
            )
            ik_attempt += 1

        self.step_sim_no_action()

        # Add flanking gears for gear_mesh.
        if self.cfg_task.name == "gear_mesh" and self.cfg_task.add_flanking_gears:
            small_gear_state = self._small_gear_asset.data.default_root_state.clone()[env_ids]
            small_gear_state[:, 0:7] = fixed_state[:, 0:7]
            small_gear_state[:, 7:] = 0.0
            self._small_gear_asset.write_root_pose_to_sim(small_gear_state[:, 0:7], env_ids=env_ids)
            self._small_gear_asset.write_root_velocity_to_sim(small_gear_state[:, 7:], env_ids=env_ids)
            self._small_gear_asset.reset()

            large_gear_state = self._large_gear_asset.data.default_root_state.clone()[env_ids]
            large_gear_state[:, 0:7] = fixed_state[:, 0:7]
            large_gear_state[:, 7:] = 0.0
            self._large_gear_asset.write_root_pose_to_sim(large_gear_state[:, 0:7], env_ids=env_ids)
            self._large_gear_asset.write_root_velocity_to_sim(large_gear_state[:, 7:], env_ids=env_ids)
            self._large_gear_asset.reset()

        # (3) Position held asset relative to EE.
        if self.profile.grasp_type == "gripper":
            self._randomize_held_asset_in_gripper(env_ids)
        # For fixed_peg, no held_asset positioning needed — peg is part of the robot.

        # Set reset gains.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = forge_utils.get_deriv_gains(
            reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale
        )

        self.step_sim_no_action()

        # Close gripper (gripper-based robots only).
        if self.profile.has_gripper:
            grasp_time = 0.0
            while grasp_time < 0.25:
                self.ctrl_target_joint_pos[env_ids, self.gripper_slice] = 0.0
                self.close_gripper_in_place()
                self.step_sim_no_action()
                grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, self.arm_slice].clone()
        self.prev_ee_pos = self.ee_pos.clone()
        self.prev_ee_quat = self.ee_quat.clone()

        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = forge_utils.get_deriv_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    def _randomize_held_asset_in_gripper(self, env_ids):
        """Position the held asset in the gripper with randomization."""
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        ee_flipped_quat, ee_flipped_pos = torch_utils.tf_combine(
            q1=self.ee_quat,
            t1=self.ee_pos,
            q2=flip_z_quat,
            t2=torch.zeros((self.num_envs, 3), device=self.device),
        )

        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )

        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=ee_flipped_quat, t1=ee_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )

        # Add randomization.
        rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        held_asset_pos_noise = 2 * (rand_sample - 0.5)
        if self.cfg_task.name == "gear_mesh":
            held_asset_pos_noise[:, 2] = -rand_sample[:, 2]

        held_asset_pos_noise_level = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        held_asset_pos_noise = held_asset_pos_noise @ torch.diag(held_asset_pos_noise_level)
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            t2=held_asset_pos_noise,
        )

        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

    # -----------------------------------------------------------------------
    # Sim helpers
    # -----------------------------------------------------------------------

    def step_sim_no_action(self):
        """Step the simulation without an action."""
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def _log_factory_metrics(self, rew_dict, curr_successes):
        """Keep track of episode statistics and log rewards."""
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

    def _log_forge_metrics(self, rew_dict, policy_success_pred):
        """Log metrics to evaluate success prediction performance."""
        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        for thresh, first_success_tx in self.first_pred_success_tx.items():
            curr_predicted_success = policy_success_pred > thresh
            first_success_idxs = torch.logical_and(curr_predicted_success, first_success_tx == 0)

            first_success_tx[:] = torch.where(first_success_idxs, self.episode_length_buf, first_success_tx)

            if torch.any(self.reset_buf):
                delay_ids = torch.logical_and(self.ep_success_times != 0, first_success_tx != 0)
                delay_times = (first_success_tx[delay_ids] - self.ep_success_times[delay_ids]).sum() / delay_ids.sum()
                if delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_all/{thresh}"] = delay_times

                correct_delay_ids = torch.logical_and(delay_ids, first_success_tx > self.ep_success_times)
                correct_delay_times = (
                    first_success_tx[correct_delay_ids] - self.ep_success_times[correct_delay_ids]
                ).sum() / correct_delay_ids.sum()
                if correct_delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_correct/{thresh}"] = correct_delay_times.item()

                pred_success_idxs = first_success_tx != 0
                true_success_preds = torch.logical_and(
                    self.ep_success_times[pred_success_idxs] > 0,
                    self.ep_success_times[pred_success_idxs] < first_success_tx[pred_success_idxs],
                )

                num_pred_success = pred_success_idxs.sum().item()
                et_prec = true_success_preds.sum() / num_pred_success
                if num_pred_success > 0:
                    self.extras[f"early_term_precision/{thresh}"] = et_prec

                true_success_idxs = self.ep_success_times > 0
                num_true_success = true_success_idxs.sum().item()
                et_recall = true_success_preds.sum() / num_true_success
                if num_true_success > 0:
                    self.extras[f"early_term_recall/{thresh}"] = et_recall
