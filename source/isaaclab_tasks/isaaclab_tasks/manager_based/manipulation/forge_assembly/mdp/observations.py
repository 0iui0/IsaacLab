# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the forge assembly (peg-in-hole) environment.

Custom observation terms that replicate direct forge's observation structure.
Standard terms (last_action, joint_pos_rel, root_pos_w, root_quat_w) come from
isaaclab.envs.mdp via wildcard import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ee_pos_rel_hole(ManagerTermBase):
    """End-effector position relative to the hole.

    Direct forge equivalent: fingertip_pos_rel_fixed (noisy EE pos minus noisy fixed pos).
    Returns 3D vector: ee_pos - hole_pos in environment-local frame.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene["robot"]
        self.hole: Articulation = env.scene["hole"]
        body_name = cfg.params.get("body_name", "panda_hand")
        body_ids, _ = self.robot.find_bodies(body_name)
        self._ee_body_idx = body_ids[0]
        # Fixed asset position observation noise (direct forge: 0.001 diagonal)
        self._fixed_pos_obs_noise = torch.zeros(env.num_envs, 3, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, body_name: str = "panda_hand") -> torch.Tensor:
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx]
        hole_pos = self.hole.data.root_pos_w
        # Direct forge: subtract (fixed_pos_obs_frame + init_fixed_pos_obs_noise)
        return ee_pos - hole_pos - self._fixed_pos_obs_noise - env.scene.env_origins

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # Randomize fixed asset position noise on reset (direct forge: diag(0.001))
        if env_ids is not None:
            noise = torch.randn(len(env_ids), 3, device=self.robot.device) * 0.001
            self._fixed_pos_obs_noise[env_ids] = noise
        else:
            self._fixed_pos_obs_noise = torch.randn(
                self.robot.num_instances, 3, device=self.robot.device
            ) * 0.001


class ee_quat_canonical(ManagerTermBase):
    """End-effector orientation as quaternion with flip and rotation noise.

    Direct forge equivalent: fingertip_quat (with flip quaternions + rotation noise).
    - Applies random ±1 flip (flip_quats)
    - Adds 0.1° rotation noise with random axis
    - Zeros quaternion components [0, 3] (x and w) after noise (direct forge convention)
    - Does NOT canonicalize (no w > 0 enforcement)
    Returns 4D quaternion (w, x, y, z).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene["robot"]
        body_name = cfg.params.get("body_name", "panda_hand")
        body_ids, _ = self.robot.find_bodies(body_name)
        self._ee_body_idx = body_ids[0]
        self._rot_noise_level_deg = 0.1  # Direct forge: fingertip_rot_deg = 0.1

    def __call__(self, env: ManagerBasedRLEnv, body_name: str = "panda_hand") -> torch.Tensor:
        from isaaclab.utils.math import quat_from_angle_axis, quat_mul

        quat = self.robot.data.body_quat_w[:, self._ee_body_idx].clone()

        # Apply flip quaternions (random ±1, direct forge: flip_quats)
        if hasattr(env, "_flip_quats"):
            quat = quat * env._flip_quats.unsqueeze(-1)

        # Add rotation noise: 0.1° with random axis (direct forge: rot_noise_level_deg = 0.1)
        rot_noise_axis = torch.randn(env.num_envs, 3, device=env.device)
        rot_noise_axis = rot_noise_axis / (torch.norm(rot_noise_axis, dim=-1, keepdim=True) + 1e-8)
        rot_noise_angle = torch.randn(env.num_envs, device=env.device) * np.deg2rad(self._rot_noise_level_deg)
        rot_noise_quat = quat_from_angle_axis(rot_noise_angle, rot_noise_axis)
        quat = quat_mul(quat, rot_noise_quat)

        # Zero quaternion components [0, 3] (x and w) - direct forge convention
        quat[:, 0] = 0.0  # w component
        quat[:, 3] = 0.0  # z component

        return quat


class ee_linvel_fd(ManagerTermBase):
    """End-effector linear velocity via finite differencing.

    Direct forge equivalent: ee_linvel computed from noisy fingertip positions.
    Returns 3D velocity vector.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene["robot"]
        body_name = cfg.params.get("body_name", "panda_hand")
        body_ids, _ = self.robot.find_bodies(body_name)
        self._ee_body_idx = body_ids[0]
        # prev_pos should be noisy (direct forge uses noisy fingertip positions)
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx]
        self._prev_pos = ee_pos + torch.randn_like(ee_pos) * 0.00025

    def __call__(self, env: ManagerBasedRLEnv, body_name: str = "panda_hand") -> torch.Tensor:
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx]
        # Direct forge adds position noise (0.00025 level) before finite diff
        pos_noise_level = 0.00025
        noise = torch.randn_like(ee_pos) * pos_noise_level
        noisy_pos = ee_pos + noise
        dt = env.step_dt
        linvel = (noisy_pos - self._prev_pos) / dt
        self._prev_pos = noisy_pos.clone()
        return linvel

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            ee_pos = self.robot.data.body_pos_w[env_ids, self._ee_body_idx]
            self._prev_pos[env_ids] = ee_pos + torch.randn_like(ee_pos) * 0.00025
        else:
            ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx]
            self._prev_pos = ee_pos + torch.randn_like(ee_pos) * 0.00025


class ee_angvel_fd(ManagerTermBase):
    """End-effector angular velocity via finite differencing of orientation.

    Direct forge equivalent: ee_angvel computed from noisy fingertip quaternions.
    Returns 3D angular velocity vector. Direct forge zeroes roll/pitch components.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene["robot"]
        body_name = cfg.params.get("body_name", "panda_hand")
        body_ids, _ = self.robot.find_bodies(body_name)
        self._ee_body_idx = body_ids[0]
        self._prev_quat = self.robot.data.body_quat_w[:, self._ee_body_idx].clone()
        # Initialize flip quaternions if not present
        if not hasattr(env, "_flip_quats"):
            env._flip_quats = torch.ones(env.num_envs, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, body_name: str = "panda_hand") -> torch.Tensor:
        from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

        curr_quat = self.robot.data.body_quat_w[:, self._ee_body_idx].clone()
        # Apply flip quaternions (direct forge: noisy_fingertip_quat * flip_quats)
        if hasattr(env, "_flip_quats"):
            curr_quat = curr_quat * env._flip_quats.unsqueeze(-1)
        dt = env.step_dt

        # Compute rotation difference
        diff_quat = quat_mul(curr_quat, quat_conjugate(self._prev_quat))
        diff_quat *= torch.sign(diff_quat[:, 0]).unsqueeze(-1)
        axis_angle = axis_angle_from_quat(diff_quat)
        angvel = axis_angle / dt

        # Direct forge zeros roll/pitch angular velocity components
        angvel[:, 0:2] = 0.0

        self._prev_quat = curr_quat.clone()
        return angvel

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            self._prev_quat[env_ids] = self.robot.data.body_quat_w[env_ids, self._ee_body_idx].clone()
        else:
            self._prev_quat = self.robot.data.body_quat_w[:, self._ee_body_idx].clone()


class ft_force_smooth_noisy(ManagerTermBase):
    """Force sensor reading: EMA-smoothed with Gaussian noise, frame-transformed.

    Direct forge equivalent: ft_force observation. Reads 6D wrench from EE body,
    applies EMA smoothing, transforms to hole frame via change_FT_frame, adds noise.
    Returns 3D force vector (force component only).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene["robot"]
        self.hole: Articulation = env.scene["hole"]
        body_name = cfg.params.get("body_name", "panda_hand")
        body_ids, _ = self.robot.find_bodies(body_name)
        self._ee_body_idx = body_ids[0]
        self._smoothing_alpha = cfg.params.get("smoothing_alpha", 0.25)
        self._noise_std = cfg.params.get("noise_std", 1.0)
        self._force_smooth = torch.zeros(env.num_envs, 6, device=env.device)

    def _change_FT_frame(self, source_F, source_T, source_pos, source_quat, target_pos, target_quat):
        """Transform force/torque from source frame to target frame.

        Direct forge equivalent: change_FT_frame (Modern Robotics eq. 3.95).
        Transforms wrench from EE body frame to hole (fixed asset) frame.
        """
        from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

        # Compute relative transform: target_T_source
        source_quat_inv = quat_conjugate(source_quat)
        rel_pos = source_pos - target_pos
        # Rotate relative position to target frame
        rel_pos_target = quat_apply(target_quat, rel_pos)

        # Rotate force to target frame: F_target = R_target * R_source^T * F_source
        source_F_world = quat_apply(source_quat, source_F)  # source to world
        target_F = quat_apply(quat_conjugate(target_quat), source_F_world)  # world to target

        # Transform torque: T_target = R * T_source + r × F_target
        source_T_world = quat_apply(source_quat, source_T)
        target_T = quat_apply(quat_conjugate(target_quat), source_T_world) + torch.cross(rel_pos_target, target_F, dim=-1)
        return target_F, target_T

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        body_name: str = "panda_hand",
        smoothing_alpha: float = 0.25,
        noise_std: float = 1.0,
    ) -> torch.Tensor:
        # Read raw wrench from PhysX (6D: force + torque) in body frame
        raw_wrench = self.robot.data.body_incoming_joint_wrench_b[:, self._ee_body_idx]

        # EMA smoothing (direct forge: alpha * raw + (1-alpha) * prev)
        self._force_smooth = smoothing_alpha * raw_wrench + (1 - smoothing_alpha) * self._force_smooth

        # Transform force from EE body frame to hole frame (direct forge: change_FT_frame)
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx]
        hole_pos = self.hole.data.root_pos_w
        hole_quat = self.hole.data.root_quat_w

        source_F = self._force_smooth[:, :3]
        source_T = self._force_smooth[:, 3:6]
        target_F, _ = self._change_FT_frame(source_F, source_T, ee_pos, ee_quat, hole_pos, hole_quat)

        # Store force norm on env for contact_force_penalty reward term
        env._force_smooth_norm = torch.norm(target_F, p=2, dim=-1)

        # Add noise
        noise = torch.randn_like(target_F) * noise_std
        return target_F + noise

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            self._force_smooth[env_ids] = 0.0
        else:
            self._force_smooth.zero_()


class contact_threshold_obs(ManagerTermBase):
    """Scalar contact penalty threshold as observation.

    Direct forge equivalent: force_threshold observation.
    Reads the threshold stored on the environment by the randomization event.
    Returns 1D tensor.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not hasattr(env, "_contact_penalty_threshold"):
            env._contact_penalty_threshold = torch.ones(env.num_envs, 1, device=env.device) * 7.5

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return env._contact_penalty_threshold


class held_pos_rel_fixed(ManagerTermBase):
    """Peg position relative to hole (privileged observation).

    Direct forge equivalent: held_pos_rel_fixed in state_order.
    Returns 3D vector: peg_pos - hole_pos.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.peg: Articulation = env.scene["peg"]
        self.hole: Articulation = env.scene["hole"]

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        peg_pos = self.peg.data.root_pos_w - env.scene.env_origins
        hole_pos = self.hole.data.root_pos_w - env.scene.env_origins
        return peg_pos - hole_pos


class task_prop_gains_obs(ManagerTermBase):
    """Task-space proportional gains as privileged observation.

    Direct forge equivalent: task_prop_gains in state_order.
    Returns 6D tensor: [px, py, pz, rx, ry, rz] gains.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not hasattr(env, "_task_prop_gains"):
            env._task_prop_gains = torch.zeros(env.num_envs, 6, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return env._task_prop_gains


class ema_factor_obs(ManagerTermBase):
    """EMA smoothing factor as privileged observation.

    Direct forge equivalent: ema_factor in state_order.
    Returns 1D tensor.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not hasattr(env, "_ema_factor"):
            env._ema_factor = torch.ones(env.num_envs, 1, device=env.device) * 0.05

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return env._ema_factor


class pos_threshold_obs(ManagerTermBase):
    """Position clipping threshold as privileged observation.

    Direct forge equivalent: pos_threshold in state_order.
    Returns 3D tensor.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not hasattr(env, "_pos_threshold"):
            env._pos_threshold = torch.zeros(env.num_envs, 3, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return env._pos_threshold


class rot_threshold_obs(ManagerTermBase):
    """Rotation clipping threshold as privileged observation.

    Direct forge equivalent: rot_threshold in state_order.
    Returns 3D tensor.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not hasattr(env, "_rot_threshold"):
            env._rot_threshold = torch.zeros(env.num_envs, 3, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return env._rot_threshold
