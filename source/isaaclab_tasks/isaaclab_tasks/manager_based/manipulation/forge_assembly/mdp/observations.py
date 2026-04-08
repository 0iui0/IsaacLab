# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the forge assembly (peg-in-hole) environment.

Custom observation terms that replicate direct forge's observation structure.
Standard terms (joint_pos_rel, root_pos_w, root_quat_w) come from
isaaclab.envs.mdp. The last_action term is custom to match direct forge's
behavior of zeroing roll/pitch action components in the observation.
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
    """Force sensor reading: EMA-smoothed with Gaussian noise, frame-transformed to hole frame.

    Direct forge equivalent: ft_force observation.

    Direct forge reads force in WORLD frame via get_link_incoming_joint_force(),
    then transforms from world frame to hole frame using change_FT_frame with
    source_frame=(identity, zeros) and target_frame=(identity, hole_pos).

    Since both frames use identity quaternion, the force vector is unchanged.
    This is equivalent to using the world-frame force directly.
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

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        body_name: str = "panda_hand",
        smoothing_alpha: float = 0.25,
        noise_std: float = 1.0,
    ) -> torch.Tensor:
        # Read raw wrench from PhysX in WORLD frame (matching direct forge)
        # Direct forge: self._robot.root_physx_view.get_link_incoming_joint_force()
        raw_wrench_world = self.robot.root_physx_view.get_link_incoming_joint_force()[:, self._ee_body_idx]

        # EMA smoothing (direct forge: alpha * raw + (1-alpha) * prev)
        self._force_smooth = smoothing_alpha * raw_wrench_world + (1 - smoothing_alpha) * self._force_smooth

        # Transform force from WORLD frame to HOLE frame
        # Direct forge uses identity quaternion for hole frame (no rotation)
        # So we just use the world-frame force directly
        target_F = self._force_smooth[:, :3]

        # Store force norm on env for contact_force_penalty reward term
        env._force_smooth_norm = torch.norm(target_F, p=2, dim=-1)

        # Add noise (Direct forge: noise ~ N(0,1) * ft_force=1.0)
        noise = torch.randn_like(target_F) * noise_std
        return target_F + noise

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset force sensor smoothing buffer to match direct forge.

        Direct forge explicitly zeroes force_sensor_world_smooth in _reset_idx
        to prevent contact penalty from firing at initialization.
        """
        if env_ids is not None:
            self._force_smooth[env_ids] = 0.0
            if hasattr(self._env, "_force_smooth_norm"):
                self._env._force_smooth_norm[env_ids] = 0.0
        else:
            self._force_smooth.zero_()
            if hasattr(self._env, "_force_smooth_norm"):
                self._env._force_smooth_norm.zero_()


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


class last_action(ManagerTermBase):
    """Last action with roll/pitch components zeroed (direct forge equivalent).

    Direct forge: prev_actions observation has action[:, 3:5] zeroed because
    only yaw rotation is used in the action space.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        actions = env.action_manager.action.clone()
        # Zero roll/pitch components (indices 3 and 4) to match direct forge
        actions[:, 3:5] = 0.0
        return actions
