# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the forge assembly (peg-in-hole) environment.

Custom event/randomization terms replicating direct forge's domain randomization.
Standard terms (reset_scene_to_default, randomize_actuator_gains,
randomize_joint_parameters, randomize_rigid_body_material) come from
isaaclab.envs.mdp via wildcard import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_env_ids(env_ids, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Convert env_ids (which may be slice or Tensor) to a proper tensor index."""
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids


class reset_peg_to_ee(ManagerTermBase):
    """Reset peg pose to the end-effector position."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene["robot"]
        self.peg: RigidObject = env.scene["peg"]
        body_name = cfg.params.get("body_name", "panda_hand")
        body_ids, _ = self.robot.find_bodies(body_name)
        self._ee_body_idx = body_ids[0]
        self._peg_offset = cfg.params.get("peg_offset", [0.0, 0.0, 0.05])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        body_name: str = "panda_hand",
        peg_offset: list[float] | None = None,
    ):
        env_ids = _resolve_env_ids(env_ids, env)
        if peg_offset is None:
            peg_offset = self._peg_offset

        from isaaclab.utils.math import quat_rotate

        ee_pos = self.robot.data.body_pos_w[env_ids, self._ee_body_idx].clone()
        ee_quat = self.robot.data.body_quat_w[env_ids, self._ee_body_idx].clone()

        offset = torch.tensor(peg_offset, device=env.device, dtype=torch.float32).unsqueeze(0).expand(len(env_ids), -1)
        offset_world = quat_rotate(ee_quat, offset)

        pos = ee_pos + offset_world
        quat = ee_quat

        root_state = self.peg.data.default_root_state[env_ids].clone()
        root_state[:, :3] = pos
        root_state[:, 3:7] = quat
        root_state[:, 7:] = 0.0

        self.peg.write_root_state_to_sim(root_state, env_ids)


class randomize_hole_pose(ManagerTermBase):
    """Randomize hole position and orientation on reset."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.hole: RigidObject = env.scene["hole"]
        self._pos_range = cfg.params.get(
            "pos_range", {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [0.0, 0.0]}
        )
        self._yaw_range = cfg.params.get("yaw_range", [-3.14, 3.14])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        pos_range: dict | None = None,
        yaw_range: list[float] | None = None,
    ):
        env_ids = _resolve_env_ids(env_ids, env)
        if pos_range is None:
            pos_range = self._pos_range
        if yaw_range is None:
            yaw_range = self._yaw_range

        import isaacsim.core.utils.torch as torch_utils

        n = len(env_ids)
        root_state = self.hole.data.default_root_state[env_ids].clone()

        for axis, (lo, hi) in pos_range.items():
            idx = {"x": 0, "y": 1, "z": 2}[axis]
            root_state[:, idx] += torch.rand(n, device=env.device) * (hi - lo) + lo

        yaw = torch.rand(n, device=env.device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
        default_quat = self.hole.data.default_root_state[env_ids, 3:7]
        yaw_quat = torch_utils.quat_from_euler_xyz(
            torch.zeros(n, device=env.device),
            torch.zeros(n, device=env.device),
            yaw,
        )
        root_state[:, 3:7] = torch_utils.quat_mul(yaw_quat, default_quat)

        self.hole.write_root_state_to_sim(root_state, env_ids)


class randomize_ema_factor(ManagerTermBase):
    """Randomize EMA smoothing factor per episode."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ema_range = cfg.params.get("ema_range", [0.025, 0.1])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        ema_range: list[float] | None = None,
    ):
        env_ids = _resolve_env_ids(env_ids, env)
        if ema_range is None:
            ema_range = self._ema_range
        if not hasattr(env, "_ema_factor"):
            env._ema_factor = torch.ones(env.num_envs, 1, device=env.device) * ema_range[0]
        env._ema_factor[env_ids] = (
            torch.rand(len(env_ids), 1, device=env.device) * (ema_range[1] - ema_range[0]) + ema_range[0]
        )


class randomize_contact_threshold(ManagerTermBase):
    """Randomize contact penalty threshold per episode."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._threshold_range = cfg.params.get("threshold_range", [5.0, 10.0])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        threshold_range: list[float] | None = None,
    ):
        env_ids = _resolve_env_ids(env_ids, env)
        if threshold_range is None:
            threshold_range = self._threshold_range
        if not hasattr(env, "_contact_penalty_threshold"):
            env._contact_penalty_threshold = torch.ones(env.num_envs, 1, device=env.device) * threshold_range[0]
        env._contact_penalty_threshold[env_ids] = (
            torch.rand(len(env_ids), 1, device=env.device) * (threshold_range[1] - threshold_range[0])
            + threshold_range[0]
        )


class randomize_task_gains(ManagerTermBase):
    """Randomize task-space proportional gains per episode.

    Direct forge: get_random_prop_gains with noise_level (multiplicative).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._noise_level = cfg.params.get("noise_level", 0.41)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        noise_level: float = 0.41,
        default_task_prop_gains: list[float] | None = None,
    ):
        env_ids = _resolve_env_ids(env_ids, env)
        if default_task_prop_gains is None:
            default_task_prop_gains = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]

        if not hasattr(env, "_task_prop_gains"):
            env._task_prop_gains = torch.zeros(env.num_envs, 6, device=env.device)
            env._task_deriv_gains = torch.zeros(env.num_envs, 6, device=env.device)

        n = len(env_ids)
        defaults = torch.tensor(default_task_prop_gains, device=env.device).unsqueeze(0).expand(n, -1)

        noise = torch.rand((n, 6), dtype=torch.float32, device=env.device)
        multiplier = 1.0 + noise * noise_level
        decrease_flag = torch.rand((n, 6), dtype=torch.float32, device=env.device) > 0.5
        multiplier = torch.where(decrease_flag, 1.0 / multiplier, multiplier)

        env._task_prop_gains[env_ids] = defaults * multiplier
        env._task_deriv_gains[env_ids] = 2.0 * torch.sqrt(env._task_prop_gains[env_ids])


class randomize_pos_threshold(ManagerTermBase):
    """Randomize position clipping threshold per episode."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._noise_level = cfg.params.get("noise_level", 0.25)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        noise_level: float = 0.25,
        default_pos_threshold: list[float] | None = None,
    ):
        env_ids = _resolve_env_ids(env_ids, env)
        if default_pos_threshold is None:
            default_pos_threshold = [0.02, 0.02, 0.02]

        if not hasattr(env, "_pos_threshold"):
            env._pos_threshold = torch.zeros(env.num_envs, 3, device=env.device)

        n = len(env_ids)
        defaults = torch.tensor(default_pos_threshold, device=env.device).unsqueeze(0).expand(n, -1)

        noise = torch.rand((n, 3), dtype=torch.float32, device=env.device)
        multiplier = 1.0 + noise * noise_level
        decrease_flag = torch.rand((n, 3), dtype=torch.float32, device=env.device) > 0.5
        multiplier = torch.where(decrease_flag, 1.0 / multiplier, multiplier)

        env._pos_threshold[env_ids] = defaults * multiplier


class randomize_rot_threshold(ManagerTermBase):
    """Randomize rotation clipping threshold per episode."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._noise_level = cfg.params.get("noise_level", 0.29)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        noise_level: float = 0.29,
        default_rot_threshold: list[float] | None = None,
    ):
        env_ids = _resolve_env_ids(env_ids, env)
        if default_rot_threshold is None:
            default_rot_threshold = [0.097, 0.097, 0.097]

        if not hasattr(env, "_rot_threshold"):
            env._rot_threshold = torch.zeros(env.num_envs, 3, device=env.device)

        n = len(env_ids)
        defaults = torch.tensor(default_rot_threshold, device=env.device).unsqueeze(0).expand(n, -1)

        noise = torch.rand((n, 3), dtype=torch.float32, device=env.device)
        multiplier = 1.0 + noise * noise_level
        decrease_flag = torch.rand((n, 3), dtype=torch.float32, device=env.device) > 0.5
        multiplier = torch.where(decrease_flag, 1.0 / multiplier, multiplier)

        env._rot_threshold[env_ids] = defaults * multiplier


class randomize_dead_zone(ManagerTermBase):
    """Randomize dead zone thresholds per episode.

    Direct forge: dead_zone = rand() * default_dead_zone (uniform [0, default]).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._default_dead_zone = cfg.params.get(
            "default_dead_zone", [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        default_dead_zone: list[float] | None = None,
    ):
        env_ids = _resolve_env_ids(env_ids, env)
        if default_dead_zone is None:
            default_dead_zone = self._default_dead_zone

        action_term = env.action_manager._terms["arm_action"]
        n = len(env_ids)
        defaults = torch.tensor(default_dead_zone, device=env.device).unsqueeze(0).expand(n, -1)
        action_term._dead_zone[env_ids] = torch.rand((n, 6), device=env.device) * defaults
