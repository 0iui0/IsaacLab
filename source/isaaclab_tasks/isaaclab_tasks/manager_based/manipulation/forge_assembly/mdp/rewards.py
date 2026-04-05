# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the forge assembly (peg-in-hole) environment.

Custom reward terms replicating direct forge's multi-scale keypoint tracking,
contact penalty, and action penalties. Standard terms (action_rate_l2, action_l2)
come from isaaclab.envs.mdp via wildcard import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# Keypoint helpers
##


def _get_keypoint_offsets(device: torch.device | None = None) -> torch.Tensor:
    """Get keypoint offsets for pose alignment (6-axis: +/-x, +/-y, +/-z)."""
    corners = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    corners = torch.tensor(corners, device=device, dtype=torch.float32)
    return torch.cat((corners, -corners), dim=0)  # (6, 3)


def _compute_keypoint_distance(
    pos1: torch.Tensor,
    quat1: torch.Tensor,
    pos2: torch.Tensor,
    quat2: torch.Tensor,
    keypoint_scale: float,
    identity_quat: torch.Tensor,
) -> torch.Tensor:
    """Compute keypoint distances between two poses.

    Returns:
        Tensor of shape (num_envs, num_keypoints) with L2 distances.
    """
    num_envs = pos1.shape[0]
    offsets = _get_keypoint_offsets(pos1.device) * keypoint_scale  # (6, 3)

    # Expand for all keypoints
    offsets_exp = offsets.unsqueeze(0).expand(num_envs, -1, -1).reshape(-1, 3)
    identity_exp = identity_quat[: offsets_exp.shape[0]]

    # Transform keypoints by each pose
    quat1_exp = quat1.unsqueeze(1).expand(-1, 6, -1).reshape(-1, 4)
    pos1_exp = pos1.unsqueeze(1).expand(-1, 6, -1).reshape(-1, 3)
    quat2_exp = quat2.unsqueeze(1).expand(-1, 6, -1).reshape(-1, 4)
    pos2_exp = pos2.unsqueeze(1).expand(-1, 6, -1).reshape(-1, 3)

    kp1, _ = combine_frame_transforms(pos1_exp, quat1_exp, offsets_exp, identity_exp)
    kp2, _ = combine_frame_transforms(pos2_exp, quat2_exp, offsets_exp, identity_exp)

    kp1 = kp1.reshape(num_envs, 6, 3)
    kp2 = kp2.reshape(num_envs, 6, 3)

    return torch.norm(kp2 - kp1, p=2, dim=-1)  # (num_envs, 6)


##
# Reward terms
##


class keypoint_peg_hole_error(ManagerTermBase):
    """Keypoint distance between peg and hole with squashing function.

    Direct forge equivalent: kp_baseline, kp_coarse, kp_fine rewards.
    Uses squashing function: d_sq = d / (1 + d/a)^b to compress large errors.
    Different (a, b) params create different reward scales (baseline/coarse/fine).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.peg: RigidObject = env.scene["peg"]
        self.hole: RigidObject = env.scene["hole"]
        self._identity_quat = (
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device, dtype=torch.float32)
            .repeat(env.num_envs * 6, 1)
            .contiguous()
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        keypoint_scale: float = 0.15,
        squash_a: float = 10.0,
        squash_b: float = 5.0,
    ) -> torch.Tensor:
        peg_pos = self.peg.data.root_pos_w - env.scene.env_origins
        peg_quat = self.peg.data.root_quat_w
        hole_pos = self.hole.data.root_pos_w - env.scene.env_origins
        hole_quat = self.hole.data.root_quat_w

        kp_dist = _compute_keypoint_distance(peg_pos, peg_quat, hole_pos, hole_quat, keypoint_scale, self._identity_quat)

        # Squashing: d_sq = d / (1 + d/a)^b
        d_sq = kp_dist / (1.0 + kp_dist / squash_a) ** squash_b
        return d_sq.mean(dim=-1)


class keypoint_peg_hole_error_exp(ManagerTermBase):
    """Exponential keypoint reward: r = 1/(exp(a*d) + b + exp(-a*d)).

    Direct forge equivalent: kp_exp reward with sum of exponentials.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.peg: RigidObject = env.scene["peg"]
        self.hole: RigidObject = env.scene["hole"]
        self._identity_quat = (
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device, dtype=torch.float32)
            .repeat(env.num_envs * 6, 1)
            .contiguous()
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        keypoint_scale: float = 0.15,
        kp_exp_coeffs: list[tuple[float, float]] = [(50, 0.0001), (300, 0.0001)],
        kp_use_sum_of_exps: bool = False,
    ) -> torch.Tensor:
        peg_pos = self.peg.data.root_pos_w - env.scene.env_origins
        peg_quat = self.peg.data.root_quat_w
        hole_pos = self.hole.data.root_pos_w - env.scene.env_origins
        hole_quat = self.hole.data.root_quat_w

        kp_dist = _compute_keypoint_distance(peg_pos, peg_quat, hole_pos, hole_quat, keypoint_scale, self._identity_quat)

        if kp_use_sum_of_exps:
            kp_dist_mean = kp_dist  # per-keypoint
        else:
            kp_dist_mean = kp_dist.mean(dim=-1, keepdim=True)  # scalar per env

        reward = torch.zeros(env.num_envs, device=env.device)
        for a, b in kp_exp_coeffs:
            reward += (1.0 / (torch.exp(a * kp_dist_mean) + b + torch.exp(-a * kp_dist_mean))).mean(dim=-1)
        return reward


class contact_force_penalty(ManagerTermBase):
    """Contact force penalty: relu(||force|| - threshold).

    Direct forge equivalent: contact_penalty reward.
    Uses the smoothed force norm stored on the environment by ft_force_smooth_noisy.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        if not hasattr(env, "_force_smooth_norm"):
            return torch.zeros(env.num_envs, device=env.device)
        force_norm = env._force_smooth_norm
        threshold = env._contact_penalty_threshold.squeeze(-1)
        return torch.nn.functional.relu(force_norm - threshold)


class action_penalty_asset(ManagerTermBase):
    """Asset-relative action penalty: normalized position + yaw error.

    Direct forge equivalent: action_penalty_asset = pos_error/threshold + |delta_yaw|/rot_threshold.
    Reads delta_pos and delta_yaw from the action term.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        # Read from action term state (set during apply_actions)
        action_term = env.action_manager._terms["arm_action"]
        pos_error = torch.norm(action_term.delta_pos, p=2, dim=-1)
        # Normalize by position threshold
        pos_threshold = action_term._pos_threshold[:, 0]  # Use x threshold as scale
        pos_error_norm = pos_error / pos_threshold
        # Yaw error
        yaw_error = torch.abs(action_term.delta_yaw.squeeze(-1))
        rot_threshold = action_term._rot_threshold[:, 2]  # Use yaw threshold
        yaw_error_norm = yaw_error / rot_threshold
        return pos_error_norm + yaw_error_norm


class peg_insertion_engaged(ManagerTermBase):
    """Binary bonus when peg is inserted > 90% of hole depth.

    Direct forge equivalent: curr_engaged reward (weight +1.0).
    Success = xy_dist < 2.5mm AND z_disp < height * engage_threshold (0.9).
    Peg insert: height=0.025, engage_threshold=0.9 → z_disp < 0.0225m.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.peg: RigidObject = env.scene["peg"]
        self.hole: RigidObject = env.scene["hole"]
        self._xy_threshold = 0.0025  # 2.5mm
        self._engage_threshold = 0.9  # 90% of hole height

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        peg_pos = self.peg.data.root_pos_w - env.scene.env_origins
        hole_pos = self.hole.data.root_pos_w - env.scene.env_origins

        xy_dist = torch.linalg.vector_norm(peg_pos[:, :2] - hole_pos[:, :2], dim=1)
        z_disp = peg_pos[:, 2] - hole_pos[:, 2]

        is_centered = xy_dist < self._xy_threshold
        is_engaged = z_disp < (0.025 * self._engage_threshold)
        return torch.logical_and(is_centered, is_engaged).float()


class peg_insertion_success(ManagerTermBase):
    """Binary bonus when peg is fully inserted.

    Direct forge equivalent: curr_success reward (weight +1.0).
    Success = xy_dist < 2.5mm AND z_disp < height * success_threshold (0.04).
    Peg insert: height=0.025, success_threshold=0.04 → z_disp < 0.001m.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.peg: RigidObject = env.scene["peg"]
        self.hole: RigidObject = env.scene["hole"]
        self._xy_threshold = 0.0025  # 2.5mm
        self._success_threshold = 0.04  # 4% of hole height

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        peg_pos = self.peg.data.root_pos_w - env.scene.env_origins
        hole_pos = self.hole.data.root_pos_w - env.scene.env_origins

        xy_dist = torch.linalg.vector_norm(peg_pos[:, :2] - hole_pos[:, :2], dim=1)
        z_disp = peg_pos[:, 2] - hole_pos[:, 2]

        is_centered = xy_dist < self._xy_threshold
        is_below = z_disp < (0.025 * self._success_threshold)
        return torch.logical_and(is_centered, is_below).float()


class success_prediction_penalty(ManagerTermBase):
    """Success prediction penalty: |true_success - predicted_success|.

    Direct forge equivalent: success_pred_error reward.
    Uses proximity-based success (peg within 3D distance threshold) since
    the hole is solid and physical insertion is not possible.

    - Reads the 7th action dimension (success prediction) and rescales from [-1,1] to [0,1]
    - Computes true success (peg within 3D proximity threshold)
    - Returns absolute difference
    - Scales by success_pred_scale (starts 0, becomes 1.0 when 25%+ envs succeed)
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.peg: RigidObject = env.scene["peg"]
        self.hole: RigidObject = env.scene["hole"]
        self._success_pred_scale = 0.0
        self._delay_until_ratio = 0.25
        # Proximity-based success thresholds
        self._xy_threshold = 0.003  # 3mm xy alignment
        self._z_threshold = 0.003   # 3mm z proximity (not penetration)

    def _get_true_successes(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute true success: peg close to hole in all axes."""
        peg_pos = self.peg.data.root_pos_w - env.scene.env_origins
        hole_pos = self.hole.data.root_pos_w - env.scene.env_origins

        xy_dist = torch.linalg.vector_norm(peg_pos[:, :2] - hole_pos[:, :2], dim=1)
        z_dist = torch.abs(peg_pos[:, 2] - hole_pos[:, 2])

        is_centered = xy_dist < self._xy_threshold
        is_close_z = z_dist < self._z_threshold
        return torch.logical_and(is_centered, is_close_z)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        # Get policy's success prediction: action dim 6, rescaled [-1,1] → [0,1]
        actions = env.action_manager.action
        policy_success_pred = (actions[:, 6] + 1.0) / 2.0

        # Get true success
        true_successes = self._get_true_successes(env)

        # Delay activation until 25% of envs are succeeding
        if true_successes.float().mean() >= self._delay_until_ratio:
            self._success_pred_scale = 1.0

        success_pred_error = (true_successes.float() - policy_success_pred).abs()

        # Store for logging
        env._success_pred_error = success_pred_error
        env._true_success_rate = true_successes.float().mean()
        env._success_pred_scale = self._success_pred_scale

        return success_pred_error * self._success_pred_scale
