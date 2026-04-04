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
