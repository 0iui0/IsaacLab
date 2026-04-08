# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for FORGE tasks matching direct version exactly.

Reward terms (matching factory_env._get_factory_rew_dict and forge_env._get_rewards):
    kp_baseline:    1.0 * squashing(keypoint_dist, [5, 4])
    kp_coarse:      1.0 * squashing(keypoint_dist, [50, 2])
    kp_fine:        1.0 * squashing(keypoint_dist, [100, 0])
    action_penalty_ee:    -scale * ||actions||_2
    action_grad_penalty:  -scale * ||actions - prev_actions||_2
    curr_engaged:    1.0 * engaged_flag
    curr_success:    1.0 * success_flag
    action_penalty_asset: -scale * (pos_err + yaw_err)
    contact_penalty: -scale * ReLU(force - threshold)
    success_pred_error: -scale * |true_success - predicted_success|
"""

from __future__ import annotations

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .observations import (
    _get_action_term,
    compute_keypoint_distance,
    get_held_base_pose,
    get_target_held_base_pose,
    squashing_fn,
)


# ---------------------------------------------------------------------------
# Factory base rewards (shared across all Factory/FORGE tasks)
# ---------------------------------------------------------------------------


def kp_baseline(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    num_keypoints: int,
    keypoint_scale: float,
    task_name: str,
    fixed_asset_cfg,
    coef: list[float],
) -> torch.Tensor:
    """Baseline keypoint distance reward with squashing function."""
    kp_dist = compute_keypoint_distance(
        env, held_cfg, fixed_cfg, num_keypoints, keypoint_scale, task_name, fixed_asset_cfg
    )
    return squashing_fn(kp_dist, coef[0], coef[1])


def kp_coarse(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    num_keypoints: int,
    keypoint_scale: float,
    task_name: str,
    fixed_asset_cfg,
    coef: list[float],
) -> torch.Tensor:
    """Coarse keypoint distance reward with squashing function."""
    kp_dist = compute_keypoint_distance(
        env, held_cfg, fixed_cfg, num_keypoints, keypoint_scale, task_name, fixed_asset_cfg
    )
    return squashing_fn(kp_dist, coef[0], coef[1])


def kp_fine(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    num_keypoints: int,
    keypoint_scale: float,
    task_name: str,
    fixed_asset_cfg,
    coef: list[float],
) -> torch.Tensor:
    """Fine keypoint distance reward with squashing function."""
    kp_dist = compute_keypoint_distance(
        env, held_cfg, fixed_cfg, num_keypoints, keypoint_scale, task_name, fixed_asset_cfg
    )
    return squashing_fn(kp_dist, coef[0], coef[1])


def action_penalty_ee(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 norm of current (EMA-smoothed) actions.

    Matches direct: torch.norm(self.actions, p=2).
    """
    action_term = _get_action_term(env)
    return torch.norm(action_term.prev_actions, p=2, dim=-1)


def action_grad_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 norm of action gradient (current - previous).

    Matches direct: torch.norm(self.actions - self.prev_actions, p=2, dim=-1).
    In the action term, prev_actions is updated at the end of apply_actions(),
    so at reward computation time: prev_actions = actions from 2 steps ago,
    and _processed_actions = current EMA-smoothed actions. The gradient is
    computed as the difference between these two.
    """
    action_term = _get_action_term(env)
    current_actions = action_term._processed_actions
    return torch.norm(current_actions - action_term.prev_actions, p=2, dim=-1)


def curr_engaged(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    engage_threshold: float,
    task_name: str,
    fixed_asset_cfg,
) -> torch.Tensor:
    """Whether held asset is engaged with fixed asset (keypoint-based).

    Matches direct: _get_curr_successes(success_threshold=engage_threshold, check_rot=False).
    """
    from .terminations import _check_success
    return _check_success(env, held_cfg, fixed_cfg, engage_threshold, False, task_name, fixed_asset_cfg).float()


def curr_success(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    success_threshold: float,
    check_rot: bool,
    task_name: str,
    fixed_asset_cfg,
    ee_success_yaw: float = 0.0,
) -> torch.Tensor:
    """Whether task is currently successful (keypoint-based).

    Matches direct: _get_curr_successes(success_threshold, check_rot).
    """
    from .terminations import _check_success
    return _check_success(
        env, held_cfg, fixed_cfg, success_threshold, check_rot, task_name, fixed_asset_cfg, ee_success_yaw
    ).float()


# ---------------------------------------------------------------------------
# FORGE-specific rewards
# ---------------------------------------------------------------------------


def action_penalty_asset(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.02,
    rot_threshold: float = 0.1,
) -> torch.Tensor:
    """Asset-relative action penalty: normalized pos error + normalized yaw error.

    Matches direct: pos_error + rot_error.
    """
    action_term = _get_action_term(env)
    pos_error = torch.norm(action_term.delta_pos, p=2, dim=-1) / pos_threshold
    rot_error = torch.abs(action_term.delta_yaw) / rot_threshold
    return pos_error + rot_error


def contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    fixed_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Contact penalty: ReLU(force_z - threshold).

    Matches direct: relu(contact_force - threshold).
    """
    action_term = _get_action_term(env)
    contact_force = torch.norm(action_term.force_sensor_smooth[:, 0:3], p=2, dim=-1)
    penalty = torch.nn.functional.relu(contact_force - action_term.contact_penalty_thresholds)
    return penalty


def success_pred_error(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    success_threshold: float,
    check_rot: bool,
    task_name: str,
    fixed_asset_cfg,
    delay_until_ratio: float = 0.25,
    ee_success_yaw: float = 0.0,
) -> torch.Tensor:
    """Success prediction error: |true_success - predicted_success|.

    Matches direct: success_pred_error with delay_until_ratio gating.

    Note: The scale for this reward is applied dynamically based on success rate.
    In the manager-based version, the weight should be set to -1.0 and a curriculum
    event enables it once enough successes have occurred.
    """
    action_term = _get_action_term(env)
    from .terminations import _check_success

    true_successes = _check_success(
        env, held_cfg, fixed_cfg, success_threshold, check_rot, task_name, fixed_asset_cfg, ee_success_yaw
    ).float()

    policy_success_pred = (action_term.prev_actions[:, 6] + 1.0) / 2.0  # Rescale [-1,1] -> [0,1]
    error = (true_successes - policy_success_pred).abs()
    return error
