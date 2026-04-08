# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for FORGE tasks matching direct version exactly.

Success detection using keypoint-based approach:
    - Compute held asset base pose and target base pose
    - Check XY distance < 0.0025m (is_centered)
    - Check Z displacement < height * success_threshold (is_close_or_below)
    - Optional rotation check for nut_thread task

The termination returns time_out=True for all envs (matching Factory behavior where
all envs reset together). The success signal is communicated via extras for logging.
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
    get_held_base_pose,
    get_target_held_base_pose,
)


def _check_success(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    success_threshold: float,
    check_rot: bool,
    task_name: str,
    fixed_asset_cfg,
    ee_success_yaw: float = 0.0,
) -> torch.Tensor:
    """Check if the task has been completed successfully.

    Matches direct: ForgeEnv._get_curr_successes().

    Args:
        env: The environment instance.
        held_cfg: Configuration for the held asset.
        fixed_cfg: Configuration for the fixed asset.
        success_threshold: Distance threshold for success.
        check_rot: Whether to check rotation (nut_thread).
        task_name: Name of the task ("peg_insert", "gear_mesh", "nut_thread").
        fixed_asset_cfg: Fixed asset configuration with height/base_height/etc.
        ee_success_yaw: Yaw threshold for rotation check.

    Returns:
        Boolean tensor of shape (num_envs,) indicating success.
    """
    held_asset: RigidObject = env.scene[held_cfg.name]
    fixed_asset: RigidObject = env.scene[fixed_cfg.name]

    held_pos = held_asset.data.root_pos_w - env.scene.env_origins
    held_quat = held_asset.data.root_quat_w
    fixed_pos = fixed_asset.data.root_pos_w - env.scene.env_origins
    fixed_quat = fixed_asset.data.root_quat_w

    held_base_pos, _ = get_held_base_pose(
        held_pos, held_quat, task_name, fixed_asset_cfg, env.num_envs, env.device
    )
    target_held_base_pos, _ = get_target_held_base_pose(
        fixed_pos, fixed_quat, task_name, fixed_asset_cfg, env.num_envs, env.device
    )

    xy_dist = torch.linalg.vector_norm(
        target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1
    )
    z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

    curr_successes = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    is_centered = xy_dist < 0.0025

    # Height threshold depends on task type
    if task_name in ("peg_insert", "gear_mesh"):
        height_threshold = fixed_asset_cfg.height * success_threshold
    elif task_name == "nut_thread":
        height_threshold = fixed_asset_cfg.thread_pitch * success_threshold
    else:
        raise NotImplementedError(f"Task '{task_name}' not implemented")

    is_close_or_below = z_disp < height_threshold
    curr_successes = torch.logical_and(is_centered, is_close_or_below)

    # Optional rotation check (for nut_thread)
    if check_rot:
        action_term = _get_action_term(env)
        _, _, curr_yaw = torch_utils.get_euler_xyz(action_term.noisy_fingertip_quat)
        curr_yaw = _wrap_yaw(curr_yaw)
        is_rotated = curr_yaw < ee_success_yaw
        curr_successes = torch.logical_and(curr_successes, is_rotated)

    return curr_successes


def _wrap_yaw(angle):
    """Wrap yaw to [-125, 235] degrees range."""
    return torch.where(angle > np.deg2rad(235), angle - 2 * np.pi, angle)


def task_success(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    success_threshold: float,
    check_rot: bool,
    task_name: str,
    fixed_asset_cfg,
    ee_success_yaw: float = 0.0,
) -> torch.Tensor:
    """Check if the task is successful (for logging/extras).

    Note: In the direct version, termination returns time_out for ALL envs
    (all environments reset together). Success is only communicated via extras.
    This function always returns False (never triggers early termination).

    Returns:
        Always returns False (time_out handles termination).
    """
    # Compute success and log in extras
    successes = _check_success(
        env, held_cfg, fixed_cfg, success_threshold, check_rot, task_name, fixed_asset_cfg, ee_success_yaw
    )

    # Log success rate in extras
    if not hasattr(env, "_ep_succeeded"):
        env._ep_succeeded = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
        env._ep_success_times = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)

    first_success = torch.logical_and(successes, torch.logical_not(env._ep_succeeded))
    env._ep_succeeded[successes] = 1
    first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
    env._ep_success_times[first_success_ids] = env.episode_length_buf[first_success_ids]

    # Never trigger early termination - let time_out handle it
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check if the episode has timed out.

    Matches direct: time_out = episode_length_buf >= max_episode_length - 1.
    """
    return env.episode_length_buf >= env.max_episode_length - 1
