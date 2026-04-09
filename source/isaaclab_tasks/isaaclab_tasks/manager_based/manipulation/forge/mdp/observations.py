# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for FORGE tasks matching direct version exactly.

Policy observations (27D total):
    fingertip_pos_rel_fixed:  3D  (noisy absolute fingertip - noisy fixed frame)
    fingertip_quat:           4D  (noisy, Gaussian noise on rotation)
    ee_linvel:                3D  (finite difference from noisy pos)
    ee_angvel:                3D  (finite difference from noisy quat)
    ft_force:                 3D  (smoothed EMA, Gaussian noise)
    force_threshold:          1D  (randomized 5-10N)
    prev_actions:             7D  (from action term, with [:,3:5]=0)

Critic observations (~64D total):
    fingertip_pos:            3D  (raw absolute)
    fingertip_quat:           4D  (raw absolute)
    ee_linvel:                3D  (raw linear velocity)
    ee_angvel:                3D  (raw angular velocity)
    joint_pos:                7D  (arm joint positions)
    held_pos:                 3D
    held_pos_rel_fixed:       3D
    held_quat:                4D
    fixed_pos:                3D
    fixed_quat:               4D
    task_prop_gains:          6D
    ema_factor:               1D
    ft_force:                 3D  (smoothed, no noise)
    pos_threshold:            3D
    rot_threshold:            3D
    force_threshold:          1D
    prev_actions:             7D
"""

from __future__ import annotations

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_action_term(env: ManagerBasedRLEnv):
    """Get the ForgeImpedanceAction term from the action manager."""
    for name, term in env.action_manager._terms.items():
        if hasattr(term, "noisy_fingertip_pos"):
            return term
    raise RuntimeError("ForgeImpedanceAction term not found in action manager")


# ---------------------------------------------------------------------------
# Policy observations
# ---------------------------------------------------------------------------


def fingertip_pos_rel_fixed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    fixed_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Fingertip position relative to fixed asset (3D, noisy).

    Matches direct: noisy_fingertip_pos - noisy_fixed_pos.
    """
    action_term = _get_action_term(env)
    noisy_fixed_pos = action_term.fixed_pos_obs_frame + action_term.init_fixed_pos_obs_noise
    return action_term.noisy_fingertip_pos - noisy_fixed_pos


def fingertip_quat(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Noisy fingertip quaternion (4D).

    Matches direct: noisy_fingertip_quat with [:,[0,3]]=0 and flip.
    """
    action_term = _get_action_term(env)
    return action_term.noisy_fingertip_quat


def ee_linvel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EE linear velocity via finite differencing from noisy pos (3D).

    Matches direct: ee_linvel_fd.
    """
    action_term = _get_action_term(env)
    return action_term.ee_linvel_fd


def ee_angvel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EE angular velocity via finite differencing from noisy quat (3D).

    Matches direct: ee_angvel_fd with [:,0:2]=0.
    """
    action_term = _get_action_term(env)
    return action_term.ee_angvel_fd


def ft_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    fixed_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Noisy force sensor in fixed asset frame (3D).

    Matches direct: noisy_force = smoothed + Gaussian noise.
    """
    action_term = _get_action_term(env)
    return action_term.noisy_force


def force_threshold(
    env: ManagerBasedRLEnv,
    lower_bound: float = 5.0,
    upper_bound: float = 10.0,
) -> torch.Tensor:
    """Randomized contact penalty threshold (1D)."""
    action_term = _get_action_term(env)
    return action_term.contact_penalty_thresholds.unsqueeze(-1)


def prev_actions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous actions (7D) from action term.

    Matches direct: prev_actions with [:,3:5]=0.
    """
    action_term = _get_action_term(env)
    actions = action_term.prev_actions.clone()
    actions[:, 3:5] = 0.0
    return actions


# ---------------------------------------------------------------------------
# Critic observations
# ---------------------------------------------------------------------------


def fingertip_pos_raw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Raw fingertip position in environment frame (3D)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_pos_w[:, asset_cfg.body_ids[0]] - env.scene.env_origins


def fingertip_quat_raw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Raw fingertip quaternion (4D)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_quat_w[:, asset_cfg.body_ids[0]]


def ee_linvel_raw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Raw fingertip linear velocity (3D)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_lin_vel_w[:, asset_cfg.body_ids[0]]


def ee_angvel_raw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Raw fingertip angular velocity (3D)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_ang_vel_w[:, asset_cfg.body_ids[0]]


def joint_pos_arm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Arm joint positions (7D)."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = robot.find_joints(["panda_joint.*"])
    return robot.data.joint_pos[:, joint_ids]


def held_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Held asset position in environment frame (3D)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def held_pos_rel_fixed(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Held asset position relative to fixed asset observation frame (3D)."""
    action_term = _get_action_term(env)
    held_asset: Articulation = env.scene[held_cfg.name]
    held_p = held_asset.data.root_pos_w - env.scene.env_origins
    return held_p - action_term.fixed_pos_obs_frame


def held_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Held asset quaternion (4D)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_quat_w


def fixed_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Fixed asset position in environment frame (3D)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def fixed_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Fixed asset quaternion (4D)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_quat_w


def task_prop_gains(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current task-space proportional gains (6D)."""
    action_term = _get_action_term(env)
    return action_term.task_prop_gains


def ema_factor(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current EMA smoothing factor (1D)."""
    action_term = _get_action_term(env)
    return action_term.ema_factor


def ft_force_raw(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    fixed_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Smoothed force for critic (3D, no noise)."""
    action_term = _get_action_term(env)
    return action_term.force_sensor_smooth[:, 0:3]


def pos_threshold(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Position action thresholds (3D)."""
    action_term = _get_action_term(env)
    return action_term.pos_threshold


def rot_threshold(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Rotation action thresholds (3D)."""
    action_term = _get_action_term(env)
    return action_term.rot_threshold


def force_threshold_critic(
    env: ManagerBasedRLEnv,
    lower_bound: float = 5.0,
    upper_bound: float = 10.0,
) -> torch.Tensor:
    """Contact penalty threshold for critic (1D)."""
    action_term = _get_action_term(env)
    return action_term.contact_penalty_thresholds.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Keypoint computation (shared between rewards and terminations)
# ---------------------------------------------------------------------------


def get_keypoint_offsets(num_keypoints: int, device: torch.device) -> torch.Tensor:
    """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""
    keypoint_offsets = torch.zeros((num_keypoints, 3), device=device)
    keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=device) - 0.5
    return keypoint_offsets


def get_held_base_pose(
    held_pos, held_quat, task_name, fixed_asset_cfg, num_envs, device
):
    """Get current held asset base pose for keypoint computation."""
    held_base_z_offset = 0.0
    held_base_x_offset = 0.0
    if task_name == "peg_insert":
        held_base_z_offset = 0.0
    elif task_name == "gear_mesh":
        gear_base_offset = fixed_asset_cfg.medium_gear_base_offset
        held_base_x_offset = gear_base_offset[0]
        held_base_z_offset = gear_base_offset[2]
    elif task_name == "nut_thread":
        held_base_z_offset = fixed_asset_cfg.base_height

    held_base_pos_local = torch.zeros((num_envs, 3), device=device)
    held_base_pos_local[:, 0] = held_base_x_offset
    held_base_pos_local[:, 2] = held_base_z_offset
    held_base_quat_local = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)
    )
    held_base_quat, held_base_pos = torch_utils.tf_combine(
        held_quat, held_pos, held_base_quat_local, held_base_pos_local
    )
    return held_base_pos, held_base_quat


def get_target_held_base_pose(
    fixed_pos, fixed_quat, task_name, fixed_asset_cfg, num_envs, device
):
    """Get target held asset base pose for keypoint computation."""
    fixed_success_pos_local = torch.zeros((num_envs, 3), device=device)
    if task_name == "peg_insert":
        fixed_success_pos_local[:, 2] = 0.0
    elif task_name == "gear_mesh":
        gear_base_offset = fixed_asset_cfg.medium_gear_base_offset
        fixed_success_pos_local[:, 0] = gear_base_offset[0]
        fixed_success_pos_local[:, 2] = gear_base_offset[2]
    elif task_name == "nut_thread":
        head_height = fixed_asset_cfg.base_height
        shank_length = fixed_asset_cfg.height
        thread_pitch = fixed_asset_cfg.thread_pitch
        fixed_success_pos_local[:, 2] = head_height + shank_length - thread_pitch * 1.5
    fixed_success_quat_local = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)
    )
    target_held_base_quat, target_held_base_pos = torch_utils.tf_combine(
        fixed_quat, fixed_pos, fixed_success_quat_local, fixed_success_pos_local
    )
    return target_held_base_pos, target_held_base_quat


def compute_keypoint_distance(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    num_keypoints: int,
    keypoint_scale: float,
    task_name: str,
    fixed_asset_cfg,
) -> torch.Tensor:
    """Compute mean keypoint distance between held and fixed assets (num_envs,)."""
    held_asset: Articulation = env.scene[held_cfg.name]
    fixed_asset: Articulation = env.scene[fixed_cfg.name]

    held_pos_w = held_asset.data.root_pos_w - env.scene.env_origins
    held_quat_w = held_asset.data.root_quat_w
    fixed_pos_w = fixed_asset.data.root_pos_w - env.scene.env_origins
    fixed_quat_w = fixed_asset.data.root_quat_w

    held_base_pos, held_base_quat = get_held_base_pose(
        held_pos_w, held_quat_w, task_name, fixed_asset_cfg, env.num_envs, env.device
    )
    target_held_base_pos, target_held_base_quat = get_target_held_base_pose(
        fixed_pos_w, fixed_quat_w, task_name, fixed_asset_cfg, env.num_envs, env.device
    )

    offsets = get_keypoint_offsets(num_keypoints, env.device)
    keypoint_offsets = offsets * keypoint_scale

    identity_quat = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
        .unsqueeze(0)
        .repeat(env.num_envs, 1)
    )

    keypoints_held = torch.zeros((env.num_envs, num_keypoints, 3), device=env.device)
    keypoints_fixed = torch.zeros((env.num_envs, num_keypoints, 3), device=env.device)

    for idx in range(num_keypoints):
        offset = keypoint_offsets[idx].repeat(env.num_envs, 1)
        _, kp_held = torch_utils.tf_combine(held_base_quat, held_base_pos, identity_quat, offset)
        keypoints_held[:, idx] = kp_held
        _, kp_fixed = torch_utils.tf_combine(
            target_held_base_quat, target_held_base_pos, identity_quat, offset
        )
        keypoints_fixed[:, idx] = kp_fixed

    return torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)


def squashing_fn(x, a, b):
    """Compute bounded reward function: 1 / (exp(ax) + b + exp(-ax))."""
    return 1.0 / (torch.exp(a * x) + b + torch.exp(-a * x))
