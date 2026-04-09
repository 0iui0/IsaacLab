# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for FORGE tasks matching direct version's randomization.

Randomization events matching ForgeEnv._reset_idx() and ForgeEventCfg:
- randomize_fixed_asset_pose: Position/orientation noise on fixed asset
- randomize_impedance_gains: Kp +/- 41% multiplicative noise, Kd = 2*sqrt(Kp)
- randomize_action_thresholds: pos +/- 25%, rot +/- 29%
- randomize_ema_factor: uniform [0.025, 0.1]
- randomize_dead_zone: [5,5,5,1,1,1] * random
- randomize_contact_threshold: uniform [5, 10]
- Interval-based dead zone re-randomization every 2s
"""

from __future__ import annotations

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .actions import get_random_prop_gains


def _get_action_term(env: ManagerBasedRLEnv):
    """Get the ForgeImpedanceAction term from the action manager."""
    for name, term in env.action_manager._terms.items():
        if hasattr(term, "noisy_fingertip_pos"):
            return term
    raise RuntimeError("ForgeImpedanceAction term not found in action manager")


def randomize_fixed_asset_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    fixed_cfg: SceneEntityCfg,
    pos_noise: list[float] = [0.05, 0.05, 0.05],
    orn_init_deg: float = 0.0,
    orn_range_deg: float = 360.0,
):
    """Randomize fixed asset position and orientation.

    Matches direct FactoryEnv.randomize_initial_state() step (1):
    - Add uniform noise to position
    - Randomize yaw orientation

    Args:
        env: The environment instance.
        env_ids: Environment IDs to randomize.
        fixed_cfg: Scene entity config for the fixed asset.
        pos_noise: Position noise range [x, y, z].
        orn_init_deg: Initial orientation in degrees.
        orn_range_deg: Orientation range in degrees.
    """
    fixed_asset: Articulation = env.scene[fixed_cfg.name]

    if env_ids is None or len(env_ids) == env.num_envs:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Get current state
    fixed_state = fixed_asset.data.root_state_w[env_ids].clone()

    # (1.a) Position noise
    rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=env.device)
    pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
    pos_noise_tensor = torch.tensor(pos_noise, dtype=torch.float32, device=env.device)
    pos_init_rand = pos_init_rand @ torch.diag(pos_noise_tensor)

    # Add noise relative to default position (remove env origin first, add noise, then add back)
    default_pos = fixed_asset.data.default_root_state[env_ids, 0:3]
    fixed_state[:, 0:3] = default_pos + pos_init_rand + env.scene.env_origins[env_ids]

    # (1.b) Orientation noise (yaw only)
    orn_init_yaw = np.deg2rad(orn_init_deg)
    orn_yaw_range = np.deg2rad(orn_range_deg)
    rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=env.device)
    orn_euler = orn_init_yaw + orn_yaw_range * rand_sample
    orn_euler[:, 0:2] = 0.0  # Only change yaw
    orn_quat = torch_utils.quat_from_euler_xyz(orn_euler[:, 0], orn_euler[:, 1], orn_euler[:, 2])
    fixed_state[:, 3:7] = orn_quat

    # (1.c) Zero velocity
    fixed_state[:, 7:] = 0.0

    # Write to simulation
    fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
    fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)


def randomize_impedance_gains(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    noise_levels: list[float] = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41],
):
    """Randomize impedance proportional and derivative gains.

    Matches direct: get_random_prop_gains() with multiplicative noise and 50% chance of 1/multiplier.
    Kd = 2 * sqrt(Kp) (critical damping).
    """
    action_term = _get_action_term(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    prop_gains = action_term.default_gains[env_ids].clone()
    prop_gains = get_random_prop_gains(prop_gains, noise_levels, len(env_ids), env.device)
    action_term.task_prop_gains[env_ids] = prop_gains
    action_term.task_deriv_gains[env_ids] = action_term._get_deriv_gains(prop_gains)


def randomize_action_thresholds(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pos_noise_levels: list[float] = [0.25, 0.25, 0.25],
    rot_noise_levels: list[float] = [0.29, 0.29, 0.29],
):
    """Randomize position and rotation action thresholds.

    Matches direct: get_random_prop_gains() on pos_threshold and rot_threshold.
    """
    action_term = _get_action_term(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    pos_threshold = action_term.default_pos_threshold[env_ids].clone()
    pos_threshold = get_random_prop_gains(pos_threshold, pos_noise_levels, len(env_ids), env.device)
    action_term.pos_threshold[env_ids] = pos_threshold

    rot_threshold = action_term.default_rot_threshold[env_ids].clone()
    rot_threshold = get_random_prop_gains(rot_threshold, rot_noise_levels, len(env_ids), env.device)
    action_term.rot_threshold[env_ids] = rot_threshold


def randomize_ema_factor(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ema_factor_range: list[float] = [0.025, 0.1],
):
    """Randomize EMA smoothing factor.

    Matches direct: ema_factor = ema_lower + rand * (ema_upper - ema_lower).
    """
    action_term = _get_action_term(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    ema_rand = torch.rand((len(env_ids), 1), dtype=torch.float32, device=env.device)
    ema_lower, ema_upper = ema_factor_range
    action_term.ema_factor[env_ids] = ema_lower + ema_rand * (ema_upper - ema_lower)


def randomize_dead_zone(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    default_dead_zone: list[float] = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0],
):
    """Randomize dead zone thresholds for the controller.

    Matches direct: dead_zone_thresholds = rand(num_envs, 6) * default_dead_zone.
    Called both at reset and every 2s interval.
    """
    action_term = _get_action_term(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    num_resets = len(env_ids)
    dead_zone = (
        torch.rand((num_resets, 6), dtype=torch.float32, device=env.device)
        * torch.tensor(default_dead_zone, dtype=torch.float32, device=env.device)
    )
    action_term.dead_zone_thresholds[env_ids] = dead_zone


def randomize_contact_threshold(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    lower_bound: float = 5.0,
    upper_bound: float = 10.0,
):
    """Randomize contact penalty thresholds.

    Matches direct: contact_penalty_thresholds = lower + rand * (upper - lower).
    """
    action_term = _get_action_term(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    contact_rand = torch.rand(len(env_ids), dtype=torch.float32, device=env.device)
    action_term.contact_penalty_thresholds[env_ids] = lower_bound + contact_rand * (upper_bound - lower_bound)


def reset_force_sensor(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """Reset force sensor smoothing buffers.

    Matches direct: force_sensor_world_smooth[:, :] = 0.0.
    """
    action_term = _get_action_term(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    action_term.force_sensor_world_smooth[env_ids] = 0.0
    action_term.force_sensor_smooth[env_ids] = 0.0


def randomize_flip_quats(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """Randomize quaternion flip for observation noise.

    Matches direct: flip_quats = ones; rand_flips > 0.5 -> flip_quats = -1.
    """
    action_term = _get_action_term(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    action_term.flip_quats[env_ids] = 1.0
    rand_flips = torch.rand(len(env_ids), device=env.device) > 0.5
    action_term.flip_quats[env_ids[rand_flips]] = -1.0


def reset_action_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """Reset action state and compute initial actions from current EE pose.

    Delegates to ForgeImpedanceAction.reset() which computes initial
    pos/yaw actions from the current fingertip pose relative to the
    fixed asset, matching direct ForgeEnv._reset_idx().
    """
    action_term = _get_action_term(env)
    action_term.reset(env_ids)


def update_fixed_pos_obs_frame(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """Compute fixed_pos_obs_frame and randomize init_fixed_pos_obs_noise.

    Matches direct FactoryEnv._reset_idx():
    1. fixed_pos_obs_frame = fixed_asset_pos + local_offset (height, base_height, offset_x)
    2. init_fixed_pos_obs_noise = randn * noise_scale

    This MUST fire before reset_action_state so that fixed_pos_obs_frame
    is correct when computing initial actions.
    """
    action_term = _get_action_term(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    cfg = action_term.cfg

    # Read fixed asset position from the scene
    fixed_asset: Articulation = env.scene[cfg.fixed_asset_name]
    fixed_pos = fixed_asset.data.root_pos_w - env.scene.env_origins
    fixed_quat = fixed_asset.data.root_quat_w

    # Compute tip position: fixed_pos + R * [offset_x, 0, height + base_height]
    identity_quat = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
        .unsqueeze(0)
        .repeat(env.num_envs, 1)
    )
    tip_offset_local = torch.zeros((env.num_envs, 3), device=env.device)
    tip_offset_local[:, 2] = cfg.fixed_asset_height + cfg.fixed_asset_base_height
    tip_offset_local[:, 0] = cfg.fixed_asset_offset_x

    _, tip_pos = torch_utils.tf_combine(
        fixed_quat, fixed_pos, identity_quat, tip_offset_local
    )
    action_term.fixed_pos_obs_frame[env_ids] = tip_pos[env_ids]

    # Randomize noise on fixed asset position observation
    noise_scale = torch.tensor(
        cfg.fixed_asset_pos_noise, dtype=torch.float32, device=env.device
    )
    noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=env.device) @ torch.diag(noise_scale)
    action_term.init_fixed_pos_obs_noise[env_ids] = noise
