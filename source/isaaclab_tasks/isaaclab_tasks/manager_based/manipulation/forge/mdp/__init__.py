# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP functions for FORGE manipulation tasks."""

from .actions import ForgeImpedanceAction, ForgeImpedanceActionCfg, get_random_prop_gains

from .observations import (
    ee_angvel,
    ee_angvel_raw,
    ee_linvel,
    ee_linvel_raw,
    ema_factor,
    fingertip_pos_raw,
    fingertip_pos_rel_fixed,
    fingertip_quat,
    fingertip_quat_raw,
    fixed_pos,
    fixed_quat,
    force_threshold,
    force_threshold_critic,
    ft_force,
    ft_force_raw,
    held_pos,
    held_pos_rel_fixed,
    held_quat,
    joint_pos_arm,
    pos_threshold,
    prev_actions,
    rot_threshold,
    task_prop_gains,
    compute_keypoint_distance,
    get_keypoint_offsets,
    squashing_fn,
)

from .rewards import (
    action_grad_penalty,
    action_penalty_asset,
    action_penalty_ee,
    contact_penalty,
    curr_engaged,
    curr_success,
    kp_baseline,
    kp_coarse,
    kp_fine,
    success_pred_error,
)

from .events import (
    randomize_action_thresholds,
    randomize_contact_threshold,
    randomize_dead_zone,
    randomize_ema_factor,
    randomize_fixed_asset_pose,
    randomize_flip_quats,
    randomize_impedance_gains,
    reset_action_state,
    reset_force_sensor,
    update_fixed_pos_obs_frame,
)

from .terminations import task_success, time_out
