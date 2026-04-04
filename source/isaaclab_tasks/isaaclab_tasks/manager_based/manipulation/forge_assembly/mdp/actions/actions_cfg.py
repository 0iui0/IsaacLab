# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action term configurations for forge assembly tasks."""

from dataclasses import MISSING

from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

from .forge_actions import ForgeAssemblyAction


@configclass
class ForgeAssemblyActionCfg(ActionTermCfg):
    """Configuration for the forge assembly action term.

    This action term replicates the direct/forge control chain:
    - 7D action space: [pos_xyz, rot_xyz, success_pred]
    - Asset-relative action frame (actions relative to hole/bolt position)
    - EMA smoothing on actions
    - Dead zone randomization
    - Jacobian transpose control with nullspace optimization
    - Gain randomization
    - Position/rotation clipping with thresholds
    """

    class_type: type = ForgeAssemblyAction

    joint_names: list[str] = MISSING
    """Joint name patterns for the arm (7 joints)."""

    body_name: str = MISSING
    """End-effector body name."""

    hole_name: str = "hole"
    """Name of the hole (fixed asset) rigid body in the scene."""

    peg_name: str = "peg"
    """Name of the peg (held asset) rigid body in the scene."""

    # Action bounds
    pos_action_bounds: list[float] = [0.02, 0.02, 0.02]
    """Position action scaling bounds."""

    rot_action_bounds: list[float] = [0.097, 0.097, 0.097]
    """Rotation action scaling bounds."""

    # Default controller parameters
    default_task_prop_gains: list[float] = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]
    """Default proportional gains [px, py, pz, rx, ry, rz]."""

    default_dead_zone: list[float] = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]
    """Default dead zone thresholds [fx, fy, fz, tx, ty, tz]."""

    pos_action_threshold: list[float] = [0.02, 0.02, 0.02]
    """Default position clipping threshold."""

    rot_action_threshold: list[float] = [0.097, 0.097, 0.097]
    """Default rotation clipping threshold."""

    # EMA parameters
    ema_factor_range: list[float] = [0.025, 0.1]
    """Range for EMA smoothing factor randomization."""

    # Gain randomization
    task_prop_gains_noise_level: float = 0.41
    """Noise level for proportional gain randomization (direct forge: ±41%)."""

    pos_threshold_noise_level: float = 0.25
    """Noise level for position threshold randomization (direct forge: ±25%)."""

    rot_threshold_noise_level: float = 0.29
    """Noise level for rotation threshold randomization (direct forge: ±29%)."""

    # Nullspace control
    kp_null: float = 10.0
    """Nullspace proportional gain."""

    kd_null: float = 5.0
    """Nullspace derivative gain."""

    default_dof_pos_tensor: list[float] = MISSING
    """Default joint positions for nullspace target (robot-specific)."""

    # Torque clamp
    torque_limit: float = 100.0
    """Maximum joint torque."""

    # Contact penalty
    contact_penalty_threshold_range: list[float] = [5.0, 10.0]
    """Range for contact penalty threshold randomization."""
