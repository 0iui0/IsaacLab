# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass


@configclass
class RobotProfile:
    """Robot-agnostic profile for the forge assembly environment.

    Encapsulates all robot-specific details so the environment code
    never depends on hardcoded joint names, body names, or DOF counts.
    """

    # --- ArticulationCfg (the robot asset itself) ---
    robot: ArticulationCfg = None

    # --- Joint layout ---
    num_arm_joints: int = 7
    """Number of arm joints (7 for Franka, 6 for UR10/CR5)."""

    has_gripper: bool = True
    """Whether the robot has a gripper."""

    arm_joint_ids: slice = slice(0, 7)
    """Slice into joint arrays for arm joints."""

    gripper_joint_ids: slice | None = slice(7, 9)
    """Slice into joint arrays for gripper joints, or None."""

    # --- Body name lookups ---
    ee_body_name: str = ""
    """Name of the end-effector body used for tracking and control."""

    left_finger_body_name: str | None = None
    """Name of the left finger body (gripper robots only)."""

    right_finger_body_name: str | None = None
    """Name of the right finger body (gripper robots only)."""

    force_sensor_body_name: str | None = None
    """Name of the force sensor body."""

    # --- IK / control ---
    default_arm_joint_pos: list = []
    """Default arm joint positions for normal operation."""

    reset_arm_joint_pos: list = []
    """Home-position arm joint positions for IK reset."""

    null_space_default_pos: list = []
    """Default joint positions for null-space controller target."""

    # --- End-effector geometry ---
    fingerpad_length: float = 0.0
    """Length of the fingerpad (0.017608 for Franka, 0.0 for fixed-peg)."""

    # --- Grasp type ---
    grasp_type: str = "gripper"
    """How the peg is held: "gripper" (Franka) or "fixed_peg" (UR10/CR5)."""

    # --- Fixed peg configuration (for UR10/CR5) ---
    peg_offset_from_ee: list = None
    """[x, y, z] offset from EE link to peg tip (meters). Only used for fixed_peg robots."""

    peg_radius: float = 0.0
    """Radius of the peg (meters). Only used for fixed_peg robots."""

    peg_height: float = 0.0
    """Height of the peg (meters). Only used for fixed_peg robots."""

    peg_material: tuple = None
    """(static_friction, dynamic_friction, restitution) for peg. Only used for fixed_peg robots."""

    peg_usd_path: str | None = None
    """Optional USD path for a pre-made peg asset. If None, a cylinder is spawned dynamically."""

    keypoint_axis: int = 2
    """Axis along which keypoint offsets are spread in the held-asset local frame.
    0=X, 1=Y, 2=Z. Must match the direction the peg/asset extends in its local frame.
    Franka peg USD: Z-axis (2). UR10 fixed peg: X-axis (0)."""
