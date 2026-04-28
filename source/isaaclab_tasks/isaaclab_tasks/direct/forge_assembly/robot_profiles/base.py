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

    def __post_init__(self):
        # Validate joint config consistency
        if len(self.default_arm_joint_pos) != self.num_arm_joints:
            raise ValueError(
                f"default_arm_joint_pos length ({len(self.default_arm_joint_pos)}) "
                f"!= num_arm_joints ({self.num_arm_joints})"
            )
        if len(self.reset_arm_joint_pos) != self.num_arm_joints:
            raise ValueError(
                f"reset_arm_joint_pos length ({len(self.reset_arm_joint_pos)}) "
                f"!= num_arm_joints ({self.num_arm_joints})"
            )
        # Validate fixed_peg requires peg config
        if self.grasp_type == "fixed_peg":
            if self.peg_offset_from_ee is None:
                raise ValueError("fixed_peg requires peg_offset_from_ee")
            if self.peg_radius <= 0:
                raise ValueError("fixed_peg requires peg_radius > 0")
            if self.peg_height <= 0:
                raise ValueError("fixed_peg requires peg_height > 0")
        # Validate gripper consistency
        if self.has_gripper and self.gripper_joint_ids is None:
            raise ValueError("has_gripper=True requires gripper_joint_ids")
        if not self.has_gripper and self.gripper_joint_ids is not None:
            raise ValueError("has_gripper=False but gripper_joint_ids is set")

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

    ik_joint_limits: list | None = None
    """Per-joint (min, max) limits clamped during IK iteration to prevent joint reversal.

    Example for 7-DOF arm: [(−6.28, 6.28), (−6.28, 6.28), (0.0, 6.28), ...]
    Set to None to skip clamping (uses URDF limits). Only needs to constrain
    joints prone to reversal (e.g. elbow joint).
    """

    # --- End-effector geometry ---
    fingerpad_length: float = 0.0
    """Length of the fingerpad (0.017608 for Franka, 0.0 for fixed-peg)."""

    ee_to_fingertip_offset: list = [0.0, 0.0, 0.0]
    """[x, y, z] offset from ee_body to fingertip center in ee_body frame.

    For Franka: ee_body = panda_fingertip_centered, so offset = [0, 0, 0].
    For Marvin Panda with ee_body = Link7_R: fingertip is ~0.129m along Link7_R -Y,
    so offset should be [0, -0.129, 0] (approximate).

    This is used to correctly position held assets relative to the actual gripper center.
    """

    # --- Grasp type ---
    grasp_type: str = "gripper"
    """How the peg is held: "gripper" (Franka) or "fixed_peg" (UR10/CR5)."""

    # --- Fixed peg configuration (for UR10/CR5) ---
    peg_offset_from_ee: list = None
    """Direction vector from EE link to peg tip, scaled by peg_height/2 at spawn time.

    Must be a unit direction vector (e.g., [1,0,0] for X, [0,0,1] for Z).
    The actual offset is computed as: direction * peg_height / 2.
    Only used for fixed_peg robots.
    """

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

    # --- Visual-only props (no physics, just rendering) ---
    visual_assets: dict[str, tuple] | None = None
    """Visual-only USD assets to spawn in the scene. Dict of {prim_name: (usd_path, (x,y,z)[, (w,x,y,z)])}.
    Optional third element is orientation quaternion. Spawned as static Xform, no collision."""

    # --- Left arm (visual-only articulation) ---
    left_arm: ArticulationCfg | None = None
    """Optional left arm Articulation. Kinematic-only, does not participate in training."""

    # --- EE frame correction ---
    ee_frame_correction: list | None = None
    """Quaternion [w,x,y,z] to correct EE body frame to a standard control frame.

    Applied when ee_to_fingertip_offset is non-zero, transforming the EE orientation
    so that Z points down (matching the control convention).
    For Marvin: [0.7071, 0.7071, 0, 0] (90° around X to align Link7_R -Y with gripper Z-down).
    For Franka: None (ee_body IS the fingertip, already correct).
    """
