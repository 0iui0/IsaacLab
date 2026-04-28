# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from .base import RobotProfile

# Resolve local asset path relative to this file (avoid import chain issues)
_LOCAL_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "isaaclab_assets", "data"))
_LOCAL_ASSET_DIR = os.path.join(_LOCAL_ROOT, "robots", "marvin")
ASSET_DIR = _LOCAL_ASSET_DIR if os.path.isdir(_LOCAL_ASSET_DIR) else f"{ISAACLAB_NUCLEUS_DIR}/Robots/marvin"

_FORGE_ASSET_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets"))


# ============================================================================
# Marvin M6 + Force Sensor + Franka Panda Gripper Profile
# ============================================================================
# Assembly structure (from GUI assembly or marvin_m6_panda.urdf):
#   Base_R -> Joint1-7_R -> Link7_R -> [fixed] force_sensor -> [fixed] panda_hand
#     -> panda_finger_joint1 -> panda_leftfinger
#     -> panda_finger_joint2 -> panda_rightfinger
#     -> panda_fingertip_centered (virtual link at fingertip center)
# ============================================================================

MARVIN_PANDA_FORGE_PROFILE = RobotProfile(
    # --- Robot ArticulationCfg ---
    robot=ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # NOTE: Using existing marvin_m6_panda.usd (from URDF conversion).
            # After GUI assembly with real force sensor STEP, replace with:
            #   marvin_m6_force_sensor.usd
            usd_path=f"{ASSET_DIR}/marvin_m6_panda.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # Marvin arm joints (7 DOF) — IK-solved for hole-above pose.
                "Joint1_R": -1.1349,
                "Joint2_R": 0.6922,
                "Joint3_R": 0.6928,
                "Joint4_R": -1.9639,
                "Joint5_R": -2.3329,
                "Joint6_R": -0.5054,
                "Joint7_R": -1.0058,
                # Franka Panda gripper joints (2 prismatic)
                "panda_finger_joint1": 0.04,  # open
                "panda_finger_joint2": 0.04,  # open (mimic joint)
            },
            # Horizontal mount: base Z → world -Y (right arm extends to the right).
            # 90° rotation around X: quat=[cos(π/4), sin(π/4), 0, 0].
            # Mounted at 40.5cm above the table surface.
            pos=(0.0, -0.234, 0.405),
            rot=(0.7071, 0.7071, 0.0, 0.0),
        ),
        actuators={
            # Marvin arm actuators (impedance control mode)
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["Joint[1-7]_R"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=108.0,  # Joint1-2 max effort
                velocity_limit_sim=3.1416,
            ),
            # Franka Panda gripper actuators (matching Franka profile)
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    ),
    # --- Joint layout ---
    num_arm_joints=7,
    has_gripper=True,
    arm_joint_ids=slice(0, 7),
    gripper_joint_ids=slice(7, 9),
    # --- Body name lookups ---
    # NOTE: After URDF→USD conversion, fixed joints may be merged.
    # The force_sensor link might be merged into Link7_R, and panda_hand
    # might be merged into force_sensor or Link7_R.
    # Use Link7_R as EE body since it's the last arm link before gripper.
    # For gripper, use panda_leftfinger/panda_rightfinger directly.
    ee_body_name="Link7_R",  # Changed from panda_fingertip_centered (merged)
    left_finger_body_name="panda_leftfinger",
    right_finger_body_name="panda_rightfinger",
    force_sensor_body_name="Link7_R",  # Force sensor merged into Link7_R
    # --- IK / control ---
    # --- IK / control ---
    # IK-solved pose above hole at (0.20, -0.45, 0.005), within joint limits.
    default_arm_joint_pos=[-1.1349, 0.6922, 0.6928, -1.9639, -2.3329, -0.5054, -1.0058],
    reset_arm_joint_pos=[-1.1349, 0.6922, 0.6928, -1.9639, -2.3329, -0.5054, -1.0058],
    null_space_default_pos=[-1.1349, 0.6922, 0.6928, -1.9639, -2.3329, -0.5054, -1.0058],
    # IK joint limits: URDF limits from marvin_m6_panda.urdf.
    ik_joint_limits=[
        (-3.1067, 3.1067),  # Joint1_R
        (-2.0944, 2.0944),  # Joint2_R
        (0.0, 3.1067),      # Joint3_R (elbow - prevent reversal + URDF)
        (-2.5307, 1.0472),  # Joint4_R
        (-3.1067, 3.1067),  # Joint5_R
        (-1.0472, 1.0472),  # Joint6_R
        (-1.5708, 1.5708),  # Joint7_R
    ],
    # --- End-effector geometry ---
    fingerpad_length=0.017608,  # Same as Franka
    # Offset from Link7_R to fingertip center in Link7_R frame.
    # Computed from URDF chain:
    #   Link7_R → force_sensor: xyz=[0, -0.1, 0], rpy=[π/2, 0, 0] (90° X rotation)
    #     After rotation: force_sensor Z-axis = Link7_R -Y-axis
    #   force_sensor → panda_hand: xyz=[0, 0, 0.0165] (along Z in force_sensor frame)
    #     In Link7_R frame: Y offset = -0.0165 (Z_fs → -Y_L7)
    #   panda_hand → fingertip_centered: xyz=[0, 0, 0.112071] (along Z in panda_hand frame)
    #     In Link7_R frame: Y offset = -0.112071 (Z_ph → -Y_L7)
    # Total Y offset = -0.1 - 0.0165 - 0.112071 = -0.228571 m
    ee_to_fingertip_offset=[0.0, -0.228571, 0.0],
    # --- Grasp type ---
    grasp_type="gripper",
    # --- EE frame correction ---
    # Link7_R -Y → force_sensor Z → gripper Z-down, so apply 90° X rotation
    ee_frame_correction=[0.7071, 0.7071, 0.0, 0.0],
    # --- Keypoint axis ---
    # Panda gripper fingertip extends along Z in its local frame
    keypoint_axis=2,
    # --- Fixed peg configuration (not used for gripper robots) ---
    peg_offset_from_ee=None,
    peg_radius=0.0,
    peg_height=0.0,
    peg_material=None,
    # --- Visual-only assets ---
    visual_assets={
        "upper_body": (
            f"{_FORGE_ASSET_DIR}/stl/000-upper-body.STL",
            (0.0, 0.0, 0.0),
        ),
    },
    # --- Left arm (from left-arm URDF, kinematic visual-only) ---
    left_arm=ArticulationCfg(
        prim_path="/World/envs/env_.*/LeftArm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_FORGE_ASSET_DIR}/urdf/marvin_m6_left/marvin_m6_left.usd",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.95, 0.95),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "Joint1_L": 1.1349,   # negated (mirror)
                "Joint2_L": -0.8667,  # negated + 10° down
                "Joint3_L": -0.1928,  # negated (mirror)
                "Joint4_L": -1.9639,
                "Joint5_L": 2.3329,   # negated (mirror)
                "Joint6_L": -0.5054,
                "Joint7_L": 1.0058,   # negated (mirror)
            },
            pos=(0.0, 0.234, 0.405),
            rot=(0.7071, -0.7071, 0.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["Joint[1-7]_L"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    ),
)


# ============================================================================
# Marvin M6 + Robotiq 2F-85 Gripper Profile (alternative)
# ============================================================================
# For future use with Robotiq gripper assembly.
# ============================================================================

MARVIN_ROBOTIQ_2F85_FORGE_PROFILE = RobotProfile(
    # --- Robot ArticulationCfg ---
    robot=ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/marvin_m6_robotiq_2f85.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "Joint1_R": 0.0,
                "Joint2_R": -1.571,
                "Joint3_R": 1.571,
                "Joint4_R": 0.0,
                "Joint5_R": -1.571,
                "Joint6_R": 0.0,
                "Joint7_R": 0.0,
                "finger_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.3),
            rot=(0.7071, 0.7071, 0.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["Joint[1-7]_R"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=108.0,
                velocity_limit_sim=3.1416,
            ),
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                stiffness=17.0,
                damping=1.5,
                effort_limit_sim=1650.0,
                velocity_limit_sim=10.0,
            ),
            "gripper_finger": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_joint"],
                stiffness=0.2,
                damping=0.001,
                effort_limit_sim=50.0,
                velocity_limit_sim=10.0,
            ),
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_knuckle_joint", "right_outer_knuckle_joint"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    ),
    # --- Joint layout ---
    num_arm_joints=7,
    has_gripper=True,
    arm_joint_ids=slice(0, 7),
    gripper_joint_ids=slice(7, 8),
    # --- Body name lookups ---
    ee_body_name="force_sensor",
    left_finger_body_name="left_outer_finger",
    right_finger_body_name="right_outer_finger",
    force_sensor_body_name="force_sensor",
    # --- IK / control ---
    default_arm_joint_pos=[0.0, -0.5, 1.0, -1.0, -0.5, 1.0, 0.0],
    reset_arm_joint_pos=[0.0, -1.571, 1.571, 0.0, -1.571, 0.0, 0.0],
    null_space_default_pos=[0.0, -0.5, 1.0, -1.0, -0.5, 1.0, 0.0],
    ik_joint_limits=[
        (-6.28, 6.28),  # Joint1_R
        (-6.28, 6.28),  # Joint2_R
        (0.0, 6.28),    # Joint3_R (elbow - prevent reversal)
        (-6.28, 6.28),  # Joint4_R
        (-6.28, 6.28),  # Joint5_R
        (-6.28, 6.28),  # Joint6_R
        (-6.28, 6.28),  # Joint7_R
    ],
    # --- End-effector geometry ---
    fingerpad_length=0.025,
    # --- Grasp type ---
    grasp_type="gripper",
    # --- Keypoint axis ---
    keypoint_axis=2,
    # --- Fixed peg configuration (not used for Marvin) ---
    peg_offset_from_ee=None,
    peg_radius=0.0,
    peg_height=0.0,
    peg_material=None,
)
