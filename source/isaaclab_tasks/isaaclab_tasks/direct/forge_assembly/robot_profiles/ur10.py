# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from .base import RobotProfile

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"

UR10_FORGE_PROFILE = RobotProfile(
    # --- Robot ArticulationCfg ---
    robot=ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
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
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -0.5,
                "elbow_joint": -0.5,
                "wrist_1_joint": -0.5,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=330.0,
                velocity_limit_sim=180.0,
            ),
        },
    ),
    # --- Joint layout ---
    num_arm_joints=6,
    has_gripper=False,
    arm_joint_ids=slice(0, 6),
    gripper_joint_ids=None,
    # --- Body name lookups ---
    ee_body_name="ee_link",
    left_finger_body_name=None,
    right_finger_body_name=None,
    force_sensor_body_name="force_sensor",  # Dynamically created
    # --- IK / control ---
    default_arm_joint_pos=[0.0, -1.712, 1.712, 0.0, -1.571, 0.0],
    reset_arm_joint_pos=[0.0, -0.5, -0.5, -0.5, 0.0, 0.0],
    null_space_default_pos=[0.0, -1.712, 1.712, 0.0, -1.571, 0.0],
    # --- End-effector geometry ---
    fingerpad_length=0.0,
    # --- Grasp type ---
    grasp_type="fixed_peg",
    # --- Fixed peg configuration ---
    # Approach axis direction: [0,1,0] = ee_link Y-axis (flange outward).
    # Actual peg offset is computed dynamically as: direction * peg_height/2
    # This ensures peg base always sits on flange surface regardless of peg_height.
    peg_offset_from_ee=[0.0, 1.0, 0.0],
    peg_radius=0.004,  # 8mm diameter peg (fits 8.1mm hole with clearance)
    peg_height=0.050,  # 50mm peg height (matches Franka peg)
    peg_material=(1.0, 1.0, 0.0),  # (static_friction, dynamic_friction, restitution)
    # USD with ArticulationRootAPI — required for contact force reporting.
    # Uses the same factory peg as Franka's held asset.
    peg_usd_path=f"{ASSET_DIR}/factory_peg_8mm.usd",
)
