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

# CR5 USD path — place converted asset under Isaac Lab assets directory.
# Convert from URDF first using:
#   python -m isaaclab.sim.converters.urdf_converter --input <urdf_path> --output <usd_path>
_CR5_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/DOBOT/CR5/cr5.usd"

CR5_FORGE_PROFILE = RobotProfile(
    # --- Robot ArticulationCfg ---
    robot=ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_CR5_USD_PATH,
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
                "joint1": 0.0,
                "joint2": -1.571,
                "joint3": 1.571,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": 0.0,
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
                effort_limit_sim=150.0,
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
    ee_body_name="Link6",
    left_finger_body_name=None,
    right_finger_body_name=None,
    force_sensor_body_name="force_sensor",  # Dynamically created
    # --- IK / control ---
    default_arm_joint_pos=[0.0, -1.571, 1.571, 0.0, 0.0, 0.0],
    reset_arm_joint_pos=[0.0, -1.571, 1.571, 0.0, 0.0, 0.0],
    null_space_default_pos=[0.0, -1.571, 1.571, 0.0, 0.0, 0.0],
    # --- End-effector geometry ---
    fingerpad_length=0.0,
    # --- Grasp type ---
    grasp_type="fixed_peg",
    # --- Fixed peg configuration ---
    peg_offset_from_ee=[0.0, 0.0, 0.15],  # Peg extends 15cm from EE link
    peg_radius=0.0125,  # 25mm diameter peg
    peg_height=0.0625,  # 62.5mm peg height
    peg_material=(1.0, 1.0, 0.0),  # (static_friction, dynamic_friction, restitution)
)
