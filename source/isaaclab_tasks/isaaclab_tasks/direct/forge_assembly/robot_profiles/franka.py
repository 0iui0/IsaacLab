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

FRANKA_FORGE_PROFILE = RobotProfile(
    # --- Robot ArticulationCfg ---
    robot=ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",
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
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
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
    ee_body_name="panda_fingertip_centered",
    left_finger_body_name="panda_leftfinger",
    right_finger_body_name="panda_rightfinger",
    force_sensor_body_name="force_sensor",
    # --- IK / control ---
    default_arm_joint_pos=[-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754],
    reset_arm_joint_pos=[1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01],
    null_space_default_pos=[-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754],
    # --- End-effector geometry ---
    fingerpad_length=0.017608,
    # --- Grasp type ---
    grasp_type="gripper",
    # --- Fixed peg configuration (not used for Franka) ---
    peg_offset_from_ee=None,
    peg_radius=0.0,
    peg_height=0.0,
    peg_material=None,
)
