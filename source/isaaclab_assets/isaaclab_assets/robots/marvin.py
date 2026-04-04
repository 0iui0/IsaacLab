# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Marvin M6-S-R-CCS-696 collaborative robot arm.

Reference: https://github.com/isaac-sim/IsaacLab
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

MARVIN_M6_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            os.path.dirname(__file__), "..", "data", "Robots", "Marvin", "M6-CCS-696", "marvin_m6.usd"
        ),
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
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Joint1_R": 0.0,
            "Joint2_R": -0.4,
            "Joint3_R": 0.8,
            "Joint4_R": -1.5,
            "Joint5_R": 0.0,
            "Joint6_R": 0.8,
            "Joint7_R": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        # Shoulder joints (J1-2): 108 Nm torque rating
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["Joint[1-2]_R"],
            effort_limit_sim=108.0,
            velocity_limit_sim=3.1416,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
        # Elbow joints (J3-4): 66 Nm torque rating
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Joint[3-4]_R"],
            effort_limit_sim=66.0,
            velocity_limit_sim=3.1416,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
        # Wrist joints (J5-7): 18 Nm torque rating
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["Joint[5-7]_R"],
            effort_limit_sim=18.0,
            velocity_limit_sim=3.1416,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""Configuration of Marvin M6-S-R-CCS-696 robot arm for operational space control."""
