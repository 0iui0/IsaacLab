# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka-specific environment configuration for peg-in-hole assembly.

Uses the Franka Emika Panda robot with ForgeAssemblyAction (Jacobian transpose
control with EMA smoothing, dead zone, and nullspace optimization) to replicate
the direct forge control chain exactly.

Robot config matches direct forge's factory_env_cfg.py exactly:
- Pure torque control (stiffness=0, damping=0 for arm joints)
- Stiff finger joints (stiffness=7500, damping=173)
- disable_gravity=True, solver_position_iteration_count=192
- Contact sensors enabled
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.forge_assembly.forge_assembly_env_cfg import (
    ForgeAssemblyEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.forge_assembly.mdp.actions.actions_cfg import (
    ForgeAssemblyActionCfg,
)


##
# Environment configuration
##


@configclass
class FrankaForgeAssemblyEnvCfg(ForgeAssemblyEnvCfg):
    """Configuration for Franka Panda peg-in-hole assembly.

    Inherits the base forge assembly config and fills in Franka-specific:
    - Robot articulation matching direct forge (pure torque control)
    - ForgeAssemblyAction term (7D asset-relative Jacobian transpose control)
    - Franka default joint positions for nullspace target
    - Scene placement matching Franka's kinematic reach
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # --- Robot: Match direct forge factory_env_cfg.py exactly ---
        # Direct forge uses franka_mimic.usd with pure torque control (stiffness=0)
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
                activate_contact_sensors=True,  # Match direct forge
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,  # Match direct forge
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,  # Match direct forge
                    solver_velocity_iteration_count=1,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,  # Match direct forge
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "panda_joint1": 0.00871,
                    "panda_joint2": -0.10368,
                    "panda_joint3": -0.00794,
                    "panda_joint4": -1.49139,
                    "panda_joint5": -0.00083,
                    "panda_joint6": 1.38774,
                    "panda_joint7": 0.0,
                    "panda_finger_joint.*": 0.02,  # Slightly open for reset
                },
            ),
            actuators={
                # Pure torque control for arm joints (stiffness=0, damping=0)
                "panda_shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[1-4]"],
                    effort_limit_sim=87.0,
                    stiffness=0.0,
                    damping=0.0,
                ),
                "panda_forearm": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[5-7]"],
                    effort_limit_sim=12.0,
                    stiffness=0.0,
                    damping=0.0,
                ),
                # Stiff finger joints for gripping (matches direct forge)
                "panda_hand": ImplicitActuatorCfg(
                    joint_names_expr=["panda_finger_joint.*"],
                    effort_limit_sim=200.0,
                    stiffness=7500.0,
                    damping=173.0,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )

        # --- Action: Forge Assembly (7D asset-relative Jacobian transpose) ---
        self.actions.arm_action = ForgeAssemblyActionCfg(
            asset_name="robot",
            joint_names=["panda_joint[1-7]"],
            body_name="panda_hand",
            hole_name="hole",
            peg_name="peg",
            pos_action_bounds=[0.05, 0.05, 0.05],  # Match direct forge exactly
            rot_action_bounds=[1.0, 1.0, 1.0],  # Match direct forge exactly
            default_task_prop_gains=[565.0, 565.0, 565.0, 28.0, 28.0, 28.0],
            default_dead_zone=[5.0, 5.0, 5.0, 1.0, 1.0, 1.0],
            pos_action_threshold=[0.02, 0.02, 0.02],  # Match direct forge exactly
            rot_action_threshold=[0.097, 0.097, 0.097],
            ema_factor_range=[0.025, 0.1],
            task_prop_gains_noise_level=0.41,
            pos_threshold_noise_level=0.25,
            rot_threshold_noise_level=0.29,
            kp_null=10.0,
            kd_null=6.3246,  # 2*sqrt(10) matching direct forge
            default_dof_pos_tensor=[-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754],
            torque_limit=100.0,
            contact_penalty_threshold_range=[5.0, 10.0],
        )

        # --- Override body_name for MDP event terms ---
        self.events.reset_peg.params["body_name"] = "panda_hand"

        # --- Override joint name patterns for randomization ---
        self.events.randomize_gains.params["asset_cfg"].joint_names = ["panda_joint.*"]
        self.events.randomize_friction.params["asset_cfg"].joint_names = ["panda_joint.*"]

        # --- Scene adjustments for Franka workspace ---
        # Franka reach ~0.855m. Hole at (0.5, 0.0, table_height)
        # Ground at z=-1.05 -> table surface at z=1.05
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.5, 0.0, 1.0)

        # --- Episode length ---
        self.episode_length_s = 10.0

        # --- num_envs matching rl_games num_actors: 128 ---
        self.scene.num_envs = 128


@configclass
class FrankaForgeAssemblyEnvCfg_PLAY(FrankaForgeAssemblyEnvCfg):
    """Play/inference configuration for Franka peg-in-hole assembly."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
