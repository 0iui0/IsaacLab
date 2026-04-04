# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Marvin M6-S-R-CCS-696 specific environment configuration for peg-in-hole assembly.

Uses the Marvin M6-S-R-CCS-696 collaborative robot arm with
 ForgeAssemblyAction (Jacobian transpose
control with EMA smoothing, dead zone, and nullspace optimization) to replicate
the direct forge control chain exactly.
"""

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab_assets.robots.marvin import MARVIN_M6_CFG

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
class MarvinM6ForgeAssemblyEnvCfg(ForgeAssemblyEnvCfg):
    """Configuration for Marvin M6-S-R-CCS-696 peg-in-hole assembly.

    Inherits the base forge assembly config and fills in Marvin M6-specific:
    - Robot articulation (Marvin M6 with 7 revolute joints)
    - ForgeAssemblyAction term (7D asset-relative Jacobian transpose control)
    - Marvin M6 default joint positions for nullspace target
    - Scene placement matching Marvin M6's kinematic reach
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # --- Robot ---
        self.scene.robot = MARVIN_M6_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # --- Action: Forge Assembly (7D asset-relative Jacobian transpose) ---
        self.actions.arm_action = ForgeAssemblyActionCfg(
            asset_name="robot",
            joint_names=["Joint[1-7]_R"],
            body_name="Link7_R",
            hole_name="hole",
            peg_name="peg",
            pos_action_bounds=[0.02, 0.02, 0.02],
            rot_action_bounds=[0.097, 0.097, 0.097],
            default_task_prop_gains=[565.0, 565.0, 565.0, 28.0, 28.0, 28.0],
            default_dead_zone=[5.0, 5.0, 5.0, 1.0, 1.0, 1.0],
            pos_action_threshold=[0.02, 0.02, 0.02],
            rot_action_threshold=[0.097, 0.097, 0.097],
            ema_factor_range=[0.025, 0.1],
            task_prop_gains_noise_level=0.41,
            pos_threshold_noise_level=0.25,
            rot_threshold_noise_level=0.29,
            kp_null=10.0,
            kd_null=5.0,
            default_dof_pos_tensor=[0.0, -0.4, 0.8, -1.5, 0.0, 0.8, 0.0],
            torque_limit=100.0,
            contact_penalty_threshold_range=[5.0, 10.0],
        )

        # --- Override body_name for MDP event terms ---
        self.events.reset_peg.params["body_name"] = "Link7_R"

        # --- Override joint name patterns for randomization ---
        self.events.randomize_gains.params["asset_cfg"].joint_names = ["Joint[1-7]_R"]
        self.events.randomize_friction.params["asset_cfg"].joint_names = ["Joint[1-7]_R"]

        # --- Scene adjustments for Marvin M6 workspace ---
        # Marvin M6 reach ~0.78m. Hole at (0.45, 0.0, table_height)
        # Ground at z=-1.05 -> table surface at z=1.05
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.lookat = (0.45, 0.0, 1.0)

        # --- Episode length ---
        self.episode_length_s = 10.0


@configclass
class MarvinM6ForgeAssemblyEnvCfg_PLAY(MarvinM6ForgeAssemblyEnvCfg):
    """Play/inference configuration for Marvin M6 peg-in-hole assembly."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
