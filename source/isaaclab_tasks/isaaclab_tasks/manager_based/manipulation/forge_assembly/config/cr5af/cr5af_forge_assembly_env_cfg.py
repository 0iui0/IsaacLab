# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CR5AF-specific environment configuration for peg-in-hole assembly.

Uses the Dobot CR5 6-axis robot arm with Operational Space Control (OSC)
for Cartesian impedance control during peg insertion.

The action space is 7D: target pose (position xyz + quaternion wxyz) for
the operational space controller, which maps to joint torques via Jacobian
transpose.
"""

import isaaclab.sim as sim_utils
from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.dobot import CR5_CFG

from isaaclab_tasks.manager_based.manipulation.forge_assembly.forge_assembly_env_cfg import (
    ForgeAssemblyEnvCfg,
)


##
# Environment configuration
##


@configclass
class CR5AFForgeAssemblyEnvCfg(ForgeAssemblyEnvCfg):
    """Configuration for Dobot CR5AF peg-in-hole assembly.

    Inherits the base forge assembly config and fills in CR5-specific:
    - Robot articulation (CR5 with 6 revolute joints)
    - OSC action term (Cartesian pose → joint torques)
    - Adjusted initial joint positions for reach workspace
    - Scene placement matching CR5's kinematic reach
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # --- Robot ---
        self.scene.robot = CR5_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        # --- Action: Operational Space Controller ---
        # target_types=["pose_abs"]: 7D action (pos xyz + quat wxyz)
        # impedance_mode="variable_kp": additional 6D stiffness per-axis
        # Total action dim = 7 (pose) + 6 (stiffness) = 13
        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            body_name="Link6",
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_abs"],
                impedance_mode="variable_kp",
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                motion_stiffness_task=100.0,
                motion_damping_ratio_task=1.0,
                motion_stiffness_limits_task=(50.0, 200.0),
                nullspace_control="position",
            ),
            nullspace_joint_pos_target="center",
            position_scale=1.0,
            orientation_scale=1.0,
            stiffness_scale=100.0,
        )

        # --- Scene adjustments for CR5 workspace ---
        # CR5 reach ~0.5m. Place hole/peg within workspace.
        # Ground at z=-1.05 so table surface is at z=1.05 (matches base height)
        # Hole at (0.5, 0.0, 1.05) is within CR5 reach
        # Peg starts 0.15m above hole

        # --- Viewer ---
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.5, 0.0, 1.0)

        # --- Episode length ---
        # 10s at 120Hz with decimation 8 = 150 steps
        # Sufficient for approach + insertion
        self.episode_length_s = 10.0


@configclass
class CR5AFForgeAssemblyEnvCfg_PLAY(CR5AFForgeAssemblyEnvCfg):
    """Play/inference configuration for CR5AF peg-in-hole assembly.

    Smaller environment count and disabled observation noise for
    visualization and evaluation.
    """

    def __post_init__(self):
        super().__post_init__()
        # Smaller scene for visualization
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable observation noise for play
        self.observations.policy.enable_corruption = False
