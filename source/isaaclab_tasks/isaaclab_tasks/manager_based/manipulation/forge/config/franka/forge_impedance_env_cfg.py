# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base Franka configuration for FORGE tasks with impedance control.

This replaces the old IK-based config (forge_ik_env_cfg.py) with impedance
control that matches the direct version exactly.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.forge.forge_env_cfg import (
    ForgeEnvCfg,
    ForgeSceneCfg,
)
from isaaclab_tasks.manager_based.manipulation.forge.mdp.actions import ForgeImpedanceActionCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

FORGE_ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"

##
# Franka-specific actuator config for impedance control
##

# For impedance control, arm joints need zero stiffness/damping since we apply torques directly.
FORGE_FRANKA_CFG = FRANKA_PANDA_HIGH_PD_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    actuators={
        "arm": FRANKA_PANDA_HIGH_PD_CFG.actuators["arm"].replace(
            stiffness={k: 0.0 for k in FRANKA_PANDA_HIGH_PD_CFG.actuators["arm"].stiffness},
            damping={k: 0.0 for k in FRANKA_PANDA_HIGH_PD_CFG.actuators["arm"].damping},
        ),
        "gripper": FRANKA_PANDA_HIGH_PD_CFG.actuators["gripper"],
    },
)


##
# Scene configurations for each task
##


@configclass
class FrankaPegInsertSceneCfg(ForgeSceneCfg):
    """Scene configuration for Peg Insert task."""

    robot = FORGE_FRANKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Fixed asset: Hole
    fixed_asset = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{FORGE_ASSET_DIR}/factory_hole_8mm.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Held asset: Peg
    held_asset = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{FORGE_ASSET_DIR}/factory_peg_8mm.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class FrankaGearMeshSceneCfg(ForgeSceneCfg):
    """Scene configuration for Gear Mesh task."""

    robot = FORGE_FRANKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Fixed asset: Gear base
    fixed_asset = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{FORGE_ASSET_DIR}/factory_gear_base.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Held asset: Medium gear
    held_asset = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{FORGE_ASSET_DIR}/factory_gear_medium.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class FrankaNutThreadSceneCfg(ForgeSceneCfg):
    """Scene configuration for Nut Thread task."""

    robot = FORGE_FRANKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Fixed asset: Bolt
    fixed_asset = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{FORGE_ASSET_DIR}/factory_bolt_m16.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Held asset: Nut
    held_asset = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{FORGE_ASSET_DIR}/factory_nut_m16.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )


##
# Fixed asset config classes for keypoint/success computation
##


@configclass
class PegInsertFixedAssetCfg:
    """Fixed asset config for peg_insert task."""
    height: float = 0.025
    base_height: float = 0.0
    medium_gear_base_offset: list = [0.0, 0.0, 0.0]
    thread_pitch: float = 0.0


@configclass
class GearMeshFixedAssetCfg:
    """Fixed asset config for gear_mesh task."""
    height: float = 0.02
    base_height: float = 0.005
    medium_gear_base_offset: list = [2.025e-2, 0.0, 0.0]
    thread_pitch: float = 0.0


@configclass
class NutThreadFixedAssetCfg:
    """Fixed asset config for nut_thread task."""
    height: float = 0.025
    base_height: float = 0.01
    medium_gear_base_offset: list = [0.0, 0.0, 0.0]
    thread_pitch: float = 0.002


##
# Base Franka FORGE environment configuration
##


@configclass
class FrankaForgeEnvCfg(ForgeEnvCfg):
    """Base Franka configuration for FORGE tasks with impedance control."""

    def __post_init__(self):
        # Post init of parent (sets sim params, decimation, etc.)
        super().__post_init__()

        # Set Franka as robot with zero-stiffness arm actuators (impedance control applies torques directly)
        self.scene.robot = FORGE_FRANKA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        # Configure impedance action term
        self.actions.arm_action = ForgeImpedanceActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_fingertip_centered",
            finger_body_names=["panda_leftfinger", "panda_rightfinger"],
            fixed_asset_name="fixed_asset",
        )

        # Observation body names are already set in ForgeEnvCfg as "panda_fingertip_centered"
