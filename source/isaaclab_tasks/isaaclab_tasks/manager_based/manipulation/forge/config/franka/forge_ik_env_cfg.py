# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base Franka configuration for FORGE tasks with differential IK control."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.forge.forge_env_cfg import (
    ForgeEnvCfg,
    ForgeSceneCfg,
    FORGE_ASSET_DIR,
)

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

# Import Factory asset configurations
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


@configclass
class FrankaForgeEnvCfg(ForgeEnvCfg):
    """Base Franka configuration for FORGE tasks."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Set Franka as robot with high PD gains for IK tracking
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        # Configure IK action with relative mode (matches FORGE direct behavior)
        # The action is relative to the fixed asset position
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,  # Match direct FORGE behavior
                ik_method="dls",
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.107]  # Fingertip offset (finger pad length)
            ),
        )

        # Set body name for observations
        self.observations.policy.fingertip_pose_rel.params["asset_cfg"] = (
            self.observations.policy.fingertip_pose_rel.params["asset_cfg"].replace(body_names="panda_hand")
        )
        self.observations.policy.fingertip_vel.params["asset_cfg"] = (
            self.observations.policy.fingertip_vel.params["asset_cfg"].replace(body_names="panda_hand")
        )


@configclass
class FrankaPegInsertSceneCfg(ForgeSceneCfg):
    """Scene configuration for Peg Insert task."""

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Fixed asset: Hole
    fixed_asset = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{FORGE_ASSET_DIR}/factory_hole_8mm.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False  # Disable articulation root for rigid body
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
                articulation_enabled=False  # Disable articulation root for rigid body
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
    """Scene configuration for Gear Mesh task with optional flanking gears."""

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Fixed asset: Gear base
    fixed_asset = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{FORGE_ASSET_DIR}/factory_gear_base.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False  # Disable articulation root for rigid body
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
                articulation_enabled=False  # Disable articulation root for rigid body
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

    # Optional flanking gears for gear_mesh task
    flanking_gear_small = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SmallGearAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{FORGE_ASSET_DIR}/factory_gear_small.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False  # Disable articulation root for rigid body
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    flanking_gear_large = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LargeGearAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{FORGE_ASSET_DIR}/factory_gear_large.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False  # Disable articulation root for rigid body
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
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
class FrankaNutThreadSceneCfg(ForgeSceneCfg):
    """Scene configuration for Nut Thread task."""

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Fixed asset: Bolt
    fixed_asset = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{FORGE_ASSET_DIR}/factory_bolt_m16.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False  # Disable articulation root for rigid body
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
                articulation_enabled=False  # Disable articulation root for rigid body
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
