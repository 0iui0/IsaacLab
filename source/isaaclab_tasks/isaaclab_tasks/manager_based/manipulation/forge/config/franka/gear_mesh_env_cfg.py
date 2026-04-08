# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Gear Mesh task configuration for FORGE with impedance control."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.forge.mdp as forge_mdp

from isaaclab_tasks.manager_based.manipulation.forge.config.franka.forge_impedance_env_cfg import (
    FrankaForgeEnvCfg,
    FrankaGearMeshSceneCfg,
    GearMeshFixedAssetCfg,
)


@configclass
class FrankaGearMeshEnvCfg(FrankaForgeEnvCfg):
    """Configuration for the Gear Mesh task.

    The robot must mesh a gear with a fixed gear base.
    Episode length: 20s, success_threshold: 0.05, contact_penalty_weight: -0.05
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene = FrankaGearMeshSceneCfg(
            num_envs=self.scene.num_envs, env_spacing=self.scene.env_spacing
        )

        self.episode_length_s = 20.0

        fixed_asset_cfg = GearMeshFixedAssetCfg()

        # Set fixed asset geometry for observation frame computation
        self.actions.arm_action.fixed_asset_height = fixed_asset_cfg.height
        self.actions.arm_action.fixed_asset_base_height = fixed_asset_cfg.base_height
        self.actions.arm_action.fixed_asset_offset_x = fixed_asset_cfg.medium_gear_base_offset[0]
        self.actions.arm_action.fixed_asset_pos_noise = [0.001, 0.001, 0.001]

        self.rewards.kp_baseline = RewTerm(
            func=forge_mdp.kp_baseline,
            params={
                "held_cfg": SceneEntityCfg("held_asset"),
                "fixed_cfg": SceneEntityCfg("fixed_asset"),
                "num_keypoints": 4,
                "keypoint_scale": 0.15,
                "task_name": "gear_mesh",
                "fixed_asset_cfg": fixed_asset_cfg,
                "coef": [5, 4],
            },
            weight=1.0,
        )

        self.rewards.kp_coarse = RewTerm(
            func=forge_mdp.kp_coarse,
            params={
                "held_cfg": SceneEntityCfg("held_asset"),
                "fixed_cfg": SceneEntityCfg("fixed_asset"),
                "num_keypoints": 4,
                "keypoint_scale": 0.15,
                "task_name": "gear_mesh",
                "fixed_asset_cfg": fixed_asset_cfg,
                "coef": [50, 2],
            },
            weight=1.0,
        )

        self.rewards.kp_fine = RewTerm(
            func=forge_mdp.kp_fine,
            params={
                "held_cfg": SceneEntityCfg("held_asset"),
                "fixed_cfg": SceneEntityCfg("fixed_asset"),
                "num_keypoints": 4,
                "keypoint_scale": 0.15,
                "task_name": "gear_mesh",
                "fixed_asset_cfg": fixed_asset_cfg,
                "coef": [100, 0],
            },
            weight=1.0,
        )

        self.rewards.curr_engaged = RewTerm(
            func=forge_mdp.curr_engaged,
            params={
                "held_cfg": SceneEntityCfg("held_asset"),
                "fixed_cfg": SceneEntityCfg("fixed_asset"),
                "engage_threshold": 0.9,
                "task_name": "gear_mesh",
                "fixed_asset_cfg": fixed_asset_cfg,
            },
            weight=1.0,
        )

        self.rewards.curr_success = RewTerm(
            func=forge_mdp.curr_success,
            params={
                "held_cfg": SceneEntityCfg("held_asset"),
                "fixed_cfg": SceneEntityCfg("fixed_asset"),
                "success_threshold": 0.05,
                "check_rot": False,
                "task_name": "gear_mesh",
                "fixed_asset_cfg": fixed_asset_cfg,
            },
            weight=1.0,
        )

        self.rewards.contact_penalty = RewTerm(
            func=forge_mdp.contact_penalty,
            params={
                "sensor_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            },
            weight=-0.05,
        )

        self.rewards.success_pred_error = RewTerm(
            func=forge_mdp.success_pred_error,
            params={
                "held_cfg": SceneEntityCfg("held_asset"),
                "fixed_cfg": SceneEntityCfg("fixed_asset"),
                "success_threshold": 0.05,
                "check_rot": False,
                "task_name": "gear_mesh",
                "fixed_asset_cfg": fixed_asset_cfg,
                "delay_until_ratio": 0.25,
            },
            weight=-1.0,
        )

        self.terminations.task_success = DoneTerm(
            func=forge_mdp.task_success,
            params={
                "held_cfg": SceneEntityCfg("held_asset"),
                "fixed_cfg": SceneEntityCfg("fixed_asset"),
                "success_threshold": 0.05,
                "check_rot": False,
                "task_name": "gear_mesh",
                "fixed_asset_cfg": fixed_asset_cfg,
            },
        )


@configclass
class FrankaGearMeshEnvCfg_PLAY(FrankaGearMeshEnvCfg):
    """Play configuration with smaller scene."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
