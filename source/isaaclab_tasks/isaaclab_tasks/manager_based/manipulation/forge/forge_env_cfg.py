# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base environment configuration for FORGE tasks.

This module provides the base configuration for all FORGE tasks which use
impedance control with EMA smoothing, matching the direct version's
policy I/O, control, and training setup.

Key differences from the IK-based version:
- Impedance control (torque) instead of Differential IK (position)
- 7D action space (6D pose + success prediction)
- EMA smoothing with randomized factor
- Multi-scale keypoint rewards matching Factory/FORGE
- Gain/threshold randomization events
- rl_games LSTM training configuration
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

# Import custom FORGE MDP functions
import isaaclab_tasks.manager_based.manipulation.forge.mdp as forge_mdp

# Import built-in MDP functions
import isaaclab.envs.mdp as lab_mdp

# Import action term
from isaaclab_tasks.manager_based.manipulation.forge.mdp.actions import ForgeImpedanceActionCfg

##
# Scene definition
##

FORGE_ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


@configclass
class ForgeSceneCfg(InteractiveSceneCfg):
    """Configuration for the FORGE scene with a robot and manipulation assets."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # robot (override in task-specific configs)
    robot: AssetBaseCfg = MISSING

    # manipulation assets
    fixed_asset: RigidObjectCfg = MISSING
    held_asset: RigidObjectCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms - FORGE uses asset-relative actions, no commands needed."""
    pass


@configclass
class ActionsCfg:
    """Action specifications: 7D impedance control.

    Action space: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, success_pred]
    - [:, 0:3] = position delta relative to fixed asset
    - [:, 3:6] = rotation delta (only yaw used, roll/pitch zeroed)
    - [:, 6]   = success prediction (stored, not used for control)
    """

    arm_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications matching direct version exactly.

    Policy observations (27D):
        fingertip_pos_rel_fixed: 3D, fingertip_quat: 4D, ee_linvel: 3D,
        ee_angvel: 3D, ft_force: 3D, force_threshold: 1D, prev_actions: 7D

    Critic observations (~64D):
        All policy obs + joint_pos(7D) + held_pose(7D) + held_pos_rel_fixed(3D)
        + fixed_pose(7D) + task_prop_gains(6D) + ema_factor(1D)
        + pos_threshold(3D) + rot_threshold(3D)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations (27D)."""

        fingertip_pos_rel_fixed = ObsTerm(
            func=forge_mdp.fingertip_pos_rel_fixed,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered"),
                "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            },
        )
        fingertip_quat = ObsTerm(
            func=forge_mdp.fingertip_quat,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered")},
        )
        ee_linvel = ObsTerm(
            func=forge_mdp.ee_linvel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered")},
        )
        ee_angvel = ObsTerm(
            func=forge_mdp.ee_angvel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered")},
        )
        ft_force = ObsTerm(
            func=forge_mdp.ft_force,
            params={
                "sensor_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            },
        )
        force_threshold = ObsTerm(func=forge_mdp.force_threshold)
        prev_actions = ObsTerm(func=forge_mdp.prev_actions)

        def __post_init__(self):
            self.enable_corruption = False  # Noise is added inside action term
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations (~64D) with full state information."""

        # Policy obs (duplicated for critic)
        fingertip_pos = ObsTerm(
            func=forge_mdp.fingertip_pos_rel_fixed,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered"),
                "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            },
        )
        fingertip_quat_c = ObsTerm(
            func=forge_mdp.fingertip_quat,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered")},
        )
        ee_linvel_c = ObsTerm(
            func=forge_mdp.ee_linvel_raw,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered")},
        )
        ee_angvel_c = ObsTerm(
            func=forge_mdp.ee_angvel_raw,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_fingertip_centered")},
        )
        # Additional state
        joint_pos = ObsTerm(
            func=forge_mdp.joint_pos_arm,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        held_pos = ObsTerm(
            func=forge_mdp.held_pos,
            params={"asset_cfg": SceneEntityCfg("held_asset")},
        )
        held_pos_rel_fixed = ObsTerm(
            func=forge_mdp.held_pos_rel_fixed,
            params={
                "held_cfg": SceneEntityCfg("held_asset"),
                "fixed_cfg": SceneEntityCfg("fixed_asset"),
            },
        )
        held_quat = ObsTerm(
            func=forge_mdp.held_quat,
            params={"asset_cfg": SceneEntityCfg("held_asset")},
        )
        fixed_pos = ObsTerm(
            func=forge_mdp.fixed_pos,
            params={"asset_cfg": SceneEntityCfg("fixed_asset")},
        )
        fixed_quat = ObsTerm(
            func=forge_mdp.fixed_quat,
            params={"asset_cfg": SceneEntityCfg("fixed_asset")},
        )
        task_prop_gains = ObsTerm(func=forge_mdp.task_prop_gains)
        ema_factor = ObsTerm(func=forge_mdp.ema_factor)
        ft_force_c = ObsTerm(
            func=forge_mdp.ft_force_raw,
            params={
                "sensor_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            },
        )
        pos_threshold = ObsTerm(func=forge_mdp.pos_threshold)
        rot_threshold = ObsTerm(func=forge_mdp.rot_threshold)
        force_threshold_c = ObsTerm(func=forge_mdp.force_threshold_critic)
        prev_actions_c = ObsTerm(func=forge_mdp.prev_actions)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for randomization events matching direct ForgeEnv._reset_idx()."""

    # Reset-level randomization (called at episode start)
    randomize_impedance_gains = EventTerm(
        func=forge_mdp.randomize_impedance_gains,
        mode="reset",
        params={"noise_levels": [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]},
    )

    randomize_action_thresholds = EventTerm(
        func=forge_mdp.randomize_action_thresholds,
        mode="reset",
        params={
            "pos_noise_levels": [0.25, 0.25, 0.25],
            "rot_noise_levels": [0.29, 0.29, 0.29],
        },
    )

    randomize_ema_factor = EventTerm(
        func=forge_mdp.randomize_ema_factor,
        mode="reset",
        params={"ema_factor_range": [0.025, 0.1]},
    )

    randomize_contact_threshold = EventTerm(
        func=forge_mdp.randomize_contact_threshold,
        mode="reset",
        params={"lower_bound": 5.0, "upper_bound": 10.0},
    )

    randomize_flip_quats = EventTerm(
        func=forge_mdp.randomize_flip_quats,
        mode="reset",
    )

    reset_force_sensor = EventTerm(
        func=forge_mdp.reset_force_sensor,
        mode="reset",
    )

    # Update fixed asset observation frame (MUST fire before reset_action_state)
    update_fixed_pos_obs_frame = EventTerm(
        func=forge_mdp.update_fixed_pos_obs_frame,
        mode="reset",
    )

    reset_action_state = EventTerm(
        func=forge_mdp.reset_action_state,
        mode="reset",
    )

    # Interval-based dead zone re-randomization (every 2s)
    randomize_dead_zone = EventTerm(
        func=forge_mdp.randomize_dead_zone,
        mode="interval",
        interval_range_s=(2.0, 2.0),
        params={"default_dead_zone": [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]},
    )

    # Mass randomization
    randomize_held_asset_mass = EventTerm(
        func=lab_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "mass_distribution_params": (-0.005, 0.005),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # Physics materials
    held_physics_material = EventTerm(
        func=lab_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    fixed_physics_material = EventTerm(
        func=lab_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("fixed_asset"),
            "static_friction_range": (0.25, 1.25),
            "dynamic_friction_range": (0.25, 0.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms matching direct Factory + FORGE rewards.

    Base rewards (from Factory):
        kp_baseline:    squashing(keypoint_dist, [5, 4])   weight=1.0
        kp_coarse:      squashing(keypoint_dist, [50, 2])   weight=1.0
        kp_fine:        squashing(keypoint_dist, [100, 0])  weight=1.0
        action_penalty_ee:    -||actions||_2                weight=0.0 (disabled)
        action_grad_penalty:  -||actions-prev||_2           weight=-0.1
        curr_engaged:   engaged_flag                        weight=1.0
        curr_success:   success_flag                        weight=1.0

    FORGE additional:
        action_penalty_asset: -(pos_err+yaw_err)           weight=-0.001
        contact_penalty:      -ReLU(force-threshold)        weight=-0.05
        success_pred_error:   -|true-predicted|             weight=-1.0 (dynamic)
    """

    # Multi-scale keypoint rewards
    kp_baseline = RewTerm(
        func=forge_mdp.kp_baseline,
        params={
            "held_cfg": SceneEntityCfg("held_asset"),
            "fixed_cfg": SceneEntityCfg("fixed_asset"),
            "num_keypoints": 4,
            "keypoint_scale": 0.15,
            "task_name": MISSING,
            "fixed_asset_cfg": MISSING,
            "coef": [5, 4],
        },
        weight=1.0,
    )

    kp_coarse = RewTerm(
        func=forge_mdp.kp_coarse,
        params={
            "held_cfg": SceneEntityCfg("held_asset"),
            "fixed_cfg": SceneEntityCfg("fixed_asset"),
            "num_keypoints": 4,
            "keypoint_scale": 0.15,
            "task_name": MISSING,
            "fixed_asset_cfg": MISSING,
            "coef": [50, 2],
        },
        weight=1.0,
    )

    kp_fine = RewTerm(
        func=forge_mdp.kp_fine,
        params={
            "held_cfg": SceneEntityCfg("held_asset"),
            "fixed_cfg": SceneEntityCfg("fixed_asset"),
            "num_keypoints": 4,
            "keypoint_scale": 0.15,
            "task_name": MISSING,
            "fixed_asset_cfg": MISSING,
            "coef": [100, 0],
        },
        weight=1.0,
    )

    # Action penalties
    action_penalty_ee = RewTerm(
        func=forge_mdp.action_penalty_ee,
        weight=0.0,  # Disabled by default (matching direct: scale=0)
    )

    action_grad_penalty = RewTerm(
        func=forge_mdp.action_grad_penalty,
        weight=-0.1,
    )

    # Engagement and success
    curr_engaged = RewTerm(
        func=forge_mdp.curr_engaged,
        params={
            "held_cfg": SceneEntityCfg("held_asset"),
            "fixed_cfg": SceneEntityCfg("fixed_asset"),
            "engage_threshold": 0.9,
            "task_name": MISSING,
            "fixed_asset_cfg": MISSING,
        },
        weight=1.0,
    )

    curr_success = RewTerm(
        func=forge_mdp.curr_success,
        params={
            "held_cfg": SceneEntityCfg("held_asset"),
            "fixed_cfg": SceneEntityCfg("fixed_asset"),
            "success_threshold": MISSING,
            "check_rot": False,
            "task_name": MISSING,
            "fixed_asset_cfg": MISSING,
        },
        weight=1.0,
    )

    # FORGE-specific rewards
    action_penalty_asset = RewTerm(
        func=forge_mdp.action_penalty_asset,
        weight=-0.001,
    )

    contact_penalty = RewTerm(
        func=forge_mdp.contact_penalty,
        params={
            "sensor_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
        },
        weight=-0.05,
    )

    success_pred_error = RewTerm(
        func=forge_mdp.success_pred_error,
        params={
            "held_cfg": SceneEntityCfg("held_asset"),
            "fixed_cfg": SceneEntityCfg("fixed_asset"),
            "success_threshold": MISSING,
            "check_rot": False,
            "task_name": MISSING,
            "fixed_asset_cfg": MISSING,
            "delay_until_ratio": 0.25,
        },
        weight=-1.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms - all envs reset together (time-based only)."""

    time_out = DoneTerm(func=forge_mdp.time_out)

    # Success logging (not used for termination, only for extras)
    task_success = DoneTerm(
        func=forge_mdp.task_success,
        params={
            "held_cfg": SceneEntityCfg("held_asset"),
            "fixed_cfg": SceneEntityCfg("fixed_asset"),
            "success_threshold": MISSING,
            "check_rot": False,
            "task_name": MISSING,
            "fixed_asset_cfg": MISSING,
        },
    )


##
# Environment configuration
##


@configclass
class ForgeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the FORGE manipulation environment with impedance control."""

    # Scene settings
    scene: ForgeSceneCfg = ForgeSceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Environment settings
    episode_length_s: float = 10.0

    def __post_init__(self):
        """Post initialization - match direct version sim params."""
        # Match direct version: decimation=8, dt=1/120
        self.decimation = 8
        self.sim.render_interval = self.decimation

        # Viewer position
        self.viewer.eye = (3.5, 3.5, 3.5)

        # Simulation settings matching direct version
        self.sim.dt = 1.0 / 120.0
        self.sim.physx = PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,
        )
        self.sim.physics_material = RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        )
