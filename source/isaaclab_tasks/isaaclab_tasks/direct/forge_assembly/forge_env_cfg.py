# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from .forge_events import randomize_dead_zone
from .forge_tasks_cfg import ASSET_DIR, ForgeGearMesh, ForgeNutThread, ForgePegInsert, ForgeTask, _make_fixed_asset_cfg
from .robot_profiles import CR5_FORGE_PROFILE, FRANKA_FORGE_PROFILE, UR10_FORGE_PROFILE, RobotProfile

# ---------------------------------------------------------------------------
# Observation / state dimension lookup tables
# ---------------------------------------------------------------------------

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "force_threshold": 1,
    "ft_force": 3,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,  # Will be overridden dynamically based on robot profile
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
    "force_threshold": 1,
    "ft_force": 3,
}


# ---------------------------------------------------------------------------
# Control config
# ---------------------------------------------------------------------------


@configclass
class CtrlCfg:
    ema_factor: float = 0.2

    pos_action_bounds: list = [0.05, 0.05, 0.05]
    rot_action_bounds: list = [1.0, 1.0, 1.0]

    pos_action_threshold: list = [0.02, 0.02, 0.02]
    rot_action_threshold: list = [0.097, 0.097, 0.097]

    reset_joints: list = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
    reset_task_prop_gains: list = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale: float = 10.0
    default_task_prop_gains: list = [100, 100, 100, 30, 30, 30]

    # Null space parameters.
    default_dof_pos_tensor: list = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null: float = 10.0
    kd_null: float = 6.3246


@configclass
class ForgeCtrlCfg(CtrlCfg):
    ema_factor_range: list = [0.025, 0.1]
    default_task_prop_gains: list = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]
    task_prop_gains_noise_level: list = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]
    pos_threshold_noise_level: list = [0.25, 0.25, 0.25]
    rot_threshold_noise_level: list = [0.29, 0.29, 0.29]
    default_dead_zone: list = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Observation randomization config
# ---------------------------------------------------------------------------


@configclass
class ObsRandCfg:
    fixed_asset_pos: list = [0.001, 0.001, 0.001]


@configclass
class ForgeObsRandCfg(ObsRandCfg):
    fingertip_pos: float = 0.00025
    fingertip_rot_deg: float = 0.1
    ft_force: float = 1.0


# ---------------------------------------------------------------------------
# Event config
# ---------------------------------------------------------------------------


@configclass
class EventCfg:
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "mass_distribution_params": (-0.005, 0.005),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    held_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
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
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("fixed_asset"),
            "static_friction_range": (0.25, 1.25),
            "dynamic_friction_range": (0.25, 0.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    dead_zone_thresholds = EventTerm(
        func=randomize_dead_zone,
        mode="interval",
        interval_range_s=(2.0, 2.0),
    )


# ---------------------------------------------------------------------------
# Main environment config (robot-agnostic)
# ---------------------------------------------------------------------------


@configclass
class ForgeEnvCfg(DirectRLEnvCfg):
    # Robot profile — must be overridden by task-specific configs.
    robot_profile: RobotProfile = FRANKA_FORGE_PROFILE

    decimation: int = 8
    action_space: int = 7

    obs_rand: ForgeObsRandCfg = ForgeObsRandCfg()
    ctrl: ForgeCtrlCfg = ForgeCtrlCfg()
    task: ForgeTask = ForgeTask()
    events: EventCfg = EventCfg()

    ft_smoothing_factor: float = 0.25

    task_name: str = "peg_insert"
    episode_length_s: float = 10.0

    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "ft_force",
        "force_threshold",
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "task_prop_gains",
        "ema_factor",
        "ft_force",
        "pos_threshold",
        "rot_threshold",
        "force_threshold",
    ]

    def __post_init__(self):
        # Compute observation/state spaces from profile-driven dimensions.
        STATE_DIM_CFG["joint_pos"] = self.robot_profile.num_arm_joints
        self.observation_space = sum(OBS_DIM_CFG[obs] for obs in self.obs_order) + self.action_space
        self.state_space = sum(STATE_DIM_CFG[state] for state in self.state_order) + self.action_space

    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
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
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, clone_in_fabric=True)

    # Robot ArticulationCfg is derived from robot_profile at runtime.
    # The environment accesses it via self.profile.robot.


# ---------------------------------------------------------------------------
# Task-specific configs: Peg Insert (robot-agnostic base)
# ---------------------------------------------------------------------------


@configclass
class ForgeTaskPegInsertCfg(ForgeEnvCfg):
    task_name = "peg_insert"
    task = ForgePegInsert()
    episode_length_s = 10.0


@configclass
class ForgeTaskGearMeshCfg(ForgeEnvCfg):
    task_name = "gear_mesh"
    task = ForgeGearMesh()
    episode_length_s = 20.0


@configclass
class ForgeTaskNutThreadCfg(ForgeEnvCfg):
    task_name = "nut_thread"
    task = ForgeNutThread()
    episode_length_s = 30.0


# ---------------------------------------------------------------------------
# Robot-specific config variants: Franka
# ---------------------------------------------------------------------------


@configclass
class FrankaForgeTaskPegInsertCfg(ForgeTaskPegInsertCfg):
    robot_profile: RobotProfile = FRANKA_FORGE_PROFILE


@configclass
class FrankaForgeTaskGearMeshCfg(ForgeTaskGearMeshCfg):
    robot_profile: RobotProfile = FRANKA_FORGE_PROFILE


@configclass
class FrankaForgeTaskNutThreadCfg(ForgeTaskNutThreadCfg):
    robot_profile: RobotProfile = FRANKA_FORGE_PROFILE


# ---------------------------------------------------------------------------
# Event config for fixed-peg robots (no held_asset events)
# ---------------------------------------------------------------------------


@configclass
class EventCfgFixedPeg(EventCfg):
    """Event configuration for fixed-peg robots (UR10/CR5).

    The peg is spawned as a collision shape on the EE link, not as a separate
    held_asset articulation. Therefore, held_asset event terms must be disabled.
    """
    # Disable held_asset events — peg is part of robot, not a separate asset
    object_scale_mass = None
    held_physics_material = None


# ---------------------------------------------------------------------------
# Robot-specific config variants: UR10
# ---------------------------------------------------------------------------


@configclass
class UR10ForgeTaskPegInsertCfg(ForgeTaskPegInsertCfg):
    """UR10 configuration for peg insertion with fixed peg.

    UR10 has a shorter effective reach than Franka and is 6-DOF (no redundancy).
    The fixed_asset (hole) is placed closer to the robot base, and the reset pose
    is more forward-reaching to improve IK convergence.
    """
    robot_profile: RobotProfile = UR10_FORGE_PROFILE
    events: EventCfgFixedPeg = EventCfgFixedPeg()

    # UR10 has 6 arm joints, so reset_joints and default_dof_pos_tensor must have 6 values
    ctrl: ForgeCtrlCfg = ForgeCtrlCfg(
        reset_joints=[0.0, -0.5, -0.5, -0.5, 0.0, 0.0],
        default_dof_pos_tensor=[0.0, -1.712, 1.712, 0.0, -1.571, 0.0],
    )

    # UR10-specific task overrides: closer fixed_asset, tighter workspace
    task = ForgePegInsert(
        # hand_init_pos[2] must place the peg just above (not inside) the hole.
        # Peg bottom = EE - (peg_offset + peg_height/2) = EE - 0.065m.
        # With 0.067, peg bottom is 2mm above hole top — close enough for
        # contact during early training but avoids interpenetration at reset.
        hand_init_pos=[0.0, 0.0, 0.067],
        hand_init_pos_noise=[0.01, 0.01, 0.005],
        fixed_asset_init_pos_noise=[0.03, 0.03, 0.01],
        # Lower contact threshold: UR10 peg is a collision-only prim (no mass),
        # so contact forces are smaller than Franka's held asset (19g mass).
        contact_penalty_threshold_range=[1.0, 3.0],
        contact_penalty_scale=0.2,
        fixed_asset=_make_fixed_asset_cfg(
            "/World/envs/env_.*/FixedAsset",
            f"{ASSET_DIR}/factory_hole_8mm.usd",
            0.05,
            pos=(0.45, 0.0, 0.05),
        ),
    )


# ---------------------------------------------------------------------------
# Robot-specific config variants: CR5
# ---------------------------------------------------------------------------


@configclass
class CR5ForgeTaskPegInsertCfg(ForgeTaskPegInsertCfg):
    """CR5 configuration for peg insertion with fixed peg.

    Note: Uses EventCfgFixedPeg to exclude held_asset events.
    """
    robot_profile: RobotProfile = CR5_FORGE_PROFILE
    events: EventCfgFixedPeg = EventCfgFixedPeg()

    # CR5 has 6 arm joints, so reset_joints and default_dof_pos_tensor must have 6 values
    # These override the 7-element Franka defaults from CtrlCfg
    ctrl: ForgeCtrlCfg = ForgeCtrlCfg(
        reset_joints=[0.0, -1.571, 1.571, 0.0, 0.0, 0.0],  # CR5 home position
        default_dof_pos_tensor=[0.0, -1.571, 1.571, 0.0, 0.0, 0.0],  # CR5 null-space position
    )
