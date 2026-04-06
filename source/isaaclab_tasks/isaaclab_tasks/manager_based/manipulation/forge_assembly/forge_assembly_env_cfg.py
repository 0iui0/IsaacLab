# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base environment configuration for forge assembly (peg-in-hole) task.

This config implements the forge peg-in-hole assembly strategy using the
manager-based framework. It mirrors the direct forge implementation's
observation/reward/action structure as closely as possible.

Action space: 7D via ForgeAssemblyAction (asset-relative Jacobian transpose)
Observations: Match direct/forge policy (24D) and critic (67D) dimensions
Rewards: Multi-scale keypoint tracking + contact penalty + action penalties
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.forge_assembly.mdp as mdp

# Factory asset directory on Nucleus (same as direct forge)
_FACTORY_ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"

# Import custom action term config
from isaaclab_tasks.manager_based.manipulation.forge_assembly.mdp.actions.actions_cfg import (
    ForgeAssemblyActionCfg,
)


##
# Scene
##


@configclass
class ForgeAssemblySceneCfg(InteractiveSceneCfg):
    """Scene for peg-in-hole assembly: robot + peg + hole."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # Hole: kinematic articulation with 0 joints (matches direct forge pattern exactly).
    # Uses Factory USD (factory_hole_8mm.usd) — a proper hollow cylinder mesh
    # that allows the peg to physically insert. Direct forge uses Articulation with 0 joints.
    hole = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Hole",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_FACTORY_ASSET_DIR}/factory_hole_8mm.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.425, 0.0, 0.423),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )

    # Peg: dynamic articulation with 0 joints held by gripper (matches direct forge pattern exactly).
    # Uses Factory USD (factory_peg_8mm.usd) matching direct forge exactly.
    peg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_FACTORY_ASSET_DIR}/factory_peg_8mm.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.425, 0.0, 0.423),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )

    # Robot: filled by subclass
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# Actions - forge assembly action term
##


@configclass
class ActionsCfg:
    """Action term using forge's asset-relative Jacobian transpose control."""

    arm_action: ActionTerm = MISSING  # Filled by robot-specific config with ForgeAssemblyActionCfg


##
# Observations - match direct/forge structure
##


@configclass
class ObservationsCfg:
    """Observations matching direct forge's policy/critic structure.

    Policy obs (24D): ee_pos_rel_hole(3) + ee_quat(4) + ee_linvel(3) + ee_angvel(3)
                    + ft_force(3) + force_threshold(1) + prev_actions(7) = 24D
    Critic obs (67D): all policy + joint_pos(7) + held_pos(3) + held_pos_rel_fixed(3)
                    + held_quat(4) + fixed_pos(3) + fixed_quat(4) + task_prop_gains(6)
                    + ema_factor(1) + pos_threshold(3) + rot_threshold(3) = 67D
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations matching direct forge."""

        # EE position relative to hole (forge: fingertip_pos_rel_fixed)
        ee_pos_rel_hole = ObsTerm(
            func=mdp.ee_pos_rel_hole,
            noise=Unoise(n_min=-0.00025, n_max=0.00025),
        )
        # EE orientation (forge: fingertip_quat)
        ee_quat = ObsTerm(
            func=mdp.ee_quat_canonical,
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        # EE linear velocity via finite difference (forge: ee_linvel)
        ee_linvel = ObsTerm(func=mdp.ee_linvel_fd)
        # EE angular velocity (forge: ee_angvel)
        ee_angvel = ObsTerm(func=mdp.ee_angvel_fd)
        # Force sensor: EMA-smoothed + noisy (forge: ft_force)
        ft_force = ObsTerm(
            func=mdp.ft_force_smooth_noisy,
            params={"smoothing_alpha": 0.25, "noise_std": 1.0},
        )
        # Contact penalty threshold (forge: force_threshold)
        force_threshold = ObsTerm(func=mdp.contact_threshold_obs)
        # Previous actions (forge: prev_actions) - 7D
        prev_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic gets everything policy sees plus privileged info."""

        # Same as policy
        ee_pos_rel_hole = ObsTerm(func=mdp.ee_pos_rel_hole)
        ee_quat = ObsTerm(func=mdp.ee_quat_canonical)
        ee_linvel = ObsTerm(func=mdp.ee_linvel_fd)
        ee_angvel = ObsTerm(func=mdp.ee_angvel_fd)
        ft_force = ObsTerm(func=mdp.ft_force_smooth_noisy, params={"smoothing_alpha": 0.25, "noise_std": 0.0})
        force_threshold = ObsTerm(func=mdp.contact_threshold_obs)
        prev_actions = ObsTerm(func=mdp.last_action)

        # Privileged: joint positions (from isaaclab.envs.mdp)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # Privileged: held asset (peg) pose (from isaaclab.envs.mdp)
        held_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("peg")},
        )
        held_pos_rel_fixed = ObsTerm(
            func=mdp.held_pos_rel_fixed,
        )
        held_quat = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("peg")},
        )
        # Privileged: fixed asset (hole) pose (from isaaclab.envs.mdp)
        fixed_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("hole")},
        )
        fixed_quat = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("hole")},
        )
        # Privileged: task proportional gains
        task_prop_gains = ObsTerm(func=mdp.task_prop_gains_obs)
        # Privileged: EMA factor
        ema_factor = ObsTerm(func=mdp.ema_factor_obs)
        # Privileged: thresholds
        pos_threshold = ObsTerm(func=mdp.pos_threshold_obs)
        rot_threshold = ObsTerm(func=mdp.rot_threshold_obs)

    # groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


##
# Events (randomization)
##


@configclass
class EventCfg:
    """Domain randomization matching direct forge."""

    # Reset
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Peg attaches to EE on reset with asset-in-gripper randomization
    # Direct forge: held_asset_pos_noise randomizes peg position relative to EE before gripping
    reset_peg = EventTerm(
        func=mdp.reset_peg_to_ee,
        mode="reset",
        params={"body_name": MISSING, "peg_offset": [0.0, 0.0, 0.05], "grip_noise_range": 0.02},
    )

    # Randomize hole pose
    # Direct forge: hand_init_pos_noise = [0.02, 0.02, 0.01] (relative EE-to-hole noise)
    # This replaces fixed_asset_init_pos_noise=[0.05,0.05,0.05] because we don't do IK
    randomize_hole = EventTerm(
        func=mdp.randomize_hole_pose,
        mode="reset",
        params={
            # Reduced from ±20mm to ±5mm xy, ±2mm z — the success threshold
            # is 2.5mm xy so randomization must be smaller for any chance of success.
            # Direct forge uses IK-based init with hand_init_pos_noise=[0.005, 0.005, 0.002].
            "pos_range": {"x": [-0.005, 0.005], "y": [-0.005, 0.005], "z": [-0.002, 0.002]},
            "yaw_range": [-3.14159, 3.14159],
        },
    )

    # Randomize controller gains (forge: +/-41%) - from isaaclab.envs.mdp
    randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.59, 1.41),
            "damping_distribution_params": (0.59, 1.41),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # Randomize joint friction - from isaaclab.envs.mdp
    randomize_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": (0.3, 0.7),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # Physics material randomization - from isaaclab.envs.mdp
    peg_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("peg", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    hole_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("hole", body_names=".*"),
            "static_friction_range": (0.25, 1.25),
            "dynamic_friction_range": (0.25, 0.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )

    # Forge-specific randomizations
    randomize_ema = EventTerm(
        func=mdp.randomize_ema_factor,
        mode="reset",
        params={"ema_range": [0.025, 0.1]},
    )

    randomize_contact_threshold = EventTerm(
        func=mdp.randomize_contact_threshold,
        mode="reset",
        params={"threshold_range": [5.0, 10.0]},
    )

    # Task gains randomization (forge: +/-41%)
    randomize_task_gains = EventTerm(
        func=mdp.randomize_task_gains,
        mode="reset",
        params={"noise_level": 0.41},
    )

    # Position/rotation threshold randomization
    randomize_pos_threshold = EventTerm(
        func=mdp.randomize_pos_threshold,
        mode="reset",
        params={"noise_level": 0.25},
    )

    randomize_rot_threshold = EventTerm(
        func=mdp.randomize_rot_threshold,
        mode="reset",
        params={"noise_level": 0.29},
    )

    # Dead zone randomization
    randomize_dead_zone = EventTerm(
        func=mdp.randomize_dead_zone,
        mode="reset",
        params={
            "default_dead_zone": [5.0, 5.0, 5.0, 1.0, 1.0, 1.0],
        },
    )


##
# Rewards
##


@configclass
class RewardsCfg:
    """Rewards matching direct forge's FULL reward structure.

    Direct forge calls super()._get_rewards() from FactoryEnv, which provides:
    - kp_baseline (+1.0): squashing_fn(keypoint_dist, a=5, b=4)
    - kp_coarse (+1.0): squashing_fn(keypoint_dist, a=50, b=2)
    - kp_fine (+1.0): squashing_fn(keypoint_dist, a=100, b=0)
    - action_penalty_ee (0.0, disabled)
    - action_grad_penalty (-0.1, from ForgeTask.action_grad_penalty_scale=0.1)
    - curr_engaged (+1.0): binary when insertion > 90% depth
    - curr_success (+1.0): binary when fully inserted

    Then forge_env._get_rewards() adds ON TOP:
    - action_penalty_asset (-0.001): pos_error + rot_error
    - contact_penalty (-0.2): relu(force - threshold)
    - success_pred_error (-1.0): |true_success - predicted| after delay

    The keypoint rewards are the PRIMARY positive signals. They were
    previously disabled due to a misunderstanding — forge DOES use them
    via super()._get_rewards().
    """

    # --- Factory keypoint rewards (PRIMARY positive signals, weight +1.0) ---
    keypoint_baseline = RewTerm(
        func=mdp.keypoint_peg_hole_error_exp,
        weight=1.0,
        params={
            "kp_exp_coeffs": [(5, 4)],
            "kp_use_sum_of_exps": False,
            "keypoint_scale": 0.15,
        },
    )
    keypoint_coarse = RewTerm(
        func=mdp.keypoint_peg_hole_error_exp,
        weight=1.0,
        params={
            "kp_exp_coeffs": [(50, 2)],
            "kp_use_sum_of_exps": False,
            "keypoint_scale": 0.15,
        },
    )
    keypoint_fine = RewTerm(
        func=mdp.keypoint_peg_hole_error_exp,
        weight=1.0,
        params={
            "kp_exp_coeffs": [(100, 0)],
            "kp_use_sum_of_exps": False,
            "keypoint_scale": 0.15,
        },
    )

    # --- Forge-specific additions (penalties) ---
    action_penalty_asset = RewTerm(
        func=mdp.action_penalty_asset,
        weight=-0.001,
    )
    contact_penalty = RewTerm(
        func=mdp.contact_force_penalty,
        weight=-0.2,
    )
    success_pred_error = RewTerm(
        func=mdp.success_prediction_penalty,
        weight=-1.0,
    )

    # --- Success bonuses (from FactoryEnv) ---
    curr_engaged = RewTerm(
        func=mdp.peg_insertion_engaged,
        weight=1.0,
    )
    curr_success = RewTerm(
        func=mdp.peg_insertion_success,
        weight=1.0,
    )

    # --- Action gradient penalty (match direct forge ForgeTask: scale=0.1) ---
    action_grad_penalty = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    # --- Disabled rewards ---
    action_penalty = RewTerm(func=mdp.action_l2, weight=0.0)


##
# Terminations
##


@configclass
class TerminationsCfg:
    """Termination conditions matching direct forge (timeout only)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Note: direct forge does NOT have early termination.
    # Peg-dropped termination is optional for safety.


##
# Environment
##


@configclass
class ForgeAssemblyEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for peg-in-hole assembly using forge strategy."""

    # Scene
    scene: ForgeAssemblySceneCfg = ForgeAssemblySceneCfg(num_envs=4096, env_spacing=2.5)
    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # Simulation
    sim: SimulationCfg = SimulationCfg(
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_collision_stack_size=2**28,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,
        ),
    )

    def __post_init__(self):
        # Episode length matching forge peg_insert (10s at 120Hz with decimation 8)
        self.episode_length_s = 10.0
        self.decimation = 8
        self.sim.render_interval = self.decimation
        self.sim.dt = 1.0 / 120.0
        self.viewer.eye = (2.5, 2.5, 2.5)
