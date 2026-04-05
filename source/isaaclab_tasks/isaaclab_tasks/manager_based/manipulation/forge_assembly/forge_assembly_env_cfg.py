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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.forge_assembly.mdp as mdp

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

    # Hole: kinematic rigid body (cylinder with hole)
    hole = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Hole",
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.03,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
                solver_position_iteration_count=64,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 1.05)),
    )

    # Peg: dynamic rigid body attached to EE (small cylinder)
    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        spawn=sim_utils.CylinderCfg(
            radius=0.012,
            height=0.06,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=64,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 1.2)),
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

    # Peg attaches to EE on reset
    reset_peg = EventTerm(
        func=mdp.reset_peg_to_ee,
        mode="reset",
        params={"body_name": MISSING, "peg_offset": [0.0, 0.0, 0.05]},
    )

    # Randomize hole pose
    randomize_hole = EventTerm(
        func=mdp.randomize_hole_pose,
        mode="reset",
        params={
            "pos_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [0.0, 0.0]},
            "yaw_range": [-3.14, 3.14],
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
    """Rewards matching direct forge's reward structure exactly.

    Direct forge overrides FactoryEnv.compute_reward and uses ONLY:
    - action_penalty_asset: pos_error + rot_error (scale -0.001)
    - contact_penalty: relu(||force|| - threshold) (scale -0.2)
    - success_pred_error: |true_success - predicted| (scale -1.0 after delay)

    Factory keypoint rewards (kp_baseline/coarse/fine) and action_grad_penalty
    are NOT used by forge — they are disabled here.
    """

    # --- Active rewards (matching direct forge) ---

    # Asset-relative action penalty (forge: action_penalty_asset, scale -0.001)
    action_penalty_asset = RewTerm(
        func=mdp.action_penalty_asset,
        weight=-0.001,
    )

    # Contact force penalty (forge: contact_penalty, scale -0.2)
    contact_penalty = RewTerm(
        func=mdp.contact_force_penalty,
        weight=-0.2,
    )

    # Success prediction error (forge: success_pred_error, scale -1.0 after delay)
    success_pred_error = RewTerm(
        func=mdp.success_prediction_penalty,
        weight=-1.0,
    )

    # --- Disabled rewards (not used by direct forge) ---

    # Keypoint rewards: disabled (direct forge does not use Factory keypoint rewards)
    keypoint_baseline = RewTerm(
        func=mdp.keypoint_peg_hole_error_exp,
        weight=0.0,
        params={
            "kp_exp_coeffs": [(5, 4)],
            "kp_use_sum_of_exps": False,
            "keypoint_scale": 0.15,
        },
    )
    keypoint_coarse = RewTerm(
        func=mdp.keypoint_peg_hole_error_exp,
        weight=0.0,
        params={
            "kp_exp_coeffs": [(50, 2)],
            "kp_use_sum_of_exps": False,
            "keypoint_scale": 0.15,
        },
    )
    keypoint_fine = RewTerm(
        func=mdp.keypoint_peg_hole_error_exp,
        weight=0.0,
        params={
            "kp_exp_coeffs": [(100, 0)],
            "kp_use_sum_of_exps": False,
            "keypoint_scale": 0.15,
        },
    )

    # Action penalty: disabled (direct forge: action_penalty_ee_scale = 0.0)
    action_penalty = RewTerm(func=mdp.action_l2, weight=0.0)

    # Action gradient penalty: disabled (direct forge defines but never uses)
    action_grad_penalty = RewTerm(func=mdp.action_rate_l2, weight=0.0)


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
