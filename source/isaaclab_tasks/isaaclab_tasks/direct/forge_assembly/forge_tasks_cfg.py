# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forge Assembly: task configurations.

Self-contained task configs inlined from factory_tasks_cfg to remove
dependency on the factory module.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


# ---------------------------------------------------------------------------
# Base asset config classes
# ---------------------------------------------------------------------------


@configclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05
    grip_offset: float = 0.0  # Z offset from default grip position (positive = deeper into gripper)


@configclass
class RobotCfg:
    robot_usd: str = ""
    franka_fingerpad_length: float = 0.017608
    friction: float = 0.75


# ---------------------------------------------------------------------------
# Base task config
# ---------------------------------------------------------------------------


@configclass
class ForgeTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ""
    duration_s: float = 5.0

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    asset_size: float = 0.0

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.015]
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0, 2.356]
    hand_init_orn_noise: list = [0.0, 0.0, 1.57]

    # Action
    unidirectional_rot: bool = False

    # Fixed Asset
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # Held Asset
    held_asset_pos_noise: list = [0.0, 0.006, 0.003]
    held_asset_rot_init: float = -90.0

    # Reward
    ee_success_yaw: float = 0.0
    action_penalty_ee_scale: float = 0.0
    action_grad_penalty_scale: float = 0.1
    action_penalty_asset_scale: float = 0.001
    ee_dist_reward_scale: float = 50.0
    ee_dist_reward_weight: float = 0.0
    z_descent_reward_scale: float = 100.0  # Z-descent reward scale (higher = stronger gradient when above hole)
    z_descent_reward_weight: float = 0.0  # Z-descent reward weight (0 = disabled)
    insertion_reward_weight: float = 0.0  # Peg insertion depth reward (0 = disabled)
    q1_reg_weight: float = 0.0  # Base joint regularization (0 = disabled)
    contact_penalty_scale: float = 0.05
    delay_until_ratio: float = 0.25
    contact_penalty_threshold_range: list = [5.0, 10.0]
    num_keypoints: int = 4
    keypoint_scale: float = 0.15
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    success_threshold: float = 0.04
    engage_threshold: float = 0.9


# ---------------------------------------------------------------------------
# Asset definitions
# ---------------------------------------------------------------------------


@configclass
class Peg8mm(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_peg_8mm.usd"
    diameter = 0.007986
    height = 0.050
    mass = 0.019


@configclass
class Hole8mm(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_hole_8mm.usd"
    diameter = 0.0081
    height = 0.025
    base_height = 0.0


@configclass
class GearBase(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_gear_base.usd"
    height = 0.02
    base_height = 0.005
    small_gear_base_offset = [5.075e-2, 0.0, 0.0]
    medium_gear_base_offset = [2.025e-2, 0.0, 0.0]
    large_gear_base_offset = [-3.025e-2, 0.0, 0.0]


@configclass
class MediumGear(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_gear_medium.usd"
    diameter = 0.03
    height: float = 0.03
    mass = 0.012


@configclass
class NutM16(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_nut_m16.usd"
    diameter = 0.024
    height = 0.01
    mass = 0.03
    friction = 0.01


@configclass
class BoltM16(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_bolt_m16.usd"
    diameter = 0.024
    height = 0.025
    base_height = 0.01
    thread_pitch = 0.002


# ---------------------------------------------------------------------------
# Helper to create asset ArticulationCfg
# ---------------------------------------------------------------------------


def _make_asset_cfg(prim_path: str, usd_path: str, mass: float, disable_gravity: bool = False) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=disable_gravity,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


def _make_fixed_asset_cfg(
    prim_path: str, usd_path: str, mass: float, pos: tuple = (0.6, 0.0, 0.05)
) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos, rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------


@configclass
class ForgePegInsert(ForgeTask):
    name = "peg_insert"
    fixed_asset_cfg = Hole8mm()
    held_asset_cfg = Peg8mm()
    asset_size = 8.0
    duration_s = 10.0

    hand_init_pos: list = [0.0, 0.0, 0.047]
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]

    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    held_asset_pos_noise: list = [0.003, 0.0, 0.003]
    held_asset_rot_init: float = 0.0

    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    success_threshold: float = 0.04
    engage_threshold: float = 0.9
    contact_penalty_scale: float = 0.2

    fixed_asset: ArticulationCfg = _make_fixed_asset_cfg(
        "/World/envs/env_.*/FixedAsset", f"{ASSET_DIR}/factory_hole_8mm.usd", 0.05
    )
    held_asset: ArticulationCfg = _make_asset_cfg(
        "/World/envs/env_.*/HeldAsset", f"{ASSET_DIR}/factory_peg_8mm.usd", 0.019, disable_gravity=True
    )


@configclass
class ForgeGearMesh(ForgeTask):
    name = "gear_mesh"
    fixed_asset_cfg = GearBase()
    held_asset_cfg = MediumGear()
    duration_s = 20.0

    small_gear_usd = f"{ASSET_DIR}/factory_gear_small.usd"
    large_gear_usd = f"{ASSET_DIR}/factory_gear_large.usd"

    small_gear_cfg: ArticulationCfg = _make_asset_cfg(
        "/World/envs/env_.*/SmallGearAsset", f"{ASSET_DIR}/factory_gear_small.usd", 0.019
    )
    large_gear_cfg: ArticulationCfg = _make_asset_cfg(
        "/World/envs/env_.*/LargeGearAsset", f"{ASSET_DIR}/factory_gear_large.usd", 0.019
    )

    add_flanking_gears = True
    add_flanking_gears_prob = 1.0

    hand_init_pos: list = [0.0, 0.0, 0.035]
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]

    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 15.0

    held_asset_pos_noise: list = [0.003, 0.0, 0.003]
    held_asset_rot_init: float = -90.0

    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    success_threshold: float = 0.05
    engage_threshold: float = 0.9

    fixed_asset: ArticulationCfg = _make_fixed_asset_cfg(
        "/World/envs/env_.*/FixedAsset", f"{ASSET_DIR}/factory_gear_base.usd", 0.05
    )
    held_asset: ArticulationCfg = _make_asset_cfg(
        "/World/envs/env_.*/HeldAsset", f"{ASSET_DIR}/factory_gear_medium.usd", 0.012, disable_gravity=True
    )


@configclass
class ForgeNutThread(ForgeTask):
    name = "nut_thread"
    fixed_asset_cfg = BoltM16()
    held_asset_cfg = NutM16()
    asset_size = 16.0
    duration_s = 30.0

    hand_init_pos: list = [0.0, 0.0, 0.015]
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 1.83]
    hand_init_orn_noise: list = [0.0, 0.0, 0.26]

    unidirectional_rot: bool = True

    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 120.0
    fixed_asset_init_orn_range_deg: float = 30.0

    held_asset_pos_noise: list = [0.0, 0.003, 0.003]
    held_asset_rot_init: float = -90.0

    ee_success_yaw = 0.0
    keypoint_coef_baseline: list = [100, 2]
    keypoint_coef_coarse: list = [500, 2]
    keypoint_coef_fine: list = [1500, 0]
    success_threshold: float = 0.375
    engage_threshold: float = 0.5
    keypoint_scale: float = 0.05

    fixed_asset: ArticulationCfg = _make_fixed_asset_cfg(
        "/World/envs/env_.*/FixedAsset", f"{ASSET_DIR}/factory_bolt_m16.usd", 0.05
    )
    held_asset: ArticulationCfg = _make_asset_cfg(
        "/World/envs/env_.*/HeldAsset", f"{ASSET_DIR}/factory_nut_m16.usd", 0.03, disable_gravity=True
    )


# ---------------------------------------------------------------------------
# Peg-insert asset pair registry
# ---------------------------------------------------------------------------
# Each entry defines a (held, fixed) pair.  During training the environment
# randomly selects a pair per environment at reset, so the policy generalises
# across different peg/hole geometries.  To add a new part (B003, B004 …)
# simply append a dict to ASSET_PAIRS below.
# ---------------------------------------------------------------------------

_LOCAL_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "stl")


def _asset_path(name: str) -> str:
    """Return absolute path to a local asset file (USDA, STL, etc.)."""
    return os.path.join(_LOCAL_ASSET_DIR, name)


ASSET_PAIRS: list[dict] = [
    # --- Factory default 8 mm peg + hole ---
    {
        "held_cfg": Peg8mm(),
        "fixed_cfg": Hole8mm(),
        "held_art": _make_asset_cfg(
            "/World/envs/env_.*/HeldAsset", f"{ASSET_DIR}/factory_peg_8mm.usd", 0.019, disable_gravity=True
        ),
        "fixed_art": _make_fixed_asset_cfg(
            "/World/envs/env_.*/FixedAsset", f"{ASSET_DIR}/factory_hole_8mm.usd", 0.05
        ),
        "held_material": {"diffuse_color": (0.75, 0.75, 0.75), "metallic": 0.6, "roughness": 0.3},
        "fixed_material": {"diffuse_color": (0.55, 0.55, 0.6), "metallic": 0.5, "roughness": 0.4},
    },
    # --- pair1 (peg=stainless steel, housing=aluminum alloy) ---
    {
        "held_cfg": HeldAssetCfg(
            usd_path=_asset_path("pair1_peg.usd"),
            diameter=0.008,
            height=0.082,
            mass=0.032,  # stainless steel (~7700 kg/m3)
            friction=0.45,  # steel-aluminum dry friction
        ),
        "fixed_cfg": FixedAssetCfg(
            usd_path=_asset_path("pair1_fixed.usd"),
            diameter=0.008,
            height=0.0096,
            base_height=0.021,
            friction=0.45,  # aluminum side of steel-aluminum contact
        ),
        "held_art": _make_asset_cfg(
            "/World/envs/env_.*/HeldAsset", _asset_path("pair1_peg.usd"), 0.032, disable_gravity=True
        ),
        "fixed_art": _make_fixed_asset_cfg(
            "/World/envs/env_.*/FixedAsset", _asset_path("pair1_fixed.usd"), 0.101
        ),
        "held_material": {
            "diffuse_color": (0.78, 0.78, 0.8),  # stainless steel
            "metallic": 1.0,
            "roughness": 0.25,
        },
        "fixed_material": {
            "diffuse_color": (0.62, 0.62, 0.65),  # aluminum alloy
            "metallic": 0.8,
            "roughness": 0.35,
        },
    },
    # --- pair2 (peg=steel shaft, fixed-asset=housing+ring magnet assembly) ---
    {
        "held_cfg": HeldAssetCfg(
            usd_path=_asset_path("pair2_peg.usd"),
            diameter=0.0061,
            height=0.066,
            mass=0.014,  # steel shaft (~7.5 g/cm³ from 1862 mm³)
            friction=0.5,
        ),
        "fixed_cfg": FixedAssetCfg(
            usd_path=_asset_path("pair2_fixed.usd"),
            diameter=0.0062,
            height=0.025,
            base_height=0.010,
            friction=0.5,
        ),
        "held_art": _make_asset_cfg(
            "/World/envs/env_.*/HeldAsset", _asset_path("pair2_peg.usd"), 0.014, disable_gravity=True
        ),
        "fixed_art": _make_fixed_asset_cfg(
            "/World/envs/env_.*/FixedAsset", _asset_path("pair2_fixed.usd"), 0.100
        ),
        "held_material": {
            "diffuse_color": (0.75, 0.75, 0.78),  # steel shaft
            "metallic": 1.0,
            "roughness": 0.3,
        },
        "fixed_material": {
            "diffuse_color": (0.5, 0.5, 0.55),  # metal housing
            "metallic": 0.7,
            "roughness": 0.4,
        },
    },
]


