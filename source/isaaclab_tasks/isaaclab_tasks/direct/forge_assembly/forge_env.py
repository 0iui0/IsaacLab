# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forge Assembly: robot-agnostic direct RL environment.

Supports Franka (gripper-based) and UR10/CR5 (fixed-peg) robots through
the RobotProfile abstraction.
"""

import math
import os

import carb
import numpy as np
import omni.usd
import torch

import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg, spawn_preview_surface
from isaaclab.sim.utils import bind_visual_material
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

_DIR = os.path.dirname(os.path.abspath(__file__))
from isaaclab.utils.math import axis_angle_from_quat, skew_symmetric_matrix

from . import forge_control, forge_utils
from .forge_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, ForgeEnvCfg
from .forge_tasks_cfg import ASSET_PAIRS as _ALL_ASSET_PAIRS
from .robot_profiles import RobotProfile


class ForgeEnv(DirectRLEnv):
    cfg: ForgeEnvCfg

    def __init__(self, cfg: ForgeEnvCfg, render_mode: str | None = None, **kwargs):
        # Resolve profile-driven dimensions before parent init.
        self.profile: RobotProfile = cfg.robot_profile
        self.num_arm_joints = self.profile.num_arm_joints
        self.arm_slice = self.profile.arm_joint_ids
        self.gripper_slice = self.profile.gripper_joint_ids
        self.cfg_task = cfg.task

        # Resolve asset pair subset from config.
        if cfg.asset_pair_indices is not None:
            ASSET_PAIRS = [_ALL_ASSET_PAIRS[i] for i in cfg.asset_pair_indices]
        else:
            ASSET_PAIRS = list(_ALL_ASSET_PAIRS)
        self._asset_pairs = ASSET_PAIRS

        super().__init__(cfg, render_mode, **kwargs)

        forge_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()

        # Success prediction (FORGE-specific).
        self.success_pred_scale = 0.0
        self.first_pred_success_tx = {}
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh] = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        # Debug: step counter for periodic exploration stats
        self._debug_step_count = 0

        # Force sensor - body index will be set in _init_tensors after scene setup
        self.force_sensor_body_idx = None
        self.force_sensor_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_sensor_world_smooth = torch.zeros((self.num_envs, 6), device=self.device)

        # Peg tip / hole top positions for direct success check (fixed-peg robots)
        self.peg_tip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hole_top_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Multi-asset pair: per-env index into ASSET_PAIRS
        self.asset_pair_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Dynamics randomization defaults.
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_dead_zone = torch.tensor(self.cfg.ctrl.default_dead_zone, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()

    # -----------------------------------------------------------------------
    # Scene setup
    # -----------------------------------------------------------------------

    @staticmethod
    def _clone_art_cfg(cfg: ArticulationCfg, prim_path: str) -> ArticulationCfg:
        """Return a deep copy of ArticulationCfg with prim_path set."""
        import copy
        new_cfg = copy.deepcopy(cfg)
        new_cfg.prim_path = prim_path
        return new_cfg

    def _apply_asset_materials(self):
        """Apply visual materials to custom asset meshes using Isaac Lab's material system.

        Called after clone_environments so all prims exist on the live stage.
        Skips pair index 0 (factory USD assets — they carry material from the USDA file).
        """
        from isaaclab.sim.utils import get_current_stage

        if self._num_asset_pairs < 2:
            return

        stage = get_current_stage()
        for i, pair in enumerate(self._asset_pairs):
            if not pair.get("fixed_material"):
                continue
            suffix = f"_{i}" if i > 0 else ""
            self._bind_pair_material(stage, self.num_envs, pair, "FixedAsset", suffix, "fixed_material")
            if self.profile.grasp_type == "gripper":
                self._bind_pair_material(stage, self.num_envs, pair, "HeldAsset", suffix, "held_material")

    @staticmethod
    def _bind_pair_material(stage, num_envs: int, pair: dict, art_prefix: str, suffix: str, material_key: str):
        """Create and bind a PreviewSurface material to mesh prims for one asset pair."""
        mat = pair.get(material_key)
        if not mat:
            return
        mat_path = f"/World/Materials/{art_prefix}{suffix}"
        spawn_preview_surface(mat_path, PreviewSurfaceCfg(
            diffuse_color=mat.get("diffuse_color", (0.6, 0.4, 0.2)),
            metallic=mat.get("metallic", 0.0),
            roughness=mat.get("roughness", 0.5),
        ))
        if not stage.GetPrimAtPath(mat_path):
            return
        # Bind material to the articulation root using Kit command (propagates to all children).
        for env_idx in range(num_envs):
            art_path = f"/World/envs/env_{env_idx}/{art_prefix}{suffix}"
            if stage.GetPrimAtPath(art_path):
                bind_visual_material(art_path, mat_path)

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        # Visual-only props from robot profile (pre-converted USD files).
        if self.profile.visual_assets:
            for name, (asset_path, pos) in self.profile.visual_assets.items():
                if not os.path.isfile(asset_path):
                    continue
                va_cfg = sim_utils.UsdFileCfg(usd_path=asset_path)
                va_cfg.func(f"/World/envs/env_.*/{name}", va_cfg, translation=pos)

        # Robot and assets - use copy_from_source=True like factory_env.py
        # No need for manual env_0 spawn when using copy_from_source=True
        self._robot = Articulation(self.profile.robot)

        # Multi-asset support: spawn all registered asset pairs for peg_insert.
        # For non-peg_insert tasks (gear_mesh, nut_thread), use the single task config.
        self._num_asset_pairs = len(self._asset_pairs) if self.cfg_task.name == "peg_insert" else 1
        self._fixed_assets = []  # list of Articulation, one per pair
        self._held_assets = []   # list of Articulation or None, one per pair

        if self._num_asset_pairs > 1:
            # Spawn all asset pairs. Pair 0 uses standard prim paths (no suffix)
            # so event configs referencing "held_asset"/"fixed_asset" still work.
            for i, pair in enumerate(self._asset_pairs):
                suffix = f"_{i}" if i > 0 else ""
                fixed_cfg = self._clone_art_cfg(
                    pair["fixed_art"],
                    f"/World/envs/env_.*/FixedAsset{suffix}",
                )
                fixed_cfg.init_state.pos = self.cfg_task.fixed_asset.init_state.pos
                self._fixed_assets.append(Articulation(fixed_cfg))
                if self.profile.grasp_type == "gripper":
                    held_cfg = self._clone_art_cfg(
                        pair["held_art"],
                        f"/World/envs/env_.*/HeldAsset{suffix}",
                    )
                    self._held_assets.append(Articulation(held_cfg))
                else:
                    self._held_assets.append(None)
        else:
            # Single-asset path: use first custom pair (index 1)
            pair = self._asset_pairs[1] if len(self._asset_pairs) > 1 else self._asset_pairs[0]
            fixed_cfg = self._clone_art_cfg(
                pair["fixed_art"],
                "/World/envs/env_.*/FixedAsset",
            )
            fixed_cfg.init_state.pos = self.cfg_task.fixed_asset.init_state.pos
            self._fixed_asset = Articulation(fixed_cfg)
            self._fixed_assets.append(self._fixed_asset)

            if self.profile.grasp_type == "gripper":
                held_cfg = self._clone_art_cfg(
                    pair["held_art"],
                    "/World/envs/env_.*/HeldAsset",
                )
                self._held_asset = Articulation(held_cfg)
                self._held_assets.append(self._held_asset)
            else:
                self._held_asset = None
                self._held_assets.append(None)

        # Left arm (mirrored visual-only articulation, kinematic)
        if self.profile.left_arm is not None:
            # Auto-convert URDF to USD on first run (output alongside URDF for mesh resolution).
            usd_path = self.profile.left_arm.spawn.usd_path
            if not os.path.isfile(usd_path):
                urdf_dir = os.path.join(_DIR, "assets/urdf/marvin_m6_left")
                urdf_file = os.path.join(urdf_dir, "marvin_m6_left.urdf")
                if os.path.isfile(urdf_file):
                    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

                    uc_cfg = UrdfConverterCfg(
                        asset_path=urdf_file,
                        usd_dir=urdf_dir,
                        usd_file_name="marvin_m6_left.usd",
                        fix_base=True,
                        merge_fixed_joints=True,
                        make_instanceable=False,
                        joint_drive=UrdfConverterCfg.JointDriveCfg(
                            drive_type="force",
                            target_type="position",
                            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
                        ),
                    )
                    UrdfConverter(uc_cfg)
            self._left_arm = Articulation(self.profile.left_arm)
        else:
            self._left_arm = None

        # For fixed-peg robots (no held_asset articulation), spawn peg collision on EE link
        if self.profile.grasp_type == "fixed_peg" and self.profile.peg_offset_from_ee is not None:
            self._spawn_peg_collision_on_ee()

        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        self.scene.clone_environments(copy_from_source=True)
        self._apply_asset_materials()
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot

        # Register all asset pair articulations
        # Always register "held_asset" and "fixed_asset" aliases pointing to
        # pair 0 so that EventCfg SceneEntityCfg references resolve correctly.
        if self._num_asset_pairs > 1:
            for i, (fa, ha) in enumerate(zip(self._fixed_assets, self._held_assets)):
                self.scene.articulations[f"fixed_asset_{i}"] = fa
                if ha is not None:
                    self.scene.articulations[f"held_asset_{i}"] = ha
            # Aliases for event config compatibility
            self.scene.articulations["fixed_asset"] = self._fixed_assets[0]
            if self._held_assets[0] is not None:
                self.scene.articulations["held_asset"] = self._held_assets[0]
        else:
            self.scene.articulations["fixed_asset"] = self._fixed_asset
            if self._held_asset is not None:
                self.scene.articulations["held_asset"] = self._held_asset

        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        # Contact sensor on robot EE for force data — no built-in debug_vis.
        _ENV_RE = "/World/envs/env_.*"
        contact_cfg = ContactSensorCfg(
            prim_path=f"{_ENV_RE}/Robot/.*",
            debug_vis=False,
        )
        self._contact_sensor = ContactSensor(contact_cfg)
        self.scene.sensors["contact_ee"] = self._contact_sensor

        # XYZ force component markers using CylinderCfg (proven to work).
        # Contact force: Red=X, Green=Y, Blue=Z
        # F/T sensor: Orange=X, Cyan=Y, Purple=Z
        self._force_viz = {}
        self._ft_viz = {}
        colors_contact = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
        colors_ft = {"x": (1.0, 0.5, 0.0), "y": (0.0, 1.0, 0.5), "z": (0.5, 0.0, 1.0)}
        for axis in ("x", "y", "z"):
            self._force_viz[axis] = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=f"/Visuals/CF_{axis.upper()}",
                    markers={
                        "shaft": sim_utils.CylinderCfg(
                            radius=0.004, height=1.0, axis="X",
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colors_contact[axis]),
                        ),
                    },
                )
            )
            self._ft_viz[axis] = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=f"/Visuals/FT_{axis.upper()}",
                    markers={
                        "shaft": sim_utils.CylinderCfg(
                            radius=0.003, height=1.0, axis="X",
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colors_ft[axis]),
                        ),
                    },
                )
            )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Patch collision meshes with SDF approximation at runtime.
        # USDA-authored physics:approximation may not be picked up by PhysX;
        # setting it via the pxr API ensures proper SDF collision for through-holes.
        self._patch_sdf_collision()

    def _patch_sdf_collision(self):
        """Force SDF approximation on fixed-asset collision meshes via runtime pxr API.

        USDA-authored physics:approximation is sometimes not recognized by PhysX.
        This re-applies the schema and attribute through the pxr API to ensure
        SDF collision works for through-holes on dynamic bodies.
        """
        import omni.usd
        from pxr import UsdPhysics, PhysxSchema

        stage = omni.usd.get_context().get_stage()
        patched = 0
        for prim in stage.Traverse():
            # Only patch meshes under FixedAsset prims
            path_str = str(prim.GetPath())
            if "FixedAsset" not in path_str:
                continue
            if prim.GetTypeName() != "Mesh":
                continue
            # Apply/re-apply collision APIs and set SDF
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            mesh_api.GetApproximationAttr().Set("sdf")
            # Ensure collision is enabled
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(True)
            # PhysX collision offsets
            if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_col = PhysxSchema.PhysxCollisionAPI(prim)
            physx_col.GetContactOffsetAttr().Set(0.005)
            physx_col.GetRestOffsetAttr().Set(0.0)
            patched += 1
        carb.log_info(f"[Forge] Patched {patched} collision meshes with SDF approximation")

    def _spawn_peg_collision_on_ee(self):
        """Spawn peg as a collision-only prim child of the EE link.

        IMPORTANT: Must be called BEFORE Articulation(robot) is created so that
        the collision shape is part of the EE link's rigid body when PhysX
        initializes the articulation. This ensures contact forces on the peg
        are reported through get_link_incoming_joint_force() on the ee_body_idx.

        Note: Peg offset is automatically computed as peg_height/2 along the
        approach axis to ensure peg base sits on the EE flange surface.
        """
        peg_radius = self.profile.peg_radius
        peg_height = self.profile.peg_height
        peg_mat = self.profile.peg_material or (1.0, 1.0, 0.0)

        # Compute peg offset dynamically: peg center offset by peg_height/2
        # so that peg base sits on EE flange surface.
        # peg_offset_from_ee defines the approach axis direction (e.g., [1,0,0] for X-axis)
        base_offset = self.profile.peg_offset_from_ee or [1.0, 0.0, 0.0]
        peg_offset = [base_offset[0] * peg_height / 2,
                      base_offset[1] * peg_height / 2,
                      base_offset[2] * peg_height / 2]

        peg_prim_path = "/World/envs/env_0/Robot/.*{}/peg".format(self.profile.ee_body_name)
        peg_cfg = sim_utils.CylinderCfg(
            radius=peg_radius,
            height=peg_height,
            axis="X",  # Peg along ee_link X-axis (perpendicular to flange)
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=peg_mat[0],
                dynamic_friction=peg_mat[1],
                restitution=peg_mat[2],
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        )
        peg_cfg.func(
            peg_prim_path,
            peg_cfg,
            translation=(peg_offset[0], peg_offset[1], peg_offset[2]),
            orientation=(1.0, 0.0, 0.0, 0.0),  # Identity - peg aligned with ee_link frame
        )

    # -----------------------------------------------------------------------
    # Multi-asset helpers
    # -----------------------------------------------------------------------

    def _gather_fixed_pos_w(self) -> torch.Tensor:
        """Gather fixed_asset root_pos_w from the correct pair per env."""
        if self._num_asset_pairs <= 1:
            return self._fixed_assets[0].data.root_pos_w
        out = torch.zeros((self.num_envs, 3), device=self.device)
        for i, art in enumerate(self._fixed_assets):
            mask = self.asset_pair_idx == i
            if mask.any():
                out[mask] = art.data.root_pos_w[mask]
        return out

    def _gather_fixed_quat_w(self) -> torch.Tensor:
        if self._num_asset_pairs <= 1:
            return self._fixed_assets[0].data.root_quat_w
        out = torch.zeros((self.num_envs, 4), device=self.device)
        for i, art in enumerate(self._fixed_assets):
            mask = self.asset_pair_idx == i
            if mask.any():
                out[mask] = art.data.root_quat_w[mask]
        return out

    def _gather_held_pos_w(self) -> torch.Tensor:
        if self._num_asset_pairs <= 1:
            return self._held_assets[0].data.root_pos_w
        out = torch.zeros((self.num_envs, 3), device=self.device)
        for i, art in enumerate(self._held_assets):
            if art is None:
                continue
            mask = self.asset_pair_idx == i
            if mask.any():
                out[mask] = art.data.root_pos_w[mask]
        return out

    def _gather_held_quat_w(self) -> torch.Tensor:
        if self._num_asset_pairs <= 1:
            return self._held_assets[0].data.root_quat_w
        out = torch.zeros((self.num_envs, 4), device=self.device)
        for i, art in enumerate(self._held_assets):
            if art is None:
                continue
            mask = self.asset_pair_idx == i
            if mask.any():
                out[mask] = art.data.root_quat_w[mask]
        return out

    def _get_active_fixed_cfg(self) -> "FixedAssetCfg":
        """Return fixed_asset_cfg for the most common pair (used for reward dims)."""
        if self._num_asset_pairs <= 1:
            idx = min(1, len(self._asset_pairs) - 1)
            return self._asset_pairs[idx]["fixed_cfg"]
        pair_idx = self.asset_pair_idx[0].item()
        return self._asset_pairs[pair_idx]["fixed_cfg"]

    def _get_active_held_cfg(self) -> "HeldAssetCfg":
        if self._num_asset_pairs <= 1:
            idx = min(1, len(self._asset_pairs) - 1)
            return self._asset_pairs[idx]["held_cfg"]
        pair_idx = self.asset_pair_idx[0].item()
        return self._asset_pairs[pair_idx]["held_cfg"]

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        if self._held_assets and any(h is not None for h in self._held_assets):
            for i, ha in enumerate(self._held_assets):
                if ha is not None:
                    pair_held_cfg = self._asset_pairs[i]["held_cfg"]
                    forge_utils.set_friction(ha, pair_held_cfg.friction, self.scene.num_envs)
        for i, fa in enumerate(self._fixed_assets):
            pair_fixed_cfg = self._asset_pairs[i]["fixed_cfg"]
            forge_utils.set_friction(fa, pair_fixed_cfg.friction, self.scene.num_envs)
        forge_utils.set_friction(self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

    def _init_tensors(self):
        """Initialize tensors once."""
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ema_factor = self.cfg.ctrl.ema_factor
        self.dead_zone_thresholds = None

        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Body indices.
        self.ee_body_idx = self._robot.body_names.index(self.profile.ee_body_name)
        if self.profile.has_gripper and self.profile.left_finger_body_name is not None:
            self.left_finger_body_idx = self._robot.body_names.index(self.profile.left_finger_body_name)
            self.right_finger_body_idx = self._robot.body_names.index(self.profile.right_finger_body_name)

        # Force sensor body index
        if self.profile.grasp_type == "fixed_peg":
            self.force_sensor_body_idx = self.ee_body_idx
        elif self.profile.force_sensor_body_name is not None:
            if self.profile.force_sensor_body_name in self._robot.body_names:
                self.force_sensor_body_idx = self._robot.body_names.index(self.profile.force_sensor_body_name)
            else:
                # Fallback to EE body when named sensor link is absent
                self.force_sensor_body_idx = self.ee_body_idx
        else:
            self.force_sensor_body_idx = None

        # Finite-differencing.
        self.last_update_timestamp = 0.0
        self.prev_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_ee_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.prev_joint_pos = torch.zeros((self.num_envs, self.num_arm_joints), device=self.device)

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    # -----------------------------------------------------------------------
    # Intermediate value computation
    # -----------------------------------------------------------------------

    def _compute_intermediate_values(self, dt, update_joint_pos=True):
        """Get values computed from raw tensors. This includes adding noise."""
        # Update contact sensor data (force arrows drawn at end of this method).
        self._contact_sensor.update(dt, force_recompute=True)

        self.fixed_pos = self._gather_fixed_pos_w() - self.scene.env_origins
        self.fixed_quat = self._gather_fixed_quat_w()

        self.ee_pos = self._robot.data.body_pos_w[:, self.ee_body_idx] - self.scene.env_origins
        self.ee_quat = self._robot.data.body_quat_w[:, self.ee_body_idx]
        self.ee_linvel = self._robot.data.body_lin_vel_w[:, self.ee_body_idx]
        self.ee_angvel = self._robot.data.body_ang_vel_w[:, self.ee_body_idx]

        # Compute fingertip pose from ee_body pose using ee_to_fingertip_offset.
        # For Franka: ee_body = panda_fingertip_centered, offset = [0,0,0] (no change)
        # For Marvin: ee_body = Link7_R, offset = [0, -0.129, 0] in Link7_R frame
        #
        # After this block, self.ee_pos and self.ee_quat are the CONTROL TARGET pose,
        # which should be the fingertip center for gripper robots.
        offset = torch.tensor(self.profile.ee_to_fingertip_offset, device=self.device)
        if torch.norm(offset) > 1e-6:
            # Transform offset from ee_body local frame to world frame
            offset_world = torch_utils.quat_apply(self.ee_quat, offset.unsqueeze(0).expand(self.num_envs, -1))
            fingertip_pos = self.ee_pos + offset_world
            # Apply robot-specific EE frame correction (e.g., Marvin Link7_R→gripper Z-down)
            if self.profile.ee_frame_correction is not None:
                frame_rot = torch.tensor(self.profile.ee_frame_correction, device=self.device)
                fingertip_quat = torch_utils.quat_mul(self.ee_quat, frame_rot.unsqueeze(0).expand(self.num_envs, -1))
            else:
                fingertip_quat = self.ee_quat
            # Store original ee_body pose for reference (needed for Jacobian lookup)
            self.ee_body_pos = self.ee_pos.clone()
            self.ee_body_quat = self.ee_quat.clone()
            # Replace ee_pos/ee_quat with fingertip pose for control/observations
            self.ee_pos = fingertip_pos
            self.ee_quat = fingertip_quat
        # For Franka: ee_body IS the fingertip center, no transformation needed
        # (ee_pos/ee_quat remain unchanged)

        if self._held_assets and any(h is not None for h in self._held_assets):
            self.held_pos = self._gather_held_pos_w() - self.scene.env_origins
            self.held_quat = self._gather_held_quat_w()
        else:
            # Fixed-peg: compute peg base position from EE pose.
            # Peg base sits at EE link origin (peg_offset_from_ee defines the direction).
            # This matches Franka's held_pos which is the peg root/base position.
            ee_pos = self._robot.data.body_pos_w[:, self.ee_body_idx] - self.scene.env_origins
            ee_quat = self._robot.data.body_quat_w[:, self.ee_body_idx]
            self.held_pos = ee_pos  # Peg base = EE position
            self.held_quat = ee_quat

            # Peg tip = held_pos + peg_height along peg direction
            peg_direction = torch.tensor(self.profile.peg_offset_from_ee, device=self.device)
            peg_tip_offset = peg_direction * self.profile.peg_height
            peg_tip_offset = peg_tip_offset.unsqueeze(0).expand(self.num_envs, -1)
            identity_quat = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(self.num_envs, 4)
            )
            _, self.peg_tip_pos = torch_utils.tf_combine(
                self.held_quat, self.held_pos, identity_quat, peg_tip_offset
            )

            # Hole top = fixed_pos + height along Z in hole frame
            hole_top_local = torch.zeros((self.num_envs, 3), device=self.device)
            if self._num_asset_pairs > 1:
                for i, pair in enumerate(self._asset_pairs):
                    mask = self.asset_pair_idx == i
                    if mask.any():
                        hole_top_local[mask, 2] = pair["fixed_cfg"].height
            else:
                hole_top_local[:, 2] = self.cfg_task.fixed_asset_cfg.height
            _, self.hole_top_pos = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, identity_quat, hole_top_local
            )

        jacobians = self._robot.root_physx_view.get_jacobians()
        mass_matrices = self._robot.root_physx_view.get_generalized_mass_matrices()

        # Compute EE Jacobian.
        if self.profile.has_gripper:
            left_jac = jacobians[:, self.left_finger_body_idx - 1, 0:6, :]
            right_jac = jacobians[:, self.right_finger_body_idx - 1, 0:6, :]
            self.ee_jacobian = (left_jac[:, :, self.arm_slice] + right_jac[:, :, self.arm_slice]) * 0.5
        else:
            self.ee_jacobian = jacobians[:, self.ee_body_idx - 1, 0:6, self.arm_slice]

        # Correct Jacobian for ee_to_fingertip_offset.
        # The body Jacobian maps joint velocities to body-origin velocities, but
        # the EE pose (fingertip center) is offset by r from the body origin.
        # Standard correction: J_tip_linear = J_body_linear + skew(r_world) @ J_body_angular
        offset = torch.tensor(self.profile.ee_to_fingertip_offset, device=self.device)
        if torch.norm(offset) > 1e-6:
            # Use ee_body_quat (stored before fingertip transform) to get offset in world frame
            ee_body_quat = self.ee_body_quat if hasattr(self, "ee_body_quat") else self.ee_quat
            r_world = torch_utils.quat_apply(ee_body_quat, offset.unsqueeze(0).expand(self.num_envs, -1))
            skew_r = skew_symmetric_matrix(r_world)  # (N, 3, 3)
            # ee_jacobian shape: (N, 6, num_arm_joints)
            # Rows 0-2: linear, Rows 3-5: angular
            J_lin = self.ee_jacobian[:, 0:3, :]  # (N, 3, num_arm_joints)
            J_ang = self.ee_jacobian[:, 3:6, :]  # (N, 3, num_arm_joints)
            J_lin_corrected = J_lin + torch.bmm(skew_r, J_ang)  # (N, 3, num_arm_joints)
            self.ee_jacobian[:, 0:3, :] = J_lin_corrected

        self.arm_mass_matrix = mass_matrices[:, self.arm_slice, self.arm_slice]
        if update_joint_pos:
            self.joint_pos = self._robot.data.joint_pos.clone()
            self.joint_vel = self._robot.data.joint_vel.clone()

        # Finite-differencing for velocities.
        self.ee_linvel_fd = (self.ee_pos - self.prev_ee_pos) / dt
        self.prev_ee_pos = self.ee_pos.clone()

        rot_diff_quat = torch_utils.quat_mul(self.ee_quat, torch_utils.quat_conjugate(self.prev_ee_quat))
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_ee_quat = self.ee_quat.clone()

        joint_diff = self.joint_pos[:, self.arm_slice] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, self.arm_slice].clone()

        self.last_update_timestamp = self._robot._data._sim_timestamp

        # --- Debug: periodic exploration stats for 6-DOF robots ---
        if self.profile.num_arm_joints == 6:
            self._debug_step_count += 1
            if self._debug_step_count % 150 == 0:
                hole_pos = self.fixed_pos_obs_frame  # hole tip position
                ee_to_hole = self.ee_pos - hole_pos
                xy_dist = torch.norm(ee_to_hole[:, 0:2], dim=1)
                z_dist = ee_to_hole[:, 2]
                total_dist = torch.norm(ee_to_hole, dim=1)
                q1_vals = self.joint_pos[:, 0]
                # How many envs within 2cm XY of hole center
                near_hole = (xy_dist < 0.02).sum().item()
                print(
                    f"[Step {self._debug_step_count:>5d}] "
                    f"EE-hole XY: mean={xy_dist.mean():.4f} max={xy_dist.max():.4f}m | "
                    f"Z: mean={z_dist.mean():+.4f} | "
                    f"Total: mean={total_dist.mean():.4f} | "
                    f"Near(<2cm): {near_hole}/{self.num_envs} | "
                    f"|q1|: mean={q1_vals.abs().mean():.3f} max={q1_vals.abs().max():.3f} rad",
                    flush=True,
                )

        # --- FORGE-specific: noise + force sensing ---
        self._compute_forge_noise(dt)

    def _compute_forge_noise(self, dt):
        """Add noise to observations for force sensing (FORGE-specific)."""
        pos_noise_level = self.cfg.obs_rand.fingertip_pos
        rot_noise_level_deg = self.cfg.obs_rand.fingertip_rot_deg

        fingertip_pos_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        fingertip_pos_noise = fingertip_pos_noise @ torch.diag(
            torch.tensor([pos_noise_level] * 3, dtype=torch.float32, device=self.device)
        )
        self.noisy_ee_pos = self.ee_pos + fingertip_pos_noise

        rot_noise_axis = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        rot_noise_axis /= torch.linalg.norm(rot_noise_axis, dim=1, keepdim=True)
        rot_noise_angle = torch.randn((self.num_envs,), dtype=torch.float32, device=self.device) * np.deg2rad(
            rot_noise_level_deg
        )
        self.noisy_ee_quat = torch_utils.quat_mul(
            self.ee_quat, torch_utils.quat_from_angle_axis(rot_noise_angle, rot_noise_axis)
        )
        self.noisy_ee_quat[:, [0, 3]] = 0.0
        self.noisy_ee_quat = self.noisy_ee_quat * self.flip_quats.unsqueeze(-1)

        # Repeat finite differencing with noisy values.
        self.ee_linvel_fd = (self.noisy_ee_pos - self.prev_ee_pos) / dt
        self.prev_ee_pos = self.noisy_ee_pos.clone()

        rot_diff_quat = torch_utils.quat_mul(self.noisy_ee_quat, torch_utils.quat_conjugate(self.prev_ee_quat))
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.ee_angvel_fd[:, 0:2] = 0.0
        self.prev_ee_quat = self.noisy_ee_quat.clone()

        # Force sensing.
        if self.force_sensor_body_idx is not None:
            self.force_sensor_world = self._robot.root_physx_view.get_link_incoming_joint_force()[
                :, self.force_sensor_body_idx
            ]
            alpha = self.cfg.ft_smoothing_factor
            self.force_sensor_world_smooth = alpha * self.force_sensor_world + (1 - alpha) * self.force_sensor_world_smooth

            self.force_sensor_smooth = torch.zeros_like(self.force_sensor_world)
            identity_quat = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            )
            self.force_sensor_smooth[:, :3], self.force_sensor_smooth[:, 3:6] = forge_utils.change_FT_frame(
                self.force_sensor_world_smooth[:, 0:3],
                self.force_sensor_world_smooth[:, 3:6],
                (identity_quat, torch.zeros((self.num_envs, 3), device=self.device)),
                (identity_quat, self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise),
            )

            force_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
            force_noise *= self.cfg.obs_rand.ft_force
            self.noisy_force = self.force_sensor_smooth[:, 0:3] + force_noise
            if self.cfg.use_ft_torque:
                torque_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
                torque_noise *= self.cfg.obs_rand.ft_force
                self.noisy_torque = self.force_sensor_smooth[:, 3:6] + torque_noise
        else:
            self.noisy_force = torch.zeros((self.num_envs, 3), device=self.device)
            if self.cfg.use_ft_torque:
                self.noisy_torque = torch.zeros((self.num_envs, 3), device=self.device)

        # Draw force visualization arrows after all sensor data is updated.
        if self.force_sensor_body_idx is not None or self._contact_sensor.is_initialized:
            self._draw_force_arrows()

    # -----------------------------------------------------------------------
    # Force-arrow debug visualization
    # -----------------------------------------------------------------------

    def _draw_force_arrows(self):
        """Draw XYZ force components as colored cylinders at EE position."""
        ee_pos = self._robot.data.body_pos_w[:, self.ee_body_idx]  # (N, 3)

        # ContactSensor net forces summed across all bodies per env
        contact_force = self._contact_sensor.data.net_forces_w.sum(dim=1)  # (N, 3)
        self._draw_xyz(self._force_viz, contact_force, ee_pos, scale=0.01)

        # F/T sensor force in world frame
        if self.force_sensor_body_idx is not None:
            ft_force = self.force_sensor_world_smooth[:, 0:3]  # (N, 3)
            self._draw_xyz(self._ft_viz, ft_force, ee_pos, scale=0.01)
        else:
            z = torch.zeros((self.num_envs, 3), device=self.device)
            q = torch.zeros((self.num_envs, 4), device=self.device)
            q[:, 0] = 1.0
            for viz in self._ft_viz.values():
                viz.visualize(translations=ee_pos, orientations=q, scales=z)

    def _draw_xyz(self, viz_dict, forces, positions, scale=0.01):
        """Draw each force component as a cylinder along its world axis.

        CylinderCfg(axis="X") creates a cylinder along X.
        We rotate: X→Y by 90° around Z, X→Z by -90° around Y.
        Negative component = flip 180° around perpendicular axis.
        """
        N = self.num_envs
        # Pre-compute base orientations for each axis (wxyz)
        # X-axis: identity [1,0,0,0]
        # Y-axis: rotate X→Y = 90° around Z = [cos45, 0,0, sin45]
        # Z-axis: rotate X→Z = -90° around Y = [cos45, 0,-sin45, 0]
        base_quats = {
            "x": torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(N, -1).clone(),
            "y": torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=self.device).expand(N, -1).clone(),
            "z": torch.tensor([0.7071, 0.0, -0.7071, 0.0], device=self.device).expand(N, -1).clone(),
        }
        # Flip quaternions for negative values (180° rotation)
        flip_quats = {
            "x": torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand(N, -1).clone(),  # 180° Z
            "y": torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).expand(N, -1).clone(),  # 180° Y
            "z": torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).expand(N, -1).clone(),  # 180° X
        }

        for i, axis in enumerate(("x", "y", "z")):
            comp = forces[:, i]  # (N,)
            abs_c = comp.abs()

            # Scale: X=length, Y=Z=1.0
            s = torch.zeros((N, 3), device=self.device)
            s[:, 0] = abs_c * scale
            s[:, 1] = 1.0
            s[:, 2] = 1.0
            s[abs_c < 0.05] = 0.0  # hide near-zero

            # Orientation: flip for negative
            neg = comp < 0
            quat = base_quats[axis].clone()
            quat[neg] = flip_quats[axis][neg]

            viz_dict[axis].visualize(translations=positions, orientations=quat, scales=s)

    # -----------------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------------

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        obs_dict, state_dict = self._get_obs_state_dict()

        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        obs_dict.update(
            {
                "fingertip_pos": self.noisy_ee_pos,
                "fingertip_pos_rel_fixed": self.noisy_ee_pos - noisy_fixed_pos,
                "fingertip_quat": self.noisy_ee_quat,
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "ft_force": self.noisy_force,
                "prev_actions": prev_actions,
            }
        )
        if self.cfg.use_ft_torque:
            obs_dict["ft_torque"] = self.noisy_torque

        state_dict.update(
            {
                "ema_factor": self.ema_factor,
                "ft_force": self.force_sensor_smooth[:, 0:3] if (self.force_sensor_body_idx is not None or self.profile.grasp_type == "fixed_peg") else torch.zeros((self.num_envs, 3), device=self.device),
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "prev_actions": prev_actions,
            }
        )
        if self.cfg.use_ft_torque:
            state_dict["ft_torque"] = self.force_sensor_smooth[:, 3:6] if (self.force_sensor_body_idx is not None or self.profile.grasp_type == "fixed_peg") else torch.zeros((self.num_envs, 3), device=self.device)

        obs_tensors = forge_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = forge_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _get_obs_state_dict(self):
        """Populate dictionaries for the policy and critic."""
        prev_actions = self.actions.clone()

        obs_dict = {
            "fingertip_pos": self.ee_pos,
            "fingertip_pos_rel_fixed": self.ee_pos - (self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise),
            "fingertip_quat": self.ee_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.ee_pos,
            "fingertip_pos_rel_fixed": self.ee_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": self.ee_quat,
            "ee_linvel": self.ee_linvel,
            "ee_angvel": self.ee_angvel,
            "joint_pos": self.joint_pos[:, self.arm_slice],
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
        }
        return obs_dict, state_dict

    # -----------------------------------------------------------------------
    # Action application
    # -----------------------------------------------------------------------

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.actions = self.ema_factor * action.clone().to(self.device) + (1 - self.ema_factor) * self.actions

    def _apply_action(self):
        """FORGE actions: targets relative to the fixed asset."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))

        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_ee_preclipped_pos = fixed_pos_action_frame + pos_actions

        rot_actions[:, 0:2] = 0.0
        yaw_min, yaw_max = self.cfg.ctrl.yaw_action_range
        rot_actions[:, 2] = np.deg2rad(yaw_min) + np.deg2rad(yaw_max - yaw_min) * (rot_actions[:, 2] + 1.0) / 2.0

        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )

        if self.profile.grasp_type == "fixed_peg":
            # UR10/CR5: peg along ee_link X-axis. To keep X pointing down,
            # apply pitch=90° then yaw. quat_mul(yaw, pitch_90) = R_yaw @ R_pitch.
            pitch_90_euler = torch.tensor([0.0, np.pi / 2, 0.0], device=self.device).repeat(self.num_envs, 1)
            quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
                roll=pitch_90_euler[:, 0], pitch=pitch_90_euler[:, 1], yaw=pitch_90_euler[:, 2]
            )
            ctrl_target_ee_preclipped_quat = torch_utils.quat_mul(bolt_frame_quat, quat_bolt_to_ee)
        else:
            # Franka: panda_hand Z-axis points down at zero pose.
            # quat_mul(roll_180, yaw) keeps Z pointing down with yaw rotation.
            rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
                roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
            )
            ctrl_target_ee_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        # Clip position targets.
        self.delta_pos = ctrl_target_ee_preclipped_pos - self.ee_pos
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_ee_pos = self.ee_pos + pos_error_clipped

        # Clip orientation targets.
        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.ee_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_ee_preclipped_quat)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        curr_yaw = forge_utils.wrap_yaw(curr_yaw)
        desired_yaw = forge_utils.wrap_yaw(desired_yaw)

        self.delta_yaw = desired_yaw - curr_yaw
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch)

        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll

        curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch)

        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
        desired_xyz[:, 1] = curr_pitch + clipped_pitch

        ctrl_target_ee_quat = torch_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_ee_pos=ctrl_target_ee_pos,
            ctrl_target_ee_quat=ctrl_target_ee_quat,
            ctrl_target_gripper_dof_pos=0.0 if self.profile.has_gripper else None,
        )

    # -----------------------------------------------------------------------
    # Control
    # -----------------------------------------------------------------------

    def generate_ctrl_signals(self, ctrl_target_ee_pos, ctrl_target_ee_quat, ctrl_target_gripper_dof_pos):
        """Compute joint torques via task-space impedance control."""
        self.joint_torque, self.applied_wrench = forge_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            ee_pos=self.ee_pos,
            ee_quat=self.ee_quat,
            ee_linvel=self.ee_linvel,
            ee_angvel=self.ee_angvel,
            jacobian=self.ee_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_ee_pos=ctrl_target_ee_pos,
            ctrl_target_ee_quat=ctrl_target_ee_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
            num_arm_joints=self.num_arm_joints,
            null_space_default_pos=self.profile.null_space_default_pos,
            dead_zone_thresholds=self.dead_zone_thresholds,
        )

        if self.gripper_slice is not None and ctrl_target_gripper_dof_pos is not None:
            self.ctrl_target_joint_pos[:, self.gripper_slice] = ctrl_target_gripper_dof_pos
            self.joint_torque[:, self.gripper_slice] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def close_gripper_in_place(self):
        """Keep gripper in current position as gripper closes."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)

        pos_actions = actions[:, 0:3] * self.pos_threshold
        ctrl_target_ee_pos = self.ee_pos + pos_actions

        rot_actions = actions[:, 3:6]
        angle = torch.norm(rot_actions, p=2, dim=-1)
        safe_angle = angle.clamp(min=1.0e-6)
        axis = rot_actions / safe_angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_ee_quat = torch_utils.quat_mul(rot_actions_quat, self.ee_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_ee_quat), dim=1)
        if self.profile.grasp_type == "fixed_peg":
            target_euler_xyz[:, 0] = 0.0
            target_euler_xyz[:, 1] = np.pi / 2  # pitch=90° keeps peg down
        else:
            target_euler_xyz[:, 0] = 3.14159  # Franka: roll=180° keeps peg down
            target_euler_xyz[:, 1] = 0.0

        ctrl_target_ee_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_ee_pos=ctrl_target_ee_pos,
            ctrl_target_ee_quat=ctrl_target_ee_quat,
            ctrl_target_gripper_dof_pos=0.0 if self.profile.has_gripper else None,
        )

    # -----------------------------------------------------------------------
    # Rewards
    # -----------------------------------------------------------------------

    def _get_rewards(self):
        """FORGE reward includes contact penalty and success prediction."""
        rew_buf = self._get_base_rewards()

        rew_dict, rew_scales = {}, {}
        pos_error = torch.norm(self.delta_pos, p=2, dim=-1) / self.cfg.ctrl.pos_action_threshold[0]
        rot_error = torch.abs(self.delta_yaw) / self.cfg.ctrl.rot_action_threshold[0]

        if self.force_sensor_body_idx is not None or self.profile.grasp_type == "fixed_peg":
            contact_force = torch.norm(self.force_sensor_smooth[:, 0:3], p=2, dim=-1, keepdim=False)
        else:
            contact_force = torch.zeros(self.num_envs, device=self.device)
        contact_penalty = torch.nn.functional.relu(contact_force - self.contact_penalty_thresholds)

        check_rot = self.cfg_task.name == "nut_thread"
        true_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )
        policy_success_pred = (self.actions[:, 6] + 1) / 2
        success_pred_error = (true_successes.float() - policy_success_pred).abs()

        if true_successes.float().mean() >= self.cfg_task.delay_until_ratio:
            self.success_pred_scale = 1.0

        rew_dict = {
            "action_penalty_asset": pos_error + rot_error,
            "contact_penalty": contact_penalty,
            "success_pred_error": success_pred_error,
        }
        rew_scales = {
            "action_penalty_asset": -self.cfg_task.action_penalty_asset_scale,
            "contact_penalty": -self.cfg_task.contact_penalty_scale,
            "success_pred_error": -self.success_pred_scale,
        }

        # Direct EE-to-target distance reward for fixed-peg robots.
        if self.profile.grasp_type == "fixed_peg" and self.cfg_task.ee_dist_reward_weight > 0:
            ee_dist = torch.norm(self.peg_tip_pos - self.hole_top_pos, p=2, dim=-1)
            ee_dist_reward = 1.0 / (1.0 + ee_dist * self.cfg_task.ee_dist_reward_scale)
            rew_dict["ee_distance"] = ee_dist_reward
            rew_scales["ee_distance"] = self.cfg_task.ee_dist_reward_weight

        # Z-descent reward: provides gradient for vertical descent independent of XY.
        # Only for fixed-peg robots. Rewards peg_tip being closer to hole_top in Z.
        # Scale: 100 means reward=0.5 when 1cm above, reward=0.91 when 1mm above.
        if self.profile.grasp_type == "fixed_peg" and self.cfg_task.z_descent_reward_weight > 0:
            z_gap = torch.clamp(self.peg_tip_pos[:, 2] - self.hole_top_pos[:, 2], min=0.0)  # positive = above hole
            z_descent_reward = 1.0 / (1.0 + z_gap * self.cfg_task.z_descent_reward_scale)
            rew_dict["z_descent"] = z_descent_reward
            rew_scales["z_descent"] = self.cfg_task.z_descent_reward_weight

        # Insertion depth reward: gradient for pushing peg INTO the hole.
        # Measures how far the peg tip has penetrated below the hole top surface.
        # Only active for fixed-peg robots. Provides the missing gradient that
        # Franka gets implicitly through keypoint rewards on the held asset.
        if self.profile.grasp_type == "fixed_peg" and self.cfg_task.insertion_reward_weight > 0:
            z_penetration = self.hole_top_pos[:, 2] - self.peg_tip_pos[:, 2]  # positive = inside hole
            hole_height = self.cfg_task.fixed_asset_cfg.height
            insertion_depth = torch.clamp(z_penetration / hole_height, 0.0, 1.0)
            rew_dict["insertion_depth"] = insertion_depth
            rew_scales["insertion_depth"] = self.cfg_task.insertion_reward_weight

        # q1 regularization: penalize base joint rotation from reset value.
        if self.profile.grasp_type == "fixed_peg" and self.cfg_task.q1_reg_weight > 0:
            q1_reset = torch.tensor(self.profile.reset_arm_joint_pos[0], device=self.device)
            q1_current = self.joint_pos[:, 0]
            q1_error = torch.abs(
                (q1_current - q1_reset + math.pi) % (2 * math.pi) - math.pi
            )
            rew_dict["q1_reg"] = -q1_error
            rew_scales["q1_reg"] = self.cfg_task.q1_reg_weight
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self._log_forge_metrics(rew_dict, policy_success_pred)
        return rew_buf

    def _get_base_rewards(self):
        """Compute base keypoint rewards."""
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        rew_dict, rew_scales = self._get_keypoint_rew_dict(curr_successes)

        rew_buf = torch.zeros_like(rew_dict["kp_coarse"])
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self.prev_actions = self.actions.clone()
        self._log_factory_metrics(rew_dict, curr_successes)
        return rew_buf

    def _get_keypoint_rew_dict(self, curr_successes):
        """Compute keypoint-based reward terms."""
        held_base_pos, held_base_quat = forge_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = forge_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )

        keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        keypoints_fixed = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        offsets = forge_utils.get_keypoint_offsets(
            self.cfg_task.num_keypoints, self.device, axis=self.profile.keypoint_axis
        )
        keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        for idx, keypoint_offset in enumerate(keypoint_offsets):
            keypoints_held[:, idx] = torch_utils.tf_combine(
                held_base_quat,
                held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
            keypoints_fixed[:, idx] = torch_utils.tf_combine(
                target_held_base_quat,
                target_held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
        keypoint_dist = torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)

        a0, b0 = self.cfg_task.keypoint_coef_baseline
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        a2, b2 = self.cfg_task.keypoint_coef_fine

        action_penalty_ee = torch.norm(self.actions, p=2)
        action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        curr_engaged = self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold, check_rot=False)

        rew_dict = {
            "kp_baseline": forge_utils.squashing_fn(keypoint_dist, a0, b0),
            "kp_coarse": forge_utils.squashing_fn(keypoint_dist, a1, b1),
            "kp_fine": forge_utils.squashing_fn(keypoint_dist, a2, b2),
            "action_penalty_ee": action_penalty_ee,
            "action_grad_penalty": action_grad_penalty,
            "curr_engaged": curr_engaged.float(),
            "curr_success": curr_successes.float(),
        }
        rew_scales = {
            "kp_baseline": 1.0,
            "kp_coarse": 1.0,
            "kp_fine": 1.0,
            "action_penalty_ee": -self.cfg_task.action_penalty_ee_scale,
            "action_grad_penalty": -self.cfg_task.action_grad_penalty_scale,
            "curr_engaged": 1.0,
            "curr_success": 1.0,
        }
        return rew_dict, rew_scales

    # -----------------------------------------------------------------------
    # Dones
    # -----------------------------------------------------------------------

    def _get_dones(self):
        """Check which environments are terminated."""
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    # -----------------------------------------------------------------------
    # Success check
    # -----------------------------------------------------------------------

    def _get_curr_successes(self, success_threshold, check_rot=False):
        """Get success mask at current timestep."""
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        if self.profile.grasp_type == "fixed_peg":
            # Direct check: peg tip vs hole top position
            xy_dist = torch.linalg.vector_norm(
                self.hole_top_pos[:, 0:2] - self.peg_tip_pos[:, 0:2], dim=1
            )
            is_centered = xy_dist < 0.0025  # 2.5mm XY alignment

            z_disp = self.peg_tip_pos[:, 2] - self.hole_top_pos[:, 2]
            insertion_threshold = self.cfg_task.fixed_asset_cfg.height * success_threshold
            is_inserted = z_disp < -insertion_threshold  # Peg tip below hole top

            curr_successes = torch.logical_and(is_centered, is_inserted)

            if check_rot:
                _, _, curr_yaw = torch_utils.get_euler_xyz(self.ee_quat)
                curr_yaw = forge_utils.wrap_yaw(curr_yaw)
                is_rotated = curr_yaw < self.cfg_task.ee_success_yaw
                curr_successes = torch.logical_and(curr_successes, is_rotated)

            return curr_successes

        # Gripper-based robots: use keypoint-based success check
        held_base_pos, held_base_quat = forge_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = forge_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )

        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))

        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name in ("peg_insert", "gear_mesh"):
            height_threshold = fixed_cfg.height * success_threshold
        elif self.cfg_task.name == "nut_thread":
            height_threshold = fixed_cfg.thread_pitch * success_threshold
        else:
            raise NotImplementedError("Task not implemented")

        is_close_or_below = torch.where(
            z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        if check_rot:
            _, _, curr_yaw = torch_utils.get_euler_xyz(self.ee_quat)
            curr_yaw = forge_utils.wrap_yaw(curr_yaw)
            is_rotated = curr_yaw < self.cfg_task.ee_success_yaw
            curr_successes = torch.logical_and(curr_successes, is_rotated)

        return curr_successes

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def _reset_idx(self, env_ids):
        """Perform full reset for specified environments."""
        super()._reset_idx(env_ids)

        # Randomize which asset pair each env uses
        if self._num_asset_pairs > 1:
            self.asset_pair_idx[env_ids] = torch.randint(
                0, self._num_asset_pairs, (len(env_ids),), device=self.device
            )

        self._set_assets_to_default_pose(env_ids)
        self._set_robot_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)

        # Reset left arm to default pose (visual-only)
        if self._left_arm is not None:
            left_default_pos = self._left_arm.data.default_joint_pos[env_ids].clone()
            left_default_vel = torch.zeros_like(left_default_pos)
            self._left_arm.write_joint_state_to_sim(left_default_pos, left_default_vel, env_ids=env_ids)
            self._left_arm.reset()

        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

        # FORGE-specific randomization.
        self._reset_forge_randomization(env_ids)

        # DEBUG: Print UR10 EE pose at reset (all envs)
        if self.profile.num_arm_joints == 6:
            import numpy as np
            import sys
            for i in env_ids[:8].cpu().numpy():  # Limit to first 8
                ee_pose = self._robot.data.body_state_w[i, self.ee_body_idx]
                ee_quat = ee_pose[3:7].cpu().numpy()
                w, x, y, z = ee_quat
                # EE X-axis (peg direction) via rotation matrix
                ee_x = np.array([1 - 2*y*y - 2*z*z, 2*x*y + 2*z*w, 2*x*z - 2*y*w])
                xdot = np.dot(ee_x, np.array([0.0, 0.0, -1.0]))
                tag = "OK" if xdot > 0.9 else "BAD"
                joints_str = ", ".join(f"{j:.3f}" for j in self._robot.data.joint_pos[i, :6].cpu().numpy())
                print(f"  [Reset] env {i}: Xdown={xdot:+.3f} [{tag}]  joints=[{joints_str}]", flush=True)
            sys.stdout.flush()

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0
        self.ep_success_times[env_ids] = 0
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh][env_ids] = 0

    def _reset_forge_randomization(self, env_ids):
        """FORGE-specific reset randomization."""
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        pos_actions = self.ee_pos - fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

        unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )
        ee_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.ee_quat)
        ee_yaw_bolt = torch_utils.get_euler_xyz(ee_quat_rel_bolt)[-1]
        ee_yaw_bolt = torch.where(ee_yaw_bolt > torch.pi / 2, ee_yaw_bolt - 2 * torch.pi, ee_yaw_bolt)
        ee_yaw_bolt = torch.where(ee_yaw_bolt < -torch.pi, ee_yaw_bolt + 2 * torch.pi, ee_yaw_bolt)

        yaw_action = (ee_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action
        self.actions[:, 6] = self.prev_actions[:, 6] = -1.0

        # EMA randomization.
        ema_rand = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device)
        ema_lower, ema_upper = self.cfg.ctrl.ema_factor_range
        self.ema_factor = ema_lower + ema_rand * (ema_upper - ema_lower)

        prop_gains = self.default_gains.clone()
        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()
        prop_gains = forge_utils.get_random_prop_gains(
            prop_gains, self.cfg.ctrl.task_prop_gains_noise_level, self.num_envs, self.device
        )
        self.pos_threshold = forge_utils.get_random_prop_gains(
            self.pos_threshold, self.cfg.ctrl.pos_threshold_noise_level, self.num_envs, self.device
        )
        self.rot_threshold = forge_utils.get_random_prop_gains(
            self.rot_threshold, self.cfg.ctrl.rot_threshold_noise_level, self.num_envs, self.device
        )
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = forge_utils.get_deriv_gains(prop_gains)

        contact_rand = torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
        contact_lower, contact_upper = self.cfg_task.contact_penalty_threshold_range
        self.contact_penalty_thresholds = contact_lower + contact_rand * (contact_upper - contact_lower)

        self.dead_zone_thresholds = (
            torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device) * self.default_dead_zone
        )

        self.force_sensor_world_smooth[:, :] = 0.0

        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        rand_flips = torch.rand(self.num_envs) > 0.5
        self.flip_quats[rand_flips] = -1.0

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        # Set held assets to default
        for i, ha in enumerate(self._held_assets):
            if ha is None:
                continue
            held_state = ha.data.default_root_state.clone()[env_ids]
            held_state[:, 0:3] += self.scene.env_origins[env_ids]
            held_state[:, 7:] = 0.0
            # Move non-active held assets far away so they don't interfere
            if self._num_asset_pairs > 1:
                non_active = self.asset_pair_idx[env_ids] != i
                held_state[non_active, 1] = -100.0
            ha.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
            ha.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
            ha.reset()

        # Set fixed assets to default
        for i, fa in enumerate(self._fixed_assets):
            fixed_state = fa.data.default_root_state.clone()[env_ids]
            fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
            fixed_state[:, 7:] = 0.0
            # Move non-active fixed assets far away
            if self._num_asset_pairs > 1:
                non_active = self.asset_pair_idx[env_ids] != i
                fixed_state[non_active, 1] = -100.0
            fa.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
            fa.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
            fa.reset()

    def _set_robot_to_default_pose(self, joints, env_ids):
        """Return robot to its default joint position."""
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self.arm_slice] = torch.tensor(joints, device=self.device)[None, :]
        if self.gripper_slice is not None:
            if self.profile.has_gripper:
                # Use max diameter across all pairs for a safe gripper opening
                max_diam = max(p["held_cfg"].diameter for p in self._asset_pairs) if self._num_asset_pairs > 1 else self.cfg_task.held_asset_cfg.diameter
                gripper_width = max_diam / 2 * 1.25
            else:
                gripper_width = 0.0
            joint_pos[:, self.gripper_slice] = gripper_width
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)
        self.step_sim_no_action()

    # -----------------------------------------------------------------------
    # IK and state randomization
    # -----------------------------------------------------------------------

    def set_pos_inverse_kinematics(self, ctrl_target_ee_pos, ctrl_target_ee_quat, env_ids):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0

        while ik_time < 0.25:
            pos_error, axis_angle_error = forge_control.get_pose_error(
                ee_pos=self.ee_pos[env_ids],
                ee_quat=self.ee_quat[env_ids],
                ctrl_target_ee_pos=ctrl_target_ee_pos[env_ids],
                ctrl_target_ee_quat=ctrl_target_ee_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            delta_dof_pos = forge_control.get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.ee_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, self.arm_slice] += delta_dof_pos[:, : self.num_arm_joints]
            # Clamp to IK joint limits to prevent joint reversal (e.g. elbow).
            if self.profile.ik_joint_limits is not None:
                lo = torch.tensor([l[0] for l in self.profile.ik_joint_limits], device=self.device)
                hi = torch.tensor([l[1] for l in self.profile.ik_joint_limits], device=self.device)
                self.joint_pos[env_ids, self.arm_slice] = torch.clamp(
                    self.joint_pos[env_ids, self.arm_slice], min=lo, max=hi
                )
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids])

            self.ctrl_target_joint_pos[env_ids, self.arm_slice] = self.joint_pos[env_ids, self.arm_slice]
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            self.step_sim_no_action(update_joint_pos=False)
            ik_time += self.physics_dt

        # Recompute error after final step_sim to get accurate final state.
        pos_error, axis_angle_error = forge_control.get_pose_error(
            ee_pos=self.ee_pos[env_ids],
            ee_quat=self.ee_quat[env_ids],
            ctrl_target_ee_pos=ctrl_target_ee_pos[env_ids],
            ctrl_target_ee_quat=ctrl_target_ee_quat[env_ids],
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )

        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self):
        """Get default relative pose between held asset and EE."""
        if self.profile.grasp_type == "fixed_peg":
            # Peg is rigidly attached; relative pose is the peg offset from EE link.
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            held_asset_relative_pos[:, 2] = self.profile.peg_offset_from_ee[2]
            held_asset_relative_quat = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            )
            return held_asset_relative_pos, held_asset_relative_quat

        if self.cfg_task.name == "peg_insert":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            if self._num_asset_pairs > 1:
                # Per-env height based on asset_pair_idx
                heights = torch.tensor([p["held_cfg"].height for p in self._asset_pairs], device=self.device)
                grip_offsets = torch.tensor([p["held_cfg"].grip_offset for p in self._asset_pairs], device=self.device)
                held_asset_relative_pos[:, 2] = heights[self.asset_pair_idx]
                held_asset_relative_pos[:, 2] -= self.profile.fingerpad_length
                held_asset_relative_pos[:, 2] += grip_offsets[self.asset_pair_idx]
            else:
                held_cfg = self._get_active_held_cfg()
                held_asset_relative_pos[:, 2] = held_cfg.height
                held_asset_relative_pos[:, 2] -= self.profile.fingerpad_length
                held_asset_relative_pos[:, 2] += held_cfg.grip_offset
        elif self.cfg_task.name == "gear_mesh":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            gear_base_offset = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset
            held_asset_relative_pos[:, 0] += gear_base_offset[0]
            held_asset_relative_pos[:, 2] += gear_base_offset[2]
            held_asset_relative_pos[:, 2] += self.cfg_task.held_asset_cfg.height / 2.0 * 1.1
        elif self.cfg_task.name == "nut_thread":
            held_asset_relative_pos = forge_utils.get_held_base_pos_local(
                self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
            )
        else:
            raise NotImplementedError("Task not implemented")

        held_asset_relative_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        if self.cfg_task.name == "nut_thread":
            initial_rot_deg = self.cfg_task.held_asset_rot_init
            rot_yaw_euler = torch.tensor([0.0, 0.0, initial_rot_deg * np.pi / 180.0], device=self.device).repeat(
                self.num_envs, 1
            )
            held_asset_relative_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_yaw_euler[:, 0], pitch=rot_yaw_euler[:, 1], yaw=rot_yaw_euler[:, 2]
            )

        return held_asset_relative_pos, held_asset_relative_quat

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform episode-level randomization."""
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1) Randomize fixed asset pose.
        fixed_state = self._fixed_assets[0].data.default_root_state.clone()[env_ids]
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]

        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        fixed_state[:, 7:] = 0.0
        # Write to all fixed asset articulations
        for i, fa in enumerate(self._fixed_assets):
            s = fa.data.default_root_state.clone()[env_ids]
            s[:, 0:3] = fixed_state[:, 0:3].clone()
            s[:, 3:7] = fixed_state[:, 3:7].clone()
            s[:, 7:] = 0.0
            # Move non-active pairs far away
            if self._num_asset_pairs > 1:
                non_active = self.asset_pair_idx[env_ids] != i
                s[non_active, 1] = -100.0
            fa.write_root_pose_to_sim(s[:, 0:7], env_ids=env_ids)
            fa.write_root_velocity_to_sim(s[:, 7:], env_ids=env_ids)
            fa.reset()

        # Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

        # Compute observation frame.
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        if self._num_asset_pairs > 1:
            # Use per-env height/base_height based on asset_pair_idx
            for i, pair in enumerate(self._asset_pairs):
                mask = self.asset_pair_idx == i
                if mask.any():
                    fixed_tip_pos_local[mask, 2] += pair["fixed_cfg"].height
                    fixed_tip_pos_local[mask, 2] += pair["fixed_cfg"].base_height
        else:
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
            fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # (2) Move EE to randomized location above fixed asset.
        bad_envs = env_ids.clone()
        ik_attempt = 0
        max_ik_attempts = 10
        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

        while bad_envs.shape[0] > 0 and ik_attempt < max_ik_attempts:
            n_bad = bad_envs.shape[0]

            above_fixed_pos = fixed_tip_pos.clone()
            above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            above_fixed_pos[bad_envs] += above_fixed_pos_rand

            hand_down_euler = (
                torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            )
            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            )

            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_ee_pos=above_fixed_pos,
                ctrl_target_ee_quat=hand_down_quat,
                env_ids=bad_envs,
            )
            pos_err_norm = torch.linalg.norm(pos_error, dim=1)
            angle_err_norm = torch.norm(aa_error, dim=1)
            pos_error = pos_err_norm > 5e-3
            angle_error = angle_err_norm > 5e-2
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            if bad_envs.shape[0] == 0:
                break

            self._set_robot_to_default_pose(
                joints=self.profile.reset_arm_joint_pos, env_ids=bad_envs
            )
            ik_attempt += 1

        # Re-write IK result to sim so the physics step doesn't drift the robot
        # (Marvin has stiffness=0/damping=0, so joints drift under gravity).
        self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

        # Add flanking gears for gear_mesh.
        if self.cfg_task.name == "gear_mesh" and self.cfg_task.add_flanking_gears:
            small_gear_state = self._small_gear_asset.data.default_root_state.clone()[env_ids]
            small_gear_state[:, 0:7] = fixed_state[:, 0:7]
            small_gear_state[:, 7:] = 0.0
            self._small_gear_asset.write_root_pose_to_sim(small_gear_state[:, 0:7], env_ids=env_ids)
            self._small_gear_asset.write_root_velocity_to_sim(small_gear_state[:, 7:], env_ids=env_ids)
            self._small_gear_asset.reset()

            large_gear_state = self._large_gear_asset.data.default_root_state.clone()[env_ids]
            large_gear_state[:, 0:7] = fixed_state[:, 0:7]
            large_gear_state[:, 7:] = 0.0
            self._large_gear_asset.write_root_pose_to_sim(large_gear_state[:, 0:7], env_ids=env_ids)
            self._large_gear_asset.write_root_velocity_to_sim(large_gear_state[:, 7:], env_ids=env_ids)
            self._large_gear_asset.reset()

        # (3) Position held asset relative to EE.
        if self.profile.grasp_type == "gripper":
            self._randomize_held_asset_in_gripper(env_ids)
        # For fixed_peg, no held_asset positioning needed — peg is part of the robot.

        # Set reset gains.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = forge_utils.get_deriv_gains(
            reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale
        )

        self.step_sim_no_action()

        # Close gripper (gripper-based robots only).
        if self.profile.has_gripper:
            grasp_time = 0.0
            while grasp_time < 0.25:
                self.ctrl_target_joint_pos[env_ids, self.gripper_slice] = 0.0
                self.close_gripper_in_place()
                self.step_sim_no_action()
                grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, self.arm_slice].clone()
        self.prev_ee_pos = self.ee_pos.clone()
        self.prev_ee_quat = self.ee_quat.clone()

        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = forge_utils.get_deriv_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    def _randomize_held_asset_in_gripper(self, env_ids):
        """Position the held asset in the gripper with randomization."""
        # NOTE: self.ee_pos and self.ee_quat are already transformed to fingertip pose
        # in _compute_intermediate_values (via ee_to_fingertip_offset). No need to apply offset here.
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        ee_flipped_quat, ee_flipped_pos = torch_utils.tf_combine(
            q1=self.ee_quat,
            t1=self.ee_pos,
            q2=flip_z_quat,
            t2=torch.zeros((self.num_envs, 3), device=self.device),
        )

        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )

        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=ee_flipped_quat, t1=ee_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )

        # Add randomization.
        rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        held_asset_pos_noise = 2 * (rand_sample - 0.5)
        if self.cfg_task.name == "gear_mesh":
            held_asset_pos_noise[:, 2] = -rand_sample[:, 2]

        held_asset_pos_noise_level = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        held_asset_pos_noise = held_asset_pos_noise @ torch.diag(held_asset_pos_noise_level)
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            t2=held_asset_pos_noise,
        )

        held_state = self._held_assets[0].data.default_root_state.clone()
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0
        # Write to all held asset articulations (only the active one matters)
        for ha in self._held_assets:
            if ha is None:
                continue
            s = ha.data.default_root_state.clone()
            s[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
            s[:, 3:7] = translated_held_asset_quat
            s[:, 7:] = 0.0
            # Move non-active pairs far away
            if self._num_asset_pairs > 1:
                pair_i = self._held_assets.index(ha)
                non_active = self.asset_pair_idx != pair_i
                s[non_active, 1] = -100.0
            ha.write_root_pose_to_sim(s[:, 0:7])
            ha.write_root_velocity_to_sim(s[:, 7:])
            ha.reset()

    # -----------------------------------------------------------------------
    # Sim helpers
    # -----------------------------------------------------------------------

    def step_sim_no_action(self, update_joint_pos=True):
        """Step the simulation without an action."""
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt, update_joint_pos=update_joint_pos)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def _log_factory_metrics(self, rew_dict, curr_successes):
        """Keep track of episode statistics and log rewards."""
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

    def _log_forge_metrics(self, rew_dict, policy_success_pred):
        """Log metrics to evaluate success prediction performance."""
        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        for thresh, first_success_tx in self.first_pred_success_tx.items():
            curr_predicted_success = policy_success_pred > thresh
            first_success_idxs = torch.logical_and(curr_predicted_success, first_success_tx == 0)

            first_success_tx[:] = torch.where(first_success_idxs, self.episode_length_buf, first_success_tx)

            if torch.any(self.reset_buf):
                delay_ids = torch.logical_and(self.ep_success_times != 0, first_success_tx != 0)
                delay_times = (first_success_tx[delay_ids] - self.ep_success_times[delay_ids]).sum() / delay_ids.sum()
                if delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_all/{thresh}"] = delay_times

                correct_delay_ids = torch.logical_and(delay_ids, first_success_tx > self.ep_success_times)
                correct_delay_times = (
                    first_success_tx[correct_delay_ids] - self.ep_success_times[correct_delay_ids]
                ).sum() / correct_delay_ids.sum()
                if correct_delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_correct/{thresh}"] = correct_delay_times.item()

                pred_success_idxs = first_success_tx != 0
                true_success_preds = torch.logical_and(
                    self.ep_success_times[pred_success_idxs] > 0,
                    self.ep_success_times[pred_success_idxs] < first_success_tx[pred_success_idxs],
                )

                num_pred_success = pred_success_idxs.sum().item()
                et_prec = true_success_preds.sum() / num_pred_success
                if num_pred_success > 0:
                    self.extras[f"early_term_precision/{thresh}"] = et_prec

                true_success_idxs = self.ep_success_times > 0
                num_true_success = true_success_idxs.sum().item()
                et_recall = true_success_preds.sum() / num_true_success
                if num_true_success > 0:
                    self.extras[f"early_term_recall/{thresh}"] = et_recall
