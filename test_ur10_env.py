#!/usr/bin/env python3
"""Test UR10 environment setup and print debug info."""

import torch

from isaaclab_tasks.direct.forge_assembly.forge_env_cfg import UR10ForgeTaskPegInsertCfg

# Create config
env_cfg = UR10ForgeTaskPegInsertCfg()

print("=" * 60)
print("UR10 Forge Assembly Environment Config")
print("=" * 60)

print(f"\nRobot Profile:")
print(f"  - num_arm_joints: {env_cfg.robot_profile.num_arm_joints}")
print(f"  - has_gripper: {env_cfg.robot_profile.has_gripper}")
print(f"  - grasp_type: {env_cfg.robot_profile.grasp_type}")
print(f"  - ee_body_name: {env_cfg.robot_profile.ee_body_name}")
print(f"  - peg_offset_from_ee: {env_cfg.robot_profile.peg_offset_from_ee}")

print(f"\nInitial Joint Positions (standing):")
print(f"  - reset_arm_joint_pos: {env_cfg.robot_profile.reset_arm_joint_pos}")

print(f"\nControl Config:")
print(f"  - reset_joints: {env_cfg.ctrl.reset_joints}")
print(f"  - default_dof_pos_tensor: {env_cfg.ctrl.default_dof_pos_tensor}")

print(f"\nTask Config:")
print(f"  - hand_init_pos: {env_cfg.task.hand_init_pos}")
print(f"  - hand_init_pos_noise: {env_cfg.task.hand_init_pos_noise}")
print(f"  - fixed_asset_init_pos_noise: {env_cfg.task.fixed_asset_init_pos_noise}")

print(f"\nEvents:")
print(f"  - held_physics_material: {env_cfg.events.held_physics_material}")
print(f"  - object_scale_mass: {env_cfg.events.object_scale_mass}")

print("\n" + "=" * 60)
print("Config check complete!")
print("=" * 60)
