#!/usr/bin/env python3
"""Test only the new forge_assembly Franka PegInsert environment."""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

ENV_ID = "Isaac-ForgeAssembly-Franka-PegInsert-Direct-v0"
NUM_ENVS = 4
NUM_STEPS = 30

print("=" * 60)
print(f"Testing: {ENV_ID}")
print(f"num_envs={NUM_ENVS}, num_steps={NUM_STEPS}")
print("=" * 60)

env_cfg = parse_env_cfg(ENV_ID, num_envs=NUM_ENVS)
print(f"Config: {type(env_cfg).__name__}")
print(f"  Profile: {env_cfg.robot_profile.ee_body_name}")
print(f"  Arm joints: {env_cfg.robot_profile.num_arm_joints}")

env = gym.make(ENV_ID, cfg=env_cfg)
obs, info = env.reset()
print(f"Reset OK. Obs shape: {obs['policy'].shape}")

total_reward = 0.0
for i in range(NUM_STEPS):
    action_dim = env.unwrapped.single_action_space.shape[0]
    action = torch.zeros(NUM_ENVS, action_dim, device=env.unwrapped.device)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += float(reward.mean())
    if i % 10 == 0:
        print(f"  Step {i}/{NUM_STEPS}, avg reward: {float(reward.mean()):.4f}")

env.close()
print(f"\nTotal reward: {total_reward:.4f}")
print("TEST PASSED!" if total_reward != 0 else "Reward is zero (may need investigation)")
print("=" * 60)

simulation_app.close()
