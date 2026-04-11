#!/usr/bin/env python3
"""Quick test: run one forge environment for a few steps.

Since two DirectRLEnv instances cannot coexist in the same process,
run this script twice with --env_id to compare behavior.

Usage (inside container):
  docker exec isaac-lab-forge bash -c 'PYTHONUNBUFFERED=1 /isaac-sim/kit/python/bin/python3 -u /workspace/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/test_compare.py --headless --num_envs 4 --env_id old'

  docker exec isaac-lab-forge bash -c 'PYTHONUNBUFFERED=1 /isaac-sim/kit/python/bin/python3 -u /workspace/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/test_compare.py --headless --num_envs 4 --env_id new'
"""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=30)
parser.add_argument("--env_id", type=str, default="new", choices=["old", "new"])
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

NUM_ENVS = args.num_envs
NUM_STEPS = args.num_steps

ENVS = {
    "old": "Isaac-Forge-PegInsert-Direct-v0",
    "new": "Isaac-ForgeAssembly-Franka-PegInsert-Direct-v0",
}

env_id = ENVS[args.env_id]
print(f"\n{'=' * 60}")
print(f"Testing [{args.env_id}]: {env_id}")
print(f"num_envs={NUM_ENVS}, num_steps={NUM_STEPS}")
print(f"{'=' * 60}")

env_cfg = parse_env_cfg(env_id, num_envs=NUM_ENVS)
print(f"Config: {type(env_cfg).__name__}")

env = gym.make(env_id, cfg=env_cfg)
obs, info = env.reset()
print(f"Reset OK. Obs shape: {obs['policy'].shape}")

action_dim = env.unwrapped.single_action_space.shape[0]
print(f"Action dim: {action_dim}")

total_reward = 0.0
for i in range(NUM_STEPS):
    action = torch.zeros(NUM_ENVS, action_dim, device=env.unwrapped.device)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += float(reward.mean())
    if i % 10 == 0:
        print(f"  Step {i}/{NUM_STEPS}, avg reward: {float(reward.mean()):.4f}")

env.close()
print(f"\n[{args.env_id}] Total reward: {total_reward:.4f}")
print(f"[{args.env_id}] TEST PASSED!")
print(f"{'=' * 60}")

simulation_app.close()
