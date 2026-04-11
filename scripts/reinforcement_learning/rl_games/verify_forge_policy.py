#!/usr/bin/env python3
"""Quick policy verification - run episodes and count successes."""
import argparse
import sys

parser = argparse.ArgumentParser(description="Quick policy verification")
parser.add_argument("--task", type=str, default="Isaac-ForgeAssembly-Franka-PegInsert-Direct-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--checkpoint", type=str, default="logs/rl_games/Forge/test/nn/Forge.pth")
parser.add_argument("--max_episodes", type=int, default=50)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import os
import torch
import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401

@hydra_task_config(args.task, "rl_games_cfg_entry_point")
def verify_policy(env_cfg, agent_cfg):
    """Run quick policy verification."""
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"

    # Create environment
    env = gym.make(args.task, cfg=env_cfg)

    # Wrap for rl-games
    rl_device = "cuda:0"
    clip_obs = math.inf
    clip_actions = math.inf
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {
        "vecenv_type": "IsaacRlgWrapper",
        "env_creator": lambda **kwargs: env
    })

    # Load agent
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = args.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(args.checkpoint)
    agent.reset()

    print(f"\nLoaded checkpoint: {args.checkpoint}")
    print(f"Running {args.max_episodes} episodes with {args.num_envs} envs...")

    # Run episodes
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    total_successes = 0
    total_steps = 0
    max_steps = env.unwrapped.max_episode_length
    success_flags = torch.zeros(args.num_envs, dtype=torch.bool, device="cuda:0")

    with torch.inference_mode():
        for step in range(max_steps):
            obs_torch = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_torch, is_deterministic=True)
            obs, rew, dones, infos = env.step(actions)
            total_steps += 1

            # Check for successes in infos
            if "successes" in infos:
                new_successes = infos["successes"] & ~success_flags
                total_successes += new_successes.sum().item()
                success_flags |= infos["successes"]

            if success_flags.all():
                break

    success_rate = total_successes / args.num_envs
    print(f"\n=== Policy Verification Results ===")
    print(f"Environments: {args.num_envs}")
    print(f"Steps run: {total_steps}")
    print(f"Successes: {total_successes} / {args.num_envs}")
    print(f"Success rate: {success_rate:.2%}")

    if success_rate >= 0.8:
        print("✓ Policy performs WELL - good convergence!")
    elif success_rate >= 0.5:
        print("~ Policy performs MODERATELY - some learning achieved")
    else:
        print("✗ Policy performs POORLY - needs more training")

    env.close()

if __name__ == "__main__":
    verify_policy()
    simulation_app.close()
