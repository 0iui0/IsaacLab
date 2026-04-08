# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka robot configurations for FORGE tasks with impedance control."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
# Using impedance control matching direct FORGE behavior exactly.

gym.register(
    id="Isaac-Forge-Peg-Insert-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.peg_insert_env_cfg:FrankaPegInsertEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Forge-Peg-Insert-v0-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.peg_insert_env_cfg:FrankaPegInsertEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-Forge-GearMesh-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gear_mesh_env_cfg:FrankaGearMeshEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Forge-GearMesh-v0-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gear_mesh_env_cfg:FrankaGearMeshEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-Forge-NutThread-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nut_thread_env_cfg:FrankaNutThreadEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)

gym.register(
    id="Isaac-Forge-NutThread-v0-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nut_thread_env_cfg:FrankaNutThreadEnvCfg_PLAY",
    },
)
