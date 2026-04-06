# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registration of Franka peg-in-hole assembly Gym environments."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Franka peg-in-hole assembly (training)
gym.register(
    id="Isaac-ForgeAssembly-Franka-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.forge_assembly.forge_assembly_env:ForgeAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.forge_assembly.config.franka.franka_forge_assembly_env_cfg:FrankaForgeAssemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# Franka peg-in-hole assembly (play/inference)
gym.register(
    id="Isaac-ForgeAssembly-Franka-Play-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.forge_assembly.forge_assembly_env:ForgeAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.forge_assembly.config.franka.franka_forge_assembly_env_cfg:FrankaForgeAssemblyEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
