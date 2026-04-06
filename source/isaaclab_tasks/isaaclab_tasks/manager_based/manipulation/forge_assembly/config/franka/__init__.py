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
    entry_point=f"{__name__}...forge_assembly_env:ForgeAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_forge_assembly_env_cfg:FrankaForgeAssemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaForgeAssemblyRNNPPORunnerCfg",
    },
)

# Franka peg-in-hole assembly (play/inference)
gym.register(
    id="Isaac-ForgeAssembly-Franka-Play-v0",
    entry_point=f"{__name__}...forge_assembly_env:ForgeAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_forge_assembly_env_cfg:FrankaForgeAssemblyEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
