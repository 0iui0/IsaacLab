# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registration of Marvin M6 peg-in-hole assembly Gym environments."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Marvin M6 peg-in-hole assembly (training)
gym.register(
    id="Isaac-ForgeAssembly-MarvinM6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.marvin_m6_forge_assembly_env_cfg:MarvinM6ForgeAssemblyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MarvinM6ForgeAssemblyRNNPPORunnerCfg",
    },
)

# Marvin M6 peg-in-hole assembly (play/inference)
gym.register(
    id="Isaac-ForgeAssembly-MarvinM6-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.marvin_m6_forge_assembly_env_cfg:MarvinM6ForgeAssemblyEnvCfg_PLAY",
    },
)
