# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registration of CR5AF peg-in-hole assembly Gym environments."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# CR5AF peg-in-hole assembly (training)
gym.register(
    id="Isaac-ForgeAssembly-CR5AF-v0",
    entry_point=f"{__name__}....forge_assembly_env:ForgeAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cr5af_forge_assembly_env_cfg:CR5AFForgeAssemblyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CR5AFForgeAssemblyRNNPPORunnerCfg",
    },
)

# CR5AF peg-in-hole assembly (play/inference)
gym.register(
    id="Isaac-ForgeAssembly-CR5AF-Play-v0",
    entry_point=f"{__name__}....forge_assembly_env:ForgeAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cr5af_forge_assembly_env_cfg:CR5AFForgeAssemblyEnvCfg_PLAY",
    },
)
