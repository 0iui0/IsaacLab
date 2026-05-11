# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
#
# ID convention: Isaac-ForgeAssembly-{Robot}-{Task}-Direct-v0
# This avoids collision with the original forge/ IDs (Isaac-Forge-*-Direct-v0).
##

# --- Franka ---

gym.register(
    id="Isaac-ForgeAssembly-Franka-PegInsert-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:FrankaForgeTaskPegInsertCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ForgeAssembly-Franka-GearMesh-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:FrankaForgeTaskGearMeshCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ForgeAssembly-Franka-NutThread-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:FrankaForgeTaskNutThreadCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)

# --- UR10 ---

gym.register(
    id="Isaac-ForgeAssembly-UR10-PegInsert-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:UR10ForgeTaskPegInsertCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# --- CR5 ---

gym.register(
    id="Isaac-ForgeAssembly-CR5-PegInsert-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:CR5ForgeTaskPegInsertCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# --- Marvin M6 + Robotiq 2F-85 ---

gym.register(
    id="Isaac-ForgeAssembly-MarvinRobotiq-PegInsert-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:MarvinForgeTaskPegInsertCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# --- Marvin M6 + Force Sensor + Franka Panda Gripper ---

gym.register(
    id="Isaac-ForgeAssembly-MarvinPanda-PegInsert-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:MarvinPandaForgeTaskPegInsertCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ForgeAssembly-MarvinPanda-PegInsertPair12-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:MarvinPandaForgeTaskPegInsertPair12Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_finetune.yaml",
    },
)

# --- Ablation experiments ---

gym.register(
    id="Isaac-ForgeAssembly-AblationB-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:MarvinPandaAblationBCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_ablation_B.yaml",
    },
)

gym.register(
    id="Isaac-ForgeAssembly-AblationC-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:MarvinPandaAblationCCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_ablation_C.yaml",
    },
)
