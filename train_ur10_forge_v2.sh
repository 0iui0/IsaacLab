#!/bin/bash
# Train UR10 Peg Insert with forge_assembly environment

set -e

cd /workspace/IsaacLab

# Use the isaac-sim python.sh wrapper which sets up all environment variables
export CARB_APP_PATH=/isaac-sim/kit
export ISAAC_PATH=/isaac-sim
export EXP_PATH=/isaac-sim/apps
export LD_PRELOAD=/isaac-sim/kit/libcarb.so

/isaac-sim/python.sh -u scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-ForgeAssembly-UR10-PegInsert-Direct-v0 \
  --headless \
  --max_envs 128 \
  --log-dir logs/rl_games/forge_assembly_ur10 \
  --run-name ur10_peg_insert_exp1
