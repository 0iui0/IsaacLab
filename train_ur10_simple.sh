#!/bin/bash
# Train UR10 Peg Insert with forge_assembly environment
# Uses Isaac Sim launcher directly

set -e

cd /workspace/IsaacLab

/isaac-sim/kit/python/bin/python3 scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-ForgeAssembly-UR10-PegInsert-Direct-v0 \
  --headless \
  --max_envs 128 \
  --log-dir logs/rl_games/forge_assembly_ur10 \
  --run-name ur10_peg_insert_exp1
