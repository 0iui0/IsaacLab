#!/bin/bash
# Train Franka Peg Insert with forge_assembly environment (for comparison)

set -e

docker exec isaac-lab-forge bash -c "
export CARB_APP_PATH=/isaac-sim/kit
export ISAAC_PATH=/isaac-sim
source /isaac-sim/kit/setup_python_env.sh 2>/dev/null || true
export USD_LIBS=/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
export PYTHONPATH=/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311:\${PYTHONPATH}
export LD_PRELOAD=/isaac-sim/kit/libcarb.so
export RESOURCE_NAME=IsaacSim

cd /workspace/IsaacLab

/isaac-sim/kit/python/bin/python3 -u scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-ForgeAssembly-Franka-PegInsert-Direct-v0 \
  --headless \
  --max_envs 128 \
  --log-dir logs/rl_games/forge_assembly_franka_new \
  --run-name franka_peg_insert_exp1
"
