#!/bin/bash
# Training script with wandb integration for UR10 Forge Assembly
# Uploads metrics to wandb.ai after training completes

set -e

PROJECT="forge-assembly"
RUN_NAME="UR10_PegInsert_$(date +%Y%m%d_%H%M%S)"
SUMMARY_DIR="/home/an/IsaacLab/logs/rl_games/Forge/test/summaries"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== UR10 Training with wandb Integration ==="
echo "Project: ${PROJECT}"
echo "Run name: ${RUN_NAME}"
echo "Summary dir: ${SUMMARY_DIR}"

# Clear old summaries
rm -f ${SUMMARY_DIR}/events.out.*

# Start training inside container
docker start isaac-lab-forge 2>/dev/null || true
docker exec -t isaac-lab-forge bash -c "
export CARB_APP_PATH=/isaac-sim/kit
export ISAAC_PATH=/isaac-sim
export EXP_PATH=/isaac-sim/apps
source /isaac-sim/setup_python_env.sh
cd /workspace/IsaacLab
/isaac-sim/kit/python/bin/python3 scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-ForgeAssembly-UR10-PegInsert-Direct-v0 \
  --headless \
  --num_envs 128 \
  --max_iterations 1500
" 2>&1 | tee /tmp/ur10_training.log || true

# Upload to wandb after training completes
echo ""
echo "=== Uploading metrics to wandb ==="
WANDB_DIR=/tmp/wandb python3 ${SCRIPT_DIR}/upload_to_wandb.py \
  --project ${PROJECT} \
  --run-name ${RUN_NAME} \
  --summary-dir ${SUMMARY_DIR} \
  --robot UR10 \
  --tags ur10 peg-insert fixed-peg

echo ""
echo "=== Training and upload complete ==="
