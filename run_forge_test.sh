#!/bin/bash
# Wrapper to run scripts inside the isaac-lab-forge container with proper env
# Usage: docker exec isaac-lab-forge bash /workspace/IsaacLab/run_forge_test.sh <script.py> [args...]

set -e

SCRIPT_DIR="/isaac-sim"

# Setup all Isaac Sim env vars
export CARB_APP_PATH=${SCRIPT_DIR}/kit
export ISAAC_PATH=${SCRIPT_DIR}
export EXP_PATH=${SCRIPT_DIR}/apps
source ${SCRIPT_DIR}/setup_python_env.sh

# Add missing pxr/usd libs paths
USD_LIBS="${SCRIPT_DIR}/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
export PYTHONPATH="${USD_LIBS}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${USD_LIBS}/bin:${LD_LIBRARY_PATH}"
export LD_PRELOAD=${SCRIPT_DIR}/kit/libcarb.so
export RESOURCE_NAME="IsaacSim"

cd /workspace/IsaacLab
${SCRIPT_DIR}/kit/python/bin/python3 "$@"
