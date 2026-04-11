#!/usr/bin/env python3
"""Check gym registration for forge_assembly environments."""

from isaaclab.app import AppLauncher
import argparse
import sys

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym

# Force import of forge_assembly to register gym envs
from isaaclab_tasks.direct import forge_assembly  # noqa: F401

print("=" * 60, file=sys.stderr)
print("Registered Forge Assembly environments:", file=sys.stderr)
print("=" * 60, file=sys.stderr)
for id in sorted(gym.registry.keys()):
    if "ForgeAssembly" in id:
        print(f"  {id}", file=sys.stderr)
print("=" * 60, file=sys.stderr)

# Try to get the UR10 env spec
try:
    spec = gym.spec("Isaac-ForgeAssembly-UR10-PegInsert-Direct-v0")
    print(f"\nUR10 env spec found!", file=sys.stderr)
except gym.error.NameNotFound as e:
    print(f"\nUR10 env NOT found: {e}", file=sys.stderr)

simulation_app.close()
