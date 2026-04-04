"""Test script to verify forge_assembly environment registration."""
import gymnasium as gym

import isaaclab_tasks  # noqa: F401

# List all forge-related environments
envs = [e for e in gym.envs.registry.keys() if "Forge" in e or "forge" in e.lower()]
for e in sorted(envs):
    print(e)

if not envs:
    print("NO FORGE ENVS FOUND")
