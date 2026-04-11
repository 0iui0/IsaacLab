# Forge Assembly - Peg Insertion Task

Robot-agnostic assembly environment for precision manipulation tasks. Currently supports:
- **Peg Insert** - Insert a peg into a hole using impedance control
- **Gear Mesh** - Mesh gears with precise alignment (Franka only)
- **Nut Thread** - Thread a nut onto a screw (Franka only)

## Quick Start

### Train a Policy

```bash
# Train with RL-Games PPO (128 environments, headless)
docker exec isaac-lab-forge bash -c "
export CARB_APP_PATH=/isaac-sim/kit
export ISAAC_PATH=/isaac-sim
export EXP_PATH=/isaac-sim/apps
source /isaac-sim/setup_python_env.sh
export USD_LIBS=/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
export PYTHONPATH=/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311:\${PYTHONPATH}
export LD_LIBRARY_PATH=/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin:\${LD_LIBRARY_PATH}
export LD_PRELOAD=/isaac-sim/kit/libcarb.so
export RESOURCE_NAME=IsaacSim
cd /workspace/IsaacLab
/isaac-sim/kit/python/bin/python3 -u scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-ForgeAssembly-Franka-PegInsert-Direct-v0 \
  --num_envs 128 \
  --headless \
  --max_iterations 1000
"
```

### Play/Visualize a Policy

```bash
# Play with GUI display (single environment for viewing)
docker exec -e DISPLAY=:10 isaac-lab-forge bash -c "
export CARB_APP_PATH=/isaac-sim/kit
export ISAAC_PATH=/isaac-sim
export EXP_PATH=/isaac-sim/apps
source /isaac-sim/setup_python_env.sh
export USD_LIBS=/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
export PYTHONPATH=/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311:\${PYTHONPATH}
export LD_LIBRARY_PATH=/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin:\${LD_LIBRARY_PATH}
export LD_PRELOAD=/isaac-sim/kit/libcarb.so
export RESOURCE_NAME=IsaacSim
cd /workspace/IsaacLab
/isaac-sim/kit/python/bin/python3 -u scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-ForgeAssembly-Franka-PegInsert-Direct-v0 \
  --num_envs 1 \
  --checkpoint logs/rl_games/Forge/test/nn/Forge.pth
"
```

### Play Headless (Quick Evaluation)

```bash
# Play headless with multiple environments for fast evaluation
docker exec isaac-lab-forge bash -c "
export CARB_APP_PATH=/isaac-sim/kit
export ISAAC_PATH=/isaac-sim
export EXP_PATH=/isaac-sim/apps
source /isaac-sim/setup_python_env.sh
export USD_LIBS=/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
export PYTHONPATH=/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311:\${PYTHONPATH}
export LD_LIBRARY_PATH=/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin:\${LD_LIBRARY_PATH}
export LD_PRELOAD=/isaac-sim/kit/libcarb.so
export RESOURCE_NAME=IsaacSim
cd /workspace/IsaacLab
/isaac-sim/kit/python/bin/python3 -u scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-ForgeAssembly-Franka-PegInsert-Direct-v0 \
  --num_envs 32 \
  --headless \
  --checkpoint logs/rl_games/Forge/test/nn/Forge.pth
"
```

## Available Tasks

| Task ID | Robot | Description | Episode Length |
|---------|-------|-------------|----------------|
| `Isaac-ForgeAssembly-Franka-PegInsert-Direct-v0` | Franka | Peg insertion into hole | 10s |
| `Isaac-ForgeAssembly-Franka-GearMesh-Direct-v0` | Franka | Gear meshing task | 20s |
| `Isaac-ForgeAssembly-Franka-NutThread-Direct-v0` | Franka | Nut threading task | 30s |

## Environment Configuration

### Observation Space (24 dimensions for Franka)
- `fingertip_pos_rel_fixed` (3) - End-effector position relative to fixed peg
- `fingertip_quat` (4) - End-effector orientation
- `ee_linvel` (3) - End-effector linear velocity
- `ee_angvel` (3) - End-effector angular velocity
- `ft_force` (3) - Force-tensor forces
- `force_threshold` (1) - Force threshold for contact detection
- `action_history` (7) - Previous actions

### Action Space (7 dimensions)
- 3 position delta commands (scaled by position threshold)
- 3 rotation delta commands (scaled by rotation threshold)
- 1 gripper command (Franka only)

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `decimation` | 8 | Control decimation (120Hz sim / 8 = 15Hz control) |
| `episode_length_s` | 10.0 | Episode duration in seconds |
| `success_threshold` | 0.04 | Peg insertion threshold (4% of peg height = 1mm) |
| `ft_smoothing_factor` | 0.25 | Force-tensor smoothing factor |

## Robot Profiles

The environment supports multiple robots through the `RobotProfile` abstraction:

```python
# Available profiles
FRANKA_FORGE_PROFILE  # 7-DOF Franka Panda with gripper
UR10_FORGE_PROFILE    # 6-DOF UR10 (fixed peg)
CR5_FORGE_PROFILE     # 6-DOF Dobot CR5 (fixed peg)
```

### Adding a New Robot

1. Create a new profile in `robot_profiles/`:
```python
from isaaclab.utils import configclass
from .base import RobotProfile

@configclass
class MyRobotProfile(RobotProfile):
    robot: ArticulationCfg = MY_ROBOT_CFG
    num_arm_joints: int = 6
    has_gripper: bool = False
    # ... other parameters
```

2. Add task configuration in `forge_env_cfg.py`:
```python
@configclass
class MyRobotForgeTaskPegInsertCfg(ForgeTaskPegInsertCfg):
    robot_profile: RobotProfile = MY_ROBOT_PROFILE
```

3. Register in `__init__.py`:
```python
gym.register(
    id="Isaac-ForgeAssembly-MyRobot-PegInsert-Direct-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg": MyRobotForgeTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml"
    },
)
```

## Training Tips

1. **Success Rate**: The peg insertion task has a tight success threshold (1mm). Expect 50-100 epochs for reasonable convergence.

2. **Environment Count**: More environments = faster training but higher GPU memory usage.
   - Minimum: 32 envs
   - Recommended: 128 envs
   - High-end GPU: 256+ envs

3. **Checkpoint Location**: Checkpoints are saved to `logs/rl_games/Forge/<run_name>/nn/`

4. **Monitor Training**:
```bash
# View training metrics with tensorboard
tensorboard --logdir logs/rl_games/Forge/
```

## Troubleshooting

### "Failed to clone in Fabric" Error
This is a warning that can be ignored for single-environment runs. For multi-env, try:
```bash
# Add --disable_fabric flag
python scripts/.../play.py --disable_fabric ...
```

### Display Issues
- Ensure X11 forwarding is enabled: `xhost +local:docker`
- Set correct DISPLAY: `echo $DISPLAY` (should be `:0` or `:10`)
- For headless runs, always use `--headless` flag

### Low Success Rate
- Check checkpoint is from the correct task
- Verify environment configuration matches training
- Increase training iterations if policy hasn't converged

## File Structure

```
forge_assembly/
├── agents/                     # RL agent configurations
│   └── rl_games_ppo_cfg.yaml   # PPO hyperparameters
├── robot_profiles/             # Robot-specific configurations
│   ├── base.py                 # RobotProfile base class
│   ├── franka.py               # Franka Panda profile
│   ├── ur10.py                 # UR10 profile
│   └── cr5.py                  # Dobot CR5 profile
├── forge_env.py                # Main environment implementation
├── forge_env_cfg.py            # Environment configurations
├── forge_tasks_cfg.py          # Task-specific configurations
├── forge_control.py            # Impedance control implementation
├── forge_events.py             # Randomization events
├── forge_utils.py              # Utility functions
└── __init__.py                 # Gym registrations
```

## References

- Original Factory/Forge implementation: NVIDIA Isaac Lab
- RL-Games: https://github.com/Denys88/rl_games
- Impedance Control: Hogan, B. (1985). Impedance control: An approach to manipulation.
