# Forge Assembly - Peg Insertion Task

Robot-agnostic assembly environment for precision manipulation tasks. Currently supports:
- **Peg Insert** - Insert a peg into a hole using impedance control
- **Gear Mesh** - Mesh gears with precise alignment (Franka only)
- **Nut Thread** - Thread a nut onto a screw (Franka only)

## Quick Start

### Training Commands (Simplified)

```bash
# Franka Panda - Default
docker exec isaac-lab-task-main bash -c '
  cd /workspace/isaaclab
  /isaac-sim/python.sh scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-ForgeAssembly-Franka-PegInsert-Direct-v0 \
    --num_envs 128 --headless
'

# Marvin M6 + Panda Gripper (需要代理下载Nucleus资产)
docker exec isaac-lab-task-main bash -c '
  cd /workspace/isaaclab
  /isaac-sim/python.sh scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-ForgeAssembly-MarvinPanda-PegInsert-Direct-v0 \
    --num_envs 16 --headless
'

# UR10 (固定peg模式)
docker exec isaac-lab-task-main bash -c '
  cd /workspace/isaaclab
  /isaac-sim/python.sh scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-ForgeAssembly-UR10-PegInsert-Direct-v0 \
    --num_envs 128 --headless
'
```

### GUI Training (调试模式)

```bash
# 不带--headless即可启动GUI
docker exec isaac-lab-task-main bash -c '
  export https_proxy=http://your_proxy
  cd /workspace/isaaclab
  /isaac-sim/python.sh scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-ForgeAssembly-MarvinPanda-PegInsert-Direct-v0 \
    --num_envs 16
'
```

### Play/Inference

```bash
docker exec isaac-lab-task-main bash -c '
  cd /workspace/isaaclab
  /isaac-sim/python.sh scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-ForgeAssembly-MarvinPanda-PegInsert-Direct-v0 \
    --num_envs 1
'
```

## URDF → USD Conversion

```bash
docker exec isaac-lab-task-main bash -c '
  /isaac-sim/python.sh /workspace/isaaclab/scripts/tools/convert_urdf.py \
    /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/assets/urdf/marvin_m6_panda/marvin_m6_panda.urdf \
    /workspace/isaaclab/source/isaaclab_assets/data/robots/marvin/marvin_m6_panda.usd \
    --merge-joints \
    --fix-base
'
```

## Available Tasks

| Task ID | Robot | Description | Episode Length |
|---------|-------|-------------|----------------|
| `Isaac-ForgeAssembly-Franka-PegInsert-Direct-v0` | Franka | Peg insertion into hole | 10s |
| `Isaac-ForgeAssembly-MarvinPanda-PegInsert-Direct-v0` | Marvin M6 + Panda | Peg insertion (gripper) | 10s |
| `Isaac-ForgeAssembly-Marvin-Robotiq-PegInsert-Direct-v0` | Marvin M6 + Robotiq | Peg insertion (gripper) | 10s |
| `Isaac-ForgeAssembly-UR10-PegInsert-Direct-v0` | UR10 | Fixed peg insertion | 10s |
| `Isaac-ForgeAssembly-CR5-PegInsert-Direct-v0` | CR5 | Fixed peg insertion | 10s |

## Robot Profiles

| Robot | DOF | Gripper | ee_body | ee_to_fingertip_offset |
|-------|-----|---------|---------|------------------------|
| Franka | 7 | Panda | panda_fingertip_centered | [0, 0, 0] |
| Marvin Panda | 7 | Panda | Link7_R | [0, -0.129, 0] |
| Marvin Robotiq | 7 | Robotiq 2F-85 | force_sensor | [0, 0, 0] |
| UR10 | 6 | Fixed Peg | ee_link | N/A |
| CR5 | 6 | Fixed Peg | ee_link | N/A |

### Key Differences: Gripper vs Fixed-Peg

**Gripper robots (Franka, Marvin)**:
- `held_asset`: Peg is a separate articulation grabbed by gripper
- Peg initialized in gripper at reset
- IK positions fingertip above hole

**Fixed-Peg robots (UR10, CR5)**:
- Peg is a collision shape spawned on EE link
- No held_asset, peg is part of robot
- IK positions EE link (with peg) above hole

## Environment Configuration

### Observation Space (24 dimensions for Franka)
- `fingertip_pos_rel_fixed` (3) - End-effector position relative to hole
- `fingertip_quat` (4) - End-effector orientation
- `ee_linvel` (3) - End-effector linear velocity
- `ee_angvel` (3) - End-effector angular velocity
- `ft_force` (3) - Force-tensor forces
- `force_threshold` (1) - Force threshold for contact detection
- `action_history` (7) - Previous actions

### Action Space (7 dimensions)
- 3 position delta commands (scaled by pos_action_threshold)
- 3 rotation delta commands (yaw only for gripper robots)
- 1 gripper command / success prediction

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `decimation` | 8 | Control decimation (120Hz sim / 8 = 15Hz control) |
| `episode_length_s` | 10.0 | Episode duration in seconds |
| `success_threshold` | 0.04 | Peg insertion threshold (4% of peg height = 1mm) |
| `ft_smoothing_factor` | 0.25 | Force-tensor smoothing factor |

### Control Parameters (ForgeCtrlCfg)

- `reset_joints`: Initial joint positions for IK reset
- `pos_action_bounds`: Max position delta per action (meters)
- `default_task_prop_gains`: Impedance gains [Kp_x, Kp_y, Kp_z, Kp_roll, Kp_pitch, Kp_yaw]

## Adding a New Robot

1. Create profile in `robot_profiles/`:
```python
from .base import RobotProfile

MY_ROBOT_FORGE_PROFILE = RobotProfile(
    robot=ArticulationCfg(...),
    num_arm_joints=7,
    has_gripper=True,
    ee_body_name="my_ee_link",
    ee_to_fingertip_offset=[0, 0, 0],  # If ee_body != fingertip center
    fingerpad_length=0.02,
    grasp_type="gripper",
)
```

2. Add task configuration in `forge_env_cfg.py`:
```python
@configclass
class MyRobotForgeTaskPegInsertCfg(ForgeTaskPegInsertCfg):
    robot_profile: RobotProfile = MY_ROBOT_FORGE_PROFILE
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

## Troubleshooting

### Body Name Not Found
```
ValueError: 'panda_fingertip_centered' is not in list
```
Cause: URDF→USD conversion merged fixed joints.
Fix: Use `ee_body_name="Link7_R"` and add `ee_to_fingertip_offset`.

### Peg Floating in Air
Cause: `ee_to_fingertip_offset` calculation incorrect.
Fix: Compute offset by tracing URDF kinematic chain from ee_body to fingertip.

**Example for Marvin M6 + Panda Gripper (Link7_R → fingertip_centered):**
1. Link7_R → force_sensor: `xyz=[0, -0.1, 0]`, `rpy=[π/2, 0, 0]` (90° X rotation)
   - After rotation: force_sensor Z-axis = Link7_R -Y-axis
2. force_sensor → panda_hand: `xyz=[0, 0, 0.0165]` (along force_sensor Z)
   - In Link7_R frame: Y offset = -0.0165
3. panda_hand → fingertip_centered: `xyz=[0, 0, 0.112071]` (along panda_hand Z)
   - In Link7_R frame: Y offset = -0.112071
4. **Total Y offset = -0.1 - 0.0165 - 0.112071 = -0.228571 m**

Set: `ee_to_fingertip_offset=[0, -0.228571, 0]`

### Hole Not Visible
Cause: Fixed asset spawn configuration issue.
Fix: Verify fixed_asset prim_path matches regex `/World/envs/env_.*/FixedAsset`.

### "Failed to clone in Fabric" Warning
Can be ignored. Use `clone_in_fabric=False` in scene config if needed.

## File Structure

```
forge_assembly/
├── agents/                     # RL agent configurations
│   └── rl_games_ppo_cfg.yaml   # PPO hyperparameters
├── robot_profiles/             # Robot-specific configurations
│   ├── base.py                 # RobotProfile base class
│   ├── franka.py               # Franka Panda profile
│   ├── marvin.py               # Marvin M6 profiles
│   ├── ur10.py                 # UR10 profile
│   └── cr5.py                  # Dobot CR5 profile
├── assets/urdf/                # Source URDF files
│   └── marvin_m6_panda/        # Marvin M6 + Panda URDF
├── forge_env.py                # Main environment implementation
├── forge_env_cfg.py            # Environment configurations
├── forge_tasks_cfg.py          # Task-specific configurations
├── forge_control.py            # Impedance control implementation
├── forge_events.py             # Randomization events
├── forge_utils.py              # Utility functions
└── __init__.py                 # Gym registrations
```

## Proxy Settings (China)

All Docker commands need proxy for Nucleus asset downloads:

```bash
export https_proxy=http://your_proxy
```

Or add to `docker-compose.yaml`:
```yaml
environment:
  - https_proxy=http://your_proxy
```

## References

- Isaac Lab: https://github.com/isaac-sim/IsaacLab
- RL-Games: https://github.com/Denys88/rl_games
- Impedance Control: Hogan, B. (1985). Impedance control: An approach to manipulation.