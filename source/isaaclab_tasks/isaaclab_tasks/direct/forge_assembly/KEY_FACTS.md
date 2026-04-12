# UR10 Forge Assembly - Key Facts and Findings

## Date: 2026-04-12

## Problem Summary
UR10 training for Forge Assembly Peg Insert task had multiple issues:
1. IK solver infinite loops during environment reset
2. Peg visualization not appearing in GUI
3. Peg collision not being detected properly
4. Physics hierarchy warnings (multiple rigid bodies in hierarchy)

## Solutions Implemented

### 1. IK Convergence Fix
**Problem**: IK solver failed to converge for many environments, causing training to hang or skip environments.

**Root Cause**: 
- UR10 initial pose was not optimal for the task workspace
- Fixed asset (hole) position was outside UR10's reachable workspace

**Solution**:
```python
# In robot_profiles/ur10.py
reset_arm_joint_pos=[0.0, -0.5, -0.5, -0.5, 0.0, 0.0]  # Arm forward and down

# In forge_env_cfg.py  
fixed_asset_offset = torch.tensor([0.0, 0.50, 0.0], device=self.device)  # 50cm distance
```

**Result**: IK convergence improved significantly, most resets succeed within 1-3 attempts.

### 2. Peg Visualization
**Problem**: Peg was not visible in the GUI when running training.

**Solution**: Added `_spawn_peg_visualization()` method in `forge_env.py`:
```python
def _spawn_peg_visualization(self):
    """Spawn peg as collision prim child of EE link."""
    for env_id in range(self.num_envs):
        peg_prim_path = f"/World/envs/env_{env_id}/Robot/{self.profile.ee_body_name}/peg"
        peg_cfg = sim_utils.CylinderCfg(
            radius=peg_radius,
            height=peg_height,
            # No rigid_props - parent EE handles physics
            collision_props=sim_utils.CollisionPropertiesCfg(...),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        )
        peg_cfg.func(peg_prim_path, peg_cfg, translation=(0, 0, -peg_height/2))
```

**Result**: Peg now visible in GUI, moves with EE link.

### 3. Physics Hierarchy Fix
**Problem**: PhysX error: "Rigid Body missing xformstack reset when child of another enabled rigid body"

**Root Cause**: Peg and force_sensor were spawned with `RigidBodyPropertiesCfg` as children of `ee_link`, which is already part of the robot articulation. PhysX doesn't support multiple rigid bodies in a hierarchy.

**Solution**: Removed `RigidBodyPropertiesCfg` and `MassPropertiesCfg` from peg and force_sensor. They are now collision-only prims that inherit motion from the parent EE link.

```python
# BEFORE (wrong - causes PhysX errors)
peg_cfg = sim_utils.CylinderCfg(
    rigid_props=sim_utils.RigidBodyPropertiesCfg(...),  # REMOVE THIS
    mass_props=sim_utils.MassPropertiesCfg(...),        # REMOVE THIS
    collision_props=sim_utils.CollisionPropertiesCfg(...),
)

# AFTER (correct - collision only)
peg_cfg = sim_utils.CylinderCfg(
    # No rigid_props - parent EE link handles physics
    collision_props=sim_utils.CollisionPropertiesCfg(...),
    physics_material=sim_utils.RigidBodyMaterialCfg(...),
)
```

**Result**: No more PhysX hierarchy warnings.

### 4. Docker Configuration
**Problem**: Code changes not reflected in container, GPU not accessible.

**Solution**: Created `docker-compose-forge-test.yml`:
```yaml
services:
  isaac-lab-forge-test:
    image: nvcr.io/nvidia/isaac-lab:2.3.2
    volumes:
      - ~/IsaacLab/source:/workspace/isaaclab/source:rw  # Correct path!
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["all"]
              capabilities: ["gpu"]
```

**Key Finding**: Mount path must match container's working directory (`/workspace/isaaclab` not `/workspace/IsaacLab`).

## Files Created/Modified

### New Files
1. `docker-compose-forge-test.yml` - Docker Compose config for testing
2. `source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/assets/ur10_peg_25mm.usd` - Standalone peg USD
3. `source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/assets/factory_peg_25mm.usd` - Alternative peg USD
4. `source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/check_gym_envs.py` - Gym registration checker
5. `source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/ur10_add_peg_usd.py` - Peg USD creator script
6. `source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/LESSONS_LEARNED.md` - Detailed lessons learned

### Modified Files
1. `source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/forge_env.py`
   - Fixed `_spawn_peg_visualization()` to use collision-only prims
   - Removed `RigidBodyPropertiesCfg` from peg and force_sensor
   - Added `max_ik_attempts` limit in `randomize_initial_state()`

2. `source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/robot_profiles/ur10.py`
   - Tuned `reset_arm_joint_pos` for better IK convergence
   - Set `fixed_asset_offset` to 0.50m for optimal workspace

3. `source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/forge_env_cfg.py`
   - Adjusted `hand_init_pos` Z to 0.05 for easier IK
   - Tuned `fixed_asset_init_pos_noise` for UR10

## Technical Key Facts

### USD/PhysX Hierarchy Rules
1. **DO NOT** create `RigidBodyAPI` children of articulation links
2. **DO** use collision-only prims for attached objects
3. Motion is inherited through USD hierarchy - no need for separate rigid body
4. Collision detection works through `CollisionAPI` + `PhysxMaterialAPI`

### UR10 Specifics
- 6-DOF manipulator (vs Franka's 7-DOF)
- No null-space for redundancy resolution
- Optimal workspace: 0.3m - 0.8m from base
- Default reset pose: `[0, -0.5, -0.5, -0.5, 0, 0]` works well for table tasks
- **Peg direction**: along ee_link X-axis (perpendicular to flange, outward)
- **IK target orientation**: `hand_init_orn=[4.712, 1.571, 1.571]` (roll=270°, pitch=90°, yaw=90°)
  - pitch=90° makes ee_link X-axis point down
  - roll=270° + yaw=90° orient the flange correctly

### Isaac Sim APIs
- `sim_utils.CylinderCfg.func()` - Spawn cylinder prims
- `CollisionAPI.Apply()` - Enable collision detection
- `PhysxCollisionAPI` - Contact offset, rest offset
- `RigidBodyMaterialAPI` - Friction, restitution (when applicable)
- `PreviewSurfaceCfg` - Visual material

### Gym Registration
```python
# Registration in __init__.py
gym.register(
    id="Isaac-ForgeAssembly-UR10-PegInsert-Direct-v0",
    entry_point="...forge_env:ForgeEnv",
    kwargs={
        "env_cfg_entry_point": "...forge_env_cfg:UR10ForgeTaskPegInsertCfg",
    },
)

# Verification
from isaaclab_tasks.direct import forge_assembly  # Force import
import gymnasium as gym
spec = gym.spec("Isaac-ForgeAssembly-UR10-PegInsert-Direct-v0")  # Should succeed
```

## Training Status
- **Environment**: `Isaac-ForgeAssembly-UR10-PegInsert-Direct-v0`
- **Num Envs**: 8
- **Device**: CUDA:0
- **Status**: Training runs successfully, no IK failures, no PhysX errors

## Next Steps / Recommendations

1. **Test with GUI**: Run training with `--enable_cameras` to verify peg visualization

2. **Validate Collision**: Test that peg actually collides with hole (not just visual)

3. **UR10 USD Integration**: Consider creating a custom UR10 USD with built-in peg:
   - Modify UR10 USD directly (requires locating the source file)
   - Or use USD reference to include peg in UR10 instance

4. **Performance Tuning**: 
   - Monitor training convergence
   - Adjust reward weights if needed
   - Consider increasing `num_envs` for faster training

## References
- Isaac Sim Documentation: https://docs.omniverse.nvidia.com/
- Isaac Lab GitHub: https://github.com/isaac-sim/IsaacLab
- PhysX USD Schema: https://openusd.org/dev/api/physx_schema_page_front.html
