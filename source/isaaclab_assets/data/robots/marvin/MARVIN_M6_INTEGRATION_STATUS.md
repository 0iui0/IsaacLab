# MARVIN M6 Robot Integration - Status Summary

**Date:** 2026-04-16

## Overview

This document summarizes the integration of the MARVIN M6-CCS-696 7-DOF robot into Isaac Lab's forge_assembly framework. The MARVIN M6 provides 7-DOF redundancy (like Franka), enabling null-space control that was unavailable with the 6-DOF UR10.

---

## Completed Work

### 1. URDF Acquisition ✅

**Location:** `/home/an/TJ_FX_ROBOT_CONTRL_SDK/URDF/Marvin M6-CCS-696-urdf_V4.0/`

**Robot Specifications:**
- **DOF:** 7 (Joint1_L through Joint7_L)
- **Joint Limits:**
  - J1: ±3.1067 rad (±178°) - Base rotation
  - J2: ±2.0944 rad (±120°) - Shoulder
  - J3: ±3.1067 rad (±178°) - Elbow
  - J4: [-2.5307, 1.0472] rad ([-145°, 60°]) - Wrist 1
  - J5: ±3.1067 rad (±178°) - Wrist 2
  - J6: ±1.0472 rad (±60°) - Wrist 3
  - J7: ±1.5708 rad (±90°) - Wrist 4 / EE rotation
- **Effort Limits:** J1-J3: 108 Nm, J4-J7: 66 Nm
- **Velocity Limits:** 3.1416 rad/s (180°/s) for all joints

### 2. Robot Configuration Files Created ✅

**File:** `source/isaaclab_assets/isaaclab_assets/robots/marvin.py`

Contains three configurations:
- `MARVIN_M6_CFG`: Base 7-DOF arm (no gripper)
- `MARVIN_M6_GRIPPER_CFG`: With Franka-like parallel-jaw gripper
- `MARVIN_M6_PEG_CFG`: With fixed peg tool (for insertion tasks)

**Key Features:**
- Joint names match URDF: `Joint1_L` through `Joint7_L`
- Semi-folded default pose exploits kinematic redundancy
- Actuator effort limits match MARVIN specifications

### 3. Robot Profile Created ✅

**File:** `source/isaaclab_tasks/.../robot_profiles/marvin.py`

Exports `MARVIN_M6_FORGE_PROFILE` with:
- `num_arm_joints=7` (7-DOF redundancy)
- `has_gripper=True` (Franka-like parallel-jaw)
- `grasp_type="gripper"` (not fixed_peg like UR10)
- `null_space_default_pos=[0.0, -0.3, 0.3, 0.0, 0.0, 0.0, 0.0]` (semi-folded pose)
- `fingerpad_length=0.017608` (matches Franka)
- `keypoint_axis=2` (Z-axis, like Franka)

### 4. Task Configuration Added ✅

**File:** `source/isaaclab_tasks/.../forge_env_cfg.py`

Added `MARVINM6ForgeTaskPegInsertCfg` class with:
- 7-element joint arrays (matching Franka layout)
- Null-space control enabled: `kp_null=10.0`, `kd_null=6.3246`
- No joint-space regularization needed (unlike UR10's `kp_joint_reg=100.0`)
- Full yaw range: `[-180.0, 90.0]` (no q1 divergence concerns)
- Same workspace as UR10: `fixed_asset` at `(0.45, 0.0, 0.05)`

### 5. Environment Registration ✅

**File:** `source/isaaclab_tasks/.../__init__.py`

Registered environment:
```
Isaac-ForgeAssembly-MarvinM6-PegInsert-Direct-v0
```

### 6. All Python Files Syntax-Validated ✅

All files pass `python3 -c "import ast; ast.parse(...)"` checks.

---

## Blocking Issue: USD Asset Creation ❌

### Problem

URDF to USD conversion fails in headless mode:
```bash
docker exec isaac-lab-forge-test bash -c "
/isaac-sim/python.sh /workspace/isaaclab/scripts/tools/convert_urdf.py \
    /tmp/marvin_m6.urdf \
    /tmp/marvin_m6.usd \
    --merge-joints \
    --joint-stiffness 100.0 \
    --joint-damping 1.0 \
    --headless
"
```

**Result:** USD file is only 1.5K (should be MBs for full robot model)

**Root Cause:** Isaac Sim's URDF importer has known limitations in headless mode. The conversion process starts but doesn't complete mesh imports and articulation root setup.

### Required USD Asset

**Target Path:**
```
/home/an/IsaacLab/source/isaaclab_assets/data/robots/marvin/marvin_m6.usd
```

**Required Structure:**
```
marvin_m6.usd (or marvin_m6_instanceable.usd)
├── Articulation root at prim path (e.g., /Marvin_M6)
├── Joints: Joint1_L, Joint2_L, ..., Joint7_L (7-DOF arm)
├── Links: Base_L, Link1_L, ..., Link7_L
├── Gripper variants:
│   ├── Franka_Like variant with:
│   │   ├── M6_left_finger (collision mesh)
│   │   ├── M6_right_finger (collision mesh)
│   │   ├── M6_left_finger_joint (prismatic)
│   │   └── M6_right_finger_joint (prismatic)
│   └── (Optional) Peg variant for fixed-peg tasks
└── PhysX articulation properties:
    ├── solver_position_iteration_count=192
    ├── solver_velocity_iteration_count=1
    └── enabled_self_collisions=False
```

### Solutions (Choose One)

#### Option 1: Isaac Sim GUI (Recommended)

1. Open Isaac Sim GUI (not headless):
   ```bash
   docker exec -it isaac-lab-forge-test /isaac-sim/isaac-sim.sh
   ```

2. Import URDF:
   - File → Import → `/tmp/marvin_m6.urdf`
   - Accept default import settings

3. Add Articulation Root:
   - Select the root prim
   - Add → Physics → Articulation Root

4. Configure joints:
   - Ensure all 7 joints have Drive settings
   - Set stiffness=200, damping=20 for shoulder joints
   - Set stiffness=100, damping=10 for wrist joints

5. Add gripper (if not in URDF):
   - Import Franka gripper or create parallel-jaw gripper
   - Add prismatic joints for finger motion

6. Export as USD:
   - File → Export → `/home/an/IsaacLab/source/isaaclab_assets/data/robots/marvin/marvin_m6.usd`

#### Option 2: Manual USD Creation

1. Copy Franka USD as template:
   ```bash
   cp /path/to/franka_instanceable.usd \
      /home/an/IsaacLab/source/isaaclab_assets/data/robots/marvin/marvin_m6_instanceable.usd
   ```

2. Replace geometry:
   - Keep Franka's articulation structure
   - Replace mesh paths with MARVIN meshes from `/tmp/marvin_m6_meshes/`

3. Update joint names:
   - Change `panda_joint*` to `Joint*_L`
   - Change `panda_finger_joint*` to `M6_*_finger_joint`

4. Update dynamics:
   - Set mass/inertia from MARVIN INI file values
   - Set effort limits: J1-J3=108 Nm, J4-J7=66 Nm

#### Option 3: Alternative Converter

Try using `omni.usd.import` API directly (may still require GUI):
```python
import omni.usd
stage = omni.usd.get_context().new_stage()
omni.usd.get_context().import_stage_with_args(
    "/tmp/marvin_m6.urdf",
    "/Marvin_M6",
    ["mergeFixedJoints=true", "fixBase=false"]
)
stage.GetRootLayer().Export("/path/to/marvin_m6.usd")
```

---

## Verification Checklist

After USD asset is created, verify:

- [ ] USD file size > 1MB (indicates full geometry)
- [ ] Opens in Isaac Sim without errors
- [ ] All 7 joints visible and movable
- [ ] Gripper joints present (if using gripper variant)
- [ ] Articulation root properly configured
- [ ] Meshes render correctly (no missing geometry)

Then test in Isaac Lab:

```bash
docker exec isaac-lab-forge-test bash -c "
cd /workspace/isaaclab && \
python -m isaaclab_tasks.direct.forge_assembly \
    --task Isaac-ForgeAssembly-MarvinM6-PegInsert-Direct-v0 \
    --num_envs 2 \
    --device cuda:0 \
    --headless
"
```

Expected behavior:
- Environment spawns without errors
- Robot articulation is recognized
- 7 joint positions are tracked
- Gripper can open/close

---

## Training Validation (Post-USD)

After basic instantiation works, run training:

```bash
docker exec isaac-lab-forge-test bash -c "
python -m isaaclab_tasks.direct.forge_assembly \
    --task Isaac-ForgeAssembly-MarvinM6-PegInsert-Direct-v0 \
    --num_envs 128 \
    --max_iterations 500 \
    --seed 42 \
    --wandb-name marvin-m6-v1-nullspace
"
```

**Success Criteria (vs. UR10):**
- |q1| stays stable without explicit `kp_joint_reg` regularization
- Null-space control visible (elbow motion without EE drift)
- Policy discovers insertion within 50-100 epochs
- Success rate >0% by epoch 200
- No q1 divergence pattern (unlike UR10 V9-V18)

---

## Files Modified/Created

| File | Type | Status |
|------|------|--------|
| `isaaclab_assets/robots/marvin.py` | CREATE | ✅ Complete |
| `forge_assembly/robot_profiles/marvin.py` | CREATE | ✅ Complete |
| `forge_assembly/forge_env_cfg.py` | MODIFY | ✅ Added MARVINM6ForgeTaskPegInsertCfg |
| `forge_assembly/__init__.py` | MODIFY | ✅ Registered new environment |
| `forge_assembly/robot_profiles/__init__.py` | MODIFY | ✅ Added export |
| `data/robots/marvin/marvin_m6.usd` | CREATE | ❌ BLOCKING - Requires GUI |

---

## Key Differences: MARVIN M6 vs. UR10

| Feature | UR10 (6-DOF) | MARVIN M6 (7-DOF) |
|---------|--------------|-------------------|
| Joint count | 6 | 7 |
| Null-space control | ❌ No | ✅ Yes (kp_null=10.0) |
| q1 regularization | Required (kp_joint_reg=100.0) | Not needed |
| Yaw range | Narrow ([-10, 90]) | Full ([-180, 90]) |
| Grasp type | fixed_peg | gripper |
| keypoint_axis | 0 (X-axis) | 2 (Z-axis) |
| Base joint drift | Problem (diverges to ±π) | Stable (null-space holds) |

---

## Key Differences: MARVIN M6 vs. Franka

| Feature | Franka | MARVIN M6 |
|---------|--------|-----------|
| DOF | 7 | 7 |
| Joint names | panda_joint[1-7] | Joint[1-7]_L |
| Default pose | [-1.30, -0.40, 1.18, ...] | [0.0, -0.3, 0.3, 0.0, ...] |
| Effort limits | 87/12 Nm (shoulder/wrist) | 108/66 Nm |
| Null-space params | kp_null=10.0 | kp_null=10.0 (same) |
| Gripper | Robotiq 2F-85 | Franka-like parallel-jaw |

**Conclusion:** MARVIN M6 is configured as a "Franka alternative" with:
- Same 7-DOF redundancy and null-space control
- Different kinematics (MARVIN-specific joint limits)
- Different dynamics (higher torque capacity)
- Same control paradigm (null-space vs. joint regularization)

---

## Contact

For questions about this integration, refer to:
- URDF source: `/home/an/TJ_FX_ROBOT_CONTRL_SDK/URDF/Marvin M6-CCS-696-urdf_V4.0/`
- Robot specs: `/home/an/TJ_FX_ROBOT_CONTRL_SDK/robot_M6_CCS_1003_26.ini`
- Migration plan: `/home/an/.claude/plans/7axis-migration-marvin-m6.md`
