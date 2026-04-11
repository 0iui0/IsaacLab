---
name: UR10 Fixed-Peg Mutex Fix
description: Fix for mutex recursion error when training UR10 fixed-peg robots in Forge Assembly
type: feedback
---

**Issue:** UR10 training failed with `carb.tasking::Mutex::lock()` assertion error during environment setup.

**Root Cause:** Attempting to spawn child prims (peg, force_sensor) inside an active robot articulation prim caused Isaac Sim's tasking system to deadlock.

**Solution:** For fixed-peg robots (UR10/CR5), do NOT spawn separate peg prims. The EE link itself serves as the peg for:
- Collision detection
- Force sensing (via `get_link_incoming_joint_force()` on ee_body_idx)  
- Task success checking (held_pos = EE position)

**Key Insight:** The spawned peg prims were purely visual and not needed for task logic. Removing them entirely avoids the mutex error while preserving all functional behavior.

**Verification:**
- UR10 training completes successfully (no mutex errors)
- Franka training still works (no regression)
- UR10 reward: ~0.77 after 2 epochs (128 envs)
- Franka reward: ~89.47 after 2 epochs (128 envs)

**How to apply:** When adding new fixed-peg robots to Forge Assembly, rely on the EE link for collision/force sensing. Do not spawn separate peg prims inside the robot articulation.
