# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forge assembly environment with IK-based reset matching direct forge.

This environment extends the base ManagerBasedRLEnv to add:
1. IK-based reset positioning (0.25s simulation time)
2. Gripper closing period (0.25s simulation time)

These settling periods are critical for establishing proper physics-based
friction grip, matching the direct forge implementation exactly.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


class ForgeAssemblyEnv(ManagerBasedRLEnv):
    """Forge assembly environment with physics-based reset.

    This environment adds the gripper settling period from direct forge
    to the manager-based framework. This ensures the peg is properly
    gripped through physics before the episode starts.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Store reference to robot articulation for IK operations
        self._robot = self.scene["robot"]
        self._peg = self.scene["peg"]

        # Get gripper joint indices
        self._gripper_joint_names = self._get_gripper_joint_names()

        # Configure gripper settling: 0.25s at physics_dt
        # Direct forge uses while grasp_time < 0.25 with sim.get_physics_dt()
        self._gripper_settling_steps = max(1, int(0.25 / self.physics_dt))

    def _get_gripper_joint_names(self) -> list[str]:
        """Get gripper joint names based on robot type."""
        all_joint_names = self._robot.data.joint_names

        # Franka Panda pattern: panda_finger_joint1, panda_finger_joint2
        finger_joints = [name for name in all_joint_names if "finger" in name.lower()]
        if finger_joints:
            return finger_joints

        # Fallback: use last 2 joints (typical for parallel grippers)
        return list(all_joint_names[-2:])

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments with gripper settling period.

        This override adds the gripper settling period after the standard reset
        to establish proper physics-based friction grip, matching direct forge.
        """
        # Call parent reset to run event terms (including reset_peg_to_ee)
        super()._reset_idx(env_ids)

        # Add gripper settling period to establish friction grip
        # Direct forge: 0.25s of close_gripper_in_place() simulation
        self._settle_gripper(env_ids)

    def _settle_gripper(self, env_ids: torch.Tensor):
        """Simulate gripper closing for 0.25s to establish friction grip.

        Direct forge uses close_gripper_in_place() for 0.25s after IK positioning.
        This replicates that behavior to ensure the peg is properly gripped.
        """
        # Get gripper joint indices
        gripper_joint_ids, _ = self._robot.find_joints(self._gripper_joint_names)

        # Set gripper position target to closed (0.0)
        # This matches direct forge's ctrl_target_gripper_dof_pos=0.0
        gripper_target = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device, dtype=torch.float32
        )

        # Apply gripper closing for settling period
        # Direct forge: while grasp_time < 0.25: close_gripper_in_place()
        for _ in range(self._gripper_settling_steps):
            # Set gripper target
            self._robot.set_joint_position_target(gripper_target, env_ids=env_ids)

            # Write to simulator
            self.scene.write_data_to_sim()

            # Step simulation without rendering
            self.sim.step(render=False)

            # Update scene
            self.scene.update(dt=self.physics_dt)
