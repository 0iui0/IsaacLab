#!/usr/bin/env python3
"""
Peg-in-Hole Assembly using Traditional Impedance Control

This script implements a classical impedance control strategy for peg-in-hole assembly,
without using reinforcement learning. It can be used to:
1. Understand the control requirements before training RL agents
2. Debug and tune control parameters in simulation
3. Transfer to real hardware after validation

Key Features:
- State machine-based assembly sequence
- Task-space impedance control (same as FORGE)
- Force feedback for contact detection
- Spiral search strategy for peg-hole alignment
- Parameter tuning interface for real-world deployment

Reference: FORGE RL framework (https://arxiv.org/abs/xxxx)
"""

import argparse
import math
import numpy as np
import torch
from datetime import datetime

from isaaclab.app import AppLauncher
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab_tasks.manager_based.manipulation.common.push import PushCfg
from isaaclab_tasks.direct.factory.forge_env import ForgeEnv, ForgeEnvCfg

# Import Forge utilities
from isaaclab_tasks.direct.factory import factory_control
from isaaclab_tasks.direct.factory import factory_utils


class TraditionalPegInHoleController:
    """
    Traditional impedance controller for peg-in-hole assembly.

    Uses a state machine to execute assembly phases:
    1. APPROACH: Move to above the hole
    2. ALIGN: Fine-tune position/orientation
    3. INSERT: Lower into the hole
    4. COMPLETE: Task done

    The controller uses the same impedance control parameters as FORGE:
    - Kp = 565 (position stiffness)
    - Kp_rot = 28 (rotation stiffness)
    - Kd = 2*sqrt(Kp) (critical damping)
    """

    def __init__(self, cfg, num_envs, device, sim_dt):
        """
        Initialize the controller.

        Args:
            cfg: Environment configuration
            num_envs: Number of parallel environments
            device: Device to create tensors on
            sim_dt: Simulation time step
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.sim_dt = sim_dt

        # Assembly phases
        self.PHASE_APPROACH = 0
        self.PHASE_ALIGN = 1
        self.PHASE_INSERT = 2
        self.PHASE_COMPLETE = 3

        # State tracking
        self.current_phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.phase_timer = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.insertion_depth = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.contact_detected = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.insertion_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Control parameters (same as FORGE)
        self.Kp = torch.tensor(cfg.ctrl.default_task_prop_gains, device=device).repeat(num_envs, 1)
        self.Kd = factory_utils.get_deriv_gains(
            self.Kp,
            cfg.ctrl.reset_rot_deriv_scale
        )

        # Force thresholds
        self.contact_force_threshold = 5.0  # N
        self.insertion_force_threshold = 15.0  # N
        self.success_force_threshold = 25.0  # N

        # Position targets
        self.approach_height = 0.05  # 5cm above hole
        self.insertion_depth_target = 0.02  # 2cm insertion

        # Spiral search parameters (for alignment)
        self.spiral_amplitude = 0.002  # 2mm radius
        self.spiral_frequency = 2.0  # Hz
        self.spiral_max_time = 5.0  # seconds

    def compute_control_signals(
        self,
        env,
        ctrl_target_pos,
        ctrl_target_quat,
        gripper_dof_pos
    ):
        """
        Compute joint torques using impedance control.

        This is the same control law used in FORGE:
        1. Compute pose error
        2. Apply PD control law in task space
        3. Map to joint space via Jacobian transpose
        4. Add null-space control

        Args:
            env: Forge environment instance
            ctrl_target_pos: Target fingertip position [num_envs, 3]
            ctrl_target_quat: Target fingertip orientation [num_envs, 4]
            gripper_dof_pos: Gripper joint position

        Returns:
            joint_torques: Joint torques [num_envs, 7]
        """
        # Get current state
        fingertip_pos = env.fingertip_midpoint_pos
        fingertip_quat = env.fingertip_midpoint_quat
        fingertip_linvel = env.fingertip_midpoint_linvel
        fingertip_angvel = env.fingertip_midpoint_angvel
        jacobian = env.fingertip_midpoint_jacobian
        arm_mass_matrix = env.arm_mass_matrix

        # Compute joint torques (impedance control)
        joint_torques, task_wrench = factory_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=env.joint_pos,
            dof_vel=env.joint_vel,
            fingertip_midpoint_pos=fingertip_pos,
            fingertip_midpoint_quat=fingertip_quat,
            fingertip_midpoint_linvel=fingertip_linvel,
            fingertip_midpoint_angvel=fingertip_angvel,
            jacobian=jacobian,
            arm_mass_matrix=arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=ctrl_target_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_quat,
            task_prop_gains=self.Kp,
            task_deriv_gains=self.Kd,
            device=self.device,
            dead_zone_thresholds=env.dead_zone_thresholds,
        )

        return joint_torques, task_wrench

    def get_force_sensor_data(self, env):
        """Get force sensor readings."""
        return env.force_sensor_smooth[:, 0:3]  # Only use force (3D), not torque (3D)

    def check_contact(self, env):
        """Detect contact based on force sensor."""
        force = self.get_force_sensor_data(env)
        force_norm = torch.norm(force, p=2, dim=-1)
        return force_norm > self.contact_force_threshold

    def check_insertion_depth(self, env):
        """Check if peg is inserted deep enough."""
        # Compute insertion depth (distance in -z direction)
        peg_bottom_pos = env.held_pos[..., 2]  # z-coordinate
        hole_top_pos = env.fixed_pos[..., 2] - env.cfg_task.fixed_asset_cfg.height

        insertion_depth = hole_top_pos - peg_bottom_pos
        return insertion_depth > self.insertion_depth_target

    def check_success(self, env):
        """Check if assembly is successful."""
        force = self.get_force_sensor_data(env)
        force_norm = torch.norm(force, p=2, dim=-1)

        # Success criteria: high force means peg is firmly inserted
        return force_norm > self.success_force_threshold

    def update_state_machine(
        self,
        env,
        dt,
        obs
    ):
        """
        Update assembly state machine based on current state.

        State transitions:
        APPROACH → ALIGN: when near target position
        ALIGN → INSERT: when aligned and contact detected
        INSERT → COMPLETE: when insertion depth reached
        """
        num_envs = env.num_envs

        # Get current state
        peg_pos = env.held_pos  # [num_envs, 3]
        peg_quat = env.held_quat  # [num_envs, 4]
        hole_pos = env.fixed_pos  # [num_envs, 3]

        # Check contact
        self.contact_detected = self.check_contact(env)

        # Check insertion depth
        self.insertion_success = self.check_insertion_depth(env)

        # State transitions
        for env_id in range(num_envs):
            current_phase = self.current_phase[env_id].item()

            if current_phase == self.PHASE_APPROACH:
                # Check if we're close enough to switch to ALIGN
                dist_to_hole = torch.norm(peg_pos[env_id, :2] - hole_pos[env_id, :2])

                if dist_to_hole < 0.01:  # Within 1cm
                    self.current_phase[env_id] = self.PHASE_ALIGN
                    self.phase_timer[env_id] = 0.0

            elif current_phase == self.PHASE_ALIGN:
                # Check if aligned and in contact
                force = self.get_force_sensor_data(env)[env_id]
                force_norm = torch.norm(force)

                # Check alignment (peg roughly above hole)
                lateral_error = torch.norm(peg_pos[env_id, :2] - hole_pos[env_id, :2])

                if lateral_error < 0.002 and self.contact_detected[env_id]:
                    self.current_phase[env_id] = self.PHASE_INSERT
                    self.phase_timer[env_id] = 0.0
                elif self.phase_timer[env_id] > self.spiral_max_time:
                    # Alignment failed, retry
                    self.current_phase[env_id] = self.PHASE_APPROACH
                    self.phase_timer[env_id] = 0.0

            elif current_phase == self.PHASE_INSERT:
                # Check if insertion is complete
                if self.insertion_success[env_id]:
                    self.current_phase[env_id] = self.PHASE_COMPLETE
                    self.phase_timer[env_id] = 0.0

            elif current_phase == self.PHASE_COMPLETE:
                # Task done
                pass

    def compute_target_pose(
        self,
        env,
        phase
    ):
        """
        Compute target pose for each assembly phase.

        Args:
            env: Forge environment instance
            phase: Current assembly phase

        Returns:
            target_pos: Target position [num_envs, 3]
            target_quat: Target orientation [num_envs, 4]
        """
        num_envs = env.num_envs
        device = env.device

        # Get current state
        peg_pos = env.held_pos  # [num_envs, 3]
        peg_quat = env.held_quat  # [num_envs, 4]
        hole_pos = env.fixed_pos  # [num_envs, 3]

        if phase == self.PHASE_APPROACH:
            # Target: 5cm above the hole center
            target_pos = hole_pos.clone()
            target_pos[:, 2] += self.approach_height

            # Orientation: vertical
            target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

        elif phase == self.PHASE_ALIGN:
            # Strategy: Spiral search to find hole
            # Current position + small spiral motion

            # Spiral parameters
            t = self.phase_timer  # [num_envs]

            # Spiral motion: x = A*cos(ωt), y = A*sin(ωt)
            spiral_radius = self.spiral_amplitude * (1.0 - t / self.spiral_max_time)

            # Current lateral offset
            lateral_offset_x = spiral_radius * torch.cos(2 * np.pi * self.spiral_frequency * t)
            lateral_offset_y = spiral_radius * torch.sin(2 * np.pi * self.spiral_frequency * t)

            # Target position: hole center + spiral offset
            target_pos = hole_pos.clone()
            target_pos[:, 0] += lateral_offset_x
            target_pos[:, 1] += lateral_offset_y
            target_pos[:, 2] = hole_pos[:, 2] + 0.01  # Slightly above

            # Orientation: maintain vertical
            target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

        elif phase == self.PHASE_INSERT:
            # Target: slowly lower into the hole
            target_pos = peg_pos.clone()
            target_pos[:, 2] -= 0.001  # Lower by 1mm per step

            # Orientation: maintain
            target_quat = peg_quat

        else:  # COMPLETE
            # Hold current pose
            target_pos = peg_pos
            target_quat = peg_quat

        return target_pos, target_quat

    def run_episode(self, env, max_steps=1000):
        """
        Run a single assembly episode.

        Args:
            env: Forge environment instance
            max_steps: Maximum number of steps

        Returns:
            success: Boolean array [num_envs] indicating success/failure
            log_data: Dictionary with episode statistics
        """
        num_envs = env.num_envs
        device = env.device

        # Reset environment
        obs, _ = env.reset()

        # Reset controller state
        self.current_phase[:] = self.PHASE_APPROACH
        self.phase_timer[:] = 0.0

        # Logging
        phase_log = []
        force_log = []
        pose_error_log = []

        for step in range(max_steps):
            # Update state machine
            self.update_state_machine(env, env.sim.dt, obs)

            # Compute target pose based on current phase
            target_pos, target_quat = self.compute_target_pose(env, self.current_phase)

            # Compute control signals (impedance control)
            joint_torques, task_wrench = self.compute_control_signals(
                env, target_pos, target_quat, gripper_dof_pos=0.0
            )

            # Apply torques
            env._robot.set_joint_effort_targets(joint_torques)

            # Step simulation
            env._scene.write_data_to_sim()
            env.sim.step()
            env._scene.update(dt=env.sim_dt)

            # Update timer
            self.phase_timer += env.sim.dt

            # Logging
            phase_log.append(self.current_phase.clone())
            force_log.append(self.get_force_sensor_data(env).norm(dim=-1))

            # Check completion
            if (self.current_phase == self.PHASE_COMPLETE).all():
                break

        # Evaluate success
        success = self.check_success(env)

        # Compile statistics
        log_data = {
            'num_steps': step + 1,
            'success': success,
            'phase_history': torch.stack(phase_log),  # [max_steps, num_envs]
            'force_history': torch.stack(force_log),  # [max_steps, num_envs]
        }

        return success, log_data


def main():
    """
    Main function to run traditional impedance control experiments.
    """

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run traditional impedance control for peg-in-hole assembly"
    )
    parser.add_argument("--task", type=str, default="Isaac-Forge-PegInsert-Direct-v0",
                       help="Name of the task")
    argparse = AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    # Clear sys.argv for Hydra
    import sys
    sys.argv = [sys.argv[0]] + hydra_args

    # Launch Isaac Sim
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    """Rest everything follows."""

    import gymnasium as gym
    import os

    from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
    from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskPegInsertCfg

    # Create environment configuration
    env_cfg = ForgeTaskPegInsertCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else "cuda:0"

    # Log directory
    log_dir = os.path.join("logs", "traditional_impedance_control", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    print(f"[INFO] Logging to: {log_dir}")
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Num environments: {env_cfg.scene.num_envs}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Create controller
    controller = TraditionalPegInHoleController(
        cfg=env.cfg,
        num_envs=env.num_envs,
        device=env.device,
        sim_dt=env.physics_dt
    )

    # Run episodes
    num_episodes = 10
    success_rates = []

    for episode in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*50}")

        success, log_data = controller.run_episode(env, max_steps=1000)

        success_rate = success.float().mean().item()
        success_rates.append(success_rate)

        print(f"Success rate: {success_rate:.2%}")
        print(f"Num steps: {log_data['num_steps']}")

        # Log statistics
        if episode == 0:
            print(f"\nPhase distribution:")
            for step, phases in enumerate(log_data['phase_history']):
                phase_names = ['APPROACH', 'ALIGN', 'INSERT', 'COMPLETE']
                for phase_idx in range(4):
                    count = (phases == phase_idx).sum().item()
                    print(f"  Step {step}: {phase_names[phase_idx]}: {count}/{env.num_envs}")

    print(f"\n{'='*50}")
    print(f"Overall success rate: {np.mean(success_rates):.2%}")
    print(f"{'='*50}")

    # Save results
    results = {
        'success_rates': success_rates,
        'mean_success_rate': np.mean(success_rates),
        'config': {
            'approach_height': controller.approach_height,
            'insertion_depth_target': controller.insertion_depth_target,
            'contact_force_threshold': controller.contact_force_threshold,
            'insertion_force_threshold': controller.insertion_force_threshold,
            'success_force_threshold': controller.success_force_threshold,
            'spiral_amplitude': controller.spiral_amplitude,
            'spiral_frequency': controller.spiral_frequency,
        }
    }

    import json
    with open(os.path.join(log_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {log_dir}/results.json")

    # Close simulation
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
