#!/usr/bin/env python3
"""Keyboard control script for UR10 manual positioning.

Run with:
  ./isaaclab.sh -p source/isaaclab_tasks/.../keyboard_control_ur10.py
"""

# Launch Isaac Sim Simulator first.
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard control for UR10")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse isaac-sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import torch

from isaaclab_tasks.direct.forge_assembly.forge_env_cfg import UR10ForgeTaskPegInsertCfg
from isaaclab_tasks.direct.forge_assembly.forge_env import ForgeEnv


def get_key():
    """Get a single keypress from the keyboard."""
    import sys
    import termios
    import tty
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key


def print_instructions():
    """Print control instructions."""
    print("\n" + "=" * 70)
    print("UR10 Manual Control - Find correct reset pose for peg insertion")
    print("=" * 70)
    print("\nGOAL: Make peg (red cylinder) point VERTICALLY DOWN toward hole")
    print("      ee_link X-axis should align with [0, 0, -1]")
    print("      ee_link flange plane should be parallel to XY plane")
    print("\nRotation control - wrist/Elbow joints only (step: 5):")
    print("  U/O     : elbow_joint (q3) - controls arm extension")
    print("  R/F     : wrist_1_joint (q4) - controls approach angle")
    print("  T/G     : wrist_2_joint (q5) - controls pitch")
    print("  Y/H     : wrist_3_joint (q6) - controls final orientation")
    print("\n  Q       : Quit and print joint angles")
    print("  SPACE   : Print current pose")
    print("=" * 70)


def euler_from_quat(q):
    """Convert quaternion to Euler angles (roll, pitch, yaw)."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1, min(1, sinp)))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def compute_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q
    return torch.tensor([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
    ])


def main():
    print_instructions()

    # Create environment - this will spawn robot at reset pose
    # Note: SimulationContext is created automatically by ForgeEnv
    env_cfg = UR10ForgeTaskPegInsertCfg()
    env = ForgeEnv(cfg=env_cfg)

    # Get robot and EE body index
    robot = env.scene["robot"]
    ee_body_idx = robot.find_bodies("ee_link")[0][0]

    # Get current joint positions (start from current reset pose)
    joint_positions = robot.data.joint_pos[0, :6].clone().tolist()
    joint_names = ["shoulder_pan (q1)", "shoulder_lift (q2)", "elbow (q3)",
                   "wrist_1 (q4)", "wrist_2 (q5)", "wrist_3 (q6)"]

    rot_step = math.radians(5)  # 5 degrees for rotation

    print("\nStarting pose (from reset_joints config):")
    for name, joint in zip(joint_names, joint_positions):
        print(f"  {name}: {joint:.4f} rad ({joint*180/math.pi:7.2f})")

    print("\nAdjusting elbow and wrist joints (q3, q4, q5, q6)...")
    print("Watch GUI: peg (red cylinder) should point DOWN")
    print("Press keys to adjust, Q to quit\n")

    while True:
        key = get_key().lower()

        if key == 'q':
            break
        elif key == 'u':
            joint_positions[2] += rot_step  # elbow
        elif key == 'o':
            joint_positions[2] -= rot_step
        elif key == 'r':
            joint_positions[3] += rot_step  # wrist_1
        elif key == 'f':
            joint_positions[3] -= rot_step
        elif key == 't':
            joint_positions[4] += rot_step  # wrist_2
        elif key == 'g':
            joint_positions[4] -= rot_step
        elif key == 'y':
            joint_positions[5] += rot_step  # wrist_3
        elif key == 'h':
            joint_positions[5] -= rot_step
        elif key == ' ':
            # Print current pose
            ee_pose = robot.data.body_state_w[0, ee_body_idx]
            ee_pos = ee_pose[:3].cpu().numpy()
            ee_quat = ee_pose[3:7].cpu().numpy()
            r, p, y = euler_from_quat(ee_quat)
            R = compute_rotation_matrix(ee_quat)
            ee_x = R[:, 0].cpu().numpy()
            ee_z = R[:, 2].cpu().numpy()
            x_dot_down = ee_x @ torch.tensor([0.0, 0.0, -1.0]).cpu().numpy()

            print(f"\nCurrent pose:")
            print(f"  Position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
            print(f"  Euler: roll={r*180/math.pi:7.1f}, pitch={p*180/math.pi:7.1f}, yaw={y*180/math.pi:7.1f}")
            print(f"  X-axis (peg): [{ee_x[0]:7.4f}, {ee_x[1]:7.4f}, {ee_x[2]:7.4f}]")
            print(f"  Z-axis:       [{ee_z[0]:7.4f}, {ee_z[1]:7.4f}, {ee_z[2]:7.4f}]")
            print(f"  Xdown: {x_dot_down:.4f} (1.0=pointing down)")

            print("\n  Joint angles:")
            for name, joint in zip(joint_names, joint_positions):
                print(f"    {name}: {joint:.4f} rad ({joint*180/math.pi:7.2f})")
            continue
        else:
            continue

        # Convert to tensor and apply
        joint_pos_tensor = torch.tensor(joint_positions, device=env.device).unsqueeze(0)
        joint_vel = torch.zeros_like(joint_pos_tensor)

        robot.write_joint_state_to_sim(joint_pos_tensor, joint_vel)
        robot.set_joint_position_target(joint_pos_tensor)

        # Step simulation
        for _ in range(10):
            env.step_sim_no_action()

        # Print current state inline
        ee_pose = robot.data.body_state_w[0, ee_body_idx]
        ee_quat_new = ee_pose[3:7].cpu().numpy()
        R = compute_rotation_matrix(ee_quat_new)
        ee_x = R[:, 0].cpu().numpy()
        x_dot_down = ee_x @ torch.tensor([0.0, 0.0, -1.0]).cpu().numpy()

        print(f"\rXdown: {x_dot_down:7.4f}  ", end='', flush=True)

    # Print final joint angles
    final_joints = robot.data.joint_pos[0, :6].cpu().numpy()

    print("\n\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print("\nJoint angles (q1-q6):")
    for name, joint in zip(joint_names, final_joints):
        print(f"  {name}: {joint:.4f} rad ({joint*180/math.pi:7.2f})")

    print("\nFor forge_env_cfg.py UR10 ctrl override:")
    print(f"  reset_joints={[round(float(j), 4) for j in final_joints]}")

    # Final EE pose
    ee_pose = robot.data.body_state_w[0, ee_body_idx]
    ee_pos = ee_pose[:3].cpu().numpy()
    ee_quat = ee_pose[3:7].cpu().numpy()
    r, p, y = euler_from_quat(ee_quat)
    R = compute_rotation_matrix(ee_quat)
    ee_x = R[:, 0].cpu().numpy()
    ee_z = R[:, 2].cpu().numpy()
    x_dot_down = ee_x @ torch.tensor([0.0, 0.0, -1.0]).cpu().numpy()

    print("\nFinal EE pose:")
    print(f"  Position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"  Euler: roll={r*180/math.pi:.1f}, pitch={p*180/math.pi:.1f}, yaw={y*180/math.pi:.1f}")
    print(f"  X-axis (peg): [{ee_x[0]:.4f}, {ee_x[1]:.4f}, {ee_x[2]:.4f}]")
    print(f"  Z-axis:       [{ee_z[0]:.4f}, {ee_z[1]:.4f}, {ee_z[2]:.4f}]")
    print(f"  Xdown: {x_dot_down:.4f} (1.0 = perfectly pointing down)")
    print("=" * 70)

    if x_dot_down > 0.99:
        print("SUCCESS! Peg is pointing down!")
    else:
        print(f"Note: Xdown = {x_dot_down:.4f}, may need further adjustment")
    print("=" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()
