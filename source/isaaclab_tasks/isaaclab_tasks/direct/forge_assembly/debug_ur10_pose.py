#!/usr/bin/env python3
"""Debug script to print UR10 joint angles and EE pose at reset."""

import torch
import math

from isaaclab.sim import SimulationContext

from .forge_env_cfg import UR10ForgeTaskPegInsertCfg
from .forge_env import ForgeEnv


def quaternion_to_euler(q):
    """Convert quaternion to Euler angles (roll, pitch, yaw) in radians."""
    x, y, z, w = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def main():
    print("=" * 60)
    print("UR10 Reset Pose Debug Script")
    print("=" * 60)

    # Print configuration
    cfg = UR10ForgeTaskPegInsertCfg()

    print("\n--- Configuration ---")
    print(f"reset_joints: {cfg.ctrl.reset_joints}")
    print(f"default_arm_joint_pos: {cfg.robot_profile.default_arm_joint_pos}")
    print(f"reset_arm_joint_pos: {cfg.robot_profile.reset_arm_joint_pos}")
    print(f"null_space_default_pos: {cfg.robot_profile.null_space_default_pos}")

    print("\n--- Task hand_init_orn ---")
    print(f"hand_init_orn: {cfg.task.hand_init_orn} (radians)")
    print(f"hand_init_orn_deg: {[x*180/math.pi for x in cfg.task.hand_init_orn]} (degrees)")
    print(f"hand_init_pos: {cfg.task.hand_init_pos}")

    print("\n--- UR10 Joint Names ---")
    ur10_joints = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    for i, joint_name in enumerate(ur10_joints):
        print(f"  q{i+1} = {joint_name}")

    print("\n--- Zero pose reference (all joints at 0) ---")
    print("  q1=0: shoulder_pan - base rotation (around Z)")
    print("  q2=0: shoulder_lift - upper arm horizontal forward")
    print("  q3=0: elbow - forearm horizontal forward (arm fully extended)")
    print("  q4=0: wrist_1 - wrist aligned with forearm")
    print("  q5=0: wrist_2 - wrist aligned")
    print("  q6=0: wrist_3 - ee_link X-axis points forward (horizontal)")

    print("\n--- Expected EE orientation at zero pose ---")
    print("  ee_link X-axis: points forward (horizontal, along robot arm)")
    print("  ee_link Y-axis: points to the side (horizontal)")
    print("  ee_link Z-axis: points upward (vertical)")

    print("\n--- To make peg (X-axis) point downward ---")
    print("  Need to rotate EE so X-axis aligns with world -Z")
    print("  This requires wrist rotation (q4, q5, q6)")

    print("\n" + "=" * 60)
    print("Running environment to get actual reset pose...")
    print("=" * 60)

    # Create simulation context
    sim_cfg = cfg.sim
    sim_cfg.device = "cuda:0"
    sim = SimulationContext(sim_cfg)

    # Set device
    device = sim.device

    # Create environment
    env = ManagerBasedRLEnvCfg
    env = ForgeEnv(cfg=cfg)

    # Reset environment to get initial state
    env.reset()

    # Get robot articulation
    robot = env.scene["robot"]

    # Get joint positions
    joint_positions = robot.data.joint_pos[:, :6].cpu().numpy()[0]

    print("\n--- Actual Joint Angles at Reset ---")
    for i, (joint_name, joint_pos) in enumerate(zip(ur10_joints, joint_positions)):
        joint_pos_deg = joint_pos * 180 / math.pi
        print(f"  q{i+1} ({joint_name}): {joint_pos:.4f} rad ({joint_pos_deg:.2f}°)")

    # Get EE pose
    ee_body_idx = robot.find_bodies("ee_link")[0][0]
    ee_pose = robot.data.body_state_w[0, ee_body_idx]
    ee_pos = ee_pose[:3].cpu().numpy()
    ee_quat = ee_pose[3:7].cpu().numpy()

    print("\n--- Actual EE Pose at Reset ---")
    print(f"  Position: {ee_pos}")
    print(f"  Quaternion: {ee_quat}")

    # Convert to Euler angles
    roll, pitch, yaw = quaternion_to_euler(ee_quat)
    roll_deg = roll * 180 / math.pi
    pitch_deg = pitch * 180 / math.pi
    yaw_deg = yaw * 180 / math.pi

    print(f"  Euler angles (roll, pitch, yaw):")
    print(f"    Roll:  {roll:.4f} rad ({roll_deg:.2f}°)")
    print(f"    Pitch: {pitch:.4f} rad ({pitch_deg:.2f}°)")
    print(f"    Yaw:   {yaw:.4f} rad ({yaw_deg:.2f}°)")

    # Compute EE axes directions
    w, x, y, z = ee_quat

    # Rotation matrix from quaternion
    R = torch.tensor([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
    ])

    # EE axes in world frame (ee_link: X=forward, Y=side, Z=up)
    ee_x_axis = R[:, 0].cpu().numpy()  # Points along peg axis
    ee_y_axis = R[:, 1].cpu().numpy()
    ee_z_axis = R[:, 2].cpu().numpy()

    print("\n--- EE Link Axes Directions (in world frame) ---")
    print(f"  X-axis (peg direction): {ee_x_axis}")
    print(f"  Y-axis: {ee_y_axis}")
    print(f"  Z-axis: {ee_z_axis}")

    print("\n--- Expected for peg pointing DOWN ---")
    print("  X-axis should be: [0, 0, -1] (pointing down, parallel to world -Z)")
    print("  Y-axis should be: horizontal (perpendicular to world Z)")
    print("  Z-axis should be: horizontal (perpendicular to world Z)")

    # Check if X-axis is pointing down
    x_dot_down = ee_x_axis @ torch.tensor([0, 0, -1]).cpu().numpy()
    print(f"\n  X-axis dot [0,0,-1]: {x_dot_down:.4f} (1.0 = perfectly aligned)")

    # Check if Z-axis is horizontal (should be perpendicular to world Z)
    z_axis_vertical = abs(ee_z_axis[2])
    print(f"  |Z-axis dot [0,0,1]|: {z_axis_vertical:.4f} (0.0 = horizontal, 1.0 = vertical)")

    print("\n" + "=" * 60)
    print("Please compare with GUI observation:")
    print("1. Do the joint angles match what you see?")
    print("2. Is the ee_link X-axis (red) pointing down?")
    print("3. If not, which joint (q3/q4/q5) appears off by ~90°?")
    print("=" * 60)


if __name__ == "__main__":
    main()
