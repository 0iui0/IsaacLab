#!/usr/bin/env python3
"""
MARVIN M6 + Robotiq_2F140 自动化组装脚本

功能：
1. 使用已有的 MARVIN M6 USD 文件
2. 从 Omniverse 云端导入 Robotiq_2F140
3. 移除 Robotiq 的 Articulation Root
4. 创建 Fixed Joint 连接机械臂和夹爪
5. 保存组合 USD

使用方法：
    /isaac-sim/python.sh scripts/tools/assemble_marvin_robotiq.py \\
        <input_usd> <output_usd> \\
        --gripper-variant 2F_140
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="Assemble MARVIN M6 + Robotiq_2F140")
parser.add_argument("input", type=str, help="Path to MARVIN M6 USD file (already converted)")
parser.add_argument("output", type=str, help="Path to output combined USD file")
parser.add_argument(
    "--gripper-variant",
    type=str,
    default="2F_140",
    choices=["2F_140", "2F_85"],
    help="Robotiq gripper variant (default: 2F_140)",
)

# 添加 AppLauncher 参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import carb
import omni.kit.app
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

import isaaclab.sim as sim_utils


def copy_prim_tree(source_stage: Usd.Stage, source_path: Sdf.Path,
                   target_stage: Usd.Stage, target_path: Sdf.Path) -> bool:
    """
    Copy a prim subtree from source stage to target stage.

    Args:
        source_stage: Source USD stage
        source_path: Path to source prim
        target_stage: Target USD stage
        target_path: Path for target prim

    Returns:
        True if successful
    """
    source_prim = source_stage.GetPrimAtPath(source_path)
    if not source_prim.IsValid():
        print(f"[ERROR] Source prim not found: {source_path}")
        return False

    # Define target prim
    target_prim = UsdGeom.Xform.Define(target_stage, target_path)
    if not target_prim:
        print(f"[ERROR] Failed to define target prim: {target_path}")
        return False

    # Copy children recursively
    for child in source_prim.GetChildren():
        child_name = child.GetName()
        copy_prim_tree(source_stage, child.GetPath(), target_stage,
                      Sdf.Path(f"{target_path}/{child_name}"))

    return True


def remove_articulation_root(stage: Usd.Stage, prim_path: str) -> None:
    """
    Remove ArticulationRootAPI from a prim.

    Args:
        stage: USD stage
        prim_path: Path to prim with ArticulationRootAPI
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[WARN] Prim not found: {prim_path}")
        return

    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Remove(prim)
        print(f"[INFO] Removed ArticulationRootAPI from: {prim_path}")
    else:
        print(f"[INFO] No ArticulationRootAPI found at: {prim_path}")


def create_fixed_joint(stage: Usd.Stage, joint_path: str, body0_path: str, body1_path: str) -> None:
    """
    Create a FixedJoint between two bodies.

    Args:
        stage: USD stage
        joint_path: Path for the joint prim
        body0_path: Path to parent body
        body1_path: Path to child body
    """
    print(f"[INFO] Creating FixedJoint: {joint_path}")
    print(f"[INFO]   Body0 (parent): {body0_path}")
    print(f"[INFO]   Body1 (child): {body1_path}")

    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    fixed_joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])

    print(f"[INFO] FixedJoint created successfully")


def assemble_marvin_robotiq(usd_path: str, output_usd_path: str, gripper_variant: str = "2F_140") -> None:
    """
    Main assembly function: MARVIN M6 + Robotiq_2F140

    Args:
        usd_path: Path to MARVIN M6 USD (already converted)
        output_usd_path: Path to output combined USD
        gripper_variant: Robotiq gripper variant (2F_140 or 2F_85)
    """
    # Create output directory if needed
    output_dir = os.path.dirname(output_usd_path)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Open existing MARVIN USD
    print(f"[INFO] Opening MARVIN USD: {usd_path}")
    if not os.path.exists(usd_path):
        print(f"[ERROR] MARVIN USD not found: {usd_path}")
        return

    # Step 2: Create new stage in memory for assembly
    print("[INFO] Creating assembly stage...")
    stage = sim_utils.create_new_stage_in_memory()

    # Step 3: Copy MARVIN prims to /World/MarvinM6
    print("[INFO] Importing MARVIN M6...")
    marvin_source_stage = Usd.Stage.Open(usd_path)
    if not marvin_source_stage:
        print(f"[ERROR] Failed to open MARVIN USD: {usd_path}")
        return

    # Copy all prims from source to /World/MarvinM6
    copy_prim_tree(marvin_source_stage, Sdf.Path("/"), stage, Sdf.Path("/World/MarvinM6"))
    print("[INFO] MARVIN M6 imported to /World/MarvinM6")

    # Step 4: Import Robotiq_2F140 from Omniverse asset library
    # Use Isaac Sim asset root path or direct S3 URL
    try:
        from isaacsim.storage.native import get_assets_root_path
        asset_root_path = get_assets_root_path()
        print(f"[INFO] Asset root path: {asset_root_path}")
    except Exception as e:
        print(f"[WARN] Could not get asset root path: {e}")
        asset_root_path = None

    # Try multiple paths for Robotiq
    if asset_root_path:
        robotiq_paths_to_try = [
            asset_root_path + "/Isaac/Robots/Robotiq/2F-140/Robotiq_2F_140_config.usd",
            asset_root_path + "/Isaac/Robots/Robotiq/2F-140/Robotiq_2F_140.usd",
        ]
    else:
        robotiq_paths_to_try = []

    # Add direct S3 URL as fallback (this is what Isaac Sim 5.1 uses)
    if gripper_variant == "2F_85":
        robotiq_paths_to_try.append(
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_config.usd"
        )
    else:
        robotiq_paths_to_try.append(
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Robotiq/2F-140/Robotiq_2F_140_config.usd"
        )

    print(f"[INFO] Importing Robotiq from:")
    robotiq_source_stage = None
    for robotiq_path in robotiq_paths_to_try:
        print(f"  Trying: {robotiq_path}")
        robotiq_source_stage = Usd.Stage.Open(robotiq_path)
        if robotiq_source_stage:
            print(f"  ✓ Success!")
            break
        else:
            print(f"  ✗ Failed")

    if not robotiq_source_stage:
        print(f"[ERROR] Failed to open Robotiq USD from all paths")
        return

    print(f"[INFO] Robotiq loaded from: {robotiq_path}")

    # Copy Robotiq prims to /World/Robotiq2F140
    copy_prim_tree(robotiq_source_stage, Sdf.Path("/"), stage, Sdf.Path("/World/Robotiq2F140"))
    print(f"[INFO] Robotiq imported to: /World/Robotiq2F140")

    # Step 5: Remove Robotiq's Articulation Root (CRITICAL!)
    print("[INFO] Removing Robotiq Articulation Root...")
    # Try common Robotiq prim paths
    possible_robotiq_paths = [
        "/World/Robotiq2F140",
        "/World/Robotiq2F140/robotiq_base_link",
        "/World/Robotiq2F140/base_link",
    ]
    for path in possible_robotiq_paths:
        remove_articulation_root(stage, path)

    # Step 6: Create Fixed Joint between MARVIN ee_link and Robotiq base
    marvin_ee_link = "/World/MarvinM6/Link7_L"
    robotiq_base_link = "/World/Robotiq2F140/robotiq_base_link"

    print(f"[INFO] Connecting {marvin_ee_link} to {robotiq_base_link}")

    create_fixed_joint(
        stage,
        "/World/MarvinM6ToRobotiqFixedJoint",
        marvin_ee_link,
        robotiq_base_link,
    )

    # Step 7: Save combined USD
    print(f"[INFO] Saving combined USD to: {output_usd_path}")
    stage.GetRootLayer().Export(output_usd_path)

    # Verify file was created
    if os.path.exists(output_usd_path):
        file_size = os.path.getsize(output_usd_path)
        print(f"[INFO] Assembly complete! File size: {file_size / 1024:.1f} KB")

        if file_size < 10000:
            print("[WARN] File size is very small - conversion may have failed!")
    else:
        print("[ERROR] Output file was not created!")


def main():
    """Main entry point."""
    print("=" * 80)
    print("MARVIN M6 + Robotiq_2F140 Assembly Script")
    print("=" * 80)

    # Validate input USD
    if not os.path.exists(args_cli.input):
        print(f"[ERROR] USD file not found: {args_cli.input}")
        sys.exit(1)

    # Run assembly
    assemble_marvin_robotiq(args_cli.input, args_cli.output, args_cli.gripper_variant)

    print("=" * 80)
    print("Assembly complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
    # Close simulation app
    simulation_app.close()
