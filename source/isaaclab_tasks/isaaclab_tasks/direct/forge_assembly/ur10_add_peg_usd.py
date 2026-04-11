#!/usr/bin/env python3
"""Create a UR10 with peg USD file using the ISAACLAB_NUCLEUS path.

This script creates a new USD file with UR10 + peg by:
1. Loading the UR10 from ISAACLAB_NUCLEUS_DIR (if available locally)
2. Or creating a simple cylinder peg USD that can be spawned with FixedJoint

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/ur10_add_peg_usd.py
"""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import os
from pxr import Usd, UsdGeom, Gf, PhysxSchema, UsdPhysics, UsdShade, Sdf


def create_peg_only_usd(output_path: str, radius: float = 0.0125, height: float = 0.0625):
    """Create a standalone peg USD that can be attached to UR10 EE link.

    This is simpler than modifying the UR10 USD and allows runtime attachment.
    """
    print(f"[INFO] Creating peg USD at: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create new stage
    stage = Usd.Stage.CreateNew(output_path)
    stage.SetDefaultPrim(stage.DefinePrim("/Peg", "Xform"))

    # Create peg root Xform
    peg_root = UsdGeom.Xform.Define(stage, "/Peg")

    # Create cylinder mesh
    cylinder = UsdGeom.Cylinder.Define(stage, "/Peg/cylinder")
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.AddTranslateOp().Set((0.0, 0.0, -height / 2))

    # Apply Collision API (not RigidBody - will be attached via FixedJoint)
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())

    # Apply PhysX collision properties
    physx_coll = PhysxSchema.PhysxCollisionAPI.Apply(cylinder.GetPrim())
    physx_coll.CreateContactOffsetAttr(0.005)
    physx_coll.CreateRestOffsetAttr(0.0)

    # Add visual material (brown/orange color)
    material = UsdShade.Material.Define(stage, "/Peg/Material")
    shader = UsdShade.Shader.Define(stage, "/Peg/Material/Shader")
    shader.CreateIdAttr("PreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.6, 0.4, 0.2))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)

    # Bind material
    UsdShade.MaterialBindingAPI.Apply(cylinder.GetPrim())
    UsdShade.MaterialBindingAPI(cylinder.GetPrim()).Bind(material)

    # Save
    stage.GetRootLayer().Save()
    print(f"[INFO] Peg USD created: {output_path}")

    # Print hierarchy
    print("\n[INFO] Prim hierarchy:")
    for prim in stage.Traverse():
        print(f"  {prim.GetPath()}")

    return True


def main():
    # Output path for the peg USD
    output_dir = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/assets"
    output_path = os.path.join(output_dir, "ur10_peg_25mm.usd")

    print("=" * 60)
    print("UR10 Peg USD Creator")
    print("=" * 60)
    print("\nNote: Modifying the UR10 USD directly is complex because:")
    print("  1. The UR10 USD is in Isaac Lab Nucleus (remote or cached)")
    print("  2. The file is instanceable and shared across environments")
    print("\nSolution: Create a standalone peg USD for runtime attachment")
    print("  - Spawn peg via FixedJoint to UR10 ee_link")
    print("  - Or use USD reference to include peg in UR10 instance")
    print("=" * 60)

    create_peg_only_usd(output_path, radius=0.0125, height=0.0625)

    print("\n[SUCCESS] Peg USD created!")
    print(f"[INFO] Path: {output_path}")
    print("\nTo use with UR10:")
    print("  1. Update ur10.py spawn config to reference this peg USD")
    print("  2. Or spawn dynamically in forge_env.py using FixedJoint")
    print("=" * 60)

    simulation_app.close()


if __name__ == "__main__":
    main()
