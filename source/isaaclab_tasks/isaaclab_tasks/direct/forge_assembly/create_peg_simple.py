#!/usr/bin/env python3
"""Simple script to create a peg USD asset.

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/create_peg_simple.py
"""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import os
from pxr import Usd, UsdGeom, Gf, PhysxSchema, UsdPhysics, UsdShade, Sdf

def create_peg_usd(output_path: str, radius: float = 0.0125, height: float = 0.0625):
    """Create a standalone peg USD file."""
    print(f"[INFO] Creating peg USD at: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create new stage
    stage = Usd.Stage.CreateNew(output_path)

    # Create peg root Xform
    peg_root = UsdGeom.Xform.Define(stage, "/Peg")

    # Create cylinder mesh
    cylinder = UsdGeom.Cylinder.Define(stage, "/Peg/cylinder")
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.AddTranslateOp().Set((0.0, 0.0, -height / 2))

    # Apply Physics API
    UsdPhysics.RigidBodyAPI.Apply(peg_root.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())

    # Apply PhysX schema
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(peg_root.GetPrim())
    physx_rb.CreateDisableGravityAttr(True)
    physx_rb.CreateLinearDampingAttr(0.0)
    physx_rb.CreateAngularDampingAttr(0.0)

    physx_coll = PhysxSchema.PhysxCollisionAPI.Apply(cylinder.GetPrim())
    physx_coll.CreateContactOffsetAttr(0.005)
    physx_coll.CreateRestOffsetAttr(0.0)

    # Create and bind material
    material = UsdShade.Material.Define(stage, "/Peg/Material")
    shader = UsdShade.Shader.Define(stage, "/Peg/Material/Shader")
    shader.CreateIdAttr("PreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.6, 0.4, 0.2))
    UsdShade.MaterialBindingAPI.Apply(cylinder.GetPrim())
    UsdShade.MaterialBindingAPI(cylinder.GetPrim()).Bind(material)

    # Save
    stage.GetRootLayer().Save()
    print(f"[INFO] Peg USD created: {output_path}")

    # Show hierarchy
    print("\n[INFO] Prim hierarchy:")
    for prim in stage.Traverse():
        print(f"  {prim.GetPath()}")

    return True


# Main
output_path = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/assets/factory_peg_25mm.usd"
create_peg_usd(output_path)
print("\n[SUCCESS] Done!")
simulation_app.close()
