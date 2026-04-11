#!/usr/bin/env python3
"""Script to create a peg USD asset that can be attached to UR10 EE link.

This creates a standalone peg USD file that can be spawned and attached
to the robot's end-effector using a fixed joint.

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/create_peg_asset.py
"""

# Start SimulationApp first to load pxr
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import os
from pxr import Usd, UsdGeom, Sdf, Gf, PhysxSchema, UsdPhysics


def create_peg_usd(output_path: str, radius: float = 0.0125, height: float = 0.0625):
    """Create a standalone peg USD file.

    Args:
        output_path: Output path for the peg USD file
        radius: Peg radius in meters (default 12.5mm for 25mm diameter)
        height: Peg height in meters (default 62.5mm)
    """
    print(f"[INFO] Creating peg USD at: {output_path}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create new stage
    stage = Usd.Stage.CreateNew(output_path)
    stage.SetDefaultPrim(stage.DefinePrim("/Peg", "Xform"))

    # Create peg root
    peg_root = UsdGeom.Xform.Define(stage, "/Peg")

    # Create cylinder mesh
    cylinder = UsdGeom.Cylinder.Define(stage, "/Peg/cylinder")
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)

    # Set transform - peg extends downward from origin
    cylinder.AddTranslateOp().Set((-0.0, -0.0, -height / 2))

    # Add visual material
    from pxr import UsdShade
    material_path = "/Peg/Material"
    material = UsdShade.Material.Define(stage, material_path)

    # Create simple preview surface material
    shader_path = material_path + "/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("PreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.6, 0.4, 0.2))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)

    # Bind material to cylinder
    UsdShade.MaterialBindingAPI.Apply(cylinder.GetPrim())
    UsdShade.MaterialBindingAPI(cylinder.GetPrim()).Bind(material)

    # Add RigidBody API
    rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(peg_root.GetPrim())
    rigid_body_api.CreateDisableGravityAttr(True)
    rigid_body_api.CreateLinearDampingAttr(0.0)
    rigid_body_api.CreateAngularDampingAttr(0.0)
    # Mass is set via RigidBodyAPI instead of PhysxRigidBodyAPI
    rb_api = UsdPhysics.RigidBodyAPI.Apply(peg_root.GetPrim())
    if rb_api is not None:
        rb_api.CreateMassAttr(0.01)

    # Add Collision API
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(cylinder.GetPrim())
    collision_api.CreateContactOffsetAttr(0.005)
    collision_api.CreateRestOffsetAttr(0.0)

    # Add Material API with friction
    mat_api = PhysxSchema.PhysxMaterialAPI.Apply(cylinder.GetPrim())
    mat_api.CreateStaticFrictionAttr(1.0)
    mat_api.CreateDynamicFrictionAttr(1.0)
    mat_api.CreateRestitutionAttr(0.0)

    # Save stage
    stage.GetRootLayer().Save()
    print(f"[INFO] Peg USD created successfully at: {output_path}")

    # Print prim hierarchy
    print("\n[INFO] Prim hierarchy:")
    for prim in stage.Traverse():
        print(f"  {prim.GetPath()} - {prim.GetPrimTypeInfo().GetTypeName()}")

    return output_path


def main():
    # Save to local IsaacLab directory instead of Nucleus
    output_path = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/assets/factory_peg_25mm.usd"

    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    create_peg_usd(output_path, radius=0.0125, height=0.0625)

    print("\n[SUCCESS] Peg USD asset created!")
    print(f"[INFO] Path: {output_path}")
    print("\n[INFO] To use this peg with UR10:")
    print("1. Update ur10.py to reference this peg USD")
    print("2. Or spawn it dynamically in forge_env.py _setup_scene()")
    print("3. Connect with FixedJoint to ee_link")

    simulation_app.close()


if __name__ == "__main__":
    main()
