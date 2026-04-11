#!/usr/bin/env python3
"""Add peg as a child of ee_link to UR10 USD file.

This script modifies the UR10 USD file to include a peg rigid body
attached to the end-effector link with a fixed joint.

Usage:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/forge_assembly/ur10_add_peg.py
"""

import carb
from omni.isaac.kit import SimulationApp

# Start Isaac Sim
simulation_app = SimulationApp({"headless": True})

import omni.usd
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
from omni.isaac.core.utils.stage import get_current_stage, save_stage
from pxr import Usd, UsdGeom, Gf, Sdf


def add_peg_to_ur10(usd_path: str):
    """Add peg as a child prim of ee_link in UR10 USD.

    Args:
        usd_path: Path to the UR10 USD file
    """
    print(f"[INFO] Opening USD file: {usd_path}")

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        omni.usd.get_context().open_stage(usd_path)
        stage = omni.usd.get_context().get_stage()

    if stage is None:
        print(f"[ERROR] Failed to open stage: {usd_path}")
        return False

    # Find ee_link prim
    ee_link_path = None
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if "ee_link" in prim_path and "Robot" not in prim_path:
            # This is the template ee_link, not an instance
            ee_link_path = prim_path
            break

    if ee_link_path is None:
        # Try common UR10 ee_link paths
        possible_paths = [
            "/ur10/ee_link",
            "/Robot/ee_link",
            "/ur10_robot/ee_link",
            "/ee_link",
        ]
        for path in possible_paths:
            prim = get_prim_at_path(path)
            if prim.IsValid():
                ee_link_path = path
                break

    if ee_link_path is None:
        print("[ERROR] Could not find ee_link in USD file")
        print("[INFO] Available prims:")
        for prim in stage.Traverse():
            print(f"  - {prim.GetPath()}")
        return False

    print(f"[INFO] Found ee_link at: {ee_link_path}")

    ee_link_prim = get_prim_at_path(ee_link_path)

    # Define peg prim as child of ee_link
    peg_path = f"{ee_link_path}/peg"
    peg_prim = define_prim(peg_path, "Xform")

    if not peg_prim.IsValid():
        print(f"[ERROR] Failed to create peg prim at: {peg_path}")
        return False

    print(f"[INFO] Created peg prim at: {peg_path}")

    # Add mesh component (cylinder) to peg
    # We'll add a reference to a cylinder or create one directly

    # Create cylinder geometry
    mesh_path = f"{peg_path}/mesh"
    mesh_prim = define_prim(mesh_path, "Cylinder")

    if mesh_prim.IsValid():
        # Set cylinder attributes
        UsdGeom.Cylinder(mesh_prim).CreateRadiusAttr(0.0125)  # 25mm diameter
        UsdGeom.Cylinder(mesh_prim).CreateHeightAttr(0.0625)  # 62.5mm height

        # Set transform (lower half extends below EE)
        mesh_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.0, 0.0, -0.03125))

        # Add material
        UsdGeom.MaterialAPI(mesh_prim)

        print(f"[INFO] Created cylinder mesh at: {mesh_path}")

    # Add RigidBody API to peg
    from omni.isaac.dynamic_control import _dynamic_control
    dc = _dynamic_control.acquire_dynamic_control_interface()

    # Add rigid body API using PhysX schema
    from pxr import PhysxSchema

    if not PhysxSchema.PhysxRigidBodyAPI.Apply(peg_prim):
        print("[WARNING] Could not apply PhysxRigidBodyAPI")
    else:
        rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Get(stage, peg_path)
        # Set rigid body properties
        rigid_body_api.CreateDisableGravityAttr(True)
        rigid_body_api.CreateLinearDampingAttr(0.0)
        rigid_body_api.CreateAngularDampingAttr(0.0)
        rigid_body_api.CreateMassAttr(0.01)
        print("[INFO] Applied RigidBody API to peg")

    # Add Collision API to peg
    if not PhysxSchema.PhysxCollisionAPI.Apply(peg_prim):
        print("[WARNING] Could not apply PhysxCollisionAPI")
    else:
        collision_api = PhysxSchema.PhysxCollisionAPI.Get(stage, peg_path)
        collision_api.CreateContactOffsetAttr(0.005)
        collision_api.CreateRestOffsetAttr(0.0)
        print("[INFO] Applied Collision API to peg")

    # Add material with friction
    if not PhysxSchema.PhysxMaterialAPI.Apply(peg_prim):
        print("[WARNING] Could not apply PhysxMaterialAPI")
    else:
        material_api = PhysxSchema.PhysxMaterialAPI.Get(stage, peg_path)
        material_api.CreateStaticFrictionAttr(1.0)
        material_api.CreateDynamicFrictionAttr(1.0)
        material_api.CreateRestitutionAttr(0.0)
        print("[INFO] Applied Material API with friction to peg")

    # Add Fixed Joint to connect peg to ee_link
    joint_path = f"{peg_path}/fixed_joint"
    joint_prim = define_prim(joint_path, "FixedJoint")

    if joint_prim.IsValid():
        # Set joint body0 (peg) and body1 (ee_link)
        joint_prim.GetAttribute("body0").Set(peg_path)
        joint_prim.GetAttribute("body1").Set(ee_link_path)

        # Set local pose transforms
        joint_prim.GetAttribute("localPose0").Set(Gf.Rang3f((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)))
        joint_prim.GetAttribute("localPose1").Set(Gf.Rang3f((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)))

        print(f"[INFO] Created fixed joint at: {joint_path}")

    # Save the stage
    output_path = usd_path.replace(".usd", "_with_peg.usd")
    save_stage(output_path)
    print(f"[INFO] Saved modified USD to: {output_path}")

    return True


def main():
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    # Path to UR10 USD file
    ur10_usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd"
    print(f"[INFO] UR10 USD path: {ur10_usd_path}")

    success = add_peg_to_ur10(ur10_usd_path)

    if success:
        print("\n[SUCCESS] Peg added to UR10 USD file!")
        print("[INFO] Update ur10.py to use the new USD file:")
        print(f'  usd_path=".../ur10_with_peg.usd"')
    else:
        print("\n[FAILED] Could not add peg to UR10 USD file")

    simulation_app.close()


if __name__ == "__main__":
    main()
