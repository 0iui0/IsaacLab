"""Batch convert STL files to USDC with SDF collision using MeshConverter.

Runs inside Isaac Sim container. Uses original (non-simplified) STL files.

Usage (inside container):
    /isaac-sim/python.sh batch_convert_sdf.py --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--stl-dir", default=os.path.dirname(os.path.abspath(__file__)))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

STL_DIR = args_cli.stl_dir
LOG_FILE = os.path.join(STL_DIR, "conversion_log.txt")

CONVERSIONS = [
    ("pair1_fixed.stl", "pair1_fixed.usd", 0.101),
    ("pair2_fixed.stl", "pair2_fixed.usd", 0.100),
    ("pair1_peg.stl", "pair1_peg.usd", 0.032),
    ("pair2_peg.stl", "pair2_peg.usd", 0.014),
]


def log(msg):
    line = f"[SDF_CONVERT] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def main():
    with open(LOG_FILE, "w") as f:
        f.write("")

    for stl_name, usd_name, mass in CONVERSIONS:
        stl_path = os.path.join(STL_DIR, stl_name)
        usd_path = os.path.join(STL_DIR, usd_name)

        if not os.path.isfile(stl_path):
            log(f"SKIP {stl_name}: file not found")
            continue

        log(f"Converting {stl_name} -> {usd_name} (mass={mass}kg)")

        cfg = MeshConverterCfg(
            asset_path=stl_path,
            force_usd_conversion=True,
            make_instanceable=False,
            usd_dir=os.path.dirname(usd_path),
            usd_file_name=os.path.basename(usd_path),
            mass_props=schemas_cfg.MassPropertiesCfg(mass=mass),
            rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
            collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
            mesh_collision_props=schemas_cfg.SDFMeshPropertiesCfg(),
        )

        converter = MeshConverter(cfg)
        usd_out = converter.usd_path
        log(f"Generated: {usd_out} ({os.path.getsize(usd_out):,} bytes)")

        # Add ArticulationRootAPI to the default prim
        try:
            asset_stage = Usd.Stage.Open(usd_out)
            default_prim = asset_stage.GetDefaultPrim()
            if default_prim.IsValid():
                if not default_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    default_prim.ApplyAPI(UsdPhysics.ArticulationRootAPI)
                    log(f"Added ArticulationRootAPI to {default_prim.GetPath()}")
            asset_stage.Save()
            log(f"Saved: {usd_path} ({os.path.getsize(usd_path):,} bytes)")
        except Exception as e:
            log(f"ERROR post-processing {usd_name}: {e}")

    log("All conversions complete!")


if __name__ == "__main__":
    main()
    simulation_app.close()
