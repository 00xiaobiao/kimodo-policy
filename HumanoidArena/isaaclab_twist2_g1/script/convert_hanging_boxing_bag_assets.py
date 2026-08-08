#!/usr/bin/env python3
# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
轉換倒吊拳擊沙袋資產：URDF → Articulation USD。

流程:
  1. 使用 MeshConverter 將 bag OBJ 轉為 USD（含 translation/rotation/scale，與 convert_boxing_bag_assets 一致）
  2. 使用 UrdfConverter 將 URDF 轉為 Articulation USD
  3. 後處理：用預旋轉的 bag USD 替換 articulation 中 bag link 的 mesh

用法:
    python scripts/convert_hanging_boxing_bag_assets.py --headless --device cuda
"""

import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.environ["PROJECT_ROOT"] = project_root

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg, UrdfConverter, UrdfConverterCfg
from isaaclab.sim.schemas import schemas_cfg


# =============================================================================
# 沙袋 mesh 變換參數（可快速測試調整）
# - BAG_TRANSLATION: 相對於 bag link 原點（掛載點）的偏移。Isaac Sim +Z 向上，
#   沙袋應在掛點下方，故 Z 為負，如 (0,0,-0.5) 表示 mesh 中心在掛點下方 0.5m
# - BAG_ROTATION: 四元數 (w,x,y,z)。若 OBJ 長軸為 X，用 (0.7071,0,-0.7071,0) 繞 Y -90° 使長軸對齊 -Z
#   若長軸為 Y：試 (0.7071,-0.7071,0,0) 或 (0.7071,0.7071,0,0)
# - BAG_SCALE: 縮放
# =============================================================================
BAG_TRANSLATION = (0.0, 0.0, -0.5)  # 負 Z = 向下（掛在錨點下方）
BAG_ROTATION = (0.7071, 0.7071, 0.0, 0.0)  # 繞 Y -90°：長軸 X → -Z（向下）
BAG_SCALE = (0.15, 0.15, 0.15)


def convert_obj_to_usd(
    obj_path: str,
    usd_path: str,
    translation=(0.0, 0.0, 0.0),
    rotation=(1.0, 0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
) -> str:
    """使用 MeshConverter 轉換 OBJ，支援 translation/rotation/scale（與 convert_boxing_bag_assets 一致）。"""
    obj_path = os.path.abspath(obj_path)
    usd_path = os.path.abspath(usd_path)
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ not found: {obj_path}")
    os.makedirs(os.path.dirname(usd_path), exist_ok=True)
    cfg = MeshConverterCfg(
        mass_props=None,
        rigid_props=None,
        # Visual override mesh should not own collisions.
        # Keep physical collision on URDF-generated bag collision prims.
        collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=False),
        asset_path=obj_path,
        force_usd_conversion=True,
        usd_dir=os.path.dirname(usd_path),
        usd_file_name=os.path.basename(usd_path),
        make_instanceable=False,
        translation=translation,
        rotation=rotation,
        scale=scale,
    )
    converter = MeshConverter(cfg)
    return converter.usd_path


def convert_urdf_to_usd(urdf_path: str, usd_path: str, fix_base: bool = True) -> str:
    urdf_path = os.path.abspath(urdf_path)
    usd_path = os.path.abspath(usd_path)
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    os.makedirs(os.path.dirname(usd_path), exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(usd_path),
        usd_file_name=os.path.basename(usd_path),
        fix_base=fix_base,
        merge_fixed_joints=False,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.3,
            ),
            target_type="none",
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"  -> {converter.usd_path}")
    return converter.usd_path


def replace_bag_mesh_in_articulation(articulation_usd: str, bag_mesh_usd: str, bag_link_name: str = "bag"):
    """Replace bag visual/collision with a mesh that already has correct transform."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(articulation_usd)
    if not stage:
        raise RuntimeError(f"Cannot open {articulation_usd}")

    bag_prim = None
    for prim in stage.Traverse():
        if prim.GetName() == bag_link_name:
            bag_prim = prim
            break

    if not bag_prim:
        print("  [WARN] Bag link not found, skipping mesh replace. Using URDF mesh.")
        stage.Save()
        return

    # Disable old visual/collision subtree to avoid mismatch with transformed mesh.
    deactivated_visual = 0
    deactivated_collision = 0
    for prim in Usd.PrimRange(bag_prim):
        if prim == bag_prim:
            continue
        path_lower = prim.GetPath().pathString.lower()
        if "visual" in path_lower:
            prim.SetActive(False)
            deactivated_visual += 1
        if "/collisions/" in path_lower or "collision" in path_lower:
            prim.SetActive(False)
            deactivated_collision += 1

    bag_mesh_abs = os.path.abspath(bag_mesh_usd)
    visual_override_path = bag_prim.GetPath().AppendChild("bag_visual_override")
    visual_override_xform = UsdGeom.Xform.Define(stage, visual_override_path).GetPrim()
    visual_override_xform.GetReferences().AddReference(bag_mesh_abs)

    # Ensure visual override does not participate in collision.
    disabled_collision = 0
    removed_collision_api = 0
    for prim in Usd.PrimRange(visual_override_xform):
        if UsdPhysics.CollisionAPI.CanApply(prim) or prim.HasAPI(UsdPhysics.CollisionAPI):
            col_api = UsdPhysics.CollisionAPI.Apply(prim)
            col_api.CreateCollisionEnabledAttr(False)
            disabled_collision += 1

        for api in (
            UsdPhysics.CollisionAPI,
            UsdPhysics.MeshCollisionAPI,
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.MassAPI,
        ):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
                removed_collision_api += 1

    # Create collision override from the same transformed mesh to keep
    # collision and visual perfectly aligned.
    collision_override_path = bag_prim.GetPath().AppendChild("bag_collision_override")
    collision_override_xform = UsdGeom.Xform.Define(stage, collision_override_path).GetPrim()
    collision_override_xform.GetReferences().AddReference(bag_mesh_abs)
    UsdGeom.Imageable(collision_override_xform).CreateVisibilityAttr("invisible")

    collision_mesh_count = 0
    for prim in Usd.PrimRange(collision_override_xform):
        if prim.IsA(UsdGeom.Mesh):
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            collision_api.CreateCollisionEnabledAttr(True)
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision_api.CreateApproximationAttr("convexHull")
            collision_mesh_count += 1

        # visual mesh should not own rigid-body/mass APIs
        for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)

    print(f"  [INFO] Added bag visual override: {visual_override_path} -> {bag_mesh_abs}")
    print(f"  [INFO] Added bag collision override: {collision_override_path} -> {bag_mesh_abs}")
    print(f"  [INFO] Deactivated visual prims: {deactivated_visual}")
    print(f"  [INFO] Deactivated collision prims: {deactivated_collision}")
    print(f"  [INFO] Disabled collision on override prims: {disabled_collision}")
    print(f"  [INFO] Removed physics APIs on override prims: {removed_collision_api}")
    print(f"  [INFO] Collision meshes configured: {collision_mesh_count}")
    stage.Save()


def main():
    assets_dir = os.path.join(project_root, "assets", "hanging_boxing_bag")
    bag_obj = os.path.join(assets_dir, "frpnchbg.obj")
    bag_mesh_usd = os.path.join(assets_dir, "hanging_bag_mesh.usd")
    urdf_path = os.path.join(assets_dir, "hanging_bag.urdf")
    articulation_usd = os.path.join(assets_dir, "hanging_bag_articulation.usd")

    print("1. Bag mesh (OBJ -> USD with rotation, like convert_boxing_bag_assets):")
    convert_obj_to_usd(
        bag_obj,
        bag_mesh_usd,
        translation=BAG_TRANSLATION,
        rotation=BAG_ROTATION,
        scale=BAG_SCALE,
    )
    print(f"   -> {bag_mesh_usd}")

    print("2. Articulation (URDF -> USD):")
    convert_urdf_to_usd(urdf_path, articulation_usd, fix_base=True)
    print(f"   -> {articulation_usd}")

    print("3. Replacing bag mesh with pre-rotated mesh...")
    try:
        replace_bag_mesh_in_articulation(articulation_usd, bag_mesh_usd)
        print("   Done.")
    except Exception as e:
        print(f"   [WARN] {e}")

    print("\nDone. Generated:")
    print(f"  - {bag_mesh_usd}")
    print(f"  - {articulation_usd}")


if __name__ == "__main__":
    main()
    simulation_app.close()
