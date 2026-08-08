"""
足球物理材質工具。
UsdFileCfg 不支援 physics_material，需於場景 reset 後動態套用。
FIFA 足球 restitution 典型區間 0.7–0.8。
"""


def apply_football_physics_material(
    ball_prim_pattern: str = "/World/envs/env_.*/Object",
    restitution: float = 0.9,
    static_friction: float = 0.5,
    dynamic_friction: float = 0.4,
) -> bool:
    """
    於場景載入後動態套用物理材質到足球 prim，使彈性接近真實足球。

    需在 Isaac Sim / Isaac Lab 環境內執行（含 isaaclab、omni 模組）。

    Args:
        ball_prim_pattern: 足球 prim 路徑或 regex，預設 /World/envs/env_.*/Object
        restitution: 恢復係數，FIFA 典型 0.7–0.8
        static_friction: 靜摩擦
        dynamic_friction: 動摩擦

    Returns:
        True 若成功套用，False 若跳過或發生錯誤
    """
    try:
        import omni.usd
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners.materials.physics_materials import spawn_rigid_body_material
        from isaaclab.sim.utils import bind_physics_material

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[football_physics_material] ⚠️ 無有效 stage，跳過")
            return False

        matches = []
        try:
            from isaaclab.sim.utils.prims import find_matching_prim_paths

            found = find_matching_prim_paths(ball_prim_pattern)
            matches = list(found) if found else []
        except (ImportError, AttributeError):
            pass
        if not matches:
            for p in ["/World/envs/env_0/Object", "/World/envs/env_0/Football"]:
                prim = stage.GetPrimAtPath(p)
                if prim and prim.IsValid():
                    matches = [p]
                    break
        if not matches:
            print(f"[football_physics_material] ⚠️ 找不到足球 prim: {ball_prim_pattern}")
            return False

        material_path = "/World/FootballPhysicsMaterial"
        mat_cfg = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="max",
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        )
        spawn_rigid_body_material(material_path, mat_cfg)

        bound = 0
        for ball_path in matches:
            try:
                bind_physics_material(ball_path, material_path, stage=stage)
                bound += 1
            except Exception:
                pass

        if bound > 0:
            print(f"[football_physics_material] ✓ 已套用 restitution={restitution} 到足球 ({bound} 處)")
            return True
        print("[football_physics_material] ⚠️ 未成功綁定（足球 geometry 可能為 instanced，需修改 USD）")
        return False

    except ImportError as e:
        print(f"[football_physics_material] ⚠️ 無法載入必要模組，跳過: {e}")
        return False
    except Exception as e:
        print(f"[football_physics_material] ⚠️ 套用失敗: {e}")
        import traceback

        traceback.print_exc()
        return False