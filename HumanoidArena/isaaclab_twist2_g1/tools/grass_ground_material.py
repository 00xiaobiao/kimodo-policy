"""
草坪 PBR 地面材質工具。
在場景載入後動態將草坪 PBR 貼圖套用到地面 prim，需在 Isaac Sim 環境內執行。
支援 AITextured 轉出的 PBR 貼圖命名（*__Png_albedo.png, *__Png_normal.png 等）。
"""

import glob
import os
from pathlib import Path


def _find_texture(textures_dir: str, patterns: list[str]) -> str | None:
    """在目錄中依多個 pattern 尋找貼圖，回傳第一個找到的完整路徑。"""
    for pat in patterns:
        matches = glob.glob(os.path.join(textures_dir, pat))
        if matches:
            return matches[0]
    return None


def apply_grass_pbr_to_ground(
    prim_path: str = "/World/GroundPlane",
    textures_dir: str = None,
    uv_scale: tuple = (25.0, 25.0),
) -> bool:
    """
    將草坪 PBR 材質套用到指定地面 prim。

    需在 Isaac Sim 環境內執行（含 pxr、omni 模組）。
    若貼圖檔案不存在則跳過，不影響場景運行。

    Args:
        prim_path: 地面 prim 路徑，預設 /World/GroundPlane
        textures_dir: 貼圖目錄，預設使用 PROJECT_ROOT/assets/objects/materials/grass_turf
        uv_scale: UV 重複倍率 (scale_u, scale_v)，越大貼圖越細密。預設 (25, 25) 使草地比例相對於機器人合理

    Returns:
        True 若成功套用 PBR 材質，False 若貼圖缺失或發生錯誤
    """
    try:
        import omni.usd
        from pxr import Sdf, UsdShade, UsdGeom, Usd, Vt, Gf
    except ImportError as e:
        print(f"[grass_ground_material] ⚠️ 無法載入 omni.usd 或 pxr，跳過草坪材質套用: {e}")
        return False

    # 解析貼圖目錄
    if textures_dir is None:
        project_root = os.environ.get("PROJECT_ROOT")
        if not project_root:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        textures_dir = str(Path(project_root) / "assets" / "objects" / "materials" / "grass_turf")

    # 支援 AITextured 命名 (*__Png_albedo.png) 及舊版 grass_albedo.png
    albedo_path = _find_texture(
        textures_dir,
        ["*__Png_albedo.png", "*_albedo.png", "grass_albedo.png"],
    )
    if not albedo_path:
        print(f"[grass_ground_material] ⚠️ Albedo 貼圖未找到，請參考 assets/objects/materials/grass_turf/README.md")
        return False

    # 確保使用絕對路徑（Omniverse 需絕對路徑解析貼圖）
    albedo_path = os.path.abspath(albedo_path)
    textures_dir = os.path.abspath(textures_dir)
    print(f"[grass_ground_material] 貼圖目錄: {textures_dir}")
    print(f"[grass_ground_material] Albedo: {albedo_path}")

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"[grass_ground_material] ⚠️ 找不到地面 prim: {prim_path}")
        return False

    # 收集所有需綁定材質的 mesh（含 prim 本身及其子孫）
    def _collect_geom_prims(p):
        prims = []
        for sub in Usd.PrimRange(p):
            if sub.IsA(UsdGeom.Mesh) or sub.IsA(UsdGeom.Cube) or sub.IsA(UsdGeom.Plane):
                prims.append(sub)
        return prims

    target_meshes = _collect_geom_prims(prim)
    if not target_meshes:
        target_meshes = [prim]

    def _ensure_st(mesh_prim):
        primvars = UsdGeom.PrimvarsAPI(mesh_prim)
        if mesh_prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(mesh_prim)
            points = mesh.GetPointsAttr().Get()
            if points:
                min_x = min(float(p[0]) for p in points)
                max_x = max(float(p[0]) for p in points)
                min_y = min(float(p[1]) for p in points)
                max_y = max(float(p[1]) for p in points)
                dx = max(max_x - min_x, 1e-6)
                dy = max(max_y - min_y, 1e-6)
                st_values = Vt.Vec2fArray(
                    [Gf.Vec2f((float(p[0]) - min_x) / dx, (float(p[1]) - min_y) / dy) for p in points]
                )
                st_primvar = primvars.GetPrimvar("st")
                if not st_primvar or not st_primvar.IsDefined():
                    st_primvar = primvars.CreatePrimvar(
                        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
                    )
                st_primvar.Set(st_values)
                st_primvar.SetInterpolation(UsdGeom.Tokens.vertex)
                return True
            face_counts = mesh.GetFaceVertexCountsAttr().Get()
            total = int(sum(face_counts)) if face_counts else 0
            if total <= 0:
                return False
            st_values = Vt.Vec2fArray([Gf.Vec2f(0.0, 0.0)] * total)
            for i in range(total):
                u = 1.0 if (i % 4) in (1, 2) else 0.0
                v = 1.0 if (i % 4) in (2, 3) else 0.0
                st_values[i] = Gf.Vec2f(u, v)
            st_primvar = primvars.GetPrimvar("st")
            if not st_primvar or not st_primvar.IsDefined():
                st_primvar = primvars.CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                )
            st_primvar.Set(st_values)
            st_primvar.SetInterpolation(UsdGeom.Tokens.faceVarying)
            return True
        if mesh_prim.IsA(UsdGeom.Cube):
            st_values = Vt.Vec2fArray(
                [Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0)] * 6
            )
            st_primvar = primvars.GetPrimvar("st")
            if not st_primvar or not st_primvar.IsDefined():
                st_primvar = primvars.CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                )
            st_primvar.Set(st_values)
            st_primvar.SetInterpolation(UsdGeom.Tokens.faceVarying)
            return True
        if mesh_prim.IsA(UsdGeom.Plane):
            st_values = Vt.Vec2fArray(
                [Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0)]
            )
            st_primvar = primvars.GetPrimvar("st")
            if not st_primvar or not st_primvar.IsDefined():
                st_primvar = primvars.CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                )
            st_primvar.Set(st_values)
            st_primvar.SetInterpolation(UsdGeom.Tokens.faceVarying)
            return True
        return False

    # Roughness 貼圖（可選，AITextured Roughness contrast 0.8 對應）
    _roughness_path = _find_texture(
        textures_dir,
        ["*__Png_roughness.png", "*_roughness.png", "grass_roughness.png"],
    )
    roughness_path = os.path.abspath(_roughness_path) if _roughness_path else None

    # 1. 建立 PBR 材質（UsdPreviewSurface + UsdUVTexture）
    mtl_path = Sdf.Path("/World/Looks/GrassTurfMaterial")
    mtl = UsdShade.Material.Define(stage, mtl_path)
    shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    # Metallic 0.0（草地為介電質）
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    # Albedo 貼圖（sRGB 色彩空間）
    albedo_tx = UsdShade.Shader.Define(stage, mtl_path.AppendPath("AlbedoTx"))
    albedo_tx.CreateIdAttr("UsdUVTexture")
    albedo_tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(albedo_path)
    albedo_tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    albedo_tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    albedo_tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    albedo_out = albedo_tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(albedo_out)

    # UV 縮放：PrimvarReader -> UsdTransform2d (scale) -> 貼圖 st
    st_reader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("StReader"))
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_reader_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    uv_transform = UsdShade.Shader.Define(stage, mtl_path.AppendPath("UVTransform"))
    uv_transform.CreateIdAttr("UsdTransform2d")
    uv_transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader_out)
    uv_transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(
        (float(uv_scale[0]), float(uv_scale[1]))
    )
    uv_out = uv_transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    albedo_tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_out)

    # Roughness 貼圖（可選，AITextured Roughness contrast 0.8）
    if roughness_path:
        roughness_tx = UsdShade.Shader.Define(stage, mtl_path.AppendPath("RoughnessTx"))
        roughness_tx.CreateIdAttr("UsdUVTexture")
        roughness_tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(roughness_path)
        roughness_tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")  # Roughness 為灰階 Linear
        roughness_tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        roughness_tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        roughness_tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_out)
        roughness_out = roughness_tx.CreateOutput("r", Sdf.ValueTypeNames.Float)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(roughness_out)
    else:
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)

    # Normal 貼圖（可選，AITextured Normal strength 2.5）
    _normal_path = _find_texture(
        textures_dir,
        ["*__Png_normal.png", "*_normal.png", "grass_normal.png"],
    )
    normal_path = os.path.abspath(_normal_path) if _normal_path else None
    if normal_path:
        normal_tx = UsdShade.Shader.Define(stage, mtl_path.AppendPath("NormalTx"))
        normal_tx.CreateIdAttr("UsdUVTexture")
        normal_tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(normal_path)
        normal_tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")  # Normal 為 Linear
        normal_tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        normal_tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        normal_tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_out)
        normal_out = normal_tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("normal", Sdf.ValueTypeNames.Float3).ConnectToSource(normal_out)

    mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # 2. 綁定材質到所有 ground mesh（覆蓋預設 checker）
    st_bound_count = 0
    for mesh_prim in target_meshes:
        if _ensure_st(mesh_prim):
            st_bound_count += 1

    bound_count = 0
    for mesh_prim in target_meshes:
        binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        binding_api.Bind(mtl)
        bound_count += 1

    # 3. 若 prim_path 為 GroundPlane，同時套用到 defaultGroundPlane（Isaac Sim 預設可能為後者）
    if prim_path == "/World/GroundPlane":
        default_prim = stage.GetPrimAtPath("/World/defaultGroundPlane")
        if default_prim and default_prim.IsValid():
            for mesh_prim in _collect_geom_prims(default_prim):
                if _ensure_st(mesh_prim):
                    st_bound_count += 1
                binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
                binding_api.Bind(mtl)
                bound_count += 1

    print(
        f"[grass_ground_material] ✅ 草坪 PBR 材質已套用到 {bound_count} 個 mesh，"
        f"已寫入 st UV 的 mesh: {st_bound_count}"
    )
    return True
