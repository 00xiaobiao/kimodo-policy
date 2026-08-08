"""
足球場標線工具。
在場景載入後動態於地面建立白色劃線，需在 Isaac Sim 環境內執行。

座標（Isaac Sim Z-up）：地面 XY，高度 Z。
重要：若 USD stage 使用 metersPerUnit=0.01（公分），會自動將「米」轉為 USD 單位。
"""

import math
from typing import Tuple

# 標線寬度與厚度 (m)
DEFAULT_LINE_WIDTH = 0.12
DEFAULT_LINE_HEIGHT = 0.005
DEFAULT_LINE_COLOR = (127.0 / 255.0, 127.0 / 255.0, 127.0 / 255.0)

# 小型球場佈局 - 從 base_scene_football_cfg_wholebody 同步（goal_y = 球門 y，grass_half = 草地半寬）
def _get_football_layout():
    """從場景配置讀取佈局，若失敗則用 fallback。"""
    try:
        from tasks.common_scene.base_scene_football_cfg_wholebody import (
            GOAL_DISTANCE,
            ROBOT_INIT_Y,
        )
        goal_y = ROBOT_INIT_Y + GOAL_DISTANCE
        grass_half = 7.0  # ground size (14,14) / 2
        return goal_y, grass_half
    except ImportError:
        return 6.0, 7.0

_LAYOUT_GOAL_Y, _LAYOUT_GRASS_HALF = _get_football_layout()
# 單位為米：球門線略在球門前，邊線距草地邊緣留距
SMALL_PITCH_GOAL_Y = _LAYOUT_GOAL_Y - 0.5
SMALL_PITCH_GRASS_HALF = _LAYOUT_GRASS_HALF - 0.5

# FIFA 標準球場尺寸 (m，供 create_pitch_lines_in_stage 使用)
FIFA_LINE_WIDTH = 0.12
FIFA_PITCH_LENGTH = 105.0
FIFA_PITCH_WIDTH = 68.0
FIFA_CENTER_RADIUS = 9.15
FIFA_CORNER_ARC_RADIUS = 1.0
FIFA_PENALTY_LENGTH = 16.5
FIFA_PENALTY_WIDTH = 40.3
FIFA_GOAL_LENGTH = 5.5
FIFA_GOAL_WIDTH = 18.32
FIFA_PENALTY_SPOT_DIST = 11.0
FIFA_PENALTY_ARC_RADIUS = 9.15


def _get_stage_scale(stage) -> float:
    """
    取得 stage 的單位比例：我們的參數一律為「米」。
    若 metersPerUnit=1，1m=1 USD 單位；若 metersPerUnit=0.01（公分），1m=100 USD 單位。
    回傳：米 -> USD 單位的乘數（= 1/metersPerUnit）
    """
    try:
        root = stage.GetRootLayer()
        mpu = root.GetMetadataByKey("metersPerUnit") if root else None
        if mpu is not None:
            mpu_f = float(mpu)
            if mpu_f > 0 and mpu_f != 1.0:
                scale = 1.0 / mpu_f
                print(f"[pitch_lines] 偵測 metersPerUnit={mpu}，米值已乘 {scale:.0f} 轉為 USD 單位")
                return scale
        return 1.0
    except Exception:
        return 1.0


def _check_stage_axes(stage) -> None:
    """檢查 stage 單位與 upAxis，若異常則印出警告。"""
    try:
        root = stage.GetRootLayer()
        up = root.GetMetadataByKey("upAxis") if root else None
        if up is not None and str(up) != "Z":
            print(f"[pitch_lines] ⚠️ upAxis={up} (非 Z-up)，標線可能錯位")
    except Exception:
        pass


def create_robot_circle(
    stage,
    center: Tuple[float, float] = (0.0, 0.0),
    radius: float = 2.0,
    parent_path: str = "/World/PitchLines",
    line_width: float = DEFAULT_LINE_WIDTH,
    line_height: float = DEFAULT_LINE_HEIGHT,
    line_color: Tuple[float, float, float] = (127.0 / 255.0, 127.0 / 255.0, 127.0 / 255.0),
    n_segments: int = 128,
) -> bool:
    """
    在機器人周圍畫一個白色圓圈（半徑 2m，可自訂）。
    使用 Mesh 環形幾何（頂點直接定義圓弧），避免 cube 拼接的鋸齒與 X/Y 縮放不一致問題。

    Args:
        stage: USD stage（omni.usd.get_context().get_stage()）
        center: 圓心 (x, y)，預設 (0, 0) 與機器人初始位置一致
        radius: 圓圈半徑 (m)
        parent_path: 父 prim 路徑
        line_width: 線寬 (m)
        line_height: 線厚度（略高於地面避免 z-fighting）
        line_color: 標線顏色
        n_segments: 圓弧分段數（越多越平滑，預設 128）

    Returns:
        True 若成功
    """
    try:
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
    except ImportError as e:
        print(f"[pitch_lines] ⚠️ 無法載入 pxr，跳過: {e}")
        return False

    scale = _get_stage_scale(stage)
    _check_stage_axes(stage)

    w = line_width * scale
    h = line_height * scale
    cx, cy = center[0] * scale, center[1] * scale
    radius = radius * scale

    # 環形 mesh：內外兩圈頂點，直接以 (x,y,z) 定義圓弧，無需 scale/rotate
    r_in = radius - w / 2
    r_out = radius + w / 2
    z = h / 2

    points = []
    for i in range(n_segments + 1):
        a = 2 * math.pi * i / n_segments
        points.append(Gf.Vec3f(cx + r_in * math.cos(a), cy + r_in * math.sin(a), z))
        points.append(Gf.Vec3f(cx + r_out * math.cos(a), cy + r_out * math.sin(a), z))

    face_vertex_counts = []
    face_vertex_indices = []
    for i in range(n_segments):
        i0 = 2 * i
        i1 = 2 * (i + 1)
        face_vertex_counts.append(4)
        face_vertex_indices.extend([i0, i0 + 1, i1 + 1, i1])

    path = f"{parent_path}/circle_ring"
    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        UsdGeom.Xform.Define(stage, parent_path)
    else:
        for child in list(parent.GetChildren()):
            stage.RemovePrim(child.GetPath())

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*line_color)])

    mat_path = f"{path}_mat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    binding.Bind(mat)

    print(
        f"[pitch_lines] 已建立圓圈（Mesh 環形）：圓心 {center}，半徑 {radius}m，線寬 {w}m，"
        f"{n_segments} 段，line_color={line_color}"
    )
    return True


def _create_rect_ring_mesh(
    stage, parent_path: str, name: str,
    gx: float, gy: float, w: float, z: float,
    line_color: Tuple[float, float, float],
) -> None:
    """
    建立矩形環狀 mesh（單一幾何體），四邊等寬、角落無重疊，形成完整矩形框。
    gx, gy = 半寬、半長；w = 線寬。
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
    # 外圈四角（逆時針）
    o0 = Gf.Vec3f(-gx - w / 2, -gy - w / 2, z)
    o1 = Gf.Vec3f(gx + w / 2, -gy - w / 2, z)
    o2 = Gf.Vec3f(gx + w / 2, gy + w / 2, z)
    o3 = Gf.Vec3f(-gx - w / 2, gy + w / 2, z)
    # 內圈四角（逆時針）
    i0 = Gf.Vec3f(-gx + w / 2, -gy + w / 2, z)
    i1 = Gf.Vec3f(gx - w / 2, -gy + w / 2, z)
    i2 = Gf.Vec3f(gx - w / 2, gy - w / 2, z)
    i3 = Gf.Vec3f(-gx + w / 2, gy - w / 2, z)
    points = Vt.Vec3fArray([o0, o1, o2, o3, i0, i1, i2, i3])
    face_vertex_counts = [4, 4, 4, 4]
    # 頂點順序使法線朝 +Z，從上方可見（避免 backface culling）
    face_vertex_indices = [
        0, 4, 5, 1,   # 下邊
        1, 5, 6, 2,   # 右邊
        2, 6, 7, 3,   # 上邊
        3, 7, 4, 0,   # 左邊
    ]
    path = f"{parent_path}/{name}"
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*line_color)])
    mat_path = f"{path}_mat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    binding.Bind(mat)


def create_small_pitch_boundary(
    stage,
    parent_path: str = "/World/PitchLines",
    goal_y: float = SMALL_PITCH_GOAL_Y,
    grass_half: float = SMALL_PITCH_GRASS_HALF,
    line_width: float = DEFAULT_LINE_WIDTH,
    line_height: float = DEFAULT_LINE_HEIGHT,
    line_color: Tuple[float, float, float] = (127.0 / 255.0, 127.0 / 255.0, 127.0 / 255.0),
    clear_parent: bool = False,
) -> bool:
    """
    建立小型球場四條邊線。所有參數單位為「米」，會依 stage 的 metersPerUnit 自動轉換。
    - 球門線 1: y=goal_y（球門側），沿 X
    - 球門線 2: y=-goal_y，沿 X
    - 邊線 1: x=-grass_half（草地邊緣），沿 Y
    - 邊線 2: x=+grass_half，沿 Y
    """
    try:
        from pxr import UsdGeom
    except ImportError as e:
        print(f"[pitch_lines] ⚠️ 無法載入 pxr: {e}")
        return False

    scale = _get_stage_scale(stage)
    w = line_width * scale
    h = line_height * scale
    goal_y_u = goal_y * scale
    grass_half_u = grass_half * scale
    z = h / 2

    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        UsdGeom.Xform.Define(stage, parent_path)
    elif clear_parent:
        for child in list(parent.GetChildren()):
            stage.RemovePrim(child.GetPath())

    # 單一矩形環狀 mesh：四邊一體、角落無重疊，形成完整矩形框
    _create_rect_ring_mesh(
        stage, parent_path, "pitch_boundary",
        grass_half_u, goal_y_u, w, z,
        line_color,
    )

    print(f"[pitch_lines] 已建立球場邊線：{2*grass_half:.1f}m × {2*goal_y:.1f}m（球門線 y=±{goal_y}，邊線 x=±{grass_half}）")
    return True


def create_goal_reference_lines(
    stage,
    parent_path: str = "/World/PitchLines",
    goal_centers: Tuple[Tuple[float, float], Tuple[float, float]] = ((-2.5, 3.0), (2.5, -3.0)),
    goal_relative_offsets: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, -0.45), (0.0, 0.45)),
    goal_line_length: float = 4.0,
    goal_line_width: float = DEFAULT_LINE_WIDTH * 0.5,
    line_height: float = DEFAULT_LINE_HEIGHT,
    line_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> bool:
    try:
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    except ImportError as e:
        print(f"[pitch_lines] ⚠️ 無法載入 pxr: {e}")
        return False

    scale = _get_stage_scale(stage)
    line_len_u = goal_line_length * scale
    line_w_u = goal_line_width * scale
    line_h_u = line_height * scale
    z = line_h_u / 2

    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        UsdGeom.Xform.Define(stage, parent_path)

    for idx, ((goal_x, goal_y), (rel_x, rel_y)) in enumerate(zip(goal_centers, goal_relative_offsets), start=1):
        px = (goal_x + rel_x) * scale
        py = (goal_y + rel_y) * scale
        path = f"{parent_path}/goal_reference_line_{idx}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube.GetPrim())
        scale_op = xform.AddScaleOp()
        scale_op.Set(Gf.Vec3d(line_len_u, line_w_u, line_h_u))
        trans_op = xform.AddTranslateOp()
        trans_op.Set(Gf.Vec3d(px, py, z))
        mat_path = f"{path}_mat"
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        binding.Bind(mat)
        UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*line_color)])
        collision_api = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collision_api.CreateCollisionEnabledAttr(False)

    print(
        f"[pitch_lines] 已建立球網參考線：length={goal_line_length}m, width={goal_line_width}m, "
        f"offsets={goal_relative_offsets}, centers={goal_centers}"
    )
    return True


def create_simple_debug_lines(
    stage,
    parent_path: str = "/World/PitchLines",
    center: Tuple[float, float] = (0.0, 0.0),
    circle_radius: float = 1.5,
    line_width: float = DEFAULT_LINE_WIDTH,
    line_height: float = DEFAULT_LINE_HEIGHT,
    line_color: Tuple[float, float, float] = (127.0 / 255.0, 127.0 / 255.0, 127.0 / 255.0),
    draw_pitch_boundary: bool = True,
    draw_goal_reference_lines: bool = False,
    goal_centers: Tuple[Tuple[float, float], Tuple[float, float]] = ((-2.5, 3.0), (2.5, -3.0)),
    goal_relative_offsets: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, -0.45), (0.0, 0.45)),
    goal_line_length: float = 2.0,
    goal_line_width: float = DEFAULT_LINE_WIDTH * 0.5,
    goal_line_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    **kwargs,
) -> bool:
    """
    建立標線：機器人周圍圓圈 + 球場四條邊線。
    對外保留此函數名稱以相容現有呼叫（sim_main、test_* 腳本）。
    """
    ok1 = create_robot_circle(
        stage,
        center=center,
        radius=circle_radius,
        parent_path=parent_path,
        line_width=line_width,
        line_height=line_height,
        line_color=line_color,
    )
    ok2 = True
    ok3 = True
    if draw_goal_reference_lines:
        ok3 = create_goal_reference_lines(
            stage,
            parent_path=parent_path,
            goal_centers=goal_centers,
            goal_relative_offsets=goal_relative_offsets,
            goal_line_length=goal_line_length,
            goal_line_width=goal_line_width,
            line_height=line_height,
            line_color=goal_line_color,
        )
    return ok1 and ok2 and ok3


def create_pitch_lines_in_stage(
    stage,
    parent_path: str = "/World/PitchLines",
    center: Tuple[float, float] = (0.0, 0.0),
    scale: float = 0.17,
    line_width: float = FIFA_LINE_WIDTH,
    line_height: float = 0.005,
    line_color: Tuple[float, float, float] = (127.0 / 255.0, 127.0 / 255.0, 127.0 / 255.0),
    axis_mode: str = "z_up",
) -> bool:
    """
    在 USD stage 中建立完整 FIFA 標準足球場白色標線。
    scale 縮放球場尺寸（供縮小版球場使用），line_width 保持正常寬度不縮放，
    使縮小場地上標線仍清晰可辨。球與球門尺寸不變，僅場地佈局縮小。

    需在 Isaac Sim 環境內執行（含 pxr、omni 模組）。

    Args:
        stage: USD stage（例如 omni.usd.get_context().get_stage()）
        parent_path: 所有標線的父 prim 路徑
        center: 球場中心 (x, y)
        scale: FIFA 縮放係數，球場尺寸按此縮小（球門與球不變）
        line_width: 標線寬度 (m)，預設 0.12 不隨 scale 縮放，保持正常可見
        line_height: 標線厚度（略高於地面以避免 z-fighting）
        line_color: RGB 標線顏色，預設白色
        axis_mode: "z_up"=地面 XY 平面(Isaac Sim 預設)，"y_up"=地面 XZ 平面

    Returns:
        True 若成功建立，False 若發生錯誤
    """
    try:
        from pxr import Gf, Sdf, UsdGeom, UsdShade
    except ImportError as e:
        print(f"[pitch_lines] ⚠️ 無法載入 pxr，跳過標線建立: {e}")
        return False

    w = line_width  # 不隨 scale 縮放，保持正常寬度
    h = line_height
    cx, cy = center

    # 場地半長、半寬
    half_len = (FIFA_PITCH_LENGTH / 2) * scale
    half_wid = (FIFA_PITCH_WIDTH / 2) * scale
    penalty_len = FIFA_PENALTY_LENGTH * scale
    penalty_wid = (FIFA_PENALTY_WIDTH / 2) * scale
    goal_len = FIFA_GOAL_LENGTH * scale
    goal_wid = (FIFA_GOAL_WIDTH / 2) * scale
    center_radius = FIFA_CENTER_RADIUS * scale
    corner_arc_radius = FIFA_CORNER_ARC_RADIUS * scale
    penalty_spot_dist = FIFA_PENALTY_SPOT_DIST * scale
    penalty_arc_radius = FIFA_PENALTY_ARC_RADIUS * scale

    def _add_arc_segments(name_prefix: str, arc_cx: float, arc_cy: float, radius: float,
                          angle_start: float, angle_end: float, n_seg: int = 12):
        """畫圓弧（多段直線近似），從 angle_start 到 angle_end 弧度。"""
        for i in range(n_seg):
            t1 = i / n_seg
            t2 = (i + 1) / n_seg
            a1 = angle_start + (angle_end - angle_start) * t1
            a2 = angle_start + (angle_end - angle_start) * t2
            x1 = arc_cx + radius * math.cos(a1)
            y1 = arc_cy + radius * math.sin(a1)
            x2 = arc_cx + radius * math.cos(a2)
            y2 = arc_cy + radius * math.sin(a2)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            seg_len = 2 * radius * math.sin((a2 - a1) / 2)
            seg_angle = (a1 + a2) / 2 + math.pi / 2
            _add_line(f"{name_prefix}_{i}", seg_len, w, mid_x, mid_y, seg_angle)

    def _add_line(name: str, length: float, width: float, px: float, py: float, angle: float):
        """建立單一標線立方體。length=沿線方向，width=線寬，h=厚度。angle 為繞法線的弧度。
        z_up: 地面 XY，法線 Z；y_up: 地面 XZ，法線 Y
        """
        path = f"{parent_path}/{name}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)  # 單位立方體 (-0.5,-0.5,-0.5) 到 (0.5,0.5,0.5)
        xform = UsdGeom.Xformable(cube.GetPrim())
        if axis_mode == "y_up":
            # 地面 XZ，Y 為上。標線平鋪於 XZ，Scale(length_x, h, width_z)，繞 Y 旋轉
            scale_op = xform.AddScaleOp()
            scale_op.Set(Gf.Vec3d(length, h, width))
            rot_op = xform.AddRotateYOp()
            rot_op.Set(math.degrees(angle))
            trans_op = xform.AddTranslateOp()
            trans_op.Set(Gf.Vec3d(px, h / 2, py))
        else:
            # 地面 XY，Z 為上（Isaac Sim 預設）
            scale_op = xform.AddScaleOp()
            scale_op.Set(Gf.Vec3d(length, width, h))
            rot_op = xform.AddRotateZOp()
            rot_op.Set(math.degrees(angle))
            trans_op = xform.AddTranslateOp()
            trans_op.Set(Gf.Vec3d(px, py, h / 2))
        # 材質
        mat_path = f"{path}_mat"
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(line_color)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        binding.Bind(mat)
        UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*line_color)])

    # 確保父 prim 存在（置於 env_0 下與球/球門同層，確保視埠渲染）
    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        UsdGeom.Xform.Define(stage, parent_path)

    # 邊線與球門線（外框四條）
    # 球門線 (68m)：在 Y = cy±half_len，沿 X 方向
    _add_line("line_goal_top", FIFA_PITCH_WIDTH * scale, w, cx, cy + half_len, 0.0)
    _add_line("line_goal_bottom", FIFA_PITCH_WIDTH * scale, w, cx, cy - half_len, 0.0)
    # 邊線 (105m)：在 X = cx±half_wid，沿 Y 方向
    _add_line("line_touch_left", FIFA_PITCH_LENGTH * scale, w, cx - half_wid, cy, math.pi / 2)
    _add_line("line_touch_right", FIFA_PITCH_LENGTH * scale, w, cx + half_wid, cy, math.pi / 2)

    # 中線：沿 X，長度 68m
    _add_line("line_center", FIFA_PITCH_WIDTH * scale, w, cx, cy, 0.0)

    # 中圈（用多段近似）
    n_arc = 24
    for i in range(n_arc):
        a1 = 2 * math.pi * i / n_arc
        a2 = 2 * math.pi * (i + 1) / n_arc
        x1 = cx + center_radius * math.cos(a1)
        y1 = cy + center_radius * math.sin(a1)
        x2 = cx + center_radius * math.cos(a2)
        y2 = cy + center_radius * math.sin(a2)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        seg_len = 2 * center_radius * math.sin(math.pi / n_arc)
        seg_angle = (a1 + a2) / 2 + math.pi / 2
        _add_line(f"center_arc_{i}", seg_len, w, mid_x, mid_y, seg_angle)

    # 禁區 + 球門區（上下球門各一；禁區 16.5m 深 x 40.3m 寬，球門區 5.5m 深 x 18.32m 寬）
    # top=+Y 端（攻方球門），bottom=-Y 端
    for side, sgn in [("bottom", -1), ("top", 1)]:
        goal_line_y = cy + sgn * half_len
        pen_front_y = goal_line_y - sgn * penalty_len  # 禁區前線朝向中線
        pen_center_y = (goal_line_y + pen_front_y) / 2
        # 禁區前線（平行球門線，沿 X）、禁區左右側線（沿 Y）
        _add_line(f"penalty_front_{side}", penalty_wid * 2, w, cx, pen_front_y, 0.0)
        _add_line(f"penalty_side_left_{side}", penalty_len, w, cx - penalty_wid, pen_center_y, math.pi / 2)
        _add_line(f"penalty_side_right_{side}", penalty_len, w, cx + penalty_wid, pen_center_y, math.pi / 2)
        # 球門區
        goal_front_y = goal_line_y - sgn * goal_len
        goal_center_y = (goal_line_y + goal_front_y) / 2
        _add_line(f"goal_front_{side}", goal_wid * 2, w, cx, goal_front_y, 0.0)
        _add_line(f"goal_side_left_{side}", goal_len, w, cx - goal_wid, goal_center_y, math.pi / 2)
        _add_line(f"goal_side_right_{side}", goal_len, w, cx + goal_wid, goal_center_y, math.pi / 2)

        # 罰球弧（以罰球點為圓心 9.15m，僅畫禁區外的弧，弧線朝中場）
        penalty_spot_y = goal_line_y - sgn * penalty_spot_dist
        d_over_r = (penalty_len - penalty_spot_dist) / penalty_arc_radius  # 5.5/9.15
        d_over_r = min(max(d_over_r, 0.0), 1.0)
        alpha = math.asin(d_over_r)
        if sgn > 0:  # top 球門：弧朝中場 (-Y)，需 sin(θ) < -d/r
            arc_start = math.pi + alpha
            arc_end = 2 * math.pi - alpha
        else:  # bottom 球門：弧朝中場 (+Y)，需 sin(θ) > d/r
            arc_start = alpha
            arc_end = math.pi - alpha
        _add_arc_segments(f"penalty_arc_{side}", cx, penalty_spot_y, penalty_arc_radius, arc_start, arc_end, n_seg=16)

    # 角球弧（四角各 1/4 圓，半徑 1m，弧線在場內）
    # 角位：(x,y) = (cx±half_wid, cy±half_len)
    corners = [
        ("tl", cx - half_wid, cy + half_len, 0, -math.pi / 2),           # 左上
        ("tr", cx + half_wid, cy + half_len, math.pi, math.pi / 2),       # 右上
        ("bl", cx - half_wid, cy - half_len, -math.pi / 2, 0),            # 左下
        ("br", cx + half_wid, cy - half_len, math.pi / 2, math.pi),       # 右下
    ]
    for cname, corn_x, corn_y, a_start, a_end in corners:
        _add_arc_segments(f"corner_arc_{cname}", corn_x, corn_y, corner_arc_radius, a_start, a_end, n_seg=8)

    print(f"[pitch_lines] 已建立完整 FIFA 標線於 {parent_path}，中心 {center}，縮放 {scale} "
          f"(場地約 {FIFA_PITCH_LENGTH*scale:.1f}m x {FIFA_PITCH_WIDTH*scale:.1f}m，線寬 {w*100:.1f}cm)")
    return True
