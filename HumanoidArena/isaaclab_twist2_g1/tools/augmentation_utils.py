import re
import random
import zlib
from typing import Any


def _usd_modules():
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    return omni, UsdLux, UsdGeom, Gf, Sdf


def _get_stage():
    omni, _UsdLux, _UsdGeom, _Gf, _Sdf = _usd_modules()
    return omni.usd.get_context().get_stage()

# ------------------------------
# 通用安全设置属性函数，避免重复创建属性
def safe_set_attr(prim, attr_name, value, usd_type):
    attr = prim.GetAttribute(attr_name)
    if not attr.IsValid():
        attr = prim.CreateAttribute(attr_name, usd_type)
    attr.Set(value)


def _set_xform_attrs(prim, *, rotation=None, position=None):
    _omni, _UsdLux, _UsdGeom, Gf, Sdf = _usd_modules()
    new_ops = []

    if position is not None:
        safe_set_attr(
            prim,
            "xformOp:translate",
            Gf.Vec3f(*(float(v) for v in position)),
            Sdf.ValueTypeNames.Float3,
        )
        new_ops.append("xformOp:translate")

    if rotation is not None:
        safe_set_attr(
            prim,
            "xformOp:rotateXYZ",
            Gf.Vec3f(*(float(v) for v in rotation)),
            Sdf.ValueTypeNames.Float3,
        )
        new_ops.append("xformOp:rotateXYZ")

    if not new_ops:
        return

    existing_order = []
    order_attr = prim.GetAttribute("xformOpOrder")
    if order_attr.IsValid():
        existing_order = [str(value) for value in (order_attr.Get() or [])]
    merged_order = [value for value in existing_order if value not in new_ops]
    merged_order.extend(new_ops)
    safe_set_attr(prim, "xformOpOrder", merged_order, Sdf.ValueTypeNames.TokenArray)


def _as_light_api(prim, type_name: str, UsdLux):
    if type_name == "DomeLight":
        return UsdLux.DomeLight(prim)
    if type_name == "DistantLight":
        return UsdLux.DistantLight(prim)
    if type_name == "SphereLight":
        return UsdLux.SphereLight(prim)
    if type_name == "RectLight":
        return UsdLux.RectLight(prim)
    return None


def _set_if_factory_exists(light, factory_name: str, value: Any) -> None:
    factory = getattr(light, factory_name, None)
    if callable(factory):
        factory().Set(value)

# ------------------------------
# 修改光源属性（颜色、强度、旋转、位置等）
def update_light(
    prim_path: str,
    color=(1.0, 1.0, 1.0),
    intensity=5000.0,
    rotation=(0.0, 0.0, 0.0),
    position=None,
    radius=None,
    enabled=None,
    temperature=None,
    cast_shadows=None,
):
    """
    更新光源属性，支持不同光源类型。
    支持参数包括颜色、强度、旋转角度、位置、半径、是否开启、色温、阴影开启等。

    Args:
        prim_path: USD Prim 路径，如 "/World/light"
        color: 光颜色 RGB tuple，范围0-1
        intensity: 光强度
        rotation: 旋转角 (度)，tuple(x,y,z)
        position: 位置坐标 tuple(x,y,z) 或 None（不改位置）
        radius: 仅 SphereLight 支持，半径
        enabled: 是否启用光源，bool 或 None
        temperature: 色温，仅部分光源支持
        cast_shadows: 是否投射阴影，bool 或 None
    """
    _omni, UsdLux, _UsdGeom, Gf, Sdf = _usd_modules()
    stage = _get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"[update_light] ❌ 找不到光源 Prim: {prim_path}")

    type_name = prim.GetTypeName()
    print(f"[update_light] ✅ 光源类型: {type_name}")

    _set_xform_attrs(prim, rotation=rotation, position=position)

    # 识别光源类型并创建对应接口
    light = _as_light_api(prim, type_name, UsdLux)
    if light is None:
        # 未知光源类型使用通用接口
        print(f"[update_light] ⚠️ 未知光源类型 {type_name}，使用通用接口设置 color 和 intensity")
        safe_set_attr(prim, "color", Gf.Vec3f(*color), Sdf.ValueTypeNames.Color3f)
        safe_set_attr(prim, "intensity", intensity, Sdf.ValueTypeNames.Float)
        return

    # 通用属性设置
    light.CreateColorAttr().Set(Gf.Vec3f(*color))
    light.CreateIntensityAttr().Set(intensity)

    # 有条件地设置其他属性
    if enabled is not None:
        _set_if_factory_exists(light, "CreateEnableAttr", bool(enabled))

    if cast_shadows is not None:
        _set_if_factory_exists(light, "CreateShadowEnableAttr", bool(cast_shadows))

    if temperature is not None:
        _set_if_factory_exists(light, "CreateTemperatureAttr", temperature)

    if radius is not None and type_name == "SphereLight":
        light.CreateRadiusAttr().Set(radius)

    print(f"[update_light] ✅ 光源 {prim_path} 设置完成")


def replace_light_with_distant(
    prim_path: str = "/World/light",
    color=(0.75, 0.75, 0.75),
    intensity=5000.0,
    angle=15.0,
    position=(-4.0, -1.0, 18.0),
    rotation=(0.0, 0.0, 0.0),
    enabled=True,
    cast_shadows=True,
):
    """Replace the light prim with a DistantLight that has a meaningful direction."""

    _omni, UsdLux, _UsdGeom, _Gf, _Sdf = _usd_modules()
    stage = _get_stage()
    if stage is None:
        raise RuntimeError("[replace_light_with_distant] USD Stage 未初始化")

    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid() and prim.GetTypeName() != "DistantLight":
        stage.RemovePrim(prim_path)

    light = UsdLux.DistantLight.Define(stage, prim_path)
    prim = light.GetPrim()
    light.CreateAngleAttr().Set(float(angle))
    update_light(
        prim_path=prim_path,
        color=color,
        intensity=float(intensity),
        rotation=rotation,
        position=position,
        enabled=enabled,
        cast_shadows=cast_shadows,
    )
    print(f"[replace_light_with_distant] DistantLight ready: {prim_path}")
    return prim


def _sample_range(rng: random.Random, value_range, default: float = 0.0) -> float:
    if value_range is None:
        return float(default)
    if len(value_range) != 2:
        raise ValueError(f"range must contain [min, max], got: {value_range!r}")
    low, high = float(value_range[0]), float(value_range[1])
    low, high = min(low, high), max(low, high)
    return float(rng.uniform(low, high)) if high > low else low


def sample_light_randomization_from_range(
    seed: int,
    rotation_ranges: dict,
    intensity_range: list | tuple | None = None,
    color_range: dict | None = None,
    position_range: dict | None = None,
) -> dict[str, Any]:
    """Sample deterministic light parameters from config ranges.

    Rotation is returned as USD rotateXYZ degrees: (roll_x, pitch_y, yaw_z).
    """

    rng = random.Random(int(seed) & 0xFFFFFFFFFFFFFFFF)
    rotation_ranges = rotation_ranges or {}
    yaw_deg = _sample_range(rng, rotation_ranges.get("yaw_deg"), 0.0)
    pitch_deg = _sample_range(rng, rotation_ranges.get("pitch_deg"), 0.0)
    roll_deg = _sample_range(rng, rotation_ranges.get("roll_deg"), 0.0)

    sampled: dict[str, Any] = {
        "rotation": (roll_deg, pitch_deg, yaw_deg),
        "intensity": None,
        "color": None,
        "position": None,
    }
    if intensity_range is not None:
        sampled["intensity"] = _sample_range(rng, intensity_range, 5000.0)
    if color_range is not None:
        sampled["color"] = (
            _sample_range(rng, color_range.get("r"), 0.75),
            _sample_range(rng, color_range.get("g"), 0.75),
            _sample_range(rng, color_range.get("b"), 0.75),
        )
    if position_range is not None:
        sampled["position"] = (
            _sample_range(rng, position_range.get("x"), 0.0),
            _sample_range(rng, position_range.get("y"), 0.0),
            _sample_range(rng, position_range.get("z"), 0.0),
        )
    return sampled


def randomize_light_from_range(
    prim_path: str,
    seed: int,
    rotation_ranges: dict,
    intensity_range: list | tuple | None = None,
    color_range: dict | None = None,
    position_range: dict | None = None,
) -> dict[str, Any]:
    sampled = sample_light_randomization_from_range(
        seed=seed,
        rotation_ranges=rotation_ranges,
        intensity_range=intensity_range,
        color_range=color_range,
        position_range=position_range,
    )
    update_light(
        prim_path=prim_path,
        color=sampled["color"] or (0.75, 0.75, 0.75),
        intensity=sampled["intensity"] if sampled["intensity"] is not None else 5000.0,
        rotation=sampled["rotation"],
        position=sampled["position"],
        enabled=True,
        cast_shadows=True,
    )
    print(
        "[randomize_light_from_range] "
        f"prim={prim_path} seed={seed} rotation_xyz={sampled['rotation']} "
        f"intensity={sampled['intensity']} color={sampled['color']} position={sampled['position']}"
    )
    return sampled


_ENV_GROUP_RE = re.compile(r"(/World/envs/env_\d+)(?:/|$)")


def _derive_group_seed(seed: int, group_name: str) -> int:
    group_hash = zlib.crc32(str(group_name).encode("utf-8")) & 0xFFFFFFFF
    return (int(seed) ^ group_hash) & 0xFFFFFFFFFFFFFFFF


def _path_group_name(prim_path: str) -> str:
    match = _ENV_GROUP_RE.search(str(prim_path))
    if match:
        return match.group(1)
    return "__global__"


def _normalize_unique_paths(prim_paths) -> list[str]:
    return sorted({str(path) for path in prim_paths})


def _sample_color_from_range(rng: random.Random, color_range: dict | None):
    if color_range is None:
        return None
    return (
        _sample_range(rng, color_range.get("r"), 1.0),
        _sample_range(rng, color_range.get("g"), 1.0),
        _sample_range(rng, color_range.get("b"), 1.0),
    )


def sample_rect_light_bar_randomization(
    seed: int,
    prim_paths,
    disable_count: int = 4,
    enabled_intensity_scale_range=None,
    color_range: dict | None = None,
) -> dict[str, Any]:
    """Choose exactly ``disable_count`` RectLights to turn off.

    The disabled count is not random. Only the selected subset and optional
    enabled-light appearance parameters are derived from ``seed``.
    """

    paths = _normalize_unique_paths(prim_paths)
    disabled_count = int(disable_count)
    if disabled_count < 0 or disabled_count > len(paths):
        raise ValueError(
            f"disable_count must be in [0, {len(paths)}], got {disable_count!r}"
        )

    rng = random.Random(int(seed) & 0xFFFFFFFFFFFFFFFF)
    disabled_paths = set(rng.sample(paths, disabled_count)) if disabled_count else set()

    lights = []
    for prim_path in paths:
        is_disabled = prim_path in disabled_paths
        if is_disabled:
            lights.append(
                {
                    "prim_path": prim_path,
                    "enabled": False,
                    "intensity_scale": 0.0,
                    "color": None,
                }
            )
            continue

        lights.append(
            {
                "prim_path": prim_path,
                "enabled": True,
                "intensity_scale": _sample_range(rng, enabled_intensity_scale_range, 1.0),
                "color": _sample_color_from_range(rng, color_range),
            }
        )

    return {
        "seed": int(seed) & 0xFFFFFFFFFFFFFFFF,
        "total_count": len(paths),
        "disable_count": disabled_count,
        "disabled_paths": sorted(disabled_paths),
        "enabled_paths": [path for path in paths if path not in disabled_paths],
        "lights": lights,
    }


def sample_grouped_rect_light_bar_randomization(
    seed: int,
    prim_paths,
    disable_count_per_group: int = 4,
    enabled_intensity_scale_range=None,
    color_range: dict | None = None,
) -> dict[str, Any]:
    groups: dict[str, list[str]] = {}
    for prim_path in _normalize_unique_paths(prim_paths):
        groups.setdefault(_path_group_name(prim_path), []).append(prim_path)

    group_states = []
    disabled_paths = []
    enabled_paths = []
    for group_name, group_paths in sorted(groups.items()):
        group_state = sample_rect_light_bar_randomization(
            seed=_derive_group_seed(seed, group_name),
            prim_paths=group_paths,
            disable_count=disable_count_per_group,
            enabled_intensity_scale_range=enabled_intensity_scale_range,
            color_range=color_range,
        )
        group_state["group"] = group_name
        group_states.append(group_state)
        disabled_paths.extend(group_state["disabled_paths"])
        enabled_paths.extend(group_state["enabled_paths"])

    return {
        "seed": int(seed) & 0xFFFFFFFFFFFFFFFF,
        "total_count": len(_normalize_unique_paths(prim_paths)),
        "disable_count_per_group": int(disable_count_per_group),
        "disabled_paths": sorted(disabled_paths),
        "enabled_paths": sorted(enabled_paths),
        "groups": group_states,
    }


def find_rect_lights_by_path_keywords(
    path_keywords=("cell_light_bars",),
    exclude_keywords=(),
    type_name: str = "RectLight",
) -> list[str]:
    _omni, UsdLux, _UsdGeom, _Gf, _Sdf = _usd_modules()
    from pxr import Usd

    stage = _get_stage()
    if stage is None:
        raise RuntimeError("[find_rect_lights_by_path_keywords] USD Stage unavailable")

    include = tuple(str(keyword).lower() for keyword in (path_keywords or ()))
    exclude = tuple(str(keyword).lower() for keyword in (exclude_keywords or ()))
    matched = []
    try:
        prim_iter = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
    except Exception:
        prim_iter = stage.Traverse()

    for prim in prim_iter:
        if not prim or not prim.IsValid() or not prim.IsActive():
            continue
        is_rect_light = False
        try:
            is_rect_light = bool(prim.IsA(UsdLux.RectLight))
        except Exception:
            is_rect_light = False
        if not is_rect_light and prim.GetTypeName() != str(type_name):
            continue
        prim_path = prim.GetPath().pathString
        path_lower = prim_path.lower()
        if include and not all(keyword in path_lower for keyword in include):
            continue
        if exclude and any(keyword in path_lower for keyword in exclude):
            continue
        matched.append(prim_path)
    return sorted(matched)


def list_light_prim_paths(limit: int = 40) -> list[str]:
    _omni, UsdLux, _UsdGeom, _Gf, _Sdf = _usd_modules()
    from pxr import Usd

    stage = _get_stage()
    if stage is None:
        return []
    try:
        prim_iter = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
    except Exception:
        prim_iter = stage.Traverse()

    paths = []
    for prim in prim_iter:
        if not prim or not prim.IsValid():
            continue
        type_name = prim.GetTypeName()
        path = prim.GetPath().pathString
        name_lower = prim.GetName().lower()
        path_lower = path.lower()
        is_light = "light" in name_lower or "light" in path_lower
        try:
            is_light = is_light or bool(prim.IsA(UsdLux.RectLight))
        except Exception:
            pass
        if not is_light:
            continue
        paths.append(f"{path} [{type_name}] active={prim.IsActive()}")
        if len(paths) >= int(limit):
            break
    return paths


def _attr_value(prim, attr_name: str, default=None):
    attr = prim.GetAttribute(attr_name)
    if not attr.IsValid():
        return default
    value = attr.Get()
    return default if value is None else value


def _light_schema_attr_value(light, factory_name: str, default=None):
    factory = getattr(light, factory_name, None)
    if not callable(factory):
        return default
    attr = factory()
    if not attr.IsValid():
        return default
    value = attr.Get()
    return default if value is None else value


def _color_tuple(value, default=(1.0, 1.0, 1.0)):
    if value is None:
        return tuple(float(v) for v in default)
    return tuple(float(value[index]) for index in range(3))


def _read_light_baseline(prim) -> dict[str, Any]:
    _omni, UsdLux, _UsdGeom, _Gf, _Sdf = _usd_modules()
    light = _as_light_api(prim, prim.GetTypeName(), UsdLux)
    if light is None:
        return {
            "color": _color_tuple(_attr_value(prim, "color", (1.0, 1.0, 1.0))),
            "intensity": float(_attr_value(prim, "intensity", 1.0)),
            "enabled": True,
            "cast_shadows": None,
        }

    return {
        "color": _color_tuple(_light_schema_attr_value(light, "GetColorAttr", (1.0, 1.0, 1.0))),
        "intensity": float(_light_schema_attr_value(light, "GetIntensityAttr", 1.0)),
        "enabled": bool(_light_schema_attr_value(light, "GetEnableAttr", True)),
        "cast_shadows": _light_schema_attr_value(light, "GetShadowEnableAttr", None),
    }


def randomize_rect_lights_by_path_keywords(
    seed: int,
    path_keywords=("cell_light_bars",),
    prim_paths=None,
    exclude_keywords=(),
    type_name: str = "RectLight",
    expected_count_per_group: int | None = None,
    disable_count_per_group: int = 4,
    disabled_intensity: float = 0.0,
    enabled_intensity_scale_range=None,
    color_range: dict | None = None,
    baseline_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    paths = (
        _normalize_unique_paths(prim_paths)
        if prim_paths is not None
        else find_rect_lights_by_path_keywords(
            path_keywords=path_keywords,
            exclude_keywords=exclude_keywords,
            type_name=type_name,
        )
    )
    if not paths and prim_paths is None:
        keywords_lower = {str(keyword).lower() for keyword in (path_keywords or ())}
        if "room" in keywords_lower and "cell_light_bars" in keywords_lower:
            paths = find_rect_lights_by_path_keywords(
                path_keywords=("Room",),
                exclude_keywords=exclude_keywords,
                type_name=type_name,
            )
            if paths:
                print(
                    "[rect_light_randomization] "
                    "fallback matched RectLights under Room after cell_light_bars lookup missed"
                )
    if not paths:
        print("[rect_light_randomization] no matching RectLights found")
        for candidate in list_light_prim_paths():
            print(f"[rect_light_randomization] light_candidate: {candidate}")
        return None

    grouped = sample_grouped_rect_light_bar_randomization(
        seed=seed,
        prim_paths=paths,
        disable_count_per_group=disable_count_per_group,
        enabled_intensity_scale_range=enabled_intensity_scale_range,
        color_range=color_range,
    )

    cache = baseline_cache if baseline_cache is not None else {}
    stage = _get_stage()
    warnings = []
    for group_state in grouped["groups"]:
        if (
            expected_count_per_group is not None
            and int(group_state["total_count"]) != int(expected_count_per_group)
        ):
            warnings.append(
                f"{group_state['group']}:expected={expected_count_per_group}:"
                f"found={group_state['total_count']}"
            )

        for light_state in group_state["lights"]:
            prim_path = light_state["prim_path"]
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                warnings.append(f"{prim_path}:missing")
                continue

            if prim_path not in cache:
                cache[prim_path] = _read_light_baseline(prim)
            baseline = cache[prim_path]

            if bool(light_state["enabled"]):
                color = light_state["color"] or baseline.get("color", (1.0, 1.0, 1.0))
                intensity = float(baseline.get("intensity", 1.0)) * float(
                    light_state["intensity_scale"]
                )
                enabled = bool(baseline.get("enabled", True))
            else:
                color = baseline.get("color", (1.0, 1.0, 1.0))
                intensity = float(disabled_intensity)
                enabled = False

            update_light(
                prim_path=prim_path,
                color=tuple(color),
                intensity=float(intensity),
                rotation=None,
                position=None,
                enabled=enabled,
                cast_shadows=baseline.get("cast_shadows"),
            )
            light_state["intensity"] = float(intensity)

    grouped["warnings"] = warnings
    if warnings:
        print("[rect_light_randomization] warnings: " + ", ".join(warnings))
    print(
        "[rect_light_randomization] "
        f"seed={grouped['seed']} total={grouped['total_count']} "
        f"disabled={grouped['disabled_paths']}"
    )
    return grouped


# ------------------------------
# 修改相机属性，支持焦距、传感器尺寸、曝光、焦点距离等
def augment_camera_appearance(
    camera_path: str,
    focal_length: float = None,
    horizontal_aperture: float = None,
    vertical_aperture: float = None,
    exposure: float = None,
    focus_distance: float = None,
):
    """
    修改静态相机的视觉成像属性，用于增强数据多样性。
    支持调整焦距、视野范围、曝光、景深等。

    Args:
        camera_path: USD 相机 Prim 路径
        focal_length: 焦距（单位 mm）
        horizontal_aperture: 传感器宽度（单位 mm）
        vertical_aperture: 传感器高度（单位 mm）
        exposure: 曝光值
        focus_distance: 聚焦距离（景深效果）
    """
    _omni, _UsdLux, UsdGeom, _Gf, _Sdf = _usd_modules()
    stage = _get_stage()
    prim = stage.GetPrimAtPath(camera_path)

    if not prim or not prim.IsValid():
        raise RuntimeError(f"[augment_camera_appearance] ❌ 找不到相机 prim: {camera_path}")

    camera = UsdGeom.Camera(prim)

    if focal_length is not None:
        camera.CreateFocalLengthAttr().Set(focal_length)

    if horizontal_aperture is not None:
        camera.CreateHorizontalApertureAttr().Set(horizontal_aperture)

    if vertical_aperture is not None:
        camera.CreateVerticalApertureAttr().Set(vertical_aperture)

    if exposure is not None:
        camera.CreateExposureAttr().Set(exposure)

    if focus_distance is not None:
        camera.CreateFocusDistanceAttr().Set(focus_distance)

    print(f"[augment_camera_appearance] ✅ 设置相机 {camera_path} 属性完成")

# --- 新增：批量修改相机（根据名称关键词匹配） ---
def batch_augment_cameras_by_name(
    names,
    focal_length=None,
    horizontal_aperture=None,
    vertical_aperture=None,
    exposure=None,
    focus_distance=None,
):
    """
    批量修改场景中所有名称包含 names 中任意关键词的相机属性。

    参数:
        names: list[str] — 相机名称关键词，如 ["front_cam", "wrist_camera"]
        其余参数: 可为单值（广播）或与匹配的相机数量一致的列表（逐个赋值）
    """
    _omni, _UsdLux, UsdGeom, _Gf, _Sdf = _usd_modules()
    stage = _get_stage()
    if stage is None:
        raise RuntimeError("[batch_augment_cameras_by_name] USD Stage 未初始化")

    matched_prims = []

    def traverse_prim(prim):
        if not prim or not prim.IsValid():
            return
        if prim.IsA(UsdGeom.Camera):
            prim_name = prim.GetName()
            if any(name in prim_name for name in names):
                matched_prims.append(prim)
        for child in prim.GetChildren():
            traverse_prim(child)

    traverse_prim(stage.GetPseudoRoot())

    if not matched_prims:
        print("[batch_augment_cameras_by_name] ⚠️ 没有找到匹配的相机")
        return

    # 参数展开工具
    def normalize(param, default=None):
        if isinstance(param, (list, tuple)):
            if len(param) == len(matched_prims):
                return param
        return [param if param is not None else default] * len(matched_prims)

    focal_lengths = normalize(focal_length)
    horiz_apertures = normalize(horizontal_aperture)
    vert_apertures = normalize(vertical_aperture)
    exposures = normalize(exposure)
    focus_distances = normalize(focus_distance)

    for i, prim in enumerate(matched_prims):
        try:
            augment_camera_appearance(
                camera_path=prim.GetPath().pathString,
                focal_length=focal_lengths[i],
                horizontal_aperture=horiz_apertures[i],
                vertical_aperture=vert_apertures[i],
                exposure=exposures[i],
                focus_distance=focus_distances[i],
            )
        except Exception as e:
            print(f"[batch_augment_cameras_by_name] 修改相机 {prim.GetPath().pathString} 出错: {e}")

    print(f"[batch_augment_cameras_by_name] ✅ 批量修改完成，目标数: {len(matched_prims)}")





# ------------------------------
# Vision Test: Replace existing light with DistantLight and enable randomization
def replace_light_with_distant(
    prim_path: str = "/World/light",
    color=(0.75, 0.75, 0.75),
    intensity=5000.0,
    angle=15.0,
    position=(-4.0, -1.0, 18.0),
    rotation=(0.0, 0.0, 0.0),
):
    """
    Delete existing light prim at prim_path and create a DistantLight in its place.
    Used in vision tests to replace default DomeLight with a directional light
    that supports rotation randomization for lighting robustness evaluation.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("[replace_light] USD Stage not initialized")

    existing = stage.GetPrimAtPath(prim_path)
    old_type = existing.GetTypeName() if existing.IsValid() else "None"
    if existing.IsValid():
        stage.RemovePrim(prim_path)
        print(f"[replace_light] removed existing light: {prim_path} (type: {old_type})")

    distlight = UsdLux.DistantLight.Define(stage, prim_path)
    distlight.CreateColorAttr().Set(Gf.Vec3f(*color))
    distlight.CreateIntensityAttr().Set(intensity)
    distlight.CreateAngleAttr().Set(angle)

    safe_set_attr(distlight.GetPrim(), "xformOp:translate", Gf.Vec3f(*position), Sdf.ValueTypeNames.Float3)
    safe_set_attr(distlight.GetPrim(), "xformOp:rotateXYZ", Gf.Vec3f(*rotation), Sdf.ValueTypeNames.Float3)

    print(f"[replace_light] DistantLight created: {prim_path} (old type: {old_type})")


def randomize_light_from_range(
    prim_path: str = "/World/light",
    seed: int = 0,
    rotation_ranges: dict = None,
    intensity_range: list = None,
    color_range: dict = None,
    position_range: dict = None,
):
    """
    Deterministically randomize DistantLight parameters within given ranges
    based on seed. Same seed always produces same result for reproducibility.

    Args:
        prim_path: Light prim path
        seed: Random seed for deterministic randomization
        rotation_ranges: {"yaw_deg": [min, max], "pitch_deg": [min, max], "roll_deg": [min, max]}
        intensity_range: [min, max]
        color_range: {"r": [min, max], "g": [min, max], "b": [min, max]}
        position_range: {"x": [min, max], "y": [min, max], "z": [min, max]}
    """
    import random as _random
    rng = _random.Random(int(seed))

    rotation = [0.0, 0.0, 0.0]
    if rotation_ranges:
        yaw = rng.uniform(*rotation_ranges.get("yaw_deg", [0.0, 0.0]))
        pitch = rng.uniform(*rotation_ranges.get("pitch_deg", [0.0, 0.0]))
        roll = rng.uniform(*rotation_ranges.get("roll_deg", [0.0, 0.0]))
        rotation = [pitch, roll, yaw]
        print(f"[randomize_light] seed={seed} rot=(pitch={pitch:.1f}, roll={roll:.1f}, yaw={yaw:.1f})")

    intensity = None
    if intensity_range and len(intensity_range) >= 2:
        intensity = rng.uniform(intensity_range[0], intensity_range[1])
        print(f"[randomize_light] seed={seed} intensity={intensity:.0f}")

    color = None
    if color_range:
        r = rng.uniform(*color_range.get("r", [0.75, 0.75]))
        g = rng.uniform(*color_range.get("g", [0.75, 0.75]))
        b = rng.uniform(*color_range.get("b", [0.75, 0.75]))
        color = (r, g, b)
        print(f"[randomize_light] seed={seed} color=({r:.2f}, {g:.2f}, {b:.2f})")

    position = None
    if position_range:
        x = rng.uniform(*position_range.get("x", [-4.0, -4.0]))
        y = rng.uniform(*position_range.get("y", [-1.0, -1.0]))
        z = rng.uniform(*position_range.get("z", [18.0, 18.0]))
        position = (x, y, z)

    update_light(
        prim_path=prim_path,
        rotation=rotation,
        intensity=intensity if intensity is not None else 5000.0,
        color=color if color is not None else (0.75, 0.75, 0.75),
        position=position,
    )
