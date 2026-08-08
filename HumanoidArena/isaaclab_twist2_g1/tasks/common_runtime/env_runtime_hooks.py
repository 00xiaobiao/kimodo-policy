"""Reusable task-side scene adjustment helpers."""

from __future__ import annotations

import zlib
from typing import Any


def apply_scene_filter_from_cfg(env_cfg: Any) -> None:
    scene_deactivate_keywords = tuple(getattr(env_cfg, "scene_deactivate_keywords", ()))
    scene_deactivate_exclude_keywords = tuple(
        getattr(env_cfg, "scene_deactivate_exclude_keywords", ())
    )
    if not scene_deactivate_keywords:
        return
    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        deactivated_paths = []
        keywords = tuple(k.lower() for k in scene_deactivate_keywords)
        exclude_keywords = tuple(k.lower() for k in scene_deactivate_exclude_keywords)
        for prim in stage.Traverse():
            if not prim.IsActive():
                continue
            prim_path = prim.GetPath().pathString
            prim_name = prim.GetName()
            path_lower = prim_path.lower()
            name_lower = prim_name.lower()
            if any((k in path_lower) or (k in name_lower) for k in exclude_keywords):
                continue
            if any((k in path_lower) or (k in name_lower) for k in keywords):
                prim.SetActive(False)
                deactivated_paths.append(prim_path)
        print(
            f"[scene_filter] deactivate keywords={scene_deactivate_keywords}, "
            f"exclude={scene_deactivate_exclude_keywords}, count={len(deactivated_paths)}"
        )
        for prim_path in deactivated_paths[:20]:
            print(f"[scene_filter] deactivated: {prim_path}")
    except Exception as exc:
        print(f"[scene_filter] deactivate failed: {exc}")


def apply_scene_reposition_from_cfg(env_cfg: Any) -> None:
    scene_reposition_rules = tuple(getattr(env_cfg, "scene_reposition_rules", ()))
    if not scene_reposition_rules:
        return
    try:
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        moved_items = []
        all_active_prims = [prim for prim in stage.Traverse() if prim.IsActive()]
        for rule in scene_reposition_rules:
            keywords = tuple(str(k).lower() for k in rule.get("keywords", ()))
            if not keywords:
                continue
            rule_name = str(rule.get("name", "rule"))
            safe_rule_name = "".join(ch if ch.isalnum() else "_" for ch in rule_name.lower())
            offset = tuple(float(v) for v in rule.get("offset", (0.0, 0.0, 0.0)))
            if abs(offset[0]) + abs(offset[1]) + abs(offset[2]) < 1e-9:
                continue
            candidates = []
            for prim in all_active_prims:
                prim_path = prim.GetPath().pathString
                prim_name = prim.GetName()
                path_lower = prim_path.lower()
                name_lower = prim_name.lower()
                if "joint" in path_lower or "/materials" in path_lower:
                    continue
                if not any((k in name_lower) or (k in path_lower) for k in keywords):
                    continue
                depth = prim_path.count("/")
                candidates.append((depth, len(prim_path), prim_path))
            if not candidates:
                continue
            candidates.sort(key=lambda item: (item[0], item[1]))
            target_path = candidates[0][2]
            target_prim = stage.GetPrimAtPath(target_path)
            if not target_prim.IsValid():
                continue
            marker_attr = target_prim.GetAttribute(
                f"userProperties:humanoidarena_reposition_applied_{safe_rule_name}"
            )
            if marker_attr.IsValid() and bool(marker_attr.Get()):
                continue
            xformable = UsdGeom.Xformable(target_prim)
            translate_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
            current_t = translate_op.Get()
            if current_t is None:
                current_t = Gf.Vec3d(0.0, 0.0, 0.0)
            new_t = Gf.Vec3d(
                float(current_t[0]) + offset[0],
                float(current_t[1]) + offset[1],
                float(current_t[2]) + offset[2],
            )
            translate_op.Set(new_t)
            if not marker_attr.IsValid():
                marker_attr = target_prim.CreateAttribute(
                    f"userProperties:humanoidarena_reposition_applied_{safe_rule_name}",
                    Sdf.ValueTypeNames.Bool,
                    custom=True,
                )
            marker_attr.Set(True)
            moved_items.append((target_path, rule_name, offset))
        print(
            f"[scene_filter] reposition rules={len(scene_reposition_rules)}, moved={len(moved_items)}"
        )
        for prim_path, rule_name, offset in moved_items[:20]:
            print(f"[scene_filter] moved({rule_name}): {prim_path}, offset={offset}")
    except Exception as exc:
        print(f"[scene_filter] reposition failed: {exc}")


def apply_optional_runtime_augments(args_cli: Any) -> None:
    if getattr(args_cli, "modify_light", False):
        try:
            from tools.augmentation_utils import update_light

            update_light(
                prim_path="/World/light",
                color=(0.75, 0.75, 0.75),
                intensity=500.0,
                radius=0.1,
                enabled=True,
                cast_shadows=True,
            )
        except Exception as exc:
            print(f"[env_runtime] modify_light failed: {exc}")

    if getattr(args_cli, "modify_camera", False):
        try:
            from tools.augmentation_utils import batch_augment_cameras_by_name

            batch_augment_cameras_by_name(
                names=["front_cam"],
                focal_length=3.0,
                horizontal_aperture=22.0,
                vertical_aperture=16.0,
                exposure=0.8,
                focus_distance=1.2,
            )
        except Exception as exc:
            print(f"[env_runtime] modify_camera failed: {exc}")
"""Vision test hooks: replace DomeLight with DistantLight and randomize per episode."""


def setup_vision_test_light(prim_path: str = "/World/light") -> None:
    """
    Called once after gym.make to replace the default light (typically DomeLight)
    with a DistantLight that supports rotation-based randomization.

    This enables vision-level testing where light direction is randomized
    per episode to evaluate model robustness to lighting variations.
    """
    try:
        from tools.augmentation_utils import replace_light_with_distant

        replace_light_with_distant(
            prim_path=prim_path,
            color=(0.75, 0.75, 0.75),
            intensity=5000.0,
            angle=15.0,
            position=(-4.0, -1.0, 18.0),
            rotation=(0.0, 0.0, 0.0),
        )
        print(f"[vision_test] DistantLight setup complete: {prim_path}")
    except Exception as exc:
        print(f"[vision_test] setup_vision_test_light failed: {exc}")


def apply_vision_light_randomization(
    prim_path: str = "/World/light",
    episode_seed: int = 0,
    rotation_ranges: dict = None,
    intensity_range: list = None,
    color_range: dict = None,
    position_range: dict = None,
) -> None:
    """
    Called per episode to deterministically randomize the DistantLight parameters.

    Uses episode_seed to derive reproducible random values within the given ranges.
    Same episode_seed always produces the same lighting configuration.

    Args:
        prim_path: Light prim path to modify
        episode_seed: Seed for deterministic randomization
        rotation_ranges: {"yaw_deg": [min, max], "pitch_deg": [min, max], "roll_deg": [min, max]}
        intensity_range: [min, max] or None
        color_range: {"r": [min, max], "g": [min, max], "b": [min, max]} or None
        position_range: {"x": [min, max], "y": [min, max], "z": [min, max]} or None
    """
    if not rotation_ranges:
        return

    try:
        from tools.augmentation_utils import randomize_light_from_range

        # Derive sub-seed for light randomization to avoid correlation with object seed
        light_seed = int(episode_seed) ^ 0x1A2B3C4D
        randomize_light_from_range(
            prim_path=prim_path,
            seed=light_seed,
            rotation_ranges=rotation_ranges,
            intensity_range=intensity_range,
            color_range=color_range,
            position_range=position_range,
        )
    except Exception as exc:
        print(f"[vision_test] apply_vision_light_randomization failed: {exc}")
