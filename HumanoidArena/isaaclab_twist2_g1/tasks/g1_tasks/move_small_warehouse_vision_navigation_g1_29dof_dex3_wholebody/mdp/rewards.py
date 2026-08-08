from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tasks.common_scene.base_scene_small_warehouse_vision_navigation import TARGET_INIT_POS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_YELLOW_BOX_TOKEN_GROUPS = (
    ("yellow",),
    ("yellowbox",),
    ("target", "box"),
    ("target", "frame"),
    ("target", "zone"),
)

_MIN_BOX_SIZE_XY_M = 0.10
_MAX_BOX_SIZE_XY_M = 3.00
_MIN_STANDING_ROOT_HEIGHT_M = 0.45
_MIN_STANDING_UP_AXIS_Z = 0.60
_LEFT_FOOT_BODY_NAMES = (
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
)
_RIGHT_FOOT_BODY_NAMES = (
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
)


def _candidate_target_paths(env_idx: int) -> list[str]:
    base = f"/World/envs/env_{env_idx}/TargetSign"
    return [base, f"{base}/PRootNode"]


def _candidate_target_ring_paths(env_idx: int) -> list[str]:
    base = f"/World/envs/env_{env_idx}/TargetSign"
    suffixes = (
        "RootNode/torus_semantic/Torus",
        "torus_semantic/Torus",
        "Root/RootNode/torus_semantic/Torus",
        "PRootNode/RootNode/torus_semantic/Torus",
        "PRootNode/torus_semantic/Torus",
        "PRootNode/Root/RootNode/torus_semantic/Torus",
    )
    return [f"{base}/{suffix}" for suffix in suffixes]


def _read_prim_world_position(path: str) -> tuple[float, float, float] | None:
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception:
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    prim = stage.GetPrimAtPath(path)
    if prim is None or not prim.IsValid() or not prim.IsActive():
        return None

    try:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        matrix = cache.GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        return (float(translation[0]), float(translation[1]), float(translation[2]))
    except Exception:
        return None


def _get_target_positions(env: "ManagerBasedRLEnv") -> list[tuple[float, float, float] | None]:
    positions: list[tuple[float, float, float] | None] = []
    for env_idx in range(env.num_envs):
        target_position = None
        for path in _candidate_target_paths(env_idx):
            target_position = _read_prim_world_position(path)
            if target_position is not None:
                break
        if target_position is None:
            env_origins = getattr(env.scene, "env_origins", None)
            if env_origins is not None:
                target_position = (
                    float(env_origins[env_idx, 0].item()) + float(TARGET_INIT_POS[0]),
                    float(env_origins[env_idx, 1].item()) + float(TARGET_INIT_POS[1]),
                    float(env_origins[env_idx, 2].item()) + float(TARGET_INIT_POS[2]),
                )
            else:
                target_position = (
                    float(TARGET_INIT_POS[0]),
                    float(TARGET_INIT_POS[1]),
                    float(TARGET_INIT_POS[2]),
                )
        positions.append(target_position)
    return positions


def _matches_yellow_box_tokens(path_lower: str, name_lower: str) -> bool:
    haystack = f"{path_lower} {name_lower}"
    return any(all(token in haystack for token in token_group) for token_group in _YELLOW_BOX_TOKEN_GROUPS)


def _box_from_aligned_range(aligned_range) -> tuple[float, float, float, float] | None:
    mn = aligned_range.GetMin()
    mx = aligned_range.GetMax()
    size_x = float(mx[0] - mn[0])
    size_y = float(mx[1] - mn[1])
    if size_x < _MIN_BOX_SIZE_XY_M or size_y < _MIN_BOX_SIZE_XY_M:
        return None
    if size_x > _MAX_BOX_SIZE_XY_M or size_y > _MAX_BOX_SIZE_XY_M:
        return None
    return (
        float(mn[0]),
        float(mx[0]),
        float(mn[1]),
        float(mx[1]),
    )


def _circle_from_aligned_range(aligned_range) -> tuple[float, float, float] | None:
    mn = aligned_range.GetMin()
    mx = aligned_range.GetMax()
    size_x = float(mx[0] - mn[0])
    size_y = float(mx[1] - mn[1])
    diameter = min(size_x, size_y)
    if diameter < _MIN_BOX_SIZE_XY_M:
        return None
    if diameter > _MAX_BOX_SIZE_XY_M:
        return None
    return (
        0.5 * float(mn[0] + mx[0]),
        0.5 * float(mn[1] + mx[1]),
        0.5 * diameter,
    )


def _candidate_box_score(path_lower: str, name_lower: str) -> int:
    haystack = f"{path_lower} {name_lower}"
    score = 0
    if "yellowbox" in haystack or "with_yellowbox" in haystack:
        score += 100
    if "yellow" in haystack:
        score += 60
    if "target" in haystack:
        score += 20
    if "box" in haystack or "frame" in haystack or "zone" in haystack:
        score += 10
    return score


def _load_stage_yellow_box_world(
    stage,
    bbox_cache,
    env_idx: int,
    target_position: tuple[float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    try:
        from pxr import Usd
    except Exception:
        return None

    room_root = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/Room")
    if room_root is None or not room_root.IsValid() or not room_root.IsActive():
        return None

    candidates: list[tuple[int, float, float, tuple[float, float, float, float]]] = []
    target_x = float(target_position[0]) if target_position is not None else None
    target_y = float(target_position[1]) if target_position is not None else None

    for prim in Usd.PrimRange(room_root):
        if prim is None or not prim.IsActive():
            continue
        path_lower = str(prim.GetPath()).lower()
        name_lower = prim.GetName().lower()
        if not _matches_yellow_box_tokens(path_lower, name_lower):
            continue
        try:
            box = _box_from_aligned_range(bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange())
        except Exception:
            continue
        if box is None:
            continue

        center_x = 0.5 * (box[0] + box[1])
        center_y = 0.5 * (box[2] + box[3])
        area_xy = (box[1] - box[0]) * (box[3] - box[2])
        if target_x is not None and target_y is not None:
            dist_to_target = float((center_x - target_x) ** 2 + (center_y - target_y) ** 2)
        else:
            dist_to_target = 0.0
        score = _candidate_box_score(path_lower, name_lower)
        candidates.append((score, dist_to_target, area_xy, box))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates[0][3]


def _load_target_sign_circle_world(
    stage,
    bbox_cache,
    env_idx: int,
) -> tuple[float, float, float] | None:
    try:
        from pxr import Usd
    except Exception:
        return None

    def circle_from_prim(prim) -> tuple[float, float, float] | None:
        if prim is None or not prim.IsValid() or not prim.IsActive():
            return None
        try:
            return _circle_from_aligned_range(bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange())
        except Exception:
            return None

    for path in _candidate_target_ring_paths(env_idx):
        circle = circle_from_prim(stage.GetPrimAtPath(path))
        if circle is not None:
            return circle

    for root_path in _candidate_target_paths(env_idx):
        root = stage.GetPrimAtPath(root_path)
        if root is None or not root.IsValid() or not root.IsActive():
            continue
        for prim in Usd.PrimRange(root):
            if "torus_semantic" not in str(prim.GetPath()).lower():
                continue
            circle = circle_from_prim(prim)
            if circle is not None:
                return circle
    return None


def _load_target_sign_box_world(
    stage,
    bbox_cache,
    env_idx: int,
) -> tuple[float, float, float, float] | None:
    for path in _candidate_target_paths(env_idx):
        prim = stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid() or not prim.IsActive():
            continue
        try:
            box = _box_from_aligned_range(bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange())
        except Exception:
            continue
        if box is not None:
            return box
    return None


def _get_stage_target_regions_world(
    env: "ManagerBasedRLEnv",
    target_positions: list[tuple[float, float, float] | None],
) -> list[tuple[str, tuple[float, ...]] | None]:
    regions: list[tuple[str, tuple[float, ...]] | None] = [None] * env.num_envs
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return regions

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        for env_idx in range(env.num_envs):
            box = _load_stage_yellow_box_world(
                stage,
                bbox_cache,
                env_idx,
                target_positions[env_idx],
            )
            if box is not None:
                regions[env_idx] = ("box", box)
                continue

            circle = _load_target_sign_circle_world(stage, bbox_cache, env_idx)
            if circle is not None:
                regions[env_idx] = ("circle", circle)
                continue

            box = _load_target_sign_box_world(stage, bbox_cache, env_idx)
            if box is not None:
                regions[env_idx] = ("box", box)
    except Exception:
        regions = [None] * env.num_envs

    return regions


def _get_body_positions_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    data = env.scene["robot"].data

    body_link_pose_w = getattr(data, "body_link_pose_w", None)
    if body_link_pose_w is not None:
        return body_link_pose_w[..., 0:3]

    body_pos_w = getattr(data, "body_pos_w", None)
    if body_pos_w is not None:
        return body_pos_w

    body_state_w = getattr(data, "body_state_w", None)
    if body_state_w is not None:
        return body_state_w[..., 0:3]

    raise AttributeError("robot body world-position tensor not found")


def _find_body_indices(body_names: list[str], required_names: tuple[str, ...]) -> tuple[int, ...] | None:
    if not all(name in body_names for name in required_names):
        return None
    return tuple(body_names.index(name) for name in required_names)


def _get_foot_body_indices(env: "ManagerBasedRLEnv") -> tuple[tuple[int, ...], tuple[int, ...]]:
    cached = getattr(env, "_swvn_foot_body_indices", None)
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached

    body_names = list(env.scene["robot"].data.body_names)
    left_indices = _find_body_indices(body_names, _LEFT_FOOT_BODY_NAMES)
    right_indices = _find_body_indices(body_names, _RIGHT_FOOT_BODY_NAMES)
    if left_indices is None or right_indices is None:
        raise ValueError(
            "failed to locate foot bodies in robot.body_names; "
            f"available bodies include: {body_names[:12]}"
        )

    env._swvn_foot_body_indices = (left_indices, right_indices)
    return left_indices, right_indices


def _root_up_axis_z(root_quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat = root_quat_wxyz / torch.linalg.vector_norm(root_quat_wxyz, dim=1, keepdim=True).clamp_min(1e-8)
    x = quat[:, 1]
    y = quat[:, 2]
    return 1.0 - 2.0 * (x * x + y * y)


def _compute_standing_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    root_state = env.scene["robot"].data.root_state_w
    root_height = root_state[:, 2]
    up_axis_z = _root_up_axis_z(root_state[:, 3:7])
    return (root_height >= _MIN_STANDING_ROOT_HEIGHT_M) & (up_axis_z >= _MIN_STANDING_UP_AXIS_Z)


def _point_in_box_xy(
    point_xy: torch.Tensor,
    box_xy: tuple[float, float, float, float],
) -> torch.Tensor:
    x_lo, x_hi, y_lo, y_hi = box_xy
    return (
        (point_xy[:, 0] >= x_lo)
        & (point_xy[:, 0] <= x_hi)
        & (point_xy[:, 1] >= y_lo)
        & (point_xy[:, 1] <= y_hi)
    )


def _point_in_circle_xy(
    point_xy: torch.Tensor,
    circle_xy: tuple[float, float, float],
) -> torch.Tensor:
    center_x, center_y, radius = circle_xy
    dx = point_xy[:, 0] - center_x
    dy = point_xy[:, 1] - center_y
    return dx * dx + dy * dy <= radius * radius


def _point_in_target_region_xy(
    point_xy: torch.Tensor,
    region: tuple[str, tuple[float, ...]],
) -> torch.Tensor:
    region_type, values = region
    if region_type == "circle":
        return _point_in_circle_xy(point_xy, (values[0], values[1], values[2]))
    return _point_in_box_xy(point_xy, (values[0], values[1], values[2], values[3]))


def compute_success_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    target_positions = _get_target_positions(env)
    target_regions_world = _get_stage_target_regions_world(env, target_positions)

    body_positions_w = _get_body_positions_w(env)
    left_foot_indices, right_foot_indices = _get_foot_body_indices(env)
    left_foot_xy = body_positions_w[:, list(left_foot_indices), 0:2]
    right_foot_xy = body_positions_w[:, list(right_foot_indices), 0:2]

    standing = _compute_standing_mask(env)
    feet_in_target = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    for env_idx in range(env.num_envs):
        target_region = target_regions_world[env_idx]
        if target_region is None:
            continue

        left_inside = _point_in_target_region_xy(left_foot_xy[env_idx], target_region).all()
        right_inside = _point_in_target_region_xy(right_foot_xy[env_idx], target_region).all()
        feet_in_target[env_idx] = bool(left_inside.item() and right_inside.item())

    return feet_in_target & standing


def compute_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Binary state reward for the warehouse navigation task.

    Reward is ``1`` only when both feet are inside the target region and the
    robot remains upright. Otherwise the reward is ``0``.
    """

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    success = compute_success_mask(env)
    reward[success] = 1.0

    return reward


__all__ = ["compute_reward", "compute_success_mask"]
