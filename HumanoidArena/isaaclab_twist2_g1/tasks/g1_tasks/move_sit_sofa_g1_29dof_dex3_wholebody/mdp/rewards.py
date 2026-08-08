from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_SOFA_SEAT_NAME_TOKENS = ("sofa_seat", "seatcushion", "l_seat", "r_seat")
_SOFA_OBJECT_NAME_TOKENS = ("sofa", "couch")
_MIN_SEAT_SIZE_XY_M = 0.20
_MAX_SEAT_SIZE_XY_M = 4.00
_SEAT_BODY_CONTACT_FORCE_THRESHOLD_N = 5.0
_SEAT_BODY_Z_BELOW_BOX_M = 0.10
_SEAT_BODY_Z_ABOVE_BOX_M = 0.35
_SEATED_GEOMETRY_FALLBACK_MIN_BODIES = 2
_SIT_BODY_PREFERRED_NAMES = (
    "pelvis",
    "pelvis_contour_link",
    "imu_in_pelvis",
    "waist_yaw_link",
    "waist_roll_link",
    "waist_pitch_link",
    "torso_link",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
)


def _box_from_aligned_range(aligned_range) -> tuple[float, float, float, float, float, float] | None:
    mn = aligned_range.GetMin()
    mx = aligned_range.GetMax()
    size_x = float(mx[0] - mn[0])
    size_y = float(mx[1] - mn[1])
    size_z = float(mx[2] - mn[2])
    if size_x < _MIN_SEAT_SIZE_XY_M or size_y < _MIN_SEAT_SIZE_XY_M or size_z <= 0.0:
        return None
    if size_x > _MAX_SEAT_SIZE_XY_M or size_y > _MAX_SEAT_SIZE_XY_M:
        return None
    return (
        float(mn[0]),
        float(mx[0]),
        float(mn[1]),
        float(mx[1]),
        float(mn[2]),
        float(mx[2]),
    )


def _candidate_sofa_seat_score(name_lower: str, path_lower: str) -> int:
    haystack = f"{name_lower} {path_lower}"
    score = 0
    if name_lower == "sofa_seat":
        score += 120
    if "sofa_seat" in haystack:
        score += 100
    if "seatcushion" in haystack:
        score += 80
    if name_lower in {"l_seat", "r_seat"}:
        score += 70
    if any(token in haystack for token in _SOFA_OBJECT_NAME_TOKENS):
        score += 40
    if "seat" in haystack:
        score += 15
    if "chair" in haystack:
        score -= 80
    return score


def _is_sofa_seat_candidate(name_lower: str, path_lower: str) -> bool:
    haystack = f"{name_lower} {path_lower}"
    if any(token in haystack for token in _SOFA_SEAT_NAME_TOKENS):
        return True
    return any(token in haystack for token in _SOFA_OBJECT_NAME_TOKENS) and "seat" in haystack


def _load_stage_sofa_seat_world(
    stage,
    bbox_cache,
    env_idx: int,
) -> tuple[float, float, float, float, float, float] | None:
    try:
        from pxr import Usd
    except Exception:
        return None

    room_root = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/Room")
    if room_root is None or not room_root.IsValid() or not room_root.IsActive():
        return None

    candidates: list[tuple[int, float, tuple[float, float, float, float, float, float]]] = []
    for prim in Usd.PrimRange(room_root):
        if prim is None or not prim.IsActive():
            continue
        name_lower = prim.GetName().lower()
        path_lower = str(prim.GetPath()).lower()
        if not _is_sofa_seat_candidate(name_lower, path_lower):
            continue
        try:
            box = _box_from_aligned_range(bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange())
        except Exception:
            continue
        if box is None:
            continue
        area_xy = (box[1] - box[0]) * (box[3] - box[2])
        score = _candidate_sofa_seat_score(name_lower, path_lower)
        candidates.append((score, area_xy, box))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], -item[1]))
    return candidates[0][2]


def _get_sofa_seat_boxes_world(
    env: "ManagerBasedRLEnv",
) -> list[tuple[float, float, float, float, float, float] | None]:
    cached = getattr(env, "_sit_sofa_seat_boxes_world", None)
    if isinstance(cached, list) and len(cached) == env.num_envs:
        return cached

    boxes: list[tuple[float, float, float, float, float, float] | None] = [None] * env.num_envs
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return boxes

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        for env_idx in range(env.num_envs):
            boxes[env_idx] = _load_stage_sofa_seat_world(stage, bbox_cache, env_idx)
    except Exception:
        boxes = [None] * env.num_envs

    if any(box is not None for box in boxes):
        env._sit_sofa_seat_boxes_world = boxes
    return boxes


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


def _get_body_contact_forces_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    data = env.scene["robot"].data

    body_net_contact_force_w = getattr(data, "body_net_contact_force_w", None)
    if body_net_contact_force_w is not None:
        return body_net_contact_force_w

    body_net_contact_forces_w = getattr(data, "body_net_contact_forces_w", None)
    if body_net_contact_forces_w is not None:
        return body_net_contact_forces_w

    try:
        contact_sensor = env.scene["contact_forces"]
    except Exception:
        contact_sensor = None

    if contact_sensor is not None:
        sensor_data = getattr(contact_sensor, "data", None)
        sensor_net_forces_w = getattr(sensor_data, "net_forces_w", None)
        sensor_body_names = list(getattr(contact_sensor, "body_names", []) or [])
        robot_body_names = list(getattr(data, "body_names", []) or [])
        if sensor_net_forces_w is not None and sensor_body_names and robot_body_names:
            out = torch.zeros(
                env.num_envs,
                len(robot_body_names),
                3,
                device=sensor_net_forces_w.device,
                dtype=sensor_net_forces_w.dtype,
            )
            sensor_name_to_idx = {name: idx for idx, name in enumerate(sensor_body_names)}
            for robot_idx, body_name in enumerate(robot_body_names):
                sensor_idx = sensor_name_to_idx.get(body_name)
                if sensor_idx is not None:
                    out[:, robot_idx, :] = sensor_net_forces_w[:, sensor_idx, :]
            return out

    raise AttributeError("robot body contact-force tensor not found on robot data or contact sensor")


def _get_sit_body_indices(env: "ManagerBasedRLEnv") -> tuple[int, ...]:
    cached = getattr(env, "_sit_sofa_body_indices", None)
    if isinstance(cached, tuple) and len(cached) > 0:
        return cached

    body_names = list(env.scene["robot"].data.body_names)
    indices: list[int] = []
    for name in _SIT_BODY_PREFERRED_NAMES:
        if name in body_names:
            idx = body_names.index(name)
            if idx not in indices:
                indices.append(idx)

    if not indices:
        for idx, name in enumerate(body_names):
            lower_name = name.lower()
            if "pelvis" in lower_name or "hip" in lower_name or "waist" in lower_name:
                indices.append(idx)

    if not indices:
        raise ValueError(
            "failed to locate pelvis/hip proxy bodies in robot.body_names; "
            f"available bodies include: {body_names[:16]}"
        )

    env._sit_sofa_body_indices = tuple(indices)
    env._sit_sofa_body_names = tuple(body_names[idx] for idx in indices)
    return env._sit_sofa_body_indices


def compute_success_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    seat_boxes_world = _get_sofa_seat_boxes_world(env)
    body_positions_w = _get_body_positions_w(env)
    body_contact_forces_w = _get_body_contact_forces_w(env)
    sit_body_indices = list(_get_sit_body_indices(env))

    sit_body_positions = body_positions_w[:, sit_body_indices, :]
    sit_body_contact_norm = torch.linalg.vector_norm(body_contact_forces_w[:, sit_body_indices, :], dim=-1)

    success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for env_idx in range(env.num_envs):
        seat_box = seat_boxes_world[env_idx]
        if seat_box is None:
            continue

        x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = seat_box
        positions = sit_body_positions[env_idx]
        contact_norm = sit_body_contact_norm[env_idx]

        inside_xy = (
            (positions[:, 0] >= x_lo)
            & (positions[:, 0] <= x_hi)
            & (positions[:, 1] >= y_lo)
            & (positions[:, 1] <= y_hi)
        )
        inside_z_band = (
            (positions[:, 2] >= z_lo - _SEAT_BODY_Z_BELOW_BOX_M)
            & (positions[:, 2] <= z_hi + _SEAT_BODY_Z_ABOVE_BOX_M)
        )
        has_contact = contact_norm >= _SEAT_BODY_CONTACT_FORCE_THRESHOLD_N
        seated_geometry = inside_xy & inside_z_band
        seated_with_contact = seated_geometry & has_contact

        if bool(torch.any(seated_with_contact).item()):
            success[env_idx] = True
            continue

        success[env_idx] = (
            int(torch.count_nonzero(seated_geometry).item()) >= _SEATED_GEOMETRY_FALLBACK_MIN_BODIES
        )

    return success


def compute_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    success = compute_success_mask(env)
    reward[success] = 1.0
    return reward


__all__ = ["compute_reward", "compute_success_mask"]
