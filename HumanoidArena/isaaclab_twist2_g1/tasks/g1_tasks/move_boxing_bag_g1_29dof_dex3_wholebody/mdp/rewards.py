from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_BOXING_TARGET_SCENE_KEYS = ("boxing_target", "boxing target", "BoxingTarget")
_BOXING_TARGET_PRIM_NAME = "BoxingTarget"
_BOXING_TARGET_FALLBACK_INIT_POS = (0.6, 0.0, 1.2)
_BOXING_TARGET_DEFAULT_RADIUS_M = 0.10
_HAND_PROXY_RADIUS_M = 0.08
_HAND_HIT_DISTANCE_FALLBACK_M = _BOXING_TARGET_DEFAULT_RADIUS_M + _HAND_PROXY_RADIUS_M

_PUNCH_BODY_NAME_GROUPS = (
    ("left_hand_palm_link", "right_hand_palm_link"),
    ("left_hand_palm", "right_hand_palm"),
    ("left_hand_base_link", "right_hand_base_link"),
    ("left_hand_camera_base_link", "right_hand_camera_base_link"),
    ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    ("left_wrist_pitch_link", "right_wrist_pitch_link"),
    ("left_wrist_roll_link", "right_wrist_roll_link"),
)
_PUNCH_BODY_TOKEN_GROUPS = (
    ("left_hand_palm", "right_hand_palm"),
    ("left_hand_base", "right_hand_base"),
    ("left_hand_camera_base", "right_hand_camera_base"),
    ("left_wrist_yaw", "right_wrist_yaw"),
    ("left_wrist_pitch", "right_wrist_pitch"),
    ("left_wrist_roll", "right_wrist_roll"),
)


def _resolve_boxing_target_asset(env: "ManagerBasedRLEnv"):
    for scene_key in _BOXING_TARGET_SCENE_KEYS:
        try:
            return env.scene[scene_key]
        except Exception:
            continue
    return None


def _get_root_positions_world(asset) -> torch.Tensor:
    data = asset.data

    root_com_pos_w = getattr(data, "root_com_pos_w", None)
    if root_com_pos_w is not None:
        return root_com_pos_w

    root_pos_w = getattr(data, "root_pos_w", None)
    if root_pos_w is not None:
        return root_pos_w

    root_state_w = getattr(data, "root_state_w", None)
    if root_state_w is not None:
        return root_state_w[:, 0:3]

    raise AttributeError("asset world-position tensor not found")


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


def _candidate_target_paths(env_idx: int) -> list[str]:
    base = f"/World/envs/env_{env_idx}/{_BOXING_TARGET_PRIM_NAME}"
    return [base, f"{base}/PRootNode"]


def _read_stage_target_world_position(path: str) -> tuple[float, float, float] | None:
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


def _get_stage_target_positions_world(
    env: "ManagerBasedRLEnv",
) -> list[tuple[float, float, float] | None]:
    positions: list[tuple[float, float, float] | None] = []
    for env_idx in range(env.num_envs):
        target_position = None
        for path in _candidate_target_paths(env_idx):
            target_position = _read_stage_target_world_position(path)
            if target_position is not None:
                break
        positions.append(target_position)
    return positions


def _get_boxing_target_positions_world(env: "ManagerBasedRLEnv") -> torch.Tensor:
    asset = _resolve_boxing_target_asset(env)
    if asset is not None:
        try:
            target_positions = _get_root_positions_world(asset)
            target_positions = target_positions.to(device=env.device, dtype=torch.float32)
            if target_positions.ndim == 1:
                target_positions = target_positions.unsqueeze(0)
            return target_positions
        except Exception:
            pass

    stage_positions = _get_stage_target_positions_world(env)
    env_origins = getattr(env.scene, "env_origins", None)
    positions = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    for env_idx, stage_position in enumerate(stage_positions):
        if stage_position is not None:
            positions[env_idx] = torch.tensor(stage_position, device=env.device, dtype=torch.float32)
            continue
        if env_origins is not None:
            positions[env_idx] = env_origins[env_idx, 0:3].to(dtype=torch.float32)
            positions[env_idx] += torch.tensor(
                _BOXING_TARGET_FALLBACK_INIT_POS,
                device=env.device,
                dtype=torch.float32,
            )
        else:
            positions[env_idx] = torch.tensor(
                _BOXING_TARGET_FALLBACK_INIT_POS,
                device=env.device,
                dtype=torch.float32,
            )
    return positions


def _read_stage_target_bounding_radius(path: str) -> float | None:
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
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        aligned_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        mn = aligned_range.GetMin()
        mx = aligned_range.GetMax()
        size_x = float(mx[0] - mn[0])
        size_y = float(mx[1] - mn[1])
        size_z = float(mx[2] - mn[2])
        return 0.5 * max(size_x, size_y, size_z)
    except Exception:
        return None


def _get_boxing_target_hit_distance_thresholds(env: "ManagerBasedRLEnv") -> torch.Tensor:
    thresholds = torch.full(
        (env.num_envs,),
        _HAND_HIT_DISTANCE_FALLBACK_M,
        device=env.device,
        dtype=torch.float32,
    )

    cached_radii = getattr(env, "_boxing_target_cached_radii_m", None)
    if not (isinstance(cached_radii, list) and len(cached_radii) == env.num_envs):
        cached_radii = [None] * env.num_envs
        for env_idx in range(env.num_envs):
            radius = None
            for path in _candidate_target_paths(env_idx):
                radius = _read_stage_target_bounding_radius(path)
                if radius is not None:
                    break
            cached_radii[env_idx] = radius
        if any(radius is not None for radius in cached_radii):
            env._boxing_target_cached_radii_m = cached_radii

    for env_idx, radius in enumerate(cached_radii):
        if radius is None:
            continue
        thresholds[env_idx] = max(float(radius), _BOXING_TARGET_DEFAULT_RADIUS_M) + _HAND_PROXY_RADIUS_M

    return thresholds


def _get_punch_body_indices(env: "ManagerBasedRLEnv") -> tuple[int, ...]:
    cached = getattr(env, "_boxing_bag_punch_body_indices", None)
    if isinstance(cached, tuple) and len(cached) > 0:
        return cached

    body_names = list(env.scene["robot"].data.body_names)
    indices: list[int] = []

    for body_group in _PUNCH_BODY_NAME_GROUPS:
        matched = [body_names.index(name) for name in body_group if name in body_names]
        if matched:
            indices = matched
            break

    if not indices:
        for token_group in _PUNCH_BODY_TOKEN_GROUPS:
            matched = [
                idx
                for idx, name in enumerate(body_names)
                if any(token in name.lower() for token in token_group)
            ]
            if matched:
                indices = matched
                break

    if not indices:
        raise ValueError(
            "failed to locate hand/punch proxy bodies in robot.body_names; "
            f"available bodies include: {body_names[:20]}"
        )

    env._boxing_bag_punch_body_indices = tuple(indices)
    env._boxing_bag_punch_body_names = tuple(body_names[idx] for idx in indices)
    return env._boxing_bag_punch_body_indices


def compute_success_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    body_positions = _get_body_positions_w(env)
    punch_body_indices = _get_punch_body_indices(env)
    punch_positions = body_positions[:, punch_body_indices, :]

    target_positions = _get_boxing_target_positions_world(env)
    thresholds = _get_boxing_target_hit_distance_thresholds(env)

    deltas = punch_positions - target_positions.unsqueeze(1)
    dist_sq = torch.sum(deltas * deltas, dim=-1)
    threshold_sq = thresholds.unsqueeze(1) * thresholds.unsqueeze(1)
    return torch.any(dist_sq <= threshold_sq, dim=1)


def compute_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    success = compute_success_mask(env)
    reward[success] = 1.0
    return reward


__all__ = ["compute_reward", "compute_success_mask"]
