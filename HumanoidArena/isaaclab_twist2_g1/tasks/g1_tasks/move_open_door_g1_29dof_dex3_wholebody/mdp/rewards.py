from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any

import torch

try:
    from tasks.common_scene.base_scene_open_door import DOOR_POS
except Exception:
    # Keep this module importable in lightweight unit tests where the runtime
    # package layout is not available.
    DOOR_POS = [-1.614, 2.314, 0.002]

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_MIN_STANDING_ROOT_HEIGHT_M = 0.45
_MIN_STANDING_UP_AXIS_Z = 0.60
_DOOR_FRAME_HALF_WIDTH_M = 0.55
_DOOR_PASSING_FORWARD_CLEARANCE_M = 0.02
_STRICT_DOOR_PASSING_FORWARD_CLEARANCE_M = 0.55
_STRICT_DOOR_LEAF_OPEN_ANGLE_DEG = 60.0
_LAST_DOOR_POSE_SOURCE = "unknown"
_LAST_DOOR_POSE_DETAIL = ""
_REWARD_DEBUG_COUNTER = 0
_REWARD_DEBUG_LAST_SUCCESS_BY_ENV: dict[int, bool] = {}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_flag_default(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[open_door_reward] invalid {name}={value!r}; using {default}")
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        print(f"[open_door_reward] invalid {name}={value!r}; using {default}")
        return default


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


def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat = quat_wxyz / torch.linalg.vector_norm(quat_wxyz, dim=1, keepdim=True).clamp_min(1e-8)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _log_door_pose_source_once(source: str, detail: str) -> None:
    global _LAST_DOOR_POSE_SOURCE, _LAST_DOOR_POSE_DETAIL
    _LAST_DOOR_POSE_SOURCE = source
    _LAST_DOOR_POSE_DETAIL = detail


def _static_door_pose_batch(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    door_pos_w = torch.tensor(DOOR_POS, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
    door_quat_wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32).repeat(
        env.num_envs, 1
    )
    return door_pos_w, door_quat_wxyz


def _try_resolve_door_pose_from_asset_data(env: "ManagerBasedRLEnv", door_asset):
    data = getattr(door_asset, "data", None)
    if data is None:
        return None

    root_state_w = getattr(data, "root_state_w", None)
    if root_state_w is not None:
        _log_door_pose_source_once("asset_data", "using root_state_w")
        return root_state_w[:, 0:3], root_state_w[:, 3:7]

    root_pos_w = getattr(data, "root_pos_w", None)
    root_quat_w = getattr(data, "root_quat_w", None)
    if root_pos_w is not None and root_quat_w is not None:
        _log_door_pose_source_once("asset_data", "using root_pos_w/root_quat_w")
        return root_pos_w, root_quat_w

    root_pose_w = getattr(data, "root_pose_w", None)
    if root_pose_w is not None:
        _log_door_pose_SOURCE = "using root_pose_w"
        _log_door_pose_source_once("asset_data", _log_door_pose_SOURCE)
        return root_pose_w[:, 0:3], root_pose_w[:, 3:7]

    return None


def _try_resolve_door_pose_from_stage(env: "ManagerBasedRLEnv"):
    try:
        import omni.usd
        from pxr import UsdGeom
    except Exception as exc:
        _log_door_pose_source_once("stage_unavailable", f"imports_failed={exc}")
        return None

    try:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            _log_door_pose_source_once("stage_unavailable", "stage_is_none")
            return None

        cache = UsdGeom.XformCache()
        pos_rows: list[list[float]] = []
        quat_rows: list[list[float]] = []
        missing_paths: list[str] = []
        for env_idx in range(int(env.num_envs)):
            prim_path = f"/World/envs/env_{env_idx}/Door"
            prim = stage.GetPrimAtPath(prim_path)
            if prim is None or not prim.IsValid() or not prim.IsActive():
                missing_paths.append(prim_path)
                pos_rows.append([float(DOOR_POS[0]), float(DOOR_POS[1]), float(DOOR_POS[2])])
                quat_rows.append([1.0, 0.0, 0.0, 0.0])
                continue

            world_matrix = cache.GetLocalToWorldTransform(prim)
            translation = world_matrix.ExtractTranslation()
            quat_wxyz = [1.0, 0.0, 0.0, 0.0]
            try:
                quat = world_matrix.ExtractRotationQuat()
                imag = quat.GetImaginary()
                quat_wxyz = [float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])]
            except Exception:
                try:
                    rotation = world_matrix.ExtractRotation().GetQuat()
                    imag = rotation.GetImaginary()
                    quat_wxyz = [float(rotation.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])]
                except Exception:
                    quat_wxyz = [1.0, 0.0, 0.0, 0.0]

            pos_rows.append([float(translation[0]), float(translation[1]), float(translation[2])])
            quat_rows.append(quat_wxyz)

        if missing_paths:
            _log_door_pose_source_once(
                "stage_world_transform_partial",
                f"using stage transform with static fallback for missing prims: {missing_paths}",
            )
        else:
            _log_door_pose_source_once("stage_world_transform", "using /World/envs/env_i/Door world transform")

        door_pos_w = torch.tensor(pos_rows, device=env.device, dtype=torch.float32)
        door_quat_wxyz = torch.tensor(quat_rows, device=env.device, dtype=torch.float32)
        return door_pos_w, door_quat_wxyz
    except Exception as exc:
        _log_door_pose_source_once("stage_world_transform_failed", str(exc))
        return None


def _resolve_door_root_pose_w(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    door_asset = None
    for scene_key in ("door", "Door"):
        try:
            door_asset = env.scene[scene_key]
            break
        except Exception:
            continue

    if door_asset is not None:
        pose_from_asset = _try_resolve_door_pose_from_asset_data(env, door_asset)
        if pose_from_asset is not None:
            return pose_from_asset

        pose_from_stage = _try_resolve_door_pose_from_stage(env)
        if pose_from_stage is not None:
            return pose_from_stage

        _log_door_pose_source_once(
            "static_fallback",
            "scene asset found but no runtime world-pose tensors or stage transform were available",
        )
        return _static_door_pose_batch(env)

    pose_from_stage = _try_resolve_door_pose_from_stage(env)
    if pose_from_stage is not None:
        return pose_from_stage

    _log_door_pose_source_once("static_fallback", "door scene asset missing; using DOOR_POS constant")
    return _static_door_pose_batch(env)


def _nan_tensor(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.full((env.num_envs,), float("nan"), device=env.device, dtype=torch.float32)


def _bool_tensor(env: "ManagerBasedRLEnv", value: bool) -> torch.Tensor:
    return torch.full((env.num_envs,), bool(value), device=env.device, dtype=torch.bool)


def _as_bool_tensor(env: "ManagerBasedRLEnv", value: Any, default: bool) -> torch.Tensor:
    if value is None:
        return _bool_tensor(env, default)
    if torch.is_tensor(value):
        return value.to(device=env.device, dtype=torch.bool)
    return torch.tensor(value, device=env.device, dtype=torch.bool)


def _as_float_tensor(env: "ManagerBasedRLEnv", value: Any) -> torch.Tensor:
    if value is None:
        return _nan_tensor(env)
    if torch.is_tensor(value):
        return value.to(device=env.device, dtype=torch.float32)
    return torch.tensor(value, device=env.device, dtype=torch.float32)


def _resolve_open_door_runtime_state(env: "ManagerBasedRLEnv") -> dict[str, Any]:
    cfg = getattr(env, "cfg", None)
    diagnostics_fn = getattr(cfg, "get_open_door_reward_diagnostics", None)
    if not callable(diagnostics_fn):
        return {
            "latch_enabled": False,
            "latch_unlocked": _bool_tensor(env, True),
            "handle_angle_deg": _nan_tensor(env),
            "leaf_angle_deg": _nan_tensor(env),
            "angle_source": ["unavailable"] * int(env.num_envs),
        }

    try:
        diagnostics = diagnostics_fn(env) or {}
    except Exception as exc:
        print(f"[open_door_reward] diagnostics_failed={exc}")
        diagnostics = {}

    latch_enabled = bool(diagnostics.get("latch_enabled", False))
    return {
        "latch_enabled": latch_enabled,
        "latch_unlocked": _as_bool_tensor(env, diagnostics.get("latch_unlocked"), default=not latch_enabled),
        "handle_angle_deg": _as_float_tensor(env, diagnostics.get("handle_angle_deg")),
        "leaf_angle_deg": _as_float_tensor(env, diagnostics.get("leaf_angle_deg")),
        "angle_source": diagnostics.get("angle_source", ["unknown"] * int(env.num_envs)),
    }


def _fmt_float(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def _tensor_float(value: torch.Tensor, env_id: int) -> float:
    try:
        return float(value[env_id].detach().cpu().item())
    except Exception:
        return float("nan")


def _tensor_bool(value: torch.Tensor, env_id: int) -> bool:
    try:
        return bool(value[env_id].detach().cpu().item())
    except Exception:
        return False


def _log_reward_debug(
    env: "ManagerBasedRLEnv",
    *,
    root_pos_w: torch.Tensor,
    door_pos_w: torch.Tensor,
    door_yaw_w: torch.Tensor,
    root_x_local: torch.Tensor,
    root_y_local: torch.Tensor,
    inside_frame_width: torch.Tensor,
    passed_door: torch.Tensor,
    standing: torch.Tensor,
    legacy_success: torch.Tensor,
    strict_success: torch.Tensor,
    success: torch.Tensor,
    runtime_state: dict[str, Any],
    strict_enabled: bool,
    strict_passed_door: torch.Tensor,
    strict_geometry_required: bool,
    strict_geometry_ok: torch.Tensor,
    strict_latch_unlocked: torch.Tensor,
    strict_leaf_open: torch.Tensor,
) -> None:
    global _REWARD_DEBUG_COUNTER
    if not _env_flag("OPEN_DOOR_REWARD_DEBUG"):
        return

    _REWARD_DEBUG_COUNTER += 1
    interval = max(1, _env_int("OPEN_DOOR_REWARD_DEBUG_INTERVAL", 20))
    log_on_success = _env_flag_default("OPEN_DOOR_REWARD_DEBUG_ON_SUCCESS", True)
    max_envs = max(1, _env_int("OPEN_DOOR_REWARD_DEBUG_MAX_ENVS", 4))

    should_log_periodic = _REWARD_DEBUG_COUNTER % interval == 0
    for env_id in range(min(int(env.num_envs), max_envs)):
        current_success = _tensor_bool(success, env_id)
        previous_success = _REWARD_DEBUG_LAST_SUCCESS_BY_ENV.get(env_id, False)
        success_edge = current_success and not previous_success
        _REWARD_DEBUG_LAST_SUCCESS_BY_ENV[env_id] = current_success

        if not should_log_periodic and not (log_on_success and success_edge):
            continue

        angle_source = runtime_state.get("angle_source", [])
        if isinstance(angle_source, (list, tuple)) and env_id < len(angle_source):
            angle_source_value = str(angle_source[env_id])
        else:
            angle_source_value = str(angle_source)

        root_pos = root_pos_w[env_id].detach().cpu().tolist()
        door_pos = door_pos_w[env_id].detach().cpu().tolist()
        print(
            "[open_door_reward] "
            f"env={env_id} strict_enabled={strict_enabled} "
            f"success={current_success} legacy_success={_tensor_bool(legacy_success, env_id)} "
            f"strict_success={_tensor_bool(strict_success, env_id)} "
            f"root_pos_w=({_fmt_float(float(root_pos[0]))},{_fmt_float(float(root_pos[1]))},{_fmt_float(float(root_pos[2]))}) "
            f"door_pos_w=({_fmt_float(float(door_pos[0]))},{_fmt_float(float(door_pos[1]))},{_fmt_float(float(door_pos[2]))}) "
            f"door_yaw_deg={_fmt_float(math.degrees(_tensor_float(door_yaw_w, env_id)))} "
            f"root_x_local={_fmt_float(_tensor_float(root_x_local, env_id))} "
            f"root_y_local={_fmt_float(_tensor_float(root_y_local, env_id))} "
            f"inside_frame_width={_tensor_bool(inside_frame_width, env_id)} "
            f"passed_door={_tensor_bool(passed_door, env_id)} "
            f"strict_passed_door={_tensor_bool(strict_passed_door, env_id)} "
            f"strict_geometry_required={strict_geometry_required} "
            f"strict_geometry_ok={_tensor_bool(strict_geometry_ok, env_id)} "
            f"standing={_tensor_bool(standing, env_id)} "
            f"latch_enabled={bool(runtime_state.get('latch_enabled', False))} "
            f"latch_unlocked={_tensor_bool(strict_latch_unlocked, env_id)} "
            f"handle_angle_deg={_fmt_float(_tensor_float(runtime_state['handle_angle_deg'], env_id))} "
            f"leaf_angle_deg={_fmt_float(_tensor_float(runtime_state['leaf_angle_deg'], env_id))} "
            f"leaf_open={_tensor_bool(strict_leaf_open, env_id)} "
            f"angle_source={angle_source_value} "
            f"door_pose_source={_LAST_DOOR_POSE_SOURCE} "
            f"door_pose_detail={_LAST_DOOR_POSE_DETAIL}"
        )


def compute_success_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    root_state = env.scene["robot"].data.root_state_w
    root_pos_w = root_state[:, 0:3]
    door_pos_w, door_quat_wxyz = _resolve_door_root_pose_w(env)

    delta_xy_w = root_pos_w[:, 0:2] - door_pos_w[:, 0:2]
    door_yaw_w = _yaw_from_quat_wxyz(door_quat_wxyz)
    cos_yaw = torch.cos(door_yaw_w)
    sin_yaw = torch.sin(door_yaw_w)
    root_x_local = cos_yaw * delta_xy_w[:, 0] + sin_yaw * delta_xy_w[:, 1]
    root_y_local = -sin_yaw * delta_xy_w[:, 0] + cos_yaw * delta_xy_w[:, 1]

    success_half_width = _env_float("OPEN_DOOR_SUCCESS_HALF_WIDTH", _DOOR_FRAME_HALF_WIDTH_M)
    legacy_forward_clearance = _env_float("OPEN_DOOR_LEGACY_SUCCESS_FORWARD_CLEARANCE", _DOOR_PASSING_FORWARD_CLEARANCE_M)
    strict_forward_clearance = _env_float(
        "OPEN_DOOR_STRICT_SUCCESS_FORWARD_CLEARANCE",
        _STRICT_DOOR_PASSING_FORWARD_CLEARANCE_M,
    )
    strict_leaf_angle_deg = abs(_env_float("OPEN_DOOR_SUCCESS_LEAF_ANGLE_DEG", _STRICT_DOOR_LEAF_OPEN_ANGLE_DEG))

    inside_frame_width = torch.abs(root_x_local) <= success_half_width
    passed_door = root_y_local > legacy_forward_clearance
    standing = _compute_standing_mask(env)
    legacy_success = inside_frame_width & passed_door & standing

    runtime_state = _resolve_open_door_runtime_state(env)
    strict_latch_unlocked = runtime_state["latch_unlocked"]
    strict_passed_door = root_y_local > strict_forward_clearance
    leaf_angle_deg = runtime_state["leaf_angle_deg"]
    leaf_angle_known = ~torch.isnan(leaf_angle_deg)
    strict_leaf_open = leaf_angle_known & (torch.abs(leaf_angle_deg) >= strict_leaf_angle_deg)

    if not _env_flag_default("OPEN_DOOR_STRICT_REQUIRE_LATCH_UNLOCKED", True):
        strict_latch_unlocked = _bool_tensor(env, True)
    if not _env_flag_default("OPEN_DOOR_STRICT_REQUIRE_LEAF_OPEN", True):
        strict_leaf_open = _bool_tensor(env, True)

    strict_geometry_required = _env_flag_default("OPEN_DOOR_STRICT_REQUIRE_GEOMETRY", False)
    strict_geometry_ok = inside_frame_width & strict_passed_door
    if not strict_geometry_required:
        strict_geometry_ok = _bool_tensor(env, True)

    strict_success = strict_geometry_ok & standing & strict_latch_unlocked & strict_leaf_open
    strict_enabled = _env_flag_default("OPEN_DOOR_STRICT_SUCCESS", True)
    success = strict_success if strict_enabled else legacy_success

    _log_reward_debug(
        env,
        root_pos_w=root_pos_w,
        door_pos_w=door_pos_w,
        door_yaw_w=door_yaw_w,
        root_x_local=root_x_local,
        root_y_local=root_y_local,
        inside_frame_width=inside_frame_width,
        passed_door=passed_door,
        standing=standing,
        legacy_success=legacy_success,
        strict_success=strict_success,
        success=success,
        runtime_state=runtime_state,
        strict_enabled=strict_enabled,
        strict_passed_door=strict_passed_door,
        strict_geometry_required=strict_geometry_required,
        strict_geometry_ok=strict_geometry_ok,
        strict_latch_unlocked=strict_latch_unlocked,
        strict_leaf_open=strict_leaf_open,
    )
    return success


def compute_reward_open_door(env: "ManagerBasedRLEnv") -> torch.Tensor:
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    success = compute_success_mask(env)
    reward[success] = 1.0
    return reward


__all__ = [
    "compute_reward_open_door",
    "compute_success_mask",
]
