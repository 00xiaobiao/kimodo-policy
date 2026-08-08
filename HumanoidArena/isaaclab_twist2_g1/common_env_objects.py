from __future__ import annotations

import zlib
import time
from typing import Any

import numpy as np
import torch


_ROOT_STATE_FIELDS = (
    ("position", slice(0, 3), "_position", np.zeros(3, dtype=np.float32)),
    ("orientation", slice(3, 7), "_orientation", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
    ("linear_velocity", slice(7, 10), "_linear_velocity", np.zeros(3, dtype=np.float32)),
    ("angular_velocity", slice(10, 13), "_angular_velocity", np.zeros(3, dtype=np.float32)),
)
_POSE_RANGE_AXES = {"x": 0, "y": 1, "z": 2}


def get_recordable_env_object_specs(env_cfg: Any) -> list[dict[str, Any]]:
    merged_specs: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for attr_name in ("recordable_env_objects", "deterministic_object_resets"):
        raw_specs = getattr(env_cfg, attr_name, []) or []
        if not isinstance(raw_specs, list):
            continue
        for raw_spec in raw_specs:
            if not isinstance(raw_spec, dict):
                continue
            scene_keys = raw_spec.get("scene_keys")
            if scene_keys is None:
                scene_keys = [raw_spec.get("scene_key")]
            scene_keys = [str(value) for value in scene_keys if value]
            prim_paths = [str(value) for value in (raw_spec.get("prim_paths", []) or []) if value]
            if not scene_keys and not prim_paths:
                continue
            fallback_name = scene_keys[0] if scene_keys else prim_paths[0].split("/")[-1] if prim_paths else ""
            record_name = str(raw_spec.get("record_name") or fallback_name)
            if not record_name:
                continue
            if record_name in seen_names:
                continue
            merged_specs.append(
                {
                    "record_name": record_name,
                    "scene_keys": scene_keys,
                    "prim_paths": prim_paths,
                    "allow_prroot_fallback": bool(raw_spec.get("allow_prroot_fallback", True)),
                    "prim_pose_write_mode": str(raw_spec.get("prim_pose_write_mode", "") or "").strip().lower(),
                    "pose_range": raw_spec.get("pose_range", {}) or {},
                    "zero_velocity_on_reset": bool(raw_spec.get("zero_velocity_on_reset", True)),
                }
            )
            seen_names.add(record_name)

    return merged_specs


def resolve_env_object_scene_key(env, env_cfg: Any, object_name: str) -> str | None:
    for spec in get_recordable_env_object_specs(env_cfg):
        if spec["record_name"] != object_name:
            continue
        for scene_key in spec["scene_keys"]:
            if scene_key in env.scene.keys():
                return scene_key
        break

    for fallback_key in (object_name,):
        if fallback_key in env.scene.keys():
            return fallback_key
    return None


def _resolve_spec_by_record_name(env_cfg: Any, object_name: str) -> dict[str, Any] | None:
    for spec in get_recordable_env_object_specs(env_cfg):
        if spec["record_name"] == object_name:
            return spec
    return None


def _resolve_prim_path(template: str, env_idx: int) -> str:
    return str(template).replace("{env_idx}", str(int(env_idx)))


def _candidate_prim_paths(
    template: str,
    env_idx: int,
    *,
    allow_prroot_fallback: bool = True,
) -> list[str]:
    base = _resolve_prim_path(template, env_idx)
    out = [base]
    # Many Prop USDs expose a transformable default prim below the mount node.
    if allow_prroot_fallback and not base.endswith("/PRootNode"):
        out.append(base.rstrip("/") + "/PRootNode")
    return out


def _should_prefer_prim_pose_write(spec: dict[str, Any] | None) -> bool:
    if not isinstance(spec, dict):
        return False
    return str(spec.get("prim_pose_write_mode", "") or "").strip().lower() == "local_matrix"


def _try_import_usd_modules():
    try:
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        return omni, Usd, UsdGeom, Gf
    except Exception:
        return None, None, None, None


def _get_stage():
    omni, _Usd, _UsdGeom, _Gf = _try_import_usd_modules()
    if omni is None:
        return None
    return omni.usd.get_context().get_stage()


def _read_prim_world_pose(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    stage = _get_stage()
    if stage is None:
        return None

    _omni, Usd, UsdGeom, _Gf = _try_import_usd_modules()
    prim = stage.GetPrimAtPath(path)
    if prim is None or not prim.IsValid() or not prim.IsActive():
        return None

    try:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        matrix = cache.GetLocalToWorldTransform(prim)
        trans = matrix.ExtractTranslation()
        quat = matrix.ExtractRotationQuat()
        pos = np.array([float(trans[0]), float(trans[1]), float(trans[2])], dtype=np.float32)
        ori = np.array(
            [
                float(quat.GetReal()),
                float(quat.GetImaginary()[0]),
                float(quat.GetImaginary()[1]),
                float(quat.GetImaginary()[2]),
            ],
            dtype=np.float32,
        )
        return pos, ori
    except Exception:
        return None


def _write_prim_world_pose(path: str, position: np.ndarray, orientation: np.ndarray) -> bool:
    stage = _get_stage()
    if stage is None:
        return False

    _omni, Usd, UsdGeom, Gf = _try_import_usd_modules()
    prim = stage.GetPrimAtPath(path)
    if prim is None or not prim.IsValid() or not prim.IsActive():
        return False

    try:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        parent = prim.GetParent()
        parent_world = (
            cache.GetLocalToWorldTransform(parent)
            if parent is not None and parent.IsValid()
            else Gf.Matrix4d(1.0)
        )

        quat = Gf.Quatd(
            float(orientation[0]),
            Gf.Vec3d(float(orientation[1]), float(orientation[2]), float(orientation[3])),
        )
        target_world = Gf.Matrix4d().SetRotate(quat)
        target_world.SetTranslateOnly(
            Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        )
        local_matrix = target_world * parent_world.GetInverse()

        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return False
        matrix_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
                matrix_op = op
                break

        if matrix_op is None:
            matrix_op = xformable.AddTransformOp()
        matrix_op.Set(local_matrix)
        return True
    except Exception:
        return False


def collect_recordable_env_object_states(env, env_cfg: Any) -> dict[str, dict[str, np.ndarray] | None]:
    env_state: dict[str, dict[str, np.ndarray] | None] = {}

    for spec in get_recordable_env_object_specs(env_cfg):
        record_name = spec["record_name"]
        prefer_prim_pose_write = _should_prefer_prim_pose_write(spec)
        scene_key = None if prefer_prim_pose_write else resolve_env_object_scene_key(env, env_cfg, record_name)

        if scene_key is not None:
            try:
                obj = env.scene[scene_key]
                root_state = obj.data.root_state_w
                env_state[record_name] = {
                    field_name: root_state[0, field_slice].detach().cpu().numpy().astype(np.float32).copy()
                    for field_name, field_slice, _, _ in _ROOT_STATE_FIELDS
                }
                continue
            except Exception:
                pass

        prim_paths = spec.get("prim_paths", []) or []
        prim_state = None
        for prim_template in prim_paths:
            for prim_path in _candidate_prim_paths(
                str(prim_template),
                0,
                allow_prroot_fallback=bool(spec.get("allow_prroot_fallback", True)),
            ):
                pose = _read_prim_world_pose(prim_path)
                if pose is None:
                    continue
                position, orientation = pose
                prim_state = {
                    "position": position.astype(np.float32).copy(),
                    "orientation": orientation.astype(np.float32).copy(),
                    "linear_velocity": np.zeros(3, dtype=np.float32),
                    "angular_velocity": np.zeros(3, dtype=np.float32),
                }
                break
            if prim_state is not None:
                break
        env_state[record_name] = prim_state

    return env_state


def add_env_object_frame_arrays(organized: dict[str, Any], data_buffer: list[dict[str, Any]]) -> None:
    if not data_buffer:
        return

    def _extract_frame_env_objects(frame: dict[str, Any]) -> dict[str, Any]:
        env_obj = frame.get("env_obj", {})
        if isinstance(env_obj, dict) and env_obj:
            return env_obj

        env_state = frame.get("env", {})
        if not isinstance(env_state, dict):
            return {}

        extracted: dict[str, Any] = {}
        for object_name, state in env_state.items():
            if object_name == "vision" or not isinstance(state, dict):
                continue
            if any(field_name in state for field_name, _, _, _ in _ROOT_STATE_FIELDS):
                extracted[object_name] = state
        return extracted

    object_names: set[str] = set()
    for frame in data_buffer:
        object_names.update(_extract_frame_env_objects(frame).keys())

    for object_name in sorted(object_names):
        states = [
            _extract_frame_env_objects(frame).get(object_name)
            for frame in data_buffer
        ]
        if not any(state is not None for state in states):
            continue
        for field_name, _, suffix, default_value in _ROOT_STATE_FIELDS:
            organized[f"env_obj_{object_name}{suffix}"] = np.array(
                [
                    np.asarray(state[field_name], dtype=np.float32) if state is not None else default_value.copy()
                    for state in states
                ],
                dtype=np.float32,
            )


def add_episode_init_env_object_fields(organized: dict[str, Any], episode_init_env: dict[str, Any] | None) -> None:
    if not isinstance(episode_init_env, dict):
        return
    for object_name, state in episode_init_env.items():
        if state is None:
            continue
        for field_name, _, suffix, default_value in _ROOT_STATE_FIELDS:
            organized[f"episode_init_env_obj_{object_name}{suffix}"] = np.asarray(
                state.get(field_name, default_value),
                dtype=np.float32,
            )


def get_current_episode_object_seed_info(env_cfg: Any) -> dict[str, Any]:
    seed_value = getattr(env_cfg, "_current_episode_object_seed", None)
    seed_source = getattr(env_cfg, "_current_episode_object_seed_source", "")
    return {
        "seed": None if seed_value is None else int(seed_value),
        "source": str(seed_source or ""),
    }


def set_current_episode_object_seed(env_cfg: Any, episode_seed: int, seed_source: str) -> tuple[int, str]:
    normalized_seed = int(episode_seed)
    normalized_source = str(seed_source or "")
    setattr(env_cfg, "_current_episode_object_seed", normalized_seed)
    setattr(env_cfg, "_current_episode_object_seed_source", normalized_source)
    return normalized_seed, normalized_source


def begin_new_episode_object_seed(env_cfg: Any) -> tuple[int, str]:
    episode_seed, seed_source = _next_episode_object_seed(env_cfg)
    return set_current_episode_object_seed(env_cfg, episode_seed, seed_source)


def ensure_current_episode_object_seed(env_cfg: Any) -> tuple[int, str]:
    seed_info = get_current_episode_object_seed_info(env_cfg)
    current_seed = seed_info.get("seed")
    current_source = str(seed_info.get("source") or "")
    if current_seed is not None:
        return int(current_seed), current_source

    return begin_new_episode_object_seed(env_cfg)


def _next_episode_object_seed(env_cfg: Any) -> tuple[int, str]:
    seed_source = str(getattr(env_cfg, "object_reset_seed_source", "time") or "time").strip().lower()
    if seed_source == "time":
        episode_seed = int(time.time_ns() & 0xFFFFFFFFFFFFFFFF)
        return episode_seed, seed_source

    if seed_source == "env_seed":
        reset_counter = int(getattr(env_cfg, "_episode_object_seed_counter", 0))
        setattr(env_cfg, "_episode_object_seed_counter", reset_counter + 1)
        base_seed = int(getattr(env_cfg, "seed", 0) or 0) & 0xFFFFFFFFFFFFFFFF
        episode_seed = (
            base_seed
            ^ ((reset_counter + 1) * 0x9E3779B185EBCA87)
        ) & 0xFFFFFFFFFFFFFFFF
        return episode_seed, seed_source

    raise ValueError(f"Unsupported object_reset_seed_source: {seed_source}")


def _make_local_spawn_rng(episode_seed: int, record_name: str, env_index: int) -> np.random.Generator:
    name_seed = zlib.crc32(record_name.encode("utf-8")) & 0xFFFFFFFF
    mixed_seed = (
        (int(episode_seed) & 0xFFFFFFFFFFFFFFFF)
        ^ name_seed
        ^ ((env_index + 1) * 0x85EBCA77)
    ) & 0xFFFFFFFFFFFFFFFF
    return np.random.default_rng(mixed_seed)


def make_local_spawn_rng(episode_seed: int, record_name: str, env_index: int) -> np.random.Generator:
    """Create a deterministic per-object, per-env RNG for custom reset logic."""
    return _make_local_spawn_rng(episode_seed, record_name, env_index)


def _sample_abs_position_with_pose_range(base_position: np.ndarray, pose_range: dict[str, Any], rng) -> np.ndarray:
    out = np.asarray(base_position, dtype=np.float32).copy()
    for axis_name, axis_idx in _POSE_RANGE_AXES.items():
        axis_range = pose_range.get(axis_name)
        if axis_range is None:
            continue
        low, high = [float(v) for v in axis_range]
        low, high = min(low, high), max(low, high)
        out[axis_idx] = float(rng.uniform(low, high)) if high > low else low
    return out


def _sample_yaw_with_pose_range(pose_range: dict[str, Any], rng) -> float | None:
    yaw_range = pose_range.get("yaw")
    if yaw_range is None:
        return None
    low, high = [float(v) for v in yaw_range]
    low, high = min(low, high), max(low, high)
    return float(rng.uniform(low, high)) if high > low else low


def _quat_wxyz_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def _quat_wxyz_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    out = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )
    return _quat_wxyz_normalize(out)


def _quat_wxyz_to_euler_xyz_deg(quat: np.ndarray) -> np.ndarray:
    quat = _quat_wxyz_normalize(quat)
    w, x, y, z = [float(v) for v in quat]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(np.asarray([roll, pitch, yaw], dtype=np.float32)).astype(np.float32)


def _apply_yaw_delta_to_orientation(base_orientation: np.ndarray, yaw_rad: float | None) -> np.ndarray:
    base = _quat_wxyz_normalize(base_orientation)
    if yaw_rad is None:
        return base
    half = 0.5 * float(yaw_rad)
    yaw_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)
    return _quat_wxyz_mul(yaw_quat, base)


def _load_prim_default_states(
    env_cfg: Any,
    record_name: str,
    prim_paths: list[str],
    num_envs: int,
    *,
    allow_prroot_fallback: bool = True,
) -> dict[int, dict[str, np.ndarray]]:
    cache = getattr(env_cfg, "_deterministic_prim_default_states", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(env_cfg, "_deterministic_prim_default_states", cache)

    key = (record_name, tuple(str(v) for v in prim_paths), int(num_envs))
    if key in cache:
        return cache[key]

    defaults: dict[int, dict[str, np.ndarray]] = {}
    for env_idx in range(num_envs):
        for prim_template in prim_paths:
            for prim_path in _candidate_prim_paths(
                str(prim_template),
                env_idx,
                allow_prroot_fallback=allow_prroot_fallback,
            ):
                pose = _read_prim_world_pose(prim_path)
                if pose is None:
                    continue
                position, orientation = pose
                defaults[env_idx] = {
                    "position": position.astype(np.float32).copy(),
                    "orientation": orientation.astype(np.float32).copy(),
                }
                break
            if env_idx in defaults:
                break
    cache[key] = defaults
    return defaults


def _apply_deterministic_object_resets_with_seed(
    env_cfg: Any,
    env,
    *,
    episode_seed: int,
    seed_source: str,
    selected_record_names: set[str] | None = None,
) -> list[str]:
    specs = get_recordable_env_object_specs(env_cfg)
    if not specs:
        return []

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    applied: list[str] = []
    setattr(env_cfg, "_current_episode_object_seed", int(episode_seed))
    setattr(env_cfg, "_current_episode_object_seed_source", str(seed_source or ""))

    scene_object_changed = False
    for spec in specs:
        record_name = spec["record_name"]
        if selected_record_names is not None and record_name not in selected_record_names:
            continue

        prefer_prim_pose_write = _should_prefer_prim_pose_write(spec)
        scene_key = None if prefer_prim_pose_write else resolve_env_object_scene_key(env, env_cfg, record_name)
        pose_range = spec.get("pose_range", {}) or {}

        if scene_key is not None:
            root_state = None
            try:
                obj = env.scene[scene_key]
                root_state = obj.data.default_root_state.clone()
            except Exception:
                try:
                    root_state = obj.data.root_state_w.clone()
                except Exception:
                    root_state = None

            if root_state is not None:
                for env_offset, env_id in enumerate(env_ids.tolist()):
                    rng = _make_local_spawn_rng(episode_seed, record_name, env_offset)
                    yaw_rad = _sample_yaw_with_pose_range(pose_range, rng)
                    root_state[env_id, 0:3] = torch.as_tensor(
                        _sample_abs_position_with_pose_range(
                            root_state[env_id, 0:3].detach().cpu().numpy(),
                            pose_range,
                            rng,
                        ),
                        device=root_state.device,
                        dtype=root_state.dtype,
                    )
                    root_state[env_id, 3:7] = torch.as_tensor(
                        _apply_yaw_delta_to_orientation(
                            root_state[env_id, 3:7].detach().cpu().numpy(),
                            yaw_rad,
                        ),
                        device=root_state.device,
                        dtype=root_state.dtype,
                    )
                    if spec.get("zero_velocity_on_reset", True):
                        root_state[env_id, 7:13] = 0.0

                obj.write_root_state_to_sim(root_state, env_ids=env_ids)
                scene_object_changed = True
                applied.append(
                    f"{record_name}->{scene_key}:episode_seed={episode_seed}:seed_source={seed_source}:pos="
                    f"{root_state[0, 0:3].detach().cpu().numpy().tolist()}"
                )
                continue

        prim_paths = [str(v) for v in (spec.get("prim_paths", []) or []) if v]
        if not prim_paths:
            continue

        defaults = _load_prim_default_states(
            env_cfg,
            record_name,
            prim_paths,
            env.num_envs,
            allow_prroot_fallback=bool(spec.get("allow_prroot_fallback", True)),
        )
        first_pos = None
        for env_offset in range(env.num_envs):
            default_state = defaults.get(env_offset)
            if default_state is None:
                continue
            rng = _make_local_spawn_rng(episode_seed, record_name, env_offset)
            yaw_rad = _sample_yaw_with_pose_range(pose_range, rng)
            target_pos = _sample_abs_position_with_pose_range(
                default_state["position"],
                pose_range,
                rng,
            )
            target_ori = _apply_yaw_delta_to_orientation(
                np.asarray(default_state["orientation"], dtype=np.float32),
                yaw_rad,
            )
            write_ok = False
            for prim_template in prim_paths:
                for prim_path in _candidate_prim_paths(
                    prim_template,
                    env_offset,
                    allow_prroot_fallback=bool(spec.get("allow_prroot_fallback", True)),
                ):
                    write_ok = _write_prim_world_pose(prim_path, target_pos, target_ori)
                    if write_ok:
                        break
                if write_ok:
                    break
            if write_ok and first_pos is None:
                first_pos = target_pos.tolist()
        if first_pos is not None:
            applied.append(
                f"{record_name}->prim:episode_seed={episode_seed}:seed_source={seed_source}:pos={first_pos}"
            )
        else:
            print(
                f"[object_reset] {record_name} prim write skipped: "
                f"no writable prim under templates={prim_paths}"
            )

    if scene_object_changed:
        env.scene.write_data_to_sim()
    return applied


def apply_deterministic_object_resets(env_cfg: Any, env, *, selected_record_names: set[str] | None = None) -> list[str]:
    if getattr(env_cfg, "_replay_initial_env_state_active", False):
        return []

    episode_seed, seed_source = _next_episode_object_seed(env_cfg)
    return _apply_deterministic_object_resets_with_seed(
        env_cfg,
        env,
        episode_seed=episode_seed,
        seed_source=seed_source,
        selected_record_names=selected_record_names,
    )


def apply_deterministic_object_resets_with_seed(
    env_cfg: Any,
    env,
    *,
    episode_seed: int,
    seed_source: str = "recorded",
    selected_record_names: set[str] | None = None,
) -> list[str]:
    return _apply_deterministic_object_resets_with_seed(
        env_cfg,
        env,
        episode_seed=int(episode_seed),
        seed_source=str(seed_source or "recorded"),
        selected_record_names=selected_record_names,
    )


def apply_explicit_env_object_states(
    env,
    env_cfg: Any,
    object_states: dict[str, Any],
    *,
    log_prefix: str = "replay_env_init",
) -> bool:
    if not object_states:
        return False

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    applied_objects = []
    scene_object_changed = False

    def _broadcast_field(value, width):
        tensor = torch.as_tensor(np.asarray(value), device=env.device, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0).repeat(env.num_envs, 1)
        elif tensor.ndim == 2 and tensor.shape[0] == env.num_envs:
            pass
        else:
            tensor = tensor.reshape(1, -1).repeat(env.num_envs, 1)
        return tensor[:, :width]

    def _pick_env_field(value, env_idx, width):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim <= 1:
            return arr.reshape(-1)[:width]
        idx = int(env_idx) if int(env_idx) < arr.shape[0] else 0
        return arr[idx].reshape(-1)[:width]

    for object_name, state in object_states.items():
        if not isinstance(state, dict):
            continue

        spec = _resolve_spec_by_record_name(env_cfg, object_name)
        prefer_prim_pose_write = _should_prefer_prim_pose_write(spec)
        scene_name = None if prefer_prim_pose_write else resolve_env_object_scene_key(env, env_cfg, object_name)
        if scene_name is not None:
            root_state = None
            try:
                asset = env.scene[scene_name]
                root_state = asset.data.default_root_state.clone()
            except Exception:
                try:
                    root_state = asset.data.root_state_w.clone()
                except Exception:
                    root_state = None

            if root_state is not None:
                if "position" in state:
                    root_state[:, 0:3] = _broadcast_field(state["position"], 3)
                if "orientation" in state:
                    root_state[:, 3:7] = _broadcast_field(state["orientation"], 4)
                if "linear_velocity" in state:
                    root_state[:, 7:10] = _broadcast_field(state["linear_velocity"], 3)
                else:
                    root_state[:, 7:10] = 0.0
                if "angular_velocity" in state:
                    root_state[:, 10:13] = _broadcast_field(state["angular_velocity"], 3)
                else:
                    root_state[:, 10:13] = 0.0

                asset.write_root_state_to_sim(root_state, env_ids=env_ids)
                scene_object_changed = True
                applied_objects.append(
                    f"{object_name}->{scene_name}:pos={root_state[0, 0:3].detach().cpu().numpy().tolist()}"
                )
                continue

        prim_paths = [str(v) for v in ((spec or {}).get("prim_paths", []) or []) if v]
        if not prim_paths:
            continue
        if "position" not in state:
            continue

        orientation_value = state.get("orientation", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        first_pos = None
        first_readback_pos = None
        first_readback_ori = None
        for env_idx in range(env.num_envs):
            pos = _pick_env_field(state["position"], env_idx, 3)
            ori = _pick_env_field(orientation_value, env_idx, 4)
            write_ok = False
            for prim_template in prim_paths:
                for prim_path in _candidate_prim_paths(
                    prim_template,
                    env_idx,
                    allow_prroot_fallback=bool((spec or {}).get("allow_prroot_fallback", True)),
                ):
                    write_ok = _write_prim_world_pose(prim_path, pos, ori)
                    if write_ok:
                        break
                if write_ok:
                    break
            if write_ok and first_pos is None:
                first_pos = pos.tolist()
                if prim_path:
                    readback = _read_prim_world_pose(prim_path)
                    if readback is not None:
                        readback_pos, readback_ori = readback
                        first_readback_pos = readback_pos.tolist()
                        first_readback_ori = readback_ori.tolist()
        if first_pos is not None:
            if first_readback_pos is not None:
                applied_objects.append(
                    f"{object_name}->prim:target_pos={first_pos}:readback_pos={first_readback_pos}:"
                    f"readback_ori={first_readback_ori}"
                )
            else:
                applied_objects.append(f"{object_name}->prim:pos={first_pos}")

    if scene_object_changed:
        env.scene.write_data_to_sim()
    if not applied_objects:
        return False

    print(f"[{log_prefix}] applied initial env object state: " + ", ".join(applied_objects))
    return True
