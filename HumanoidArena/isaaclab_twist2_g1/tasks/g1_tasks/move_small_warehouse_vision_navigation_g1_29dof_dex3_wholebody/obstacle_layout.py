from __future__ import annotations

import numpy as np


OBSTACLE_RECORD_NAMES = (
    "obstacle_01_a",
    "obstacle_01_b",
    "obstacle_01_c",
    "obstacle_02_a",
    "obstacle_02_b",
    "obstacle_02_c",
)


def should_randomize_target_sign(
    *,
    is_startup_reset: bool,
    startup_randomization_enabled: bool,
    reset_randomization_enabled: bool,
) -> bool:
    if is_startup_reset:
        return bool(startup_randomization_enabled)
    return bool(reset_randomization_enabled)


def _normalize_range(value) -> tuple[float, float]:
    low, high = [float(v) for v in value]
    return (low, high) if low <= high else (high, low)


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


def _apply_yaw_delta(base_orientation: np.ndarray, yaw_rad: float) -> np.ndarray:
    base = _quat_wxyz_normalize(base_orientation)
    half = 0.5 * float(yaw_rad)
    yaw_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)
    return _quat_wxyz_mul(yaw_quat, base)


def _sample_axis_in_range(
    axis_bounds: tuple[float, float],
    rng: np.random.Generator,
) -> float:
    axis_min, axis_max = _normalize_range(axis_bounds)
    return float(rng.uniform(axis_min, axis_max)) if axis_max > axis_min else axis_min


def build_obstacle_layout_states(
    *,
    default_states: dict[str, dict[str, np.ndarray] | None],
    episode_seed: int,
    x_range: tuple[float, float] | None = None,
    x_ranges: dict[str, tuple[float, float]] | None = None,
    y_range: tuple[float, float],
    z_ranges: dict[str, tuple[float, float]] | None = None,
    yaw_range: tuple[float, float],
) -> dict[str, dict[str, np.ndarray]]:
    yaw_min, yaw_max = _normalize_range(yaw_range)
    rng = np.random.default_rng(int(episode_seed) & 0xFFFFFFFFFFFFFFFF)

    states: dict[str, dict[str, np.ndarray]] = {}
    for obstacle_name in OBSTACLE_RECORD_NAMES:
        default_state = default_states.get(obstacle_name)
        if not isinstance(default_state, dict):
            continue

        obstacle_x_range = (x_ranges or {}).get(obstacle_name, x_range)
        if obstacle_x_range is None:
            raise ValueError(f"Missing x range for obstacle layout record: {obstacle_name}")

        sampled_x = _sample_axis_in_range(obstacle_x_range, rng)
        sampled_y = _sample_axis_in_range(y_range, rng)
        z_range = (z_ranges or {}).get(obstacle_name)
        if z_range is not None:
            z_min, z_max = _normalize_range(z_range)
            sampled_z = float(rng.uniform(z_min, z_max)) if z_max > z_min else z_min
        else:
            sampled_z = float(np.asarray(default_state["position"], dtype=np.float32)[2])
        yaw_rad = float(rng.uniform(yaw_min, yaw_max)) if yaw_max > yaw_min else yaw_min

        base_orientation = np.asarray(default_state["orientation"], dtype=np.float32)
        states[obstacle_name] = {
            "position": np.asarray([sampled_x, sampled_y, sampled_z], dtype=np.float32),
            "orientation": _apply_yaw_delta(base_orientation, yaw_rad),
            "linear_velocity": np.zeros(3, dtype=np.float32),
            "angular_velocity": np.zeros(3, dtype=np.float32),
        }

    return states


def build_single_asset_randomized_state(
    *,
    default_state: dict[str, np.ndarray] | None,
    episode_seed: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float] | None,
    yaw_range: tuple[float, float],
) -> dict[str, np.ndarray] | None:
    if not isinstance(default_state, dict):
        return None

    rng = np.random.default_rng(int(episode_seed) & 0xFFFFFFFFFFFFFFFF)
    x_min, x_max = _normalize_range(x_range)
    y_min, y_max = _normalize_range(y_range)
    yaw_min, yaw_max = _normalize_range(yaw_range)

    sampled_x = float(rng.uniform(x_min, x_max)) if x_max > x_min else x_min
    sampled_y = float(rng.uniform(y_min, y_max)) if y_max > y_min else y_min
    if z_range is not None:
        z_min, z_max = _normalize_range(z_range)
        sampled_z = float(rng.uniform(z_min, z_max)) if z_max > z_min else z_min
    else:
        sampled_z = float(np.asarray(default_state["position"], dtype=np.float32)[2])
    yaw_rad = float(rng.uniform(yaw_min, yaw_max)) if yaw_max > yaw_min else yaw_min

    base_orientation = np.asarray(default_state["orientation"], dtype=np.float32)
    return {
        "position": np.asarray([sampled_x, sampled_y, sampled_z], dtype=np.float32),
        "orientation": _apply_yaw_delta(base_orientation, yaw_rad),
        "linear_velocity": np.zeros(3, dtype=np.float32),
        "angular_velocity": np.zeros(3, dtype=np.float32),
    }


__all__ = [
    "OBSTACLE_RECORD_NAMES",
    "build_obstacle_layout_states",
    "build_single_asset_randomized_state",
    "should_randomize_target_sign",
]
