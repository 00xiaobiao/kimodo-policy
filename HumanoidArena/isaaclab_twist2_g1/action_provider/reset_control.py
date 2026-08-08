"""Shared Redis reset helpers for IsaacLab action providers."""

from __future__ import annotations

import json
import time
from typing import Any

try:
    import redis
except ImportError:  # pragma: no cover - runtime dependency
    redis = None


RESET_TRIGGER_KEY = "isaac_reset_trigger"
RESET_COMPLETE_KEY = "isaac_reset_complete_unitree_g1_with_hands"
TWIST2_INPUT_READY_KEY = "isaac_input_ready_twist2_unitree_g1_with_hands"
SONIC_INPUT_READY_KEY = "isaac_input_ready_sonic_unitree_g1_with_hands"
SONIC_JOINT29_INPUT_READY_KEY = "isaac_input_ready_sonic_joint29_unitree_g1_with_hands"

GMR_FULL_QPOS_KEY = "gmr_full_qpos_unitree_g1_with_hands"
GMR_JOINT_POS_KEY = "gmr_joint_pos_unitree_g1_with_hands"
GMR_JOINT_VEL_KEY = "gmr_joint_vel_unitree_g1_with_hands"
GMR_BODY_POS_KEY = "gmr_body_pos_unitree_g1_with_hands"
GMR_BODY_QUAT_W_KEY = "gmr_body_quat_w_unitree_g1_with_hands"
GMR_FRAME_INDEX_KEY = "gmr_frame_index_unitree_g1_with_hands"

_INPUT_READY_KEYS = {
    "twist2": TWIST2_INPUT_READY_KEY,
    "sonic": SONIC_INPUT_READY_KEY,
    "sonic_joint29": SONIC_JOINT29_INPUT_READY_KEY,
}

_INPUT_STREAM_KEYS = {
    "twist2": (
        "action_body_unitree_g1_with_hands",
        "action_hand_left_unitree_g1_with_hands",
        "action_hand_right_unitree_g1_with_hands",
        "action_neck_unitree_g1_with_hands",
        "human_smplx_data_unitree_g1_with_hands",
        "human_info_unitree_g1_with_hands",
        "recording_control_unitree_g1_with_hands",
        "controller_data",
        "t_action",
    ),
    "sonic": (
        "human_smplx_data_unitree_g1_with_hands",
        "action_hand_left_unitree_g1_with_hands",
        "action_hand_right_unitree_g1_with_hands",
        "controller_data",
        "recording_control_unitree_g1_with_hands",
    ),
    "sonic_joint29": (
        "action_body_unitree_g1_with_hands",
        "action_neck_unitree_g1_with_hands",
        "human_smplx_data_unitree_g1_with_hands",
        "human_info_unitree_g1_with_hands",
        GMR_FULL_QPOS_KEY,
        GMR_JOINT_POS_KEY,
        GMR_JOINT_VEL_KEY,
        GMR_BODY_POS_KEY,
        GMR_BODY_QUAT_W_KEY,
        GMR_FRAME_INDEX_KEY,
        "action_hand_left_unitree_g1_with_hands",
        "action_hand_right_unitree_g1_with_hands",
        "controller_data",
        "recording_control_unitree_g1_with_hands",
        "t_action",
    ),
}


def create_redis_client(host: str = "localhost", port: int = 6379, *, decode_responses: bool = False):
    if redis is None:
        raise RuntimeError("redis package is not installed")
    return redis.Redis(host=host, port=port, db=0, decode_responses=decode_responses)


def get_input_ready_key(backend: str) -> str:
    if backend not in _INPUT_READY_KEYS:
        raise ValueError(f"Unsupported backend for input ready key: {backend}")
    return _INPUT_READY_KEYS[backend]


def publish_input_ready(
    backend: str,
    *,
    source: str = "startup",
    host: str = "localhost",
    port: int = 6379,
    redis_client: Any | None = None,
) -> dict[str, Any]:
    client = redis_client or create_redis_client(host=host, port=port)
    ready_key = get_input_ready_key(backend)
    now_realtime = time.time()
    payload = {
        "backend": backend,
        "source": str(source),
        "epoch_id": int(now_realtime * 1_000_000),
        "ready_timestamp_ms": int(now_realtime * 1000),
        "ready_timestamp_realtime": now_realtime,
        "ready_timestamp_monotonic": time.monotonic(),
    }
    pipe = client.pipeline()
    stream_keys = _INPUT_STREAM_KEYS.get(backend, ())
    if stream_keys:
        pipe.delete(*stream_keys)
    pipe.set(ready_key, json.dumps(payload))
    pipe.execute()
    return payload


def publish_reset_command(
    reset_category: str = "3",
    *,
    host: str = "localhost",
    port: int = 6379,
    redis_client: Any | None = None,
) -> None:
    client = redis_client or create_redis_client(host=host, port=port)
    payload = {
        "reset_category": str(reset_category),
        "timestamp": int(time.time() * 1000),
    }
    client.set(RESET_TRIGGER_KEY, json.dumps(payload))
    client.expire(RESET_TRIGGER_KEY, 5)


def read_reset_trigger(
    *,
    host: str = "localhost",
    port: int = 6379,
    redis_client: Any | None = None,
) -> dict[str, Any] | None:
    client = redis_client or create_redis_client(host=host, port=port, decode_responses=True)
    raw = client.get(RESET_TRIGGER_KEY)
    if not raw:
        return None
    return json.loads(raw)


def clear_reset_trigger(
    *,
    host: str = "localhost",
    port: int = 6379,
    redis_client: Any | None = None,
) -> None:
    client = redis_client or create_redis_client(host=host, port=port)
    client.delete(RESET_TRIGGER_KEY)


def publish_reset_complete(
    *,
    host: str = "localhost",
    port: int = 6379,
    redis_client: Any | None = None,
) -> None:
    client = redis_client or create_redis_client(host=host, port=port)
    payload = {
        "status": "complete",
        "timestamp": int(time.time() * 1000),
    }
    client.set(RESET_COMPLETE_KEY, json.dumps(payload))
    client.expire(RESET_COMPLETE_KEY, 5)


def consume_reset_complete(
    *,
    host: str = "localhost",
    port: int = 6379,
    redis_client: Any | None = None,
) -> bool:
    client = redis_client or create_redis_client(host=host, port=port, decode_responses=True)
    raw = client.get(RESET_COMPLETE_KEY)
    if not raw:
        return False
    data = json.loads(raw)
    if data.get("status") != "complete":
        return False
    client.delete(RESET_COMPLETE_KEY)
    return True
