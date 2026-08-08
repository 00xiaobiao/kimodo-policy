"""Builders for ZMQ wire-format messages on the 'command', 'planner', and 'pose' topics."""

import json
import struct
from typing import Sequence

import numpy as np

HEADER_SIZE = 1280


def _build_header(fields: list, version: int = 1, count: int = 1) -> bytes:
    header = {
        "v": version,
        "endian": "le",
        "count": count,
        "fields": fields,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def build_command_message(
    start: bool, stop: bool, planner: bool, delta_heading: float | None = None
) -> bytes:
    fields = [
        {"name": "start", "dtype": "u8", "shape": [1]},
        {"name": "stop", "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    payload = b"".join(
        (
            struct.pack("B", 1 if start else 0),
            struct.pack("B", 1 if stop else 0),
            struct.pack("B", 1 if planner else 0),
        )
    )

    if delta_heading is not None:
        fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
        payload += struct.pack("<f", float(delta_heading))

    header = _build_header(fields, version=1, count=1)
    return b"command" + header + payload


def build_planner_message(
    mode: int,
    movement: Sequence[float],
    facing: Sequence[float],
    speed: float = -1.0,
    height: float = -1.0,
    upper_body_position: Sequence[float] | None = None,
    upper_body_velocity: Sequence[float] | None = None,
    left_hand_position: Sequence[float] | None = None,
    right_hand_position: Sequence[float] | None = None,
    vr_3pt_position: Sequence[float] | None = None,
    vr_3pt_orientation: Sequence[float] | None = None,
    vr_3pt_compliance: Sequence[float] | None = None,
) -> bytes:
    if len(movement) != 3:
        raise ValueError("movement must have length 3")
    if len(facing) != 3:
        raise ValueError("facing must have length 3")

    fields = [
        {"name": "mode", "dtype": "i32", "shape": [1]},
        {"name": "movement", "dtype": "f32", "shape": [3]},
        {"name": "facing", "dtype": "f32", "shape": [3]},
        {"name": "speed", "dtype": "f32", "shape": [1]},
        {"name": "height", "dtype": "f32", "shape": [1]},
    ]

    payload = b"".join(
        (
            struct.pack("<i", int(mode)),
            struct.pack("<fff", float(movement[0]), float(movement[1]), float(movement[2])),
            struct.pack("<fff", float(facing[0]), float(facing[1]), float(facing[2])),
            struct.pack("<f", float(speed)),
            struct.pack("<f", float(height)),
        )
    )

    if upper_body_position is not None:
        fields.append(
            {"name": "upper_body_position", "dtype": "f32", "shape": [len(upper_body_position)]}
        )
        for value in upper_body_position:
            payload += struct.pack("<f", float(value))

    if upper_body_velocity is not None:
        fields.append(
            {"name": "upper_body_velocity", "dtype": "f32", "shape": [len(upper_body_velocity)]}
        )
        for value in upper_body_velocity:
            payload += struct.pack("<f", float(value))

    if left_hand_position is not None:
        fields.append({"name": "left_hand_joints", "dtype": "f32", "shape": [len(left_hand_position)]})
        for value in left_hand_position:
            payload += struct.pack("<f", float(value))

    if right_hand_position is not None:
        fields.append(
            {"name": "right_hand_joints", "dtype": "f32", "shape": [len(right_hand_position)]}
        )
        for value in right_hand_position:
            payload += struct.pack("<f", float(value))

    if vr_3pt_position is not None:
        fields.append({"name": "vr_position", "dtype": "f32", "shape": [len(vr_3pt_position)]})
        for value in vr_3pt_position:
            payload += struct.pack("<f", float(value))

    if vr_3pt_orientation is not None:
        fields.append(
            {"name": "vr_orientation", "dtype": "f32", "shape": [len(vr_3pt_orientation)]}
        )
        for value in vr_3pt_orientation:
            payload += struct.pack("<f", float(value))

    if vr_3pt_compliance is not None:
        fields.append({"name": "vr_compliance", "dtype": "f32", "shape": [len(vr_3pt_compliance)]})
        for value in vr_3pt_compliance:
            payload += struct.pack("<f", float(value))

    header = _build_header(fields, version=1, count=1)
    return b"planner" + header + payload


def pack_pose_message(pose_data: dict, topic: str = "pose", version: int = 3) -> bytes:
    fields = []
    binary_data = []

    for key, value in pose_data.items():
        if not isinstance(value, np.ndarray):
            continue

        if value.dtype == np.float32:
            dtype_str = "f32"
        elif value.dtype == np.float64:
            dtype_str = "f64"
        elif value.dtype == np.int32:
            dtype_str = "i32"
        elif value.dtype == np.int64:
            dtype_str = "i64"
        elif value.dtype == bool:
            dtype_str = "bool"
        else:
            dtype_str = "f32"
            value = value.astype(np.float32)

        fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})

        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))

        binary_data.append(value.tobytes())

    header_bytes = _build_header(fields, version=version, count=1)
    topic_bytes = topic.encode("utf-8")
    data_bytes = b"".join(binary_data)
    return topic_bytes + header_bytes + data_bytes
