from __future__ import annotations

"""Runtime bridge between the Kimodo 40D action schema and SONIC Protocol v1.

The SONIC repository intentionally remains an external runtime.  This module
only owns the boundary between the two projects:

* SONIC ``g1_debug`` state (MuJoCo order) -> Kimodo 64D state (canonical /
  IsaacLab order),
* Kimodo 40D semantic reference action -> SONIC streamed joint reference,
* deterministic packed ZMQ Protocol v1 messages.

No hardware command is issued by this module.  The caller decides whether to
publish the returned packet to the SONIC C++ process.
"""

import json
import struct
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation


# SONIC's C++ output publishes body_q/body_dq in MuJoCo order.  The source and
# target arrays are the official mapping in policy_parameters.hpp.  Kimodo's
# canonical 29D order is the same order used by SONIC Protocol v1's
# ``joint_pos``/``joint_vel`` fields (IsaacLab order, pelvis omitted).
ISAACLAB_TO_MUJOCO = np.asarray(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
MUJOCO_TO_ISAACLAB = np.asarray(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)

SONIC_PROTOCOL_VERSION = 1
# SONIC Protocol v1 uses a fixed 4096-byte JSON header.  Keep this value
# explicit here instead of relying on the size of the current field list: the
# C++ subscriber always advances the payload offset by exactly 4096 bytes.
SONIC_HEADER_SIZE = 4096
KIMODO_STATE_DIM = 64
KIMODO_ACTION_DIM = 40


class BridgeSafetyError(ValueError):
    """Raised when a state/action would be unsafe or ambiguous to publish."""


@dataclass(frozen=True)
class SafetyLimits:
    min_root_z: float = 0.50
    max_root_z: float = 1.20
    max_joint_abs: float = 4.0
    max_joint_step: float = 0.75
    max_joint_velocity: float = 35.0
    max_root_xy_step: float = 0.15
    max_root_z_step: float = 0.12
    max_root_xy_abs: float = 10.0
    max_reference_age_s: float = 0.25


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise BridgeSafetyError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise BridgeSafetyError(f"{name} contains NaN or Inf")
    return arr.copy()


def normalize_quat_wxyz(quat: Any, name: str = "quaternion") -> np.ndarray:
    q = _finite_array(quat, (4,), name)
    norm = float(np.linalg.norm(q))
    if norm < 1e-7:
        raise BridgeSafetyError(f"{name} has near-zero norm")
    return q / norm


def optional_scalar(value: Any, name: str) -> float | None:
    """Decode a finite scalar that may arrive as a Python/NumPy scalar/list."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 1 or not np.isfinite(arr[0]):
        raise BridgeSafetyError(f"{name} must be one finite scalar")
    return float(arr[0])


def quat_mul_wxyz(q1: Any, q2: Any) -> np.ndarray:
    a = normalize_quat_wxyz(q1, "q1")
    b = normalize_quat_wxyz(q2, "q2")
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return normalize_quat_wxyz(
        np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float32,
        ),
        "quaternion product",
    )


def quat_conjugate_wxyz(quat: Any) -> np.ndarray:
    q = normalize_quat_wxyz(quat)
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_heading_wxyz(quat: Any) -> np.ndarray:
    """Return the yaw-only heading of a WXYZ quaternion."""
    q = normalize_quat_wxyz(quat)
    w, x, y, z = q
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32)


def rot6d_row_to_quat_wxyz(rot6d: Any) -> np.ndarray:
    """Decode Kimodo's row-stacked first-two-columns 6D rotation."""
    flat = _finite_array(rot6d, (6,), "root rotation 6D")
    col0 = flat[[0, 2, 4]]
    col1 = flat[[1, 3, 5]]
    n0 = float(np.linalg.norm(col0))
    if n0 < 1e-7:
        raise BridgeSafetyError("root rotation 6D first column is degenerate")
    col0 = col0 / n0
    col1 = col1 - np.dot(col0, col1) * col0
    n1 = float(np.linalg.norm(col1))
    if n1 < 1e-7:
        raise BridgeSafetyError("root rotation 6D columns are collinear")
    col1 = col1 / n1
    col2 = np.cross(col0, col1)
    rot = np.stack((col0, col1, col2), axis=1)
    quat_xyzw = SciPyRotation.from_matrix(rot).as_quat().astype(np.float32)
    return normalize_quat_wxyz(quat_xyzw[[3, 0, 1, 2]], "root reference quaternion")


def _build_header(fields: list[dict[str, Any]], version: int = 1) -> bytes:
    header = {"v": int(version), "endian": "le", "count": 1, "fields": fields}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(encoded) > SONIC_HEADER_SIZE:
        raise ValueError(f"SONIC message header is too large: {len(encoded)}")
    return encoded.ljust(SONIC_HEADER_SIZE, b"\x00")


def pack_pose_v1(
    *,
    joint_pos_isaaclab: Any,
    joint_vel_isaaclab: Any,
    body_quat_wxyz: Any,
    root_position: Any,
    frame_index: int,
    timestamp_monotonic: float | None = None,
    catch_up: bool = True,
    left_hand_joints: Any | None = None,
    right_hand_joints: Any | None = None,
) -> bytes:
    """Pack one SONIC Protocol-v1 pose frame.

    ``joint_pos`` and ``joint_vel`` are sent in Kimodo's canonical/IsaacLab
    order required by SONIC Protocol v1.  Kimodo's action codec emits this
    order and the SONIC streamed ``MotionSequence`` consumes it directly;
    **no MuJoCo permutation is applied at this boundary**.  The root
    quaternion is WXYZ and the root position is in the streamed reference
    frame (x-forward, y-left, z-up).
    """
    q = _finite_array(joint_pos_isaaclab, (29,), "joint_pos")
    dq = _finite_array(joint_vel_isaaclab, (29,), "joint_vel")
    quat = normalize_quat_wxyz(body_quat_wxyz, "body_quat")
    root = _finite_array(root_position, (3,), "root_position")
    fields: list[dict[str, Any]] = [
        {"name": "joint_pos", "dtype": "f32", "shape": [1, 29]},
        {"name": "joint_vel", "dtype": "f32", "shape": [1, 29]},
        {"name": "body_quat", "dtype": "f32", "shape": [1, 1, 4]},
        {"name": "root_position", "dtype": "f32", "shape": [1, 3]},
        {"name": "frame_index", "dtype": "i64", "shape": [1]},
        {"name": "catch_up", "dtype": "bool", "shape": [1]},
    ]
    arrays = [
        np.ascontiguousarray(q.reshape(1, 29), dtype=np.float32),
        np.ascontiguousarray(dq.reshape(1, 29), dtype=np.float32),
        np.ascontiguousarray(quat.reshape(1, 1, 4), dtype=np.float32),
        np.ascontiguousarray(root.reshape(1, 3), dtype=np.float32),
        np.asarray([int(frame_index)], dtype=np.int64),
        np.asarray([bool(catch_up)], dtype=np.bool_),
    ]
    if timestamp_monotonic is not None:
        fields.append({"name": "timestamp_monotonic", "dtype": "f64", "shape": [1]})
        arrays.append(np.asarray([float(timestamp_monotonic)], dtype=np.float64))
    for name, value in (("left_hand_joints", left_hand_joints), ("right_hand_joints", right_hand_joints)):
        if value is None:
            continue
        hand = _finite_array(value, (7,), name)
        fields.append({"name": name, "dtype": "f32", "shape": [1, 7]})
        arrays.append(np.ascontiguousarray(hand.reshape(1, 7), dtype=np.float32))
    payload = b"".join(arr.tobytes() for arr in arrays)
    return b"pose" + _build_header(fields, SONIC_PROTOCOL_VERSION) + payload


def state_msg_to_kimodo_state(
    state_msg: dict[str, Any],
    *,
    initial_root_quat_wxyz: np.ndarray | None,
    previous_state: tuple[np.ndarray, float] | None = None,
    now_monotonic: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Convert a SONIC ``g1_debug`` message to state and raw canonical values.

    Returns ``(state64, q29, dq29, root_quat, timestamp, initial_root_quat)``.
    The C++ publisher provides ``body_q``/``body_dq`` in MuJoCo order and
    ``base_quat`` in WXYZ.
    """
    if "body_q" not in state_msg or "base_quat" not in state_msg:
        raise BridgeSafetyError("g1_debug must contain body_q and base_quat")
    source_timestamp = optional_scalar(
        state_msg.get("state_monotonic_timestamp"),
        "g1_debug.state_monotonic_timestamp",
    )
    current_timestamp = float(
        source_timestamp
        if source_timestamp is not None and source_timestamp > 0.0
        else (now_monotonic if now_monotonic is not None else time.monotonic())
    )
    q_mj = _finite_array(state_msg["body_q"], (29,), "g1_debug.body_q")
    if "body_dq" in state_msg:
        dq_mj = _finite_array(state_msg["body_dq"], (29,), "g1_debug.body_dq")
    elif previous_state is not None:
        prev_q, prev_t = previous_state
        current_t = current_timestamp
        dt = max(current_t - float(prev_t), 1e-4)
        dq_mj = (q_mj - prev_q) / dt
    else:
        dq_mj = np.zeros(29, dtype=np.float32)
    q = q_mj[MUJOCO_TO_ISAACLAB]
    dq = dq_mj[MUJOCO_TO_ISAACLAB]
    root_quat = normalize_quat_wxyz(state_msg["base_quat"], "g1_debug.base_quat")
    if initial_root_quat_wxyz is None:
        initial_root_quat_wxyz = root_quat.copy()
    initial_root_quat_wxyz = normalize_quat_wxyz(initial_root_quat_wxyz, "initial root quaternion")
    relative = quat_mul_wxyz(quat_conjugate_wxyz(quat_heading_wxyz(initial_root_quat_wxyz)), root_quat)
    rel_xyzw = relative[[1, 2, 3, 0]]
    state_rot = SciPyRotation.from_quat(rel_xyzw).as_matrix().astype(np.float32)
    # Match the training/runtime row-stacked 6D convention.
    rot6d = state_rot[:, :2].reshape(6).astype(np.float32)
    state = np.concatenate((rot6d, q, dq), axis=0).astype(np.float32)
    if state.shape != (KIMODO_STATE_DIM,) or not np.isfinite(state).all():
        raise BridgeSafetyError("constructed Kimodo state is invalid")
    timestamp = current_timestamp
    return state, q, dq, root_quat, timestamp, initial_root_quat_wxyz


class KimodoActionRuntime:
    """Decode 40D Kimodo actions into SONIC Protocol-v1 reference frames."""

    def __init__(self, limits: SafetyLimits | None = None, max_root_delta_deg: float = 26.0, reference_rate_hz: float = 50.0):
        self.limits = limits or SafetyLimits()
        self.max_root_delta_rad = np.deg2rad(float(max_root_delta_deg)) if max_root_delta_deg > 0 else None
        self.reference_rate_hz = max(float(reference_rate_hz), 1.0)
        self.reset()

    def reset(self) -> None:
        self.reference_root_position = np.zeros(3, dtype=np.float32)
        self.previous_reference_z: float | None = None
        self.previous_joint_q: np.ndarray | None = None
        self.previous_reference_quat: np.ndarray | None = None
        self.frame_index = 0

    def decode(self, action: Any, *, current_root_quat_wxyz: Any) -> dict[str, Any]:
        action = _finite_array(action, (KIMODO_ACTION_DIM,), "Kimodo action")
        # Validate the measured quaternion at the boundary.  SONIC applies
        # the initial-heading alignment to the streamed reference quaternion;
        # the bridge must not apply that heading a second time.
        normalize_quat_wxyz(current_root_quat_wxyz, "current root quaternion")
        root_z = float(action[2])
        if not self.limits.min_root_z <= root_z <= self.limits.max_root_z:
            raise BridgeSafetyError(f"action.root_z={root_z:.4f} outside [{self.limits.min_root_z}, {self.limits.max_root_z}]")
        q_ref = rot6d_row_to_quat_wxyz(action[3:9])
        # Enforce the same per-step root safety bound used by the sim runtime.
        if self.previous_reference_quat is not None and self.max_root_delta_rad is not None:
            rel = quat_mul_wxyz(quat_conjugate_wxyz(self.previous_reference_quat), q_ref)
            angle = 2.0 * np.arccos(np.clip(abs(float(rel[0])), -1.0, 1.0))
            if angle > self.max_root_delta_rad:
                # Slerp without importing the larger runtime module.
                dot = float(np.dot(self.previous_reference_quat, q_ref))
                q1 = q_ref if dot >= 0 else -q_ref
                dot = abs(dot)
                t = self.max_root_delta_rad / max(angle, 1e-8)
                if dot > 0.9995:
                    q_ref = normalize_quat_wxyz(self.previous_reference_quat + t * (q1 - self.previous_reference_quat))
                else:
                    theta = np.arccos(np.clip(dot, -1.0, 1.0))
                    q_ref = normalize_quat_wxyz(
                        (np.sin((1.0 - t) * theta) * self.previous_reference_quat + np.sin(t * theta) * q1)
                        / max(np.sin(theta), 1e-8)
                    )
        previous_z = root_z if self.previous_reference_z is None else self.previous_reference_z
        # Action XY is local to the reference root. Convert it to the streamed
        # MuJoCo reference frame before accumulating root_position.
        rot = SciPyRotation.from_quat(q_ref[[1, 2, 3, 0]]).as_matrix().astype(np.float32)
        local_delta = np.zeros(3, dtype=np.float32)
        local_delta[:2] = action[:2]
        dz = root_z - previous_z
        if abs(float(rot[2, 2])) > 1e-6:
            local_delta[2] = (dz - rot[2, 0] * local_delta[0] - rot[2, 1] * local_delta[1]) / rot[2, 2]
        world_delta = rot.dot(local_delta)
        if self.previous_reference_z is not None and abs(root_z - previous_z) > self.limits.max_root_z_step:
            raise BridgeSafetyError("action.root_z has an implausible one-frame jump")
        if np.linalg.norm(world_delta[:2]) > self.limits.max_root_xy_step:
            raise BridgeSafetyError("action.root_p has an implausible one-frame XY jump")
        self.reference_root_position[:2] += world_delta[:2]
        self.reference_root_position[2] = root_z
        if np.linalg.norm(self.reference_root_position[:2]) > self.limits.max_root_xy_abs:
            raise BridgeSafetyError("action.root_p exceeded the configured XY workspace")
        q = action[9:38].copy()
        if np.max(np.abs(q)) > self.limits.max_joint_abs:
            raise BridgeSafetyError("action.joint_pos contains an implausible angle")
        if self.previous_joint_q is not None and np.max(np.abs(q - self.previous_joint_q)) > self.limits.max_joint_step:
            raise BridgeSafetyError("action.joint_pos has an implausible one-frame jump")
        dq = np.zeros(29, dtype=np.float32) if self.previous_joint_q is None else (q - self.previous_joint_q) * self.reference_rate_hz
        if np.max(np.abs(dq)) > self.limits.max_joint_velocity:
            raise BridgeSafetyError("action.joint_pos implies an implausible joint velocity")
        hand = np.clip(action[38:40], 0.0, 1.0).astype(np.float32)
        result = {
            "joint_pos": q,
            "joint_vel": dq,
            "body_quat": q_ref,
            "root_position": self.reference_root_position.copy(),
            "hand_binary": hand,
            "frame_index": self.frame_index,
        }
        self.previous_reference_z = root_z
        self.previous_reference_quat = q_ref.copy()
        self.previous_joint_q = q.copy()
        self.frame_index += 1
        return result
