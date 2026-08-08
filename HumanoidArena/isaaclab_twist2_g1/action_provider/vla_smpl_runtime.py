from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

try:
    from pico_server.sonic_tools.trl.utils.rotation_conversion import decompose_rotation_aa
    from pico_server.sonic_tools.trl.utils.torch_transform import compute_human_joints
    from pico_server.pico_server_pose_only import process_smpl_joints
except ImportError:
    decompose_rotation_aa = None
    compute_human_joints = None
    process_smpl_joints = None


CANONICAL_G1_JOINT_NAMES_29 = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

TWIST2_G1_JOINT_NAMES_29 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

TWIST2_TO_CANONICAL_INDEX_29 = [
    TWIST2_G1_JOINT_NAMES_29.index(name) for name in CANONICAL_G1_JOINT_NAMES_29
]
CANONICAL_TO_TWIST2_INDEX_29 = [
    CANONICAL_G1_JOINT_NAMES_29.index(name) for name in TWIST2_G1_JOINT_NAMES_29
]

SMPL_JOINT_ORDER_24 = [
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
]

SMPL_PARENT_INDEX_24 = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]

SMPL_POSE_ORDER_21 = [
    name for name in SMPL_JOINT_ORDER_24 if name not in ("Pelvis", "Left_Hand", "Right_Hand")
]

SMPL_POSE_TO_JOINT_INDEX_21 = [SMPL_JOINT_ORDER_24.index(name) for name in SMPL_POSE_ORDER_21]
OFFICIAL_WRIST_INDICES = np.array([23, 24, 25, 26, 27, 28], dtype=np.int64)

VLA_SMPL_STATE_DIM = 64
VLA_SMPL_ACTION_DIM = 40
VLA_SMPL_HAND_DIM = 2


def quat_normalize_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return (quat / np.clip(norm, 1e-12, None)).astype(np.float32)


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out.astype(np.float32)


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = quat_normalize_wxyz(q1)
    q2 = quat_normalize_wxyz(q2)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    out = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    return quat_normalize_wxyz(out)


def quat_to_rot6d_wxyz(quat: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat, dtype=np.float32)[..., [1, 2, 3, 0]]
    rot = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    return rot[..., :, :2].reshape(rot.shape[:-2] + (6,)).astype(np.float32)


def quat_to_roll_pitch_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = quat_normalize_wxyz(np.asarray(quat, dtype=np.float32).reshape(-1))
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}")
    w, x, y, z = quat
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    return np.array([roll, pitch], dtype=np.float32)


def quat_from_roll_pitch_yaw_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    quat_xyzw = R.from_euler("xyz", [float(roll), float(pitch), float(yaw)], degrees=False).as_quat()
    return np.asarray(quat_xyzw[[3, 0, 1, 2]], dtype=np.float32)


def yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    quat_xyzw = quat_normalize_wxyz(np.asarray(quat, dtype=np.float32).reshape(4))[[1, 2, 3, 0]]
    return float(R.from_quat(quat_xyzw).as_euler("xyz", degrees=False)[2])


def rot6d_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    flat = rot6d.reshape(-1, 6)
    col0 = np.stack([flat[:, 0], flat[:, 2], flat[:, 4]], axis=-1)
    col1 = np.stack([flat[:, 1], flat[:, 3], flat[:, 5]], axis=-1)

    col0_norm = np.linalg.norm(col0, axis=-1, keepdims=True)
    degenerate_col0 = col0_norm.squeeze(-1) < 1e-8
    if np.any(degenerate_col0):
        col0 = col0.copy()
        col0[degenerate_col0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        col0_norm = np.linalg.norm(col0, axis=-1, keepdims=True)
    col0 = col0 / np.clip(col0_norm, 1e-12, None)

    col1 = col1 - np.sum(col0 * col1, axis=-1, keepdims=True) * col0
    col1_norm = np.linalg.norm(col1, axis=-1, keepdims=True)
    degenerate_col1 = col1_norm.squeeze(-1) < 1e-8
    if np.any(degenerate_col1):
        col1 = col1.copy()
        fallback = np.zeros_like(col1)
        fallback[:] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        alt = np.zeros_like(col1)
        alt[:] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        use_alt = np.abs(col0[:, 1]) > 0.9
        fallback[use_alt] = alt[use_alt]
        col1[degenerate_col1] = fallback[degenerate_col1]
        col1 = col1 - np.sum(col0 * col1, axis=-1, keepdims=True) * col0
        col1_norm = np.linalg.norm(col1, axis=-1, keepdims=True)
    col1 = col1 / np.clip(col1_norm, 1e-12, None)

    col2 = np.cross(col0, col1)
    col2 = col2 / np.clip(np.linalg.norm(col2, axis=-1, keepdims=True), 1e-12, None)

    rot = np.stack([col0, col1, col2], axis=-1)
    return rot.reshape(rot6d.shape[:-1] + (3, 3)).astype(np.float32)


def rot6d_to_quat_wxyz(rot6d: np.ndarray) -> np.ndarray:
    rot = rot6d_to_rotation_matrix(rot6d)
    quat_xyzw = R.from_matrix(rot.reshape(-1, 3, 3)).as_quat().astype(np.float32)
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    return quat_wxyz.reshape(rot.shape[:-2] + (4,)).astype(np.float32)


def _convert_rot6d_col_to_row(rot6d: np.ndarray) -> np.ndarray:
    """Convert column-stacked 6D rotation to row-stacked format used by rot6d_to_rotation_matrix()."""
    arr = np.asarray(rot6d, dtype=np.float32)
    flat = arr.reshape(-1, 6)
    row_flat = flat[:, [0, 3, 1, 4, 2, 5]]
    return row_flat.reshape(arr.shape).astype(np.float32)


def rot6d_to_quat_wxyz_with_layout(rot6d: np.ndarray, layout: str = "row") -> np.ndarray:
    layout = str(layout).strip().lower()
    if layout == "row":
        return rot6d_to_quat_wxyz(rot6d)
    if layout == "col":
        return rot6d_to_quat_wxyz(_convert_rot6d_col_to_row(rot6d))
    raise ValueError(f"Unsupported rot6d layout: {layout}. Expected one of ['row', 'col'].")


def quat_angle_between_wxyz(q_ref: np.ndarray, q_cur: np.ndarray) -> float:
    """Return smallest rotation angle (radians) between two wxyz quaternions."""
    q_ref = quat_normalize_wxyz(np.asarray(q_ref, dtype=np.float32).reshape(4))
    q_cur = quat_normalize_wxyz(np.asarray(q_cur, dtype=np.float32).reshape(4))
    rel = quat_mul_wxyz(quat_conjugate_wxyz(q_ref), q_cur)
    w = float(np.clip(np.abs(rel[0]), -1.0, 1.0))
    return float(2.0 * np.arccos(w))


def quat_slerp_shortest_wxyz(q_start: np.ndarray, q_end: np.ndarray, t: float) -> np.ndarray:
    """Shortest-path SLERP in wxyz format."""
    q0 = quat_normalize_wxyz(np.asarray(q_start, dtype=np.float32).reshape(4))
    q1 = quat_normalize_wxyz(np.asarray(q_end, dtype=np.float32).reshape(4))
    t = float(np.clip(t, 0.0, 1.0))

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return quat_normalize_wxyz(out)

    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    if sin_theta_0 < 1e-8:
        return q0.copy()

    theta = theta_0 * t
    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(np.sin(theta) / sin_theta_0)
    out = s0 * q0 + s1 * q1
    return quat_normalize_wxyz(out)


def rot6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    rot = rot6d_to_rotation_matrix(rot6d)
    rotvec = R.from_matrix(rot.reshape(-1, 3, 3)).as_rotvec().astype(np.float32)
    return rotvec.reshape(rot.shape[:-2] + (3,)).astype(np.float32)


def reorder_twist2_to_canonical_29(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.shape[-1] != 29:
        raise ValueError(f"Expected last dimension 29, got {arr.shape}")
    return arr[..., TWIST2_TO_CANONICAL_INDEX_29].astype(np.float32)


def reorder_canonical_to_twist2_29(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.shape[-1] != 29:
        raise ValueError(f"Expected last dimension 29, got {arr.shape}")
    return arr[..., CANONICAL_TO_TWIST2_INDEX_29].astype(np.float32)


def build_vla_observation_state(
    *,
    root_orientation_wxyz: np.ndarray,
    joint_pos_canonical_29: np.ndarray,
    joint_vel_canonical_29: np.ndarray,
) -> np.ndarray:
    state = np.concatenate(
        [
            quat_to_rot6d_wxyz(root_orientation_wxyz).reshape(-1),
            np.asarray(joint_pos_canonical_29, dtype=np.float32).reshape(-1),
            np.asarray(joint_vel_canonical_29, dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)
    if state.shape != (VLA_SMPL_STATE_DIM,):
        raise ValueError(f"Unexpected VLA observation.state shape: {state.shape}")
    return state


def build_vla_action(
    *,
    root_xy_delta_world: np.ndarray,
    root_z: np.ndarray | float,
    root_rot6d: np.ndarray,
    joint_pos_canonical_29: np.ndarray,
    hand_binary: np.ndarray,
) -> np.ndarray:
    action = np.concatenate(
        [
            np.asarray(root_xy_delta_world, dtype=np.float32).reshape(2),
            np.asarray(root_z, dtype=np.float32).reshape(1),
            np.asarray(root_rot6d, dtype=np.float32).reshape(6),
            np.asarray(joint_pos_canonical_29, dtype=np.float32).reshape(29),
            np.asarray(hand_binary, dtype=np.float32).reshape(2),
        ],
        axis=0,
    ).astype(np.float32)
    if action.shape != (VLA_SMPL_ACTION_DIM,):
        raise ValueError(f"Unexpected VLA action shape: {action.shape}")
    return action


def _wrap_to_pi(angle: float) -> float:
    return ((float(angle) + np.pi) % (2.0 * np.pi)) - np.pi


def compute_planar_velocity_from_delta(root_trans_world_delta: np.ndarray, control_dt: float) -> np.ndarray:
    dt = max(float(control_dt), 1e-6)
    delta = np.asarray(root_trans_world_delta, dtype=np.float32).reshape(3)
    return (delta[:2] / dt).astype(np.float32)


def compute_yaw_rate_from_quats(
    prev_root_quat_wxyz: np.ndarray | None,
    curr_root_quat_wxyz: np.ndarray,
    control_dt: float,
) -> np.ndarray:
    if prev_root_quat_wxyz is None:
        return np.zeros((1,), dtype=np.float32)
    dt = max(float(control_dt), 1e-6)
    prev_yaw = R.from_quat(np.asarray(prev_root_quat_wxyz, dtype=np.float32)[[1, 2, 3, 0]]).as_euler(
        "xyz", degrees=False
    )[2]
    curr_yaw = R.from_quat(np.asarray(curr_root_quat_wxyz, dtype=np.float32)[[1, 2, 3, 0]]).as_euler(
        "xyz", degrees=False
    )[2]
    yaw_delta = _wrap_to_pi(curr_yaw - prev_yaw)
    return np.array([yaw_delta / dt], dtype=np.float32)


def _require_smpl_utils() -> None:
    if compute_human_joints is None or decompose_rotation_aa is None:
        raise RuntimeError(
            "SMPL runtime utilities are unavailable. Expected pico_server.sonic_tools.trl utils to be importable."
        )


def _require_sonic_pose_processing() -> None:
    _require_smpl_utils()
    if process_smpl_joints is None:
        raise RuntimeError(
            "SONIC pose processing utilities are unavailable. Expected pico_server.pico_server_pose_only to be importable."
        )


def _build_full_local_axis_angle_24(body_pose_local_axis_angle_21: np.ndarray) -> np.ndarray:
    local_axis_angle = np.zeros((24, 3), dtype=np.float32)
    body_pose_local_axis_angle_21 = np.asarray(body_pose_local_axis_angle_21, dtype=np.float32).reshape(21, 3)
    for pose_idx, joint_idx in enumerate(SMPL_POSE_TO_JOINT_INDEX_21):
        local_axis_angle[joint_idx] = body_pose_local_axis_angle_21[pose_idx]
    return local_axis_angle


def build_global_smpl_quats_wxyz(
    root_quat_wxyz: np.ndarray,
    body_pose_local_axis_angle_21: np.ndarray,
) -> np.ndarray:
    root_quat_wxyz = quat_normalize_wxyz(np.asarray(root_quat_wxyz, dtype=np.float32).reshape(4))
    local_axis_angle = _build_full_local_axis_angle_24(body_pose_local_axis_angle_21)
    local_quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (24, 1))
    local_quats[0] = root_quat_wxyz

    for joint_idx in range(1, 24):
        local_quats[joint_idx] = _rotvec_to_quat_wxyz(local_axis_angle[joint_idx]).reshape(4)

    global_quats = np.zeros((24, 4), dtype=np.float32)
    global_quats[0] = root_quat_wxyz
    for joint_idx in range(1, 24):
        parent_idx = SMPL_PARENT_INDEX_24[joint_idx]
        global_quats[joint_idx] = quat_mul_wxyz(global_quats[parent_idx], local_quats[joint_idx])
    return global_quats.astype(np.float32)


def _rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_rotvec(np.asarray(rotvec, dtype=np.float32).reshape(-1, 3)).as_quat().astype(np.float32)
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    return quat_wxyz.reshape(np.asarray(rotvec).shape[:-1] + (4,)).astype(np.float32)


def build_smpl_joint_positions_world(
    *,
    root_axis_angle: np.ndarray,
    body_pose_local_axis_angle_21: np.ndarray,
    root_trans_world: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    _require_smpl_utils()
    body_pose = torch.from_numpy(
        np.asarray(body_pose_local_axis_angle_21, dtype=np.float32).reshape(1, 63)
    ).float().to(device)
    global_orient = torch.from_numpy(np.asarray(root_axis_angle, dtype=np.float32).reshape(1, 3)).float().to(device)
    joints = compute_human_joints(body_pose=body_pose, global_orient=global_orient)
    joints_world = joints[0].detach().cpu().numpy().astype(np.float32)
    joints_world = joints_world + np.asarray(root_trans_world, dtype=np.float32).reshape(1, 3)
    return joints_world.astype(np.float32)


def build_smpl_joints_local(
    *,
    root_quat_wxyz: np.ndarray,
    joints_world: np.ndarray,
    root_trans_world: np.ndarray,
) -> np.ndarray:
    joints_root = np.asarray(joints_world, dtype=np.float32) - np.asarray(root_trans_world, dtype=np.float32).reshape(1, 3)
    root_rot = R.from_quat(np.asarray(root_quat_wxyz, dtype=np.float32)[[1, 2, 3, 0]])
    joints_local = root_rot.inv().apply(joints_root)
    return joints_local.astype(np.float32)


def compute_wrist_joint_pos_from_smpl_pose(body_pose_local_axis_angle_21: np.ndarray) -> np.ndarray:
    _require_smpl_utils()

    body_pose = np.asarray(body_pose_local_axis_angle_21, dtype=np.float32).reshape(1, 21, 3)
    joint_pos = np.zeros(29, dtype=np.float32)

    smpl_l_elbow_aa = body_pose[:, 17]
    smpl_l_wrist_aa = body_pose[:, 19]
    smpl_r_elbow_aa = body_pose[:, 18]
    smpl_r_wrist_aa = body_pose[:, 20]

    def _safe_decompose(rotation_aa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rotation_aa = np.asarray(rotation_aa, dtype=np.float32)
        small = np.linalg.norm(rotation_aa, axis=-1) < 1e-8
        ident = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        twist = np.tile(ident, (rotation_aa.shape[0], 1))
        swing = np.tile(ident, (rotation_aa.shape[0], 1))
        if np.any(~small):
            twist_valid, swing_valid = decompose_rotation_aa(
                rotation_aa[~small], np.array([0, 1, 0], dtype=np.float32)
            )
            twist[~small] = np.asarray(twist_valid, dtype=np.float32)
            swing[~small] = np.asarray(swing_valid, dtype=np.float32)
        return twist, swing

    g1_l_elbow_q_twist, g1_l_elbow_q_swing = _safe_decompose(smpl_l_elbow_aa)
    g1_r_elbow_q_twist, g1_r_elbow_q_swing = _safe_decompose(smpl_r_elbow_aa)

    def _safe_quat_batch_wxyz(quat_batch: np.ndarray) -> np.ndarray:
        quat_batch = np.asarray(quat_batch, dtype=np.float32)
        if quat_batch.ndim != 2 or quat_batch.shape[1] != 4:
            raise ValueError(f"Expected quaternion batch shape (N, 4), got {quat_batch.shape}")
        out = quat_batch.copy()
        invalid_values = ~np.isfinite(out).all(axis=-1)
        if np.any(invalid_values):
            out[invalid_values] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        norms = np.linalg.norm(out, axis=-1, keepdims=True)
        invalid = norms.squeeze(-1) < 1e-8
        if np.any(invalid):
            out[invalid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            norms = np.linalg.norm(out, axis=-1, keepdims=True)
        return (out / np.clip(norms, 1e-12, None)).astype(np.float32)

    g1_l_elbow_q_swing = _safe_quat_batch_wxyz(g1_l_elbow_q_swing)
    g1_r_elbow_q_swing = _safe_quat_batch_wxyz(g1_r_elbow_q_swing)

    l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler("XYZ", degrees=False)
    r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler("XYZ", degrees=False)

    l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
    r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

    joint_pos[23] = (l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0])[0].astype(np.float32)
    joint_pos[25] = l_wrist_euler[:, 1][0].astype(np.float32)
    joint_pos[27] = (l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2])[0].astype(np.float32)

    joint_pos[24] = (-(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0]))[0].astype(np.float32)
    joint_pos[26] = (-r_wrist_euler[:, 1])[0].astype(np.float32)
    joint_pos[28] = (r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2])[0].astype(np.float32)

    return joint_pos.astype(np.float32)


def compute_anchor_rot6d_relative_to_base(
    *,
    root_quat_wxyz: np.ndarray,
    base_quat_wxyz: np.ndarray,
) -> np.ndarray:
    rel_quat_wxyz = quat_mul_wxyz(
        quat_conjugate_wxyz(np.asarray(base_quat_wxyz, dtype=np.float32).reshape(4)),
        np.asarray(root_quat_wxyz, dtype=np.float32).reshape(4),
    )
    return quat_to_rot6d_wxyz(rel_quat_wxyz).astype(np.float32)


def rotate_vector_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat_xyzw = quat_normalize_wxyz(np.asarray(quat, dtype=np.float32).reshape(4))[[1, 2, 3, 0]]
    vec = np.asarray(vec, dtype=np.float32).reshape(3)
    return R.from_quat(quat_xyzw).inv().apply(vec).astype(np.float32)


def build_smplx_frame_dict(
    *,
    root_trans_world: np.ndarray,
    root_quat_wxyz: np.ndarray,
    body_pose_local_axis_angle_21: np.ndarray,
    device: torch.device,
) -> dict[str, list[np.ndarray]]:
    joints_world = build_smpl_joint_positions_world(
        root_axis_angle=R.from_quat(np.asarray(root_quat_wxyz, dtype=np.float32)[[1, 2, 3, 0]]).as_rotvec().astype(np.float32),
        body_pose_local_axis_angle_21=body_pose_local_axis_angle_21,
        root_trans_world=root_trans_world,
        device=device,
    )
    global_quats = build_global_smpl_quats_wxyz(root_quat_wxyz, body_pose_local_axis_angle_21)
    return {
        joint_name: [
            joints_world[joint_idx].astype(np.float32),
            global_quats[joint_idx].astype(np.float32),
        ]
        for joint_idx, joint_name in enumerate(SMPL_JOINT_ORDER_24)
    }


def build_sonic_reference_frame(
    *,
    runtime_frame: "UnifiedSMPLActionFrame",
    device: torch.device,
) -> dict[str, np.ndarray]:
    _require_sonic_pose_processing()
    body_pose = torch.from_numpy(runtime_frame.body_pose_local_axis_angle.reshape(1, 63)).float().to(device)
    global_orient = torch.from_numpy(runtime_frame.root_axis_angle.reshape(1, 3)).float().to(device)
    transl = torch.from_numpy(runtime_frame.root_trans_world.reshape(1, 3)).float().to(device)
    processed = process_smpl_joints(body_pose=body_pose, global_orient=global_orient, transl=transl)

    smpl_pose = processed["smpl_pose"][0].detach().cpu().numpy().reshape(21, 3).astype(np.float32)
    smpl_joints_local = processed["smpl_joints_local"][0].detach().cpu().numpy().astype(np.float32)
    body_quat_w = processed["global_orient_quat"][0].detach().cpu().numpy().astype(np.float32)
    root_z = processed["adjusted_transl"][0, 2].detach().cpu().numpy().reshape(1).astype(np.float32)

    joints_world = build_smpl_joint_positions_world(
        root_axis_angle=runtime_frame.root_axis_angle,
        body_pose_local_axis_angle_21=runtime_frame.body_pose_local_axis_angle,
        root_trans_world=runtime_frame.root_trans_world,
        device=device,
    )
    global_quats = build_global_smpl_quats_wxyz(runtime_frame.root_quat_wxyz, runtime_frame.body_pose_local_axis_angle)
    wrist_ref = compute_wrist_joint_pos_from_smpl_pose(runtime_frame.body_pose_local_axis_angle)
    return {
        "smpl_pose": smpl_pose,
        "smpl_joints_local": smpl_joints_local.astype(np.float32),
        "body_quat_w": body_quat_w.astype(np.float32),
        "wrist_ref": wrist_ref[OFFICIAL_WRIST_INDICES].astype(np.float32),
        "root_z": root_z.astype(np.float32),
        "smplx_frame": {
            joint_name: [
                joints_world[joint_idx].astype(np.float32),
                global_quats[joint_idx].astype(np.float32),
            ]
            for joint_idx, joint_name in enumerate(SMPL_JOINT_ORDER_24)
        },
    }


def build_twist2_mimic_obs(
    *,
    runtime_frame: "UnifiedSMPLActionFrame",
    control_dt: float,
) -> np.ndarray:
    root_quat_wxyz = quat_normalize_wxyz(runtime_frame.root_quat_wxyz.astype(np.float32, copy=False))
    root_lin_vel_world = np.array(
        [
            runtime_frame.root_xy_delta_world[0] / max(float(control_dt), 1e-6),
            runtime_frame.root_xy_delta_world[1] / max(float(control_dt), 1e-6),
            0.0,
        ],
        dtype=np.float32,
    )
    xy_vel = rotate_vector_inverse_wxyz(root_quat_wxyz, root_lin_vel_world)[:2].astype(np.float32)
    roll_pitch = quat_to_roll_pitch_wxyz(root_quat_wxyz)
    yaw_vel = compute_yaw_rate_from_quats(
        runtime_frame.prev_root_quat_wxyz,
        root_quat_wxyz,
        control_dt,
    )
    joint_pos_twist2 = reorder_canonical_to_twist2_29(runtime_frame.joint_pos_canonical_29)
    mimic_obs = np.concatenate(
        [
            xy_vel,
            np.array([runtime_frame.root_z], dtype=np.float32),
            roll_pitch,
            yaw_vel,
            joint_pos_twist2,
        ],
        axis=0,
    ).astype(np.float32)
    if mimic_obs.shape != (35,):
        raise ValueError(f"Unexpected TWIST2 mimic_obs shape: {mimic_obs.shape}")
    return mimic_obs


def build_sonic_joint29_payload(
    *,
    runtime_frame: "UnifiedSMPLActionFrame",
    control_dt: float,
) -> dict[str, np.ndarray]:
    if runtime_frame.prev_joint_pos_canonical_29 is None:
        joint_vel = np.zeros((29,), dtype=np.float32)
    else:
        joint_vel = (
            (runtime_frame.joint_pos_canonical_29 - runtime_frame.prev_joint_pos_canonical_29)
            / max(float(control_dt), 1e-6)
        ).astype(np.float32)
    return {
        "body_pos": runtime_frame.body_pos_world.astype(np.float32, copy=True),
        "body_quat_w": runtime_frame.root_quat_wxyz.astype(np.float32, copy=True),
        "joint_pos": runtime_frame.joint_pos_canonical_29.astype(np.float32, copy=True),
        "joint_vel": joint_vel,
    }


@dataclass
class CanonicalPoseActionRecorder:
    prev_body_pos_world: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_body_pos_world = None

    def step(
        self,
        *,
        body_pos_world: np.ndarray,
        body_quat_wxyz: np.ndarray,
        joint_pos_canonical_29: np.ndarray,
        hand_binary: np.ndarray,
    ) -> np.ndarray:
        body_pos_world = np.asarray(body_pos_world, dtype=np.float32).reshape(3)
        if self.prev_body_pos_world is None:
            root_xy_delta = np.zeros((2,), dtype=np.float32)
        else:
            root_xy_delta = (body_pos_world[:2] - self.prev_body_pos_world[:2]).astype(np.float32)
        self.prev_body_pos_world = body_pos_world.copy()
        return build_vla_action(
            root_xy_delta_world=root_xy_delta,
            root_z=body_pos_world[2],
            root_rot6d=quat_to_rot6d_wxyz(body_quat_wxyz).reshape(6),
            joint_pos_canonical_29=joint_pos_canonical_29,
            hand_binary=hand_binary,
        )


@dataclass
class Twist2ActionMimicRecorder:
    control_dt: float
    yaw_world: float = 0.0
    initialized: bool = False

    def reset(self) -> None:
        self.yaw_world = 0.0
        self.initialized = False

    def step(
        self,
        *,
        mimic_obs: np.ndarray,
        hand_binary: np.ndarray,
    ) -> np.ndarray:
        mimic_obs = np.asarray(mimic_obs, dtype=np.float32).reshape(-1)
        if mimic_obs.shape != (35,):
            raise ValueError(f"Expected TWIST2 mimic_obs shape (35,), got {mimic_obs.shape}")
        xy_vel_local = mimic_obs[0:2].astype(np.float32, copy=False)
        root_z = float(mimic_obs[2])
        roll = float(mimic_obs[3])
        pitch = float(mimic_obs[4])
        yaw_vel = float(mimic_obs[5])
        joint_pos_twist2 = mimic_obs[6:35].astype(np.float32, copy=False)

        if not self.initialized:
            self.initialized = True
        self.yaw_world = _wrap_to_pi(self.yaw_world + yaw_vel * float(self.control_dt))

        root_quat_wxyz = quat_from_roll_pitch_yaw_wxyz(roll=roll, pitch=pitch, yaw=self.yaw_world)
        local_delta = np.array(
            [
                xy_vel_local[0] * float(self.control_dt),
                xy_vel_local[1] * float(self.control_dt),
                0.0,
            ],
            dtype=np.float32,
        )
        root_xy_delta_world = R.from_quat(root_quat_wxyz[[1, 2, 3, 0]]).apply(local_delta)[:2].astype(np.float32)
        return build_vla_action(
            root_xy_delta_world=root_xy_delta_world,
            root_z=root_z,
            root_rot6d=quat_to_rot6d_wxyz(root_quat_wxyz).reshape(6),
            joint_pos_canonical_29=reorder_twist2_to_canonical_29(joint_pos_twist2),
            hand_binary=hand_binary,
        )


def _prefer_workspace_gmr() -> None:
    candidate_roots: list[Path] = []

    gmr_root_env = os.environ.get("GMR_ROOT")
    if gmr_root_env:
        candidate_roots.append(Path(gmr_root_env).expanduser())

    candidate_roots.append(Path(__file__).resolve().parents[2] / "GMR")

    for gmr_root in candidate_roots:
        package_dir = gmr_root / "general_motion_retargeting"
        unitree_g1_xml = gmr_root / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
        if package_dir.is_dir() and unitree_g1_xml.is_file():
            gmr_root_str = str(gmr_root)
            if gmr_root_str not in sys.path:
                sys.path.insert(0, gmr_root_str)
            os.environ.setdefault("GMR_ROOT", gmr_root_str)
            return


def create_twist2_gmr_retarget(actual_human_height: float):
    _prefer_workspace_gmr()
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    return GMR(
        src_human="xrobot",
        tgt_robot="unitree_g1",
        actual_human_height=float(actual_human_height),
    )


@dataclass
class UnifiedSMPLActionFrame:
    root_xy_delta_world: np.ndarray
    root_z: float
    body_pos_world: np.ndarray
    root_orient_rot6d: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_pos_canonical_29: np.ndarray
    hand_binary: np.ndarray
    prev_root_quat_wxyz: np.ndarray | None
    prev_joint_pos_canonical_29: np.ndarray | None


class UnifiedSMPLActionRuntime:
    def __init__(self, root_rot6d_layout: str = "row", max_root_delta_deg: float | None = None) -> None:
        self._root_rot6d_layout = str(root_rot6d_layout).strip().lower()
        if self._root_rot6d_layout not in {"row", "col", "auto"}:
            raise ValueError(
                f"Unsupported root_rot6d_layout={root_rot6d_layout}. "
                "Expected one of ['row', 'col', 'auto']."
            )
        self._max_root_delta_rad: float | None = None
        self.set_max_root_delta_deg(max_root_delta_deg)
        self.reset()

    def reset(self) -> None:
        self._body_xy_world = np.zeros((2,), dtype=np.float32)
        self._prev_root_quat_wxyz: np.ndarray | None = None
        self._prev_joint_pos_canonical_29: np.ndarray | None = None
        self._last_selected_root_rot6d_layout: str = "row"

    def prime_root_quat(self, root_quat_wxyz: np.ndarray) -> None:
        """Prime previous root orientation for continuity-based rot6d layout selection."""
        self._prev_root_quat_wxyz = quat_normalize_wxyz(
            np.asarray(root_quat_wxyz, dtype=np.float32).reshape(4)
        )

    def set_max_root_delta_deg(self, max_root_delta_deg: float | None) -> None:
        if max_root_delta_deg is None:
            self._max_root_delta_rad = None
            return
        value = float(max_root_delta_deg)
        if not np.isfinite(value) or value <= 0.0:
            self._max_root_delta_rad = None
            return
        self._max_root_delta_rad = float(np.deg2rad(value))

    def step(self, action: np.ndarray) -> UnifiedSMPLActionFrame:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (VLA_SMPL_ACTION_DIM,):
            raise ValueError(
                f"Expected unified canonical action dim {VLA_SMPL_ACTION_DIM}, got {action.shape}"
            )

        root_xy_delta_world = action[0:2].astype(np.float32, copy=True)
        root_z = float(action[2])
        root_orient_rot6d = action[3:9].astype(np.float32, copy=True)
        joint_pos_canonical_29 = action[9:38].astype(np.float32, copy=True)
        hand_binary = action[38:40].astype(np.float32, copy=True)

        if self._root_rot6d_layout in {"row", "col"}:
            root_quat_wxyz = rot6d_to_quat_wxyz_with_layout(
                root_orient_rot6d,
                layout=self._root_rot6d_layout,
            ).reshape(4).astype(np.float32)
            self._last_selected_root_rot6d_layout = self._root_rot6d_layout
        else:
            quat_row = rot6d_to_quat_wxyz_with_layout(root_orient_rot6d, layout="row").reshape(4).astype(np.float32)
            quat_col = rot6d_to_quat_wxyz_with_layout(root_orient_rot6d, layout="col").reshape(4).astype(np.float32)
            if self._prev_root_quat_wxyz is None:
                root_quat_wxyz = quat_row
                self._last_selected_root_rot6d_layout = "row"
            else:
                row_delta = quat_angle_between_wxyz(self._prev_root_quat_wxyz, quat_row)
                col_delta = quat_angle_between_wxyz(self._prev_root_quat_wxyz, quat_col)
                if col_delta < row_delta:
                    root_quat_wxyz = quat_col
                    self._last_selected_root_rot6d_layout = "col"
                else:
                    root_quat_wxyz = quat_row
                    self._last_selected_root_rot6d_layout = "row"

        if self._max_root_delta_rad is not None and self._prev_root_quat_wxyz is not None:
            root_delta = quat_angle_between_wxyz(self._prev_root_quat_wxyz, root_quat_wxyz)
            if root_delta > self._max_root_delta_rad:
                blend = self._max_root_delta_rad / max(root_delta, 1e-8)
                root_quat_wxyz = quat_slerp_shortest_wxyz(
                    self._prev_root_quat_wxyz,
                    root_quat_wxyz,
                    blend,
                )

        prev_root_quat = None if self._prev_root_quat_wxyz is None else self._prev_root_quat_wxyz.copy()
        prev_joint_pos = (
            None if self._prev_joint_pos_canonical_29 is None else self._prev_joint_pos_canonical_29.copy()
        )
        self._body_xy_world = self._body_xy_world + root_xy_delta_world
        body_pos_world = np.array(
            [self._body_xy_world[0], self._body_xy_world[1], root_z],
            dtype=np.float32,
        )
        self._prev_root_quat_wxyz = root_quat_wxyz.copy()
        self._prev_joint_pos_canonical_29 = joint_pos_canonical_29.copy()

        return UnifiedSMPLActionFrame(
            root_xy_delta_world=root_xy_delta_world,
            root_z=root_z,
            body_pos_world=body_pos_world,
            root_orient_rot6d=root_orient_rot6d,
            root_quat_wxyz=root_quat_wxyz,
            joint_pos_canonical_29=joint_pos_canonical_29,
            hand_binary=hand_binary,
            prev_root_quat_wxyz=prev_root_quat,
            prev_joint_pos_canonical_29=prev_joint_pos,
        )
