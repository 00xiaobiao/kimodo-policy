#!/usr/bin/env python3
"""
Replay SONIC POSE (Protocol v3) from a recorded TWIST2 debug .npz.

This publishes a ZMQ "pose" topic compatible with `SonicActionProvider`:
  - smpl_joints      (N, 24, 3) float32
  - smpl_pose        (N, 21, 3) float32   (axis-angle, excludes Pelvis + 2 hands)
  - body_quat_w      (N, 4)     float32   (qw,qx,qy,qz), uses Pelvis quat
  - joint_pos        (N, 29)    float32   (prefer recorded robot_twist2_inference_qpos when available)
  - joint_vel        (N, 29)    float32   (from recorded robot_qvel_before_decimation when available)
  - left_hand_joints (N, 7)     float32   (if available)
  - right_hand_joints(N, 7)     float32   (if available)
  - frame_index      (N,)       int64

Typical usage:
  python tools/sonic_pose_npz_replay_server.py --npz xxx.npz --port 5556 --fps 30
Then in another terminal:
  bash run_sonic.sh --encoder ... --decoder ...   (and set SONIC_ZMQ_PORT=5556)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from pico_server.sonic_tools.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

try:
    import zmq
except ImportError as e:
    raise SystemExit(f"Missing dependency: pyzmq. ({e})")

def _quat_to_axis_angle_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to axis-angle (x,y,z)."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected (...,4) quaternion, got {q.shape}")

    # Normalize
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.clip(norm, 1e-12, None)

    w = np.clip(q[..., 0], -1.0, 1.0)
    v = q[..., 1:4]
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(1.0 - w * w, 0.0))

    # When s is tiny, direction doesn't matter much. Use v as-is.
    axis = np.where(s[..., None] < 1e-8, v, v / s[..., None])
    aa = axis * angle[..., None]
    return aa.astype(np.float32)


def _quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected (...,4) quaternion, got {q.shape}")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return (q / np.clip(norm, 1e-12, None)).astype(np.float32)


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1).astype(np.float32)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _quat_normalize_wxyz(a).astype(np.float64)
    b = _quat_normalize_wxyz(b).astype(np.float64)

    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)

    out = np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )
    return _quat_normalize_wxyz(out)


def _quat_angle_deg_wxyz(q: np.ndarray) -> float:
    q = _quat_normalize_wxyz(q)
    w = float(np.clip(q[..., 0], -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(abs(w))))


def _quat_to_rotation_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected (...,4) quaternion, got {q.shape}")

    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.clip(norm, 1e-12, None)

    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot = np.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return rot.astype(np.float32)


def _make_root_local_joints(joints_world: np.ndarray, root_quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert world-space joints to pelvis-centered, root-local joints."""
    joints_world = np.asarray(joints_world, dtype=np.float32)
    if joints_world.shape != (24, 3):
        raise ValueError(f"Expected (24,3) joints, got {joints_world.shape}")

    root_pos = joints_world[0:1]
    joints_centered = joints_world - root_pos

    root_rot = _quat_to_rotation_matrix_wxyz(np.asarray(root_quat_wxyz, dtype=np.float32))
    root_rot_inv = np.swapaxes(root_rot, -1, -2)
    joints_local = joints_centered @ root_rot_inv
    return joints_local.astype(np.float32)


def _process_root_quat_for_sonic(raw_root_quat_wxyz: np.ndarray) -> np.ndarray:
    """Match SONIC's official root processing: Y-up→Z-up, then remove SMPL base rot."""
    raw_root_quat_wxyz = _quat_normalize_wxyz(raw_root_quat_wxyz)

    y_to_z_quat_wxyz = np.array(
        [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],
        dtype=np.float32,
    )
    smpl_base_rot_conj_wxyz = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)

    root_quat_z_up = _quat_mul_wxyz(y_to_z_quat_wxyz, raw_root_quat_wxyz)
    root_quat_sonic = _quat_mul_wxyz(root_quat_z_up, smpl_base_rot_conj_wxyz)
    return _quat_normalize_wxyz(root_quat_sonic)


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

# 21 = 24 - Pelvis - 2 hands
SMPL_POSE_ORDER_21 = [n for n in SMPL_JOINT_ORDER_24 if n not in ("Pelvis", "Left_Hand", "Right_Hand")]


def _load_human_smplx_frames(npz: dict) -> list[dict]:
    if "human_smplx_data" not in npz:
        raise KeyError("NPZ missing 'human_smplx_data'")
    raw = npz["human_smplx_data"]
    # stored as a single JSON string
    s = str(raw.item()) if hasattr(raw, "item") else str(raw)
    frames = json.loads(s)
    if not isinstance(frames, list) or not frames:
        raise ValueError("human_smplx_data is not a non-empty list")
    return frames


def _coerce_joint_frame(
    value: np.ndarray | None,
    *,
    field_name: str,
) -> np.ndarray:
    if value is None:
        return np.zeros((1, 29), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 29:
        raise ValueError(f"{field_name} must have 29 values, got shape {arr.shape}")
    return arr.reshape(1, 29)


def _frame_to_pose_fields(
    frame: dict,
    frame_index: int,
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
    joint_pos: np.ndarray | None,
    joint_vel: np.ndarray | None,
) -> dict[str, np.ndarray]:
    # Build raw world-space joints first from the recorded frame.
    joints_world = np.zeros((24, 3), dtype=np.float32)
    for i, name in enumerate(SMPL_JOINT_ORDER_24):
        # each entry: [pos3, quat4]
        pos3 = frame[name][0]
        # N*24*3(pos3)
        joints_world[i, :] = np.asarray(pos3, dtype=np.float32)

    # body_quat_w: official SONIC semantics use processed global root orientation,
    # not the raw stored pelvis quaternion directly.
    # 跟关节的四元数
    body_quat_raw = np.asarray(frame["Pelvis"][1], dtype=np.float32)
    body_quat_proc = _process_root_quat_for_sonic(body_quat_raw)
    body_quat_w = body_quat_proc.reshape(1, 4)

    # smpl_joints: (1,24,3) must follow SONIC's expected root-local semantics.
    joints = _make_root_local_joints(joints_world, body_quat_proc).reshape(1, 24, 3)

    # smpl_pose: (1,21,3) axis-angle from quat
    pose = np.zeros((1, 21, 3), dtype=np.float32)
    for i, name in enumerate(SMPL_POSE_ORDER_21):
        quat = np.asarray(frame[name][1], dtype=np.float32)
        pose[0, i, :] = _quat_to_axis_angle_wxyz(quat)

    fields: dict[str, np.ndarray] = {
        "smpl_joints": joints,
        "smpl_pose": pose,
        "body_quat_w": body_quat_w,
        "joint_pos": _coerce_joint_frame(joint_pos, field_name="joint_pos"),
        "joint_vel": _coerce_joint_frame(joint_vel, field_name="joint_vel"),
        "frame_index": np.asarray([frame_index], dtype=np.int64),
    }

    if left_hand is not None:
        fields["left_hand_joints"] = left_hand.reshape(1, -1).astype(np.float32)
    if right_hand is not None:
        fields["right_hand_joints"] = right_hand.reshape(1, -1).astype(np.float32)
    if frame_index < 3:
        delta_quat = _quat_mul_wxyz(_quat_conjugate_wxyz(body_quat_raw), body_quat_proc)
        print(
            "[REPLAY][ROOT_QUAT] "
            f"frame={frame_index} "
            f"raw={np.array2string(body_quat_raw, precision=5, separator=', ')} "
            f"proc={np.array2string(body_quat_proc, precision=5, separator=', ')} "
            f"delta_angle_deg={_quat_angle_deg_wxyz(delta_quat):.2f}",
            flush=True,
        )
    return fields


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(
        description="Replay SONIC POSE (Protocol v3) from recorded .npz (same CLI style as pico_server_pose_only.py)"
    )
    ap.add_argument("--npz", required=True, type=str, help="Recorded .npz containing human_smplx_data")
    ap.add_argument("--host", default="127.0.0.1", type=str, help="Bind host (default: 127.0.0.1)")
    ap.add_argument("--port", default=5556, type=int, help="ZMQ server port (default: 5556)")
    ap.add_argument("--fps", default=30.0, type=float, help="Publish FPS (default: 30)")
    ap.add_argument("--loop", action="store_true", help="Loop frames")
    ap.add_argument(
        "--vis_vr3pt",
        action="store_true",
        help="Enable VR 3-point pose visualization (same as pico_server_pose_only.py)",
    )
    ap.add_argument(
        "--vis_smpl",
        action="store_true",
        help="Enable SMPL body joint visualization (same as pico_server_pose_only.py)",
    )
    args = ap.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    d = np.load(npz_path, allow_pickle=True)
    frames = _load_human_smplx_frames(d)

    left_hand = d["human_hand_left"] if "human_hand_left" in d else None
    right_hand = d["human_hand_right"] if "human_hand_right" in d else None
    robot_qpos_target = d["robot_twist2_inference_qpos"] if "robot_twist2_inference_qpos" in d else None
    robot_qpos_actual = d["robot_qpos_before_decimation"] if "robot_qpos_before_decimation" in d else None
    robot_qvel = d["robot_qvel_before_decimation"] if "robot_qvel_before_decimation" in d else None
    robot_qpos = robot_qpos_target if robot_qpos_target is not None else robot_qpos_actual

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print("=" * 60)
    print("SONIC Pose Replay Server (from .npz)")
    print("=" * 60)
    print(f"Port: {args.port}")
    print(f"NPZ: {npz_path.name}  frames={len(frames)}  fps={args.fps}  loop={args.loop}")
    print(f"VR 3pt visualization: {args.vis_vr3pt}")
    print(f"SMPL visualization: {args.vis_smpl}")
    print(f"Publishing tcp://{args.host}:{args.port} (topic='pose', protocol v3)")
    if robot_qpos is not None:
        qpos = np.asarray(robot_qpos, dtype=np.float32)
        print(
            f"Recorded joint_pos: shape={qpos.shape} "
            f"range=[{qpos.min():.4f}, {qpos.max():.4f}]"
        )
        if robot_qpos_target is not None:
            print("Recorded joint_pos source: robot_twist2_inference_qpos")
        else:
            print("Recorded joint_pos source: robot_qpos_before_decimation")
    else:
        print("Recorded joint_pos: unavailable, fallback to zeros")
    if robot_qvel is not None:
        qvel = np.asarray(robot_qvel, dtype=np.float32)
        print(
            f"Recorded joint_vel: shape={qvel.shape} "
            f"range=[{qvel.min():.4f}, {qvel.max():.4f}]"
        )
    else:
        print("Recorded joint_vel: unavailable, fallback to zeros")
    print("=" * 60)

    dt = 1.0 / float(args.fps)
    i = 0
    global_frame = 0
    try:
        while True:
            if i >= len(frames):
                if args.loop:
                    i = 0
                else:
                    break

            lh = left_hand[i] if left_hand is not None and i < len(left_hand) else None
            rh = right_hand[i] if right_hand is not None and i < len(right_hand) else None
            jp = robot_qpos[i] if robot_qpos is not None and i < len(robot_qpos) else None
            jv = robot_qvel[i] if robot_qvel is not None and i < len(robot_qvel) else None

            fields = _frame_to_pose_fields(
                frames[i],
                frame_index=global_frame,
                left_hand=lh,
                right_hand=rh,
                joint_pos=jp,
                joint_vel=jv,
            )
            msg = pack_pose_message(fields, topic="pose", version=3)
            sock.send(msg)

            time.sleep(dt)
            i += 1
            global_frame += 1
    except KeyboardInterrupt:
        pass
    finally:
        sock.close(0)
        ctx.term()

    print("[sonic_pose_npz_replay_server] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
