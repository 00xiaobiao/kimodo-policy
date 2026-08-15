from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from action_provider.vision_video import read_rgb_video_mp4
from action_provider.vla_robot_current_local_runtime_v3 import (
    ACTION_NAMES,
    ROBOT_CURRENT_LOCAL_V3_SCHEMA_VERSION,
    STATE_NAMES,
    VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM,
    VLA_ROBOT_CURRENT_LOCAL_V3_STATE_DIM,
    build_vla_rotlocal_v3_action,
    build_vla_rotlocal_v3_observation_state,
    quat_heading_wxyz,
)
from action_provider.vla_smpl_runtime import (
    quat_conjugate_wxyz,
    quat_from_roll_pitch_yaw_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_to_rot6d_wxyz,
    reorder_twist2_to_canonical_29,
)


VLA_STATE_DIM = VLA_ROBOT_CURRENT_LOCAL_V3_STATE_DIM
VLA_ACTION_DIM = VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM
SCHEMA_VERSION = ROBOT_CURRENT_LOCAL_V3_SCHEMA_VERSION


def decode_scalar(value) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return decode_scalar(value.item())
        if value.size == 1:
            return decode_scalar(value.reshape(-1)[0])
        return str(value.tolist())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def find_npz_files(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.rglob("*.npz") if path.is_file())


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    elif output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_root}. "
            "Use --overwrite to replace it."
        )


def normalize_task(task_value, npz_path: Path, input_root: Path) -> str:
    if task_value is not None:
        task = decode_scalar(task_value).strip()
        if task:
            return task
    try:
        rel_parent = npz_path.parent.relative_to(input_root)
        return rel_parent.as_posix()
    except ValueError:
        return npz_path.parent.as_posix()


def load_vision_rgb_and_indices(
    data: np.lib.npyio.NpzFile,
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if "vision_rgb" in data:
        vision_rgb = np.asarray(data["vision_rgb"], dtype=np.uint8)
    elif "vision_rgb_video_path" in data:
        video_rel = decode_scalar(data["vision_rgb_video_path"])
        video_path = npz_path.parent / video_rel
        vision_rgb = read_rgb_video_mp4(video_path)
        if vision_rgb.size == 0:
            raise ValueError(f"{npz_path} video has no frames: {video_path}")
    else:
        raise KeyError(f"{npz_path} missing vision data: expected 'vision_rgb' or 'vision_rgb_video_path'")

    if "vision_frame_indices" in data:
        frame_indices = np.asarray(data["vision_frame_indices"], dtype=np.int64)
    else:
        frame_indices = np.arange(vision_rgb.shape[0], dtype=np.int64)
    return vision_rgb, frame_indices


def inspect_image_shape(npz_paths: list[Path]) -> tuple[int, int, int]:
    for path in npz_paths:
        with np.load(path, allow_pickle=True) as data:
            vision_rgb, _ = load_vision_rgb_and_indices(data, path)
            if vision_rgb.ndim != 4:
                raise ValueError(f"{path} has unexpected vision_rgb shape: {vision_rgb.shape}")
            if vision_rgb.shape[0] == 0:
                continue
            return tuple(int(v) for v in vision_rgb.shape[1:])
    raise ValueError("No RGB frames found in the input dataset.")


def validate_isaac_time_rows(
    data: np.lib.npyio.NpzFile,
    *,
    num_frames: int,
    npz_path: Path,
) -> None:
    if "episode_step" not in data:
        return
    episode_step = np.asarray(data["episode_step"], dtype=np.int64).reshape(-1)
    if episode_step.shape != (num_frames,):
        raise ValueError(f"{npz_path} has unexpected episode_step shape: {episode_step.shape}")
    if episode_step.size > 1 and not np.all(np.diff(episode_step) == 1):
        raise ValueError(f"{npz_path} episode_step is not contiguous Isaac control time")


def validate_video_frame_indices(
    frame_indices: np.ndarray,
    *,
    num_frames: int,
    npz_path: Path,
    strict: bool,
) -> None:
    frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if np.any(frame_indices < 0) or np.any(frame_indices >= num_frames):
        raise ValueError(f"{npz_path} has out-of-range vision_frame_indices for {num_frames} Isaac rows")
    if strict:
        expected = np.arange(num_frames, dtype=np.int64)
        if frame_indices.shape != expected.shape or not np.array_equal(frame_indices, expected):
            raise ValueError(
                f"{npz_path} vision_frame_indices are not a full Isaac-time 0..N-1 sequence. "
                "Use --allow-missing-video-frames only if you accept dropping Isaac rows without images."
            )


def build_features(image_shape: tuple[int, int, int], use_videos: bool) -> dict[str, dict]:
    return {
        "observation.images.front": {
            "dtype": "video" if use_videos else "image",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": ACTION_NAMES,
        },
    }


def write_v31_protocol_metadata(
    output_root: Path,
    *,
    backend_source: str,
    fps: int,
) -> None:
    info_path = output_root / "meta" / "info.json"
    if not info_path.is_file():
        return
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["vla_protocol"] = {
        "schema": SCHEMA_VERSION,
        "version": "3.1",
        "action_dim": VLA_ACTION_DIM,
        "action_layout": "root_xy_delta_z_rot6d_joints29_hands2",
        "rotation_6d_layout": "row",
        "action_semantics": "reference_pose_not_robot_current_residual",
        "root_xy_delta_frame": "current_reference_base_frame",
        "root_rotation_frame": "episode_reference_frame",
        "fps": float(fps),
        "control_dt": 1.0 / float(fps),
        "dt_source": "system_control_frequency_or_metadata",
        "backend_source": str(backend_source),
        "twist2_reference_source": "mimic_integrated" if backend_source == "twist2" else None,
        "sonic_reference_source": "body_pos_body_quat" if backend_source == "sonic" else None,
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def build_observation_state(
    *,
    initial_robot_orientation: np.ndarray,
    root_orientation: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
) -> np.ndarray:
    state = build_vla_rotlocal_v3_observation_state(
        initial_robot_orientation_wxyz=initial_robot_orientation,
        root_orientation_wxyz=root_orientation,
        joint_pos_canonical_29=joint_pos,
        joint_vel_canonical_29=joint_vel,
    )
    if state.shape != (len(STATE_NAMES),):
        raise ValueError(f"Unexpected observation.state v3 shape: {state.shape}, expected {(len(STATE_NAMES),)}")
    return state


def extract_canonical_state_v3(
    *,
    root_orientation: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
) -> np.ndarray:
    root_orientation = np.asarray(root_orientation, dtype=np.float32)
    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    joint_vel = np.asarray(joint_vel, dtype=np.float32)
    if root_orientation.shape != (joint_pos.shape[0], 4):
        raise ValueError(f"Unexpected robot root orientation shape: {root_orientation.shape}")
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 29:
        raise ValueError(f"Unexpected joint_pos shape: {joint_pos.shape}")
    if joint_vel.shape != joint_pos.shape:
        raise ValueError(f"Unexpected joint_vel shape: {joint_vel.shape}")

    initial_robot_orientation = root_orientation[0]
    return np.stack(
        [
            build_observation_state(
                initial_robot_orientation=initial_robot_orientation,
                root_orientation=root_orientation[i],
                joint_pos=joint_pos[i],
                joint_vel=joint_vel[i],
            )
            for i in range(joint_pos.shape[0])
        ],
        axis=0,
    ).astype(np.float32)


def get_hand_binary_arrays(data: np.lib.npyio.NpzFile, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    left = None
    right = None
    for key in ("vla_action_raw", "vla_action_hand_binary", "vla_action_hand_binary_2", "vla_action"):
        if key not in data:
            continue
        raw = np.asarray(data[key], dtype=np.float32)
        if raw.ndim == 2 and raw.shape == (num_frames, VLA_ACTION_DIM):
            left = raw[:, 38]
            right = raw[:, 39]
            break
        if raw.ndim == 2 and raw.shape == (num_frames, 2):
            left = raw[:, 0]
            right = raw[:, 1]
            break
    if left is None and "pico_left_grip_binary" in data:
        left = np.asarray(data["pico_left_grip_binary"], dtype=np.float32)
    if right is None and "pico_right_grip_binary" in data:
        right = np.asarray(data["pico_right_grip_binary"], dtype=np.float32)
    if left is None:
        left = np.zeros(num_frames, dtype=np.float32)
    if right is None:
        right = np.zeros(num_frames, dtype=np.float32)
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.shape != (num_frames,) or right.shape != (num_frames,):
        raise ValueError(f"Unexpected hand binary shapes: left={left.shape} right={right.shape}")
    return left, right


def _find_action_mimic_slice(data: np.lib.npyio.NpzFile) -> tuple[int, int]:
    default_slice = (0, 35)
    if "observation_semantics" not in data:
        return default_slice
    try:
        semantics = json.loads(str(data["observation_semantics"]))
        action_mimic = semantics["structure"]["obs_full"]["components"]["action_mimic"]["dims"]
        start, end = int(action_mimic[0]), int(action_mimic[1])
        if end - start == 35:
            return start, end
    except Exception:
        pass
    return default_slice


def extract_twist2_action_mimic(data: np.lib.npyio.NpzFile, num_frames: int) -> np.ndarray:
    if "robot_action_mimic" in data:
        mimic = np.asarray(data["robot_action_mimic"], dtype=np.float32)
        if mimic.shape == (num_frames, 35):
            return mimic
    obs_buf = np.asarray(data["robot_obs_buf"], dtype=np.float32)
    start, end = _find_action_mimic_slice(data)
    mimic = obs_buf[:, start:end]
    if mimic.shape != (num_frames, 35):
        raise ValueError(f"Unexpected TWIST2 action_mimic shape: {mimic.shape}")
    return mimic.astype(np.float32)


def infer_sonic_fps(npz_paths: list[Path], fps_override: int | None) -> int:
    if fps_override is not None:
        return fps_override
    freqs = []
    for path in npz_paths[: min(len(npz_paths), 32)]:
        with np.load(path, allow_pickle=True) as data:
            if "meta_control_dt" not in data:
                continue
            control_dt = float(np.asarray(data["meta_control_dt"]).reshape(-1)[0])
            if control_dt > 0:
                freqs.append(1.0 / control_dt)
    if not freqs:
        raise ValueError("Could not infer SONIC fps from meta_control_dt.")
    fps = int(round(float(np.median(freqs))))
    if fps <= 0:
        raise ValueError(f"Invalid inferred fps: {fps}")
    return fps


def infer_twist2_fps(npz_paths: list[Path], fps_override: int | None) -> int:
    if fps_override is not None:
        return fps_override
    freqs = []
    for path in npz_paths[: min(len(npz_paths), 32)]:
        with np.load(path, allow_pickle=True) as data:
            if "system_control_frequency" not in data:
                continue
            raw = np.asarray(data["system_control_frequency"], dtype=np.float32).reshape(-1)
            if raw.size > 0:
                freqs.append(float(np.median(raw)))
    if not freqs:
        raise ValueError("Could not infer TWIST2 fps from system_control_frequency.")
    fps = int(round(float(np.median(freqs))))
    if fps <= 0:
        raise ValueError(f"Invalid inferred fps: {fps}")
    return fps


def infer_twist2_control_dt(data: np.lib.npyio.NpzFile, fps: int) -> float:
    if "system_control_frequency" in data:
        freq_values = np.asarray(data["system_control_frequency"], dtype=np.float32).reshape(-1)
        if freq_values.size > 0 and float(np.median(freq_values)) > 0.0:
            return 1.0 / float(np.median(freq_values))
    return 1.0 / float(fps)


def align_source_heading_to_robot_wxyz(
    initial_robot_quat_wxyz: np.ndarray,
    initial_source_quat_wxyz: np.ndarray,
) -> np.ndarray:
    return quat_mul_wxyz(
        quat_heading_wxyz(initial_robot_quat_wxyz),
        quat_conjugate_wxyz(quat_heading_wxyz(initial_source_quat_wxyz)),
    )


def rotate_world_vector_xy_wxyz(quat_wxyz: np.ndarray, xy: np.ndarray) -> np.ndarray:
    vec = np.zeros((3,), dtype=np.float32)
    vec[:2] = np.asarray(xy, dtype=np.float32).reshape(2)
    quat_xyzw = quat_normalize_wxyz(np.asarray(quat_wxyz, dtype=np.float32).reshape(4))[[1, 2, 3, 0]]
    return R.from_quat(quat_xyzw).apply(vec)[:2].astype(np.float32)


def rotate_world_vector_wxyz(quat_wxyz: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    quat_xyzw = quat_normalize_wxyz(np.asarray(quat_wxyz, dtype=np.float32).reshape(4))[[1, 2, 3, 0]]
    return R.from_quat(quat_xyzw).apply(np.asarray(xyz, dtype=np.float32).reshape(3)).astype(np.float32)


def rotate_world_delta_to_target_base_local_xy(
    target_root_quat_wxyz: np.ndarray,
    world_delta_xyz: np.ndarray,
) -> np.ndarray:
    quat_xyzw = quat_normalize_wxyz(np.asarray(target_root_quat_wxyz, dtype=np.float32).reshape(4))[[1, 2, 3, 0]]
    return R.from_quat(quat_xyzw).inv().apply(np.asarray(world_delta_xyz, dtype=np.float32).reshape(3))[:2].astype(np.float32)


def _load_sonic_body_fields(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    if "human_body_pos" in data:
        body_pos = np.asarray(data["human_body_pos"], dtype=np.float32)
    elif "robot_root_position" in data:
        body_pos = np.asarray(data["robot_root_position"], dtype=np.float32)
    else:
        raise KeyError("SONIC rotlocal v3 requires human_body_pos or robot_root_position")

    if "human_body_quat_w" in data:
        body_quat = np.asarray(data["human_body_quat_w"], dtype=np.float32)
    elif "robot_root_orientation" in data:
        body_quat = np.asarray(data["robot_root_orientation"], dtype=np.float32)
    else:
        raise KeyError("SONIC rotlocal v3 requires human_body_quat_w or robot_root_orientation")
    return body_pos, body_quat


def _load_sonic_joint_targets(data: np.lib.npyio.NpzFile, num_frames: int) -> np.ndarray:
    if "human_joint_pos" in data:
        return np.asarray(data["human_joint_pos"], dtype=np.float32)
    if "vla_action_joint_pos_29" in data:
        return np.asarray(data["vla_action_joint_pos_29"], dtype=np.float32)
    if "vla_action_raw" in data:
        action_raw = np.asarray(data["vla_action_raw"], dtype=np.float32)
        if action_raw.shape == (num_frames, VLA_ACTION_DIM):
            return action_raw[:, 9:38]
        raise ValueError(f"Unexpected vla_action_raw shape: {action_raw.shape}")
    raise KeyError("SONIC rotlocal v3 requires human_joint_pos, vla_action_joint_pos_29, or vla_action_raw")


def build_sonic_rotlocal_v3_actions(
    data: np.lib.npyio.NpzFile,
    *,
    num_frames: int,
    robot_root_orientation: np.ndarray,
) -> np.ndarray:
    body_pos, body_quat = _load_sonic_body_fields(data)
    joint_targets = _load_sonic_joint_targets(data, num_frames)

    robot_root_orientation = np.asarray(robot_root_orientation, dtype=np.float32)
    if robot_root_orientation.shape != (num_frames, 4):
        raise ValueError(f"Unexpected robot root orientation shape: {robot_root_orientation.shape}")
    if body_pos.shape != (num_frames, 3):
        raise ValueError(f"Unexpected SONIC body_pos shape: {body_pos.shape}")
    if body_quat.shape != (num_frames, 4):
        raise ValueError(f"Unexpected SONIC body_quat shape: {body_quat.shape}")
    if joint_targets.shape != (num_frames, 29):
        raise ValueError(f"Unexpected SONIC joint target shape: {joint_targets.shape}")

    left_binary, right_binary = get_hand_binary_arrays(data, num_frames)
    source_heading_anchor_inv = quat_conjugate_wxyz(quat_heading_wxyz(body_quat[0]))

    actions = np.zeros((num_frames, VLA_ACTION_DIM), dtype=np.float32)
    for i in range(num_frames):
        target_quat = quat_mul_wxyz(source_heading_anchor_inv, body_quat[i])
        if i == 0:
            root_ref_base_local_xy_delta = np.zeros((2,), dtype=np.float32)
        else:
            source_world_delta = (body_pos[i] - body_pos[i - 1]).astype(np.float32)
            target_world_delta = rotate_world_vector_wxyz(source_heading_anchor_inv, source_world_delta)
            root_ref_base_local_xy_delta = rotate_world_delta_to_target_base_local_xy(target_quat, target_world_delta)
        actions[i] = build_vla_rotlocal_v3_action(
            root_ref_base_local_xy_delta=root_ref_base_local_xy_delta,
            root_z=body_pos[i, 2],
            root_ref_rot6d=quat_to_rot6d_wxyz(target_quat).reshape(6),
            joint_pos_canonical_29=joint_targets[i],
            hand_binary=np.array([left_binary[i], right_binary[i]], dtype=np.float32),
        )
    return actions


def build_twist2_rotlocal_v3_actions(
    data: np.lib.npyio.NpzFile,
    *,
    num_frames: int,
    control_dt: float,
    robot_root_orientation: np.ndarray,
) -> np.ndarray:
    mimic = extract_twist2_action_mimic(data, num_frames)
    robot_root_orientation = np.asarray(robot_root_orientation, dtype=np.float32)
    if robot_root_orientation.shape != (num_frames, 4):
        raise ValueError(f"Unexpected robot root orientation shape: {robot_root_orientation.shape}")

    left_binary, right_binary = get_hand_binary_arrays(data, num_frames)
    actions = np.zeros((num_frames, VLA_ACTION_DIM), dtype=np.float32)
    yaw_target = 0.0
    for i in range(num_frames):
        xy_vel_local = mimic[i, 0:2]
        root_z = float(mimic[i, 2])
        roll = float(mimic[i, 3])
        pitch = float(mimic[i, 4])
        yaw_vel = float(mimic[i, 5])
        if i > 0:
            yaw_target = ((yaw_target + yaw_vel * float(control_dt) + np.pi) % (2.0 * np.pi)) - np.pi
        target_quat = quat_from_roll_pitch_yaw_wxyz(roll=roll, pitch=pitch, yaw=yaw_target)
        root_ref_base_local_xy_delta = (
            np.zeros((2,), dtype=np.float32)
            if i == 0
            else xy_vel_local.astype(np.float32, copy=False) * float(control_dt)
        )
        actions[i] = build_vla_rotlocal_v3_action(
            root_ref_base_local_xy_delta=root_ref_base_local_xy_delta,
            root_z=root_z,
            root_ref_rot6d=quat_to_rot6d_wxyz(target_quat).reshape(6),
            joint_pos_canonical_29=reorder_twist2_to_canonical_29(mimic[i, 6:35]),
            hand_binary=np.array([left_binary[i], right_binary[i]], dtype=np.float32),
        )
    return actions
