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
from action_provider.vla_smpl_runtime import (
    CANONICAL_G1_JOINT_NAMES_29,
    VLA_SMPL_ACTION_DIM,
    VLA_SMPL_STATE_DIM,
    build_vla_action,
    build_vla_observation_state,
    quat_from_roll_pitch_yaw_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_to_rot6d_wxyz,
    reorder_twist2_to_canonical_29,
    rot6d_to_quat_wxyz,
    yaw_from_quat_wxyz,
)


VLA_STATE_DIM = VLA_SMPL_STATE_DIM
VLA_ACTION_DIM = VLA_SMPL_ACTION_DIM
CANONICAL_JOINT_NAMES_29 = CANONICAL_G1_JOINT_NAMES_29

STATE_NAMES = [
    *[f"state.root_rot6d.{idx}" for idx in range(6)],
    *[f"state.dof_pos.{name}" for name in CANONICAL_JOINT_NAMES_29],
    *[f"state.dof_vel.{name}" for name in CANONICAL_JOINT_NAMES_29],
]

ACTION_NAMES = [
    "action.root_xy_delta.x",
    "action.root_xy_delta.y",
    "action.root_z",
    *[f"action.root_rot6d.{idx}" for idx in range(6)],
    *[f"action.joint_pos.{name}" for name in CANONICAL_JOINT_NAMES_29],
    "action.hand_binary.left",
    "action.hand_binary.right",
]


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


def build_observation_state(
    *,
    root_orientation: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
) -> np.ndarray:
    state = build_vla_observation_state(
        root_orientation_wxyz=root_orientation,
        joint_pos_canonical_29=joint_pos,
        joint_vel_canonical_29=joint_vel,
    )
    if state.shape != (len(STATE_NAMES),):
        raise ValueError(f"Unexpected observation.state shape: {state.shape}, expected {(len(STATE_NAMES),)}")
    return state


def build_action(
    *,
    root_xy_delta: np.ndarray,
    root_z: np.ndarray | float,
    root_rot6d: np.ndarray,
    joint_pos: np.ndarray,
    hand_binary: np.ndarray,
) -> np.ndarray:
    action = build_vla_action(
        root_xy_delta_world=root_xy_delta,
        root_z=root_z,
        root_rot6d=root_rot6d,
        joint_pos_canonical_29=joint_pos,
        hand_binary=hand_binary,
    )
    if action.shape != (len(ACTION_NAMES),):
        raise ValueError(f"Unexpected action shape: {action.shape}, expected {(len(ACTION_NAMES),)}")
    return action


def _resolve_sonic_heading_align_quat(
    data: np.lib.npyio.NpzFile,
    *,
    num_frames: int,
) -> np.ndarray | None:
    if "anchor_heading_align_quat_wxyz" in data:
        align_quat = np.asarray(data["anchor_heading_align_quat_wxyz"], dtype=np.float32)
        if align_quat.shape == (4,):
            align_quat = np.repeat(align_quat.reshape(1, 4), num_frames, axis=0)
        if align_quat.shape != (num_frames, 4):
            raise ValueError(
                "Unexpected anchor_heading_align_quat_wxyz shape: "
                f"{align_quat.shape}, expected {(num_frames, 4)}"
            )
        align_quat = quat_normalize_wxyz(align_quat)
        if "anchor_use_heading_align" in data:
            use_align = np.asarray(data["anchor_use_heading_align"]).reshape(-1)
            if use_align.size == 1:
                use_align = np.repeat(use_align, num_frames)
            if use_align.shape != (num_frames,):
                raise ValueError(
                    f"Unexpected anchor_use_heading_align shape: {use_align.shape}, expected {(num_frames,)}"
                )
            identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            align_quat = align_quat.copy()
            align_quat[~use_align.astype(bool)] = identity
        return align_quat.astype(np.float32)

    if "human_body_quat_w" in data and "robot_root_orientation" in data:
        ref_quat = np.asarray(data["human_body_quat_w"], dtype=np.float32)
        base_quat = np.asarray(data["robot_root_orientation"], dtype=np.float32)
        if ref_quat.shape != (num_frames, 4) or base_quat.shape != (num_frames, 4):
            raise ValueError(
                "Unexpected quaternion shapes for fallback heading align: "
                f"human_body_quat_w={ref_quat.shape} robot_root_orientation={base_quat.shape}"
            )
        yaw_delta = yaw_from_quat_wxyz(base_quat[0]) - yaw_from_quat_wxyz(ref_quat[0])
        align0 = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=yaw_delta)
        return np.repeat(align0.reshape(1, 4), num_frames, axis=0).astype(np.float32)
    return None


def _apply_heading_align_to_action(
    action: np.ndarray,
    *,
    align_quat_wxyz: np.ndarray,
) -> np.ndarray:
    if action.shape[1] != VLA_ACTION_DIM:
        raise ValueError(f"Unexpected action shape {action.shape}, expected (_, {VLA_ACTION_DIM})")
    if align_quat_wxyz.shape != (action.shape[0], 4):
        raise ValueError(
            f"Unexpected align quaternion shape {align_quat_wxyz.shape}, expected {(action.shape[0], 4)}"
        )

    out = np.asarray(action, dtype=np.float32).copy()
    align_quat_wxyz = quat_normalize_wxyz(align_quat_wxyz)

    root_quat = rot6d_to_quat_wxyz(out[:, 3:9])
    aligned_root_quat = quat_mul_wxyz(align_quat_wxyz, root_quat)
    out[:, 3:9] = quat_to_rot6d_wxyz(aligned_root_quat)

    rot_mats = R.from_quat(align_quat_wxyz[:, [1, 2, 3, 0]]).as_matrix().astype(np.float32)
    xy_delta = out[:, :2]
    xy_delta_3d = np.concatenate([xy_delta, np.zeros((xy_delta.shape[0], 1), dtype=np.float32)], axis=1)
    rotated_xy_delta = np.einsum("nij,nj->ni", rot_mats, xy_delta_3d, optimize=True)[:, :2]
    out[:, :2] = rotated_xy_delta.astype(np.float32)
    return out.astype(np.float32)


def get_hand_binary_arrays(data: np.lib.npyio.NpzFile, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    left = None
    right = None
    if "vla_action" in data:
        hand_binary = np.asarray(data["vla_action"], dtype=np.float32)
        if hand_binary.ndim == 2 and hand_binary.shape == (num_frames, VLA_ACTION_DIM):
            left = hand_binary[:, 38]
            right = hand_binary[:, 39]
    if left is None and "vla_action_hand_binary_2" in data:
        hand_binary = np.asarray(data["vla_action_hand_binary_2"], dtype=np.float32)
        if hand_binary.ndim == 2 and hand_binary.shape == (num_frames, 2):
            left = hand_binary[:, 0]
            right = hand_binary[:, 1]
    if left is None and "vla_action_hand_binary" in data:
        hand_binary = np.asarray(data["vla_action_hand_binary"], dtype=np.float32)
        if hand_binary.ndim == 2 and hand_binary.shape == (num_frames, 2):
            left = hand_binary[:, 0]
            right = hand_binary[:, 1]
    if left is None and "pico_left_grip_binary" in data:
        left = np.asarray(data["pico_left_grip_binary"], dtype=np.float32)
    if right is None and "pico_right_grip_binary" in data:
        right = np.asarray(data["pico_right_grip_binary"], dtype=np.float32)
    if left is None:
        left = np.zeros(num_frames, dtype=np.float32)
    if right is None:
        right = np.zeros(num_frames, dtype=np.float32)
    if left.shape != (num_frames,) or right.shape != (num_frames,):
        raise ValueError(f"Unexpected hand binary shapes: left={left.shape} right={right.shape}")
    return left, right


def extract_canonical_state(
    *,
    data: np.lib.npyio.NpzFile,
    root_orientation: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
) -> np.ndarray:
    if "vla_state" in data:
        state = np.asarray(data["vla_state"], dtype=np.float32)
        if state.ndim == 2 and state.shape[1] == VLA_STATE_DIM:
            return state
    if (
        "vla_state_root_rot6d" in data
        and "vla_state_dof_pos_29" in data
        and "vla_state_dof_vel_29" in data
    ):
        root_rot6d = np.asarray(data["vla_state_root_rot6d"], dtype=np.float32)
        dof_pos = np.asarray(data["vla_state_dof_pos_29"], dtype=np.float32)
        dof_vel = np.asarray(data["vla_state_dof_vel_29"], dtype=np.float32)
        if (
            root_rot6d.shape == (joint_pos.shape[0], 6)
            and dof_pos.shape == (joint_pos.shape[0], 29)
            and dof_vel.shape == (joint_pos.shape[0], 29)
        ):
            return np.concatenate([root_rot6d, dof_pos, dof_vel], axis=1).astype(np.float32)
    return np.stack(
        [
            build_observation_state(
                root_orientation=root_orientation[i],
                joint_pos=joint_pos[i],
                joint_vel=joint_vel[i],
            )
            for i in range(joint_pos.shape[0])
        ],
        axis=0,
    ).astype(np.float32)


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


def build_twist2_actions_from_recording(
    data: np.lib.npyio.NpzFile,
    *,
    num_frames: int,
    control_dt: float,
) -> np.ndarray:
    if "vla_action" in data:
        action = np.asarray(data["vla_action"], dtype=np.float32)
        if action.ndim == 2 and action.shape[1] == VLA_ACTION_DIM:
            return action
    if (
        "vla_action_root_xy_delta" in data
        and "vla_action_root_z" in data
        and "vla_action_root_rot6d" in data
        and "vla_action_joint_pos_29" in data
    ):
        root_xy_delta = np.asarray(data["vla_action_root_xy_delta"], dtype=np.float32)
        root_z = np.asarray(data["vla_action_root_z"], dtype=np.float32)
        root_rot6d = np.asarray(data["vla_action_root_rot6d"], dtype=np.float32)
        joint_pos = np.asarray(data["vla_action_joint_pos_29"], dtype=np.float32)
        if (
            root_xy_delta.shape == (num_frames, 2)
            and root_z.shape == (num_frames, 1)
            and root_rot6d.shape == (num_frames, 6)
            and joint_pos.shape == (num_frames, 29)
        ):
            left_binary, right_binary = get_hand_binary_arrays(data, num_frames)
            return np.concatenate(
                [
                    root_xy_delta,
                    root_z,
                    root_rot6d,
                    joint_pos,
                    np.stack([left_binary, right_binary], axis=1).astype(np.float32),
                ],
                axis=1,
            ).astype(np.float32)

    mimic = extract_twist2_action_mimic(data, num_frames)
    left_binary, right_binary = get_hand_binary_arrays(data, num_frames)

    actions = np.zeros((num_frames, VLA_ACTION_DIM), dtype=np.float32)
    yaw_world = 0.0
    for i in range(num_frames):
        xy_vel_local = mimic[i, 0:2]
        root_z = float(mimic[i, 2])
        roll = float(mimic[i, 3])
        pitch = float(mimic[i, 4])
        yaw_vel = float(mimic[i, 5])
        yaw_world = ((yaw_world + yaw_vel * float(control_dt) + np.pi) % (2.0 * np.pi)) - np.pi
        root_quat_wxyz = quat_from_roll_pitch_yaw_wxyz(roll=roll, pitch=pitch, yaw=yaw_world)
        local_delta = np.array(
            [xy_vel_local[0] * float(control_dt), xy_vel_local[1] * float(control_dt), 0.0],
            dtype=np.float32,
        )
        root_xy_delta = R.from_quat(root_quat_wxyz[[1, 2, 3, 0]]).apply(local_delta)[:2].astype(np.float32)
        actions[i] = build_action(
            root_xy_delta=root_xy_delta,
            root_z=root_z,
            root_rot6d=quat_to_rot6d_wxyz(root_quat_wxyz).reshape(6),
            joint_pos=reorder_twist2_to_canonical_29(mimic[i, 6:35]),
            hand_binary=np.array([left_binary[i], right_binary[i]], dtype=np.float32),
        )
    return actions


def _sonic_vla_action_is_heading_aligned(data: np.lib.npyio.NpzFile) -> bool:
    if "vla_action_heading_aligned" not in data:
        return False
    raw = np.asarray(data["vla_action_heading_aligned"]).reshape(-1)
    if raw.size == 0:
        return False
    return bool(raw[0])


def build_sonic_actions_from_recording(
    data: np.lib.npyio.NpzFile,
    *,
    num_frames: int,
    align_heading_targets: bool = False,
) -> np.ndarray:
    if "vla_action" in data:
        action = np.asarray(data["vla_action"], dtype=np.float32)
        if action.ndim == 2 and action.shape[1] == VLA_ACTION_DIM:
            if align_heading_targets and (not _sonic_vla_action_is_heading_aligned(data)):
                align_quat = _resolve_sonic_heading_align_quat(data, num_frames=num_frames)
                if align_quat is not None:
                    return _apply_heading_align_to_action(action, align_quat_wxyz=align_quat)
            return action
    if (
        "vla_action_root_xy_delta" in data
        and "vla_action_root_z" in data
        and "vla_action_root_rot6d" in data
        and "vla_action_joint_pos_29" in data
    ):
        root_xy_delta = np.asarray(data["vla_action_root_xy_delta"], dtype=np.float32)
        root_z = np.asarray(data["vla_action_root_z"], dtype=np.float32)
        root_rot6d = np.asarray(data["vla_action_root_rot6d"], dtype=np.float32)
        joint_pos = np.asarray(data["vla_action_joint_pos_29"], dtype=np.float32)
        if (
            root_xy_delta.shape == (num_frames, 2)
            and root_z.shape == (num_frames, 1)
            and root_rot6d.shape == (num_frames, 6)
            and joint_pos.shape == (num_frames, 29)
        ):
            left_binary, right_binary = get_hand_binary_arrays(data, num_frames)
            action = np.concatenate(
                [
                    root_xy_delta,
                    root_z,
                    root_rot6d,
                    joint_pos,
                    np.stack([left_binary, right_binary], axis=1).astype(np.float32),
                ],
                axis=1,
            ).astype(np.float32)
            if align_heading_targets and (not _sonic_vla_action_is_heading_aligned(data)):
                align_quat = _resolve_sonic_heading_align_quat(data, num_frames=num_frames)
                if align_quat is not None:
                    return _apply_heading_align_to_action(action, align_quat_wxyz=align_quat)
            return action

    if "vla_action_joint_pos_29" in data:
        joint_targets = np.asarray(data["vla_action_joint_pos_29"], dtype=np.float32)
    elif "final_body_action_29dof" in data:
        joint_targets = np.asarray(data["final_body_action_29dof"], dtype=np.float32)
    elif "decoder_target_action" in data:
        joint_targets = np.asarray(data["decoder_target_action"], dtype=np.float32)
    else:
        joint_targets = np.asarray(data["robot_qpos_before_decimation"], dtype=np.float32)

    if "human_body_quat_w" in data:
        body_quat = np.asarray(data["human_body_quat_w"], dtype=np.float32)
    else:
        body_quat = np.asarray(data["robot_root_orientation"], dtype=np.float32)

    if "robot_root_position" in data:
        body_pos = np.asarray(data["robot_root_position"], dtype=np.float32)
    else:
        body_pos = np.zeros((num_frames, 3), dtype=np.float32)

    left_binary, right_binary = get_hand_binary_arrays(data, num_frames)

    actions = np.zeros((num_frames, VLA_ACTION_DIM), dtype=np.float32)
    prev_body_pos = None
    for i in range(num_frames):
        if prev_body_pos is None:
            root_xy_delta = np.zeros((2,), dtype=np.float32)
        else:
            root_xy_delta = (body_pos[i, :2] - prev_body_pos[:2]).astype(np.float32)
        prev_body_pos = body_pos[i].copy()
        actions[i] = build_action(
            root_xy_delta=root_xy_delta,
            root_z=body_pos[i, 2],
            root_rot6d=quat_to_rot6d_wxyz(body_quat[i]).reshape(6),
            joint_pos=joint_targets[i],
            hand_binary=np.array([left_binary[i], right_binary[i]], dtype=np.float32),
        )
    if align_heading_targets:
        align_quat = _resolve_sonic_heading_align_quat(data, num_frames=num_frames)
        if align_quat is not None:
            return _apply_heading_align_to_action(actions, align_quat_wxyz=align_quat)
    return actions
