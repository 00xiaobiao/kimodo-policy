#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from action_provider.vision_video import read_rgb_video_mp4
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_INPUT_ROOT = Path(
    "/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/datasets/sonic_0407_vla")
DEFAULT_REPO_ID = "local/sonic_0407_vla"

SONIC_JOINT_NAMES = [
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

SONIC_DEFAULT_POS = np.array(
    [
        -0.312,
        -0.312,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.669,
        0.669,
        0.2,
        0.2,
        -0.363,
        -0.363,
        0.2,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.6,
        0.6,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

SMPL_JOINT_NAMES = [
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

STATE_NAMES = [
    "state.ang_vel_local_x",
    "state.ang_vel_local_y",
    "state.ang_vel_local_z",
    "state.gravity_dir_x",
    "state.gravity_dir_y",
    "state.gravity_dir_z",
    *[f"state.dof_pos_delta.{name}" for name in SONIC_JOINT_NAMES],
    *[f"state.dof_vel.{name}" for name in SONIC_JOINT_NAMES],
    *[f"state.last_body_raw_action.{name}" for name in SONIC_JOINT_NAMES],
    "state.left_hand_binary",
    "state.right_hand_binary",
]

ACTION_NAMES = [
    *[
        f"action.body.smpl_joint.{joint_name}.{axis}"
        for joint_name in SMPL_JOINT_NAMES
        for axis in ("x", "y", "z")
    ],
    *[f"action.body.anchor_rot6d.{idx}" for idx in range(6)],
    *[f"action.body.wrist_ref.{idx}" for idx in range(6)],
    "action.hand.left_binary",
    "action.hand.right_binary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SONIC IsaacLab recordings into a LeRobot VLA dataset with "
            "`observation.images.front`, `observation.state` (95D), and `action` (86D)."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--robot-type", type=str, default="unitree_g1_sonic_vla")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-images", action="store_true")
    return parser.parse_args()


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


def infer_fps(npz_paths: list[Path], fps_override: int | None) -> int:
    if fps_override is not None:
        return fps_override
    if not npz_paths:
        raise ValueError("No npz files found.")
    freqs = []
    for path in npz_paths[: min(len(npz_paths), 32)]:
        with np.load(path, allow_pickle=True) as data:
            if "meta_control_dt" in data:
                control_dt = float(np.asarray(data["meta_control_dt"]).reshape(-1)[0])
                if control_dt > 0:
                    freqs.append(1.0 / control_dt)
    fps = int(round(float(np.median(freqs))))
    if fps <= 0:
        raise ValueError(f"Invalid inferred fps: {fps}")
    return fps


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


def load_vision_rgb_and_indices(data: np.lib.npyio.NpzFile, npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if "vision_rgb" in data:
        vision_rgb = np.asarray(data["vision_rgb"], dtype=np.uint8)
    elif "vision_rgb_video_path" in data:
        video_rel = decode_scalar(data["vision_rgb_video_path"])
        vision_rgb = read_rgb_video_mp4(npz_path.parent / video_rel)
    else:
        raise KeyError(f"{npz_path} missing vision data")
    if "vision_frame_indices" in data:
        indices = np.asarray(data["vision_frame_indices"], dtype=np.int64)
    else:
        indices = np.arange(vision_rgb.shape[0], dtype=np.int64)
    return vision_rgb, indices


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
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
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


def gravity_dir_from_base_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = quat_normalize_wxyz(quat)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    gravity_quat = np.array([0.0, gravity_world[0], gravity_world[1], gravity_world[2]], dtype=np.float32)
    rotated = quat_mul_wxyz(quat_mul_wxyz(quat_conjugate_wxyz(quat), gravity_quat), quat)
    return rotated[1:].astype(np.float32)


def get_hand_binary_arrays(data: np.lib.npyio.NpzFile, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    if "vla_action_hand_binary" in data:
        hand = np.asarray(data["vla_action_hand_binary"], dtype=np.float32)
        if hand.shape == (num_frames, 2):
            return hand[:, 0], hand[:, 1]
    left = np.asarray(data["pico_left_grip_binary"], dtype=np.float32) if "pico_left_grip_binary" in data else None
    right = np.asarray(data["pico_right_grip_binary"], dtype=np.float32) if "pico_right_grip_binary" in data else None
    if left is None:
        left = np.zeros(num_frames, dtype=np.float32)
    if right is None:
        right = np.zeros(num_frames, dtype=np.float32)
    if left.shape != (num_frames,):
        raise ValueError(f"Unexpected pico_left_grip_binary shape: {left.shape}, expected {(num_frames,)}")
    if right.shape != (num_frames,):
        raise ValueError(f"Unexpected pico_right_grip_binary shape: {right.shape}, expected {(num_frames,)}")
    return left, right


def get_body_token(data: np.lib.npyio.NpzFile, frame_index: int) -> np.ndarray:
    if "vla_action_body_token" in data:
        token = np.asarray(data["vla_action_body_token"][frame_index], dtype=np.float32).reshape(-1)
    else:
        smpl = np.asarray(data["encoder_smpl_joint_window"][frame_index, -1], dtype=np.float32).reshape(-1)
        anchor = np.asarray(data["encoder_anchor_window"][frame_index, -1], dtype=np.float32).reshape(-1)
        wrist = np.asarray(data["encoder_wrist_window"][frame_index, -1], dtype=np.float32).reshape(-1)
        token = np.concatenate([smpl, anchor, wrist], axis=0).astype(np.float32)
    if token.shape != (84,):
        raise ValueError(f"Unexpected SONIC body token shape: {token.shape}, expected {(84,)}")
    return token


def build_observation_state(
    *,
    root_ang_vel_local: np.ndarray,
    root_orientation: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    prev_raw_action: np.ndarray,
    prev_hand_binary: np.ndarray,
) -> np.ndarray:
    gravity_dir = gravity_dir_from_base_quat_wxyz(root_orientation)
    state = np.concatenate(
        [
            np.asarray(root_ang_vel_local, dtype=np.float32).reshape(-1),
            gravity_dir.reshape(-1),
            (np.asarray(qpos, dtype=np.float32).reshape(-1) - SONIC_DEFAULT_POS),
            np.asarray(qvel, dtype=np.float32).reshape(-1),
            np.asarray(prev_raw_action, dtype=np.float32).reshape(-1),
            np.asarray(prev_hand_binary, dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)
    if state.shape != (95,):
        raise ValueError(f"Unexpected SONIC observation.state shape: {state.shape}, expected {(95,)}")
    return state


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    npz_paths = find_npz_files(input_root)
    if args.limit is not None:
        npz_paths = npz_paths[: args.limit]
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found under {input_root}")

    fps = infer_fps(npz_paths, args.fps)
    image_shape = inspect_image_shape(npz_paths)
    features = build_features(image_shape=image_shape, use_videos=not args.use_images)
    prepare_output_root(output_root, overwrite=args.overwrite)

    logging.info("Found %d SONIC episodes under %s", len(npz_paths), input_root)
    logging.info("Using fps=%d image_shape=%s", fps, image_shape)
    logging.info("Writing LeRobot dataset to %s", output_root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        robot_type=args.robot_type,
        fps=fps,
        features=features,
        use_videos=not args.use_images,
    )

    total_frames = 0
    converted_episodes = 0
    try:
        for episode_index, npz_path in enumerate(npz_paths, start=1):
            frames_in_episode = 0
            with np.load(npz_path, allow_pickle=True) as data:
                qpos = np.asarray(data["robot_qpos_before_decimation"], dtype=np.float32)
                qvel = np.asarray(data["robot_qvel_before_decimation"], dtype=np.float32)
                root_orientation = np.asarray(data["robot_root_orientation"], dtype=np.float32)
                root_ang_vel_local = np.asarray(data["robot_root_ang_vel_local"], dtype=np.float32)
                raw_action = np.asarray(data["decoder_raw_action"], dtype=np.float32)
                vision_rgb, vision_frame_indices = load_vision_rgb_and_indices(data, npz_path)
                left_binary, right_binary = get_hand_binary_arrays(data, qpos.shape[0])

                if qpos.ndim != 2 or qpos.shape[1] != 29:
                    raise ValueError(f"{npz_path} has unexpected robot_qpos_before_decimation shape: {qpos.shape}")
                if qvel.shape != qpos.shape:
                    raise ValueError(f"{npz_path} has unexpected robot_qvel_before_decimation shape: {qvel.shape}")
                if raw_action.shape != qpos.shape:
                    raise ValueError(f"{npz_path} has unexpected decoder_raw_action shape: {raw_action.shape}")
                if root_orientation.shape != (qpos.shape[0], 4):
                    raise ValueError(f"{npz_path} has unexpected robot_root_orientation shape: {root_orientation.shape}")
                if root_ang_vel_local.shape != (qpos.shape[0], 3):
                    raise ValueError(
                        f"{npz_path} has unexpected robot_root_ang_vel_local shape: {root_ang_vel_local.shape}"
                    )
                if vision_rgb.ndim != 4:
                    raise ValueError(f"{npz_path} has unexpected vision_rgb shape: {vision_rgb.shape}")
                if vision_rgb.shape[0] != len(vision_frame_indices):
                    raise ValueError(
                        f"{npz_path} has mismatched vision buffers: "
                        f"{vision_rgb.shape[0]} rgb frames vs {len(vision_frame_indices)} indices"
                    )

                task = normalize_task(data.get("task"), npz_path, input_root)
                for rgb_index, frame_index_raw in enumerate(vision_frame_indices):
                    frame_index = int(frame_index_raw)
                    if frame_index < 0 or frame_index >= qpos.shape[0]:
                        logging.warning("Skip invalid frame index %d in %s", frame_index, npz_path)
                        continue

                    prev_index = max(frame_index - 1, 0)
                    prev_raw_action = np.zeros(29, dtype=np.float32) if frame_index == 0 else raw_action[prev_index]
                    prev_hand_binary = np.zeros(2, dtype=np.float32)
                    if frame_index > 0:
                        prev_hand_binary[:] = np.array([left_binary[prev_index], right_binary[prev_index]], dtype=np.float32)

                    obs_state = build_observation_state(
                        root_ang_vel_local=root_ang_vel_local[frame_index],
                        root_orientation=root_orientation[frame_index],
                        qpos=qpos[frame_index],
                        qvel=qvel[frame_index],
                        prev_raw_action=prev_raw_action,
                        prev_hand_binary=prev_hand_binary,
                    )
                    action = np.concatenate(
                        [
                            get_body_token(data, frame_index),
                            np.array([left_binary[frame_index], right_binary[frame_index]], dtype=np.float32),
                        ],
                        axis=0,
                    ).astype(np.float32)

                    dataset.add_frame(
                        {
                            "task": task,
                            "observation.images.front": np.ascontiguousarray(vision_rgb[rgb_index]),
                            "observation.state": np.ascontiguousarray(obs_state),
                            "action": np.ascontiguousarray(action),
                        }
                    )
                    frames_in_episode += 1

            if frames_in_episode == 0:
                logging.warning("Skip empty episode: %s", npz_path)
                dataset.clear_episode_buffer(delete_images=True)
                continue

            dataset.save_episode()
            total_frames += frames_in_episode
            converted_episodes += 1
            logging.info(
                "[%d/%d] saved %s with %d frames",
                episode_index,
                len(npz_paths),
                npz_path.relative_to(input_root),
                frames_in_episode,
            )
    finally:
        dataset.finalize()

    logging.info(
        "Finished conversion: %d episodes, %d frames. Stats were written by LeRobot metadata.",
        converted_episodes,
        total_frames,
    )


if __name__ == "__main__":
    main()
