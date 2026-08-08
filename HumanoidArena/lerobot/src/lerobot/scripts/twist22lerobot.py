#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_INPUT_ROOT = Path(
    "/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/0401"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/datasets/twist2_0401_vla")
DEFAULT_REPO_ID = "local/twist2-0401-vla"

OBS_PROPRIO_START = 35
OBS_PROPRIO_END = 127
FUTURE_OBS_START = 1397
FUTURE_OBS_END = 1432

TWIST2_ACTION_JOINT_NAMES = [
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

ACTION_NAMES = [
    "command.xy_vel_x",
    "command.xy_vel_y",
    "command.z_pos",
    "command.roll",
    "command.pitch",
    "command.yaw_vel",
    *[f"command.joint_target.{name}" for name in TWIST2_ACTION_JOINT_NAMES],
    "command.left_grip_binary",
    "command.right_grip_binary",
]

STATE_NAMES = [
    "state.ang_vel_scaled_x",
    "state.ang_vel_scaled_y",
    "state.ang_vel_scaled_z",
    "state.roll",
    "state.pitch",
    *[f"state.dof_pos_delta.{name}" for name in TWIST2_ACTION_JOINT_NAMES],
    *[f"state.dof_vel_scaled.{name}" for name in TWIST2_ACTION_JOINT_NAMES],
    *[f"state.last_action.{name}" for name in TWIST2_ACTION_JOINT_NAMES],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert TWIST2 IsaacLab recordings into a LeRobot VLA-style dataset. "
            "Each npz file becomes one episode with "
            "`observation.images.front`, `observation.state`, and `action`."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory that will be searched recursively for TWIST2 .npz recordings.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Local directory for the generated LeRobot dataset.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="LeRobot dataset repo id stored in metadata.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="unitree_g1_twist2",
        help="Robot type stored in LeRobot metadata.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override dataset fps. By default it is inferred from system_control_frequency.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only convert the first N episodes. Useful for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before conversion if it already exists.",
    )
    parser.add_argument(
        "--use-images",
        action="store_true",
        help="Store RGB frames as images instead of encoded videos.",
    )
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
            raw = np.asarray(data["system_control_frequency"])
            freqs.append(float(np.median(raw)))

    fps = int(round(float(np.median(freqs))))
    if fps <= 0:
        raise ValueError(f"Invalid inferred fps: {fps}")
    return fps


def inspect_image_shape(npz_paths: list[Path]) -> tuple[int, int, int]:
    for path in npz_paths:
        with np.load(path, allow_pickle=True) as data:
            vision_rgb = np.asarray(data["vision_rgb"])
            if vision_rgb.ndim != 4:
                raise ValueError(f"{path} has unexpected vision_rgb shape: {vision_rgb.shape}")
            if vision_rgb.shape[0] == 0:
                continue
            return tuple(int(v) for v in vision_rgb.shape[1:])
    raise ValueError("No RGB frames found in the input dataset.")


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


def get_grip_binary_arrays(data: np.lib.npyio.NpzFile, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
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


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    elif output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_root}. "
            "Use --overwrite to replace it."
        )


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

    logging.info("Found %d episodes under %s", len(npz_paths), input_root)
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
                obs_buf = np.asarray(data["robot_obs_buf"], dtype=np.float32)
                vision_rgb = np.asarray(data["vision_rgb"], dtype=np.uint8)
                vision_frame_indices = np.asarray(data["vision_frame_indices"], dtype=np.int64)
                left_grip_binary, right_grip_binary = get_grip_binary_arrays(data, obs_buf.shape[0])

                if obs_buf.ndim != 2 or obs_buf.shape[1] < FUTURE_OBS_END:
                    raise ValueError(f"{npz_path} has unexpected robot_obs_buf shape: {obs_buf.shape}")
                if vision_rgb.ndim != 4:
                    raise ValueError(f"{npz_path} has unexpected vision_rgb shape: {vision_rgb.shape}")
                if vision_rgb.shape[0] != len(vision_frame_indices):
                    raise ValueError(
                        f"{npz_path} has mismatched vision buffers: "
                        f"{vision_rgb.shape[0]} rgb frames vs {len(vision_frame_indices)} indices"
                    )

                task = normalize_task(data.get("task"), npz_path, input_root)
                for rgb_index, frame_index in enumerate(vision_frame_indices):
                    frame_index = int(frame_index)
                    if frame_index < 0 or frame_index >= obs_buf.shape[0]:
                        logging.warning("Skip invalid frame index %d in %s", frame_index, npz_path)
                        continue

                    dataset.add_frame(
                        {
                            "task": task,
                            "observation.images.front": np.ascontiguousarray(vision_rgb[rgb_index]),
                            "observation.state": np.ascontiguousarray(
                                obs_buf[frame_index, OBS_PROPRIO_START:OBS_PROPRIO_END]
                            ),
                            "action": np.ascontiguousarray(
                                np.concatenate(
                                    [
                                        obs_buf[frame_index, FUTURE_OBS_START:FUTURE_OBS_END],
                                        np.array(
                                            [
                                                left_grip_binary[frame_index],
                                                right_grip_binary[frame_index],
                                            ],
                                            dtype=np.float32,
                                        ),
                                    ],
                                    axis=0,
                                )
                            ),
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
