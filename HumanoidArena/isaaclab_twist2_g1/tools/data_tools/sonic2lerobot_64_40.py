#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from smpl_lerobot_common import (
    build_features,
    build_sonic_actions_from_recording,
    extract_canonical_state,
    find_npz_files,
    load_vision_rgb_and_indices,
    inspect_image_shape,
    normalize_task,
    prepare_output_root,
)


DEFAULT_INPUT_ROOT = Path(
    "/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/HOI_grapcup/sonic_v2"
)
# DEFAULT_OUTPUT_ROOT = Path("/home/dreams/Users/taowen/HumanoidArena/lerobot/outputs/HumanoidArena_datasets/HOI_football/0409_sonic_smpl_pose6d")
DEFAULT_OUTPUT_ROOT = Path("/home/dreams/Users/taowen/HumanoidArena/lerobot/outputs/HumanoidArena_datasets/HOI_grap_cup/sonic_grapcup_0423")
DEFAULT_REPO_ID = "local/sonic_grapcup_0423"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SONIC IsaacLab recordings into a LeRobot dataset with unified "
            "SMPL-pose action labels: root translation delta, root orientation rot6d, "
            "body local pose rot6d, and hand binary."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--robot-type", type=str, default="unitree_g1_smpl_pose6d_vla")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-images", action="store_true")
    parser.set_defaults(align_heading_targets=True)
    parser.add_argument(
        "--align-heading-targets",
        dest="align_heading_targets",
        action="store_true",
        help="Rotate SONIC action root orientation/xy delta into heading-aligned frame.",
    )
    parser.add_argument(
        "--no-align-heading-targets",
        dest="align_heading_targets",
        action="store_false",
        help="Keep raw SONIC heading in action targets.",
    )
    return parser.parse_args()


def infer_fps(npz_paths: list[Path], fps_override: int | None) -> int:
    if fps_override is not None:
        return fps_override
    if not npz_paths:
        raise ValueError("No npz files found.")

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
            skipped_invalid_indices = 0
            skipped_missing_smpl = 0

            with np.load(npz_path, allow_pickle=True) as data:
                qpos = np.asarray(data["robot_qpos_before_decimation"], dtype=np.float32)
                qvel = np.asarray(data["robot_qvel_before_decimation"], dtype=np.float32)
                root_orientation = np.asarray(data["robot_root_orientation"], dtype=np.float32)
                vision_rgb, vision_frame_indices = load_vision_rgb_and_indices(data, npz_path)
                obs_state = extract_canonical_state(
                    data=data,
                    root_orientation=root_orientation,
                    joint_pos=qpos,
                    joint_vel=qvel,
                )
                action = build_sonic_actions_from_recording(
                    data,
                    num_frames=qpos.shape[0],
                    align_heading_targets=args.align_heading_targets,
                )

                if qpos.ndim != 2 or qpos.shape[1] != 29:
                    raise ValueError(f"{npz_path} has unexpected robot_qpos_before_decimation shape: {qpos.shape}")
                if qvel.shape != qpos.shape:
                    raise ValueError(f"{npz_path} has unexpected robot_qvel_before_decimation shape: {qvel.shape}")
                if root_orientation.shape != (qpos.shape[0], 4):
                    raise ValueError(f"{npz_path} has unexpected robot_root_orientation shape: {root_orientation.shape}")
                if obs_state.shape != (qpos.shape[0], 64):
                    raise ValueError(f"{npz_path} has unexpected canonical observation.state shape: {obs_state.shape}")
                if action.shape != (qpos.shape[0], 40):
                    raise ValueError(f"{npz_path} has unexpected canonical action shape: {action.shape}")
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
                        skipped_invalid_indices += 1
                        continue

                    dataset.add_frame(
                        {
                            "task": task,
                            "observation.images.front": np.ascontiguousarray(vision_rgb[rgb_index]),
                            "observation.state": np.ascontiguousarray(obs_state[frame_index]),
                            "action": np.ascontiguousarray(action[frame_index]),
                        }
                    )
                    frames_in_episode += 1

            if frames_in_episode == 0:
                logging.warning("Skip empty episode: %s", npz_path)
                dataset.clear_episode_buffer(delete_images=True)
                continue

            if skipped_invalid_indices or skipped_missing_smpl:
                logging.warning(
                    "%s skipped %d out-of-range frames and %d frames with missing raw SMPL",
                    npz_path.relative_to(input_root),
                    skipped_invalid_indices,
                    skipped_missing_smpl,
                )

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
