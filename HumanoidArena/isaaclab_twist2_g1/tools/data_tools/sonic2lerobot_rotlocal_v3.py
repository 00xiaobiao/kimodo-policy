#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from smpl_lerobot_v3_common import (
    SCHEMA_VERSION,
    build_features,
    build_sonic_rotlocal_v3_actions,
    extract_canonical_state_v3,
    find_npz_files,
    infer_sonic_fps,
    inspect_image_shape,
    load_vision_rgb_and_indices,
    normalize_task,
    prepare_output_root,
    validate_isaac_time_rows,
    validate_video_frame_indices,
    write_v31_protocol_metadata,
)


DEFAULT_INPUT_ROOT = Path(
    "/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/HOI_grap_cup/sonic"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/dreams/Users/taowen/HumanoidArena/lerobot/outputs/HumanoidArena_datasets_v3_1/HOI_grap_cup/sonic_refpose_v3_1"
)
DEFAULT_REPO_ID = "local/sonic_refpose_v3_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SONIC IsaacLab recordings into LeRobot refpose v3.1 data. "
            "The action target is a 40D canonical input at Isaac control time: "
            "reference-base-local xy delta, root z, episode-reference root rot6d, "
            "29 canonical joints, hand binary."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--robot-type", type=str, default="unitree_g1_refpose_v3_1")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-images", action="store_true")
    parser.set_defaults(strict_video_frames=True)
    parser.add_argument(
        "--strict-video-frames",
        dest="strict_video_frames",
        action="store_true",
        help="Require vision_frame_indices to be exactly 0..N-1 for each Isaac control row.",
    )
    parser.add_argument(
        "--allow-missing-video-frames",
        dest="strict_video_frames",
        action="store_false",
        help="Allow conversion when images do not cover every Isaac control row.",
    )
    return parser.parse_args()


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

    fps = infer_sonic_fps(npz_paths, args.fps)
    image_shape = inspect_image_shape(npz_paths)
    features = build_features(image_shape=image_shape, use_videos=not args.use_images)
    prepare_output_root(output_root, overwrite=args.overwrite)

    logging.info("Found %d SONIC episodes under %s", len(npz_paths), input_root)
    logging.info("Using schema=%s fps=%d image_shape=%s", SCHEMA_VERSION, fps, image_shape)
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
            with np.load(npz_path, allow_pickle=True) as data:
                qpos = np.asarray(data["robot_qpos_before_decimation"], dtype=np.float32)
                qvel = np.asarray(data["robot_qvel_before_decimation"], dtype=np.float32)
                root_orientation = np.asarray(data["robot_root_orientation"], dtype=np.float32)
                if qpos.ndim != 2 or qpos.shape[1] != 29:
                    raise ValueError(f"{npz_path} has unexpected robot_qpos_before_decimation shape: {qpos.shape}")
                if qvel.shape != qpos.shape:
                    raise ValueError(f"{npz_path} has unexpected robot_qvel_before_decimation shape: {qvel.shape}")
                if root_orientation.shape != (qpos.shape[0], 4):
                    raise ValueError(f"{npz_path} has unexpected robot_root_orientation shape: {root_orientation.shape}")

                validate_isaac_time_rows(data, num_frames=qpos.shape[0], npz_path=npz_path)
                vision_rgb, vision_frame_indices = load_vision_rgb_and_indices(data, npz_path)
                if vision_rgb.ndim != 4:
                    raise ValueError(f"{npz_path} has unexpected vision_rgb shape: {vision_rgb.shape}")
                if vision_rgb.shape[0] != len(vision_frame_indices):
                    raise ValueError(
                        f"{npz_path} has mismatched vision buffers: "
                        f"{vision_rgb.shape[0]} rgb frames vs {len(vision_frame_indices)} indices"
                    )
                validate_video_frame_indices(
                    vision_frame_indices,
                    num_frames=qpos.shape[0],
                    npz_path=npz_path,
                    strict=args.strict_video_frames,
                )

                obs_state = extract_canonical_state_v3(
                    root_orientation=root_orientation,
                    joint_pos=qpos,
                    joint_vel=qvel,
                )
                action = build_sonic_rotlocal_v3_actions(
                    data,
                    num_frames=qpos.shape[0],
                    robot_root_orientation=root_orientation,
                )
                if obs_state.shape != (qpos.shape[0], 64):
                    raise ValueError(f"{npz_path} has unexpected v3 observation.state shape: {obs_state.shape}")
                if action.shape != (qpos.shape[0], 40):
                    raise ValueError(f"{npz_path} has unexpected v3 action shape: {action.shape}")

                task = normalize_task(data.get("task"), npz_path, input_root)
                frames_in_episode = 0
                for rgb_index, frame_index_raw in enumerate(vision_frame_indices):
                    frame_index = int(frame_index_raw)
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

            dataset.save_episode()
            total_frames += frames_in_episode
            converted_episodes += 1
            logging.info(
                "[%d/%d] saved %s with %d Isaac-time frames",
                episode_index,
                len(npz_paths),
                npz_path.relative_to(input_root),
                frames_in_episode,
            )
    finally:
        dataset.finalize()

    write_v31_protocol_metadata(output_root, backend_source="sonic", fps=fps)

    logging.info(
        "Finished v3.1 conversion: %d episodes, %d frames. Stats were written by LeRobot metadata.",
        converted_episodes,
        total_frames,
    )


if __name__ == "__main__":
    main()
