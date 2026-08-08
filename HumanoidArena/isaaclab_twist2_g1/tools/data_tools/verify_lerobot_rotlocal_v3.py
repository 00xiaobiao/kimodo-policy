#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from action_provider.vla_robot_current_local_runtime_v3 import ACTION_NAMES, STATE_NAMES
from action_provider.vla_smpl_runtime import quat_to_rot6d_wxyz
from smpl_lerobot_v3_common import (
    build_sonic_rotlocal_v3_actions,
    build_twist2_rotlocal_v3_actions,
    extract_canonical_state_v3,
    find_npz_files,
    infer_twist2_control_dt,
    load_vision_rgb_and_indices,
    reorder_twist2_to_canonical_29,
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_lerobot_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot info.json: {info_path}")
    return load_json(info_path)


def load_lerobot_fps(dataset_root: Path) -> int | None:
    info = load_lerobot_info(dataset_root)
    fps = info.get("fps")
    return int(fps) if fps is not None else None


def _feature_names(info: dict, key: str) -> list[str] | None:
    features = info.get("features", {})
    feature = features.get(key, {}) if isinstance(features, dict) else {}
    names = feature.get("names") if isinstance(feature, dict) else None
    return list(names) if isinstance(names, list) else None


def verify_feature_names(dataset_root: Path) -> None:
    info = load_lerobot_info(dataset_root)
    state_names = _feature_names(info, "observation.state")
    action_names = _feature_names(info, "action")
    if state_names is not None and state_names != STATE_NAMES:
        raise AssertionError("observation.state feature names do not match v3.1 STATE_NAMES")
    if action_names is not None and action_names != ACTION_NAMES:
        raise AssertionError("action feature names do not match v3.1 ACTION_NAMES")
    robot_type = str(info.get("robot_type", ""))
    if robot_type and robot_type != "unitree_g1_refpose_v3_1":
        raise AssertionError(f"Expected robot_type=unitree_g1_refpose_v3_1, got {robot_type}")


def load_lerobot_rows(dataset_root: Path) -> dict[int, dict[str, np.ndarray]]:
    data_paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No LeRobot parquet files found under {dataset_root / 'data'}")

    df = pd.concat([pd.read_parquet(path) for path in data_paths], ignore_index=True)
    required_columns = {"episode_index", "frame_index", "timestamp", "observation.state", "action"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"LeRobot parquet missing required columns: {missing}")

    episodes: dict[int, dict[str, np.ndarray]] = {}
    for episode_index, ep_df in df.groupby("episode_index", sort=True):
        ep_df = ep_df.sort_values("frame_index")
        actions = np.stack(ep_df["action"].to_numpy()).astype(np.float32)
        states = np.stack(ep_df["observation.state"].to_numpy()).astype(np.float32)
        if actions.ndim != 2 or actions.shape[1] != 40:
            raise AssertionError(f"episode {episode_index} action dim mismatch: {actions.shape}")
        if states.ndim != 2 or states.shape[1] != 64:
            raise AssertionError(f"episode {episode_index} observation.state dim mismatch: {states.shape}")
        if not np.isfinite(actions).all() or not np.isfinite(states).all():
            raise AssertionError(f"episode {episode_index} contains NaN/inf in state/action")
        episodes[int(episode_index)] = {
            "frame_index": ep_df["frame_index"].to_numpy(dtype=np.int64),
            "timestamp": ep_df["timestamp"].to_numpy(dtype=np.float64),
            "state": states,
            "action": actions,
        }
    return episodes


def verify_lerobot_clock(
    episode_rows: dict[int, dict[str, np.ndarray]],
    *,
    fps: int | None,
    atol: float,
) -> None:
    for episode_index, rows in episode_rows.items():
        frame_index = rows["frame_index"]
        expected_frame_index = np.arange(frame_index.shape[0], dtype=np.int64)
        if not np.array_equal(frame_index, expected_frame_index):
            raise AssertionError(f"episode {episode_index} frame_index is not 0..N-1")
        if fps is not None:
            expected_timestamp = expected_frame_index.astype(np.float64) / float(fps)
            if not np.allclose(rows["timestamp"], expected_timestamp, atol=atol, rtol=0.0):
                raise AssertionError(f"episode {episode_index} timestamp is not frame_index / fps")


def _warn_if_hand_binary_degenerate(dataset_rows: dict[str, np.ndarray], episode_label: str) -> None:
    hand = dataset_rows["action"][:, 38:40]
    for idx, side in enumerate(("left", "right")):
        values = hand[:, idx]
        if values.size > 8 and float(np.std(values)) < 1e-8:
            logging.warning("%s hand_binary.%s is constant at %.3f", episode_label, side, float(values[0]))


def verify_sonic_episode(
    *,
    npz_path: Path,
    dataset_rows: dict[str, np.ndarray],
    atol: float,
) -> int:
    with np.load(npz_path, allow_pickle=True) as data:
        qpos = np.asarray(data["robot_qpos_before_decimation"], dtype=np.float32)
        qvel = np.asarray(data["robot_qvel_before_decimation"], dtype=np.float32)
        root_orientation = np.asarray(data["robot_root_orientation"], dtype=np.float32)
        _, vision_frame_indices = load_vision_rgb_and_indices(data, npz_path)
        expected_action = build_sonic_rotlocal_v3_actions(
            data,
            num_frames=qpos.shape[0],
            robot_root_orientation=root_orientation,
        )
        expected_state = extract_canonical_state_v3(
            root_orientation=root_orientation,
            joint_pos=qpos,
            joint_vel=qvel,
        )
        raw_quat = np.asarray(data["human_body_quat_w"], dtype=np.float32) if "human_body_quat_w" in data else None

    if dataset_rows["action"].shape[0] != len(vision_frame_indices):
        raise AssertionError(
            f"{npz_path} frame count mismatch: dataset={dataset_rows['action'].shape[0]} "
            f"source vision frames={len(vision_frame_indices)}"
        )
    indices = np.asarray(vision_frame_indices, dtype=np.int64)
    expected_action = expected_action[indices]
    expected_state = expected_state[indices]
    np.testing.assert_allclose(dataset_rows["action"], expected_action, atol=atol, rtol=0.0)
    np.testing.assert_allclose(dataset_rows["state"], expected_state, atol=atol, rtol=0.0)

    if raw_quat is not None:
        raw_rot6d = quat_to_rot6d_wxyz(raw_quat[indices]).reshape(-1, 6)
        raw_diff = float(np.max(np.abs(raw_rot6d - expected_action[:, 3:9])))
        if raw_diff <= atol and raw_rot6d.shape[0] > 1:
            logging.warning("%s v3.1 action rot6d equals raw SONIC body quat; check episode heading distribution", npz_path)

    _warn_if_hand_binary_degenerate(dataset_rows, npz_path.as_posix())
    return int(expected_action.shape[0])


def verify_twist2_episode(
    *,
    npz_path: Path,
    dataset_rows: dict[str, np.ndarray],
    fps: int | None,
    atol: float,
) -> int:
    with np.load(npz_path, allow_pickle=True) as data:
        qpos = reorder_twist2_to_canonical_29(
            np.asarray(data["robot_qpos_before_decimation"], dtype=np.float32)
        )
        qvel = reorder_twist2_to_canonical_29(
            np.asarray(data["robot_qvel_before_decimation"], dtype=np.float32)
        )
        root_orientation = np.asarray(data["robot_root_orientation"], dtype=np.float32)
        _, vision_frame_indices = load_vision_rgb_and_indices(data, npz_path)
        control_dt = infer_twist2_control_dt(data, fps or 50)
        expected_action = build_twist2_rotlocal_v3_actions(
            data,
            num_frames=qpos.shape[0],
            control_dt=control_dt,
            robot_root_orientation=root_orientation,
        )
        expected_state = extract_canonical_state_v3(
            root_orientation=root_orientation,
            joint_pos=qpos,
            joint_vel=qvel,
        )

    if dataset_rows["action"].shape[0] != len(vision_frame_indices):
        raise AssertionError(
            f"{npz_path} frame count mismatch: dataset={dataset_rows['action'].shape[0]} "
            f"source vision frames={len(vision_frame_indices)}"
        )
    indices = np.asarray(vision_frame_indices, dtype=np.int64)
    expected_action = expected_action[indices]
    expected_state = expected_state[indices]
    np.testing.assert_allclose(dataset_rows["action"], expected_action, atol=atol, rtol=0.0)
    np.testing.assert_allclose(dataset_rows["state"], expected_state, atol=atol, rtol=0.0)
    _warn_if_hand_binary_degenerate(dataset_rows, npz_path.as_posix())
    return int(expected_action.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify refpose v3.1 LeRobot data against source npz files.")
    parser.add_argument("--backend", choices=("sonic", "twist2"), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_root = args.input_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    npz_paths = find_npz_files(input_root)
    if args.limit is not None:
        npz_paths = npz_paths[: args.limit]
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found under {input_root}")

    verify_feature_names(dataset_root)
    rows_by_episode = load_lerobot_rows(dataset_root)
    fps = load_lerobot_fps(dataset_root)
    verify_lerobot_clock(rows_by_episode, fps=fps, atol=args.atol)

    if len(rows_by_episode) < len(npz_paths):
        raise AssertionError(
            f"Dataset has fewer episodes than source paths: {len(rows_by_episode)} < {len(npz_paths)}"
        )

    total_frames = 0
    for episode_index, npz_path in enumerate(npz_paths):
        rows = rows_by_episode[episode_index]
        if args.backend == "sonic":
            frames = verify_sonic_episode(npz_path=npz_path, dataset_rows=rows, atol=args.atol)
        else:
            frames = verify_twist2_episode(npz_path=npz_path, dataset_rows=rows, fps=fps, atol=args.atol)
        total_frames += frames
        logging.info("verified episode %d: %s (%d frames)", episode_index, npz_path.relative_to(input_root), frames)

    logging.info("Verified %d %s v3 episodes, %d frames", len(npz_paths), args.backend, total_frames)


if __name__ == "__main__":
    main()
