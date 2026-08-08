#!/usr/bin/env python3
"""Backfill doubledesk hammer/basket init object states into existing replay npz files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_DATA_ROOT = (
    "/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/HOI_double_desk"
)
DEFAULT_HAMMER_POS = (-3.1370425, -3.0097883, 1.0106319)
DEFAULT_HAMMER_ORI = (0.7139051, 0.0, 0.0, -0.70024246)
DEFAULT_BASKET_POS = (-3.7, -3.2, 0.8)
DEFAULT_BASKET_ORI = (1.0, 0.0, 0.0, 0.0)


def _parse_vec(text: str, expected_dim: int) -> np.ndarray:
    values = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if len(values) != expected_dim:
        raise ValueError(f"Expected {expected_dim} values, got {len(values)} from '{text}'")
    return np.asarray(values, dtype=np.float32)


def _infer_num_frames(data: dict[str, np.ndarray]) -> int:
    priority = (
        "robot_qpos_before_decimation",
        "robot_twist2_inference_qpos",
        "final_body_action_29dof",
        "markers_episode_step",
    )
    for key in priority:
        value = data.get(key)
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] > 0:
            return int(value.shape[0])

    for value in data.values():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] > 0:
            return int(value.shape[0])
    return 1


def _make_episode_init_fields(prefix: str, pos: np.ndarray, ori: np.ndarray) -> dict[str, np.ndarray]:
    return {
        f"episode_init_env_obj_{prefix}_position": pos.astype(np.float32).copy(),
        f"episode_init_env_obj_{prefix}_orientation": ori.astype(np.float32).copy(),
        f"episode_init_env_obj_{prefix}_linear_velocity": np.zeros(3, dtype=np.float32),
        f"episode_init_env_obj_{prefix}_angular_velocity": np.zeros(3, dtype=np.float32),
    }


def _make_env_frame_fields(prefix: str, pos: np.ndarray, ori: np.ndarray, num_frames: int) -> dict[str, np.ndarray]:
    pos_frames = np.repeat(pos.reshape(1, 3), num_frames, axis=0).astype(np.float32)
    ori_frames = np.repeat(ori.reshape(1, 4), num_frames, axis=0).astype(np.float32)
    vel_frames = np.zeros((num_frames, 3), dtype=np.float32)
    return {
        f"env_obj_{prefix}_position": pos_frames,
        f"env_obj_{prefix}_orientation": ori_frames,
        f"env_obj_{prefix}_linear_velocity": vel_frames.copy(),
        f"env_obj_{prefix}_angular_velocity": vel_frames.copy(),
    }


def patch_file(npz_path: Path, hammer_pos, hammer_ori, basket_pos, basket_ori, dry_run: bool) -> bool:
    with np.load(npz_path, allow_pickle=True) as src:
        payload = {k: src[k] for k in src.files}

    num_frames = _infer_num_frames(payload)

    payload.update(_make_episode_init_fields("hammer", hammer_pos, hammer_ori))
    payload.update(_make_episode_init_fields("basket", basket_pos, basket_ori))
    payload.update(_make_env_frame_fields("hammer", hammer_pos, hammer_ori, num_frames))
    payload.update(_make_env_frame_fields("basket", basket_pos, basket_ori, num_frames))

    if dry_run:
        return False

    tmp_path = npz_path.with_suffix(npz_path.suffix + ".tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    tmp_path.replace(npz_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Patch doubledesk replay npz env object init fields.")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--hammer-pos", type=str, default=",".join(map(str, DEFAULT_HAMMER_POS)))
    parser.add_argument("--hammer-ori", type=str, default=",".join(map(str, DEFAULT_HAMMER_ORI)))
    parser.add_argument("--basket-pos", type=str, default=",".join(map(str, DEFAULT_BASKET_POS)))
    parser.add_argument("--basket-ori", type=str, default=",".join(map(str, DEFAULT_BASKET_ORI)))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    npz_files = sorted(data_root.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No npz files found under: {data_root}")

    hammer_pos = _parse_vec(args.hammer_pos, 3)
    hammer_ori = _parse_vec(args.hammer_ori, 4)
    basket_pos = _parse_vec(args.basket_pos, 3)
    basket_ori = _parse_vec(args.basket_ori, 4)

    updated = 0
    skipped = 0
    for path in npz_files:
        try:
            changed = patch_file(
                path,
                hammer_pos=hammer_pos,
                hammer_ori=hammer_ori,
                basket_pos=basket_pos,
                basket_ori=basket_ori,
                dry_run=args.dry_run,
            )
            if changed:
                updated += 1
        except Exception as exc:
            skipped += 1
            print(f"[skip] {path}: {exc}")

    mode = "DRY-RUN checked" if args.dry_run else "patched"
    print(f"{mode} {len(npz_files)} files under {data_root} (updated={updated}, skipped={skipped})")
    print(
        "hammer_pos=", hammer_pos.tolist(),
        "basket_pos=", basket_pos.tolist(),
    )


if __name__ == "__main__":
    main()
