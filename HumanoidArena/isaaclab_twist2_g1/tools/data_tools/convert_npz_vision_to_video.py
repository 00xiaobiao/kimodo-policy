#!/usr/bin/env python3
"""Offline migration: convert episode NPZ vision arrays to sidecar MP4 video format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from action_provider.vision_video import write_rgb_video_mp4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_root",
        type=Path,
        required=True,
        help="Root directory containing episode .npz files.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="**/*.npz",
        help="Glob pattern under input_root.",
    )
    parser.add_argument(
        "--keep-depth",
        action="store_true",
        help="Keep depth array and convert it to float16 instead of dropping.",
    )
    parser.add_argument(
        "--overwrite-existing-video",
        action="store_true",
        help="Overwrite existing MP4 sidecar files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversions without writing files.",
    )
    return parser.parse_args()


def _decode_scalar(value) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _decode_scalar(value.item())
        if value.size == 1:
            return _decode_scalar(value.reshape(-1)[0])
        return str(value.tolist())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _infer_fps(data: np.lib.npyio.NpzFile) -> float:
    if "vision_rgb_video_fps" in data:
        try:
            return float(np.asarray(data["vision_rgb_video_fps"]).reshape(-1)[0])
        except Exception:
            pass
    if "meta_control_dt" in data:
        try:
            control_dt = float(np.asarray(data["meta_control_dt"]).reshape(-1)[0])
            if control_dt > 1e-6:
                return float(np.clip(1.0 / control_dt, 1.0, 240.0))
        except Exception:
            pass
    if "system_control_frequency" in data:
        try:
            freq = np.asarray(data["system_control_frequency"], dtype=np.float32).reshape(-1)
            if freq.size > 0:
                return float(np.clip(np.nanmedian(freq), 1.0, 240.0))
        except Exception:
            pass
    return 30.0


def _needs_convert(data: np.lib.npyio.NpzFile) -> bool:
    return "vision_rgb" in data and "vision_rgb_video_path" not in data


def convert_one(
    npz_path: Path,
    *,
    keep_depth: bool,
    overwrite_existing_video: bool,
    dry_run: bool,
) -> bool:
    with np.load(npz_path, allow_pickle=True) as data:
        if not _needs_convert(data):
            return False

        vision_rgb = np.asarray(data["vision_rgb"])
        if vision_rgb.ndim != 4:
            raise ValueError(f"{npz_path}: unexpected vision_rgb shape {vision_rgb.shape}")
        frame_indices = (
            np.asarray(data["vision_frame_indices"], dtype=np.int32)
            if "vision_frame_indices" in data
            else np.arange(vision_rgb.shape[0], dtype=np.int32)
        )
        fps = _infer_fps(data)

        task = _decode_scalar(data["task"]) if "task" in data else npz_path.stem
        ts = None
        if "save_timestamp_us" in data:
            try:
                ts = int(np.asarray(data["save_timestamp_us"]).reshape(-1)[0])
            except Exception:
                ts = None
        if ts is None:
            ts = int(npz_path.stat().st_mtime_ns // 1000)

        video_rel = Path("videos") / f"{task}_{ts}_front_rgb.mp4"
        video_abs = npz_path.parent / video_rel

        if dry_run:
            reuse = video_abs.exists() and not overwrite_existing_video
            mode = "reuse-existing-video" if reuse else "encode-video"
            print(
                f"[DRY-RUN] {npz_path} -> {video_abs} "
                f"(fps={fps:.2f}, frames={vision_rgb.shape[0]}, mode={mode})"
            )
            return True

        if video_abs.exists() and not overwrite_existing_video:
            written = int(vision_rgb.shape[0])
            print(f"[REUSE] {npz_path} uses existing video: {video_abs}")
        else:
            written = write_rgb_video_mp4(vision_rgb, video_abs, fps=fps)

        new_payload: dict[str, np.ndarray] = {}
        for key in data.files:
            if key == "vision_rgb":
                continue
            if key == "vision_depth" and not keep_depth:
                continue
            value = data[key]
            if key == "vision_depth" and keep_depth:
                value = np.asarray(value, dtype=np.float16)
            new_payload[key] = value

        new_payload["vision_storage_format"] = np.array("video_v1")
        new_payload["vision_rgb_video_path"] = np.array(str(video_rel))
        new_payload["vision_rgb_video_fps"] = np.array(fps, dtype=np.float32)
        new_payload["vision_rgb_video_num_frames"] = np.array(int(written), dtype=np.int32)
        new_payload["vision_frame_indices"] = frame_indices

    tmp_base = npz_path.parent / f"{npz_path.stem}.tmp_write"
    tmp_npz = Path(str(tmp_base) + ".npz")
    if tmp_npz.exists():
        tmp_npz.unlink(missing_ok=True)
    np.savez_compressed(tmp_base, **new_payload)
    if tmp_npz.resolve() == npz_path.resolve():
        raise RuntimeError(f"Temporary path collision for {npz_path}: {tmp_npz}")
    tmp_npz.replace(npz_path)

    print(f"[OK] {npz_path} -> {video_abs} ({written} frames, fps={fps:.2f})")
    return True


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    npz_paths = sorted(p for p in input_root.glob(args.glob) if p.is_file())
    if not npz_paths:
        raise FileNotFoundError(f"No npz files found under {input_root} with glob {args.glob!r}")

    converted = 0
    skipped = 0
    for npz_path in npz_paths:
        changed = convert_one(
            npz_path,
            keep_depth=args.keep_depth,
            overwrite_existing_video=args.overwrite_existing_video,
            dry_run=args.dry_run,
        )
        if changed:
            converted += 1
        else:
            skipped += 1

    print(f"Done. converted={converted} skipped={skipped} total={len(npz_paths)}")


if __name__ == "__main__":
    main()
