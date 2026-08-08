#!/usr/bin/env python3
"""Extract one frame at a given second from front/left/right rerecord videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VIEW_SUFFIXES = {
    "front": "_front_rgb.mp4",
    "left": "_left_wrist_rgb.mp4",
    "right": "_right_wrist_rgb.mp4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one frame at a given second from the front/left/right view videos."
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to any one of the three videos in the same multiview set.",
    )
    parser.add_argument(
        "--second",
        type=float,
        required=True,
        help="Timestamp in seconds to extract.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output PNG files. Defaults to <video_dir>/frames_at_<second>s.",
    )
    return parser.parse_args()


def resolve_group(video_path: Path) -> tuple[Path, dict[str, Path]]:
    name = video_path.name
    matched_suffix = None
    for suffix in VIEW_SUFFIXES.values():
        if name.endswith(suffix):
            matched_suffix = suffix
            break
    if matched_suffix is None:
        raise ValueError(
            f"Unsupported video name: {video_path.name}. Expected one of: {list(VIEW_SUFFIXES.values())}"
        )

    stem_prefix = name[: -len(matched_suffix)]
    group = {}
    for view, suffix in VIEW_SUFFIXES.items():
        candidate = video_path.parent / f"{stem_prefix}{suffix}"
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing {view} view video: {candidate}")
        group[view] = candidate
    return video_path.parent, group


def extract_frame(video_path: Path, second: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise RuntimeError(f"Invalid FPS for video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = max(0, min(int(round(second * fps)), max(0, frame_count - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(
                f"Failed to read frame at second={second} (frame_index={frame_index}) from {video_path}"
            )
        return frame_index, fps, frame_bgr
    finally:
        cap.release()


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_dir, group = resolve_group(video_path)
    second_tag = str(args.second).replace(".", "p")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (video_dir / f"frames_at_{second_tag}s").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for view, path in group.items():
        frame_index, fps, frame_bgr = extract_frame(path, args.second)
        output_path = output_dir / f"{path.stem}_t{second_tag}s.png"
        if not cv2.imwrite(str(output_path), frame_bgr):
            raise RuntimeError(f"Failed to write image: {output_path}")
        print(
            f"[extract_multiview_video_frames] view={view} "
            f"video={path} fps={fps:.3f} frame_index={frame_index} output={output_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
