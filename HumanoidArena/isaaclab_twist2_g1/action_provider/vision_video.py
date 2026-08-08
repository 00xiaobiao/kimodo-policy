"""Utilities for storing per-episode vision as compressed video files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC frame, got shape={arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.shape[2] != 3:
        raise ValueError(f"Expected 3 channels, got shape={arr.shape}")
    if arr.dtype in (np.float32, np.float64):
        if float(np.nanmax(arr)) <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255)
        else:
            arr = np.clip(arr, 0, 255)
        return arr.astype(np.uint8)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def write_rgb_video_mp4(
    frames: Iterable[np.ndarray],
    output_path: str | Path,
    *,
    fps: float,
    codec: str = "mp4v",
) -> int:
    """Encode RGB frames into MP4 and return number of written frames."""
    frames = list(frames)
    if not frames:
        return 0

    first = _to_uint8_rgb(frames[0])
    h, w = int(first.shape[0]), int(first.shape[1])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        float(max(1.0, fps)),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")

    written = 0
    try:
        for frame in frames:
            rgb = _to_uint8_rgb(frame)
            if rgb.shape[0] != h or rgb.shape[1] != w:
                raise ValueError(
                    f"Frame size mismatch while writing {output_path}: "
                    f"expected {(h, w)}, got {(rgb.shape[0], rgb.shape[1])}"
                )
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            written += 1
    finally:
        writer.release()

    return written


def read_rgb_video_mp4(video_path: str | Path) -> np.ndarray:
    """Decode MP4 video into RGB uint8 frames with shape [T, H, W, 3]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for reading: {video_path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        cap.release()

    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames, axis=0).astype(np.uint8)
