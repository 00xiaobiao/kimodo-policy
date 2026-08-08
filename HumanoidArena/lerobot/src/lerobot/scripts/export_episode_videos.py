#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export one video per episode from a local LeRobot dataset.

Usage examples:

Dry run only:
    python -m lerobot.scripts.export_episode_videos \
        --dataset-root /path/to/dataset \
        --dryrun

Export all episode videos:
    python -m lerobot.scripts.export_episode_videos \
        --dataset-root /path/to/dataset
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class EpisodeSegment:
    """A contiguous run of frames for one episode inside one video file."""

    video_path: Path
    local_start: int
    local_end: int

    @property
    def num_frames(self) -> int:
        return self.local_end - self.local_start + 1


@dataclass(slots=True)
class EpisodeSummary:
    """Aggregated metadata for one episode."""

    episode_index: int
    segments: list[EpisodeSegment] = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        return sum(segment.num_frames for segment in self.segments)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to a local LeRobot dataset root containing meta/, data/, and videos/.",
    )
    parser.add_argument(
        "--video-key",
        type=str,
        default=None,
        help="Video feature key to export. Defaults to the only video key found in meta/info.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save episode mp4 files. Defaults to <dataset-root>/episode_videos/<video-key>/.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Only print episode counts, frame counts, and durations; do not export videos.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="ffmpeg executable to use for exporting videos.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output mp4 files.",
    )
    return parser.parse_args()


def load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    with info_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def discover_video_key(info: dict, requested_key: str | None) -> str:
    video_keys = [
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    if requested_key:
        if requested_key not in video_keys:
            raise ValueError(
                f"Requested --video-key '{requested_key}' not found. Available video keys: {video_keys}"
            )
        return requested_key
    if len(video_keys) != 1:
        raise ValueError(f"Expected exactly one video key, found {video_keys}. Please pass --video-key.")
    return video_keys[0]


def iter_episode_runs(episode_indices: list[int]) -> list[tuple[int, int, int]]:
    """Return contiguous runs as (episode_index, start_row, end_row)."""
    if not episode_indices:
        return []

    runs: list[tuple[int, int, int]] = []
    current_episode = int(episode_indices[0])
    start = 0
    for row_idx in range(1, len(episode_indices)):
        episode = int(episode_indices[row_idx])
        if episode != current_episode:
            runs.append((current_episode, start, row_idx - 1))
            current_episode = episode
            start = row_idx
    runs.append((current_episode, start, len(episode_indices) - 1))
    return runs


def collect_episode_summaries(dataset_root: Path, video_key: str) -> tuple[dict[int, EpisodeSummary], float]:
    info = load_info(dataset_root)
    fps = float(info["fps"])

    parquet_files = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    summaries: dict[int, EpisodeSummary] = {}
    for parquet_path in parquet_files:
        relative_data_path = parquet_path.relative_to(dataset_root / "data")
        video_path = (dataset_root / "videos" / video_key / relative_data_path).with_suffix(".mp4")
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Missing matching video file for {parquet_path}: expected {video_path}"
            )

        df = pd.read_parquet(parquet_path, columns=["episode_index"])
        runs = iter_episode_runs(df["episode_index"].astype(int).tolist())
        for episode_index, local_start, local_end in runs:
            summary = summaries.setdefault(episode_index, EpisodeSummary(episode_index=episode_index))
            summary.segments.append(
                EpisodeSegment(
                    video_path=video_path,
                    local_start=local_start,
                    local_end=local_end,
                )
            )

    return summaries, fps


def print_dryrun(summaries: dict[int, EpisodeSummary], fps: float, video_key: str) -> None:
    print(f"video_key: {video_key}")
    print(f"fps: {fps:.2f}")
    print(f"total_episodes: {len(summaries)}")
    for episode_index in sorted(summaries):
        summary = summaries[episode_index]
        duration_s = summary.total_frames / fps
        print(
            f"episode={episode_index:03d} | "
            f"frames={summary.total_frames} | "
            f"duration_s={duration_s:.3f} | "
            f"segments={len(summary.segments)}"
        )


def ensure_ffmpeg(ffmpeg_bin: str) -> None:
    if shutil.which(ffmpeg_bin) is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {ffmpeg_bin}")


def export_segment(
    ffmpeg_bin: str,
    segment: EpisodeSegment,
    fps: float,
    output_path: Path,
    overwrite: bool,
) -> None:
    end_frame_exclusive = segment.local_end + 1
    cmd = [
        ffmpeg_bin,
        "-y" if overwrite else "-n",
        "-i",
        str(segment.video_path),
        "-vf",
        f"trim=start_frame={segment.local_start}:end_frame={end_frame_exclusive},setpts=PTS-STARTPTS",
        "-r",
        str(int(round(fps))),
        "-an",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def concat_segments(ffmpeg_bin: str, segment_paths: list[Path], output_path: Path, overwrite: bool) -> None:
    concat_list = output_path.parent / f"{output_path.stem}_concat.txt"
    try:
        with concat_list.open("w", encoding="utf-8") as file:
            for segment_path in segment_paths:
                file.write(f"file '{segment_path.as_posix()}'\n")
        cmd = [
            ffmpeg_bin,
            "-y" if overwrite else "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        concat_list.unlink(missing_ok=True)


def export_all_episodes(
    summaries: dict[int, EpisodeSummary],
    fps: float,
    ffmpeg_bin: str,
    output_dir: Path,
    overwrite: bool,
) -> None:
    ensure_ffmpeg(ffmpeg_bin)
    output_dir.mkdir(parents=True, exist_ok=True)

    for episode_index in sorted(summaries):
        summary = summaries[episode_index]
        output_path = output_dir / f"episode_{episode_index:03d}.mp4"
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )

        with tempfile.TemporaryDirectory(prefix=f"ep_{episode_index:03d}_", dir=output_dir) as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            segment_paths: list[Path] = []
            for segment_idx, segment in enumerate(summary.segments):
                segment_path = tmp_dir / f"segment_{segment_idx:03d}.mp4"
                export_segment(ffmpeg_bin, segment, fps, segment_path, overwrite=True)
                segment_paths.append(segment_path)

            if len(segment_paths) == 1:
                shutil.move(str(segment_paths[0]), str(output_path))
            else:
                concat_segments(ffmpeg_bin, segment_paths, output_path, overwrite=overwrite)

        duration_s = summary.total_frames / fps
        print(
            f"saved {output_path} | frames={summary.total_frames} | "
            f"duration_s={duration_s:.3f} | segments={len(summary.segments)}"
        )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    info = load_info(dataset_root)
    video_key = discover_video_key(info, args.video_key)
    summaries, fps = collect_episode_summaries(dataset_root, video_key)

    print_dryrun(summaries, fps, video_key)
    if args.dryrun:
        return

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else dataset_root / "episode_videos" / video_key
    )
    export_all_episodes(summaries, fps, args.ffmpeg_bin, output_dir, args.overwrite)


if __name__ == "__main__":
    main()
