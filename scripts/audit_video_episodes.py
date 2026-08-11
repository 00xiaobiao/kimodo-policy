#!/usr/bin/env python3
"""Count episodes whose training video interval cannot be decoded by PyAV."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import av

from data.multisource_dataset import (
    EpisodeRecord,
    VIDEO_READ_MAX_ATTEMPTS,
    VIDEO_READ_RETRY_DELAY_SECONDS,
)
from scripts.audit_invalid_episodes import AUDITED_SOURCES, _load_audit_inputs


LOGGER = logging.getLogger("audit_video_episodes")
_WORKER_ACTION_CHUNK = 0


class VideoEpisodeDecodeError(RuntimeError):
    """The relevant interval of an episode video cannot supply a frame."""


def _initialize_worker(action_chunk: int) -> None:
    global _WORKER_ACTION_CHUNK
    _WORKER_ACTION_CHUNK = int(action_chunk)


def _last_training_timestamp(
    episode: EpisodeRecord, action_chunk: int
) -> float:
    last_cut = max(0, episode.target_length - int(action_chunk))
    return episode.video_from_timestamp + last_cut / episode.target_fps


def _decode_interval_once(
    episode: EpisodeRecord, end_timestamp: float
) -> tuple[int, bool]:
    start_timestamp = episode.video_from_timestamp
    expected_interval_frames = max(
        1,
        int(
            round(
                max(0.0, end_timestamp - start_timestamp)
                * episode.source_fps
            )
        )
        + 1,
    )
    with av.open(str(episode.video_path)) as container:
        if not container.streams.video:
            raise VideoEpisodeDecodeError(
                f"No video stream in {episode.video_path}"
            )
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        container.seek(max(0, int(start_timestamp * av.time_base)))
        decoded_frames = 0
        interval_frames = 0
        reached_end = False
        for frame in container.decode(stream):
            decoded_frames += 1
            if frame.pts is None:
                interval_frames += 1
                if interval_frames >= expected_interval_frames:
                    reached_end = True
                    break
                continue
            frame_time = float(frame.pts * stream.time_base)
            if frame_time + 1e-6 < start_timestamp:
                continue
            interval_frames += 1
            if frame_time + 1e-6 >= end_timestamp:
                reached_end = True
                break
        if decoded_frames == 0:
            raise VideoEpisodeDecodeError(
                f"Could not decode any frame from {episode.video_path} "
                f"after seeking to {start_timestamp:.3f}s"
            )
        return decoded_frames, reached_end


def _probe_timestamp_once(video_path: Path, timestamp: float) -> None:
    """Mirror training behavior: any decoded frame after the seek is usable."""
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise VideoEpisodeDecodeError(f"No video stream in {video_path}")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        container.seek(max(0, int(timestamp * av.time_base)))
        for _ in container.decode(stream):
            return
    raise VideoEpisodeDecodeError(
        f"Could not decode a frame from {video_path} after seeking to "
        f"{timestamp:.3f}s"
    )


def _scan_episode_video(episode: EpisodeRecord) -> dict[str, Any]:
    end_timestamp = _last_training_timestamp(
        episode, _WORKER_ACTION_CHUNK
    )
    for attempt in range(VIDEO_READ_MAX_ATTEMPTS):
        try:
            decoded_frames, reached_end = _decode_interval_once(
                episode, end_timestamp
            )
            if not reached_end and end_timestamp > episode.video_from_timestamp:
                _probe_timestamp_once(episode.video_path, end_timestamp)
            return {
                "source": episode.source,
                "status": "valid",
                "decoded_frames": decoded_frames,
            }
        except av.error.BlockingIOError:
            if attempt == VIDEO_READ_MAX_ATTEMPTS - 1:
                raise
            time.sleep(VIDEO_READ_RETRY_DELAY_SECONDS * (2**attempt))
    raise RuntimeError("Unreachable video retry state")


def _episode_context(episode: EpisodeRecord) -> dict[str, Any]:
    return {
        "source": episode.source,
        "episode_id": episode.episode_id,
        "task_id": episode.task_id,
        "task_name": episode.task_name,
        "video_path": str(episode.video_path),
        "video_from_timestamp": episode.video_from_timestamp,
        "last_training_timestamp": _last_training_timestamp(
            episode, _WORKER_ACTION_CHUNK
        ),
    }


def _audit_video_episode(task: tuple[str, EpisodeRecord]) -> dict[str, Any]:
    _, episode = task
    try:
        return _scan_episode_video(episode)
    except (av.error.FFmpegError, VideoEpisodeDecodeError, OSError) as error:
        result = _episode_context(episode)
        result.update(
            {
                "status": "invalid",
                "error_type": type(error).__name__,
                "reason": str(error),
            }
        )
        return result
    except Exception as error:
        result = _episode_context(episode)
        result.update(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "reason": str(error),
            }
        )
        return result


def _parse_args() -> argparse.Namespace:
    default_config = (
        PROJECT_ROOT
        / "scripts"
        / "pt_UnifoLM_HumanoidEveryday_HIW_gbs1024_20w.yaml"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Decode each episode's trainable video interval with PyAV and count "
            "episodes that would fail video loading during training."
        )
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel video decoders (default: min(16, CPU count)).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("video_invalid_episodes.jsonl"),
        help="JSONL file containing video-invalid episodes and scan errors.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Summary JSON path (default: <output>.summary.json).",
    )
    parser.add_argument(
        "--limit-per-source",
        type=int,
        default=None,
        help="Debug only: scan at most this many episodes from each source.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N scanned episodes.",
    )
    return parser.parse_args()


def _print_summary(summary: Mapping[str, Any]) -> None:
    print()
    print(
        f"{'Source':<24} {'Total':>8} {'Usable':>8} "
        f"{'BadVideo':>10} {'Errors':>8} {'BadVideo%':>11}"
    )
    print("-" * 82)
    for source in (*AUDITED_SOURCES, "ALL"):
        values = (
            summary["sources"][source]
            if source != "ALL"
            else summary["overall"]
        )
        print(
            f"{source:<24} {values['total']:>8} {values['valid']:>8} "
            f"{values['invalid']:>10} {values['error']:>8} "
            f"{values['invalid_percent']:>10.3f}%"
        )


def main() -> int:
    args = _parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if args.limit_per_source is not None and args.limit_per_source <= 0:
        raise ValueError("--limit-per-source must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    tasks, _, action_chunk = _load_audit_inputs(
        args.config.resolve(), args.limit_per_source
    )
    if not tasks:
        raise RuntimeError("No episodes were discovered")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or args.output.with_suffix(
        args.output.suffix + ".summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        source: Counter(
            {"total": 0, "valid": 0, "invalid": 0, "error": 0}
        )
        for source in AUDITED_SOURCES
    }
    error_types = {source: Counter() for source in AUDITED_SOURCES}
    started = time.monotonic()
    _initialize_worker(action_chunk)

    if args.workers == 1:
        results = map(_audit_video_episode, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(action_chunk,),
        )
        results = executor.map(_audit_video_episode, tasks, chunksize=1)

    try:
        with args.output.open("w", encoding="utf-8") as output_file:
            for completed, result in enumerate(results, start=1):
                source = result["source"]
                status = result["status"]
                counts[source]["total"] += 1
                counts[source][status] += 1
                if status != "valid":
                    output_file.write(
                        json.dumps(result, ensure_ascii=False) + "\n"
                    )
                    error_types[source][result["error_type"]] += 1
                if completed % args.progress_every == 0 or completed == len(tasks):
                    elapsed = max(time.monotonic() - started, 1e-6)
                    LOGGER.info(
                        "Progress %d/%d (%.2f episodes/s)",
                        completed,
                        len(tasks),
                        completed / elapsed,
                    )
    finally:
        if executor is not None:
            executor.shutdown()

    source_summary: dict[str, Any] = {}
    overall = Counter({"total": 0, "valid": 0, "invalid": 0, "error": 0})
    for source in AUDITED_SOURCES:
        source_counts = counts[source]
        total = source_counts["total"]
        source_summary[source] = {
            **dict(source_counts),
            "invalid_percent": 100.0 * source_counts["invalid"] / max(total, 1),
            "error_types": dict(sorted(error_types[source].items())),
        }
        overall.update(source_counts)
    overall_summary = {
        **dict(overall),
        "invalid_percent": 100.0 * overall["invalid"] / max(overall["total"], 1),
    }
    summary = {
        "config": str(args.config.resolve()),
        "workers": args.workers,
        "elapsed_seconds": time.monotonic() - started,
        "sources": source_summary,
        "overall": overall_summary,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _print_summary(summary)
    print(f"\nBad-video/error details: {args.output.resolve()}")
    print(f"Summary JSON: {summary_path.resolve()}")
    if overall["error"]:
        print(
            f"Warning: {overall['error']} episodes had unexpected scan errors; "
            "inspect status=error entries in the JSONL file."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
