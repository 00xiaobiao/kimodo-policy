#!/usr/bin/env python3
"""Audit the three pretraining datasets with the training quality rules."""

from __future__ import annotations

import argparse
import json
import logging
import math
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

import numpy as np
from omegaconf import OmegaConf

from data.multisource_dataset import (
    ADAPTER_BY_SOURCE,
    EpisodeRecord,
    ParquetEpisodeReader,
    SOURCE_HIW500,
    SOURCE_HUMANOID_EVERYDAY,
    SOURCE_UNIFOLM,
    UNIFOLM_FIRST_ROOT_JUMP_THRESHOLD_METERS,
    UNIFOLM_INTERNAL_ROOT_JUMP_THRESHOLD_METERS,
    UNIFOLM_JOINT_JUMP_THRESHOLD_RADIANS,
    UNIFOLM_MAX_INITIAL_TRIM_FRAMES,
    UNIFOLM_ROOT_ROTATION_JUMP_THRESHOLD_DEGREES,
    _as_matrix,
    _hiw_joint_discontinuity,
    _unifolm_motion_discontinuity,
)


LOGGER = logging.getLogger("audit_invalid_episodes")
AUDITED_SOURCES = (
    SOURCE_UNIFOLM,
    SOURCE_HUMANOID_EVERYDAY,
    SOURCE_HIW500,
)

_WORKER_SELECTIONS: dict[str, dict[str, Any]] = {}
_WORKER_ACTION_CHUNK = 0
_WORKER_READER: ParquetEpisodeReader | None = None


def _initialize_worker(
    selections: Mapping[str, Mapping[str, Any]], action_chunk: int
) -> None:
    global _WORKER_SELECTIONS, _WORKER_ACTION_CHUNK, _WORKER_READER
    _WORKER_SELECTIONS = {
        source: dict(selection) for source, selection in selections.items()
    }
    _WORKER_ACTION_CHUNK = int(action_chunk)
    _WORKER_READER = ParquetEpisodeReader()


def _motion_discontinuity(
    current: np.ndarray,
    desired: np.ndarray,
    selection: Mapping[str, Any],
) -> tuple[int, dict[str, float | int] | None]:
    return _unifolm_motion_discontinuity(
        current,
        desired,
        first_root_jump_threshold=selection.get(
            "first_root_jump_threshold",
            UNIFOLM_FIRST_ROOT_JUMP_THRESHOLD_METERS,
        ),
        root_jump_threshold=selection.get(
            "internal_root_jump_threshold",
            UNIFOLM_INTERNAL_ROOT_JUMP_THRESHOLD_METERS,
        ),
        joint_jump_threshold=selection.get(
            "joint_jump_threshold_radians",
            UNIFOLM_JOINT_JUMP_THRESHOLD_RADIANS,
        ),
        root_rotation_jump_threshold_degrees=selection.get(
            "root_rotation_jump_threshold_degrees",
            UNIFOLM_ROOT_ROTATION_JUMP_THRESHOLD_DEGREES,
        ),
        max_initial_trim_frames=selection.get(
            "max_initial_trim_frames", UNIFOLM_MAX_INITIAL_TRIM_FRAMES
        ),
    )


def _remaining_target_length(episode: EpisodeRecord, frame_offset: int) -> int:
    source_length = episode.source_length - int(frame_offset)
    if source_length <= 0:
        return 0
    return int(
        round((source_length - 1) * episode.target_fps / episode.source_fps)
    ) + 1


def _episode_context(episode: EpisodeRecord) -> dict[str, Any]:
    return {
        "source": episode.source,
        "episode_id": episode.episode_id,
        "task_id": episode.task_id,
        "task_name": episode.task_name,
        "data_path": str(episode.data_path),
        "video_path": str(episode.video_path),
    }


def _json_safe_metrics(issue: Mapping[str, float | int]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, value in issue.items():
        if isinstance(value, float) and not math.isfinite(value):
            metrics[key] = str(value)
        else:
            metrics[key] = value
    return metrics


def _invalid_result(
    episode: EpisodeRecord,
    reason: str,
    *,
    frame_offset: int = 0,
    issue: Mapping[str, float | int] | None = None,
) -> dict[str, Any]:
    result = _episode_context(episode)
    result.update(
        {
            "status": "invalid",
            "reason": reason,
            "frame_offset": int(frame_offset),
        }
    )
    if issue is not None:
        result["metrics"] = _json_safe_metrics(issue)
    return result


def _motion_issue_reason(issue: Mapping[str, float | int]) -> str:
    transition_frame = int(issue["transition_frame"])
    return (
        "motion discontinuity at frames "
        f"{transition_frame}->{transition_frame + 1}: "
        f"root={float(issue['root_jump']):.3f} m, "
        f"joint={float(issue['joint_jump']):.3f} rad, "
        "root_rotation="
        f"{float(issue['root_rotation_jump_degrees']):.1f} deg"
    )


def _joint_issue_reason(issue: Mapping[str, float | int]) -> str:
    transition_frame = int(issue["transition_frame"])
    return (
        "joint discontinuity at frames "
        f"{transition_frame}->{transition_frame + 1}: "
        f"joint={float(issue['joint_jump']):.3f} rad"
    )


def _audit_episode(task: tuple[str, EpisodeRecord]) -> dict[str, Any]:
    source, episode = task
    selection = _WORKER_SELECTIONS[source]
    reader = _WORKER_READER
    if reader is None:
        raise RuntimeError("Audit worker was not initialized")

    try:
        if source == SOURCE_UNIFOLM:
            table = reader.read(
                episode,
                [
                    "observation.state.robot_q_current",
                    "action.robot_q_desired",
                    "observation.state.hand_state",
                    "action.hand_cmd",
                ],
            )
            current = _as_matrix(
                table["observation.state.robot_q_current"],
                36,
                "UnifoLM current q",
            )
            desired = _as_matrix(
                table["action.robot_q_desired"], 36, "UnifoLM desired q"
            )
            hand_width = 2 if episode.metadata["hand_type"] == "dex1" else 12
            _as_matrix(
                table["observation.state.hand_state"],
                hand_width,
                "UnifoLM hand state",
            )
            _as_matrix(
                table["action.hand_cmd"],
                hand_width,
                "UnifoLM hand command",
            )
            frame_offset, issue = _motion_discontinuity(
                current, desired, selection
            )
            if issue is not None:
                return _invalid_result(
                    episode,
                    _motion_issue_reason(issue),
                    frame_offset=frame_offset,
                    issue=issue,
                )

        elif source == SOURCE_HUMANOID_EVERYDAY:
            table = reader.read(
                episode,
                [
                    "observation.arm_joints",
                    "observation.leg_joints",
                    "observation.hand_joints",
                    "observation.odometry.position",
                    "observation.odometry.quat",
                    "action",
                ],
            )
            arm = _as_matrix(
                table["observation.arm_joints"],
                14,
                "HumanoidEveryday arm",
            )
            leg = _as_matrix(
                table["observation.leg_joints"],
                15,
                "HumanoidEveryday leg",
            )
            _as_matrix(
                table["observation.hand_joints"],
                14,
                "HumanoidEveryday hand",
            )
            action = _as_matrix(
                table["action"], 28, "HumanoidEveryday action"
            )
            root_position = _as_matrix(
                table["observation.odometry.position"],
                3,
                "HumanoidEveryday odometry position",
            )
            root_quaternion = _as_matrix(
                table["observation.odometry.quat"],
                4,
                "HumanoidEveryday odometry quaternion",
            )
            observed_q = np.concatenate((leg, arm), axis=1)
            target_q = np.concatenate((leg, action[:, 14:28]), axis=1)
            current = np.concatenate(
                (root_position, root_quaternion, observed_q), axis=1
            )
            desired = np.concatenate(
                (root_position, root_quaternion, target_q), axis=1
            )
            frame_offset, issue = _motion_discontinuity(
                current, desired, selection
            )
            if issue is not None:
                return _invalid_result(
                    episode,
                    _motion_issue_reason(issue),
                    frame_offset=frame_offset,
                    issue=issue,
                )

        elif source == SOURCE_HIW500:
            table = reader.read(
                episode,
                ["observation.state", "observation.state.wbc", "action"],
            )
            joint_q = _as_matrix(
                table["observation.state"], 29, "HIW joint state"
            )
            _as_matrix(
                table["observation.state.wbc"], 23, "HIW WBC state"
            )
            _as_matrix(table["action"], 23, "HIW action")
            frame_offset, issue = _hiw_joint_discontinuity(
                joint_q,
                joint_jump_threshold=selection.get(
                    "joint_jump_threshold_radians",
                    UNIFOLM_JOINT_JUMP_THRESHOLD_RADIANS,
                ),
                max_initial_trim_frames=selection.get(
                    "max_initial_trim_frames", UNIFOLM_MAX_INITIAL_TRIM_FRAMES
                ),
            )
            if issue is not None:
                return _invalid_result(
                    episode,
                    _joint_issue_reason(issue),
                    frame_offset=frame_offset,
                    issue=issue,
                )
        else:
            raise ValueError(f"Unsupported audit source: {source}")

        if _remaining_target_length(episode, frame_offset) < _WORKER_ACTION_CHUNK:
            return _invalid_result(
                episode,
                "episode is too short after initial reset trimming",
                frame_offset=frame_offset,
            )
        return {"source": source, "status": "valid"}
    except Exception as error:
        result = _episode_context(episode)
        result.update(
            {
                "status": "error",
                "reason": f"{type(error).__name__}: {error}",
            }
        )
        return result


def _reason_category(reason: str) -> str:
    for category in (
        "motion discontinuity",
        "joint discontinuity",
        "episode is too short",
    ):
        if reason.startswith(category):
            return category
    return "other"


def _parse_args() -> argparse.Namespace:
    default_config = (
        PROJECT_ROOT
        / "scripts"
        / "pt_UnifoLM_HumanoidEveryday_HIW_gbs1024_20w.yaml"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Count episodes rejected by the exact motion-continuity rules used "
            "by the three-source pretraining dataset. Video decoding is not checked."
        )
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel parquet readers (default: min(16, CPU count)).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("invalid_episodes.jsonl"),
        help="JSONL file containing invalid episodes and read errors.",
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
        help="Debug only: audit at most this many episodes from each source.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N audited episodes.",
    )
    return parser.parse_args()


def _load_audit_inputs(
    config_path: Path, limit_per_source: int | None
) -> tuple[
    list[tuple[str, EpisodeRecord]], dict[str, dict[str, Any]], int
]:
    config = OmegaConf.load(config_path)
    roots = OmegaConf.to_container(config.main.dataset_roots, resolve=True)
    selections = OmegaConf.to_container(
        config.main.dataset_selection, resolve=True
    )
    if not isinstance(roots, dict) or not isinstance(selections, dict):
        raise TypeError("dataset_roots and dataset_selection must be mappings")

    action_chunk = int(config.main.action_chunk)
    target_fps = float(config.model.fps)
    tasks: list[tuple[str, EpisodeRecord]] = []
    normalized_selections: dict[str, dict[str, Any]] = {}
    for source in AUDITED_SOURCES:
        if source not in roots:
            raise KeyError(f"Missing dataset root for {source}")
        root = Path(str(roots[source])).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"{source} dataset root does not exist: {root}")
        selection = dict(selections.get(source, {}))
        adapter = ADAPTER_BY_SOURCE[source](
            root=root,
            selection=selection,
            target_fps=target_fps,
            action_chunk=action_chunk,
        )
        episodes = adapter.episodes
        if limit_per_source is not None:
            episodes = episodes[:limit_per_source]
        normalized_selections[source] = selection
        tasks.extend((source, episode) for episode in episodes)
        LOGGER.info("Discovered %s episodes for %s", len(episodes), source)
    return tasks, normalized_selections, action_chunk


def _print_summary(summary: Mapping[str, Any]) -> None:
    print()
    print(
        f"{'Source':<24} {'Total':>8} {'Valid':>8} "
        f"{'Invalid':>9} {'Errors':>8} {'Invalid%':>10}"
    )
    print("-" * 79)
    for source in (*AUDITED_SOURCES, "ALL"):
        values = summary["sources"][source] if source != "ALL" else summary["overall"]
        print(
            f"{source:<24} {values['total']:>8} {values['valid']:>8} "
            f"{values['invalid']:>9} {values['error']:>8} "
            f"{values['invalid_percent']:>9.3f}%"
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
    tasks, selections, action_chunk = _load_audit_inputs(
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
        source: Counter({"total": 0, "valid": 0, "invalid": 0, "error": 0})
        for source in AUDITED_SOURCES
    }
    reasons = {source: Counter() for source in AUDITED_SOURCES}
    started = time.monotonic()
    _initialize_worker(selections, action_chunk)

    if args.workers == 1:
        results = map(_audit_episode, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(selections, action_chunk),
        )
        results = executor.map(_audit_episode, tasks, chunksize=8)

    try:
        with args.output.open("w", encoding="utf-8") as output_file:
            for completed, result in enumerate(results, start=1):
                source = result["source"]
                status = result["status"]
                counts[source]["total"] += 1
                counts[source][status] += 1
                if status != "valid":
                    output_file.write(
                        json.dumps(result, ensure_ascii=False, allow_nan=False)
                        + "\n"
                    )
                    reasons[source][_reason_category(result["reason"])] += 1
                if completed % args.progress_every == 0 or completed == len(tasks):
                    elapsed = max(time.monotonic() - started, 1e-6)
                    LOGGER.info(
                        "Progress %d/%d (%.1f episodes/s)",
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
            "reasons": dict(sorted(reasons[source].items())),
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
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    _print_summary(summary)
    print(f"\nInvalid/error details: {args.output.resolve()}")
    print(f"Summary JSON: {summary_path.resolve()}")
    if overall["error"]:
        print(
            f"Warning: {overall['error']} episodes could not be audited; "
            "inspect status=error entries in the JSONL file."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
