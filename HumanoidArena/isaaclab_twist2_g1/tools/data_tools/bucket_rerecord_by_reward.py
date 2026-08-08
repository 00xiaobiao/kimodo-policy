#!/usr/bin/env python3
"""Offline bucket rerecorded samples by max reward."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np


BACKEND_ROOTS = {
    "sonic": ("sonic", "sonic_multicam_rerecord"),
    "twist2": ("twist2", "twist2_multicam_rerecord"),
}

SUMMARY_START_RE = re.compile(r"^\[(?P<index>\d+)/(?P<total>\d+)\] rerecording (?P<source>.+)$")
SUMMARY_SUCCESS_RE = re.compile(
    r"^\s+success -> (?P<output>.+?)"
    r"(?: final_reward=(?P<final_reward>-?\d+(?:\.\d+)?|<missing>))?"
    r"(?: max_reward=(?P<max_reward>-?\d+(?:\.\d+)?|<missing>))?"
    r"(?: any_success=(?P<any_success>true|false|1|0|<missing>))?$"
)
WORKER_FINAL_REWARD_RE = re.compile(r"Final rerecord reward=(?P<reward>-?\d+(?:\.\d+)?)")
WORKER_MAX_REWARD_RE = re.compile(
    r"Max rerecord reward=(?P<reward>-?\d+(?:\.\d+)?) any_success=(?P<any_success>true|false|1|0)"
)
WORKER_RENAME_RE = re.compile(r"Renaming to final file: (?P<filename>[^\s]+\.npz)")
WORKER_STABLE_OUTPUT_RE = re.compile(r"detected stable rerecord output: (?P<path>.+\.npz)")


@dataclass(frozen=True)
class RerecordPair:
    backend: str
    source_npz: Path
    rerecorded_npz: Path
    final_reward: float | None
    max_reward: float | None
    any_success: bool | None
    origin: str
    priority: int


@dataclass(frozen=True)
class PlannedMove:
    source: Path
    destination: Path
    label: str
    root: Path


@dataclass(frozen=True)
class PairStats:
    backend: str
    source_root: Path
    rerecord_root: Path
    mapped_pairs: int
    matched_pairs: int
    missing_source_mappings: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move source/rerecord samples whose rerecorded max reward matches the target "
            "into sibling *_bad directories while preserving subdirectory layout."
        )
    )
    parser.add_argument(
        "dataset_root",
        type=str,
        help="Dataset directory such as isaaclab_twist2_g1/recording_data/HSI_pp_box.",
    )
    parser.add_argument(
        "--backend",
        choices=["all", "sonic", "twist2"],
        default="all",
        help="Which backend family to process.",
    )
    parser.add_argument(
        "--target-reward",
        type=float,
        default=0.0,
        help="Target rerecorded max reward used to identify bad samples.",
    )
    parser.add_argument(
        "--reward-tol",
        type=float,
        default=5e-5,
        help="Absolute tolerance for reward comparison. 5e-5 matches the displayed 0.0000 precision.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually move files. Without this flag the script only prints the plan.",
    )
    return parser.parse_args()


def _parse_float_token(value: object) -> float | None:
    if value in (None, "", "<missing>"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_bool_token(value: object) -> bool | None:
    if value in (None, "", "<missing>"):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    try:
        return bool(int(value))
    except Exception:
        return None


def read_npz_metrics(npz_path: Path) -> tuple[float | None, float | None, bool | None]:
    if not npz_path.is_file():
        return None, None, None
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            def _read_float(key: str) -> float | None:
                if key not in data:
                    return None
                value = np.asarray(data[key], dtype=np.float32).reshape(-1)
                if value.size == 0:
                    return None
                return float(value[0])

            def _read_bool(key: str) -> bool | None:
                if key not in data:
                    return None
                value = np.asarray(data[key]).reshape(-1)
                if value.size == 0:
                    return None
                return bool(value[0])

            final_reward = _read_float("rerecord_final_reward")
            max_reward = _read_float("rerecord_max_reward")
            if max_reward is None:
                max_reward = final_reward
            any_success = _read_bool("rerecord_any_success")
            if any_success is None and max_reward is not None:
                any_success = bool(max_reward > 1e-6)
            return final_reward, max_reward, any_success
    except Exception:
        return None, None, None


def parse_manifest_pairs(manifest_path: Path, backend: str) -> dict[str, RerecordPair]:
    pairs: dict[str, RerecordPair] = {}
    if not manifest_path.is_file():
        return pairs
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if payload.get("status") != "success":
            continue
        source = Path(str(payload.get("source", ""))).resolve()
        rerecorded = Path(str(payload.get("rerecorded_npz", ""))).resolve()
        if not source.name or not rerecorded.name:
            continue
        file_final_reward, file_max_reward, file_any_success = read_npz_metrics(rerecorded)
        final_reward = _parse_float_token(payload.get("rerecord_final_reward"))
        max_reward = _parse_float_token(payload.get("rerecord_max_reward"))
        any_success = _parse_bool_token(payload.get("rerecord_any_success"))
        if final_reward is None:
            final_reward = file_final_reward
        if max_reward is None:
            max_reward = file_max_reward if file_max_reward is not None else final_reward
        if any_success is None:
            any_success = file_any_success
        if any_success is None and max_reward is not None:
            any_success = bool(max_reward > 1e-6)
        pair = RerecordPair(
            backend=backend,
            source_npz=source,
            rerecorded_npz=rerecorded,
            final_reward=final_reward,
            max_reward=max_reward,
            any_success=any_success,
            origin="manifest",
            priority=30,
        )
        pairs[str(source)] = pair
    return pairs


def parse_summary_pairs(summary_path: Path, backend: str) -> dict[str, RerecordPair]:
    pairs: dict[str, RerecordPair] = {}
    if not summary_path.is_file():
        return pairs

    current_source: Path | None = None
    for raw_line in summary_path.read_text(encoding="utf-8").splitlines():
        start_match = SUMMARY_START_RE.match(raw_line)
        if start_match:
            current_source = Path(start_match.group("source")).resolve()
            continue
        success_match = SUMMARY_SUCCESS_RE.match(raw_line)
        if success_match and current_source is not None:
            output = Path(success_match.group("output")).resolve()
            file_final_reward, file_max_reward, file_any_success = read_npz_metrics(output)
            final_reward = _parse_float_token(success_match.group("final_reward"))
            max_reward = _parse_float_token(success_match.group("max_reward"))
            any_success = _parse_bool_token(success_match.group("any_success"))
            if final_reward is None:
                final_reward = file_final_reward
            if max_reward is None:
                max_reward = file_max_reward if file_max_reward is not None else final_reward
            if any_success is None:
                any_success = file_any_success
            if any_success is None and max_reward is not None:
                any_success = bool(max_reward > 1e-6)
            pair = RerecordPair(
                backend=backend,
                source_npz=current_source,
                rerecorded_npz=output,
                final_reward=final_reward,
                max_reward=max_reward,
                any_success=any_success,
                origin="summary",
                priority=10,
            )
            pairs[str(current_source)] = pair
            current_source = None
    return pairs


def _extract_command_value(tokens: list[str], key: str) -> str | None:
    for index, token in enumerate(tokens):
        if token == key and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def parse_worker_log_pair(
    log_path: Path,
    *,
    backend: str,
    source_root: Path,
    rerecord_root: Path,
) -> RerecordPair | None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    command_line = next((line for line in text.splitlines() if line.startswith("COMMAND: ")), "")
    if not command_line:
        return None
    try:
        tokens = shlex.split(command_line[len("COMMAND: ") :])
    except ValueError:
        return None
    replay_file = _extract_command_value(tokens, "--replay_file")
    if replay_file is None:
        return None
    source_npz = Path(replay_file).resolve()

    final_reward_match = WORKER_FINAL_REWARD_RE.search(text)
    max_reward_match = WORKER_MAX_REWARD_RE.search(text)
    final_reward = float(final_reward_match.group("reward")) if final_reward_match is not None else None
    max_reward = float(max_reward_match.group("reward")) if max_reward_match is not None else None
    any_success = _parse_bool_token(max_reward_match.group("any_success")) if max_reward_match is not None else None

    filename_match = WORKER_RENAME_RE.search(text)
    if filename_match is not None:
        rerecord_filename = filename_match.group("filename")
    else:
        stable_match = WORKER_STABLE_OUTPUT_RE.search(text)
        if stable_match is None:
            return None
        rerecord_filename = Path(stable_match.group("path")).name

    try:
        rel_parent = source_npz.relative_to(source_root).parent
    except ValueError:
        return None
    rerecorded_npz = (rerecord_root / rel_parent / rerecord_filename).resolve()
    file_final_reward, file_max_reward, file_any_success = read_npz_metrics(rerecorded_npz)
    if final_reward is None:
        final_reward = file_final_reward
    if max_reward is None:
        max_reward = file_max_reward if file_max_reward is not None else final_reward
    if any_success is None:
        any_success = file_any_success
    if any_success is None and max_reward is not None:
        any_success = bool(max_reward > 1e-6)

    return RerecordPair(
        backend=backend,
        source_npz=source_npz,
        rerecorded_npz=rerecorded_npz,
        final_reward=final_reward,
        max_reward=max_reward,
        any_success=any_success,
        origin=f"worker_log:{log_path.name}",
        priority=20,
    )


def merge_pairs(existing: RerecordPair | None, candidate: RerecordPair) -> tuple[RerecordPair, str | None]:
    if existing is None:
        return candidate, None
    warning = None
    if existing.rerecorded_npz != candidate.rerecorded_npz:
        warning = (
            f"conflicting rerecorded path for {existing.source_npz}: "
            f"{existing.rerecorded_npz} vs {candidate.rerecorded_npz}"
        )
    if existing.final_reward is not None and candidate.final_reward is not None:
        if not math.isclose(existing.final_reward, candidate.final_reward, abs_tol=1e-6):
            warning = (
                f"conflicting final_reward for {existing.source_npz}: "
                f"{existing.final_reward} vs {candidate.final_reward}"
            )
    if existing.max_reward is not None and candidate.max_reward is not None:
        if not math.isclose(existing.max_reward, candidate.max_reward, abs_tol=1e-6):
            warning = (
                f"conflicting max_reward for {existing.source_npz}: "
                f"{existing.max_reward} vs {candidate.max_reward}"
            )
    if (
        existing.any_success is not None
        and candidate.any_success is not None
        and existing.any_success != candidate.any_success
    ):
        warning = (
            f"conflicting any_success for {existing.source_npz}: "
            f"{existing.any_success} vs {candidate.any_success}"
        )
    if candidate.priority > existing.priority:
        merged = replace(
            candidate,
            final_reward=candidate.final_reward if candidate.final_reward is not None else existing.final_reward,
            max_reward=candidate.max_reward if candidate.max_reward is not None else existing.max_reward,
            any_success=candidate.any_success if candidate.any_success is not None else existing.any_success,
        )
        return merged, warning
    if existing.final_reward is None and candidate.final_reward is not None:
        existing = replace(existing, final_reward=candidate.final_reward)
    if existing.max_reward is None and candidate.max_reward is not None:
        existing = replace(existing, max_reward=candidate.max_reward)
    if existing.any_success is None and candidate.any_success is not None:
        existing = replace(existing, any_success=candidate.any_success)
    return existing, warning


def load_backend_pairs(dataset_root: Path, backend: str) -> tuple[list[RerecordPair], PairStats, list[str]]:
    source_root_name, rerecord_root_name = BACKEND_ROOTS[backend]
    source_root = (dataset_root / source_root_name).resolve()
    rerecord_root = (dataset_root / rerecord_root_name).resolve()
    warnings: list[str] = []
    if not source_root.is_dir() or not rerecord_root.is_dir():
        return [], PairStats(backend, source_root, rerecord_root, 0, 0, 0), warnings

    merged: dict[str, RerecordPair] = {}

    if backend == "sonic":
        manifest_path = rerecord_root / "rerecord_manifest.jsonl"
        for pair in parse_manifest_pairs(manifest_path, backend).values():
            merged[str(pair.source_npz)] = pair
    else:
        summary_path = rerecord_root / "rerecord_conversion.log"
        for pair in parse_summary_pairs(summary_path, backend).values():
            merged[str(pair.source_npz)] = pair

    log_dir = rerecord_root / "rerecord_logs"
    if log_dir.is_dir():
        for log_path in sorted(log_dir.glob("*.log")):
            pair = parse_worker_log_pair(
                log_path,
                backend=backend,
                source_root=source_root,
                rerecord_root=rerecord_root,
            )
            if pair is None:
                continue
            current = merged.get(str(pair.source_npz))
            merged_pair, warning = merge_pairs(current, pair)
            if warning is not None:
                warnings.append(warning)
            merged[str(pair.source_npz)] = merged_pair

    source_npz_count = sum(1 for _ in source_root.rglob("*.npz"))
    pairs = sorted(merged.values(), key=lambda pair: str(pair.source_npz))
    stats = PairStats(
        backend=backend,
        source_root=source_root,
        rerecord_root=rerecord_root,
        mapped_pairs=len(pairs),
        matched_pairs=0,
        missing_source_mappings=max(0, source_npz_count - len(pairs)),
    )
    return pairs, stats, warnings


def reward_matches(value: float | None, target_reward: float, tol: float) -> bool:
    if value is None:
        return False
    return math.isclose(float(value), float(target_reward), abs_tol=float(tol))


def bad_root_for(root: Path) -> Path:
    return root.with_name(f"{root.name}_bad")


def collect_matching_video_files(npz_path: Path) -> list[Path]:
    video_dir = npz_path.parent / "videos"
    if not video_dir.is_dir():
        return []
    prefix = f"{npz_path.stem}_"
    return sorted(
        path.resolve()
        for path in video_dir.iterdir()
        if path.is_file() and path.name.startswith(prefix)
    )


def maybe_add_move(
    moves: list[PlannedMove],
    seen_sources: set[Path],
    *,
    source: Path,
    root: Path,
    label: str,
) -> None:
    if source in seen_sources:
        return
    try:
        rel_path = source.relative_to(root)
    except ValueError:
        return
    moves.append(
        PlannedMove(
            source=source,
            destination=bad_root_for(root) / rel_path,
            label=label,
            root=root,
        )
    )
    seen_sources.add(source)


def build_move_plan(
    pairs: Iterable[RerecordPair],
    *,
    target_reward: float,
    reward_tol: float,
) -> tuple[list[PlannedMove], list[RerecordPair]]:
    moves: list[PlannedMove] = []
    matched_pairs: list[RerecordPair] = []
    seen_sources: set[Path] = set()
    for pair in pairs:
        if not reward_matches(pair.max_reward, target_reward, reward_tol):
            continue
        matched_pairs.append(pair)
        source_root_name, rerecord_root_name = BACKEND_ROOTS[pair.backend]
        source_root = pair.source_npz.parents[1] if pair.source_npz.parent.name != "videos" else pair.source_npz.parents[2]
        rerecord_root = (
            pair.rerecorded_npz.parents[1]
            if pair.rerecorded_npz.parent.name != "videos"
            else pair.rerecorded_npz.parents[2]
        )

        while source_root.name != source_root_name and source_root != source_root.parent:
            source_root = source_root.parent
        while rerecord_root.name != rerecord_root_name and rerecord_root != rerecord_root.parent:
            rerecord_root = rerecord_root.parent

        maybe_add_move(
            moves,
            seen_sources,
            source=pair.source_npz,
            root=source_root,
            label=f"{pair.backend}:source_npz",
        )
        for video_path in collect_matching_video_files(pair.source_npz):
            maybe_add_move(
                moves,
                seen_sources,
                source=video_path,
                root=source_root,
                label=f"{pair.backend}:source_video",
            )

        maybe_add_move(
            moves,
            seen_sources,
            source=pair.rerecorded_npz,
            root=rerecord_root,
            label=f"{pair.backend}:rerecord_npz",
        )
        for video_path in collect_matching_video_files(pair.rerecorded_npz):
            maybe_add_move(
                moves,
                seen_sources,
                source=video_path,
                root=rerecord_root,
                label=f"{pair.backend}:rerecord_video",
            )
    return moves, matched_pairs


def cleanup_empty_parents(path: Path, stop_at: Path) -> None:
    current = path.resolve()
    stop_at = stop_at.resolve()
    while current != stop_at and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def apply_moves(moves: Iterable[PlannedMove]) -> tuple[int, list[str]]:
    moved_count = 0
    warnings: list[str] = []
    for move in moves:
        source = move.source.resolve()
        destination = move.destination.resolve()
        if not source.exists():
            if destination.exists():
                warnings.append(f"already moved: {source} -> {destination}")
                continue
            warnings.append(f"missing source: {source}")
            continue
        if destination.exists():
            raise FileExistsError(f"destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved_count += 1

        cleanup_empty_parents(source.parent, move.root)
    return moved_count, warnings


def print_plan(moves: list[PlannedMove], matched_pairs: list[RerecordPair], stats_by_backend: list[PairStats]) -> None:
    print(f"matched_pairs={len(matched_pairs)}")
    print(f"planned_moves={len(moves)}")
    for stats in stats_by_backend:
        print(
            f"[{stats.backend}] mapped_pairs={stats.mapped_pairs} "
            f"missing_source_mappings={stats.missing_source_mappings}"
        )
    for pair in matched_pairs[:10]:
        final_text = "<missing>" if pair.final_reward is None else f"{pair.final_reward:.4f}"
        max_text = "<missing>" if pair.max_reward is None else f"{pair.max_reward:.4f}"
        any_success_text = "<missing>" if pair.any_success is None else str(bool(pair.any_success)).lower()
        print(
            f"  match backend={pair.backend} final_reward={final_text} max_reward={max_text} any_success={any_success_text} "
            f"source={pair.source_npz} rerecord={pair.rerecorded_npz}"
        )
    if len(matched_pairs) > 10:
        print(f"  ... {len(matched_pairs) - 10} more matched pairs")
    for move in moves[:20]:
        print(f"  move[{move.label}] {move.source} -> {move.destination}")
    if len(moves) > 20:
        print(f"  ... {len(moves) - 20} more planned moves")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    backends = list(BACKEND_ROOTS) if args.backend == "all" else [args.backend]
    all_pairs: list[RerecordPair] = []
    stats_by_backend: list[PairStats] = []
    warnings: list[str] = []

    for backend in backends:
        pairs, stats, backend_warnings = load_backend_pairs(dataset_root, backend)
        all_pairs.extend(pairs)
        stats_by_backend.append(stats)
        warnings.extend(backend_warnings)

    moves, matched_pairs = build_move_plan(
        all_pairs,
        target_reward=args.target_reward,
        reward_tol=args.reward_tol,
    )
    stats_by_backend = [
        replace(
            stats,
            matched_pairs=sum(1 for pair in matched_pairs if pair.backend == stats.backend),
        )
        for stats in stats_by_backend
    ]

    print(f"dataset_root={dataset_root}")
    print(f"backend={args.backend}")
    print(f"target_reward={args.target_reward:.4f}")
    print(f"reward_tol={args.reward_tol}")
    print(f"mode={'apply' if args.apply else 'dry-run'}")
    print_plan(moves, matched_pairs, stats_by_backend)

    if warnings:
        print("warnings:")
        for warning in warnings[:20]:
            print(f"  {warning}")
        if len(warnings) > 20:
            print(f"  ... {len(warnings) - 20} more warnings")

    if not args.apply:
        return 0

    moved_count, apply_warnings = apply_moves(moves)
    print(f"moved_files={moved_count}")
    if apply_warnings:
        print("apply_warnings:")
        for warning in apply_warnings[:20]:
            print(f"  {warning}")
        if len(apply_warnings) > 20:
            print(f"  ... {len(apply_warnings) - 20} more apply warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
