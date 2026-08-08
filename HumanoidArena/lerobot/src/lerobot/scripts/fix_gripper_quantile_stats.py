#!/usr/bin/env python3

"""Patch LeRobot dataset stats for hand-binary action dimensions.

This script recursively scans a directory tree, finds local LeRobot datasets,
and rewrites ``meta/stats.json`` for ``action.hand_binary.left`` and
``action.hand_binary.right`` so their normalization stays stable across all
current humanoid training policies.

It always applies the following fixed binary stats:
- q01 = 0.0
- q99 = 1.0
- min = 0.0
- max = 1.0
- mean = 0.5
- std = 0.5

This keeps raw parquet values unchanged at ``0/1`` while making:
- QUANTILES map to ``-1/1``
- MIN_MAX map to ``-1/1``
- MEAN_STD map to ``-1/1``

Example:

```bash
python src/lerobot/scripts/fix_gripper_quantile_stats.py \
    --root /ai/Yichi/taowen/dataset
```
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_PREFIXES = ("HOI_", "HSI_")
HAND_BINARY_FEATURES = ("action.hand_binary.left", "action.hand_binary.right")
DEFAULT_BACKUP_SUFFIX = ".gripper_fix.bak"
DEFAULT_TARGET_VALUES = {
    "q01": 0.0,
    "q99": 1.0,
    "min": 0.0,
    "max": 1.0,
    "mean": 0.5,
    "std": 0.5,
}


@dataclass(slots=True)
class PatchResult:
    dataset_root: Path
    stats_path: Path
    modified: bool
    reason: str
    changes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory to recursively scan for LeRobot datasets.",
    )
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=None,
        help=(
            "Only patch datasets whose path contains a directory starting with this prefix. "
            f"Defaults to {DEFAULT_PREFIXES}. Repeatable."
        ),
    )
    parser.add_argument(
        "--q01",
        type=float,
        default=DEFAULT_TARGET_VALUES["q01"],
        help="Replacement q01 value for hand-binary dimensions. Defaults to 0.0.",
    )
    parser.add_argument(
        "--q99",
        type=float,
        default=DEFAULT_TARGET_VALUES["q99"],
        help="Replacement q99 value for hand-binary dimensions. Defaults to 1.0.",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=DEFAULT_TARGET_VALUES["min"],
        help="Replacement min value for hand-binary dimensions. Defaults to 0.0.",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=DEFAULT_TARGET_VALUES["max"],
        help="Replacement max value for hand-binary dimensions. Defaults to 1.0.",
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=DEFAULT_TARGET_VALUES["mean"],
        help="Replacement mean value for hand-binary dimensions. Defaults to 0.5.",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=DEFAULT_TARGET_VALUES["std"],
        help="Replacement std value for hand-binary dimensions. Defaults to 0.5.",
    )
    parser.add_argument(
        "--backup",
        dest="backup",
        action="store_true",
        help="Create a backup copy of stats.json before modifying it.",
    )
    parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help="Do not create a backup copy.",
    )
    parser.set_defaults(backup=True)
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=DEFAULT_BACKUP_SUFFIX,
        help=(
            "Suffix appended to stats.json for backups. "
            f"Defaults to {DEFAULT_BACKUP_SUFFIX!r}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose per-dataset logging.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")


def iter_dataset_roots(search_root: Path, include_prefixes: tuple[str, ...]) -> Iterable[Path]:
    seen: set[Path] = set()
    for info_path in sorted(search_root.rglob("meta/info.json")):
        dataset_root = info_path.parent.parent
        stats_path = dataset_root / "meta" / "stats.json"
        if not stats_path.is_file():
            continue
        if dataset_root in seen:
            continue
        if include_prefixes and not any(
            part.startswith(prefix) for part in dataset_root.parts for prefix in include_prefixes
        ):
            continue
        seen.add(dataset_root)
        yield dataset_root


def find_hand_binary_indices(action_names: list[str]) -> tuple[int, int] | None:
    try:
        left_idx = action_names.index(HAND_BINARY_FEATURES[0])
        right_idx = action_names.index(HAND_BINARY_FEATURES[1])
    except ValueError:
        return None
    return left_idx, right_idx


def ensure_stat_length(stats: dict, stat_key: str, required_index: int) -> list[float] | None:
    values = stats.get(stat_key)
    if not isinstance(values, list):
        return None
    if len(values) <= required_index:
        return None
    return values


def maybe_set_value(values: list[float], index: int, new_value: float, label: str, changes: list[str]) -> None:
    old_value = values[index]
    if old_value == new_value:
        return
    values[index] = new_value
    changes.append(f"{label}[{index}] {old_value!r} -> {new_value!r}")


def patch_single_dataset(
    dataset_root: Path,
    *,
    targets: dict[str, float],
    backup: bool,
    backup_suffix: str,
    dry_run: bool,
) -> PatchResult:
    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "stats.json"

    info = load_json(info_path)
    stats = load_json(stats_path)

    action_feature = info.get("features", {}).get("action", {})
    action_names = action_feature.get("names")
    if not isinstance(action_names, list):
        return PatchResult(dataset_root, stats_path, False, "missing_action_names", [])

    indices = find_hand_binary_indices(action_names)
    if indices is None:
        return PatchResult(dataset_root, stats_path, False, "no_hand_binary_features", [])
    left_idx, right_idx = indices

    action_stats = stats.get("action")
    if not isinstance(action_stats, dict):
        return PatchResult(dataset_root, stats_path, False, "missing_action_stats", [])

    required_index = max(left_idx, right_idx)
    stat_arrays: dict[str, list[float]] = {}
    for stat_key in targets:
        values = ensure_stat_length(action_stats, stat_key, required_index)
        if values is None:
            return PatchResult(dataset_root, stats_path, False, f"missing_action_{stat_key}", [])
        stat_arrays[stat_key] = values

    changes: list[str] = []
    for idx in (left_idx, right_idx):
        for stat_key, new_value in targets.items():
            maybe_set_value(stat_arrays[stat_key], idx, new_value, f"action.{stat_key}", changes)

    if not changes:
        return PatchResult(dataset_root, stats_path, False, "already_patched", [])

    if dry_run:
        return PatchResult(dataset_root, stats_path, True, "dry_run", changes)

    if backup:
        backup_path = stats_path.with_name(stats_path.name + backup_suffix)
        if not backup_path.exists():
            shutil.copy2(stats_path, backup_path)

    dump_json(stats_path, stats)
    return PatchResult(dataset_root, stats_path, True, "patched", changes)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root does not exist or is not a directory: {root}")
    if args.q99 == args.q01:
        raise ValueError("q99 must be different from q01")
    if args.max_value == args.min_value:
        raise ValueError("max-value must be different from min-value")
    if args.std <= 0:
        raise ValueError("std must be positive")

    targets = {
        "q01": args.q01,
        "q99": args.q99,
        "min": args.min_value,
        "max": args.max_value,
        "mean": args.mean,
        "std": args.std,
    }

    include_prefixes = tuple(args.include_prefix or DEFAULT_PREFIXES)
    dataset_roots = list(iter_dataset_roots(root, include_prefixes=include_prefixes))
    if not dataset_roots:
        logging.warning("No matching datasets found under %s", root)
        return

    logging.info("Found %d dataset(s) under %s", len(dataset_roots), root)

    scanned = 0
    modified = 0
    skipped = 0
    for dataset_root in dataset_roots:
        scanned += 1
        result = patch_single_dataset(
            dataset_root,
            targets=targets,
            backup=args.backup,
            backup_suffix=args.backup_suffix,
            dry_run=args.dry_run,
        )

        if result.modified:
            modified += 1
            status = result.reason.upper()
            logging.info("[%s] %s", status, result.dataset_root)
            for change in result.changes:
                logging.info("  %s", change)
        else:
            skipped += 1
            logging.debug("[SKIP:%s] %s", result.reason, result.dataset_root)

    logging.info(
        "Done. scanned=%d modified=%d skipped=%d dry_run=%s",
        scanned,
        modified,
        skipped,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
