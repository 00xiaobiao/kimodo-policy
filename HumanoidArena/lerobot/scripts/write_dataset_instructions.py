#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATASET_TOKENS = ("merged", "sonic", "twist2")


def iter_target_datasets(task_root: Path, dataset_tokens: tuple[str, ...]) -> list[Path]:
    token_set = tuple(token.lower() for token in dataset_tokens)
    dataset_dirs: list[Path] = []
    for child in sorted(task_root.iterdir()):
        if not child.is_dir():
            continue
        if not any(token in child.name.lower() for token in token_set):
            continue
        tasks_path = child / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            continue
        dataset_dirs.append(child)
    return dataset_dirs


def write_instruction(tasks_path: Path, instruction: str) -> None:
    df = pd.DataFrame({"task_index": [0]}, index=pd.Index([instruction], name="task"))
    df.to_parquet(tasks_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline writer for LeRobot task instructions across merged/sonic/twist2 datasets."
    )
    parser.add_argument("task_root", type=Path, help="Task root directory, e.g. /ai/Yichi/taowen/dataset/HOI_double_desk")
    parser.add_argument("instruction", type=str, help="English instruction to write into meta/tasks.parquet")
    parser.add_argument(
        "--dataset-tokens",
        nargs="+",
        default=list(DEFAULT_DATASET_TOKENS),
        help="Subdirectory name tokens to match. Defaults to: merged sonic twist2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which datasets would be updated without modifying files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_root = args.task_root.resolve()
    if not task_root.is_dir():
        raise FileNotFoundError(f"Task root does not exist or is not a directory: {task_root}")

    dataset_dirs = iter_target_datasets(task_root, tuple(args.dataset_tokens))
    if not dataset_dirs:
        print(f"No matching datasets found under {task_root}")
        return 1

    print(f"task_root: {task_root}")
    print(f"instruction: {args.instruction}")
    print(f"dataset_tokens: {tuple(args.dataset_tokens)}")

    for dataset_dir in dataset_dirs:
        tasks_path = dataset_dir / "meta" / "tasks.parquet"
        if args.dry_run:
            print(f"DRY_RUN {tasks_path}")
            continue
        write_instruction(tasks_path, args.instruction)
        print(f"UPDATED {tasks_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
