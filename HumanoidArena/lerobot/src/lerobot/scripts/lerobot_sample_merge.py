#!/usr/bin/env python3

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Randomly sample episodes from two local LeRobot datasets, then merge them.

Usage example:

```shell
lerobot-sample-merge \
    --source-a /path/to/dataset_a \
    --source-b /path/to/dataset_b \
    --output-dir /path/to/output_dataset \
    --sample-ratio 0.5 \
    --seed 42
```
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.datasets.dataset_tools import merge_datasets, split_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


@dataclass(slots=True)
class DatasetSamplePlan:
    label: str
    root: Path
    repo_id: str
    fps: int
    robot_type: str | None
    total_episodes: int
    selected_episode_indices: list[int]

    @property
    def selected_episode_count(self) -> int:
        return len(self.selected_episode_indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-a",
        type=Path,
        required=True,
        help="Path to the first local LeRobot dataset root containing meta/, data/, and videos/.",
    )
    parser.add_argument(
        "--source-b",
        type=Path,
        required=True,
        help="Path to the second local LeRobot dataset root containing meta/, data/, and videos/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output path for the merged LeRobot dataset.",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help="Repo id written into the merged dataset metadata. Defaults to the output directory name.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.5,
        help="Fraction of episodes to sample from each source dataset. Defaults to 0.5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for episode sampling. Defaults to 42.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before writing the merged dataset.",
    )
    return parser.parse_args()


def _load_local_dataset(dataset_root: Path, label: str) -> LeRobotDataset:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {dataset_root}")

    for required_dir in ("meta", "data"):
        required_path = dataset_root / required_dir
        if not required_path.is_dir():
            raise FileNotFoundError(f"{label} is missing required directory: {required_path}")

    return LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)


def _feature_differences(
    left_features: dict[str, Any], right_features: dict[str, Any]
) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    left_keys = set(left_features)
    right_keys = set(right_features)

    only_left = sorted(left_keys - right_keys)
    only_right = sorted(right_keys - left_keys)

    mismatched: dict[str, dict[str, Any]] = {}
    for key in sorted(left_keys & right_keys):
        if left_features[key] != right_features[key]:
            mismatched[key] = {
                "source_a": left_features[key],
                "source_b": right_features[key],
            }

    return only_left, only_right, mismatched


def validate_pair_is_mergeable(dataset_a: LeRobotDataset, dataset_b: LeRobotDataset) -> None:
    issues: list[str] = []

    if dataset_a.meta.fps != dataset_b.meta.fps:
        issues.append(f"fps mismatch: {dataset_a.meta.fps} vs {dataset_b.meta.fps}")

    if dataset_a.meta.robot_type != dataset_b.meta.robot_type:
        issues.append(f"robot_type mismatch: {dataset_a.meta.robot_type} vs {dataset_b.meta.robot_type}")

    only_a, only_b, mismatched = _feature_differences(dataset_a.meta.features, dataset_b.meta.features)
    if only_a:
        issues.append(f"feature keys only in source_a: {only_a}")
    if only_b:
        issues.append(f"feature keys only in source_b: {only_b}")
    if mismatched:
        feature_lines = [f"feature metadata mismatch for keys: {sorted(mismatched)}"]
        for key, diff in mismatched.items():
            feature_lines.append(
                f"  {key}: source_a={json.dumps(diff['source_a'], ensure_ascii=True)} "
                f"source_b={json.dumps(diff['source_b'], ensure_ascii=True)}"
            )
        issues.extend(feature_lines)

    if issues:
        raise ValueError(
            "Source datasets are not compatible with LeRobot's official merge pipeline:\n"
            + "\n".join(f"- {issue}" for issue in issues)
        )


def compute_sample_count(total_episodes: int, sample_ratio: float) -> int:
    if total_episodes <= 0:
        raise ValueError("Source dataset must contain at least one episode")
    if not 0 < sample_ratio <= 1:
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")

    return min(total_episodes, max(1, math.floor(total_episodes * sample_ratio + 0.5)))


def select_episode_indices(total_episodes: int, sample_ratio: float, rng: random.Random) -> list[int]:
    sample_count = compute_sample_count(total_episodes, sample_ratio)
    return sorted(rng.sample(range(total_episodes), k=sample_count))


def _build_sample_plan(
    dataset: LeRobotDataset,
    label: str,
    sample_ratio: float,
    rng: random.Random,
) -> DatasetSamplePlan:
    return DatasetSamplePlan(
        label=label,
        root=dataset.root,
        repo_id=dataset.repo_id,
        fps=dataset.meta.fps,
        robot_type=dataset.meta.robot_type,
        total_episodes=dataset.meta.total_episodes,
        selected_episode_indices=select_episode_indices(dataset.meta.total_episodes, sample_ratio, rng),
    )


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)


def _write_sampling_manifest(
    output_dir: Path,
    output_repo_id: str,
    sample_ratio: float,
    seed: int,
    plan_a: DatasetSamplePlan,
    plan_b: DatasetSamplePlan,
) -> None:
    manifest = {
        "output_repo_id": output_repo_id,
        "sample_ratio": sample_ratio,
        "seed": seed,
        "sources": {
            plan.label: {
                "root": str(plan.root),
                "repo_id": plan.repo_id,
                "fps": plan.fps,
                "robot_type": plan.robot_type,
                "total_episodes": plan.total_episodes,
                "selected_episode_count": plan.selected_episode_count,
                "selected_episode_indices": plan.selected_episode_indices,
            }
            for plan in (plan_a, plan_b)
        },
    }

    manifest_path = output_dir / "sampling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sample_and_merge_two_datasets(
    source_a_root: Path,
    source_b_root: Path,
    output_dir: Path,
    output_repo_id: str | None = None,
    sample_ratio: float = 0.5,
    seed: int = 42,
    overwrite: bool = False,
) -> LeRobotDataset:
    dataset_a = _load_local_dataset(source_a_root, "source_a")
    dataset_b = _load_local_dataset(source_b_root, "source_b")
    validate_pair_is_mergeable(dataset_a, dataset_b)

    output_repo_id = output_repo_id or output_dir.name
    _prepare_output_dir(output_dir, overwrite)

    rng = random.Random(seed)
    plan_a = _build_sample_plan(dataset_a, "source_a", sample_ratio, rng)
    plan_b = _build_sample_plan(dataset_b, "source_b", sample_ratio, rng)

    logging.info(
        "Sampling %s: %s/%s episodes",
        plan_a.root,
        plan_a.selected_episode_count,
        plan_a.total_episodes,
    )
    logging.info(
        "Sampling %s: %s/%s episodes",
        plan_b.root,
        plan_b.selected_episode_count,
        plan_b.total_episodes,
    )

    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}_tmp_", dir=output_dir.parent) as tmp_dir:
        temp_root = Path(tmp_dir)

        subset_a = split_dataset(
            dataset_a,
            splits={"selected": plan_a.selected_episode_indices},
            output_dir=temp_root / "source_a",
        )["selected"]
        subset_b = split_dataset(
            dataset_b,
            splits={"selected": plan_b.selected_episode_indices},
            output_dir=temp_root / "source_b",
        )["selected"]

        merged_dataset = merge_datasets(
            datasets=[subset_a, subset_b],
            output_repo_id=output_repo_id,
            output_dir=output_dir,
        )

    _write_sampling_manifest(output_dir, output_repo_id, sample_ratio, seed, plan_a, plan_b)
    return merged_dataset


def main() -> None:
    args = parse_args()
    init_logging()
    merged_dataset = sample_and_merge_two_datasets(
        source_a_root=args.source_a,
        source_b_root=args.source_b,
        output_dir=args.output_dir,
        output_repo_id=args.output_repo_id,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    manifest_path = merged_dataset.root / "sampling_manifest.json"
    logging.info("Merged dataset written to %s", merged_dataset.root)
    logging.info(
        "Created merged dataset with %s episodes and %s frames",
        merged_dataset.meta.total_episodes,
        merged_dataset.meta.total_frames,
    )
    logging.info("Sampling manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
