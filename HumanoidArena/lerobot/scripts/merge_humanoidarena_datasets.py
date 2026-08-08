#!/ai/Yichi/0_Systems/miniconda3/envs/lerobot2/bin/python
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LEROBOT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = LEROBOT_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.io_utils import load_info, load_stats, write_info, write_stats, write_tasks


DEFAULT_DATASETS_ROOT = Path(
    "/ai/Yichi/taowen/dataset_v3/HumanoidArena_datasets_v3"
)
DEFAULT_OUTPUT_BASE = Path(
    "/ai/Yichi/taowen/dataset_v3/HumanoidArena_merged_datasets_v3"
)
DEFAULT_REPO_PREFIX = "local"


@dataclass(frozen=True)
class SourceDataset:
    path: Path
    task_name: str
    dataset_name: str
    info: dict[str, Any]
    tasks: pd.DataFrame
    episodes: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge HumanoidArena LeRobot v3 datasets for VLA training."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "sonic", "twist2", "csv"],
        required=True,
        help="all: merge all 16 datasets; sonic/twist2: merge that family; csv: merge paths from CSV.",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT,
        help=f"Root containing task folders. Default: {DEFAULT_DATASETS_ROOT}",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="CSV file used by --mode csv. Supports a path column or a single-column CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Exact output dataset folder. Defaults under HumanoidArena_merged_datasets_v2.",
    )
    parser.add_argument(
        "--repo-id",
        help="Repo id to print for training. Default: local/<output folder name>.",
    )
    parser.add_argument(
        "--video-mode",
        choices=["symlink", "copy", "hardlink"],
        default="symlink",
        help="How to materialize videos in the merged dataset. Default: symlink.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output-root first if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the merge plan.",
    )
    return parser.parse_args()


def default_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root
    names = {
        "all": "all_16_rotlocal_v3",
        "sonic": "sonic_8_rotlocal_v3",
        "twist2": "twist2_8_rotlocal_v3",
    }
    if args.mode == "csv":
        stem = args.csv.stem if args.csv is not None else "csv"
        return DEFAULT_OUTPUT_BASE / f"{stem}_merged_rotlocal_v3"
    return DEFAULT_OUTPUT_BASE / names[args.mode]


def normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()


def validate_dataset_dir(path: Path) -> None:
    required = [
        path / "meta" / "info.json",
        path / "meta" / "tasks.parquet",
        path / "meta" / "stats.json",
        path / "meta" / "episodes",
        path / "data",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{path} is not a complete LeRobot dataset; missing: {missing}")


def read_episode_files(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'meta' / 'episodes'}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def load_source(path: Path, datasets_root: Path) -> SourceDataset:
    path = normalize_path(path)
    validate_dataset_dir(path)
    try:
        rel = path.relative_to(datasets_root.resolve())
        task_name = rel.parts[0] if len(rel.parts) >= 2 else path.parent.name
    except ValueError:
        task_name = path.parent.name
    return SourceDataset(
        path=path,
        task_name=task_name,
        dataset_name=path.name,
        info=load_info(path),
        tasks=pd.read_parquet(path / "meta" / "tasks.parquet"),
        episodes=read_episode_files(path),
    )


def discover_datasets(args: argparse.Namespace) -> list[Path]:
    root = normalize_path(args.datasets_root)
    if not root.is_dir():
        raise FileNotFoundError(f"datasets root does not exist: {root}")

    paths: list[Path] = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for dataset_dir in sorted(p for p in task_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            if not (dataset_dir / "meta" / "info.json").exists():
                continue
            name = dataset_dir.name.lower()
            if args.mode == "all":
                if name in {"sonic_rotlocal_v3", "twist2_rotlocal_v3"}:
                    paths.append(dataset_dir)
            elif args.mode == "sonic":
                if "sonic" in name:
                    paths.append(dataset_dir)
            elif args.mode == "twist2":
                if "twist2" in name:
                    paths.append(dataset_dir)
    return paths


def read_csv_paths(path: Path) -> list[Path]:
    if path is None:
        raise ValueError("--csv is required when --mode csv")
    path = normalize_path(path)
    rows: list[list[str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            cleaned = [cell.strip() for cell in row]
            if not cleaned or not any(cleaned):
                continue
            if cleaned[0].startswith("#"):
                continue
            rows.append(cleaned)

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    header_tokens = {"dataset_path", "dataset_root", "path", "root", "dataset"}
    first = [cell.lower() for cell in rows[0]]
    if any(cell in header_tokens for cell in first):
        header = first
        try:
            col_idx = next(i for i, cell in enumerate(header) if cell in header_tokens)
        except StopIteration as exc:
            raise ValueError(f"CSV header must include one of {sorted(header_tokens)}") from exc
        return [Path(row[col_idx]) for row in rows[1:] if len(row) > col_idx and row[col_idx]]

    if len(rows[0]) == 1:
        return [Path(row[0]) for row in rows if row and row[0]]

    # Header-less multi-column CSV: use the first absolute-looking path cell in each row.
    paths: list[Path] = []
    for row in rows:
        selected = next((cell for cell in row if cell.startswith("/")), row[0])
        paths.append(Path(selected))
    return paths


def select_sources(args: argparse.Namespace) -> list[SourceDataset]:
    raw_paths = read_csv_paths(args.csv) if args.mode == "csv" else discover_datasets(args)

    seen: OrderedDict[str, Path] = OrderedDict()
    for path in raw_paths:
        normalized = str(normalize_path(path))
        seen.setdefault(normalized, normalize_path(path))

    sources = [load_source(path, normalize_path(args.datasets_root)) for path in seen.values()]
    if not sources:
        raise RuntimeError(f"No datasets selected for mode={args.mode}")
    validate_compatible_sources(sources)
    return sources


def validate_compatible_sources(sources: list[SourceDataset]) -> None:
    first = sources[0]
    feature_ref = first.info["features"]
    fps_ref = first.info["fps"]
    robot_ref = first.info["robot_type"]
    codebase_ref = first.info["codebase_version"]

    for source in sources[1:]:
        if source.info["features"] != feature_ref:
            raise ValueError(f"Feature mismatch: {source.path} differs from {first.path}")
        if source.info["fps"] != fps_ref:
            raise ValueError(f"FPS mismatch: {source.path} differs from {first.path}")
        if source.info["robot_type"] != robot_ref:
            raise ValueError(f"robot_type mismatch: {source.path} differs from {first.path}")
        if source.info["codebase_version"] != codebase_ref:
            raise ValueError(f"codebase_version mismatch: {source.path} differs from {first.path}")


def build_global_tasks(sources: list[SourceDataset]) -> tuple[pd.DataFrame, list[dict[int, int]]]:
    task_to_global: OrderedDict[str, int] = OrderedDict()
    source_maps: list[dict[int, int]] = []

    for source in sources:
        task_map: dict[int, int] = {}
        tasks = source.tasks.sort_values("task_index")
        for task, row in tasks.iterrows():
            task_name = str(task)
            if task_name not in task_to_global:
                task_to_global[task_name] = len(task_to_global)
            task_map[int(row["task_index"])] = task_to_global[task_name]
        source_maps.append(task_map)

    task_df = pd.DataFrame(
        {"task_index": list(task_to_global.values())},
        index=pd.Index(list(task_to_global.keys()), name="task"),
    )
    return task_df, source_maps


def chunk_file_from_linear(linear_idx: int, chunk_size: int) -> tuple[int, int]:
    return linear_idx // chunk_size, linear_idx % chunk_size


def materialize_video(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported video mode: {mode}")


def scalar_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(values)
    if arr.size == 0:
        raise ValueError("Cannot compute stats for an empty scalar column")
    arr_float = arr.astype(np.float64)
    return {
        "min": np.array([arr.min()]),
        "max": np.array([arr.max()]),
        "mean": np.array([arr_float.mean()]),
        "std": np.array([arr_float.std()]),
        "count": np.array([arr.size]),
        "q01": np.array([np.quantile(arr_float, 0.01)]),
        "q10": np.array([np.quantile(arr_float, 0.10)]),
        "q50": np.array([np.quantile(arr_float, 0.50)]),
        "q90": np.array([np.quantile(arr_float, 0.90)]),
        "q99": np.array([np.quantile(arr_float, 0.99)]),
    }


def update_stats(
    output_root: Path,
    sources: list[SourceDataset],
    scalar_columns: dict[str, list[np.ndarray]],
) -> None:
    source_stats = [load_stats(source.path) for source in sources]
    if any(stats is None for stats in source_stats):
        raise FileNotFoundError("At least one source dataset is missing meta/stats.json")
    merged_stats = aggregate_stats(source_stats)

    for key, chunks in scalar_columns.items():
        if chunks:
            merged_stats[key] = scalar_stats(np.concatenate(chunks))

    write_stats(merged_stats, output_root)


def data_file_ids(episodes: pd.DataFrame) -> list[tuple[int, int]]:
    pairs = zip(episodes["data/chunk_index"], episodes["data/file_index"], strict=False)
    return sorted({(int(chunk), int(file)) for chunk, file in pairs})


def video_file_ids(episodes: pd.DataFrame, video_key: str) -> list[tuple[int, int]]:
    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    pairs = zip(episodes[chunk_col], episodes[file_col], strict=False)
    return sorted({(int(chunk), int(file)) for chunk, file in pairs})


def merge_datasets(
    sources: list[SourceDataset],
    output_root: Path,
    repo_id: str,
    video_mode: str,
    overwrite: bool,
) -> None:
    output_root = normalize_path(output_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True)
    task_df, source_task_maps = build_global_tasks(sources)

    first_info = json.loads(json.dumps(sources[0].info))
    chunk_size = int(first_info.get("chunks_size", 1000))
    video_keys = [
        key for key, feature in first_info["features"].items() if feature.get("dtype") == "video"
    ]

    episode_offset = 0
    frame_offset = 0
    data_file_counter = 0
    video_file_counters = {key: 0 for key in video_keys}
    episode_frames: list[pd.DataFrame] = []
    scalar_columns: dict[str, list[np.ndarray]] = {
        "index": [],
        "episode_index": [],
        "task_index": [],
    }
    manifest_sources: list[dict[str, Any]] = []

    for source_idx, source in enumerate(sources):
        source_task_map = source_task_maps[source_idx]
        data_map: dict[tuple[int, int], tuple[int, int]] = {}
        video_maps: dict[str, dict[tuple[int, int], tuple[int, int]]] = {key: {} for key in video_keys}

        for src_chunk, src_file in data_file_ids(source.episodes):
            dst_chunk, dst_file = chunk_file_from_linear(data_file_counter, chunk_size)
            data_file_counter += 1
            src_path = source.path / first_info["data_path"].format(
                chunk_index=src_chunk,
                file_index=src_file,
            )
            dst_path = output_root / first_info["data_path"].format(
                chunk_index=dst_chunk,
                file_index=dst_file,
            )
            df = pd.read_parquet(src_path)
            df["episode_index"] = df["episode_index"].astype("int64") + episode_offset
            df["index"] = df["index"].astype("int64") + frame_offset
            mapped_task_index = df["task_index"].map(source_task_map)
            if mapped_task_index.isna().any():
                missing = sorted(set(df.loc[mapped_task_index.isna(), "task_index"].tolist()))
                raise ValueError(f"Missing task_index mapping for {source.path}: {missing}")
            df["task_index"] = mapped_task_index.astype("int64")

            for key in scalar_columns:
                scalar_columns[key].append(df[key].to_numpy(copy=True))

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst_path, index=False)
            data_map[(src_chunk, src_file)] = (dst_chunk, dst_file)

        for video_key in video_keys:
            for src_chunk, src_file in video_file_ids(source.episodes, video_key):
                dst_chunk, dst_file = chunk_file_from_linear(video_file_counters[video_key], chunk_size)
                video_file_counters[video_key] += 1
                src_path = source.path / first_info["video_path"].format(
                    video_key=video_key,
                    chunk_index=src_chunk,
                    file_index=src_file,
                )
                dst_path = output_root / first_info["video_path"].format(
                    video_key=video_key,
                    chunk_index=dst_chunk,
                    file_index=dst_file,
                )
                materialize_video(src_path, dst_path, video_mode)
                video_maps[video_key][(src_chunk, src_file)] = (dst_chunk, dst_file)

        ep_df = source.episodes.copy()
        ep_df["episode_index"] = ep_df["episode_index"].astype("int64") + episode_offset
        ep_df["dataset_from_index"] = ep_df["dataset_from_index"].astype("int64") + frame_offset
        ep_df["dataset_to_index"] = ep_df["dataset_to_index"].astype("int64") + frame_offset

        for idx, row in ep_df.iterrows():
            src_key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
            ep_df.at[idx, "data/chunk_index"] = data_map[src_key][0]
            ep_df.at[idx, "data/file_index"] = data_map[src_key][1]
            for video_key in video_keys:
                chunk_col = f"videos/{video_key}/chunk_index"
                file_col = f"videos/{video_key}/file_index"
                src_video_key = (int(row[chunk_col]), int(row[file_col]))
                ep_df.at[idx, chunk_col] = video_maps[video_key][src_video_key][0]
                ep_df.at[idx, file_col] = video_maps[video_key][src_video_key][1]

        ep_df["meta/episodes/chunk_index"] = 0
        ep_df["meta/episodes/file_index"] = 0
        episode_frames.append(ep_df)

        manifest_sources.append(
            {
                "path": str(source.path),
                "task_name": source.task_name,
                "dataset_name": source.dataset_name,
                "total_episodes": int(source.info["total_episodes"]),
                "total_frames": int(source.info["total_frames"]),
                "task_index_map": {str(k): int(v) for k, v in source_task_map.items()},
            }
        )
        episode_offset += int(source.info["total_episodes"])
        frame_offset += int(source.info["total_frames"])

    episodes_df = pd.concat(episode_frames, ignore_index=True)
    episodes_path = output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_df.to_parquet(episodes_path, index=False)

    info = first_info
    info["total_episodes"] = int(episode_offset)
    info["total_frames"] = int(frame_offset)
    info["total_tasks"] = int(len(task_df))
    info["splits"] = {"train": f"0:{episode_offset}"}

    write_tasks(task_df, output_root)
    write_info(info, output_root)
    update_stats(output_root, sources, scalar_columns)

    manifest = {
        "repo_id": repo_id,
        "root": str(output_root),
        "video_mode": video_mode,
        "total_sources": len(sources),
        "total_tasks": len(task_df),
        "total_episodes": int(episode_offset),
        "total_frames": int(frame_offset),
        "sources": manifest_sources,
        "tasks": [
            {"task_index": int(row["task_index"]), "task": str(task)}
            for task, row in task_df.iterrows()
        ],
    }
    with (output_root / "merge_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)
        f.write("\n")


def print_plan(sources: list[SourceDataset], output_root: Path, repo_id: str, video_mode: str) -> None:
    task_df, _ = build_global_tasks(sources)
    total_episodes = sum(int(source.info["total_episodes"]) for source in sources)
    total_frames = sum(int(source.info["total_frames"]) for source in sources)
    print(f"selected datasets : {len(sources)}")
    print(f"language tasks    : {len(task_df)}")
    print(f"total episodes    : {total_episodes}")
    print(f"total frames      : {total_frames}")
    print(f"output root       : {output_root}")
    print(f"repo id           : {repo_id}")
    print(f"video mode        : {video_mode}")
    print("")
    for source in sources:
        task_names = [str(task) for task in source.tasks.sort_values("task_index").index]
        print(
            f"- {source.task_name}/{source.dataset_name}: "
            f"episodes={source.info['total_episodes']} frames={source.info['total_frames']}"
        )
        for task_name in task_names:
            print(f"  task: {task_name}")


def main() -> None:
    args = parse_args()
    output_root = default_output_root(args)
    repo_id = args.repo_id or f"{DEFAULT_REPO_PREFIX}/{output_root.name}"

    sources = select_sources(args)
    print_plan(sources, output_root, repo_id, args.video_mode)

    if args.dry_run:
        print("\ndry run only; no files were written")
        return

    merge_datasets(
        sources=sources,
        output_root=output_root,
        repo_id=repo_id,
        video_mode=args.video_mode,
        overwrite=args.overwrite,
    )
    print("\nmerge complete")
    print(f"--dataset.repo_id={repo_id}")
    print(f"--dataset.root={output_root}")


if __name__ == "__main__":
    main()
