#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

FIELDNAMES = [
    "task",
    "model_path",
    "model_label",
    "seed",
    "episode_index",
    "success",
    "failure_reason",
    "episode_steps",
    "max_steps",
    "final_reward",
    "final_reward_scaled",
    "max_reward",
    "max_reward_scaled",
    "video_path",
    "log_path",
    "vla_trace_path",
    "server_url",
    "returncode",
]

PRESETS = {
    "stack_only": [r"stack expects a non-empty TensorList"],
    "server_bug": [
        r"stack expects a non-empty TensorList",
        r"HTTP 500 from .*?/infer",
        r"predict_action_chunk",
        r"infer_chunk",
    ],
    "provider_bug": [
        r"encoder/decoder not loaded",
        r"Encoder/Decoder missing during runtime",
        r"refusing to fall back to default pose",
        r"loaded .* on CPU instead of cuda",
        r"requested GPU .* available_providers",
        r"CUDNN_STATUS",
        r"CUBLAS_STATUS",
        r"libonnxruntime_providers_",
        r"libcublas",
        r"CUDAExecutionProvider unavailable",
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quarantine eval episodes with infra/code-level errors and refresh summaries.")
    parser.add_argument("--batch-dir", action="append", required=True, help="Batch dir(s) under eval_results.")
    parser.add_argument("--preset", action="append", choices=sorted(PRESETS), default=[], help="Regex preset(s) for matching problematic episode logs.")
    parser.add_argument("--pattern", action="append", default=[], help="Additional regex pattern(s) matched against episode log text.")
    parser.add_argument("--reason", type=str, default="infra_issue", help="Reason tag stored in quarantine manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be moved.")
    return parser


def collect_patterns(presets: list[str], raw_patterns: list[str]) -> list[re.Pattern[str]]:
    merged: list[str] = []
    for preset in presets:
        merged.extend(PRESETS[preset])
    merged.extend(raw_patterns)
    if not merged:
        raise SystemExit("No match patterns specified. Use --preset and/or --pattern.")
    return [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in merged]


def iter_episode_logs(batch_dir: Path) -> list[Path]:
    logs_dir = batch_dir / "logs"
    if not logs_dir.is_dir():
        return []
    return sorted([p for p in logs_dir.glob("*.log") if not p.name.startswith("server__worker_")])


def log_matches(path: Path, patterns: list[re.Pattern[str]]) -> bool:
    text = path.read_text(errors="replace")
    return any(pattern.search(text) for pattern in patterns)


def load_episode_row(batch_dir: Path, stem: str) -> dict | None:
    episode_json = batch_dir / "episodes" / f"{stem}.json"
    if not episode_json.is_file():
        return None
    try:
        return json.loads(episode_json.read_text())
    except Exception:
        return None


def gather_paths_for_episode(batch_dir: Path, stem: str, row: dict | None) -> list[Path]:
    paths: list[Path] = []
    candidates = [
        batch_dir / "logs" / f"{stem}.log",
        batch_dir / "episodes" / f"{stem}.json",
    ]
    if row:
        for key in ("log_path", "video_path", "vla_trace_path"):
            value = row.get(key)
            if value:
                candidates.append(Path(value).expanduser())

    for video_dir in (batch_dir / "videos" / "success", batch_dir / "videos" / "failure"):
        if video_dir.is_dir():
            candidates.extend(sorted(video_dir.glob(f"{stem}__*.mp4")))
    traces_dir = batch_dir / "vla_traces"
    if traces_dir.is_dir():
        candidates.extend(sorted(traces_dir.glob(f"{stem}*")))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists() and candidate not in seen:
            seen.add(candidate)
            paths.append(candidate)
    return paths


def load_results(batch_dir: Path) -> list[dict]:
    rows: list[dict] = []
    episodes_dir = batch_dir / "episodes"
    if not episodes_dir.is_dir():
        return rows
    for result_path in sorted(episodes_dir.glob("*.json")):
        try:
            rows.append(json.loads(result_path.read_text()))
        except Exception:
            continue
    return rows


def write_summary(batch_dir: Path, results: list[dict]) -> None:
    jsonl_path = batch_dir / "summary.jsonl"
    csv_path = batch_dir / "summary.csv"
    summary_json_path = batch_dir / "summary.json"

    with open(jsonl_path, "w", encoding="utf-8") as fp:
        for row in results:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in results:
            writer.writerow({name: row.get(name, "") for name in FIELDNAMES})

    per_model: dict[str, dict] = {}
    for row in results:
        model_label = row["model_label"]
        stats = per_model.setdefault(model_label, {"model_path": row["model_path"], "episodes": 0, "successes": 0})
        stats["episodes"] += 1
        stats["successes"] += int(bool(row.get("success")))
    for stats in per_model.values():
        stats["success_rate"] = stats["successes"] / stats["episodes"] if stats["episodes"] else 0.0

    total_successes = sum(int(bool(row.get("success"))) for row in results)
    summary = {
        "total_episodes": len(results),
        "total_successes": total_successes,
        "overall_success_rate": total_successes / len(results) if results else 0.0,
        "per_model": per_model,
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")


def quarantine_batch(batch_dir: Path, patterns: list[re.Pattern[str]], reason: str, dry_run: bool) -> int:
    batch_dir = batch_dir.expanduser().resolve()
    if not batch_dir.is_dir():
        print(f"skip missing batch_dir={batch_dir}")
        return 0

    matched_logs = [path for path in iter_episode_logs(batch_dir) if log_matches(path, patterns)]
    stems = sorted({path.stem for path in matched_logs})
    if not stems:
        print(f"batch={batch_dir} matched_episodes=0")
        return 0

    quarantine_root = batch_dir / "quarantine" / datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_rows = []
    move_map: dict[Path, list[Path]] = defaultdict(list)

    for stem in stems:
        row = load_episode_row(batch_dir, stem)
        file_paths = gather_paths_for_episode(batch_dir, stem, row)
        for file_path in file_paths:
            move_map[Path(file_path)].append(Path(stem))
        manifest_rows.append(
            {
                "episode_stem": stem,
                "reason": reason,
                "log_path": str((batch_dir / "logs" / f"{stem}.log").resolve()),
                "episode_json": str((batch_dir / "episodes" / f"{stem}.json").resolve()),
                "model_label": (row or {}).get("model_label", ""),
                "model_path": (row or {}).get("model_path", ""),
                "seed": (row or {}).get("seed", ""),
                "episode_index": (row or {}).get("episode_index", ""),
                "success": (row or {}).get("success", ""),
                "failure_reason": (row or {}).get("failure_reason", ""),
            }
        )

    print(f"batch={batch_dir}")
    print(f"  matched_episode_logs={len(matched_logs)} unique_episodes={len(stems)} files_to_move={len(move_map)} dry_run={dry_run}")

    if dry_run:
        for row in manifest_rows[:10]:
            print(
                "  sample_episode={episode_stem} model={model_label} seed={seed} episode={episode_index}".format(
                    episode_stem=row.get("episode_stem"),
                    model_label=row.get("model_label"),
                    seed=row.get("seed"),
                    episode_index=row.get("episode_index"),
                )
            )
        return len(stems)

    quarantine_root.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(move_map):
        dest_path = quarantine_root / source_path.relative_to(batch_dir)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(dest_path))

    manifest_path = quarantine_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch_dir": str(batch_dir),
                "reason": reason,
                "episodes": manifest_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    results = load_results(batch_dir)
    write_summary(batch_dir, results)
    success_rate = sum(int(bool(r.get("success"))) for r in results) / len(results) if results else 0.0
    print(f"  quarantined_to={quarantine_root}")
    print(f"  refreshed_summary episodes={len(results)} overall_success_rate={success_rate:.4f}")
    return len(stems)


def main() -> int:
    args = build_parser().parse_args()
    patterns = collect_patterns(args.preset, args.pattern)
    total = 0
    for batch_dir in args.batch_dir:
        total += quarantine_batch(Path(batch_dir), patterns, args.reason, args.dry_run)
    print(f"total_quarantined_episodes={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
