#!/usr/bin/env python3
"""
Compute per-model success rate metrics (mean, std) across seeds
from batch VLA evaluation result directories.

Usage:
    python tools/compute_test_metrics.py [--backend sonic|twist2] [--summary] <result_dir1> [result_dir2 ...]

Each result_dir must contain an episodes/ subdirectory with JSON files.

Output: CSV to stdout with columns:
    task, model_label, arch, backend, seeds, repeats_per_seed,
    seed_0, seed_1, seed_2, mean, std, total_eps, total_successes

With --summary: also appends a cross-task aggregate table by (arch, group).
"""

import argparse
import csv
import json
import os
import statistics
import sys
import sys
from collections import defaultdict
from pathlib import Path


def compute_metrics(result_dir: str, backend_filter: str | None) -> dict[str, dict]:
    ep_dir = Path(result_dir) / "episodes"
    if not ep_dir.is_dir():
        print(f"Warning: episodes dir not found: {ep_dir}", file=sys.stderr)
        return {}

    model_data: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"s": 0, "t": 0})
    )

    for fpath in sorted(ep_dir.glob("*.json")):
        try:
            d = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            continue

        ml = d.get("model_label", "?")
        if not ml or ml == "?":
            continue

        if backend_filter and backend_filter not in ml:
            continue

        seed = d.get("seed", -1)
        try:
            seed = int(seed)
        except (ValueError, TypeError):
            seed = -1

        model_data[ml][seed]["t"] += 1
        if d.get("success", False):
            model_data[ml][seed]["s"] += 1

    if not model_data:
        return {}

    task_name = Path(result_dir).name
    metrics: dict[str, dict] = {}

    for ml in sorted(model_data):
        seeds_data = model_data[ml]

        arch = "unknown"
        for a in ["act", "diffusion", "mtp"]:
            if a in ml:
                arch = a
                break
        data_backend = "unknown"
        for b in ["sonic", "twist2"]:
            if b in ml:
                data_backend = b
                break

        # Task keyword -> HOI/HSI group mapping
        TASK_GROUP_KEYWORDS = {
            "doubledesk": "HOI",
            "double_desk": "HOI",
            "football": "HOI",
            "pp_box": "HOI",
            "pickplace_box": "HOI",
            "boxing": "HSI",
            "open_door": "HSI",
            "sit_sofa": "HSI",
            "vision_navi": "HSI",
            "vision_navigation": "HSI",
        }

        task_lower = task_name.lower()
        dir_lower = os.path.basename(result_dir).lower()

        # Detect task group: HOI or HSI
        group = "unknown"

        # 1. Check task keyword mapping
        for keyword, grp in TASK_GROUP_KEYWORDS.items():
            if keyword in task_lower or keyword in dir_lower:
                group = grp
                break

        # 2. Fallback: check for HOI_ / HSI_ in directory name
        if group == "unknown":
            if "hoi_" in dir_lower or "_hoi_" in dir_lower or "_hoi" in dir_lower:
                group = "HOI"
            elif "hsi_" in dir_lower or "_hsi_" in dir_lower or "_hsi" in dir_lower:
                group = "HSI"

        per_seed_rates: dict[int, float] = {}
        for seed in sorted(seeds_data):
            d = seeds_data[seed]
            rate = d["s"] / d["t"] if d["t"] > 0 else 0.0
            per_seed_rates[seed] = rate

        rates_list = [per_seed_rates[s] for s in sorted(per_seed_rates)]
        mean_val = statistics.mean(rates_list) if rates_list else 0.0
        std_val = statistics.stdev(rates_list) if len(rates_list) > 1 else 0.0

        total_eps = sum(d["t"] for d in seeds_data.values())
        total_succ = sum(d["s"] for d in seeds_data.values())
        n_seeds = len(per_seed_rates)
        repeats_per = max(d["t"] for d in seeds_data.values()) if seeds_data else 0

        metrics[ml] = {
            "task": task_name,
            "model_label": ml,
            "arch": arch,
            "backend": data_backend,
            "group": group,
            "seeds": n_seeds,
            "repeats_per_seed": repeats_per,
            "seed_0": per_seed_rates.get(0),
            "seed_1": per_seed_rates.get(1),
            "seed_2": per_seed_rates.get(2),
            "mean": mean_val,
            "std": std_val,
            "total_eps": total_eps,
            "total_successes": total_succ,
        }

    return metrics


def fmt_rate(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def compute_summary(all_metrics: list[dict]) -> list[dict]:
    """Aggregate per-model metrics into (arch, group) summary rows."""
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)

    for row in all_metrics:
        key = (row["arch"], row["group"])
        groups[key].append(row["mean"])

    summary = []
    for (arch, group), means in sorted(groups.items()):
        agg_mean = statistics.mean(means) if means else 0.0
        agg_std = statistics.stdev(means) if len(means) > 1 else 0.0
        means_str = " | ".join(f"{m:.4f}" for m in means)
        summary.append(
            {
                "arch": arch,
                "group": group,
                "num_tasks": len(means),
                "task_means": means_str,
                "aggregate_mean": agg_mean,
                "aggregate_std": agg_std,
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-model success rate metrics from batch eval results."
    )
    parser.add_argument(
        "result_dirs",
        nargs="+",
        help="One or more result directories containing episodes/ subdirectory.",
    )
    parser.add_argument(
        "--backend",
        choices=["sonic", "twist2"],
        default=None,
        help="Filter model_label to only include models trained on given backend data.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write CSV to file instead of stdout.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=False,
        help="Also compute and output cross-task aggregate summary by (arch, group).",
    )
    args = parser.parse_args()

    all_metrics: list[dict] = []
    for d in args.result_dirs:
        m = compute_metrics(d, args.backend)
        if m:
            all_metrics.extend(m.values())

    if not all_metrics:
        print("No metrics computed.", file=sys.stderr)
        sys.exit(1)

    fieldnames = [
        "task",
        "model_label",
        "arch",
        "backend",
        "seeds",
        "repeats_per_seed",
        "seed_0",
        "seed_1",
        "seed_2",
        "mean",
        "std",
        "total_eps",
        "total_successes",
    ]

    out = open(args.output, "w", newline="") if args.output else sys.stdout
    writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for row in all_metrics:
        row_out = dict(row)
        row_out["seed_0"] = fmt_rate(row.get("seed_0"))
        row_out["seed_1"] = fmt_rate(row.get("seed_1"))
        row_out["seed_2"] = fmt_rate(row.get("seed_2"))
        row_out["mean"] = fmt_rate(row.get("mean"))
        row_out["std"] = fmt_rate(row.get("std"))
        writer.writerow(row_out)

    # Summary section
    if args.summary:
        summary_rows = compute_summary(all_metrics)
        if summary_rows:
            # Write a blank line and "SUMMARY" header row (as a comment line within CSV)
            if args.output:
                out.write("\n")
                out.write("SUMMARY (aggregate by arch + HOI/HSI group)\n")

            summary_fields = [
                "arch",
                "group",
                "num_tasks",
                "task_means",
                "aggregate_mean",
                "aggregate_std",
            ]
            swriter = csv.DictWriter(
                out, fieldnames=summary_fields, extrasaction="ignore"
            )
            swriter.writeheader()
            for sr in summary_rows:
                sr_out = dict(sr)
                sr_out["aggregate_mean"] = fmt_rate(sr.get("aggregate_mean"))
                sr_out["aggregate_std"] = fmt_rate(sr.get("aggregate_std"))
                swriter.writerow(sr_out)

            # Also print to stderr for terminal visibility
            print("\n=== Cross-Task Aggregate Summary ===", file=sys.stderr)
            for sr in summary_rows:
                print(
                    f"  [{sr['arch']:>9}] {sr['group']:>3}: "
                    f"mean={sr['aggregate_mean']:.4f}  "
                    f"std={sr['aggregate_std']:.4f}  "
                    f"(from {sr['num_tasks']} tasks: {sr['task_means']})",
                    file=sys.stderr,
                )

    if args.output:
        out.close()

    if args.summary:
        print(
            f"\nDone: {len(all_metrics)} per-model + {len(compute_summary(all_metrics))} summary entries.",
            file=sys.stderr,
        )
    else:
        print(f"\nDone: {len(all_metrics)} model entries written.", file=sys.stderr)


if __name__ == "__main__":
    main()
