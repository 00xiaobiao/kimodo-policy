#!/usr/bin/env python3
"""Print late-training W&B metric outliers and local metric windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot_wandb import metric_series, read_wandb_run, resolve_wandb_file


METRICS = (
    "train/loss",
    "train/motion_loss",
    "train/root_loss",
    "train/body_loss",
    "train/hand_loss",
    "train/control_grad_norm",
    "train/hand_grad_norm",
    "perf/data_time",
    "perf/compute_time",
)


def value_at(rows: list[dict], step: int, metric: str) -> float | None:
    row = next((row for row in rows if int(row.get("_step", -1)) == step), None)
    if row is None:
        return None
    for key in (metric, metric.replace("/", ".")):
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wandb_path", type=Path)
    parser.add_argument("--min-step", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--center-step", type=int)
    parser.add_argument("--radius", type=int, default=4)
    args = parser.parse_args()

    run = read_wandb_run(resolve_wandb_file(args.wandb_path), quiet=True)
    print(f"rows={len(run.rows)}")
    candidates: dict[str, list[tuple[int, float]]] = {}
    for metric in METRICS:
        steps, values = metric_series(run.rows, (metric,))
        keep = steps >= args.min_step
        steps, values = steps[keep], values[keep]
        if not len(values):
            continue
        candidates[metric] = list(zip(steps.astype(int).tolist(), values.tolist()))
        order = np.argsort(values)[-args.top_k :][::-1]
        print(f"\n{metric} top {args.top_k} after step {args.min_step}:")
        for index in order:
            print(f"  step={int(steps[index])} value={values[index]:.9g}")

    if args.center_step is None:
        return
    print(f"\nMetrics around step {args.center_step} (+/- {args.radius}):")
    for step in range(args.center_step - args.radius, args.center_step + args.radius + 1):
        fields = [f"step={step}"]
        for metric in METRICS:
            value = value_at(run.rows, step, metric)
            if value is not None:
                fields.append(f"{metric}={value:.8g}")
        print(" | ".join(fields))


if __name__ == "__main__":
    main()
