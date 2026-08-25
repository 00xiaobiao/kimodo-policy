#!/usr/bin/env python3
"""Summarize selected metrics from one offline W&B run."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--tail", type=int, default=10000)
    args = parser.parse_args()
    run = read_wandb_run(resolve_wandb_file(args.path), quiet=False)
    print("rows", len(run.rows), "file", run.wandb_file)
    for metric in METRICS:
        steps, values = metric_series(run.rows, (metric,))
        if not len(values):
            continue
        tail = values[-args.tail:]
        print(
            metric,
            "last_step", int(steps[-1]),
            "last", float(values[-1]),
            "tail_mean", statistics.fmean(tail),
            "tail_std", statistics.pstdev(tail),
            "tail_min", float(tail.min()),
            "tail_max", float(tail.max()),
            "all_max", float(values.max()),
        )


if __name__ == "__main__":
    main()
