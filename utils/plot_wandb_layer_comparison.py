#!/usr/bin/env python3
"""Compare offline W&B training curves for ControlNet layer ablations.

Example:
    python utils/plot_wandb_layer_comparison.py \
        --run 4 /path/to/controlnet4/wandb \
        --run 8 /path/to/controlnet8/wandb \
        --run 16 /path/to/controlnet16/wandb \
        --output log/ablation/controlnet_layers_detach_true_wandb_comparison.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_wandb import _downsample_indices, _ema, metric_series, read_wandb_run, resolve_wandb_file


COLORS = {
    "4": "#2563EB",
    "8": "#059669",
    "16": "#EA580C",
}
METRICS = (
    ("train/loss", "Total loss", "Loss", False),
    ("train/root_loss", "Root loss", "Loss", False),
    ("train/body_loss", "Body loss", "Loss", False),
    ("train/hand_loss", "Hand loss", "Loss", False),
    ("train/control_grad_norm", "Control gradient norm", "L2 norm", True),
    ("train/hand_grad_norm", "Hand gradient norm", "L2 norm", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "WAND_B_PATH"),
        required=True,
        help="A legend label and a W&B directory or .wandb file. Repeat for each run.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ema-span", type=int, default=1_000)
    parser.add_argument("--raw-points", type=int, default=4_000)
    return parser.parse_args()


def plot_metric(
    axis: plt.Axes,
    runs: list[tuple[str, object]],
    metric: str,
    title: str,
    ylabel: str,
    log_scale: bool,
    ema_span: int,
    raw_points: int,
) -> None:
    plotted = False
    for label, run in runs:
        steps, values = metric_series(run.rows, (metric,))
        if not len(values):
            continue
        color = COLORS.get(label, "#475569")
        raw = _downsample_indices(len(values), raw_points)
        visible_values = np.maximum(values, 1.0e-10) if log_scale else values
        axis.plot(
            steps[raw] / 1_000,
            visible_values[raw],
            color=color,
            alpha=0.055,
            linewidth=0.55,
        )
        axis.plot(
            steps / 1_000,
            _ema(visible_values, ema_span),
            color=color,
            linewidth=2.2,
            label=f"{label} layers",
        )
        plotted = True

    axis.set_title(title, fontsize=13, pad=9)
    axis.set_xlabel("Training step (thousands)")
    axis.set_ylabel(ylabel)
    axis.grid(True, color="#CBD5E1", alpha=0.55, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    if log_scale:
        axis.set_yscale("log")
    if plotted:
        axis.legend(frameon=False, fontsize=9, loc="best")
    else:
        axis.text(0.5, 0.5, "Metric not logged", ha="center", va="center", transform=axis.transAxes)


def main() -> None:
    args = parse_args()
    if len(args.run) < 2:
        raise SystemExit("At least two --run LABEL PATH arguments are required.")

    runs = []
    for label, input_path in args.run:
        wandb_file = resolve_wandb_file(Path(input_path))
        print(f"Reading {label}-layer run: {wandb_file}", file=sys.stderr)
        runs.append((label, read_wandb_run(wandb_file)))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(18, 9.8))
    figure.subplots_adjust(
        left=0.055,
        right=0.992,
        bottom=0.07,
        top=0.885,
        wspace=0.14,
        hspace=0.24,
    )
    figure.suptitle(
        "ControlNet depth ablation: detached root-to-body gradient",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
        y=0.975,
    )
    figure.text(
        0.5,
        0.935,
        "MSE loss, 50w updates, 4-GPU training. Faint lines are raw W&B values; solid lines are EMA (span 1,000).",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )

    for axis, (metric, title, ylabel, log_scale) in zip(axes.flat, METRICS):
        plot_metric(
            axis,
            runs,
            metric,
            title,
            ylabel,
            log_scale,
            args.ema_span,
            args.raw_points,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, facecolor="white")
    print(f"Saved comparison figure to {args.output}")


if __name__ == "__main__":
    main()
