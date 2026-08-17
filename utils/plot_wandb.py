#!/usr/bin/env python3
"""Generate a Kimodo training dashboard from a local W&B run.

The input may be a ``wandb`` directory, an ``offline-run-*`` directory, or a
single ``run-*.wandb`` file.  When ``--output`` is omitted, the image is saved
as ``training_dashboard.png`` next to the input ``wandb`` directory.

Example:
    python utils/plot_wandb_training_dashboard.py \
        log/my_experiment/2026-08-17_00-00-00/wandb
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Missing dependency 'numpy'. Install it with: pip install numpy") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "Missing dependency 'matplotlib'. Install it with: pip install matplotlib"
    ) from exc

try:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Missing dependency 'wandb'. Install it with: pip install wandb") from exc


BLUE = "#1f77b4"
RED = "#d62728"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"


@dataclass
class WandbRunData:
    wandb_file: Path
    rows: list[dict[str, Any]]
    config: dict[str, Any]
    display_name: str


def _json_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _item_key(item: Any) -> str:
    nested_key = list(getattr(item, "nested_key", ()))
    if nested_key:
        return ".".join(nested_key)
    return str(getattr(item, "key", ""))


def _apply_config_items(config: dict[str, Any], items: Iterable[Any]) -> None:
    for item in items:
        key = _item_key(item)
        if not key:
            continue
        value = _json_value(item.value_json)
        cursor = config
        parts = key.split(".")
        for part in parts[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[parts[-1]] = value


def _deep_merge(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Expand dotted config keys while preserving nested W&B values."""
    normalized: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            value = _normalize_config(value)
        cursor = normalized
        parts = str(key).split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        leaf = parts[-1]
        if isinstance(value, dict) and isinstance(cursor.get(leaf), dict):
            _deep_merge(cursor[leaf], value)
        else:
            cursor[leaf] = value
    return normalized


def resolve_wandb_file(input_path: Path) -> Path:
    path = input_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"W&B path does not exist: {path}")
    if path.is_file():
        if path.suffix != ".wandb":
            raise ValueError(f"Expected a .wandb file, got: {path}")
        return path

    candidates = sorted(
        {candidate.resolve() for candidate in path.rglob("*.wandb") if candidate.is_file()},
        key=lambda candidate: (candidate.stat().st_mtime_ns, str(candidate)),
    )
    if not candidates:
        raise FileNotFoundError(f"No run-*.wandb file found under: {path}")
    if len(candidates) > 1:
        print(
            f"Found {len(candidates)} W&B runs; selecting the newest: {candidates[-1]}",
            file=sys.stderr,
        )
    return candidates[-1]


def default_output_path(input_path: Path, wandb_file: Path) -> Path:
    resolved_input = input_path.expanduser().resolve()
    search = [resolved_input] if resolved_input.is_dir() else []
    search.extend(wandb_file.parents)
    for directory in search:
        if directory.name == "wandb":
            return directory.parent / "training_dashboard.png"
    if resolved_input.is_dir() and resolved_input.name.startswith(("offline-run-", "run-")):
        return resolved_input.parent / "training_dashboard.png"
    return wandb_file.parent / "training_dashboard.png"


def read_wandb_run(wandb_file: Path, *, quiet: bool = False) -> WandbRunData:
    store = DataStore()
    store.open_for_scan(str(wandb_file))

    rows_by_step: dict[int, dict[str, Any]] = {}
    config: dict[str, Any] = {}
    display_name = ""
    history_records = 0
    fallback_step = 0

    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)

        if record.HasField("run"):
            if record.run.display_name:
                display_name = record.run.display_name
            _apply_config_items(config, record.run.config.update)
        if record.HasField("config"):
            _apply_config_items(config, record.config.update)
        if not record.HasField("history"):
            continue

        row: dict[str, Any] = {}
        for item in record.history.item:
            key = _item_key(item)
            if key:
                row[key] = _json_value(item.value_json)

        if "_step" in row:
            try:
                step = int(row["_step"])
            except (TypeError, ValueError, OverflowError):
                step = fallback_step
        elif record.history.HasField("step"):
            step = int(record.history.step.num)
        else:
            step = fallback_step
        fallback_step = max(fallback_step + 1, step + 1)

        existing = rows_by_step.setdefault(step, {"_step": step})
        existing.update(row)
        existing["_step"] = step
        history_records += 1
        if not quiet and history_records % 50_000 == 0:
            print(f"Parsed {history_records:,} W&B history records...", file=sys.stderr)

    rows = [rows_by_step[step] for step in sorted(rows_by_step)]
    if not rows:
        raise RuntimeError(f"No W&B history records found in: {wandb_file}")

    config = _normalize_config(config)
    if not display_name:
        display_name = wandb_file.parent.name
    return WandbRunData(wandb_file, rows, config, display_name)


def _nested_get(mapping: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _infer_run_title(run: WandbRunData) -> str:
    """Prefer the experiment directory used by this project's log layout."""
    for parent in run.wandb_file.parents:
        if parent.name != "wandb":
            continue
        training_run_dir = parent.parent
        timestamp_pattern = r"\d{4}-\d{2}-\d{2}[_-]\d{2}-\d{2}-\d{2}"
        if re.fullmatch(timestamp_pattern, training_run_dir.name):
            return training_run_dir.parent.name
        break

    display_name = run.display_name.strip()
    if display_name:
        return display_name
    configured_name = str(_nested_get(run.config, "main.wandb_run_name", "")).strip()
    if configured_name:
        return configured_name
    return run.wandb_file.parent.name


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def metric_series(
    rows: Sequence[dict[str, Any]], aliases: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    steps: list[float] = []
    values: list[float] = []
    for row in rows:
        value = None
        for key in aliases:
            candidates = (key, key.replace("/", "."))
            matched_key = next((candidate for candidate in candidates if candidate in row), None)
            if matched_key is not None:
                value = _as_finite_float(row[matched_key])
                if value is not None:
                    break
        if value is None:
            continue
        step = _as_finite_float(row.get("_step"))
        if step is None:
            continue
        steps.append(step)
        values.append(value)
    return np.asarray(steps, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    if not len(values):
        return values.copy()
    output = np.empty_like(values, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    output[0] = values[0]
    for index in range(1, len(values)):
        output[index] = alpha * values[index] + (1.0 - alpha) * output[index - 1]
    return output


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if not len(values):
        return values.copy()
    values = np.asarray(values, dtype=np.float64)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    output = np.empty_like(values)
    prefix = min(window, len(values))
    output[:prefix] = cumulative[1 : prefix + 1] / np.arange(1, prefix + 1)
    if len(values) > window:
        output[window:] = (
            cumulative[window + 1 :] - cumulative[1 : len(values) - window + 1]
        ) / window
    return output


def _downsample_indices(length: int, max_points: int) -> np.ndarray:
    stride = max(1, int(math.ceil(length / max_points)))
    return np.arange(0, length, stride)


def _empty_panel(axis: Any, title: str, message: str = "Metric not logged") -> None:
    axis.set_title(title, fontsize=15)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)


def _loss_panel(
    axis: Any,
    rows: Sequence[dict[str, Any]],
    aliases: Sequence[str],
    title: str,
    *,
    ema_span: int,
    raw_points: int,
    max_step: float,
) -> None:
    steps, values = metric_series(rows, aliases)
    if not len(values):
        _empty_panel(axis, title)
        return
    raw = _downsample_indices(len(values), raw_points)
    axis.plot(steps[raw], values[raw], color=BLUE, alpha=0.075, linewidth=0.7)
    axis.plot(steps, _ema(values, ema_span), color=BLUE, linewidth=2.2)
    axis.set_title(title, fontsize=15)
    axis.set_ylabel("Loss")

    visible = values[steps >= max(1.0, max_step * 0.002)]
    if not len(visible):
        visible = values
    upper = float(np.nanpercentile(visible, 99.5))
    if math.isfinite(upper) and upper > 0:
        axis.set_ylim(max(0.0, float(np.nanmin(values)) * 0.9), upper * 1.08)


def _gradient_panel(
    axis: Any,
    rows: Sequence[dict[str, Any]],
    aliases: Sequence[str],
    title: str,
    *,
    ema_span: int,
    raw_points: int,
) -> None:
    steps, values = metric_series(rows, aliases)
    if not len(values):
        _empty_panel(axis, title)
        return
    values = np.maximum(values, 1.0e-8)
    raw = _downsample_indices(len(values), raw_points)
    axis.plot(steps[raw], values[raw], color=BLUE, alpha=0.08, linewidth=0.7)
    axis.plot(steps, np.maximum(_ema(values, ema_span), 1.0e-8), color=BLUE, linewidth=2.0)
    axis.set_yscale("log")
    axis.set_title(title, fontsize=15)
    axis.set_ylabel("L2 norm")


def _clip_panel(
    axis: Any,
    rows: Sequence[dict[str, Any]],
    *,
    norm_aliases: Sequence[str],
    clipped_aliases: Sequence[str],
    scale_aliases: Sequence[str],
    threshold: float,
    title: str,
    ema_span: int,
) -> float | None:
    rate_steps, clipped = metric_series(rows, clipped_aliases)
    norm_steps, norms = metric_series(rows, norm_aliases)
    scale_steps, scales = metric_series(rows, scale_aliases)

    if not len(clipped) and len(norms):
        rate_steps = norm_steps
        clipped = (norms > threshold).astype(np.float64)
    if not len(scales) and len(norms):
        scale_steps = norm_steps
        scales = np.minimum(1.0, threshold / np.maximum(norms, 1.0e-12))
    if not len(clipped) and not len(scales):
        _empty_panel(axis, title)
        return None

    lines = []
    if len(clipped):
        clipped = np.clip(clipped, 0.0, 1.0)
        line = axis.plot(
            rate_steps,
            _rolling_mean(clipped, 1000),
            color=RED,
            linewidth=2.0,
            label="Clip rate (rolling 1k)",
        )
        lines.extend(line)
    axis.set_ylim(-0.03, 1.03)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.tick_params(axis="y", colors=RED)
    axis.set_ylabel("Clip rate", color=RED)
    axis.set_title(title, fontsize=15)

    twin = axis.twinx()
    if len(scales):
        scales = np.clip(scales, 0.0, 1.0)
        line = twin.plot(
            scale_steps,
            _ema(scales, ema_span),
            color=BLUE,
            linewidth=2.0,
            label="Clip scale",
        )
        lines.extend(line)
    twin.set_ylim(0.0, 1.03)
    twin.tick_params(axis="y", colors=BLUE)
    twin.set_ylabel("Clip scale", color=BLUE)
    if lines:
        axis.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=9)
    return float(np.mean(clipped)) if len(clipped) else None


def _learning_rate_panel(axis: Any, rows: Sequence[dict[str, Any]]) -> None:
    specifications = (
        (("train/lr_control_adapter",), "Control adapter", BLUE, "-"),
        (("train/lr_control_backbone",), "Control backbone", ORANGE, "--"),
        (("train/lr_hand",), "Hand", GREEN, ":"),
    )
    plotted = False
    for aliases, label, color, style in specifications:
        steps, values = metric_series(rows, aliases)
        if len(values):
            axis.plot(steps, np.maximum(values, 1.0e-12), color=color, linestyle=style, linewidth=1.8, label=label)
            plotted = True
    axis.set_title("Learning-rate schedules", fontsize=15)
    if not plotted:
        axis.text(0.5, 0.5, "Metric not logged", ha="center", va="center", transform=axis.transAxes)
        return
    axis.set_yscale("log")
    axis.set_ylabel("Learning rate")
    axis.legend(loc="best", fontsize=9)


def _timing_panel(axis: Any, rows: Sequence[dict[str, Any]], raw_points: int) -> None:
    plotted = False
    for aliases, label, color in (
        (("perf/compute_time",), "Compute", BLUE),
        (("perf/data_time",), "Data", ORANGE),
    ):
        steps, values = metric_series(rows, aliases)
        if len(values):
            raw = _downsample_indices(len(values), raw_points)
            axis.plot(steps[raw], np.maximum(values[raw], 1.0e-5), color=color, linewidth=1.2, label=label)
            plotted = True
    axis.set_title("Step timing", fontsize=15)
    if not plotted:
        axis.text(0.5, 0.5, "Metric not logged", ha="center", va="center", transform=axis.transAxes)
        return
    axis.set_yscale("log")
    axis.set_ylabel("Seconds")
    axis.legend(loc="best", fontsize=9)


def _throughput_panel(axis: Any, rows: Sequence[dict[str, Any]]) -> None:
    steps: list[float] = []
    times: list[float] = []
    for row in rows:
        step = _as_finite_float(row.get("_step"))
        compute = _as_finite_float(
            row.get("perf/compute_time", row.get("perf.compute_time"))
        )
        data = _as_finite_float(row.get("perf/data_time", row.get("perf.data_time")))
        if step is None or compute is None:
            continue
        steps.append(step)
        times.append(max(compute + (data or 0.0), 1.0e-6))
    axis.set_title("Effective throughput (rolling 2k)", fontsize=15)
    if not times:
        axis.text(0.5, 0.5, "Metric not logged", ha="center", va="center", transform=axis.transAxes)
        return
    axis.plot(
        np.asarray(steps),
        1.0 / _rolling_mean(np.asarray(times), 2000),
        color=BLUE,
        linewidth=2.0,
    )
    axis.set_ylabel("Steps / second")


def build_dashboard(
    run: WandbRunData,
    output_path: Path,
    *,
    title: str | None,
    ema_span: int,
    raw_points: int,
) -> tuple[float | None, float | None]:
    history_steps = np.asarray([float(row["_step"]) for row in run.rows])
    max_step = float(np.max(history_steps))
    configured_steps = _nested_get(run.config, "main.max_steps")
    configured_steps = _as_finite_float(configured_steps)
    total_steps = max(max_step, configured_steps or 0.0)
    if total_steps <= 0:
        total_steps = max(1.0, float(len(run.rows)))

    control_threshold = _as_finite_float(
        _nested_get(run.config, "main.gradient.grad_clip_norm", 1.0)
    ) or 1.0
    hand_threshold = _as_finite_float(
        _nested_get(run.config, "main.gradient.hand_grad_clip_norm", 1.0)
    ) or 1.0

    run_title = title or _infer_run_title(run)
    run_title = run_title.replace("_", " ")
    figure, axes = plt.subplots(4, 3, figsize=(22.58, 20.04), dpi=150, sharex=True)
    figure.suptitle(
        f"{run_title}\nTraining dashboard - {int(max_step):,} steps",
        fontsize=23,
        y=0.985,
    )

    loss_specs = (
        (axes[0, 0], ("train/loss",), "Total loss"),
        (axes[0, 1], ("train/motion_loss",), "Motion loss"),
        (axes[0, 2], ("train/body_loss",), "Body loss"),
        (axes[1, 0], ("train/root_loss",), "Root loss"),
        (axes[1, 1], ("train/hand_loss",), "Hand loss"),
    )
    for axis, aliases, panel_title in loss_specs:
        _loss_panel(
            axis,
            run.rows,
            aliases,
            panel_title,
            ema_span=ema_span,
            raw_points=raw_points,
            max_step=total_steps,
        )

    _gradient_panel(
        axes[1, 2],
        run.rows,
        ("train/control_grad_norm",),
        "Control gradient norm",
        ema_span=ema_span,
        raw_points=raw_points,
    )
    _gradient_panel(
        axes[2, 0],
        run.rows,
        ("train/hand_grad_norm",),
        "Hand gradient norm",
        ema_span=ema_span,
        raw_points=raw_points,
    )
    control_clip_rate = _clip_panel(
        axes[2, 1],
        run.rows,
        norm_aliases=("train/control_grad_norm",),
        clipped_aliases=("train/control_was_clipped",),
        scale_aliases=("train/control_clip_scale",),
        threshold=control_threshold,
        title="Control gradient clipping",
        ema_span=ema_span,
    )
    hand_clip_rate = _clip_panel(
        axes[2, 2],
        run.rows,
        norm_aliases=("train/hand_grad_norm",),
        clipped_aliases=("train/hand_was_clipped",),
        scale_aliases=("train/hand_clip_scale",),
        threshold=hand_threshold,
        title="Hand gradient clipping",
        ema_span=ema_span,
    )

    _learning_rate_panel(axes[3, 0], run.rows)
    _timing_panel(axes[3, 1], run.rows, raw_points)
    _throughput_panel(axes[3, 2], run.rows)

    for row in axes:
        for axis in row:
            axis.grid(True, alpha=0.23)
            axis.set_xlabel("Training step")
            axis.ticklabel_format(axis="x", style="plain")
            axis.margins(x=0)
            axis.set_xlim(0.0, total_steps)
    for axis in axes[:-1].flat:
        axis.tick_params(labelbottom=False)

    figure.text(
        0.5,
        0.018,
        f"Raw curves are lightly downsampled; solid curves use EMA span={ema_span} steps.",
        ha="center",
        fontsize=11,
        color="gray",
    )
    figure.tight_layout(rect=[0.025, 0.04, 0.98, 0.95], h_pad=1.45, w_pad=2.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return control_clip_rate, hand_clip_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "wandb_path",
        type=Path,
        help="wandb directory, offline-run directory, or run-*.wandb file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="output PNG path (default: training_dashboard.png next to wandb/)",
    )
    parser.add_argument("--title", help="override the run title shown above the dashboard")
    parser.add_argument("--ema-span", type=int, default=500, help="EMA smoothing span")
    parser.add_argument(
        "--raw-points",
        type=int,
        default=12_000,
        help="maximum raw points drawn per panel",
    )
    parser.add_argument("--quiet", action="store_true", help="hide parsing progress")
    args = parser.parse_args()
    if args.ema_span <= 0:
        parser.error("--ema-span must be positive")
    if args.raw_points <= 0:
        parser.error("--raw-points must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        wandb_file = resolve_wandb_file(args.wandb_path)
        output_path = (
            args.output.expanduser().resolve()
            if args.output
            else default_output_path(args.wandb_path, wandb_file)
        )
        print(f"W&B run: {wandb_file}")
        run = read_wandb_run(wandb_file, quiet=args.quiet)
        control_rate, hand_rate = build_dashboard(
            run,
            output_path,
            title=args.title,
            ema_span=args.ema_span,
            raw_points=args.raw_points,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Parsed steps: {len(run.rows):,} (last step: {int(run.rows[-1]['_step']):,})")
    if control_rate is not None:
        print(f"Control gradient clipped: {control_rate:.2%}")
    if hand_rate is not None:
        print(f"Hand gradient clipped: {hand_rate:.2%}")
    print(f"Saved dashboard: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
