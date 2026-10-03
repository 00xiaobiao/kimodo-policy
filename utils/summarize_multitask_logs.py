#!/usr/bin/env python3
"""Summarize multi-task training logs and W&B file availability.

This uses only the Python standard library so it can run on the RTX host's
system Python without importing the training environment.
"""

from __future__ import annotations

import glob
import json
import os
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINE_RE = re.compile(
    r"Step: (\d+)/(\d+) \| Loss: ([0-9.eE+-]+) \| Motion: ([0-9.eE+-]+)"
    r" \| Root: ([0-9.eE+-]+) \| Body: ([0-9.eE+-]+) \| Hand: ([0-9.eE+-]+)"
    r" \| Grad-Control: ([0-9.eE+-]+) \| Grad-Hand: ([0-9.eE+-]+)"
    r" \| Data: ([0-9.eE+-]+)s \| Compute: ([0-9.eE+-]+)s"
)


def config_for(log_dir: Path) -> dict:
    candidates = sorted(log_dir.glob("checkpoint_*/config.json"))
    if candidates:
        # Prefer the highest checkpoint number.
        candidates.sort(key=lambda p: int(p.parent.name.split("_")[-1]))
        try:
            return json.loads(candidates[-1].read_text())
        except Exception:
            return {}
    return {}


def summarize_log(path: Path) -> dict | None:
    rows: list[tuple[float, ...]] = []
    with path.open(errors="ignore") as stream:
        for line in stream:
            match = LINE_RE.search(line)
            if match:
                rows.append(tuple(float(x) for x in match.groups()))
    if not rows:
        return None
    # Columns: step, max_step, total, motion, root, body, hand, ctrl_grad,
    # hand_grad, data_time, compute_time.
    tail = rows[-10_000:]
    result = {"n": len(rows), "first_step": int(rows[0][0]), "last_step": int(rows[-1][0])}
    for idx, name in ((2, "loss"), (3, "motion"), (4, "root"), (5, "body"),
                      (6, "hand"), (7, "control_grad"), (8, "hand_grad"),
                      (9, "data_time"), (10, "compute_time")):
        all_values = [row[idx] for row in rows]
        tail_values = [row[idx] for row in tail]
        result[name] = {
            "last": all_values[-1],
            "tail_mean": statistics.fmean(tail_values),
            "tail_std": statistics.pstdev(tail_values),
            "tail_min": min(tail_values),
            "tail_max": max(tail_values),
            "all_max": max(all_values),
        }
    # Count unusually large loss/gradient excursions in the final 100k steps.
    late = rows[-100_000:]
    loss_median = statistics.median(row[2] for row in late)
    grad_median = statistics.median(row[7] for row in late)
    result["late_loss_spikes_gt5x_median"] = sum(row[2] > max(5 * loss_median, 0.1) for row in late)
    result["late_grad_spikes_gt5x_median"] = sum(row[7] > max(5 * grad_median, 0.5) for row in late)
    return result


def main() -> None:
    for path in sorted(ROOT.glob("log/**/multi_task*/*/training.log")):
        log_dir = path.parent
        cfg = config_for(log_dir)
        summary = summarize_log(path)
        if summary is None:
            continue
        print(json.dumps({
            "experiment": str(log_dir.parent.relative_to(ROOT / "log")),
            "run": log_dir.name,
            "config": {
                "layers": cfg.get("model", {}).get("controlnet_num_layers"),
                "detach": cfg.get("model", {}).get("detach_root_control_for_body"),
                "fusion": cfg.get("model", {}).get("control_fusion_mode", "both"),
                "loss_type": cfg.get("training", {}).get("loss", {}).get("motion_loss_type"),
                "max_steps": cfg.get("main", {}).get("max_steps"),
            },
            "summary": summary,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
