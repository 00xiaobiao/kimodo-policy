#!/usr/bin/env python3
"""Audit deterministic training batches without decoding their video frames.

The training sampler maps every global sample ordinal to a stable
episode/window from the configured seed.  This tool uses that mapping to
report the data composition and motion-label ranges for specified steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasetloader import MultiSourceG1Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_dataset(config) -> MultiSourceG1Dataset:
    return MultiSourceG1Dataset(
        dataset_roots=OmegaConf.to_container(config.main.dataset_roots, resolve=True),
        action_history=int(config.main.action_history),
        action_chunk=int(config.main.action_chunk),
        sample_stride=int(config.main.get("sample_stride", 1)),
        episode_cache_size=int(config.main.get("episode_cache_size", 8)),
        dataset_selection=OmegaConf.to_container(
            config.main.dataset_selection, resolve=True
        ),
        sampling=OmegaConf.to_container(config.main.get("sampling", {}), resolve=True),
        target_fps=float(config.model.fps),
        sampling_seed=int(config.main.seed),
    )


def summarize_window(motion: torch.Tensor, hand: torch.Tensor, start: int, end: int) -> dict:
    window = motion[start:end].float()
    deltas = window[1:] - window[:-1]
    hand_window = hand[start:end].float()
    switches = (hand_window[1:] != hand_window[:-1]).any(dim=1).sum()
    return {
        "motion_abs_max": float(window.abs().max()),
        "motion_delta_abs_max": float(deltas.abs().max()),
        "motion_delta_abs_p99": float(torch.quantile(deltas.abs().flatten(), 0.99)),
        "hand_switches": int(switches),
    }


def sample_record(dataset: MultiSourceG1Dataset, ordinal: int) -> dict:
    rng, group_slot = dataset._sampling_state_for_index(ordinal)
    if group_slot != 0 or dataset.windows_per_episode != 1:
        raise RuntimeError("This auditor currently supports windows_per_episode=1")
    adapter, episode, original_cut = dataset._sample_record(rng)
    episode_motion = dataset._episode_motion(adapter, episode)
    if episode_motion.get("skip_episode", False):
        raise RuntimeError(
            f"Deterministic sample maps to invalid episode {episode.episode_id}: "
            f"{episode_motion.get('quality_issue')}"
        )
    cut = original_cut - int(episode_motion.get("frame_offset", 0))
    start = max(0, cut - dataset.action_history)
    end = cut + dataset.action_chunk
    if end > int(episode_motion["target_motion"].shape[0]):
        raise RuntimeError(f"Invalid selected window ordinal={ordinal}")
    metrics = summarize_window(
        episode_motion["target_motion"], episode_motion["target_hand"], start, end
    )
    return {
        "ordinal": int(ordinal),
        "task": episode.task_name,
        "episode_id": episode.episode_id,
        "cut_index": int(original_cut),
        "motion_cut_index": int(cut),
        **metrics,
    }


def step_summary(dataset: MultiSourceG1Dataset, step: int, world_size: int) -> dict:
    batch_size = 32
    start = int(step) * batch_size * int(world_size)
    samples = [
        sample_record(dataset, ordinal)
        for ordinal in range(start, start + batch_size * int(world_size))
    ]
    for metric in ("motion_abs_max", "motion_delta_abs_max", "motion_delta_abs_p99"):
        samples.sort(key=lambda item: item[metric], reverse=True)
    return {
        "step": int(step),
        "ordinal_range": [start, start + len(samples) - 1],
        "task_counts": dict(sorted(Counter(sample["task"] for sample in samples).items())),
        "batch_metrics": {
            "motion_abs_max": max(sample["motion_abs_max"] for sample in samples),
            "motion_delta_abs_max": max(sample["motion_delta_abs_max"] for sample in samples),
            "motion_delta_abs_p99_mean": sum(sample["motion_delta_abs_p99"] for sample in samples) / len(samples),
            "hand_switches_total": sum(sample["hand_switches"] for sample in samples),
        },
        "top_motion_delta_samples": sorted(
            samples, key=lambda item: item["motion_delta_abs_max"], reverse=True
        )[:12],
    }


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    dataset = build_dataset(config)
    payload = {
        "config": str(args.config),
        "world_size": args.world_size,
        "dataset_summary": dataset.source_summary,
        "steps": [step_summary(dataset, step, args.world_size) for step in args.steps],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for result in payload["steps"]:
        metrics = result["batch_metrics"]
        print(
            f"step={result['step']} ordinals={result['ordinal_range']} "
            f"delta_max={metrics['motion_delta_abs_max']:.4g} "
            f"delta_p99_mean={metrics['motion_delta_abs_p99_mean']:.4g} "
            f"hand_switches={metrics['hand_switches_total']} "
            f"tasks={result['task_counts']}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
