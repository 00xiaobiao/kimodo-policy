import json
import random

import numpy as np
import pytest

from lerobot.scripts.lerobot_sample_merge import (
    sample_and_merge_two_datasets,
    select_episode_indices,
    validate_pair_is_mergeable,
)


def _build_dataset(factory, root, repo_id, fps, total_episodes=4, frames_per_episode=5):
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
    }

    dataset = factory(
        root=root,
        repo_id=repo_id,
        fps=fps,
        features=features,
        use_videos=False,
    )

    for episode_index in range(total_episodes):
        for frame_index in range(frames_per_episode):
            dataset.add_frame(
                {
                    "action": np.array([episode_index, frame_index], dtype=np.float32),
                    "observation.state": np.array(
                        [episode_index, frame_index, episode_index + frame_index], dtype=np.float32
                    ),
                    "task": f"task_{episode_index % 2}",
                }
            )
        dataset.save_episode()

    dataset.finalize()
    return dataset


def test_validate_pair_is_mergeable_rejects_fps_mismatch(tmp_path, empty_lerobot_dataset_factory):
    dataset_a = _build_dataset(
        empty_lerobot_dataset_factory,
        tmp_path / "dataset_a",
        repo_id="dataset_a",
        fps=30,
    )
    dataset_b = _build_dataset(
        empty_lerobot_dataset_factory,
        tmp_path / "dataset_b",
        repo_id="dataset_b",
        fps=50,
    )

    with pytest.raises(ValueError, match="fps mismatch: 30 vs 50"):
        validate_pair_is_mergeable(dataset_a, dataset_b)


def test_sample_and_merge_two_datasets(tmp_path, empty_lerobot_dataset_factory):
    dataset_a = _build_dataset(
        empty_lerobot_dataset_factory,
        tmp_path / "dataset_a",
        repo_id="dataset_a",
        fps=30,
    )
    dataset_b = _build_dataset(
        empty_lerobot_dataset_factory,
        tmp_path / "dataset_b",
        repo_id="dataset_b",
        fps=30,
    )

    output_dir = tmp_path / "merged_dataset"
    seed = 7
    merged_dataset = sample_and_merge_two_datasets(
        source_a_root=dataset_a.root,
        source_b_root=dataset_b.root,
        output_dir=output_dir,
        output_repo_id="merged_dataset",
        sample_ratio=0.5,
        seed=seed,
    )

    assert merged_dataset.meta.total_episodes == 4
    assert merged_dataset.meta.total_frames == 20

    manifest = json.loads((output_dir / "sampling_manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_repo_id"] == "merged_dataset"
    assert manifest["sample_ratio"] == 0.5
    assert manifest["seed"] == seed

    rng = random.Random(seed)
    expected_a = select_episode_indices(dataset_a.meta.total_episodes, 0.5, rng)
    expected_b = select_episode_indices(dataset_b.meta.total_episodes, 0.5, rng)
    assert manifest["sources"]["source_a"]["selected_episode_indices"] == expected_a
    assert manifest["sources"]["source_b"]["selected_episode_indices"] == expected_b
