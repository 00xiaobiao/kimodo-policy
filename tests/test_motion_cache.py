import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch

from data.motion_cache import (
    load_motion_cache_manifest,
    motion_cache_signature,
    prepare_motion_cache,
)
from data.multisource_dataset import MultiSourceG1Dataset


class _Episode:
    def __init__(self, source, episode_id):
        self.source = source
        self.episode_id = episode_id
        self.task_id = f"{source}::task"
        self.task_name = "task"
        self.instruction = f"instruction for {source}"
        self.first_cut = 0
        self.target_length = 8
        self.target_fps = 30.0
        self.video_from_timestamp = 0.0
        self.video_path = Path(f"/dataset/{source}/{episode_id}.mp4")
        self.metadata = {}
        self.sample_count = 4
        self.cache_key = (
            source,
            self.task_id,
            episode_id,
            f"/dataset/{source}/{episode_id}.parquet",
            0,
            8,
        )


class _Adapter:
    def __init__(self):
        self.calls = []

    def load_episode(self, episode):
        self.calls.append((episode.source, episode.episode_id))
        if episode.episode_id == "invalid":
            return {"skip_episode": True, "quality_issue": "synthetic jump"}
        motion = torch.arange(8 * 417, dtype=torch.float32).reshape(8, 417)
        hand = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        return {
            "target_motion": motion,
            "observed_motion": motion + 1,
            "observed_motion_valid": torch.ones(8, 417, dtype=torch.bool),
            "target_hand": hand,
            "observed_hand": hand + 1,
            "target_hand_valid": torch.ones(8, 2, dtype=torch.bool),
            "observed_hand_valid": torch.ones(8, 2, dtype=torch.bool),
            "target_motion_source": "synthetic",
        }


def _dataset_for_attach(records, adapters):
    dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
    dataset._episodes_by_source = records
    dataset._episodes_by_source_task = {}
    dataset._source_names = list(records)
    dataset._all_episode_records = [
        record for source in dataset._source_names for record in records[source]
    ]
    dataset._length = sum(episode.sample_count for _, episode in dataset._all_episode_records)
    dataset._configured_source_weights = {}
    dataset._source_weights = None
    dataset._episode_weights = None
    dataset.sampling_mode = "window_proportional"
    dataset._motion_cache_dir = None
    dataset._motion_cache_signature = None
    dataset._motion_cache_invalid_tokens = set()
    dataset._runtime_invalid_episodes = set()
    dataset._episode_cache = OrderedDict()
    dataset.episode_cache_size = 8
    dataset.action_history = 3
    dataset.action_chunk = 4
    dataset.sample_stride = 1
    dataset.sampling_seed = 1234
    dataset.windows_per_episode = 1
    dataset.video_cache_size = 0
    dataset._video_cache = OrderedDict()
    dataset._text_embeddings = {}
    dataset._read_video_frame = lambda *_args, **_kwargs: torch.zeros(3, 2, 2)
    dataset.adapters = adapters
    return dataset


def _distributed_cache_process(rank, world_size, init_file, cache_dir, signature):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        adapter = _Adapter()
        episodes = [
            _Episode("HIW500", f"distributed-{index}") for index in range(8)
        ]
        records = {"HIW500": [(adapter, episode) for episode in episodes]}
        dataset = SimpleNamespace(
            _source_names=["HIW500"],
            _episodes_by_source=records,
            adapters={"HIW500": adapter},
        )
        prepare_motion_cache(
            dataset,
            cache_dir,
            signature,
            rank=rank,
            world_size=world_size,
            workers_per_rank=2,
        )
    finally:
        torch.distributed.destroy_process_group()


class MotionCacheTest(unittest.TestCase):
    def test_cached_motion_is_identical_and_arena_stays_live(self):
        pretrain_adapter = _Adapter()
        arena_adapter = _Adapter()
        good = _Episode("HIW500", "good")
        invalid = _Episode("HIW500", "invalid")
        arena = _Episode("HumanoidArena", "arena")
        records = {
            "HIW500": [(pretrain_adapter, good), (pretrain_adapter, invalid)],
            "HumanoidArena": [(arena_adapter, arena)],
        }
        dataset_for_build = SimpleNamespace(
            _source_names=["HIW500", "HumanoidArena"],
            _episodes_by_source=records,
            adapters={"HIW500": pretrain_adapter, "HumanoidArena": arena_adapter},
        )
        live_motion = pretrain_adapter.load_episode(good)
        pretrain_adapter.calls.clear()

        with tempfile.TemporaryDirectory() as temporary_dir:
            signature = motion_cache_signature({"test": "cache-equivalence"})
            cache_dir = prepare_motion_cache(dataset_for_build, temporary_dir, signature)
            pretrain_adapter.calls.clear()

            cached_dataset = _dataset_for_attach(records, dataset_for_build.adapters)
            cached_dataset.attach_motion_cache(cache_dir, signature)
            self.assertEqual(
                [episode.episode_id for _, episode in cached_dataset._all_episode_records],
                ["good", "arena"],
            )
            cached_motion = cached_dataset._episode_motion(pretrain_adapter, good)
            self.assertEqual(pretrain_adapter.calls, [])
            for key, expected in live_motion.items():
                actual = cached_motion[key]
                if torch.is_tensor(expected):
                    self.assertTrue(torch.equal(actual, expected), key)
                    self.assertEqual(actual.dtype, expected.dtype, key)
                    self.assertEqual(tuple(actual.shape), tuple(expected.shape), key)
                else:
                    self.assertEqual(actual, expected, key)

            # HumanoidArena is deliberately not cached and still calls its adapter.
            cached_dataset._episode_motion(arena_adapter, arena)
            self.assertEqual(arena_adapter.calls, [("HumanoidArena", "arena")])

    def test_cached_and_live_samples_are_identical(self):
        adapter = _Adapter()
        good = _Episode("HIW500", "good")
        records = {"HIW500": [(adapter, good)]}
        dataset_for_build = SimpleNamespace(
            _source_names=["HIW500"],
            _episodes_by_source=records,
            adapters={"HIW500": adapter},
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            signature = motion_cache_signature({"test": "sample-equivalence"})
            cache_dir = prepare_motion_cache(dataset_for_build, temporary_dir, signature)
            live_dataset = _dataset_for_attach(records, {"HIW500": adapter})
            cached_dataset = _dataset_for_attach(records, {"HIW500": adapter})
            cached_dataset.attach_motion_cache(cache_dir, signature)
            adapter.calls.clear()

            for index in range(8):
                live_sample = live_dataset[index]
                cached_sample = cached_dataset[index]
                self.assertEqual(live_sample.keys(), cached_sample.keys())
                for key, expected in live_sample.items():
                    actual = cached_sample[key]
                    if torch.is_tensor(expected):
                        self.assertTrue(torch.equal(actual, expected), key)
                        self.assertEqual(actual.dtype, expected.dtype, key)
                        self.assertEqual(tuple(actual.shape), tuple(expected.shape), key)
                    else:
                        self.assertEqual(actual, expected, key)

    def test_parallel_workers_produce_the_same_cache_payloads(self):
        adapter = _Adapter()
        episodes = [_Episode("HIW500", f"good-{index}") for index in range(4)]
        records = {"HIW500": [(adapter, episode) for episode in episodes]}
        dataset_for_build = SimpleNamespace(
            _source_names=["HIW500"],
            _episodes_by_source=records,
            adapters={"HIW500": adapter},
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            signature = motion_cache_signature({"test": "parallel-cache"})
            cache_dir = prepare_motion_cache(
                dataset_for_build,
                temporary_dir,
                signature,
                workers_per_rank=2,
            )
            manifest = load_motion_cache_manifest(cache_dir, signature)
            self.assertEqual(
                manifest["stats"],
                {"valid": 4, "invalid": 0, "error": 0, "total": 4},
            )
            self.assertEqual(adapter.calls, [])
            cached_dataset = _dataset_for_attach(records, {"HIW500": adapter})
            cached_dataset.attach_motion_cache(cache_dir, signature)
            for index in range(4):
                sample = cached_dataset[index]
                self.assertEqual(tuple(sample["gt_motion"].shape), (7, 417))

    def test_two_ranks_use_all_configured_workers_without_duplicate_episodes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            init_file = str(Path(temporary_dir) / "distributed_init")
            cache_dir = str(Path(temporary_dir) / "cache")
            signature = motion_cache_signature({"test": "distributed-cache"})
            torch.multiprocessing.spawn(
                _distributed_cache_process,
                args=(2, init_file, cache_dir, signature),
                nprocs=2,
                join=True,
            )
            manifest = load_motion_cache_manifest(cache_dir, signature)
            self.assertEqual(
                manifest["stats"],
                {"valid": 8, "invalid": 0, "error": 0, "total": 8},
            )
            self.assertEqual(len(manifest["entries"]), 8)


if __name__ == "__main__":
    unittest.main()
