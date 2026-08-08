import tempfile
import unittest
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import BatchSampler
from torch.optim.lr_scheduler import LambdaLR
from accelerate.data_loader import BatchSamplerShard

from train import (
    ResumableOrdinalSampler,
    _capture_rng_state,
    _load_text_embedding_cache,
    _load_rank_rng_state,
    _load_training_checkpoint,
    _process_seed,
    _restore_rng_state,
    _rng_state_path,
    _sample_ordinal_range,
    _save_rank_rng_state,
    _save_task_text_embedding,
    _task_text_embedding_cache_path,
    _worker_seed,
    build_dataloader,
)


class _TinyTrainableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = nn.Linear(3, 2)
        self.frozen = nn.Linear(3, 2)
        self.frozen.requires_grad_(False)


class _TinyDataset:
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return index


class TrainingResumeTest(unittest.TestCase):
    @staticmethod
    def _stochastic_train_step(model, optimizer, scheduler):
        optimizer.zero_grad()
        inputs = torch.rand(5, 3)
        random_scale = random.random() + float(np.random.random())
        loss = model.trainable(inputs).square().mean() * random_scale
        loss.backward()
        optimizer.step()
        scheduler.step()

    def test_python_numpy_and_torch_rng_round_trip(self):
        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(404)
        state = _capture_rng_state(rank=0, world_size=1)

        expected_python = [random.random() for _ in range(4)]
        expected_numpy = np.random.random(4)
        expected_torch = torch.rand(4)
        expected_cuda = torch.rand(4, device="cuda") if torch.cuda.is_available() else None

        for _ in range(10):
            random.random()
            np.random.random()
            torch.rand(1)
            if torch.cuda.is_available():
                torch.rand(1, device="cuda")
        _restore_rng_state(state, expected_rank=0, expected_world_size=1)

        self.assertEqual([random.random() for _ in range(4)], expected_python)
        np.testing.assert_array_equal(np.random.random(4), expected_numpy)
        torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)
        if expected_cuda is not None:
            torch.testing.assert_close(
                torch.rand(4, device="cuda"), expected_cuda, rtol=0, atol=0
            )

    def test_per_rank_rng_file_round_trip_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            _save_rank_rng_state(directory, rank=2, world_size=4)

            path = Path(_rng_state_path(directory, 2))
            payload = _load_rank_rng_state(directory, rank=2, expected_world_size=4)

            self.assertTrue(path.is_file())
            self.assertEqual(payload["rank"], 2)
            self.assertEqual(payload["world_size"], 4)
            with self.assertRaisesRegex(RuntimeError, "world_size"):
                _load_rank_rng_state(directory, rank=2, expected_world_size=8)

    def test_old_checkpoint_without_rng_file_remains_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(
                _load_rank_rng_state(directory, rank=0, expected_world_size=1)
            )

    def test_new_checkpoint_rejects_missing_rank_rng_file(self):
        model = _TinyTrainableModel()
        optimizer = torch.optim.AdamW(model.trainable.parameters(), lr=0.01)
        scheduler = LambdaLR(optimizer, lambda step: 1.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint_0"
            checkpoint_path.mkdir()
            torch.save(
                {
                    "global_step": 0,
                    "model": {
                        name: parameter.detach().clone()
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    },
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "world_size": 2,
                    "rng_state_version": 1,
                },
                checkpoint_path / "training_state.pt",
            )

            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                _load_training_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    str(checkpoint_path),
                    expected_world_size=2,
                )

    def test_checkpoint_rejects_inconsistent_sample_ordinal(self):
        model = _TinyTrainableModel()
        optimizer = torch.optim.AdamW(model.trainable.parameters(), lr=0.01)
        scheduler = LambdaLR(optimizer, lambda step: 1.0)
        optimizer.step()
        scheduler.step()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint_1"
            checkpoint_path.mkdir()
            torch.save(
                {
                    "global_step": 1,
                    "model": {
                        name: parameter.detach().clone()
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    },
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "world_size": 1,
                    "data_state": {
                        "sampling_version": 1,
                        "next_sample_ordinal": 7,
                        "batch_size": 4,
                        "gradient_accumulation_steps": 2,
                        "world_size": 1,
                    },
                },
                checkpoint_path / "training_state.pt",
            )

            with self.assertRaisesRegex(RuntimeError, "next_sample_ordinal"):
                _load_training_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    str(checkpoint_path),
                    expected_world_size=1,
                )

    def test_resumed_stochastic_step_matches_uninterrupted_training(self):
        random.seed(11)
        np.random.seed(22)
        torch.manual_seed(33)
        source_model = _TinyTrainableModel()
        source_optimizer = torch.optim.AdamW(
            source_model.trainable.parameters(), lr=0.01
        )
        source_scheduler = LambdaLR(source_optimizer, lambda step: 0.95 ** step)
        self._stochastic_train_step(
            source_model, source_optimizer, source_scheduler
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint_1"
            checkpoint_path.mkdir()
            _save_rank_rng_state(str(checkpoint_path), rank=0, world_size=1)
            torch.save(
                {
                    "global_step": 1,
                    "model": {
                        name: parameter.detach().clone()
                        for name, parameter in source_model.named_parameters()
                        if parameter.requires_grad
                    },
                    "optimizer": source_optimizer.state_dict(),
                    "scheduler": source_scheduler.state_dict(),
                    "world_size": 1,
                    "rng_state_version": 1,
                    "data_state": {
                        "sampling_version": 1,
                        "next_sample_ordinal": 5,
                        "batch_size": 5,
                        "gradient_accumulation_steps": 1,
                        "world_size": 1,
                    },
                },
                checkpoint_path / "training_state.pt",
            )

            self._stochastic_train_step(
                source_model, source_optimizer, source_scheduler
            )
            uninterrupted_state = {
                name: parameter.detach().clone()
                for name, parameter in source_model.named_parameters()
                if parameter.requires_grad
            }
            uninterrupted_optimizer = source_optimizer.state_dict()

            random.seed(101)
            np.random.seed(202)
            torch.manual_seed(303)
            resumed_model = _TinyTrainableModel()
            resumed_optimizer = torch.optim.AdamW(
                resumed_model.trainable.parameters(), lr=0.01
            )
            resumed_scheduler = LambdaLR(
                resumed_optimizer, lambda step: 0.95 ** step
            )
            resumed_step = _load_training_checkpoint(
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
                str(checkpoint_path),
                expected_world_size=1,
            )
            resumed_rng = _load_rank_rng_state(
                str(checkpoint_path), rank=0, expected_world_size=1
            )
            _restore_rng_state(
                resumed_rng, expected_rank=0, expected_world_size=1
            )
            self._stochastic_train_step(
                resumed_model, resumed_optimizer, resumed_scheduler
            )

        self.assertEqual(resumed_step, 1)
        self.assertEqual(resumed_scheduler.last_epoch, source_scheduler.last_epoch)
        self.assertEqual(
            resumed_scheduler.get_last_lr(), source_scheduler.get_last_lr()
        )
        for name, parameter in resumed_model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(
                    parameter, uninterrupted_state[name], rtol=0, atol=0
                )
        resumed_optimizer_state = resumed_optimizer.state_dict()["state"]
        for parameter_id, expected_state in uninterrupted_optimizer["state"].items():
            for state_name, expected_value in expected_state.items():
                actual_value = resumed_optimizer_state[parameter_id][state_name]
                if torch.is_tensor(expected_value):
                    torch.testing.assert_close(
                        actual_value, expected_value, rtol=0, atol=0
                    )
                else:
                    self.assertEqual(actual_value, expected_value)

    def test_resume_ordinal_range_skips_all_consumed_global_samples(self):
        config = SimpleNamespace(
            main=SimpleNamespace(
                batch_size=4,
                max_steps=10,
                gradient=SimpleNamespace(grad_accumulation_steps=2),
            )
        )

        start, stop = _sample_ordinal_range(config, resume_step=3, world_size=4)
        sampler = ResumableOrdinalSampler(start, stop)

        self.assertEqual(start, 3 * 2 * 4 * 4)
        self.assertEqual(stop, 10 * 2 * 4 * 4)
        self.assertEqual(next(iter(sampler)), start)
        self.assertEqual(len(sampler), stop - start)
        self.assertEqual((stop - start) % (4 * 4), 0)

    def test_accelerate_rank_shards_cover_each_resumed_ordinal_once(self):
        batch_size = 3
        world_size = 4
        start = 120
        stop = 168
        rank_batches = []
        for rank in range(world_size):
            batch_sampler = BatchSampler(
                ResumableOrdinalSampler(start, stop),
                batch_size=batch_size,
                drop_last=True,
            )
            shard = BatchSamplerShard(
                batch_sampler,
                num_processes=world_size,
                process_index=rank,
                split_batches=False,
            )
            rank_batches.append(list(shard))

        reconstructed = []
        for microstep in range(len(rank_batches[0])):
            for rank in range(world_size):
                reconstructed.extend(rank_batches[rank][microstep])

        self.assertEqual(reconstructed, list(range(start, stop)))

    def test_text_cache_without_instruction_metadata_is_rejected(self):
        task_id = "HumanoidArena::task-id"
        task_instructions = {task_id: "Open the door."}
        task_cache_names = {task_id: "HSI_open_door"}
        with tempfile.TemporaryDirectory() as directory:
            cache_path = _task_text_embedding_cache_path(
                directory, task_id, task_cache_names[task_id]
            )
            Path(cache_path).parent.mkdir(parents=True)
            torch.save(
                {
                    "version": 3,
                    "source": "HumanoidArena",
                    "task_id": task_id,
                    "task_name": "HSI_open_door",
                    "embedding": torch.zeros(1, 4096),
                },
                cache_path,
            )

            loaded, missing = _load_text_embedding_cache(
                directory,
                task_instructions,
                task_cache_names,
                feature_dim=4096,
            )

        self.assertEqual(loaded, {})
        self.assertEqual(missing, [task_id])

    def test_natural_language_cache_keeps_task_id_lookup_key(self):
        task_id = "HumanoidArena::task-id"
        task_instructions = {task_id: "Open the door."}
        task_cache_names = {task_id: "HSI_open_door"}
        embedding = torch.randn(1, 16)
        with tempfile.TemporaryDirectory() as directory:
            _save_task_text_embedding(
                directory,
                task_id,
                task_cache_names[task_id],
                task_instructions[task_id],
                embedding,
            )

            loaded, missing = _load_text_embedding_cache(
                directory,
                task_instructions,
                task_cache_names,
                feature_dim=16,
            )

        self.assertEqual(missing, [])
        torch.testing.assert_close(loaded[task_id], embedding)

    def test_text_cache_is_grouped_by_source_and_sanitized_task_name(self):
        cache_path = Path(
            _task_text_embedding_cache_path(
                "/cache",
                "UnifoLM_WBT_Dataset::Dex1/task::Open the drawer.",
                "Dex1/task",
            )
        )

        self.assertEqual(cache_path.parent, Path("/cache/UnifoLM_WBT_Dataset"))
        self.assertRegex(cache_path.name, r"^Dex1_task__[0-9a-f]{12}\.pt$")

    def test_unreadable_per_task_cache_is_treated_as_missing(self):
        task_id = "HIW500::sweep_floor"
        task_instructions = {task_id: "Sweep the floor."}
        task_cache_names = {task_id: "sweep_floor"}
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(
                _task_text_embedding_cache_path(
                    directory, task_id, task_cache_names[task_id]
                )
            )
            cache_path.parent.mkdir(parents=True)
            cache_path.write_bytes(b"not a torch cache")

            loaded, missing = _load_text_embedding_cache(
                directory,
                task_instructions,
                task_cache_names,
                feature_dim=4096,
            )

        self.assertEqual(loaded, {})
        self.assertEqual(missing, [task_id])

    def test_restores_complete_training_state(self):
        torch.manual_seed(7)
        source_model = _TinyTrainableModel()
        source_optimizer = torch.optim.AdamW(source_model.trainable.parameters(), lr=0.01)
        source_scheduler = LambdaLR(source_optimizer, lambda step: 0.9 ** step)

        for _ in range(4):
            source_optimizer.zero_grad()
            source_model.trainable(torch.ones(2, 3)).square().mean().backward()
            source_optimizer.step()
            source_scheduler.step()

        trainable_state = {
            name: parameter.detach().clone()
            for name, parameter in source_model.named_parameters()
            if parameter.requires_grad
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint_4"
            checkpoint_path.mkdir()
            torch.save(
                {
                    "global_step": 4,
                    "model": trainable_state,
                    "optimizer": source_optimizer.state_dict(),
                    "scheduler": source_scheduler.state_dict(),
                    "world_size": 4,
                },
                checkpoint_path / "training_state.pt",
            )

            torch.manual_seed(19)
            resumed_model = _TinyTrainableModel()
            resumed_optimizer = torch.optim.AdamW(resumed_model.trainable.parameters(), lr=0.01)
            resumed_scheduler = LambdaLR(resumed_optimizer, lambda step: 0.9 ** step)
            global_step = _load_training_checkpoint(
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
                str(checkpoint_path),
                expected_world_size=4,
            )

        self.assertEqual(global_step, 4)
        self.assertEqual(resumed_scheduler.last_epoch, 4)
        self.assertEqual(resumed_scheduler.get_last_lr(), source_scheduler.get_last_lr())
        for name, parameter in resumed_model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter, trainable_state[name])
        source_optimizer_state = source_optimizer.state_dict()["state"]
        resumed_optimizer_state = resumed_optimizer.state_dict()["state"]
        self.assertEqual(source_optimizer_state.keys(), resumed_optimizer_state.keys())
        for parameter_id in source_optimizer_state:
            torch.testing.assert_close(
                resumed_optimizer_state[parameter_id]["exp_avg"],
                source_optimizer_state[parameter_id]["exp_avg"],
            )

        resumed_optimizer.step()
        resumed_scheduler.step()
        self.assertEqual(resumed_scheduler.last_epoch, 5)

    def test_resume_seed_uses_global_step_offset(self):
        initial_seeds = {
            _worker_seed(42, 0, 4, 8, process_index, worker_id)
            for process_index in range(4)
            for worker_id in range(8)
        }
        resumed_seeds = {
            _worker_seed(42, 40000, 4, 8, process_index, worker_id)
            for process_index in range(4)
            for worker_id in range(8)
        }

        self.assertEqual(len(initial_seeds), 32)
        self.assertEqual(len(resumed_seeds), 32)
        self.assertTrue(initial_seeds.isdisjoint(resumed_seeds))
        self.assertEqual(_process_seed(42, 40000, 4), 160042)

    def test_dataloader_uses_spawn_and_single_batch_prefetch(self):
        config = SimpleNamespace(
            main=SimpleNamespace(
                cpu_workers_num=4,
                batch_size=16,
                seed=42,
                max_steps=100000,
                gradient=SimpleNamespace(grad_accumulation_steps=2),
            )
        )

        with patch("train.DataLoader") as data_loader:
            build_dataloader(
                config,
                train_dataset=object(),
                process_index=2,
                world_size=4,
                resume_step=60000,
            )

        kwargs = data_loader.call_args.kwargs
        self.assertEqual(kwargs["num_workers"], 4)
        self.assertEqual(kwargs["multiprocessing_context"], "spawn")
        self.assertEqual(kwargs["prefetch_factor"], 1)
        self.assertTrue(kwargs["persistent_workers"])
        self.assertIsInstance(kwargs["sampler"], ResumableOrdinalSampler)
        self.assertEqual(kwargs["sampler"].start, 60000 * 16 * 4 * 2)

    def test_spawn_dataloader_can_fetch_a_batch(self):
        config = SimpleNamespace(
            main=SimpleNamespace(cpu_workers_num=2, batch_size=4, seed=42)
        )
        data_loader = build_dataloader(config, train_dataset=_TinyDataset())

        batch = next(iter(data_loader))

        self.assertEqual(batch.shape, (4,))


if __name__ == "__main__":
    unittest.main()
