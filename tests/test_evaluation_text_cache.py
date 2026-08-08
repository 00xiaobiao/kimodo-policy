import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.humanoidarena_server import (
    _load_text_embedding_cache,
    _resolve_cached_task,
)


class EvaluationTextCacheTest(unittest.TestCase):
    def test_loads_new_per_task_directory_and_resolves_raw_task_id(self):
        raw_task_id = "Isaac-Move-Football-Single-G129-Dex3-Wholebody"
        task_id = f"HumanoidArena::{raw_task_id}"
        embedding = torch.arange(12, dtype=torch.bfloat16).reshape(1, 12)
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            torch.save(
                {
                    "version": 3,
                    "source": "HumanoidArena",
                    "task_id": task_id,
                    "task_name": "HOI_football",
                    "instruction": "Kick the football into the goal.",
                    "embedding": embedding,
                },
                cache_dir / "HOI_football__test.pt",
            )

            embeddings, instructions, aliases = _load_text_embedding_cache(cache_dir)

        self.assertEqual(set(embeddings), {task_id})
        torch.testing.assert_close(embeddings[task_id], embedding)
        self.assertEqual(instructions[task_id], "Kick the football into the goal.")
        self.assertEqual(_resolve_cached_task(raw_task_id, aliases), task_id)
        self.assertEqual(_resolve_cached_task(task_id, aliases), task_id)
        self.assertEqual(_resolve_cached_task("HOI_football", aliases), task_id)

    def test_keeps_legacy_aggregate_cache_compatible(self):
        task_id = "legacy-task"
        embedding = torch.ones(1, 8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "legacy.pt"
            torch.save(
                {
                    "embeddings": {task_id: embedding},
                    "instructions": {task_id: "Do the task."},
                },
                cache_path,
            )

            embeddings, instructions, aliases = _load_text_embedding_cache(cache_path)

        torch.testing.assert_close(embeddings[task_id], embedding)
        self.assertEqual(instructions[task_id], "Do the task.")
        self.assertEqual(_resolve_cached_task(task_id, aliases), task_id)

    def test_rejects_invalid_new_cache_shape(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "invalid.pt"
            torch.save(
                {
                    "source": "HumanoidArena",
                    "task_id": "HumanoidArena::task",
                    "instruction": "Do the task.",
                    "embedding": torch.zeros(2, 4),
                },
                cache_path,
            )

            with self.assertRaisesRegex(ValueError, r"Expected text embedding \[1, D\]"):
                _load_text_embedding_cache(cache_path)


if __name__ == "__main__":
    unittest.main()
