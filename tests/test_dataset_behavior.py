import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import torch

from data.datasetloader import HumanoidArenaDataset


class DatasetBehaviorTest(unittest.TestCase):
    def test_individual_task_id_resolves_to_natural_language(self):
        instruction = HumanoidArenaDataset._resolve_task_instruction(
            "Isaac-Move-Open-Door-G129-Dex3-Wholebody",
            {0: "Open the door."},
            is_aggregate=False,
        )

        self.assertEqual(instruction, "Open the door.")

    def test_aggregate_task_id_resolves_through_task_index(self):
        task_text_by_index = {
            0: "Put the hammer into the basket.",
            5: "Open the door.",
            7: "Move to the yellow marked area.",
        }

        instruction = HumanoidArenaDataset._resolve_task_instruction(
            "Isaac-Move-Open-Door-G129-Dex3-Wholebody",
            task_text_by_index,
            is_aggregate=True,
        )

        self.assertEqual(instruction, "Open the door.")

    def test_embeddings_are_indexed_by_task_id_not_instruction(self):
        dataset = HumanoidArenaDataset.__new__(HumanoidArenaDataset)
        dataset._task_instructions = {"task-id": "Open the door."}
        embedding = np.zeros((1, 4096), dtype=np.float32)

        dataset.set_text_embeddings({"task-id": embedding})

        self.assertIn("task-id", dataset._text_embeddings)
        self.assertNotIn("Open the door.", dataset._text_embeddings)

    def test_individual_roots_replace_duplicate_aggregate_collections(self):
        dataset_root = Path("/datasets/humanoid")
        individual = [
            dataset_root / "HSI_sit_sofa" / "sonic_refpose_v3_1",
            dataset_root / "HOI_football" / "sonic_refpose_v3_1",
        ]
        aggregate = [
            dataset_root
            / "HumanoidArena_merged_datasets_v3_1"
            / "all_16_refpose_v3_1",
            dataset_root
            / "HumanoidArena_merged_datasets_v3_1"
            / "sonic_8_refpose_v3_1",
        ]

        selected = HumanoidArenaDataset._prefer_individual_task_roots(
            dataset_root,
            individual + aggregate,
        )

        self.assertEqual(selected, individual)

    def test_aggregate_roots_remain_usable_when_they_are_the_only_source(self):
        dataset_root = Path("/datasets/humanoid")
        aggregate = [
            dataset_root
            / "HumanoidArena_merged_datasets_v3_1"
            / "all_16_refpose_v3_1"
        ]

        selected = HumanoidArenaDataset._prefer_individual_task_roots(
            dataset_root,
            aggregate,
        )

        self.assertEqual(selected, aggregate)

    def test_direct_backend_aggregate_root_is_recognized(self):
        dataset_root = Path(
            "/datasets/humanoid/HumanoidArena_merged_datasets_v3_1/sonic_8_refpose_v3_1"
        )

        self.assertTrue(
            HumanoidArenaDataset._is_aggregate_task_root(dataset_root, dataset_root)
        )

    def test_backend_aggregate_roots_replace_combined_duplicate(self):
        dataset_root = Path("/datasets/humanoid")
        aggregate_root = dataset_root / "HumanoidArena_merged_datasets_v3_1"
        combined = aggregate_root / "all_16_refpose_v3_1"
        backend_roots = [
            aggregate_root / "sonic_8_refpose_v3_1",
            aggregate_root / "twist2_8_refpose_v3_1",
        ]

        selected = HumanoidArenaDataset._prefer_individual_task_roots(
            dataset_root,
            [combined, *backend_roots],
        )

        self.assertEqual(selected, backend_roots)

    def test_aggregate_backend_prefilter_accepts_task_specific_selection(self):
        dataset = HumanoidArenaDataset.__new__(HumanoidArenaDataset)
        dataset.dataset_selection = {"HSI_sit_sofa": {"sonic"}}

        self.assertTrue(dataset._backend_may_be_selected("sonic"))
        self.assertFalse(dataset._backend_may_be_selected("twist2"))

    def test_video_frame_read_retries_transient_pyav_failure(self):
        dataset = HumanoidArenaDataset.__new__(HumanoidArenaDataset)
        frame = MagicMock()
        frame.pts = None
        frame.to_ndarray.side_effect = [
            av.error.BlockingIOError(11, "Resource temporarily unavailable"),
            np.zeros((4, 6, 3), dtype=np.uint8),
        ]
        stream = MagicMock()
        container = MagicMock()
        container.streams.video = [stream]
        container.decode.return_value = [frame]
        container.__enter__.return_value = container

        with patch("data.datasetloader.av.open", return_value=container) as open_video, patch(
            "data.datasetloader.time.sleep"
        ) as sleep:
            image = dataset._read_video_frame(Path("sample.mp4"), 1.5)

        self.assertEqual(image.shape, (3, 4, 6))
        self.assertEqual(image.dtype, torch.uint8)
        self.assertTrue(image.is_contiguous())
        self.assertEqual(open_video.call_count, 2)
        self.assertEqual(stream.codec_context.thread_count, 1)
        sleep.assert_called_once_with(0.05)


if __name__ == "__main__":
    unittest.main()
