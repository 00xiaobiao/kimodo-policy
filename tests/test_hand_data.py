import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from data.datasetloader import HumanoidArenaDataset
from evaluation.humanoidarena_server import _select_hand_execution_prefix
from motion.g1_reference import resample_hand_binary, resample_hand_binary_chunk


class HandDataTest(unittest.TestCase):
    def test_dataset_extracts_action_dimensions_38_and_39(self):
        dataset = HumanoidArenaDataset.__new__(HumanoidArenaDataset)
        dataset.episodes = [
            {
                "data_path": Path("episode.parquet"),
                "data_from_index": 0,
                "data_to_index": 3,
                "source_length": 3,
                "source_fps": 30.0,
                "target_fps": 30.0,
            }
        ]
        dataset._episode_cache = OrderedDict()
        dataset._data_file_cache = {}
        dataset.episode_cache_size = 2
        decoder = MagicMock()
        decoder.decode.return_value = {
            "local_rot_mats": torch.zeros(3, 1, 3, 3),
            "root_positions": torch.zeros(3, 3),
        }
        representation = MagicMock(return_value=torch.zeros(3, 417))
        dataset._get_motion_tools = MagicMock(
            return_value=(decoder, representation)
        )
        actions = np.zeros((3, 40), dtype=np.float32)
        actions[:, 38:40] = np.asarray([[0, 1], [1, 1], [1, 0]])
        table = MagicMock()
        table.to_pydict.return_value = {
            "index": np.arange(3),
            "action": list(actions),
        }

        with patch("data.datasetloader.pq.read_table", return_value=table), patch(
            "data.datasetloader.resample_motion",
            return_value=(
                decoder.decode.return_value["local_rot_mats"],
                decoder.decode.return_value["root_positions"],
            ),
        ):
            episode = dataset._load_episode_motion(0)

        torch.testing.assert_close(
            episode["hand_binary"], torch.from_numpy(actions[:, 38:40])
        )

    def test_sequence_resampling_is_binary_nearest_neighbor(self):
        source = torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.float32)

        target = resample_hand_binary(source, source_fps=30, target_fps=50)

        self.assertEqual(tuple(target.shape), (4, 2))
        self.assertTrue(torch.logical_or(target == 0, target == 1).all())
        torch.testing.assert_close(target[0], source[0])
        torch.testing.assert_close(target[-1], source[-1])

    def test_invalid_non_binary_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            resample_hand_binary(
                torch.tensor([[0.0, 0.5], [1.0, 0.0]]),
                source_fps=30,
                target_fps=30,
            )

    def test_fifteen_executed_model_frames_become_twenty_five_control_frames(self):
        prediction = torch.cat(
            (torch.zeros(10, 2), torch.ones(40, 2)), dim=0
        )

        executed = _select_hand_execution_prefix(prediction, execution_frames=15)
        control = resample_hand_binary_chunk(
            executed,
            source_fps=30,
            target_fps=50,
            previous_hand_binary=torch.zeros(2),
        )

        self.assertEqual(tuple(executed.shape), (15, 2))
        self.assertEqual(tuple(control.shape), (25, 2))
        self.assertTrue(torch.logical_or(control == 0, control == 1).all())


if __name__ == "__main__":
    unittest.main()
