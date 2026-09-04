import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from data.common import EpisodeRecord, SOURCE_SIMPLE
from data.simple_hand import SIMPLE_RIGHT_HAND_CLOSE
from data.simple_loader import SimpleReplayAdapter


def _fixed(values, width):
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.asarray(values, dtype=np.float32).reshape(-1)), width
    )


class SimpleReplayAdapterTest(unittest.TestCase):
    def _write_episode(self, root: Path, task_name: str, episode_index: int = 7):
        episode_root = root / task_name / f"episode_{episode_index:06d}"
        data_path = episode_root / "data/chunk-000/file-000.parquet"
        video_path = episode_root / "videos/observation.images.front/chunk-000/file-000.mp4"
        metadata_path = episode_root / "meta/episodes/chunk-000.parquet"
        data_path.parent.mkdir(parents=True)
        video_path.parent.mkdir(parents=True)
        metadata_path.parent.mkdir(parents=True)
        length = 4
        state = np.zeros((length, 64), dtype=np.float32)
        state[:, :6] = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
        action = np.zeros((length, 40), dtype=np.float32)
        action[-1, 39] = 1
        pq.write_table(
            pa.table({"observation.state": _fixed(state, 64), "action": _fixed(action, 40)}),
            data_path,
        )
        video_path.write_bytes(b"test video")
        pq.write_table(
            pa.table({"task_index": pa.array([0], type=pa.int64()), "task": pa.array(["Pick up the object."])}),
            episode_root / "meta/tasks.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    "episode_index": pa.array([episode_index], type=pa.int64()),
                    "tasks": pa.array([[0]], type=pa.list_(pa.int64())),
                    "length": pa.array([length], type=pa.int64()),
                    "data/chunk_index": pa.array([0], type=pa.int64()),
                    "data/file_index": pa.array([0], type=pa.int64()),
                    "videos/observation.images.front/chunk_index": pa.array([0], type=pa.int64()),
                    "videos/observation.images.front/file_index": pa.array([0], type=pa.int64()),
                    "videos/observation.images.front/from_timestamp": pa.array([0.0], type=pa.float32()),
                }
            ),
            metadata_path,
        )
        (episode_root / "meta/info.json").write_text(
            json.dumps(
                {
                    "fps": 50,
                    "vla_protocol": {"schema": "unitree_g1_gmt_refpose_v3_1"},
                    "features": {
                        "observation.images.front": {"dtype": "video", "shape": [4, 6, 3]},
                        "observation.state": {"shape": [64]},
                        "action": {"shape": [40]},
                    },
                }
            )
        )
        (episode_root / "validation.json").write_text(
            json.dumps(
                {
                    "frames": length,
                    "source_frames": length,
                    "recorded_frames": length,
                    "state_dim": 64,
                    "action_dim": 40,
                    "kimodo_dim": 417,
                    "target_kimodo_dim": 417,
                }
            )
        )
        return episode_root, state, action

    def test_discovery_uses_single_episode_row_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_episode(root, "G1WholebodyBendPickTeleop-v0")
            adapter = SimpleReplayAdapter(
                root=root,
                selection={"task": "G1WholebodyBendPickTeleop-v0"},
                target_fps=30,
                action_chunk=2,
            )

            self.assertEqual(adapter.source_name, SOURCE_SIMPLE)
            self.assertEqual(len(adapter.episodes), 1)
            episode = adapter.episodes[0]
            self.assertEqual(episode.task_id, "Simple::G1WholebodyBendPickTeleop-v0")
            self.assertEqual(episode.instruction, "Pick up the object.")
            self.assertEqual(episode.metadata, {"row_start": 0, "row_end": 4, "sample_stride": 1})
            self.assertEqual(episode.target_length, 3)
            self.assertEqual(episode.sample_count, 3)

    def test_discovery_skips_episode_without_completed_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode_root, _, _ = self._write_episode(
                root, "G1WholebodyBendPickTeleop-v0"
            )
            (episode_root / "validation.json").unlink()

            adapter = SimpleReplayAdapter(
                root=root,
                selection={"task": "G1WholebodyBendPickTeleop-v0"},
                target_fps=30,
                action_chunk=2,
            )

            self.assertEqual(adapter.episodes, [])

    def test_discovery_accepts_successful_source_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode_root, _, _ = self._write_episode(
                root, "G1WholebodyCloseDoorTeleop-v0"
            )
            report_path = episode_root / "validation.json"
            report = json.loads(report_path.read_text())
            report["source_frames"] = report["recorded_frames"] + 3
            report_path.write_text(json.dumps(report))

            adapter = SimpleReplayAdapter(
                root=root,
                selection={"task": "G1WholebodyCloseDoorTeleop-v0"},
                target_fps=30,
                action_chunk=2,
            )

            self.assertEqual(len(adapter.episodes), 1)
            self.assertEqual(adapter.episodes[0].source_length, 4)

    def test_load_episode_uses_replay_actions_for_observed_and_target_hands(self):
        adapter = SimpleReplayAdapter.__new__(SimpleReplayAdapter)
        state = np.zeros((4, 64), dtype=np.float32)
        state[:, :6] = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
        action = np.zeros((4, 40), dtype=np.float32)
        action[:, 38:40] = np.asarray([[0, 0], [0, 1], [0, 1], [0, 0]], dtype=np.float32)
        adapter.reader = MagicMock()
        adapter.reader.read.return_value = {"observation.state": state, "action": action}
        decoded = {
            "local_rot_mats": torch.eye(3).reshape(1, 1, 3, 3).repeat(4, 1, 1, 1),
            "root_positions": torch.zeros(4, 3),
        }
        decoder = MagicMock()
        decoder.decode_joint_configuration_pose.return_value = decoded
        decoder.decode_action_pose.return_value = decoded
        adapter._decoder = MagicMock(return_value=decoder)
        adapter._motion_feature_mask = MagicMock(return_value=torch.ones(417, dtype=torch.bool))
        adapter._finalize_motion = MagicMock(return_value={"loaded": True})
        episode = EpisodeRecord(
            source=SOURCE_SIMPLE,
            task_id="Simple::task",
            task_name="task",
            instruction="Pick up the object.",
            episode_id="task:000000",
            data_path=Path("episode.parquet"),
            source_length=4,
            source_fps=50,
            target_fps=30,
            video_path=Path("episode.mp4"),
            video_from_timestamp=0,
            first_cut=0,
            sample_count=1,
            metadata={"row_start": 0, "row_end": 4},
        )

        result = adapter.load_episode(episode)

        self.assertEqual(result, {"loaded": True})
        finalize_args = adapter._finalize_motion.call_args.args
        np.testing.assert_array_equal(finalize_args[5], action[:, 38:40])
        np.testing.assert_array_equal(finalize_args[6], action[:, 38:40])
        np.testing.assert_array_equal(finalize_args[7], np.ones((4, 2), dtype=bool))
        np.testing.assert_array_equal(finalize_args[8], np.ones((4, 2), dtype=bool))

    def test_load_episode_continuous_uses_measured_history_and_source_target(self):
        adapter = SimpleReplayAdapter.__new__(SimpleReplayAdapter)
        adapter.selection = {"hand_control_mode": "continuous"}
        state = np.zeros((4, 64), dtype=np.float32)
        state[:, :6] = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
        action = np.zeros((4, 40), dtype=np.float32)
        source_action = np.zeros((4, 36), dtype=np.float32)
        source_action[:, 7:14] = 0.75 * SIMPLE_RIGHT_HAND_CLOSE
        hand_q = np.zeros((4, 14), dtype=np.float32)
        hand_q[:, 7:14] = 0.25 * SIMPLE_RIGHT_HAND_CLOSE
        adapter.reader = MagicMock()
        adapter.reader.read.return_value = {
            "observation.state": state,
            "action": action,
            "observation.hand_q": hand_q,
            "source.action": source_action,
        }
        decoded = {
            "local_rot_mats": torch.eye(3).reshape(1, 1, 3, 3).repeat(4, 1, 1, 1),
            "root_positions": torch.zeros(4, 3),
        }
        decoder = MagicMock()
        decoder.decode_joint_configuration_pose.return_value = decoded
        decoder.decode_action_pose.return_value = decoded
        adapter._decoder = MagicMock(return_value=decoder)
        adapter._motion_feature_mask = MagicMock(return_value=torch.ones(417, dtype=torch.bool))
        adapter._finalize_motion = MagicMock(return_value={"loaded": True})
        episode = EpisodeRecord(
            source=SOURCE_SIMPLE,
            task_id="Simple::task",
            task_name="task",
            instruction="Pick up the object.",
            episode_id="task:000000",
            data_path=Path("episode.parquet"),
            source_length=4,
            source_fps=50,
            target_fps=30,
            video_path=Path("episode.mp4"),
            video_from_timestamp=0,
            first_cut=0,
            sample_count=1,
            metadata={"row_start": 0, "row_end": 4},
        )

        result = adapter.load_episode(episode)

        self.assertEqual(result, {"loaded": True})
        finalize_args = adapter._finalize_motion.call_args.args
        expected_observed = np.tile(np.asarray([0.0, 0.25], dtype=np.float32), (4, 1))
        expected_target = np.tile(np.asarray([0.0, 0.75], dtype=np.float32), (4, 1))
        np.testing.assert_allclose(finalize_args[5], expected_observed)
        np.testing.assert_allclose(finalize_args[6], expected_target)
        self.assertEqual(
            adapter._finalize_motion.call_args.kwargs["hand_resampling"], "linear"
        )

    def test_finalize_motion_appends_success_terminal_hold_before_representation(self):
        adapter = SimpleReplayAdapter.__new__(SimpleReplayAdapter)
        adapter.action_chunk = 3
        adapter.target_fps = 30
        representation = MagicMock()

        def encode_motion(local_rotations, root_positions, to_normalize=False):
            self.assertFalse(to_normalize)
            encoded = torch.zeros(root_positions.shape[0], 417)
            encoded[:, :3] = root_positions
            encoded[:, 3] = torch.arange(root_positions.shape[0])
            return encoded

        representation.side_effect = encode_motion
        adapter._motion_representation = MagicMock(return_value=representation)
        episode = EpisodeRecord(
            source=SOURCE_SIMPLE,
            task_id="Simple::task",
            task_name="task",
            instruction="Pick up the object.",
            episode_id="task:000000",
            data_path=Path("episode.parquet"),
            source_length=2,
            source_fps=30,
            target_fps=30,
            video_path=Path("episode.mp4"),
            video_from_timestamp=0,
            first_cut=0,
            sample_count=2,
            metadata={"row_start": 0, "row_end": 2},
        )
        local_rotations = torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 1, 1, 1)
        observed_root = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        target_root = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        observed_hand = torch.tensor([[0.0, 0.0], [0.25, 0.5]])
        target_hand = torch.tensor([[0.0, 0.0], [0.75, 1.0]])
        hand_valid = torch.ones(2, 2, dtype=torch.bool)

        result = adapter._finalize_motion(
            episode,
            local_rotations,
            observed_root,
            local_rotations,
            target_root,
            observed_hand,
            target_hand,
            hand_valid,
            hand_valid,
            observed_motion_valid=torch.ones(417, dtype=torch.bool),
            target_motion_source="action",
            hand_resampling="linear",
        )

        self.assertEqual(tuple(result["target_motion"].shape), (4, 417))
        torch.testing.assert_close(
            result["target_motion"][:, 0],
            torch.tensor([0.0, 2.0, 2.0, 2.0]),
        )
        torch.testing.assert_close(
            result["target_motion"][:, 3],
            torch.tensor([0.0, 1.0, 1.0, 1.0]),
        )
        torch.testing.assert_close(
            result["target_hand"],
            torch.tensor(
                [[0.0, 0.0], [0.75, 1.0], [0.75, 1.0], [0.75, 1.0]]
            ),
        )
        self.assertTrue(result["target_hand_valid"].all())
        target_representation_call = representation.call_args_list[1]
        torch.testing.assert_close(
            target_representation_call.args[1],
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
