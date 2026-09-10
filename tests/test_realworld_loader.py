import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

from data.common import EpisodeRecord, SOURCE_REAL_WORLD
from data.realworld_loader import (
    REALWORLD_HAND_CLOSE_POSES,
    REALWORLD_HAND_JOINT_NAMES_14,
    RealWorldAdapter,
    project_realworld_hand_closure,
)


def _packed_names_and_values(closure: np.ndarray) -> tuple[list[str], np.ndarray]:
    names = [f"body_{index}" for index in range(29)] + list(
        reversed(REALWORLD_HAND_JOINT_NAMES_14)
    )
    values = np.zeros((closure.shape[0], 43), dtype=np.float32)
    natural_hand_q = (
        closure[:, :, None] * REALWORLD_HAND_CLOSE_POSES[None]
    ).reshape(-1, 14)
    name_to_index = {name: index for index, name in enumerate(names)}
    for hand_index, name in enumerate(REALWORLD_HAND_JOINT_NAMES_14):
        values[:, name_to_index[name]] = natural_hand_q[:, hand_index]
    return names, values


def _episode(field_names, available_columns, hand_feature_names=None):
    return EpisodeRecord(
        source=SOURCE_REAL_WORLD,
        task_id="RealWorld::data",
        task_name="data",
        instruction="Pack the bottles.",
        episode_id="0",
        data_path=Path("episode.parquet"),
        source_length=4,
        source_fps=30,
        target_fps=30,
        video_path=Path("episode.mp4"),
        video_from_timestamp=0,
        first_cut=0,
        sample_count=1,
        metadata={
            "row_start": 0,
            "row_end": 4,
            "field_names": field_names,
            "optional_field_names": {
                "root_target_valid_field": "action.root_target_valid",
                "root_target_discontinuous_field": "action.root_target_discontinuous",
            },
            "available_columns": tuple(available_columns),
            "optional_columns": (),
            "hand_feature_names": hand_feature_names or {},
        },
    )


def _adapter_and_body_table(selection):
    adapter = RealWorldAdapter.__new__(RealWorldAdapter)
    adapter.selection = selection
    adapter.action_chunk = 2
    adapter.reader = MagicMock()
    adapter._decoder = MagicMock()
    adapter._motion_feature_mask = MagicMock(
        return_value=torch.ones(417, dtype=torch.bool)
    )
    adapter._finalize_motion = MagicMock(return_value={"loaded": True})

    decoded = {
        "local_rot_mats": torch.eye(3).reshape(1, 1, 3, 3).repeat(4, 1, 1, 1),
        "root_positions": torch.zeros(4, 3),
    }
    adapter._decoder.return_value.decode_joint_configuration_pose.return_value = decoded
    table = {
        "observation.joint_q": np.zeros((4, 29), dtype=np.float32),
        "observation.root_q_relative": np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (4, 1)
        ),
        "action.joint_q": np.zeros((4, 29), dtype=np.float32),
        "action.root_p": np.zeros((4, 3), dtype=np.float32),
        "action.root_z": np.full(4, 0.8, dtype=np.float32),
        "action.root_q": np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (4, 1)
        ),
    }
    return adapter, table


class RealWorldAdapterTest(unittest.TestCase):
    def test_projection_uses_feature_names_and_realworld_close_pose(self):
        expected = np.asarray([[0.25, 0.75], [1.0, 0.5]], dtype=np.float32)
        names, values = _packed_names_and_values(expected)

        np.testing.assert_allclose(
            project_realworld_hand_closure(values, names), expected, atol=1e-6
        )

        duplicate_names = names.copy()
        duplicate_names[0] = duplicate_names[1]
        with self.assertRaisesRegex(ValueError, "duplicate feature names"):
            project_realworld_hand_closure(values, duplicate_names)

    def test_continuous_mode_uses_measured_state_and_wbc_target(self):
        adapter, table = _adapter_and_body_table({"hand_control_mode": "continuous"})
        observed_closure = np.tile(
            np.asarray([0.25, 0.5], dtype=np.float32), (4, 1)
        )
        target_closure = np.tile(
            np.asarray([0.75, 1.0], dtype=np.float32), (4, 1)
        )
        observed_names, table["observation.state"] = _packed_names_and_values(
            observed_closure
        )
        target_names, table["action.wbc"] = _packed_names_and_values(target_closure)
        adapter.reader.read.return_value = table
        field_names = {
            "observed_joint_field": "observation.joint_q",
            "observed_root_orientation_field": "observation.root_q_relative",
            "observed_hand_state_field": "observation.state",
            "target_joint_field": "action.joint_q",
            "target_root_position_field": "action.root_p",
            "target_root_height_field": "action.root_z",
            "target_root_orientation_field": "action.root_q",
            "target_hand_wbc_field": "action.wbc",
        }
        episode = _episode(
            field_names,
            table,
            {"observed": observed_names, "target": target_names},
        )

        result = adapter.load_episode(episode)

        self.assertEqual(result, {"loaded": True})
        finalize_args = adapter._finalize_motion.call_args.args
        np.testing.assert_allclose(finalize_args[5], observed_closure, atol=1e-6)
        np.testing.assert_allclose(finalize_args[6], target_closure, atol=1e-6)
        np.testing.assert_array_equal(finalize_args[7], np.ones((4, 2), dtype=bool))
        np.testing.assert_array_equal(finalize_args[8], np.ones((4, 2), dtype=bool))
        self.assertEqual(
            adapter._finalize_motion.call_args.kwargs["hand_resampling"], "linear"
        )
        self.assertIn(
            "action.wbc hand closure",
            adapter._finalize_motion.call_args.kwargs["target_motion_source"],
        )

    def test_binary_mode_keeps_existing_hand_fields(self):
        adapter, table = _adapter_and_body_table({})
        observed_hand = np.tile(
            np.asarray([0.0, 1.0], dtype=np.float32), (4, 1)
        )
        target_hand = np.tile(
            np.asarray([1.0, 0.0], dtype=np.float32), (4, 1)
        )
        table["observation.hand_binary"] = observed_hand
        table["action.hand_binary"] = target_hand
        adapter.reader.read.return_value = table
        field_names = {
            "observed_joint_field": "observation.joint_q",
            "observed_root_orientation_field": "observation.root_q_relative",
            "observed_hand_field": "observation.hand_binary",
            "target_joint_field": "action.joint_q",
            "target_root_position_field": "action.root_p",
            "target_root_height_field": "action.root_z",
            "target_root_orientation_field": "action.root_q",
            "target_hand_field": "action.hand_binary",
        }
        episode = _episode(field_names, table)

        result = adapter.load_episode(episode)

        self.assertEqual(result, {"loaded": True})
        finalize_args = adapter._finalize_motion.call_args.args
        np.testing.assert_array_equal(finalize_args[5], observed_hand)
        np.testing.assert_array_equal(finalize_args[6], target_hand)
        self.assertEqual(
            adapter._finalize_motion.call_args.kwargs["hand_resampling"], "binary"
        )

    def test_continuous_observation_with_binary_action(self):
        adapter, table = _adapter_and_body_table(
            {
                "hand_control_mode": "binary",
                "hand_observation_mode": "continuous",
            }
        )
        observed_closure = np.asarray(
            [[0.0, 0.25], [0.1, 0.5], [0.2, 0.75], [0.3, 1.0]],
            dtype=np.float32,
        )
        observed_names, table["observation.state"] = _packed_names_and_values(
            observed_closure
        )
        target_hand = np.asarray(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            dtype=np.float32,
        )
        table["action.hand_binary"] = target_hand
        adapter.reader.read.return_value = table
        field_names = {
            "observed_joint_field": "observation.joint_q",
            "observed_root_orientation_field": "observation.root_q_relative",
            "observed_hand_state_field": "observation.state",
            "target_joint_field": "action.joint_q",
            "target_root_position_field": "action.root_p",
            "target_root_height_field": "action.root_z",
            "target_root_orientation_field": "action.root_q",
            "target_hand_field": "action.hand_binary",
        }
        episode = _episode(
            field_names,
            table,
            {"observed": observed_names},
        )

        result = adapter.load_episode(episode)

        self.assertEqual(result, {"loaded": True})
        finalize_args = adapter._finalize_motion.call_args.args
        np.testing.assert_allclose(finalize_args[5], observed_closure, atol=1e-6)
        np.testing.assert_array_equal(finalize_args[6], target_hand)
        finalize_kwargs = adapter._finalize_motion.call_args.kwargs
        self.assertEqual(finalize_kwargs["observed_hand_resampling"], "linear")
        self.assertEqual(finalize_kwargs["target_hand_resampling"], "binary")
        self.assertNotIn(
            "action.wbc hand closure", finalize_kwargs["target_motion_source"]
        )


if __name__ == "__main__":
    unittest.main()
