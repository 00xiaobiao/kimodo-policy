import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from data.multisource_dataset import (
    EpisodeRecord,
    SOURCE_HIW500,
    SOURCE_HUMANOID_EVERYDAY,
    SOURCE_UNIFOLM,
)
from scripts import audit_invalid_episodes as audit


def _episode(source: str, *, metadata=None, source_length: int = 6) -> EpisodeRecord:
    return EpisodeRecord(
        source=source,
        task_id="task",
        task_name="task",
        instruction="task",
        episode_id="1",
        data_path=Path("episode.parquet"),
        source_length=source_length,
        source_fps=30.0,
        target_fps=30.0,
        video_path=Path("episode.mp4"),
        video_from_timestamp=0.0,
        first_cut=0,
        sample_count=1,
        metadata=dict(metadata or {"row_start": 0, "row_end": source_length}),
    )


class InvalidEpisodeAuditTest(unittest.TestCase):
    def setUp(self):
        audit._initialize_worker(
            {
                SOURCE_UNIFOLM: {"joint_jump_threshold_radians": 0.5},
                SOURCE_HUMANOID_EVERYDAY: {
                    "joint_jump_threshold_radians": 0.5
                },
                SOURCE_HIW500: {"joint_jump_threshold_radians": 0.5},
            },
            action_chunk=2,
        )
        audit._WORKER_READER = MagicMock()

    def test_unifolm_internal_jump_is_counted_as_invalid(self):
        current = np.zeros((6, 36), dtype=np.float32)
        desired = np.zeros((6, 36), dtype=np.float32)
        current[:, 3] = 1.0
        desired[:, 3] = 1.0
        desired[3:, 8] = 0.8
        audit._WORKER_READER.read.return_value = {
            "observation.state.robot_q_current": current,
            "action.robot_q_desired": desired,
            "observation.state.hand_state": np.zeros((6, 2), dtype=np.float32),
            "action.hand_cmd": np.zeros((6, 2), dtype=np.float32),
        }
        episode = _episode(SOURCE_UNIFOLM, metadata={"hand_type": "dex1"})

        result = audit._audit_episode((SOURCE_UNIFOLM, episode))

        self.assertEqual(result["status"], "invalid")
        self.assertIn("joint=0.800 rad", result["reason"])
        self.assertEqual(result["metrics"]["transition_frame"], 2)

    def test_hiw_internal_jump_is_counted_as_invalid(self):
        joint_q = np.zeros((6, 29), dtype=np.float32)
        joint_q[3:, 4] = 0.8
        audit._WORKER_READER.read.return_value = {
            "observation.state": joint_q,
            "observation.state.wbc": np.zeros((6, 23), dtype=np.float32),
            "action": np.zeros((6, 23), dtype=np.float32),
        }
        episode = _episode(SOURCE_HIW500)

        result = audit._audit_episode((SOURCE_HIW500, episode))

        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["metrics"]["transition_frame"], 2)

    def test_everyday_continuous_episode_is_valid(self):
        root_quaternion = np.zeros((6, 4), dtype=np.float32)
        root_quaternion[:, 0] = 1.0
        audit._WORKER_READER.read.return_value = {
            "observation.arm_joints": np.zeros((6, 14), dtype=np.float32),
            "observation.leg_joints": np.zeros((6, 15), dtype=np.float32),
            "observation.hand_joints": np.zeros((6, 14), dtype=np.float32),
            "observation.odometry.position": np.zeros((6, 3), dtype=np.float32),
            "observation.odometry.quat": root_quaternion,
            "action": np.zeros((6, 28), dtype=np.float32),
        }
        episode = _episode(SOURCE_HUMANOID_EVERYDAY)

        result = audit._audit_episode((SOURCE_HUMANOID_EVERYDAY, episode))

        self.assertEqual(result, {"source": SOURCE_HUMANOID_EVERYDAY, "status": "valid"})

    def test_parquet_read_failure_is_reported_separately(self):
        audit._WORKER_READER.read.side_effect = OSError("truncated parquet")
        episode = _episode(SOURCE_HIW500)

        result = audit._audit_episode((SOURCE_HIW500, episode))

        self.assertEqual(result["status"], "error")
        self.assertIn("truncated parquet", result["reason"])


if __name__ == "__main__":
    unittest.main()
