import unittest
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from data.playback.replay_capture import (
    BODY_NAMES,
    CANONICAL_NAMES,
    CAPTURE_MODE_EXPERT_ALIGNED,
    expert_aligned_frames,
    hand_binary,
    matrix_from_rot6d,
    reference_root_from_source_action,
    rot6d_from_matrix,
    validate_expert_alignment,
    validate_output,
    write_output,
)


class PlaybackReferenceActionTest(unittest.TestCase):
    @staticmethod
    def _expert_episode(length=4):
        body = (
            np.arange(29, dtype=np.float32)[None] / 100
            + np.arange(length, dtype=np.float32)[:, None] / 1000
        )
        hand = (
            np.arange(14, dtype=np.float32)[None] / 200
            + np.arange(length, dtype=np.float32)[:, None] / 2000
        )
        action = np.arange(length * 36, dtype=np.float32).reshape(length, 36) / 1000
        action[:, 31] = np.linspace(0.72, 0.75, length)
        action[:, 32:34] = np.asarray([0.2, -0.1], dtype=np.float32)
        action[:, 35] = np.linspace(0.0, 0.3, length)
        done = np.zeros(length, dtype=bool)
        done[-1] = True
        return {
            "observation.leg_joints": body[:, :15],
            "observation.arm_joints": body[:, 15:],
            "observation.hand_joints": hand,
            "states": np.arange(length * 32, dtype=np.float32).reshape(length, 32),
            "action": action,
            "next.done": done,
            "task_index": np.zeros(length, dtype=np.int64),
        }, body, hand

    def test_reference_root_uses_synced_navigation_command(self):
        source_action = np.zeros((4, 36), dtype=np.float32)
        source_action[:, 31] = [0.74, 0.75, 0.76, 0.77]
        source_action[:, 32:34] = [
            [5.0, 5.0],  # frame zero must still have a zero displacement
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
        source_action[:, 35] = [0.0, 0.0, np.pi / 2, np.pi / 2]

        reference = reference_root_from_source_action(source_action, fps=50)

        np.testing.assert_allclose(
            reference["local_xy_delta"],
            [[0.0, 0.0], [0.02, 0.0], [0.02, 0.02], [0.0, 0.0]],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            reference["root_p_relative"],
            [[0.0, 0.0, 0.0], [0.02, 0.0, 0.01], [0.0, 0.02, 0.02], [0.0, 0.02, 0.03]],
            atol=1e-6,
        )
        np.testing.assert_allclose(reference["root_q_relative"][0], [1, 0, 0, 0], atol=1e-7)
        np.testing.assert_allclose(
            reference["root_q_relative"][2],
            [np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
            atol=1e-6,
        )

    def test_rot6d_round_trip_uses_arena_row_layout(self):
        source_action = np.zeros((3, 36), dtype=np.float32)
        source_action[:, 31] = 0.74
        source_action[:, 35] = [0.0, 0.3, -0.4]
        rotations = reference_root_from_source_action(source_action, fps=50)[
            "rotation_matrices"
        ]

        decoded = matrix_from_rot6d(rot6d_from_matrix(rotations))

        np.testing.assert_allclose(decoded, rotations, atol=1e-6)

    def test_expert_aligned_frames_copy_realized_expert_pose(self):
        episode, body_source, hand = self._expert_episode()
        frames = expert_aligned_frames(episode, fps=50)
        source_indices = {name: index for index, name in enumerate(BODY_NAMES)}
        expected_body = body_source[
            :, [source_indices[name] for name in CANONICAL_NAMES]
        ]

        np.testing.assert_array_equal(frames["joint_q"], expected_body)
        np.testing.assert_array_equal(frames["state"][:, 6:35], expected_body)
        np.testing.assert_array_equal(frames["target_joint_q"], expected_body)
        np.testing.assert_array_equal(frames["action"][:, 9:38], expected_body)
        np.testing.assert_array_equal(frames["hand_q"], hand)
        np.testing.assert_array_equal(frames["target_hand_q"], hand)
        np.testing.assert_array_equal(
            frames["hand_closure"], frames["action_hand_closure"]
        )
        np.testing.assert_array_equal(frames["source_action"], episode["action"])
        np.testing.assert_array_equal(
            frames["action"][:, 38:40],
            np.stack([hand_binary(row) for row in episode["action"]], axis=0),
        )
        np.testing.assert_array_equal(frames["done"], episode["next.done"])
        np.testing.assert_array_equal(
            frames["root_p"][:, 2], episode["action"][:, 31]
        )

    def test_expert_aligned_output_copies_source_video_and_audits_all_fields(self):
        length = 60
        episode, _, _ = self._expert_episode(length=length)
        frames = expert_aligned_frames(episode, fps=50)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_video = root / "source.mp4"
            writer = cv2.VideoWriter(
                str(source_video), cv2.VideoWriter_fourcc(*"mp4v"), 50, (32, 24)
            )
            self.assertTrue(writer.isOpened())
            for frame_index in range(length):
                writer.write(
                    np.full((24, 32, 3), frame_index * 40, dtype=np.uint8)
                )
            writer.release()

            output = root / "output"
            write_output(
                output,
                0,
                "pick up the object",
                {},
                frames,
                [],
                50,
                capture_mode=CAPTURE_MODE_EXPERT_ALIGNED,
                source_video=source_video,
            )
            report = validate_expert_alignment(
                output, episode, source_video, fps=50
            )
            protocol_report = validate_output(output, 50)

            copied_video = (
                output
                / "videos/observation.images.front/chunk-000/file-000.mp4"
            )
            self.assertEqual(source_video.read_bytes(), copied_video.read_bytes())
            self.assertTrue(report["expert_alignment_passed"])
            self.assertEqual(report["expert_alignment_max_error"], 0.0)
            self.assertTrue(protocol_report["quality_passed"])
            self.assertEqual(protocol_report["joint_tracking_rmse_rad"], 0.0)
            info = json.loads((output / "meta/info.json").read_text())
            self.assertEqual(info["capture_mode"], CAPTURE_MODE_EXPERT_ALIGNED)
            self.assertEqual(info["video_frame_count"], length)
            episode_meta = json.loads(
                (output / "meta/episodes.jsonl").read_text()
            )
            self.assertIsNone(episode_meta["success"])


if __name__ == "__main__":
    unittest.main()
