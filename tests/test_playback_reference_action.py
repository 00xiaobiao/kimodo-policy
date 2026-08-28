import unittest

import numpy as np

from data.playback.replay_capture import (
    matrix_from_rot6d,
    reference_root_from_source_action,
    rot6d_from_matrix,
)


class PlaybackReferenceActionTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
