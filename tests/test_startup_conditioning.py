import math
import unittest

import torch

from model.kimodo_policy import KimodoPolicy
from motion.feature_utils import compute_vel_angle, compute_vel_xyz


class StartupConditioningTest(unittest.TestCase):
    def test_first_heading_uses_history_only(self):
        history_motion = torch.zeros(3, 4, 2)
        history_mask = torch.tensor(
            [
                [False, False, False, False],
                [False, False, True, True],
                [True, True, True, True],
            ]
        )
        history_motion[1, 2] = torch.tensor([0.0, 1.0])
        history_motion[1, 3] = torch.tensor([-1.0, 0.0])
        history_motion[2, 0] = torch.tensor([0.0, -1.0])
        history_motion[2, 1:] = torch.tensor([1.0, 0.0])

        headings = KimodoPolicy._first_history_heading(
            history_motion,
            history_mask,
            slice(0, 2),
        )

        expected = torch.tensor([0.0, math.pi / 2, -math.pi / 2])
        torch.testing.assert_close(headings, expected)

    def test_left_padded_xyz_velocity_repeats_actual_endpoint(self):
        positions = torch.zeros(1, 6, 1, 3)
        positions[0, 3:, 0, 0] = torch.tensor([10.0, 12.0, 15.0])
        valid_mask = torch.tensor([[False, False, False, True, True, True]])

        velocity = compute_vel_xyz(
            positions,
            fps=1.0,
            valid_mask=valid_mask,
        )

        torch.testing.assert_close(
            velocity[0, :, 0, 0],
            torch.tensor([0.0, 0.0, 0.0, 2.0, 3.0, 3.0]),
        )

    def test_left_padded_angular_velocity_repeats_actual_endpoint(self):
        angles = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.1, 0.3]])
        valid_mask = torch.tensor([[False, False, False, True, True, True]])

        velocity = compute_vel_angle(
            angles,
            fps=1.0,
            valid_mask=valid_mask,
        )

        torch.testing.assert_close(
            velocity,
            torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.2, 0.2]]),
        )

    def test_prefix_lengths_remain_backward_compatible(self):
        positions = torch.zeros(1, 6, 1, 3)
        positions[0, :3, 0, 0] = torch.tensor([10.0, 12.0, 15.0])

        velocity = compute_vel_xyz(
            positions,
            fps=1.0,
            lengths=torch.tensor([3]),
        )

        torch.testing.assert_close(
            velocity[0, :, 0, 0],
            torch.tensor([2.0, 3.0, 3.0, 0.0, 0.0, 0.0]),
        )


if __name__ == "__main__":
    unittest.main()
