from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
for path in (_THIS_DIR, _PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from sonic2lerobot_clean_64_40 import build_clean_keep_mask


def test_build_clean_keep_mask_filters_frame_index_anomalies() -> None:
    keep_mask, counters = build_clean_keep_mask(
        raw_frame_index=np.array([10, 11, 11, 15], dtype=np.int64),
        realtime_timestamps=np.array([0.00, 0.02, 0.04, 0.06], dtype=np.float64),
        qpos=np.zeros((4, 29), dtype=np.float32),
        action=np.zeros((4, 40), dtype=np.float32),
        control_dt=0.02,
        drop_nonmonotonic_frame_index=True,
        max_frame_index_gap=1,
        drop_nonpositive_realtime_dt=True,
        max_realtime_gap_scale=2.5,
        drop_held_action_moving_state=True,
        held_action_eps=1e-6,
        moving_qpos_threshold=0.02,
    )

    np.testing.assert_array_equal(keep_mask, np.array([True, True, False, False]))
    assert counters == {
        "nonmonotonic_frame_index": 1,
        "large_frame_gap": 1,
        "nonpositive_realtime_dt": 0,
        "large_realtime_gap": 0,
        "held_action_moving_state": 0,
    }


def test_build_clean_keep_mask_filters_timing_and_held_action_anomalies() -> None:
    qpos = np.zeros((5, 29), dtype=np.float32)
    qpos[3, 0] = 0.03
    qpos[4, 0] = 0.03

    action = np.zeros((5, 40), dtype=np.float32)
    action[4, 0] = 0.7

    keep_mask, counters = build_clean_keep_mask(
        raw_frame_index=np.array([100, 101, 102, 103, 104], dtype=np.int64),
        realtime_timestamps=np.array([1.00, 1.02, 1.02, 1.039, 1.089], dtype=np.float64),
        qpos=qpos,
        action=action,
        control_dt=0.02,
        drop_nonmonotonic_frame_index=True,
        max_frame_index_gap=1,
        drop_nonpositive_realtime_dt=True,
        max_realtime_gap_scale=2.0,
        drop_held_action_moving_state=True,
        held_action_eps=1e-6,
        moving_qpos_threshold=0.02,
    )

    np.testing.assert_array_equal(keep_mask, np.array([True, True, False, False, False]))
    assert counters == {
        "nonmonotonic_frame_index": 0,
        "large_frame_gap": 0,
        "nonpositive_realtime_dt": 1,
        "large_realtime_gap": 1,
        "held_action_moving_state": 1,
    }
