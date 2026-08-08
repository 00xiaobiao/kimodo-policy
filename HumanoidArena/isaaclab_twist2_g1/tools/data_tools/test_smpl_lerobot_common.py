from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
for path in (_THIS_DIR, _PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from smpl_lerobot_common import (
    VLA_ACTION_DIM,
    VLA_STATE_DIM,
    build_sonic_actions_from_recording,
    build_twist2_actions_from_recording,
    extract_canonical_state,
    get_hand_binary_arrays,
)


def _open_npz(tmp_path: Path, **arrays) -> np.lib.npyio.NpzFile:
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, **arrays)
    return np.load(npz_path, allow_pickle=True)


def test_get_hand_binary_prefers_full_canonical_action(tmp_path: Path) -> None:
    vla_action = np.zeros((2, VLA_ACTION_DIM), dtype=np.float32)
    vla_action[:, 38:40] = np.array([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32)
    legacy_token = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with _open_npz(
        tmp_path,
        vla_action=vla_action,
        vla_action_hand_binary=legacy_token,
    ) as data:
        left, right = get_hand_binary_arrays(data, num_frames=2)

    np.testing.assert_allclose(left, np.array([0.2, 0.9], dtype=np.float32))
    np.testing.assert_allclose(right, np.array([0.8, 0.1], dtype=np.float32))


def test_extract_canonical_state_supports_split_fields(tmp_path: Path) -> None:
    num_frames = 3
    root_rot6d = np.arange(num_frames * 6, dtype=np.float32).reshape(num_frames, 6)
    dof_pos = np.arange(num_frames * 29, dtype=np.float32).reshape(num_frames, 29)
    dof_vel = -dof_pos.copy()

    with _open_npz(
        tmp_path,
        vla_state_root_rot6d=root_rot6d,
        vla_state_dof_pos_29=dof_pos,
        vla_state_dof_vel_29=dof_vel,
    ) as data:
        state = extract_canonical_state(
            data=data,
            root_orientation=np.zeros((num_frames, 4), dtype=np.float32),
            joint_pos=np.zeros((num_frames, 29), dtype=np.float32),
            joint_vel=np.zeros((num_frames, 29), dtype=np.float32),
        )

    assert state.shape == (num_frames, VLA_STATE_DIM)
    np.testing.assert_allclose(state[:, :6], root_rot6d)
    np.testing.assert_allclose(state[:, 6:35], dof_pos)
    np.testing.assert_allclose(state[:, 35:64], dof_vel)


def test_build_twist2_actions_supports_split_canonical_fields(tmp_path: Path) -> None:
    num_frames = 2
    root_xy_delta = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    root_z = np.array([[0.8], [0.9]], dtype=np.float32)
    root_rot6d = np.arange(num_frames * 6, dtype=np.float32).reshape(num_frames, 6)
    joint_pos = np.arange(num_frames * 29, dtype=np.float32).reshape(num_frames, 29)
    hand_binary = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    with _open_npz(
        tmp_path,
        vla_action_root_xy_delta=root_xy_delta,
        vla_action_root_z=root_z,
        vla_action_root_rot6d=root_rot6d,
        vla_action_joint_pos_29=joint_pos,
        vla_action_hand_binary_2=hand_binary,
    ) as data:
        action = build_twist2_actions_from_recording(data, num_frames=num_frames, control_dt=1.0 / 30.0)

    assert action.shape == (num_frames, VLA_ACTION_DIM)
    np.testing.assert_allclose(action[:, :2], root_xy_delta)
    np.testing.assert_allclose(action[:, 2:3], root_z)
    np.testing.assert_allclose(action[:, 3:9], root_rot6d)
    np.testing.assert_allclose(action[:, 9:38], joint_pos)
    np.testing.assert_allclose(action[:, 38:40], hand_binary)


def test_build_twist2_actions_old_protocol_ignores_missing_human_smplx(tmp_path: Path) -> None:
    num_frames = 3
    robot_action_mimic = np.zeros((num_frames, 35), dtype=np.float32)
    robot_action_mimic[:, 2] = np.array([0.8, 0.81, 0.82], dtype=np.float32)
    robot_action_mimic[:, 6:35] = np.arange(num_frames * 29, dtype=np.float32).reshape(num_frames, 29)
    human_smplx_data = np.array(json.dumps([None, None, None]))

    with _open_npz(
        tmp_path,
        robot_action_mimic=robot_action_mimic,
        pico_left_grip_binary=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        pico_right_grip_binary=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        human_smplx_data=human_smplx_data,
    ) as data:
        action = build_twist2_actions_from_recording(data, num_frames=num_frames, control_dt=1.0 / 30.0)

    assert action.shape == (num_frames, VLA_ACTION_DIM)
    np.testing.assert_allclose(action[:, 2], np.array([0.8, 0.81, 0.82], dtype=np.float32))
    np.testing.assert_allclose(action[:, 38], np.array([0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(action[:, 39], np.array([1.0, 0.0, 1.0], dtype=np.float32))


def test_build_sonic_actions_supports_split_canonical_fields(tmp_path: Path) -> None:
    num_frames = 2
    root_xy_delta = np.array([[0.0, 0.0], [0.05, -0.1]], dtype=np.float32)
    root_z = np.array([[0.82], [0.83]], dtype=np.float32)
    root_rot6d = np.arange(num_frames * 6, dtype=np.float32).reshape(num_frames, 6)
    joint_pos = np.arange(num_frames * 29, dtype=np.float32).reshape(num_frames, 29)
    hand_binary = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    with _open_npz(
        tmp_path,
        vla_action_root_xy_delta=root_xy_delta,
        vla_action_root_z=root_z,
        vla_action_root_rot6d=root_rot6d,
        vla_action_joint_pos_29=joint_pos,
        vla_action_hand_binary_2=hand_binary,
    ) as data:
        action = build_sonic_actions_from_recording(data, num_frames=num_frames)

    assert action.shape == (num_frames, VLA_ACTION_DIM)
    np.testing.assert_allclose(action[:, :2], root_xy_delta)
    np.testing.assert_allclose(action[:, 2:3], root_z)
    np.testing.assert_allclose(action[:, 3:9], root_rot6d)
    np.testing.assert_allclose(action[:, 9:38], joint_pos)
    np.testing.assert_allclose(action[:, 38:40], hand_binary)


def test_build_sonic_actions_applies_heading_alignment_to_root(tmp_path: Path) -> None:
    num_frames = 1
    root_xy_delta = np.array([[1.0, 0.0]], dtype=np.float32)
    root_z = np.array([[0.82]], dtype=np.float32)
    # identity rotation: x=[1,0,0], y=[0,1,0]
    root_rot6d = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    joint_pos = np.zeros((num_frames, 29), dtype=np.float32)
    hand_binary = np.zeros((num_frames, 2), dtype=np.float32)
    # +90deg yaw align quaternion in wxyz.
    heading_align = np.array([[0.70710677, 0.0, 0.0, 0.70710677]], dtype=np.float32)

    with _open_npz(
        tmp_path,
        vla_action_root_xy_delta=root_xy_delta,
        vla_action_root_z=root_z,
        vla_action_root_rot6d=root_rot6d,
        vla_action_joint_pos_29=joint_pos,
        vla_action_hand_binary_2=hand_binary,
        anchor_heading_align_quat_wxyz=heading_align,
        anchor_use_heading_align=np.array([[True]], dtype=np.bool_),
    ) as data:
        action = build_sonic_actions_from_recording(
            data,
            num_frames=num_frames,
            align_heading_targets=True,
        )

    np.testing.assert_allclose(action[:, :2], np.array([[0.0, 1.0]], dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(
        action[:, 3:9],
        np.array([[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        atol=1e-5,
    )


def test_build_sonic_actions_does_not_double_align_prealigned_vla_action(tmp_path: Path) -> None:
    num_frames = 1
    aligned_action = np.zeros((num_frames, VLA_ACTION_DIM), dtype=np.float32)
    aligned_action[:, :2] = np.array([[0.0, 1.0]], dtype=np.float32)
    aligned_action[:, 2:3] = np.array([[0.82]], dtype=np.float32)
    aligned_action[:, 3:9] = np.array([[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    heading_align = np.array([[0.70710677, 0.0, 0.0, 0.70710677]], dtype=np.float32)

    with _open_npz(
        tmp_path,
        vla_action=aligned_action,
        vla_action_heading_aligned=np.array(True, dtype=np.bool_),
        anchor_heading_align_quat_wxyz=heading_align,
        anchor_use_heading_align=np.array([[True]], dtype=np.bool_),
    ) as data:
        action = build_sonic_actions_from_recording(
            data,
            num_frames=num_frames,
            align_heading_targets=True,
        )

    np.testing.assert_allclose(action, aligned_action, atol=1e-5)
