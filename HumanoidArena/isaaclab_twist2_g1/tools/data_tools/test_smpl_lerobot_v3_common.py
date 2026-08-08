from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (_THIS_DIR, _PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from action_provider.vla_smpl_runtime import (
    quat_angle_between_wxyz,
    quat_from_roll_pitch_yaw_wxyz,
    quat_to_rot6d_wxyz,
    reorder_twist2_to_canonical_29,
)
from action_provider.vla_robot_current_local_runtime_v3 import (
    UnifiedRobotCurrentLocalActionRuntimeV3,
    build_twist2_mimic_obs_v3,
)
from smpl_lerobot_v3_common import (
    build_features,
    build_sonic_rotlocal_v3_actions,
    build_twist2_rotlocal_v3_actions,
    extract_canonical_state_v3,
)


def test_sonic_v3_action_is_reference_pose_not_robot_current_residual(tmp_path: Path) -> None:
    path = tmp_path / "sonic.npz"
    robot_quat_a = np.stack(
        [
            quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0),
            quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0),
        ],
        axis=0,
    )
    robot_quat_b = np.stack(
        [
            quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=-np.pi / 2.0),
            quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=-np.pi / 2.0),
        ],
        axis=0,
    )
    source_quat = np.stack(
        [
            quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 4.0),
            quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0),
        ],
        axis=0,
    )
    hand = np.array([[0.25, 0.75], [0.5, 1.0]], dtype=np.float32)
    np.savez(
        path,
        human_body_pos=np.array([[0.0, 0.0, 0.8], [1.0, 0.0, 0.82]], dtype=np.float32),
        human_body_quat_w=source_quat.astype(np.float32),
        human_joint_pos=np.zeros((2, 29), dtype=np.float32),
        vla_action_hand_binary=hand,
    )

    with np.load(path, allow_pickle=True) as data:
        actions_a = build_sonic_rotlocal_v3_actions(
            data,
            num_frames=2,
            robot_root_orientation=robot_quat_a,
        )
        actions_b = build_sonic_rotlocal_v3_actions(
            data,
            num_frames=2,
            robot_root_orientation=robot_quat_b,
        )

    identity_rot6d = quat_to_rot6d_wxyz(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)).reshape(6)
    ref_step_rot6d = quat_to_rot6d_wxyz(quat_from_roll_pitch_yaw_wxyz(0.0, 0.0, np.pi / 4.0)).reshape(6)
    np.testing.assert_allclose(actions_a, actions_b, atol=1e-6)
    np.testing.assert_allclose(actions_a[0, 3:9], identity_rot6d, atol=1e-6)
    np.testing.assert_allclose(actions_a[1, 3:9], ref_step_rot6d, atol=1e-6)
    np.testing.assert_allclose(actions_a[:, 38:40], hand)


def test_twist2_v3_action_integrates_reference_yaw_from_zero(tmp_path: Path) -> None:
    path = tmp_path / "twist2.npz"
    robot_quat = np.stack(
        [
            quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0),
            quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0),
        ],
        axis=0,
    )
    mimic = np.zeros((2, 35), dtype=np.float32)
    mimic[:, 0:2] = np.array([[0.5, -0.25], [1.0, 0.75]], dtype=np.float32)
    mimic[:, 2] = 0.8
    mimic[1, 5] = 2.0
    mimic[:, 6:35] = np.arange(58, dtype=np.float32).reshape(2, 29) * 0.01
    hand = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    np.savez(path, robot_action_mimic=mimic, vla_action_hand_binary=hand)

    with np.load(path, allow_pickle=True) as data:
        actions = build_twist2_rotlocal_v3_actions(
            data,
            num_frames=2,
            control_dt=0.02,
            robot_root_orientation=robot_quat,
        )

    identity_rot6d = quat_to_rot6d_wxyz(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)).reshape(6)
    yaw_step_rot6d = quat_to_rot6d_wxyz(quat_from_roll_pitch_yaw_wxyz(0.0, 0.0, 0.04)).reshape(6)
    np.testing.assert_allclose(actions[0, 0:2], np.zeros(2, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(actions[1, 0:2], mimic[1, 0:2] * 0.02, atol=1e-6)
    np.testing.assert_allclose(actions[0, 3:9], identity_rot6d, atol=1e-6)
    np.testing.assert_allclose(actions[1, 3:9], yaw_step_rot6d, atol=1e-6)
    np.testing.assert_allclose(actions[:, 9:38], reorder_twist2_to_canonical_29(mimic[:, 6:35]))
    np.testing.assert_allclose(actions[:, 38:40], hand)


def test_twist2_v3_keeps_base_local_xy_for_command_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "twist2_pitch.npz"
    robot_quat = np.stack(
        [
            quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=np.pi / 6.0, yaw=0.0),
            quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=np.pi / 6.0, yaw=0.0),
        ],
        axis=0,
    )
    mimic = np.zeros((2, 35), dtype=np.float32)
    mimic[:, 0] = 1.0
    mimic[:, 2] = 0.8
    mimic[:, 4] = np.pi / 6.0
    np.savez(path, robot_action_mimic=mimic)

    with np.load(path, allow_pickle=True) as data:
        actions = build_twist2_rotlocal_v3_actions(
            data,
            num_frames=2,
            control_dt=0.02,
            robot_root_orientation=robot_quat,
        )

    np.testing.assert_allclose(actions[0, 0:2], np.zeros(2, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(actions[1, 0:2], np.array([0.02, 0.0], dtype=np.float32), atol=1e-6)


def test_twist2_v3_action_roundtrips_to_native_mimic_after_first_frame(tmp_path: Path) -> None:
    path = tmp_path / "twist2_roundtrip.npz"
    robot_quat = np.tile(
        quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0),
        (3, 1),
    )
    mimic = np.zeros((3, 35), dtype=np.float32)
    mimic[:, 0:2] = np.array([[0.0, 0.0], [0.4, -0.2], [0.6, 0.3]], dtype=np.float32)
    mimic[:, 2] = 0.8
    mimic[1:, 5] = np.array([1.0, -0.5], dtype=np.float32)
    mimic[:, 6:35] = np.arange(87, dtype=np.float32).reshape(3, 29) * 0.01
    np.savez(path, robot_action_mimic=mimic)

    with np.load(path, allow_pickle=True) as data:
        actions = build_twist2_rotlocal_v3_actions(
            data,
            num_frames=3,
            control_dt=0.02,
            robot_root_orientation=robot_quat,
        )

    runtime = UnifiedRobotCurrentLocalActionRuntimeV3()
    reconstructed = []
    for action in actions:
        frame = runtime.step(
            action,
            current_robot_quat_wxyz=robot_quat[0],
            current_robot_xy_world=np.zeros(2, dtype=np.float32),
        )
        reconstructed.append(build_twist2_mimic_obs_v3(runtime_frame=frame, control_dt=0.02))
    reconstructed = np.stack(reconstructed, axis=0)

    np.testing.assert_allclose(reconstructed[1:, 0:6], mimic[1:, 0:6], atol=1e-5)
    np.testing.assert_allclose(reconstructed[1:, 6:35], mimic[1:, 6:35], atol=1e-6)


def test_v3_state_is_heading_canonical() -> None:
    initial = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0)
    current = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0)
    state = extract_canonical_state_v3(
        root_orientation=np.stack([initial, current], axis=0),
        joint_pos=np.zeros((2, 29), dtype=np.float32),
        joint_vel=np.zeros((2, 29), dtype=np.float32),
    )
    identity_rot6d = quat_to_rot6d_wxyz(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)).reshape(6)

    np.testing.assert_allclose(state[:, :6], np.tile(identity_rot6d, (2, 1)), atol=1e-6)
    assert state.shape == (2, 64)


def test_v3_feature_names_are_rotlocal() -> None:
    features = build_features((64, 64, 3), use_videos=False)

    assert features["observation.state"]["names"][0] == "state.root_heading_canonical_rot6d.0"
    assert features["action"]["names"][0] == "action.root_ref_base_local_xy_delta.x"
    assert features["action"]["names"][3] == "action.root_ref_rot6d.0"
    assert features["action"]["shape"] == (40,)


def test_v3_common_imports_do_not_pull_v2_runtime() -> None:
    import smpl_lerobot_v3_common

    assert "vla_local_delta_runtime_v2" not in Path(smpl_lerobot_v3_common.__file__).read_text()
