import tempfile
import unittest
import random
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from data.multisource_dataset import (
    BaseSourceAdapter,
    EpisodeRecord,
    HIW_G1_JOINT_FEATURE_NAMES_29,
    HIW_WBC_FEATURE_NAMES_23,
    HumanoidArenaAdapter,
    MultiSourceG1Dataset,
    ParquetEpisodeReader,
    SOURCE_HIW500,
    SOURCE_HUMANOID_ARENA,
    SOURCE_HUMANOID_EVERYDAY,
    SOURCE_UNIFOLM,
    VideoFrameDecodeError,
    _canonicalize_kimodo_window_translation,
    _episode_local_root_from_odometry,
    _hiw_hand_from_events,
    _hiw_joint_discontinuity,
    _hiw_root_from_wbc,
    _humanoid_everyday_instruction,
    _stereo_crop,
    _unifolm_motion_discontinuity,
    _unifolm_root_discontinuity,
    _validate_named_feature,
)
from motion.g1_reference import (
    CANONICAL_G1_JOINT_NAMES_29,
    HumanoidArenaActionDecoder,
    UNITREE_G1_JOINT_NAMES_29,
    resample_motion,
)
from skeleton.definitions import G1Skeleton34


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = PROJECT_ROOT / "skeleton/assets/g1skel34/xml/g1.xml"


def _episode(
    *,
    source=SOURCE_HUMANOID_ARENA,
    task_id="task-a",
    episode_id="0",
    data_path=Path("episode.parquet"),
    source_length=8,
    source_fps=30.0,
    metadata=None,
):
    return EpisodeRecord(
        source=source,
        task_id=task_id,
        task_name=task_id,
        instruction=task_id,
        episode_id=episode_id,
        data_path=data_path,
        source_length=source_length,
        source_fps=source_fps,
        target_fps=30.0,
        video_path=Path("episode.mp4"),
        video_from_timestamp=0.0,
        first_cut=0,
        sample_count=1,
        metadata=dict(metadata or {"row_start": 0, "row_end": source_length}),
    )


class MultiSourceDatasetTest(unittest.TestCase):
    def test_arena_uses_action_hand_as_observed_history_state(self):
        adapter = HumanoidArenaAdapter.__new__(HumanoidArenaAdapter)
        state = np.zeros((4, 64), dtype=np.float32)
        state[:, :6] = np.asarray(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32
        )
        actions = np.zeros((4, 40), dtype=np.float32)
        actions[:, 38:40] = np.asarray(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            dtype=np.float32,
        )
        adapter.reader = MagicMock()
        adapter.reader.read.return_value = {
            "observation.state": state,
            "action": actions,
        }
        decoder = MagicMock()
        decoded = {
            "local_rot_mats": torch.eye(3).reshape(1, 1, 3, 3).repeat(4, 1, 1, 1),
            "root_positions": torch.zeros(4, 3),
        }
        decoder.decode_joint_configuration_pose.return_value = decoded
        decoder.decode_action_pose.return_value = decoded
        adapter._decoder = MagicMock(return_value=decoder)
        adapter._motion_feature_mask = MagicMock(
            return_value=torch.ones(417, dtype=torch.bool)
        )
        adapter._finalize_motion = MagicMock(return_value={"loaded": True})

        result = adapter.load_episode(_episode(source_length=4))

        self.assertEqual(result, {"loaded": True})
        finalize_args = adapter._finalize_motion.call_args.args
        np.testing.assert_array_equal(finalize_args[5], actions[:, 38:40])
        np.testing.assert_array_equal(finalize_args[6], actions[:, 38:40])
        np.testing.assert_array_equal(
            finalize_args[7], np.ones((4, 2), dtype=bool)
        )
        np.testing.assert_array_equal(
            finalize_args[8], np.ones((4, 2), dtype=bool)
        )

    def test_hiw_root_integrates_base_twist_not_torso_rpy(self):
        wbc = np.zeros((4, 23), dtype=np.float32)
        wbc[:, 6] = 0.74
        wbc[:, 3:6] = np.asarray([1.2, 0.8, -0.4], dtype=np.float32)
        wbc[0, 0] = 1.0
        wbc[1, 2] = np.pi / 2
        wbc[2, 0] = 1.0

        positions, rotations = _hiw_root_from_wbc(wbc, fps=1.0)

        np.testing.assert_allclose(
            positions,
            np.asarray(
                [
                    [0.0, 0.0, 0.74],
                    [1.0, 0.0, 0.74],
                    [1.0, 0.0, 0.74],
                    [1.0, 1.0, 0.74],
                ],
                dtype=np.float32,
            ),
            atol=1e-6,
        )
        torch.testing.assert_close(rotations[0], torch.eye(3))
        torch.testing.assert_close(
            rotations[-1],
            torch.tensor(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_hiw_root_uses_exact_constant_twist_integration(self):
        wbc = np.zeros((2, 23), dtype=np.float32)
        wbc[:, 6] = 0.7
        wbc[0, 0] = 1.0
        wbc[0, 2] = np.pi / 2

        positions, _ = _hiw_root_from_wbc(wbc, fps=1.0)

        expected = 2.0 / np.pi
        np.testing.assert_allclose(
            positions[1], np.asarray([expected, expected, 0.7]), atol=1e-6
        )

    def test_hiw_hand_events_are_latched_closed_states(self):
        wbc = np.zeros((7, 23), dtype=np.float32)
        wbc[:, [19, 21]] = 10.0
        wbc[2, 19] = 0.0
        wbc[3, 19] = 10.0
        wbc[4, 20] = 1.0
        wbc[5, 20] = 0.0
        wbc[1, 21] = 0.0
        wbc[2, 21] = 10.0
        wbc[5, 22] = 1.0

        hand = _hiw_hand_from_events(wbc)

        np.testing.assert_array_equal(
            hand[:, 0], np.asarray([0, 0, 1, 1, 0, 0, 0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            hand[:, 1], np.asarray([0, 1, 1, 1, 1, 0, 0], dtype=np.float32)
        )

    def test_hiw_initial_joint_reset_is_logically_trimmed(self):
        joint_q = np.zeros((6, 29), dtype=np.float32)
        joint_q[0, 4] = 0.8

        frame_offset, issue = _hiw_joint_discontinuity(joint_q)

        self.assertEqual(frame_offset, 1)
        self.assertIsNone(issue)

    def test_hiw_internal_joint_jump_is_rejected(self):
        joint_q = np.zeros((6, 29), dtype=np.float32)
        joint_q[3:, 4] = 0.8

        frame_offset, issue = _hiw_joint_discontinuity(joint_q)

        self.assertEqual(frame_offset, 0)
        self.assertIsNotNone(issue)
        self.assertEqual(issue["transition_frame"], 2)
        self.assertAlmostEqual(issue["joint_jump"], 0.8)

    def test_hiw_named_schema_accepts_released_joint_name_typo(self):
        names = list(HIW_G1_JOINT_FEATURE_NAMES_29)
        names[21] = "kLeftWristyaw.q"
        features = {
            "observation.state": {
                "shape": [29],
                "names": names,
            }
        }

        _validate_named_feature(
            features,
            "observation.state",
            HIW_G1_JOINT_FEATURE_NAMES_29,
        )

    def test_hiw_named_schema_rejects_reordered_wbc_fields(self):
        names = list(HIW_WBC_FEATURE_NAMES_23)
        names[0], names[1] = names[1], names[0]
        features = {
            "observation.state.wbc": {
                "shape": [23],
                "names": names,
            }
        }

        with self.assertRaisesRegex(ValueError, "field at index 0"):
            _validate_named_feature(
                features,
                "observation.state.wbc",
                HIW_WBC_FEATURE_NAMES_23,
            )

    def test_unifolm_initial_root_reset_is_logically_trimmed(self):
        current = torch.zeros(6, 36).numpy()
        desired = torch.zeros(6, 36).numpy()
        current[0, 0] = 1.2

        frame_offset, internal_max, internal_frame = _unifolm_root_discontinuity(
            current, desired
        )

        self.assertEqual(frame_offset, 1)
        self.assertEqual(internal_max, 0.0)
        self.assertIsNone(internal_frame)

    def test_unifolm_internal_root_reset_is_rejected(self):
        current = torch.zeros(6, 36).numpy()
        desired = torch.zeros(6, 36).numpy()
        current[3:, 0] = 1.0

        frame_offset, internal_max, internal_frame = _unifolm_root_discontinuity(
            current, desired
        )

        self.assertEqual(frame_offset, 0)
        self.assertAlmostEqual(internal_max, 1.0)
        self.assertEqual(internal_frame, 2)

    def test_unifolm_initial_joint_reset_is_logically_trimmed(self):
        current = np.zeros((6, 36), dtype=np.float32)
        desired = np.zeros((6, 36), dtype=np.float32)
        current[:, 3] = 1.0
        desired[:, 3] = 1.0
        current[0, 7] = 1.2

        frame_offset, issue = _unifolm_motion_discontinuity(current, desired)

        self.assertEqual(frame_offset, 1)
        self.assertIsNone(issue)

    def test_unifolm_initial_rotation_reset_is_logically_trimmed(self):
        current = np.zeros((6, 36), dtype=np.float32)
        desired = np.zeros((6, 36), dtype=np.float32)
        current[:, 3] = 1.0
        desired[:, 3] = 1.0
        current[0, 3:7] = np.asarray(
            [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32
        )

        frame_offset, issue = _unifolm_motion_discontinuity(current, desired)

        self.assertEqual(frame_offset, 1)
        self.assertIsNone(issue)

    def test_unifolm_internal_joint_jump_is_rejected(self):
        current = np.zeros((6, 36), dtype=np.float32)
        desired = np.zeros((6, 36), dtype=np.float32)
        current[:, 3] = 1.0
        desired[:, 3] = 1.0
        desired[3:, 8] = 0.8

        frame_offset, issue = _unifolm_motion_discontinuity(current, desired)

        self.assertEqual(frame_offset, 0)
        self.assertIsNotNone(issue)
        self.assertEqual(issue["transition_frame"], 2)
        self.assertAlmostEqual(issue["joint_jump"], 0.8)

    def test_arena_selection_has_exactly_one_task_or_merged_dataset(self):
        self.assertEqual(
            HumanoidArenaAdapter._parse_selection(
                {"task": "HOI_football", "backend": "sonic"}
            ),
            {"mode": "task", "task": "HOI_football", "backend": "sonic"},
        )
        self.assertEqual(
            HumanoidArenaAdapter._parse_selection(
                {"merged": "all_16_refpose_v3_1"}
            ),
            {"mode": "merged", "merged": "all_16_refpose_v3_1"},
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            HumanoidArenaAdapter._parse_selection({})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            HumanoidArenaAdapter._parse_selection(
                {"HOI_football": "sonic", "HSI_sit_sofa": "sonic"}
            )
        with self.assertRaisesRegex(ValueError, "Unsupported HumanoidArena merged"):
            HumanoidArenaAdapter._parse_selection({"merged": "unknown"})

    def test_arena_merged_manifest_preserves_per_source_backend(self):
        episode_sources = HumanoidArenaAdapter._merged_episode_sources(
            {
                "total_episodes": 3,
                "sources": [
                    {
                        "task_name": "HOI_football",
                        "dataset_name": "sonic_refpose_v3_1",
                        "total_episodes": 2,
                    },
                    {
                        "task_name": "HOI_football",
                        "dataset_name": "twist2_refpose_v3_1",
                        "total_episodes": 1,
                    },
                ],
            }
        )

        self.assertEqual(
            episode_sources,
            [
                ("HOI_football", "sonic"),
                ("HOI_football", "sonic"),
                ("HOI_football", "twist2"),
            ],
        )

    def test_arena_merged_training_excludes_grap_cup_only(self):
        self.assertTrue(
            HumanoidArenaAdapter._skip_merged_training_task("HOI_grap_cup")
        )
        self.assertFalse(
            HumanoidArenaAdapter._skip_merged_training_task("HOI_football")
        )
        self.assertEqual(
            HumanoidArenaAdapter._parse_selection(
                {"task": "HOI_grap_cup", "backend": "sonic"}
            ),
            {"mode": "task", "task": "HOI_grap_cup", "backend": "sonic"},
        )

    def test_everyday_task_catalog_description_overrides_stale_episode_text(self):
        instruction = _humanoid_everyday_instruction(
            {
                "task": "Articulated/adjust_the_angle_of_a_phone_stand",
                "description": "Tilt the phone stand upward.",
            },
            {"instruction": "Wipe the desk."},
        )

        self.assertEqual(instruction, "Tilt the phone stand upward.")

    def test_everyday_odometry_is_rebased_to_initial_heading(self):
        half_angle = np.pi / 4.0
        root_quaternion = np.asarray(
            [
                [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        root_position = np.asarray(
            [[10.0, 20.0, 0.75], [10.0, 21.0, 0.80]], dtype=np.float32
        )

        local_position, local_rotation = _episode_local_root_from_odometry(
            root_position, root_quaternion
        )

        np.testing.assert_allclose(
            local_position,
            np.asarray([[0.0, 0.0, 0.75], [1.0, 0.0, 0.80]], dtype=np.float32),
            atol=1e-6,
        )
        torch.testing.assert_close(
            local_rotation[0], torch.eye(3), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            local_rotation[1],
            torch.tensor(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_horizontally_packed_stereo_video_is_cropped_to_one_eye(self):
        feature = {"shape": [480, 1280, 3]}

        self.assertEqual(_stereo_crop(feature, {}), (0, 0, 640, 480))
        self.assertEqual(
            _stereo_crop(feature, {"stereo_view": "right"}),
            (640, 0, 1280, 480),
        )
        self.assertIsNone(_stereo_crop({"shape": [480, 640, 3]}, {}))

    def test_unitree_joint_order_is_the_public_g1_order(self):
        expected = (
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
            "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
            "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        )
        self.assertEqual(UNITREE_G1_JOINT_NAMES_29, expected)
        self.assertEqual(set(expected), set(CANONICAL_G1_JOINT_NAMES_29))

    def test_generic_configuration_decoder_preserves_joint_values(self):
        decoder = HumanoidArenaActionDecoder(G1Skeleton34(), XML_PATH, fps=30.0)
        unitree_q = torch.linspace(-0.4, 0.4, 29).repeat(30, 1)
        root = torch.tensor([[1.0, 2.0, 0.8]]).repeat(30, 1)
        quaternion_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(30, 1)

        decoded = decoder.decode_joint_configuration(
            unitree_q,
            root,
            root_quaternions=quaternion_wxyz,
            joint_names=UNITREE_G1_JOINT_NAMES_29,
        )
        encoded = decoder.encode(
            decoded["local_rot_mats"], decoded["root_positions"]
        )
        expected_canonical = unitree_q[:, [
            UNITREE_G1_JOINT_NAMES_29.index(name)
            for name in CANONICAL_G1_JOINT_NAMES_29
        ]]
        torch.testing.assert_close(
            encoded[:, 9:38], expected_canonical, rtol=1e-5, atol=1e-5
        )

    def test_pose_only_decoder_preserves_final_training_motion_exactly(self):
        decoder = HumanoidArenaActionDecoder(
            G1Skeleton34(), XML_PATH, fps=50.0
        )
        generator = torch.Generator().manual_seed(9012)
        frame_count = 24
        joint_positions = (
            torch.randn(frame_count, 29, generator=generator) * 0.1
        )
        root_positions = (
            torch.randn(frame_count, 3, generator=generator) * 0.02
        )
        root_positions[:, 2] += 0.8
        root_quaternions = torch.zeros(frame_count, 4)
        root_quaternions[:, 0] = 1.0
        observed_pose_only = decoder.decode_joint_configuration_pose(
            joint_positions,
            root_positions,
            root_quaternions=root_quaternions,
        )
        observed_full = decoder.decode_joint_configuration(
            joint_positions,
            root_positions,
            root_quaternions=root_quaternions,
        )
        actions = torch.zeros(frame_count, 40)
        actions[:, :2] = torch.randn(
            frame_count, 2, generator=generator
        ) * 0.005
        actions[:, 2] = 0.8
        yaw = torch.randn(frame_count, generator=generator) * 0.1
        actions[:, 3:9] = torch.stack(
            (
                torch.cos(yaw),
                -torch.sin(yaw),
                torch.sin(yaw),
                torch.cos(yaw),
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
            ),
            dim=-1,
        )
        actions[:, 9:38] = (
            torch.randn(frame_count, 29, generator=generator) * 0.1
        )
        target_pose_only = decoder.decode_action_pose(actions)
        target_full = decoder.decode(actions)
        episode = _episode(source_length=frame_count, source_fps=50.0)
        hands = torch.zeros(frame_count, 2)
        valid = torch.ones(frame_count, 2, dtype=torch.bool)

        adapter = BaseSourceAdapter.__new__(BaseSourceAdapter)
        adapter.target_fps = 30.0
        adapter._representation = None
        old_path = adapter._finalize_motion(
            episode,
            observed_full["local_rot_mats"],
            observed_full["root_positions"],
            target_full["local_rot_mats"],
            target_full["root_positions"],
            hands,
            hands,
            valid,
            valid,
        )
        new_path = adapter._finalize_motion(
            episode,
            observed_pose_only["local_rot_mats"],
            observed_pose_only["root_positions"],
            target_pose_only["local_rot_mats"],
            target_pose_only["root_positions"],
            hands,
            hands,
            valid,
            valid,
        )

        self.assertEqual(set(old_path), set(new_path))
        for key in old_path:
            if torch.is_tensor(old_path[key]):
                self.assertTrue(
                    torch.equal(old_path[key], new_path[key]),
                    msg=f"Training field changed: {key}",
                )
            else:
                self.assertEqual(old_path[key], new_path[key])

    def test_fifty_hz_motion_resamples_to_thirty_hz_with_shared_endpoints(self):
        source_frames = 51
        rotations = torch.eye(3).reshape(1, 1, 3, 3).repeat(
            source_frames, 2, 1, 1
        )
        root = torch.zeros(source_frames, 3)
        root[:, 0] = torch.linspace(0.0, 1.0, source_frames)

        resampled_rotations, resampled_root = resample_motion(
            rotations, root, source_fps=50.0, target_fps=30.0
        )

        self.assertEqual(resampled_rotations.shape[0], 31)
        torch.testing.assert_close(resampled_root[0], root[0])
        torch.testing.assert_close(resampled_root[-1], root[-1])
        torch.testing.assert_close(
            resampled_root[:, 0], torch.linspace(0.0, 1.0, 31),
            rtol=1e-6, atol=1e-6,
        )

    def test_window_translation_uses_first_valid_target_and_preserves_features(self):
        gt_motion = torch.arange(5 * 417, dtype=torch.float32).reshape(5, 417)
        gt_motion[:, 0] = torch.tensor([99.0, 10.0, 11.5, 14.0, 18.0])
        gt_motion[:, 1] = torch.tensor([7.0, 0.82, 0.83, 0.84, 0.85])
        gt_motion[:, 2] = torch.tensor([88.0, -3.0, -2.0, 1.0, 5.0])
        condition_motion = gt_motion.clone()
        condition_motion[:, 0] = torch.tensor([77.0, 9.5, 10.5, 0.0, 0.0])
        condition_motion[:, 2] = torch.tensor([66.0, -3.5, -2.5, 0.0, 0.0])
        gt_mask = torch.tensor([False, True, True, True, True])
        condition_mask = torch.zeros(5, 417, dtype=torch.bool)
        condition_mask[1:3] = True
        before_gt = gt_motion.clone()
        before_condition = condition_motion.clone()
        before_condition_mask = condition_mask.clone()
        before_delta_xz = torch.diff(gt_motion[gt_mask][:, [0, 2]], dim=0)

        origin = _canonicalize_kimodo_window_translation(
            gt_motion, condition_motion, condition_mask, gt_mask
        )

        torch.testing.assert_close(origin, torch.tensor([10.0, -3.0]))
        torch.testing.assert_close(gt_motion[1, [0, 2]], torch.zeros(2))
        torch.testing.assert_close(
            gt_motion[:, 0], torch.tensor([99.0, 0.0, 1.5, 4.0, 8.0])
        )
        torch.testing.assert_close(
            gt_motion[:, 2], torch.tensor([88.0, 0.0, 1.0, 4.0, 8.0])
        )
        torch.testing.assert_close(
            condition_motion[:, 0], torch.tensor([77.0, -0.5, 0.5, 0.0, 0.0])
        )
        torch.testing.assert_close(
            condition_motion[:, 2], torch.tensor([66.0, -0.5, 0.5, 0.0, 0.0])
        )
        unchanged_features = [1, *range(3, 417)]
        self.assertTrue(
            torch.equal(
                gt_motion[:, unchanged_features],
                before_gt[:, unchanged_features],
            )
        )
        self.assertTrue(
            torch.equal(
                condition_motion[:, unchanged_features],
                before_condition[:, unchanged_features],
            )
        )
        self.assertTrue(torch.equal(condition_mask, before_condition_mask))
        torch.testing.assert_close(
            torch.diff(gt_motion[gt_mask][:, [0, 2]], dim=0), before_delta_xz
        )

    def test_window_translation_future_only_starts_at_planar_origin(self):
        gt_motion = torch.zeros(4, 417)
        gt_motion[:, 0] = torch.tensor([2.0, 3.0, 5.0, 8.0])
        gt_motion[:, 1] = 0.9
        gt_motion[:, 2] = torch.tensor([-4.0, -2.0, 1.0, 5.0])
        condition_motion = torch.zeros_like(gt_motion)
        condition_mask = torch.zeros_like(gt_motion, dtype=torch.bool)
        gt_mask = torch.ones(4, dtype=torch.bool)

        _canonicalize_kimodo_window_translation(
            gt_motion, condition_motion, condition_mask, gt_mask
        )

        torch.testing.assert_close(gt_motion[0, [0, 2]], torch.zeros(2))
        torch.testing.assert_close(gt_motion[:, 1], torch.full((4,), 0.9))
        self.assertFalse(condition_mask.any())
        self.assertFalse(condition_motion.any())

    def test_window_translation_keeps_arena_missing_root_zero_and_masked(self):
        gt_motion = torch.zeros(3, 417)
        gt_motion[:, 0] = torch.tensor([4.0, 5.0, 7.0])
        gt_motion[:, 2] = torch.tensor([6.0, 8.0, 11.0])
        condition_motion = torch.zeros_like(gt_motion)
        condition_motion[:, 3:9] = 0.25
        condition_mask = torch.zeros_like(gt_motion, dtype=torch.bool)
        condition_mask[:2, 3:9] = True
        gt_mask = torch.ones(3, dtype=torch.bool)

        _canonicalize_kimodo_window_translation(
            gt_motion, condition_motion, condition_mask, gt_mask
        )

        torch.testing.assert_close(gt_motion[0, [0, 2]], torch.zeros(2))
        self.assertFalse(condition_motion[:, [0, 2]].any())
        self.assertFalse(condition_mask[:, [0, 2]].any())
        torch.testing.assert_close(
            condition_motion[:2, 3:9], torch.full((2, 6), 0.25)
        )

    def test_parquet_reader_returns_only_the_episode_row_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.parquet"
            table = pa.table(
                {
                    "index": list(range(10, 20)),
                    "value": list(range(10)),
                }
            )
            pq.write_table(table, path, row_group_size=3)
            episode = _episode(
                data_path=path,
                source_length=4,
                metadata={"dataset_from_index": 13, "dataset_to_index": 17},
            )

            reader = ParquetEpisodeReader()
            with patch(
                "data.multisource_dataset.pq.ParquetFile",
                wraps=pq.ParquetFile,
            ) as open_parquet:
                result = reader.read(episode, ["value"])
                repeated = reader.read(episode, ["value"])

            self.assertEqual(result["value"], [3, 4, 5, 6])
            self.assertEqual(repeated, result)
            self.assertEqual(open_parquet.call_count, 1)
            reader.close()

    def test_history_conditions_on_state_while_clean_motion_uses_action(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.action_history = 2
        dataset.action_chunk = 2
        episode = _episode(source_length=6)
        motion = {
            "observed_motion": torch.arange(6).reshape(6, 1).repeat(1, 417).float(),
            "observed_motion_valid": torch.ones(6, 417, dtype=torch.bool),
            "target_motion": (100 + torch.arange(6)).reshape(6, 1).repeat(1, 417).float(),
            "observed_hand": torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 1]]).float(),
            "target_hand": torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0], [1, 0], [0, 1]]).float(),
            "observed_hand_valid": torch.tensor([[1, 0], [1, 1], [0, 1], [1, 1], [1, 1], [1, 1]]).bool(),
            "target_hand_valid": torch.tensor([[1, 1], [1, 1], [1, 0], [0, 1], [1, 1], [1, 1]]).bool(),
        }
        dataset._sample_record = MagicMock(return_value=(MagicMock(), episode, 3))
        dataset._episode_motion = MagicMock(return_value=motion)
        dataset._read_video_frame = MagicMock(return_value=torch.zeros(3, 4, 5, dtype=torch.uint8))
        dataset._text_embeddings = {}

        sample = dataset[0]

        torch.testing.assert_close(
            sample["gt_motion"][:, 0],
            torch.tensor([0.0, 1.0, 2.0, 3.0]),
        )
        torch.testing.assert_close(
            sample["condition_motion"][:, 0],
            torch.tensor([-100.0, -99.0, 0.0, 0.0]),
        )
        self.assertTrue(sample["condition_motion_mask"][:2].all())
        self.assertFalse(sample["condition_motion_mask"][2:].any())
        torch.testing.assert_close(sample["gt_hand"][:2], motion["observed_hand"][1:3])
        torch.testing.assert_close(sample["gt_hand"][2:], motion["target_hand"][3:5])
        torch.testing.assert_close(sample["gt_hand_mask"][:2], motion["observed_hand_valid"][1:3])
        torch.testing.assert_close(sample["gt_hand_mask"][2:], motion["target_hand_valid"][3:5])
        self.assertTrue(sample["gt_mask"].all())

    def test_cut_zero_keeps_left_padding_zero_and_anchors_first_future_frame(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.action_history = 2
        dataset.action_chunk = 2
        episode = _episode(source_length=4)
        target_motion = torch.zeros(4, 417)
        target_motion[:, 0] = torch.tensor([6.0, 7.0, 9.0, 12.0])
        target_motion[:, 1] = 0.8
        target_motion[:, 2] = torch.tensor([-2.0, 0.0, 3.0, 7.0])
        motion = {
            "observed_motion": torch.full((4, 417), 3.0),
            "observed_motion_valid": torch.ones(4, 417, dtype=torch.bool),
            "target_motion": target_motion,
            "observed_hand": torch.zeros(4, 2),
            "target_hand": torch.ones(4, 2),
            "observed_hand_valid": torch.ones(4, 2, dtype=torch.bool),
            "target_hand_valid": torch.ones(4, 2, dtype=torch.bool),
        }
        dataset._sample_record = MagicMock(return_value=(MagicMock(), episode, 0))
        dataset._episode_motion = MagicMock(return_value=motion)
        dataset._read_video_frame = MagicMock(
            return_value=torch.zeros(3, 4, 5, dtype=torch.uint8)
        )
        dataset._text_embeddings = {}

        sample = dataset[0]

        self.assertFalse(sample["gt_mask"][:2].any())
        self.assertFalse(sample["gt_motion"][:2].any())
        self.assertFalse(sample["condition_motion"][:2].any())
        self.assertFalse(sample["condition_motion_mask"].any())
        torch.testing.assert_close(
            sample["gt_motion"][2:, [0, 2]],
            torch.tensor([[0.0, 0.0], [1.0, 2.0]]),
        )
        torch.testing.assert_close(
            sample["gt_motion"][2:, 1], torch.full((2,), 0.8)
        )

    def test_logical_frame_offset_keeps_motion_and_video_time_aligned(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.action_history = 2
        dataset.action_chunk = 2
        episode = _episode(source=SOURCE_UNIFOLM, source_length=6)
        motion = {
            "observed_motion": torch.arange(5).reshape(5, 1).repeat(1, 417).float(),
            "observed_motion_valid": torch.ones(5, 417, dtype=torch.bool),
            "target_motion": (100 + torch.arange(5)).reshape(5, 1).repeat(1, 417).float(),
            "observed_hand": torch.zeros(5, 2),
            "target_hand": torch.ones(5, 2),
            "observed_hand_valid": torch.ones(5, 2, dtype=torch.bool),
            "target_hand_valid": torch.ones(5, 2, dtype=torch.bool),
            "frame_offset": 1,
            "source_frame_offset": 1,
        }
        dataset._sample_record = MagicMock(return_value=(MagicMock(), episode, 3))
        dataset._episode_motion = MagicMock(return_value=motion)
        dataset._read_video_frame = MagicMock(
            return_value=torch.zeros(3, 4, 5, dtype=torch.uint8)
        )
        dataset._text_embeddings = {}

        sample = dataset[0]

        torch.testing.assert_close(
            sample["gt_motion"][:, 0], torch.tensor([0.0, 1.0, 2.0, 3.0])
        )
        torch.testing.assert_close(
            sample["condition_motion"][:, 0], torch.tensor([-100.0, -99.0, 0.0, 0.0])
        )
        dataset._read_video_frame.assert_called_once_with(
            episode.video_path,
            episode.video_from_timestamp + 3 / episode.target_fps,
            crop=None,
        )
        self.assertEqual(sample["cut_index"], 3)
        self.assertEqual(sample["motion_cut_index"], 2)
        self.assertEqual(sample["source_frame_offset"], 1)

    def test_partial_state_feature_mask_only_applies_to_history(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.action_history = 2
        dataset.action_chunk = 2
        episode = _episode(source_length=6)
        feature_mask = torch.zeros(6, 417, dtype=torch.bool)
        feature_mask[:, 3:9] = True
        motion = {
            "observed_motion": torch.ones(6, 417),
            "observed_motion_valid": feature_mask,
            "target_motion": torch.full((6, 417), 2.0),
            "observed_hand": torch.zeros(6, 2),
            "target_hand": torch.ones(6, 2),
            "observed_hand_valid": torch.zeros(6, 2, dtype=torch.bool),
            "target_hand_valid": torch.ones(6, 2, dtype=torch.bool),
        }
        dataset._sample_record = MagicMock(return_value=(MagicMock(), episode, 3))
        dataset._episode_motion = MagicMock(return_value=motion)
        dataset._read_video_frame = MagicMock(
            return_value=torch.zeros(3, 4, 5, dtype=torch.uint8)
        )
        dataset._text_embeddings = {}

        sample = dataset[0]

        self.assertTrue(sample["condition_motion_mask"][:2, 3:9].all())
        self.assertFalse(sample["condition_motion_mask"][:2, :3].any())
        self.assertFalse(sample["condition_motion_mask"][:2, 9:].any())
        self.assertFalse(sample["condition_motion_mask"][2:].any())
        self.assertTrue(sample["gt_mask"].all())

    def test_invalid_internal_reset_episode_is_resampled(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.action_history = 2
        dataset.action_chunk = 2
        dataset._episode_cache = OrderedDict()
        invalid_episode = _episode(
            source=SOURCE_UNIFOLM, episode_id="invalid", source_length=6
        )
        valid_episode = _episode(
            source=SOURCE_UNIFOLM, episode_id="valid", source_length=6
        )
        valid_motion = {
            "observed_motion": torch.zeros(6, 417),
            "target_motion": torch.ones(6, 417),
            "observed_hand": torch.zeros(6, 2),
            "target_hand": torch.ones(6, 2),
            "observed_hand_valid": torch.ones(6, 2, dtype=torch.bool),
            "target_hand_valid": torch.ones(6, 2, dtype=torch.bool),
        }
        dataset._sample_record = MagicMock(
            side_effect=[
                (MagicMock(), invalid_episode, 2),
                (MagicMock(), invalid_episode, 2),
                (MagicMock(), valid_episode, 2),
            ]
        )
        dataset._episode_motion = MagicMock(
            side_effect=[
                {
                    "skip_episode": True,
                    "quality_issue": "internal root jump 1.000 m",
                },
                valid_motion,
            ]
        )
        dataset._read_video_frame = MagicMock(
            return_value=torch.zeros(3, 4, 5, dtype=torch.uint8)
        )
        dataset._text_embeddings = {}

        sample = dataset[0]

        self.assertEqual(sample["episode_id"], "valid")
        self.assertEqual(dataset._sample_record.call_count, 3)
        self.assertEqual(dataset._episode_motion.call_count, 2)
        self.assertIn(
            invalid_episode.cache_key,
            dataset._runtime_invalid_episodes,
        )
        self.assertTrue(sample["gt_mask"].all())

    def test_video_decode_failure_blacklists_episode_and_resamples(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.action_history = 2
        dataset.action_chunk = 2
        dataset._episode_cache = OrderedDict()
        invalid_episode = _episode(episode_id="invalid-video", source_length=6)
        valid_episode = _episode(episode_id="valid-video", source_length=6)
        motion = {
            "observed_motion": torch.zeros(6, 417),
            "target_motion": torch.ones(6, 417),
            "observed_hand": torch.zeros(6, 2),
            "target_hand": torch.ones(6, 2),
            "observed_hand_valid": torch.ones(6, 2, dtype=torch.bool),
            "target_hand_valid": torch.ones(6, 2, dtype=torch.bool),
        }
        dataset._sample_record = MagicMock(
            side_effect=[
                (MagicMock(), invalid_episode, 2),
                (MagicMock(), invalid_episode, 2),
                (MagicMock(), valid_episode, 2),
            ]
        )
        dataset._episode_motion = MagicMock(side_effect=[motion, motion])
        dataset._read_video_frame = MagicMock(
            side_effect=[
                VideoFrameDecodeError("invalid packet"),
                torch.zeros(3, 4, 5, dtype=torch.uint8),
            ]
        )
        dataset._text_embeddings = {}

        sample = dataset[0]

        self.assertEqual(sample["episode_id"], "valid-video")
        self.assertEqual(dataset._sample_record.call_count, 3)
        self.assertEqual(dataset._episode_motion.call_count, 2)
        self.assertEqual(dataset._read_video_frame.call_count, 2)
        self.assertIn(
            invalid_episode.cache_key,
            dataset._runtime_invalid_episodes,
        )

    def test_invalid_video_packet_is_wrapped_with_path_context(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.video_cache_size = 0
        dataset._video_cache = OrderedDict()
        error = av.error.InvalidDataError(
            1094995529, "Invalid data found when processing input"
        )
        with patch(
            "data.multisource_dataset.av.open", side_effect=error
        ), self.assertRaises(VideoFrameDecodeError) as raised:
            dataset._read_video_frame(Path("broken.mp4"), timestamp=1.25)

        self.assertIn("broken.mp4", str(raised.exception))
        self.assertIn("1.250s", str(raised.exception))

    def test_video_container_cache_reuses_open_file_without_changing_pixels(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.video_cache_size = 2
        dataset._video_cache = OrderedDict()
        dataset.adapters = {}

        first_frame = MagicMock()
        first_frame.pts = 30
        first_frame.to_ndarray.return_value = np.full((2, 3, 3), 17, dtype=np.uint8)
        second_frame = MagicMock()
        second_frame.pts = 60
        second_frame.to_ndarray.return_value = np.full((2, 3, 3), 29, dtype=np.uint8)
        stream = MagicMock()
        stream.time_base = Fraction(1, 30)
        stream.codec_context.thread_count = 0
        container = MagicMock()
        container.streams.video = [stream]
        container.decode.side_effect = [[first_frame], [second_frame]]

        with patch("data.multisource_dataset.av.open", return_value=container) as open_video:
            first = dataset._read_video_frame(Path("sample.mp4"), 1.0)
            second = dataset._read_video_frame(Path("sample.mp4"), 2.0)

        self.assertEqual(open_video.call_count, 1)
        self.assertTrue(torch.equal(first, torch.full((3, 2, 3), 17, dtype=torch.uint8)))
        self.assertTrue(torch.equal(second, torch.full((3, 2, 3), 29, dtype=torch.uint8)))
        self.assertEqual(container.seek.call_count, 2)
        container.seek.assert_any_call(
            30, backward=True, any_frame=False, stream=stream
        )
        container.seek.assert_any_call(
            60, backward=True, any_frame=False, stream=stream
        )

        dataset.close()
        container.close.assert_called_once_with()

    def test_cache_key_separates_same_episode_id_across_tasks_and_files(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.episode_cache_size = 8
        dataset._episode_cache = OrderedDict()
        adapter = MagicMock()
        adapter.load_episode.side_effect = [
            {"target_motion": torch.tensor([1.0])},
            {"target_motion": torch.tensor([2.0])},
        ]
        first = _episode(task_id="task-a", episode_id="0", data_path=Path("a.parquet"))
        second = _episode(task_id="task-b", episode_id="0", data_path=Path("b.parquet"))

        first_motion = dataset._episode_motion(adapter, first)
        second_motion = dataset._episode_motion(adapter, second)

        self.assertEqual(adapter.load_episode.call_count, 2)
        self.assertEqual(first_motion["target_motion"].item(), 1.0)
        self.assertEqual(second_motion["target_motion"].item(), 2.0)

    def test_nested_and_legacy_dataset_selection_remain_compatible(self):
        roots = {
            SOURCE_HUMANOID_ARENA: Path("arena"),
            SOURCE_HUMANOID_EVERYDAY: Path("everyday"),
            SOURCE_HIW500: Path("hiw"),
            SOURCE_UNIFOLM: Path("unifolm"),
        }
        legacy = MultiSourceG1Dataset._normalize_selection(
            {"HSI_sit_sofa": "sonic"}, roots
        )
        nested = MultiSourceG1Dataset._normalize_selection(
            {
                "HumanoidEveryday": {"tasks": ["*"]},
                "HIW500": True,
                "UnifoLM_WBT_Dataset": False,
            },
            roots,
        )

        self.assertEqual(legacy, {SOURCE_HUMANOID_ARENA: {"HSI_sit_sofa": "sonic"}})
        self.assertEqual(
            nested,
            {
                SOURCE_HUMANOID_EVERYDAY: {"tasks": ["*"]},
                SOURCE_HIW500: {},
            },
        )

    def test_explicitly_disabling_every_source_is_an_error(self):
        roots = {
            SOURCE_HUMANOID_ARENA: Path("arena"),
            SOURCE_HIW500: Path("hiw"),
        }

        with self.assertRaisesRegex(ValueError, "disables every"):
            MultiSourceG1Dataset._normalize_selection(
                {"HumanoidArena": False, "HIW500": None}, roots
            )

    def test_source_balanced_sampler_uses_configured_source_weights(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.sampling_mode = "source_balanced"
        dataset._source_names = [SOURCE_HUMANOID_ARENA, SOURCE_HIW500]
        dataset._source_weights = [1.0, 3.0]
        arena_record = (MagicMock(), _episode(source=SOURCE_HUMANOID_ARENA))
        hiw_record = (MagicMock(), _episode(source=SOURCE_HIW500))
        dataset._episodes_by_source = {
            SOURCE_HUMANOID_ARENA: [arena_record],
            SOURCE_HIW500: [hiw_record],
        }
        dataset.sample_stride = 1

        with patch("data.multisource_dataset.random.choices", return_value=[SOURCE_HIW500]) as choices:
            _, selected, cut = dataset._sample_record()

        choices.assert_called_once_with(
            dataset._source_names, weights=dataset._source_weights, k=1
        )
        self.assertEqual(selected.source, SOURCE_HIW500)
        self.assertEqual(cut, 0)

    def test_episode_uniform_sampler_flattens_all_sources_before_sampling(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.sampling_mode = "episode_uniform"
        arena_record = (MagicMock(), _episode(source=SOURCE_HUMANOID_ARENA))
        hiw_record = (MagicMock(), _episode(source=SOURCE_HIW500))
        dataset._all_episode_records = [arena_record, hiw_record]
        dataset.sample_stride = 1

        with patch(
            "data.multisource_dataset.random.choice", return_value=hiw_record
        ) as choice:
            _, selected, cut = dataset._sample_record()

        choice.assert_called_once_with(dataset._all_episode_records)
        self.assertEqual(selected.source, SOURCE_HIW500)
        self.assertEqual(cut, 0)

    def test_source_task_balanced_sampler_selects_source_then_task(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.sampling_mode = "source_task_balanced"
        dataset._source_names = [SOURCE_UNIFOLM, SOURCE_HIW500]
        dataset._source_weights = [3.0, 1.0]
        first = (
            MagicMock(),
            _episode(source=SOURCE_UNIFOLM, task_id="task-a", episode_id="a"),
        )
        second = (
            MagicMock(),
            _episode(source=SOURCE_UNIFOLM, task_id="task-b", episode_id="b"),
        )
        dataset._episodes_by_source_task = {
            SOURCE_UNIFOLM: {"task-a": [first], "task-b": [second]},
            SOURCE_HIW500: {},
        }
        dataset.sample_stride = 1
        rng = MagicMock()
        rng.choices.return_value = [SOURCE_UNIFOLM]
        rng.choice.side_effect = ["task-b", second]
        rng.randrange.return_value = 0

        _, selected, cut = dataset._sample_record(rng)

        rng.choices.assert_called_once_with(
            dataset._source_names, weights=dataset._source_weights, k=1
        )
        self.assertEqual(rng.choice.call_args_list[0].args[0], ["task-a", "task-b"])
        self.assertIs(selected, second[1])
        self.assertEqual(cut, 0)

    def test_window_proportional_sampler_weights_each_episode_by_start_points(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.sampling_mode = "window_proportional"
        short = (MagicMock(), _episode(source_length=8))
        long = (MagicMock(), _episode(source_length=20))
        short[1].sample_count = 2
        long[1].sample_count = 10
        dataset._all_episode_records = [short, long]
        dataset._episode_weights = [2, 10]
        dataset.sample_stride = 1

        with patch(
            "data.multisource_dataset.random.choices", return_value=[long]
        ) as choices:
            _, selected, _ = dataset._sample_record()

        choices.assert_called_once_with(
            dataset._all_episode_records,
            weights=dataset._episode_weights,
            k=1,
        )
        self.assertIs(selected, long[1])

    def test_default_group_size_preserves_the_original_sampling_sequence(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.sampling_seed = 19
        dataset.sampling_mode = "window_proportional"
        dataset.sample_stride = 2
        records = [
            (MagicMock(), _episode(source_length=100, episode_id=str(index)))
            for index in range(5)
        ]
        for index, (_, episode) in enumerate(records):
            episode.first_cut = index
            episode.sample_count = 20 + index
        dataset._all_episode_records = records
        dataset._episode_weights = [episode.sample_count for _, episode in records]

        original = []
        grouped_default = []
        for index in range(64):
            original.append(
                dataset._sample_record(dataset._rng_for_index(index))[1:]
            )
            rng, group_slot = dataset._sampling_state_for_index(index)
            grouped_default.append(
                dataset._sample_grouped_record(rng, group_slot)[1:]
            )

        self.assertEqual(
            [(episode.episode_id, cut) for episode, cut in grouped_default],
            [(episode.episode_id, cut) for episode, cut in original],
        )

    def test_grouped_sampling_reuses_episode_and_avoids_duplicate_windows(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.sampling_seed = 23
        dataset.sampling_mode = "window_proportional"
        dataset.windows_per_episode = 4
        dataset.sample_stride = 1
        records = [
            (MagicMock(), _episode(source_length=100, episode_id=str(index)))
            for index in range(8)
        ]
        for _, episode in records:
            episode.sample_count = 80
        dataset._all_episode_records = records
        dataset._episode_weights = [episode.sample_count for _, episode in records]

        groups = []
        for group_index in range(2):
            group = []
            for index in range(group_index * 4, group_index * 4 + 4):
                rng, group_slot = dataset._sampling_state_for_index(index)
                _, episode, cut = dataset._sample_grouped_record(rng, group_slot)
                group.append((episode.episode_id, cut))
            groups.append(group)

        for group in groups:
            self.assertEqual(len({episode_id for episode_id, _ in group}), 1)
            self.assertEqual(len({cut for _, cut in group}), 4)
        first_group_again = []
        for index in range(4):
            rng, group_slot = dataset._sampling_state_for_index(index)
            _, episode, cut = dataset._sample_grouped_record(rng, group_slot)
            first_group_again.append((episode.episode_id, cut))
        self.assertEqual(first_group_again, groups[0])

    def test_grouped_sampling_uses_only_windows_after_logical_frame_trim(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.action_history = 4
        dataset.action_chunk = 2
        dataset.sample_stride = 1
        dataset.sampling_seed = 37
        dataset.sampling_mode = "episode_uniform"
        dataset.windows_per_episode = 4
        dataset._episode_cache = OrderedDict()
        dataset._runtime_invalid_episodes = set()
        dataset._text_embeddings = {}
        adapter = MagicMock()
        episode = _episode(source_length=12, episode_id="trimmed")
        episode.sample_count = 11
        motion = {
            "observed_motion": torch.zeros(9, 417),
            "observed_motion_valid": torch.ones(9, 417, dtype=torch.bool),
            "target_motion": torch.ones(9, 417),
            "observed_hand": torch.zeros(9, 2),
            "target_hand": torch.ones(9, 2),
            "observed_hand_valid": torch.ones(9, 2, dtype=torch.bool),
            "target_hand_valid": torch.ones(9, 2, dtype=torch.bool),
            "frame_offset": 3,
            "source_frame_offset": 3,
        }
        dataset._all_episode_records = [(adapter, episode)]
        dataset._episode_motion = MagicMock(return_value=motion)
        dataset._read_video_frame = MagicMock(
            return_value=torch.zeros(3, 4, 5, dtype=torch.uint8)
        )

        samples = [dataset[index] for index in range(4)]

        self.assertEqual({sample["episode_id"] for sample in samples}, {"trimmed"})
        self.assertEqual(len({sample["cut_index"] for sample in samples}), 4)
        self.assertTrue(all(sample["cut_index"] >= 3 for sample in samples))
        self.assertTrue(all(sample["motion_cut_index"] >= 0 for sample in samples))

    def test_grouped_trim_rejection_preserves_legacy_episode_probability(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.action_history = 4
        dataset.action_chunk = 2
        dataset.sample_stride = 1
        dataset.sampling_seed = 43
        dataset.sampling_mode = "episode_uniform"
        dataset.windows_per_episode = 4
        dataset._episode_cache = OrderedDict()
        dataset._runtime_invalid_episodes = set()
        dataset._text_embeddings = {}
        trimmed_adapter = MagicMock()
        valid_adapter = MagicMock()
        trimmed = _episode(source_length=12, episode_id="trimmed")
        valid = _episode(source_length=12, episode_id="valid")
        trimmed.sample_count = valid.sample_count = 11
        trimmed_motion = {
            "observed_motion": torch.zeros(9, 417),
            "observed_motion_valid": torch.ones(9, 417, dtype=torch.bool),
            "target_motion": torch.ones(9, 417),
            "observed_hand": torch.zeros(9, 2),
            "target_hand": torch.ones(9, 2),
            "observed_hand_valid": torch.ones(9, 2, dtype=torch.bool),
            "target_hand_valid": torch.ones(9, 2, dtype=torch.bool),
            "frame_offset": 3,
        }
        valid_motion = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in trimmed_motion.items()
            if key != "frame_offset"
        }
        valid_motion = {
            **valid_motion,
            "observed_motion": torch.zeros(12, 417),
            "observed_motion_valid": torch.ones(12, 417, dtype=torch.bool),
            "target_motion": torch.ones(12, 417),
            "observed_hand": torch.zeros(12, 2),
            "target_hand": torch.ones(12, 2),
            "observed_hand_valid": torch.ones(12, 2, dtype=torch.bool),
            "target_hand_valid": torch.ones(12, 2, dtype=torch.bool),
        }

        def load_motion(adapter, _episode_record):
            return trimmed_motion if adapter is trimmed_adapter else valid_motion

        dataset._episode_motion = MagicMock(side_effect=load_motion)
        dataset._read_video_frame = MagicMock(
            return_value=torch.zeros(3, 4, 5, dtype=torch.uint8)
        )

        def episode_sequence(rng):
            attempt = getattr(rng, "_test_attempt", 0)
            rng._test_attempt = attempt + 1
            return (
                (trimmed_adapter, trimmed)
                if attempt == 0
                else (valid_adapter, valid)
            )

        with patch.object(dataset, "_sample_episode", side_effect=episode_sequence), patch(
            "data.multisource_dataset.random.Random.randrange",
            autospec=True,
            side_effect=lambda rng, stop: stop - 1,
        ):
            samples = [dataset[index] for index in range(4)]

        self.assertEqual({sample["episode_id"] for sample in samples}, {"valid"})

    def test_grouped_retry_stays_synchronized_after_invalid_episode(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.action_history = 4
        dataset.action_chunk = 2
        dataset.sample_stride = 1
        dataset.sampling_seed = 41
        dataset.sampling_mode = "episode_uniform"
        dataset.windows_per_episode = 4
        dataset._episode_cache = OrderedDict()
        dataset._runtime_invalid_episodes = set()
        dataset._text_embeddings = {}
        invalid_adapter = MagicMock()
        valid_adapter = MagicMock()
        invalid = _episode(source_length=12, episode_id="invalid")
        valid = _episode(source_length=12, episode_id="valid")
        invalid.sample_count = valid.sample_count = 11
        dataset._all_episode_records = [
            (invalid_adapter, invalid),
            (valid_adapter, valid),
        ]
        valid_motion = {
            "observed_motion": torch.zeros(12, 417),
            "observed_motion_valid": torch.ones(12, 417, dtype=torch.bool),
            "target_motion": torch.ones(12, 417),
            "observed_hand": torch.zeros(12, 2),
            "target_hand": torch.ones(12, 2),
            "observed_hand_valid": torch.ones(12, 2, dtype=torch.bool),
            "target_hand_valid": torch.ones(12, 2, dtype=torch.bool),
        }

        def load_motion(adapter, _episode_record):
            if adapter is invalid_adapter:
                return {"skip_episode": True, "quality_issue": "invalid"}
            return valid_motion

        dataset._episode_motion = MagicMock(side_effect=load_motion)
        dataset._read_video_frame = MagicMock(
            return_value=torch.zeros(3, 4, 5, dtype=torch.uint8)
        )

        with patch.object(
            dataset,
            "_sample_episode",
            side_effect=lambda rng: (
                (invalid_adapter, invalid)
                if rng.random() < 1.0
                else (valid_adapter, valid)
            ),
        ):
            # Keep the episode sequence deterministic while forcing the first
            # attempt invalid and the second valid for every group slot.
            def deterministic_episode(rng):
                attempt = getattr(rng, "_test_attempt", 0)
                rng._test_attempt = attempt + 1
                return (
                    (invalid_adapter, invalid)
                    if attempt == 0
                    else (valid_adapter, valid)
                )

            dataset._sample_episode.side_effect = deterministic_episode
            samples = [dataset[index] for index in range(4)]

        self.assertEqual({sample["episode_id"] for sample in samples}, {"valid"})
        self.assertEqual(len({sample["cut_index"] for sample in samples}), 4)

    def test_grouped_sampling_resume_uses_the_same_absolute_ordinal_sequence(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.sampling_seed = 29
        dataset.sampling_mode = "episode_uniform"
        dataset.windows_per_episode = 4
        dataset.sample_stride = 1
        records = [
            (MagicMock(), _episode(source_length=100, episode_id=str(index)))
            for index in range(6)
        ]
        for _, episode in records:
            episode.sample_count = 80
        dataset._all_episode_records = records

        def sample(indices):
            result = []
            for index in indices:
                rng, group_slot = dataset._sampling_state_for_index(index)
                _, episode, cut = dataset._sample_grouped_record(rng, group_slot)
                result.append((episode.episode_id, cut))
            return result

        uninterrupted = sample(range(32))
        resumed = sample(range(16)) + sample(range(16, 32))

        self.assertEqual(resumed, uninterrupted)

    def test_grouped_window_proportional_sampling_keeps_episode_weights(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.sampling_seed = 31
        dataset.sampling_mode = "window_proportional"
        dataset.windows_per_episode = 4
        dataset.sample_stride = 1
        short = (MagicMock(), _episode(episode_id="short", source_length=20))
        long = (MagicMock(), _episode(episode_id="long", source_length=100))
        short[1].sample_count = 20
        long[1].sample_count = 80
        dataset._all_episode_records = [short, long]
        dataset._episode_weights = [20, 80]

        selected_long = 0
        group_count = 10000
        for group_index in range(group_count):
            rng, group_slot = dataset._sampling_state_for_index(group_index * 4)
            _, episode, _ = dataset._sample_grouped_record(rng, group_slot)
            selected_long += episode.episode_id == "long"

        self.assertAlmostEqual(selected_long / group_count, 0.8, delta=0.02)

    def test_sample_ordinal_is_independent_of_process_random_state(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.sampling_seed = 42
        dataset.sampling_mode = "episode_uniform"
        dataset.sample_stride = 1
        records = [
            (MagicMock(), _episode(source_length=20, episode_id=str(index)))
            for index in range(5)
        ]
        for _, episode in records:
            episode.sample_count = 10
        dataset._all_episode_records = records

        first_rng = dataset._rng_for_index(123456)
        first = dataset._sample_record(first_rng)[1:]
        random.seed(999)
        for _ in range(100):
            random.random()
        second_rng = dataset._rng_for_index(123456)
        second = dataset._sample_record(second_rng)[1:]

        self.assertEqual(first[0].episode_id, second[0].episode_id)
        self.assertEqual(first[1], second[1])

    def test_different_ordinals_produce_a_stable_sampling_sequence(self):
        dataset = MultiSourceG1Dataset.__new__(MultiSourceG1Dataset)
        dataset.sampling_seed = 7
        dataset.sampling_mode = "episode_uniform"
        dataset.sample_stride = 1
        records = [
            (MagicMock(), _episode(source_length=100, episode_id=str(index)))
            for index in range(4)
        ]
        for _, episode in records:
            episode.sample_count = 90
        dataset._all_episode_records = records

        sequence_a = [
            dataset._sample_record(dataset._rng_for_index(index))[1:]
            for index in range(16)
        ]
        sequence_b = [
            dataset._sample_record(dataset._rng_for_index(index))[1:]
            for index in range(16)
        ]
        compact_a = [(episode.episode_id, cut) for episode, cut in sequence_a]
        compact_b = [(episode.episode_id, cut) for episode, cut in sequence_b]

        self.assertEqual(compact_a, compact_b)
        self.assertGreater(len(set(compact_a)), 1)


if __name__ == "__main__":
    unittest.main()
