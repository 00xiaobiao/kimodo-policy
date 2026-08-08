import tempfile
import unittest
import random
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from data.multisource_dataset import (
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
    _canonicalize_kimodo_window_translation,
    _humanoid_everyday_instruction,
    _stereo_crop,
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

    def test_everyday_task_catalog_description_overrides_stale_episode_text(self):
        instruction = _humanoid_everyday_instruction(
            {
                "task": "Articulated/adjust_the_angle_of_a_phone_stand",
                "description": "Tilt the phone stand upward.",
            },
            {"instruction": "Wipe the desk."},
        )

        self.assertEqual(instruction, "Tilt the phone stand upward.")

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

            result = ParquetEpisodeReader().read(episode, ["value"])

            self.assertEqual(result["value"], [3, 4, 5, 6])

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
        self.assertEqual(dataset._sample_record.call_count, 2)
        self.assertTrue(sample["gt_mask"].all())

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
