import unittest
from pathlib import Path

import numpy as np
import torch

from motion.g1_reference import (
    CANONICAL_G1_JOINT_NAMES_29,
    HumanoidArenaActionDecoder,
    resample_motion_chunk,
)
from skeleton.definitions import G1Skeleton34


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = PROJECT_ROOT / "skeleton/assets/g1skel34/xml/g1.xml"


class G1ReferenceTest(unittest.TestCase):
    def test_chunk_resampling_distributes_root_motion_evenly(self):
        source_frames = 50
        local_rot_mats = torch.eye(3).reshape(1, 1, 3, 3).repeat(
            source_frames, 1, 1, 1
        )
        root_positions = torch.zeros(source_frames, 3)
        root_positions[:, 0] = torch.arange(source_frames, dtype=torch.float32)
        previous_root = torch.tensor([-1.0, 0.0, 0.0])

        _, resampled_root = resample_motion_chunk(
            local_rot_mats,
            root_positions,
            source_fps=30.0,
            target_fps=50.0,
            previous_local_rot_mat=local_rot_mats[0],
            previous_root_position=previous_root,
        )

        self.assertEqual(resampled_root.shape[0], 83)
        deltas = torch.diff(
            torch.cat((previous_root[None], resampled_root), dim=0), dim=0
        )[:, 0]
        torch.testing.assert_close(
            deltas,
            torch.full_like(deltas, 50.0 / 83.0),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(resampled_root[-1], root_positions[-1])

    def test_fifteen_model_frames_become_twenty_five_control_frames(self):
        source_frames = 15
        local_rot_mats = torch.eye(3).reshape(1, 1, 3, 3).repeat(
            source_frames, 1, 1, 1
        )
        root_positions = torch.zeros(source_frames, 3)

        resampled_rotations, resampled_root = resample_motion_chunk(
            local_rot_mats,
            root_positions,
            source_fps=30.0,
            target_fps=50.0,
            previous_local_rot_mat=local_rot_mats[0],
            previous_root_position=root_positions[0],
        )

        self.assertEqual(resampled_rotations.shape[0], 25)
        self.assertEqual(resampled_root.shape[0], 25)

    def test_root_planar_gauge_translation_preserves_encoded_actions(self):
        """A shared local-coordinate translation must not change root deltas."""
        decoder = HumanoidArenaActionDecoder(G1Skeleton34(), XML_PATH, fps=30.0)
        source_frames = 7
        local_rot_mats = torch.eye(3).reshape(1, 1, 3, 3).repeat(
            source_frames, 34, 1, 1
        )
        root_positions = torch.tensor(
            [
                [0.10, 0.82, -0.20],
                [0.16, 0.82, -0.16],
                [0.23, 0.82, -0.11],
                [0.31, 0.82, -0.05],
                [0.40, 0.82, 0.02],
                [0.50, 0.82, 0.10],
                [0.61, 0.82, 0.19],
            ],
            dtype=torch.float32,
        )
        previous_root_position = torch.tensor([0.04, 0.82, -0.24])
        planar_offset = torch.tensor([12.0, 0.0, -7.0])

        resampled_rotations, resampled_roots = resample_motion_chunk(
            local_rot_mats,
            root_positions,
            source_fps=30.0,
            target_fps=50.0,
            previous_local_rot_mat=local_rot_mats[0],
            previous_root_position=previous_root_position,
        )
        shifted_rotations, shifted_roots = resample_motion_chunk(
            local_rot_mats,
            root_positions + planar_offset,
            source_fps=30.0,
            target_fps=50.0,
            previous_local_rot_mat=local_rot_mats[0],
            previous_root_position=previous_root_position + planar_offset,
        )

        baseline = decoder.encode(
            resampled_rotations,
            resampled_roots,
            previous_root_position=previous_root_position,
        )
        shifted = decoder.encode(
            shifted_rotations,
            shifted_roots,
            previous_root_position=previous_root_position + planar_offset,
        )

        torch.testing.assert_close(shifted, baseline, rtol=1e-5, atol=1e-5)

    def test_root_boundary_gauge_mismatch_changes_only_first_planar_delta(self):
        """A boundary from another local gauge can create a false first step."""
        decoder = HumanoidArenaActionDecoder(G1Skeleton34(), XML_PATH, fps=30.0)
        local_rot_mats = torch.eye(3).reshape(1, 1, 3, 3).repeat(3, 34, 1, 1)
        root_positions = torch.tensor(
            [[0.00, 0.82, 0.00], [0.10, 0.82, 0.02], [0.20, 0.82, 0.04]],
            dtype=torch.float32,
        )
        previous_root_position = torch.tensor([-0.10, 0.82, -0.02])
        wrong_boundary = previous_root_position + torch.tensor([1.0, 0.0, 0.0])

        aligned = decoder.encode(
            local_rot_mats,
            root_positions,
            previous_root_position=previous_root_position,
        )
        mismatched = decoder.encode(
            local_rot_mats,
            root_positions,
            previous_root_position=wrong_boundary,
        )

        self.assertGreater(
            float(torch.linalg.vector_norm(mismatched[0, :2] - aligned[0, :2])),
            0.5,
        )
        torch.testing.assert_close(mismatched[1:, :2], aligned[1:, :2])
        torch.testing.assert_close(mismatched[:, 2:], aligned[:, 2:])

    def test_decoded_joint_positions_match_mujoco(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("mujoco is not installed")

        skeleton = G1Skeleton34()
        decoder = HumanoidArenaActionDecoder(skeleton, XML_PATH, fps=30.0)
        actions = torch.zeros(30, 40)
        actions[:, 2] = 0.8
        actions[:, 3:9] = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        actions[:, 9:38] = 0.25
        decoded_positions = decoder.decode(actions)["posed_joints"][0].numpy()

        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        data = mujoco.MjData(model)
        data.qpos[:3] = np.array([0.0, 0.0, 0.8])
        data.qpos[3] = 1.0
        for joint_name in CANONICAL_G1_JOINT_NAMES_29:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            data.qpos[model.jnt_qposadr[joint_id]] = 0.25
        mujoco.mj_forward(model, data)

        coordinate_change = decoder.mujoco_to_kimodo.numpy()
        expected_positions = np.empty((skeleton.nbjoints, 3), dtype=np.float32)
        expected_positions.fill(np.nan)
        pelvis_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        expected_positions[0] = coordinate_change @ data.xpos[pelvis_id]
        xml_joint_names = [
            model.joint(index).name
            for index in range(model.njnt)
            if model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
        ]
        for joint_name, skeleton_index in zip(
            xml_joint_names, decoder.xml_to_skeleton.tolist()
        ):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            expected_positions[skeleton_index] = (
                coordinate_change @ data.xanchor[joint_id]
            )

        valid = np.isfinite(expected_positions).all(axis=1)
        np.testing.assert_allclose(
            decoded_positions[valid],
            expected_positions[valid],
            rtol=0.0,
            atol=2e-5,
        )

    def test_action_pose_decoder_matches_full_decoder_pose_exactly(self):
        decoder = HumanoidArenaActionDecoder(
            G1Skeleton34(), XML_PATH, fps=30.0
        )
        generator = torch.Generator().manual_seed(1234)
        actions = torch.zeros(24, 40)
        actions[:, :3] = torch.randn(24, 3, generator=generator) * 0.01
        actions[:, 2] += 0.8
        root_angles = torch.randn(24, generator=generator) * 0.2
        actions[:, 3:9] = torch.stack(
            (
                torch.cos(root_angles),
                -torch.sin(root_angles),
                torch.sin(root_angles),
                torch.cos(root_angles),
                torch.zeros_like(root_angles),
                torch.zeros_like(root_angles),
            ),
            dim=-1,
        )
        actions[:, 9:38] = torch.randn(24, 29, generator=generator) * 0.1

        pose = decoder.decode_action_pose(actions)
        full = decoder.decode(actions)

        self.assertEqual(set(pose), {"local_rot_mats", "root_positions"})
        self.assertTrue(torch.equal(pose["local_rot_mats"], full["local_rot_mats"]))
        self.assertTrue(torch.equal(pose["root_positions"], full["root_positions"]))

    def test_configuration_pose_decoder_matches_full_decoder_pose_exactly(self):
        decoder = HumanoidArenaActionDecoder(
            G1Skeleton34(), XML_PATH, fps=30.0
        )
        generator = torch.Generator().manual_seed(5678)
        joint_positions = torch.randn(32, 29, generator=generator) * 0.1
        root_positions = torch.randn(32, 3, generator=generator) * 0.05
        root_positions[:, 2] += 0.8
        yaw = torch.randn(32, generator=generator) * 0.2
        root_quaternions = torch.stack(
            (
                torch.cos(yaw / 2),
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                torch.sin(yaw / 2),
            ),
            dim=-1,
        )
        planar_origin = torch.tensor([0.03, -0.04])

        pose = decoder.decode_joint_configuration_pose(
            joint_positions,
            root_positions,
            root_quaternions=root_quaternions,
            planar_origin=planar_origin,
        )
        full = decoder.decode_joint_configuration(
            joint_positions,
            root_positions,
            root_quaternions=root_quaternions,
            planar_origin=planar_origin,
        )

        self.assertEqual(set(pose), {"local_rot_mats", "root_positions"})
        self.assertTrue(torch.equal(pose["local_rot_mats"], full["local_rot_mats"]))
        self.assertTrue(torch.equal(pose["root_positions"], full["root_positions"]))


if __name__ == "__main__":
    unittest.main()
