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


if __name__ == "__main__":
    unittest.main()
