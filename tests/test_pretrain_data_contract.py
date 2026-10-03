import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from model.kimodo_policy import KimodoPolicy
from model.modules.diffusion import Diffusion
from motion.representation.kimodo_motionrep import KimodoMotionRep
from skeleton.definitions import G1Skeleton34


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATS_PATH = PROJECT_ROOT / "checkpoints/Kimodo-G1-RP-v1/stats/motion"


class _ContractImageEncoder(nn.Module):
    def forward(self, images):
        if images.dtype != torch.uint8:
            raise TypeError("DataLoader images must remain uint8 before DINO preprocessing")
        if tuple(images.shape[1:]) != (3, 480, 640):
            raise ValueError(f"Unexpected egoview shape {tuple(images.shape)}")
        return torch.zeros(images.shape[0], 196, 8, device=images.device)


class _ContractControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.01))
        self.body_injection_layers = (1,)
        self.root_hint_fusion = nn.Identity()
        self.body_hint_fusion = nn.Identity()

    def forward(self, timesteps, image_features, sequence_length, future_start):
        if sequence_length != 150 or future_start != 100:
            raise ValueError("Kimodo must receive 100 history + 50 future frames")
        tokens = self.scale * torch.ones_like(image_features)
        return {1: tokens}, {1: tokens}


class _ContractDenoiser(nn.Module):
    def forward(
        self,
        x,
        root_control_tokens,
        body_control_tokens,
        return_body_hidden=False,
        **_kwargs,
    ):
        control = root_control_tokens[1].mean(dim=(1, 2), keepdim=True)
        predicted = control.reshape(x.shape[0], 1, 1).expand_as(x)
        if return_body_hidden:
            hidden = body_control_tokens[1].mean(dim=1, keepdim=True).expand(
                x.shape[0], x.shape[1], 8
            )
            return predicted, hidden
        return predicted


class _ContractHandHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, noisy_future_hand, **_kwargs):
        return self.scale * torch.ones_like(noisy_future_hand)


def _build_contract_policy():
    policy = KimodoPolicy.__new__(KimodoPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        action_history=100,
        action_chunk=50,
        root_loss_weight=2.0,
        body_loss_weight=1.0,
        hand_loss_weight=1.0,
        hand_transition_loss_weight=0.2,
    )
    policy.representation = KimodoMotionRep(
        skeleton=G1Skeleton34(), fps=30, stats_path=str(STATS_PATH)
    )
    policy.denoiser = _ContractDenoiser()
    policy.image_encoder = _ContractImageEncoder()
    policy.controlnet = _ContractControlNet()
    policy.hand_head = _ContractHandHead()
    policy.diffusion = Diffusion(num_base_steps=1000)
    policy.text_encoder = None
    return policy


class PretrainDataContractTest(unittest.TestCase):
    def _batch(self, batch_size=3):
        generator = torch.Generator().manual_seed(123)
        gt_motion = torch.randn(
            batch_size, 150, 417, generator=generator
        ) * 0.05
        gt_motion[:, :, 1] = 0.8
        gt_motion[:, :, 3] = 1.0
        gt_motion[:, :, 4] = 0.0
        gt_motion[:, 0, 0] = 0.0
        gt_motion[:, 0, 2] = 0.0
        condition_motion = gt_motion.clone()
        condition_motion[:, 100:] = 0.0
        condition_motion_mask = torch.zeros(
            batch_size, 150, 417, dtype=torch.bool
        )
        condition_motion_mask[:, :100] = True
        gt_mask = torch.ones(batch_size, 150, dtype=torch.bool)
        gt_hand = torch.randint(
            0, 2, (batch_size, 150, 2), generator=generator
        ).float()
        gt_hand_mask = torch.ones(batch_size, 150, 2, dtype=torch.bool)
        return {
            "instruction": ["test instruction"] * batch_size,
            "egoview": torch.zeros(
                batch_size, 3, 480, 640, dtype=torch.uint8
            ),
            "gt_motion": gt_motion,
            "condition_motion": condition_motion,
            "condition_motion_mask": condition_motion_mask,
            "gt_hand": gt_hand,
            "gt_hand_mask": gt_hand_mask,
            "gt_mask": gt_mask,
            "text_embedding": torch.zeros(
                batch_size, 1, 4096, dtype=torch.bfloat16
            ),
            "text_length": torch.ones(batch_size, dtype=torch.long),
        }

    def test_batch_shapes_dtypes_and_masks_match_kimodo_contract(self):
        batch = self._batch()

        self.assertEqual(tuple(batch["egoview"].shape), (3, 3, 480, 640))
        self.assertEqual(batch["egoview"].dtype, torch.uint8)
        self.assertEqual(tuple(batch["gt_motion"].shape), (3, 150, 417))
        self.assertEqual(batch["gt_motion"].dtype, torch.float32)
        self.assertEqual(tuple(batch["gt_hand"].shape), (3, 150, 2))
        self.assertEqual(tuple(batch["text_embedding"].shape), (3, 1, 4096))
        self.assertEqual(batch["text_embedding"].dtype, torch.bfloat16)
        self.assertFalse(batch["condition_motion_mask"][:, 100:].any())
        self.assertTrue(
            torch.all(
                ~batch["condition_motion_mask"]
                | batch["gt_mask"].unsqueeze(-1)
            )
        )
        self.assertEqual(set(torch.unique(batch["gt_hand"]).tolist()), {0.0, 1.0})

    @unittest.skipUnless(
        STATS_PATH.is_dir(), "Kimodo motion statistics are not installed"
    )
    def test_real_417d_representation_normalizes_batch_without_shape_drift(self):
        policy = _build_contract_policy()
        batch = self._batch()

        normalized = policy.representation.normalize(batch["gt_motion"])
        restored = policy.representation.unnormalize(normalized)

        self.assertEqual(tuple(normalized.shape), (3, 150, 417))
        self.assertTrue(torch.isfinite(normalized).all())
        torch.testing.assert_close(restored, batch["gt_motion"], rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(
        STATS_PATH.is_dir(), "Kimodo motion statistics are not installed"
    )
    def test_kimodo_training_entry_accepts_batch_and_backpropagates(self):
        policy = _build_contract_policy()
        batch = self._batch()

        output = policy(
            batch["instruction"],
            batch["egoview"],
            batch["gt_motion"],
            batch["gt_mask"],
            condition_motion=batch["condition_motion"],
            condition_motion_mask=batch["condition_motion_mask"],
            gt_hand=batch["gt_hand"],
            gt_hand_mask=batch["gt_hand_mask"],
            text_feat=batch["text_embedding"],
            text_length=batch["text_length"],
        )
        output["loss"].backward()

        for name in (
            "loss",
            "motion_loss",
            "root_loss",
            "body_loss",
            "hand_loss",
            "hand_state_loss",
            "hand_transition_loss",
        ):
            self.assertTrue(torch.isfinite(output[name]), msg=name)
        self.assertIsNotNone(policy.controlnet.scale.grad)
        self.assertIsNotNone(policy.hand_head.scale.grad)

    @unittest.skipUnless(
        STATS_PATH.is_dir(), "Kimodo motion statistics are not installed"
    )
    def test_both_motion_loss_modes_forward_and_backward(self):
        component_keys = (
            "root_position_loss",
            "root_heading_loss",
            "joint_position_loss",
            "joint_velocity_loss",
            "joint_rotation_loss",
            "foot_contact_loss",
            "fk_loss",
        )
        weights = KimodoPolicy.KIMODO_SMOOTH_L1_WEIGHTS

        for motion_loss_type in ("mse", "kimodo_smooth_l1"):
            with self.subTest(motion_loss_type=motion_loss_type):
                policy = _build_contract_policy()
                policy.config.motion_loss_type = motion_loss_type
                batch = self._batch(batch_size=1)

                output = policy(
                    batch["instruction"],
                    batch["egoview"],
                    batch["gt_motion"],
                    batch["gt_mask"],
                    condition_motion=batch["condition_motion"],
                    condition_motion_mask=batch["condition_motion_mask"],
                    gt_hand=batch["gt_hand"],
                    gt_hand_mask=batch["gt_hand_mask"],
                    text_feat=batch["text_embedding"],
                    text_length=batch["text_length"],
                )
                output["loss"].backward()

                self.assertTrue(torch.isfinite(output["loss"]))
                self.assertTrue(torch.isfinite(output["motion_loss"]))
                self.assertTrue(torch.isfinite(policy.controlnet.scale.grad).all())
                self.assertTrue(torch.isfinite(policy.hand_head.scale.grad).all())
                if motion_loss_type == "mse":
                    self.assertTrue(all(key not in output for key in component_keys))
                    target = policy.representation.normalize(batch["gt_motion"])[
                        :, policy.config.action_history :
                    ]
                    prediction = torch.full_like(
                        target,
                        policy.controlnet.scale.detach(),
                    )
                    root_slice = policy.representation.root_slice
                    body_slice = policy.representation.body_slice
                    expected_root = nn.functional.mse_loss(
                        prediction[..., root_slice], target[..., root_slice]
                    )
                    expected_body = nn.functional.mse_loss(
                        prediction[..., body_slice], target[..., body_slice]
                    )
                    expected_motion_loss = (
                        policy.config.root_loss_weight * expected_root
                        + policy.config.body_loss_weight * expected_body
                    )
                    torch.testing.assert_close(
                        output["motion_loss"], expected_motion_loss
                    )
                else:
                    self.assertTrue(all(key in output for key in component_keys))
                    self.assertGreater(output["fk_loss"].item(), 0.0)
                    expected_motion_loss = sum(
                        weights[key] * output[key] for key in component_keys
                    )
                    torch.testing.assert_close(
                        output["motion_loss"], expected_motion_loss
                    )

    @unittest.skipUnless(
        STATS_PATH.is_dir(), "Kimodo motion statistics are not installed"
    )
    def test_kimodo_smooth_l1_weights_are_configurable(self):
        policy = _build_contract_policy()
        policy.config.motion_loss_type = "kimodo_smooth_l1"
        policy.config.kimodo_smooth_l1_weights = {
            "root_position": 1.5,
            "joint_velocity": 0.5,
            "fk": 0.0,
        }
        batch = self._batch(batch_size=1)

        output = policy(
            batch["instruction"],
            batch["egoview"],
            batch["gt_motion"],
            batch["gt_mask"],
            condition_motion=batch["condition_motion"],
            condition_motion_mask=batch["condition_motion_mask"],
            gt_hand=batch["gt_hand"],
            gt_hand_mask=batch["gt_hand_mask"],
            text_feat=batch["text_embedding"],
            text_length=batch["text_length"],
        )
        weights = policy._resolve_kimodo_smooth_l1_weights()
        expected_motion_loss = sum(
            weights[key] * output[key]
            for key in KimodoPolicy.KIMODO_SMOOTH_L1_WEIGHTS
        )

        self.assertEqual(weights["root_position_loss"], 1.5)
        self.assertEqual(weights["joint_velocity_loss"], 0.5)
        self.assertEqual(weights["joint_position_loss"], 10.0)
        self.assertEqual(weights["fk_loss"], 0.0)
        self.assertEqual(output["fk_loss"].item(), 0.0)
        torch.testing.assert_close(output["motion_loss"], expected_motion_loss)

    @unittest.skipUnless(
        STATS_PATH.is_dir(), "Kimodo motion statistics are not installed"
    )
    def test_fk_loss_uses_the_417d_joint_position_coordinate_system(self):
        policy = _build_contract_policy()
        representation = policy.representation
        batch_size, frame_count = 1, 30
        joint_count = representation.skeleton.nbjoints
        local_rotations = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(
            batch_size, frame_count, joint_count, 1, 1
        )
        root_positions = torch.zeros(batch_size, frame_count, 3)
        root_positions[..., 0] = torch.linspace(0.0, 1.0, frame_count)
        root_positions[..., 1] = 0.8
        features = representation(
            local_rotations,
            root_positions,
            to_normalize=True,
            lengths=torch.tensor([frame_count]),
        )
        valid = torch.ones(batch_size, frame_count, dtype=torch.bool)

        exact_fk_loss = policy._kimodo_fk_position_loss(features, features, valid)
        torch.testing.assert_close(
            exact_fk_loss,
            torch.zeros_like(exact_fk_loss),
            atol=1e-6,
            rtol=0,
        )

        prediction = features.detach().clone()
        rotation_slice = representation.slice_dict["global_rot_data"]
        prediction[..., rotation_slice.start] += 0.1
        prediction.requires_grad_(True)
        fk_loss = policy._kimodo_fk_position_loss(prediction, features, valid)
        fk_loss.backward()

        self.assertGreater(fk_loss.item(), 0.0)
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad[..., rotation_slice].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
