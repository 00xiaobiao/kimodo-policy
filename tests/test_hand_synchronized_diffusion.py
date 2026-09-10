import unittest
from types import SimpleNamespace

import torch
from torch import nn

from model.kimodo_policy import KimodoPolicy
from model.modules.diffusion import Diffusion


class _FakeRepresentation:
    motion_rep_dim = 4
    slice_dict = {"global_root_heading": slice(0, 2)}
    root_slice = slice(0, 2)
    body_slice = slice(2, 4)

    @staticmethod
    def normalize(motion):
        return motion

    @staticmethod
    def unnormalize(motion):
        return motion

    @staticmethod
    def inverse(motion, **kwargs):
        batch_size, frames, _ = motion.shape
        local_rot_mats = torch.eye(3, device=motion.device).reshape(1, 1, 1, 3, 3)
        return {
            "local_rot_mats": local_rot_mats.expand(batch_size, frames, 1, 3, 3),
            "root_positions": torch.zeros(
                batch_size, frames, 3, device=motion.device
            ),
        }


class _FakeImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, images):
        return torch.zeros(images.shape[0], 196, 8, device=images.device)


class _FakeControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.body_injection_layers = (1,)
        self.root_hint_fusion = nn.Identity()
        self.body_hint_fusion = nn.Identity()
        self.timesteps = []

    def forward(self, timesteps, image_features, sequence_length, future_start):
        self.timesteps.append(timesteps.detach().clone())
        tokens = torch.zeros(
            image_features.shape[0], 196, 8, device=image_features.device
        )
        return {1: tokens}, {1: tokens}


class _FakeDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.timesteps = []
        self.return_hidden_flags = []
        self.inputs = []
        self.motion_masks = []
        self.observed_motions = []

    def forward(
        self,
        x,
        timesteps,
        return_body_hidden=False,
        motion_mask=None,
        observed_motion=None,
        **kwargs,
    ):
        self.timesteps.append(timesteps.detach().clone())
        self.return_hidden_flags.append(bool(return_body_hidden))
        self.inputs.append(x.detach().clone())
        self.motion_masks.append(
            None if motion_mask is None else motion_mask.detach().clone()
        )
        self.observed_motions.append(
            None if observed_motion is None else observed_motion.detach().clone()
        )
        predicted_clean = torch.zeros_like(x)
        if return_body_hidden:
            body_hidden = torch.zeros(x.shape[0], x.shape[1], 8, device=x.device)
            return predicted_clean, body_hidden
        return predicted_clean


class _FakeHandHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.timesteps = []

    def forward(self, noisy_future_hand, timesteps, **kwargs):
        self.timesteps.append(timesteps.detach().clone())
        return torch.zeros_like(noisy_future_hand)


class _FixedHandHead(_FakeHandHead):
    def __init__(self, delta):
        super().__init__()
        self.delta = torch.as_tensor(delta, dtype=torch.float32)

    def forward(self, noisy_future_hand, **kwargs):
        self.timesteps.append(kwargs["timesteps"].detach().clone())
        return self.delta.to(
            device=noisy_future_hand.device, dtype=noisy_future_hand.dtype
        ).expand_as(noisy_future_hand)


class _RecordingSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, use_timesteps, x_t, pred_xstart, timestep):
        family = "hand" if x_t.shape[-1] == 2 else "motion"
        self.calls.append((family, timestep.detach().clone()))
        return pred_xstart


class _KeepNoiseSampler(nn.Module):
    def forward(self, use_timesteps, x_t, pred_xstart, timestep):
        return x_t


def _build_test_policy(sampler):
    policy = KimodoPolicy.__new__(KimodoPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(action_history=2, action_chunk=3)
    policy.representation = _FakeRepresentation()
    policy.denoiser = _FakeDenoiser()
    policy.image_encoder = _FakeImageEncoder()
    policy.controlnet = _FakeControlNet()
    policy.hand_head = _FakeHandHead()
    policy.diffusion = Diffusion(num_base_steps=1000)
    policy.sampler = sampler
    policy.text_encoder = None
    return policy


class HandSynchronizedDiffusionTest(unittest.TestCase):
    def test_rtc_temporal_weights_freeze_then_decay(self):
        weights = KimodoPolicy._rtc_temporal_weights(
            6,
            2,
            1.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        torch.testing.assert_close(weights[:2], torch.ones(2))
        self.assertTrue(torch.all(weights[2:-1] > weights[3:]))
        self.assertGreater(float(weights[2]), 0.5)
        self.assertLess(float(weights[-1]), 0.5)
        self.assertTrue(torch.all((weights >= 0) & (weights <= 1)))

    def test_rtc_inpaints_motion_and_hand_but_not_window_local_root_xz(self):
        policy = _build_test_policy(_KeepNoiseSampler())
        policy.representation.slice_dict["smooth_root_pos"] = slice(0, 3)
        motion_reference = torch.tensor(
            [
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
                [30.0, 31.0, 32.0, 33.0],
            ]
        )
        hand_reference = torch.tensor(
            [[-1.0, 1.0], [0.5, -0.5], [0.25, -0.25]]
        )

        def predict(with_rtc):
            kwargs = {}
            if with_rtc:
                kwargs.update(
                    rtc_motion_reference=motion_reference,
                    rtc_hand_reference=hand_reference,
                    rtc_overlap_frames=3,
                    rtc_frozen_frames=1,
                    rtc_ramp_power=1.0,
                )
            return policy.predict_future(
                instruction="Open the door.",
                egoview=torch.zeros(1, 3, 2, 2),
                history_motion=torch.empty(1, 0, 4),
                hand_history=torch.empty(1, 0, 2),
                diffusion_steps=1,
                squeeze_batch=False,
                text_feat=torch.zeros(1, 1, 6),
                generator=torch.Generator().manual_seed(123),
                **kwargs,
            )

        baseline = predict(False)
        rtc = predict(True)

        # The frozen frame keeps compatible body features exactly.
        torch.testing.assert_close(
            rtc["motion_features"][0, 0, [1, 3]],
            motion_reference[0, [1, 3]],
        )
        # Root x/z keep the fresh window's values instead of importing the old
        # window's incompatible translation gauge.
        torch.testing.assert_close(
            rtc["motion_features"][0, :, [0, 2]],
            baseline["motion_features"][0, :, [0, 2]],
        )
        torch.testing.assert_close(rtc["hand_clean"][0, 0], hand_reference[0])

    def test_rtc_rejects_invalid_configuration_and_reference(self):
        policy = _build_test_policy(_KeepNoiseSampler())
        common = dict(
            instruction="Open the door.",
            egoview=torch.zeros(1, 3, 2, 2),
            history_motion=torch.empty(1, 0, 4),
            hand_history=torch.empty(1, 0, 2),
            diffusion_steps=1,
            squeeze_batch=False,
            text_feat=torch.zeros(1, 1, 6),
        )

        with self.assertRaisesRegex(ValueError, "rtc_overlap_frames"):
            policy.predict_future(
                **common,
                rtc_motion_reference=torch.zeros(2, 4),
                rtc_overlap_frames=0,
            )
        with self.assertRaisesRegex(ValueError, "rtc_motion_reference shape"):
            policy.predict_future(
                **common,
                rtc_motion_reference=torch.zeros(2, 5),
                rtc_overlap_frames=2,
            )
        with self.assertRaisesRegex(ValueError, r"clean \[-1, 1\]"):
            policy.predict_future(
                **common,
                rtc_motion_reference=torch.zeros(2, 4),
                rtc_hand_reference=torch.full((2, 2), 2.0),
                rtc_overlap_frames=2,
            )

    def test_training_uses_state_condition_and_action_clean_target(self):
        policy = _build_test_policy(_KeepNoiseSampler())
        policy.hand_head = None
        policy.config.root_loss_weight = 1.0
        policy.config.body_loss_weight = 1.0
        target_action = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
        state_condition = target_action + 100.0
        condition_mask = torch.zeros(1, 5, 4, dtype=torch.bool)
        condition_mask[:, :2, (0, 2)] = True

        output = policy.training_kimodo_policy_controlnet(
            instruction=["Open the door."],
            egoview=torch.zeros(1, 3, 2, 2),
            gt_motion=target_action,
            gt_mask=torch.ones(1, 5, dtype=torch.bool),
            condition_motion=state_condition,
            condition_motion_mask=condition_mask,
            text_feat=torch.zeros(1, 1, 6),
        )

        self.assertTrue(torch.isfinite(output["loss"]))
        recorded_mask = policy.denoiser.motion_masks[0]
        self.assertTrue(torch.equal(recorded_mask.bool(), condition_mask))
        recorded_condition = policy.denoiser.observed_motions[0]
        torch.testing.assert_close(
            recorded_condition[condition_mask], state_condition[condition_mask]
        )
        self.assertTrue(
            torch.equal(
                recorded_condition[~condition_mask],
                torch.zeros_like(recorded_condition[~condition_mask]),
            )
        )

    def test_partial_history_feature_mask_only_hard_overwrites_known_values(self):
        policy = _build_test_policy(_KeepNoiseSampler())
        history = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
        )
        feature_mask = torch.tensor(
            [[[True, False, True, False], [True, False, True, False]]]
        )

        policy.predict_future(
            instruction="Open the door.",
            egoview=torch.zeros(1, 3, 2, 2),
            history_motion=history,
            history_feature_mask=feature_mask,
            hand_history=torch.zeros(1, 2, 2),
            diffusion_steps=1,
            squeeze_batch=False,
            text_feat=torch.zeros(1, 1, 6),
            generator=torch.Generator().manual_seed(7),
        )

        first_input = policy.denoiser.inputs[0][:, :2]
        torch.testing.assert_close(first_input[feature_mask], history[feature_mask])
        self.assertTrue(
            torch.equal(
                policy.denoiser.motion_masks[0][:, :2].bool(), feature_mask
            )
        )
        self.assertTrue(
            torch.equal(
                policy.denoiser.observed_motions[0][:, :2][feature_mask],
                history[feature_mask],
            )
        )
        self.assertTrue(
            torch.equal(
                policy.denoiser.observed_motions[0][:, :2][~feature_mask],
                torch.zeros_like(history[~feature_mask]),
            )
        )

    def test_prediction_returns_same_window_history_root_boundary(self):
        policy = _build_test_policy(_KeepNoiseSampler())

        def inverse_with_time_root(motion, **kwargs):
            batch_size, frames, _ = motion.shape
            roots = torch.zeros(batch_size, frames, 3)
            roots[:, :, 0] = torch.arange(frames)
            local_rot_mats = torch.eye(3).reshape(1, 1, 1, 3, 3)
            return {
                "local_rot_mats": local_rot_mats.expand(
                    batch_size, frames, 1, 3, 3
                ),
                "root_positions": roots,
            }

        policy.representation.inverse = inverse_with_time_root

        output = policy.predict_future(
            instruction="Open the door.",
            egoview=torch.zeros(1, 3, 2, 2),
            history_motion=torch.zeros(1, 2, 4),
            hand_history=torch.zeros(1, 2, 2),
            diffusion_steps=1,
            squeeze_batch=False,
            text_feat=torch.zeros(1, 1, 6),
            generator=torch.Generator().manual_seed(7),
        )

        self.assertEqual(tuple(output["history_last_root_position"].shape), (1, 3))
        torch.testing.assert_close(
            output["history_last_root_position"], torch.tensor([[1.0, 0.0, 0.0]])
        )

        startup_output = policy.predict_future(
            instruction="Open the door.",
            egoview=torch.zeros(1, 3, 2, 2),
            history_motion=torch.empty(1, 0, 4),
            hand_history=torch.empty(1, 0, 2),
            diffusion_steps=1,
            squeeze_batch=False,
            text_feat=torch.zeros(1, 1, 6),
            generator=torch.Generator().manual_seed(7),
        )
        torch.testing.assert_close(
            startup_output["history_last_root_position"],
            startup_output["root_positions"][:, 0],
        )

    def test_motion_and_hand_use_identical_timesteps_in_one_loop(self):
        policy = _build_test_policy(_RecordingSampler())

        policy.predict_future(
            instruction="Open the door.",
            egoview=torch.zeros(1, 3, 2, 2),
            history_motion=torch.empty(1, 0, 4),
            hand_history=torch.empty(1, 0, 2),
            diffusion_steps=4,
            squeeze_batch=False,
            text_feat=torch.zeros(1, 1, 6),
        )

        expected = [999, 666, 333, 0]
        motion_steps = [int(timestep.item()) for timestep in policy.denoiser.timesteps]
        control_steps = [int(timestep.item()) for timestep in policy.controlnet.timesteps]
        hand_steps = [int(timestep.item()) for timestep in policy.hand_head.timesteps]
        self.assertEqual(motion_steps, expected)
        self.assertEqual(control_steps, expected)
        self.assertEqual(hand_steps, expected)
        self.assertTrue(all(policy.denoiser.return_hidden_flags))
        self.assertEqual(
            [family for family, _ in policy.sampler.calls],
            ["motion", "hand"] * 4,
        )
        self.assertEqual(
            [int(timestep.item()) for _, timestep in policy.sampler.calls],
            [3, 3, 2, 2, 1, 1, 0, 0],
        )

    def test_explicit_generator_reproduces_motion_and_hand_noise(self):
        policy = _build_test_policy(_KeepNoiseSampler())

        def predict(seed):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            return policy.predict_future(
                instruction="Open the door.",
                egoview=torch.zeros(1, 3, 2, 2),
                history_motion=torch.empty(1, 0, 4),
                hand_history=torch.empty(1, 0, 2),
                diffusion_steps=4,
                squeeze_batch=False,
                text_feat=torch.zeros(1, 1, 6),
                generator=generator,
            )

        first = predict(1234)
        second = predict(1234)
        different = predict(5678)

        self.assertTrue(torch.equal(first["motion_features"], second["motion_features"]))
        self.assertTrue(torch.equal(first["hand_clean"], second["hand_clean"]))
        self.assertFalse(torch.equal(first["motion_features"], different["motion_features"]))
        self.assertFalse(torch.equal(first["hand_clean"], different["hand_clean"]))

    def test_continuous_hand_history_is_accepted_and_exposed_as_closure(self):
        policy = _build_test_policy(_KeepNoiseSampler())
        policy.config.hand_control_mode = "continuous"

        output = policy.predict_future(
            instruction="Pick up the object.",
            egoview=torch.zeros(1, 3, 2, 2),
            history_motion=torch.zeros(1, 2, 4),
            hand_history=torch.tensor([[[0.0, 0.25], [0.5, 1.0]]]),
            diffusion_steps=1,
            squeeze_batch=False,
            text_feat=torch.zeros(1, 1, 6),
        )

        self.assertIn("hand_closure", output)
        self.assertTrue(torch.all((output["hand_closure"] >= 0) & (output["hand_closure"] <= 1)))

    def test_continuous_hand_prediction_is_reconstructed_from_measured_state(self):
        policy = _build_test_policy(_RecordingSampler())
        policy.config.hand_control_mode = "continuous"
        policy.config.action_chunk = 3
        policy.hand_head = _FixedHandHead([0.1, -0.2])

        output = policy.predict_future(
            instruction="Pick up the object.",
            egoview=torch.zeros(1, 3, 2, 2),
            history_motion=torch.zeros(1, 2, 4),
            hand_history=torch.tensor([[[0.2, 0.8], [0.25, 0.75]]]),
            diffusion_steps=1,
            squeeze_batch=False,
            text_feat=torch.zeros(1, 1, 6),
        )

        torch.testing.assert_close(
            output["hand_delta"], torch.tensor([[[0.1, -0.2]]]).expand(1, 3, 2)
        )
        torch.testing.assert_close(
            output["hand_closure"], torch.tensor([[[0.35, 0.55]]]).expand(1, 3, 2)
        )
        torch.testing.assert_close(
            output["hand_binary"], torch.tensor([[[0.0, 1.0]]]).expand(1, 3, 2)
        )

    def test_binary_hand_history_still_rejects_intermediate_values(self):
        policy = _build_test_policy(_KeepNoiseSampler())

        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            policy.predict_future(
                instruction="Pick up the object.",
                egoview=torch.zeros(1, 3, 2, 2),
                history_motion=torch.zeros(1, 2, 4),
                hand_history=torch.full((1, 2, 2), 0.5),
                diffusion_steps=1,
                squeeze_batch=False,
                text_feat=torch.zeros(1, 1, 6),
            )

    def test_continuous_training_accepts_intermediate_targets(self):
        policy = _build_test_policy(_KeepNoiseSampler())
        policy.config.hand_control_mode = "continuous"
        policy.config.root_loss_weight = 1.0
        policy.config.body_loss_weight = 1.0
        policy.config.hand_loss_weight = 1.0
        policy.config.hand_transition_loss_weight = 0.2
        output = policy.training_kimodo_policy_controlnet(
            instruction=["Pick up the object."],
            egoview=torch.zeros(1, 3, 2, 2),
            gt_motion=torch.zeros(1, 5, 4),
            gt_mask=torch.ones(1, 5, dtype=torch.bool),
            gt_hand=torch.tensor(
                [[[0.0, 0.0], [0.1, 0.3], [0.4, 0.7], [0.8, 1.0], [1.0, 1.0]]]
            ),
            text_feat=torch.zeros(1, 1, 6),
        )
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_binary_training_accepts_continuous_observation_history(self):
        policy = _build_test_policy(_KeepNoiseSampler())
        policy.config.hand_control_mode = "binary"
        policy.config.hand_observation_mode = "continuous"
        policy.config.root_loss_weight = 1.0
        policy.config.body_loss_weight = 1.0
        policy.config.hand_loss_weight = 1.0
        policy.config.hand_transition_loss_weight = 0.2
        gt_hand = torch.tensor(
            [[[0.1, 0.25], [0.2, 0.75], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]]
        )

        output = policy.training_kimodo_policy_controlnet(
            instruction=["Pick up the object."],
            egoview=torch.zeros(1, 3, 2, 2),
            gt_motion=torch.zeros(1, 5, 4),
            gt_mask=torch.ones(1, 5, dtype=torch.bool),
            gt_hand=gt_hand,
            text_feat=torch.zeros(1, 1, 6),
        )

        self.assertTrue(torch.isfinite(output["loss"]))

    def test_binary_training_still_rejects_intermediate_targets(self):
        policy = _build_test_policy(_KeepNoiseSampler())
        policy.config.root_loss_weight = 1.0
        policy.config.body_loss_weight = 1.0
        gt_hand = torch.zeros(1, 5, 2)
        gt_hand[:, 2:] = 0.5
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            policy.training_kimodo_policy_controlnet(
                instruction=["Pick up the object."],
                egoview=torch.zeros(1, 3, 2, 2),
                gt_motion=torch.zeros(1, 5, 4),
                gt_mask=torch.ones(1, 5, dtype=torch.bool),
                gt_hand=gt_hand,
                text_feat=torch.zeros(1, 1, 6),
            )


if __name__ == "__main__":
    unittest.main()
