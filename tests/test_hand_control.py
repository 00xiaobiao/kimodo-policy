import os
import tempfile
import unittest

import torch
from torch import nn

from model.kimodo_policy import KimodoPolicy
from model.modules.hand_control import HandTokenDiffusionDenoiser


class _RecordingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence_shape = None
        self.padding_mask_shape = None

    def forward(self, sequence, src_key_padding_mask=None):
        self.sequence_shape = tuple(sequence.shape)
        self.padding_mask_shape = tuple(src_key_padding_mask.shape)
        return sequence


class HandControlTest(unittest.TestCase):
    def _head(self, **overrides):
        config = {
            "input_dim": 16,
            "hidden_dim": 32,
            "future_token_count": 50,
            "image_token_count": 196,
            "num_layers": 4,
            "num_heads": 4,
            "ffn_dim": 64,
            "max_diffusion_steps": 1000,
        }
        config.update(overrides)
        return HandTokenDiffusionDenoiser(**config)

    def test_four_layer_encoder_receives_298_separate_tokens(self):
        head = self._head()
        self.assertEqual(len(head.encoder.layers), 4)
        recorder = _RecordingEncoder()
        head.encoder = recorder

        output = head(
            visual_tokens=torch.randn(2, 196, 16),
            body_future_tokens=torch.randn(2, 50, 16),
            noisy_future_hand=torch.randn(2, 50, 2),
            current_hand_state=torch.zeros(2, 2),
            timesteps=torch.tensor([7, 19]),
        )

        self.assertEqual(recorder.sequence_shape, (2, 298, 32))
        self.assertEqual(recorder.padding_mask_shape, (2, 298))
        self.assertEqual(tuple(output.shape), (2, 50, 2))

    def test_visual_and_body_conditions_are_detached(self):
        head = self._head(
            future_token_count=5,
            image_token_count=7,
            num_layers=1,
        )
        visual = torch.randn(2, 7, 16, requires_grad=True)
        body = torch.randn(2, 5, 16, requires_grad=True)
        noisy_hand = torch.randn(2, 5, 2, requires_grad=True)

        output = head(
            visual,
            body,
            noisy_hand,
            torch.zeros(2, 2),
            torch.tensor([1, 2]),
        )
        output.square().mean().backward()

        self.assertIsNone(visual.grad)
        self.assertIsNone(body.grad)
        self.assertGreater(noisy_hand.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            sum(
                parameter.grad.abs().sum().item()
                for parameter in head.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )

    def test_last_valid_hand_state_defaults_open(self):
        history = torch.tensor(
            [
                [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            ]
        )
        mask = torch.tensor([[True, False, True], [False, False, False]])

        current = KimodoPolicy._last_valid_hand_state(history, mask)

        torch.testing.assert_close(current[0], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(current[1], torch.tensor([0.0, 0.0]))

    def test_strict_checkpoint_requires_enabled_hand_weights(self):
        policy = KimodoPolicy.__new__(KimodoPolicy)
        nn.Module.__init__(policy)
        policy.controlnet = nn.Linear(2, 2)
        policy.hand_head = nn.Linear(2, 2)
        checkpoint = {
            "global_step": 12,
            "model": {
                "controlnet.weight": policy.controlnet.weight.detach().clone(),
                "controlnet.bias": policy.controlnet.bias.detach().clone(),
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "training_state.pt")
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(RuntimeError, "No hand-head weights"):
                policy.load_controlnet_checkpoint(path, strict=True)
            self.assertEqual(policy.load_controlnet_checkpoint(path, strict=False), 12)


if __name__ == "__main__":
    unittest.main()
