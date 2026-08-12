import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from model.kimodo_policy import (
    KimodoPolicy,
    _masked_all_binary,
    _masked_all_finite,
    _masked_mean,
)
from model.modules.backbone import pad_x_and_mask_to_fixed_size
from model.modules.diffusion import DDIMSampler, Diffusion
from model.modules.dinov3_wrapper import DINOv3Encoder
from model.modules.denoiser import TwostageDenoiser
from evaluation.humanoidarena_server import (
    _dtype_from_name,
    _resolve_execution_frames,
    _select_execution_prefix,
)
from utils.geometry import cont6d_to_matrix


class _DtypeCheckingVisionModel(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.projection = nn.Linear(3, 4, dtype=dtype)
        self.config = SimpleNamespace(
            image_size=2,
            hidden_size=4,
            num_register_tokens=0,
        )

    def forward(self, pixel_values: torch.Tensor):
        if pixel_values.dtype != self.projection.weight.dtype:
            raise TypeError(
                f"input dtype {pixel_values.dtype} != weight dtype {self.projection.weight.dtype}"
            )
        pooled = pixel_values.mean(dim=(-2, -1))
        token = self.projection(pooled).unsqueeze(1)
        patch_tokens = token.expand(-1, 4, -1)
        return SimpleNamespace(last_hidden_state=torch.cat((token, patch_tokens), dim=1))


class _RecordingImageProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, images, return_tensors, do_rescale):
        self.calls.append(
            {
                "images": images,
                "return_tensors": return_tensors,
                "do_rescale": do_rescale,
            }
        )
        return {"pixel_values": images.float()}


class _RootStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, *args, **kwargs):
        return self.value.expand(x.shape[0], x.shape[1], 1)


class _BodyStage(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x[..., :1]


class _MotionRep:
    body_slice = slice(1, 2)

    def global_root_to_local_root(self, root_features, **kwargs):
        return root_features


class ModelRobustnessTest(unittest.TestCase):
    def test_masked_operations_keep_one_compiled_graph_when_mask_counts_change(self):
        compile_count = 0

        def counting_backend(graph_module, _example_inputs):
            nonlocal compile_count
            compile_count += 1
            return graph_module.forward

        def masked_operations(values, mask):
            return (
                _masked_all_finite(values, mask),
                _masked_all_binary(values, mask),
                _masked_mean(values, mask),
            )

        compiled = torch.compile(
            masked_operations,
            backend=counting_backend,
            dynamic=False,
        )
        values = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        dense_mask = torch.ones_like(values, dtype=torch.bool)
        sparse_mask = torch.tensor(
            [[True, False, False], [False, True, False]],
            dtype=torch.bool,
        )

        dense_finite, dense_binary, dense_mean = compiled(values, dense_mask)
        sparse_finite, sparse_binary, sparse_mean = compiled(values, sparse_mask)

        self.assertEqual(compile_count, 1)
        self.assertTrue(dense_finite)
        self.assertTrue(dense_binary)
        self.assertTrue(sparse_finite)
        self.assertTrue(sparse_binary)
        torch.testing.assert_close(dense_mean, values.mean())
        torch.testing.assert_close(sparse_mean, torch.tensor(0.0))

    def test_masked_validation_ignores_only_unselected_values(self):
        values = torch.tensor([[0.0, float("nan"), 2.0]])
        mask = torch.tensor([[True, False, True]])

        self.assertTrue(_masked_all_finite(values, mask))
        self.assertFalse(_masked_all_binary(values, mask))

    def test_strict_controlnet_checkpoint_loading_rejects_missing_parameters(self):
        policy = KimodoPolicy.__new__(KimodoPolicy)
        nn.Module.__init__(policy)
        policy.controlnet = nn.Sequential(nn.Linear(2, 2))
        checkpoint = {
            "model": {
                "controlnet.0.weight": policy.controlnet[0].weight.detach().clone(),
            },
            "global_step": 17,
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = os.path.join(directory, "training_state.pt")
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "0.bias"):
                policy.load_controlnet_checkpoint(checkpoint_path, strict=True)
            self.assertEqual(
                policy.load_controlnet_checkpoint(checkpoint_path, strict=False),
                17,
            )

    def test_degenerate_6d_rotations_remain_finite(self):
        rotations = cont6d_to_matrix(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                ]
            )
        )

        self.assertTrue(torch.isfinite(rotations).all())

    def test_training_noise_uses_base_schedule_after_ddim_respace(self):
        diffusion = Diffusion(num_base_steps=1000)
        use_timesteps, _ = diffusion.space_timesteps(10)
        diffusion.calc_diffusion_vars(use_timesteps)
        clean = torch.ones(1, 2, 3)
        noise = torch.zeros_like(clean)

        sample = diffusion.q_sample(clean, torch.tensor([999]), noise)

        expected = diffusion.sqrt_alphas_cumprod_base[999] * clean
        torch.testing.assert_close(sample, expected)

    def test_diffusion_schedule_stays_fp32_for_low_precision_model(self):
        diffusion = Diffusion(num_base_steps=100).to(dtype=torch.bfloat16)

        for buffer in diffusion.buffers():
            self.assertEqual(buffer.dtype, torch.float32)

        use_timesteps, _ = diffusion.space_timesteps(10)
        diffusion.calc_diffusion_vars(use_timesteps)
        sampler = DDIMSampler(diffusion)
        noisy = torch.randn(2, 4, 5, dtype=torch.bfloat16)
        predicted_clean = torch.randn_like(noisy)
        sample = sampler(
            use_timesteps,
            noisy,
            predicted_clean,
            torch.tensor([9, 5]),
        )
        final_sample = sampler(
            use_timesteps,
            noisy,
            predicted_clean,
            torch.tensor([0, 0]),
        )

        self.assertEqual(sample.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(sample).all())
        self.assertTrue(torch.isfinite(final_sample).all())
        self.assertGreater(diffusion.sqrt_recipm1_alphas_cumprod[0].item(), 0.0)

    def test_text_padding_crops_oversized_inputs(self):
        features = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)
        mask = torch.ones(2, 7, dtype=torch.bool)

        cropped_features, cropped_mask = pad_x_and_mask_to_fixed_size(features, mask, 5)

        torch.testing.assert_close(cropped_features, features[:, :5])
        self.assertTrue(torch.equal(cropped_mask, mask[:, :5]))

    def test_dino_input_matches_low_precision_model_dtype(self):
        encoder = DINOv3Encoder.__new__(DINOv3Encoder)
        nn.Module.__init__(encoder)
        encoder.model = _DtypeCheckingVisionModel(torch.bfloat16)
        encoder.image_processor = _RecordingImageProcessor()
        encoder.image_size = 2
        encoder.output_dim = 4
        encoder.num_register_tokens = 0

        output = encoder(torch.ones(1, 3, 2, 2, dtype=torch.float32))

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(tuple(output.shape), (1, 4, 4))
        self.assertFalse(encoder.image_processor.calls[0]["do_rescale"])

    def test_dino_uint8_input_uses_official_processor_rescaling(self):
        encoder = DINOv3Encoder.__new__(DINOv3Encoder)
        nn.Module.__init__(encoder)
        encoder.model = _DtypeCheckingVisionModel(torch.float32)
        encoder.image_processor = _RecordingImageProcessor()
        encoder.image_size = 2
        encoder.output_dim = 4
        encoder.num_register_tokens = 0
        images = torch.full((1, 3, 2, 2), 255, dtype=torch.uint8)

        output = encoder(images)

        self.assertEqual(tuple(output.shape), (1, 4, 4))
        call = encoder.image_processor.calls[0]
        self.assertIs(call["images"], images)
        self.assertEqual(call["return_tensors"], "pt")
        self.assertTrue(call["do_rescale"])

    def test_dino_rejects_numerically_unsafe_fp16_weights(self):
        encoder = DINOv3Encoder.__new__(DINOv3Encoder)
        nn.Module.__init__(encoder)
        encoder.model = _DtypeCheckingVisionModel(torch.float16)
        encoder.image_processor = _RecordingImageProcessor()
        encoder.image_size = 2
        encoder.output_dim = 4
        encoder.num_register_tokens = 0

        with self.assertRaisesRegex(RuntimeError, "DINOv3 FP16"):
            encoder(torch.ones(1, 3, 2, 2, dtype=torch.float32))

    def test_server_rejects_fp16_mode(self):
        with self.assertRaisesRegex(ValueError, "Unsupported dtype"):
            _dtype_from_name("fp16")

    def test_execution_frames_resolve_against_prediction_chunk(self):
        self.assertEqual(_resolve_execution_frames(0, 50), 50)
        self.assertEqual(_resolve_execution_frames(15, 50), 15)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _resolve_execution_frames(-1, 50)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            _resolve_execution_frames(51, 50)

    def test_execution_prefix_truncates_all_prediction_outputs(self):
        future_features = torch.arange(50 * 3, dtype=torch.float32).reshape(50, 3)
        local_rot_mats = torch.eye(3).reshape(1, 1, 3, 3).repeat(50, 2, 1, 1)
        root_positions = torch.arange(50 * 3, dtype=torch.float32).reshape(50, 3)

        future, rotations, roots = _select_execution_prefix(
            future_features,
            local_rot_mats,
            root_positions,
            execution_frames=15,
        )

        self.assertEqual(tuple(future.shape), (15, 3))
        self.assertEqual(tuple(rotations.shape), (15, 2, 3, 3))
        self.assertEqual(tuple(roots.shape), (15, 3))
        torch.testing.assert_close(future[-1], future_features[14])
        torch.testing.assert_close(roots[-1], root_positions[14])

    def test_training_can_explicitly_detach_root_from_body_loss(self):
        denoiser = TwostageDenoiser.__new__(TwostageDenoiser)
        nn.Module.__init__(denoiser)
        denoiser.motion_mask_mode = "none"
        denoiser.motion_rep = _MotionRep()
        denoiser.root_model = _RootStage()
        denoiser.body_model = _BodyStage()
        inputs = torch.zeros(1, 3, 2)
        mask = torch.ones(1, 3, dtype=torch.bool)
        text = torch.zeros(1, 1, 1)
        timesteps = torch.zeros(1, dtype=torch.long)

        detached_output = denoiser(
            inputs,
            mask,
            text,
            torch.ones(1, 1, dtype=torch.bool),
            timesteps,
            detach_root_for_body=True,
        )
        detached_output[..., 1:].sum().backward()
        self.assertEqual(denoiser.root_model.value.grad.abs().item(), 0.0)

        denoiser.root_model.value.grad = None
        coupled_output = denoiser(
            inputs,
            mask,
            text,
            torch.ones(1, 1, dtype=torch.bool),
            timesteps,
            detach_root_for_body=False,
        )
        coupled_output[..., 1:].sum().backward()
        self.assertGreater(denoiser.root_model.value.grad.abs().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
