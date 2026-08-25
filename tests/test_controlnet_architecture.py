import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from model.modules.backbone import PositionalEncoding, TransformerEncoderBlock
from model.modules.controlnet import ControlNet


class _Stage(nn.Module):
    def __init__(self, depth: int = 4, latent_dim: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = 2
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout=0.0)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=2,
            dim_feedforward=16,
            dropout=0.0,
            batch_first=True,
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=depth)


class _Denoiser(nn.Module):
    def __init__(self, depth: int = 4):
        super().__init__()
        self.root_model = _Stage(depth=depth)
        self.body_model = _Stage(depth=depth)


class _RecordingFuser(nn.Module):
    def __init__(self, impulse_layer=None):
        super().__init__()
        self.impulse_layer = impulse_layer
        self.queries = []

    def forward(self, layer_index, visual_tokens, future_motion_tokens):
        self.queries.append((layer_index, future_motion_tokens.detach().clone()))
        if layer_index == self.impulse_layer:
            return torch.ones_like(future_motion_tokens)
        return torch.zeros_like(future_motion_tokens)


class ControlNetArchitectureTest(unittest.TestCase):
    @staticmethod
    def _controlnet(
        detach_root_control_for_body: bool = False,
        control_fusion_mode: str = "both",
        controlnet_scale: int = 1,
        num_control_layers: int = 2,
        denoiser=None,
    ):
        return ControlNet(
            denoiser or _Denoiser(),
            image_feat_dim=6,
            image_token_count=4,
            motion_token_count=6,
            num_control_layers=num_control_layers,
            controlnet_scale=controlnet_scale,
            future_token_count=2,
            token_mlp_hidden_dims=(6, 4),
            detach_root_control_for_body=detach_root_control_for_body,
            control_fusion_mode=control_fusion_mode,
        )

    @staticmethod
    def _backbone(num_layers: int = 4):
        return TransformerEncoderBlock(
            input_dim=6,
            output_dim=5,
            skeleton=SimpleNamespace(nbjoints=1),
            llm_shape=[1, 10],
            use_text_mask=True,
            latent_dim=8,
            ff_size=16,
            num_layers=num_layers,
            num_heads=2,
            activation="gelu",
            dropout=0.0,
            pe_dropout=0.0,
            norm_first=False,
            num_text_tokens_override=3,
            input_first_heading_angle=True,
        )

    def test_sparse_control_layers_return_visual_tokens(self):
        controlnet = self._controlnet()
        self.assertEqual(controlnet.root_injection_layers, (1, 3))
        self.assertEqual(controlnet.body_injection_layers, (1, 3))
        self.assertEqual(len(controlnet.root_layers), 2)
        self.assertEqual(len(controlnet.body_layers), 2)

        root_tokens, body_tokens = controlnet(
            timesteps=torch.tensor([3, 7]),
            image_features=torch.randn(2, 4, 6),
            sequence_length=6,
            future_start=4,
        )
        self.assertEqual(tuple(root_tokens), (1, 3))
        self.assertEqual(tuple(body_tokens), (1, 3))
        for visual_tokens in [*root_tokens.values(), *body_tokens.values()]:
            self.assertEqual(tuple(visual_tokens.shape), (2, 4, 8))

    def test_controlnet_scale_one_preserves_legacy_structure(self):
        torch.manual_seed(23)
        denoiser = _Denoiser()
        controlnet = self._controlnet(denoiser=denoiser)
        self.assertEqual(controlnet.controlnet_scale, 1)
        self.assertEqual(controlnet.root_injection_layers, (1, 3))
        self.assertEqual(controlnet.root_branch_injection_layers, (0, 1))
        self.assertEqual(len(controlnet.root_layers), 2)
        for branch_layer, source_layer in zip(
            controlnet.root_layers,
            (denoiser.root_model.seqTransEncoder.layers[1],
             denoiser.root_model.seqTransEncoder.layers[3]),
        ):
            for name, value in source_layer.state_dict().items():
                torch.testing.assert_close(value, branch_layer.state_dict()[name])

    def test_scaled_controlnet_has_random_prefix_and_copied_suffix_per_segment(self):
        torch.manual_seed(29)
        denoiser = _Denoiser(depth=16)
        controlnet = self._controlnet(
            denoiser=denoiser,
            num_control_layers=4,
            controlnet_scale=8,
        )
        self.assertEqual(controlnet.root_injection_layers, (3, 7, 11, 15))
        self.assertEqual(controlnet.root_branch_injection_layers, (7, 15, 23, 31))
        self.assertEqual(len(controlnet.root_layers), 32)
        self.assertEqual(len(controlnet.body_layers), 32)

        source_layers = denoiser.root_model.seqTransEncoder.layers
        branch_layers = controlnet.root_layers
        for segment_index in range(4):
            branch_start = segment_index * 8
            source_start = segment_index * 4
            # The last four branch layers copy the corresponding Kimodo span.
            for offset in range(4):
                source = source_layers[source_start + offset]
                copied = branch_layers[branch_start + 4 + offset]
                for name, value in source.state_dict().items():
                    torch.testing.assert_close(value, copied.state_dict()[name])
            # The first four layers are independently initialized, not copies.
            for offset in range(4):
                random_layer = branch_layers[branch_start + offset]
                source = source_layers[source_start + offset]
                self.assertTrue(
                    any(
                        not torch.equal(value, source.state_dict()[name])
                        for name, value in random_layer.state_dict().items()
                    )
                )

        root_tokens, body_tokens = controlnet(
            timesteps=torch.tensor([3, 7]),
            image_features=torch.randn(2, 4, 6),
            sequence_length=6,
            future_start=4,
        )
        self.assertEqual(tuple(root_tokens), (3, 7, 11, 15))
        self.assertEqual(tuple(body_tokens), (3, 7, 11, 15))

        # Closed-loop check: the branch emits keys at the frozen Kimodo layer
        # indices, so the real backbone accepts and fuses all four hints.
        backbone = self._backbone(num_layers=16)
        output = backbone(
            x=torch.randn(2, 6, 6),
            x_pad_mask=torch.ones(2, 6, dtype=torch.bool),
            text_feat=torch.randn(2, 3, 10),
            text_feat_pad_mask=torch.ones(2, 3, dtype=torch.bool),
            timesteps=torch.tensor([1, 2]),
            first_heading_angle=torch.zeros(2),
            control_visual_tokens=root_tokens,
            hint_fuser=controlnet.root_hint_fusion,
            future_start=4,
        )
        self.assertEqual(tuple(output.shape), (2, 6, 5))

    def test_equivalent_total_depths_copy_the_same_kimodo_layers(self):
        """Changing injection grouping must not change copied Kimodo layers."""
        torch.manual_seed(31)
        source = _Denoiser(depth=16)
        scaled = self._controlnet(
            denoiser=source,
            num_control_layers=4,
            controlnet_scale=2,
        )
        legacy_grouping = self._controlnet(
            denoiser=copy.deepcopy(source),
            num_control_layers=8,
            controlnet_scale=1,
        )

        self.assertEqual(scaled.root_branch_injection_layers, (1, 3, 5, 7))
        self.assertEqual(legacy_grouping.root_branch_injection_layers, tuple(range(8)))
        self.assertEqual(scaled.root_injection_layers, (3, 7, 11, 15))
        self.assertEqual(legacy_grouping.root_injection_layers, (1, 3, 5, 7, 9, 11, 13, 15))

        for scaled_layer, legacy_layer in zip(scaled.root_layers, legacy_grouping.root_layers):
            for name, value in legacy_layer.state_dict().items():
                torch.testing.assert_close(value, scaled_layer.state_dict()[name])
        for scaled_layer, legacy_layer in zip(scaled.body_layers, legacy_grouping.body_layers):
            for name, value in legacy_layer.state_dict().items():
                torch.testing.assert_close(value, scaled_layer.state_dict()[name])

    def test_non_positive_controlnet_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "controlnet_scale"):
            self._controlnet(controlnet_scale=0)

    def test_root_to_body_detach_preserves_values_and_isolates_body_gradients(self):
        torch.manual_seed(5)
        shared = self._controlnet(detach_root_control_for_body=False)
        isolated = copy.deepcopy(shared)
        isolated.detach_root_control_for_body = True
        shared.train()
        isolated.train()
        timesteps = torch.tensor([3, 7])
        image_features = torch.randn(2, 4, 6)

        shared_root, shared_body = shared(
            timesteps=timesteps,
            image_features=image_features,
            sequence_length=6,
            future_start=4,
        )
        isolated_root, isolated_body = isolated(
            timesteps=timesteps,
            image_features=image_features,
            sequence_length=6,
            future_start=4,
        )
        for layer_index in shared_root:
            torch.testing.assert_close(
                shared_root[layer_index], isolated_root[layer_index]
            )
        for layer_index in shared_body:
            torch.testing.assert_close(
                shared_body[layer_index], isolated_body[layer_index]
            )

        sum(value.square().mean() for value in shared_body.values()).backward()
        sum(value.square().mean() for value in isolated_body.values()).backward()

        shared_root_grad = sum(
            parameter.grad.abs().sum()
            for parameter in shared.root_layers.parameters()
            if parameter.grad is not None
        )
        isolated_root_grads = [
            parameter.grad for parameter in isolated.root_layers.parameters()
        ]
        shared_body_grad = sum(
            parameter.grad.abs().sum()
            for parameter in shared.body_layers.parameters()
            if parameter.grad is not None
        )
        isolated_body_grad = sum(
            parameter.grad.abs().sum()
            for parameter in isolated.body_layers.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(shared_root_grad.item(), 0.0)
        self.assertTrue(all(gradient is None for gradient in isolated_root_grads))
        self.assertGreater(shared.image_projection.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNone(isolated.image_projection.weight.grad)
        self.assertGreater(shared_body_grad.item(), 0.0)
        self.assertGreater(isolated_body_grad.item(), 0.0)

    def test_dual_path_fusion_is_zero_initialized_without_learnable_queries(self):
        torch.manual_seed(7)
        controlnet = self._controlnet()
        parameter_names = dict(controlnet.named_parameters())
        self.assertFalse(any("future_queries" in name for name in parameter_names))

        root_tokens, body_tokens = controlnet(
            timesteps=torch.tensor([3, 7]),
            image_features=torch.randn(2, 4, 6),
            sequence_length=6,
            future_start=4,
        )
        future_motion_tokens = torch.randn(2, 2, 8)
        for fuser, visual_tokens_by_layer in (
            (controlnet.root_hint_fusion, root_tokens),
            (controlnet.body_hint_fusion, body_tokens),
        ):
            for layer_index, visual_tokens in visual_tokens_by_layer.items():
                hint = fuser(layer_index, visual_tokens, future_motion_tokens)
                self.assertEqual(tuple(hint.shape), (2, 2, 8))
                self.assertTrue(torch.equal(hint, torch.zeros_like(hint)))

    def test_default_mode_matches_explicit_both_structure(self):
        default_controlnet = self._controlnet()
        explicit_controlnet = self._controlnet(control_fusion_mode="both")
        self.assertEqual(
            tuple(default_controlnet.state_dict()),
            tuple(explicit_controlnet.state_dict()),
        )

    def test_control_fusion_modes_only_construct_selected_paths(self):
        for mode in ("both", "cross_attn", "mlp"):
            with self.subTest(mode=mode):
                controlnet = self._controlnet(control_fusion_mode=mode)
                parameter_names = tuple(name for name, _ in controlnet.named_parameters())
                has_mlp = any("visual_token_mlp" in name for name in parameter_names)
                has_cross_attention = any(
                    "shared_cross_attention" in name for name in parameter_names
                )
                self.assertEqual(has_mlp, mode in {"both", "mlp"})
                self.assertEqual(
                    has_cross_attention, mode in {"both", "cross_attn"}
                )

    def test_control_fusion_modes_preserve_shape_and_zero_initialization(self):
        for mode in ("both", "cross_attn", "mlp"):
            with self.subTest(mode=mode):
                controlnet = self._controlnet(control_fusion_mode=mode)
                root_tokens, body_tokens = controlnet(
                    timesteps=torch.tensor([3, 7]),
                    image_features=torch.randn(2, 4, 6),
                    sequence_length=6,
                    future_start=4,
                )
                future_motion_tokens = torch.randn(2, 2, 8)
                for fuser, visual_tokens_by_layer in (
                    (controlnet.root_hint_fusion, root_tokens),
                    (controlnet.body_hint_fusion, body_tokens),
                ):
                    for layer_index, visual_tokens in visual_tokens_by_layer.items():
                        hint = fuser(layer_index, visual_tokens, future_motion_tokens)
                        self.assertEqual(tuple(hint.shape), (2, 2, 8))
                        self.assertTrue(torch.equal(hint, torch.zeros_like(hint)))

    def test_invalid_control_fusion_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "control_fusion_mode"):
            self._controlnet(control_fusion_mode="invalid")

    def test_future_motion_hidden_is_cross_attention_query_and_both_paths_get_gradients(self):
        torch.manual_seed(11)
        controlnet = self._controlnet()
        fuser = controlnet.root_hint_fusion
        projection = fuser.projections[0]
        nn.init.eye_(projection.fusion_zero_projection.weight)
        nn.init.zeros_(projection.fusion_zero_projection.bias)

        visual_tokens = torch.randn(2, 4, 8, requires_grad=True)
        future_motion_tokens = torch.randn(2, 2, 8, requires_grad=True)
        mlp_tokens = projection.visual_token_mlp(visual_tokens)
        attended_tokens = fuser.shared_cross_attention(
            future_motion_tokens,
            visual_tokens,
        )
        hint = fuser(
            layer_index=1,
            visual_tokens=visual_tokens,
            future_motion_tokens=future_motion_tokens,
        )

        torch.testing.assert_close(hint, mlp_tokens + attended_tokens)
        hint.square().mean().backward()
        first_mlp_linear = next(
            module for module in projection.visual_token_mlp.network if isinstance(module, nn.Linear)
        )
        self.assertGreater(first_mlp_linear.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            fuser.shared_cross_attention.attention.in_proj_weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(future_motion_tokens.grad.abs().sum().item(), 0.0)
        self.assertGreater(visual_tokens.grad.abs().sum().item(), 0.0)

    def test_future_range_must_match_configured_query_count(self):
        controlnet = self._controlnet()
        with self.assertRaisesRegex(ValueError, "exactly match"):
            controlnet(
                timesteps=torch.tensor([3]),
                image_features=torch.randn(1, 4, 6),
                sequence_length=6,
                future_start=3,
            )

    def test_backbone_passes_each_layers_current_future_hidden_to_fuser(self):
        backbone = self._backbone()
        fuser = _RecordingFuser()
        output = backbone(
            x=torch.randn(2, 6, 6),
            x_pad_mask=torch.ones(2, 6, dtype=torch.bool),
            text_feat=torch.randn(2, 3, 10),
            text_feat_pad_mask=torch.ones(2, 3, dtype=torch.bool),
            timesteps=torch.tensor([1, 2]),
            first_heading_angle=torch.zeros(2),
            control_visual_tokens={
                1: torch.randn(2, 4, 8),
                3: torch.randn(2, 4, 8),
            },
            hint_fuser=fuser,
            future_start=4,
        )
        self.assertEqual(tuple(output.shape), (2, 6, 5))
        self.assertEqual([layer for layer, _ in fuser.queries], [1, 3])
        for _, query in fuser.queries:
            self.assertEqual(tuple(query.shape), (2, 2, 8))

    def test_later_layer_query_contains_earlier_injection_effect(self):
        torch.manual_seed(19)
        baseline_backbone = self._backbone()
        injected_backbone = copy.deepcopy(baseline_backbone)
        baseline_fuser = _RecordingFuser()
        injected_fuser = _RecordingFuser(impulse_layer=1)
        common = dict(
            x=torch.randn(1, 6, 6),
            x_pad_mask=torch.ones(1, 6, dtype=torch.bool),
            text_feat=torch.randn(1, 3, 10),
            text_feat_pad_mask=torch.ones(1, 3, dtype=torch.bool),
            timesteps=torch.tensor([1]),
            first_heading_angle=torch.zeros(1),
            control_visual_tokens={
                1: torch.randn(1, 4, 8),
                3: torch.randn(1, 4, 8),
            },
            future_start=4,
        )
        baseline_backbone(**common, hint_fuser=baseline_fuser)
        injected_backbone(**common, hint_fuser=injected_fuser)

        torch.testing.assert_close(
            baseline_fuser.queries[0][1],
            injected_fuser.queries[0][1],
        )
        self.assertFalse(
            torch.allclose(
                baseline_fuser.queries[1][1],
                injected_fuser.queries[1][1],
            )
        )

    def test_backbone_and_real_dynamic_fusion_integrate(self):
        controlnet = self._controlnet()
        backbone = self._backbone()
        root_tokens, _ = controlnet(
            timesteps=torch.tensor([3, 7]),
            image_features=torch.randn(2, 4, 6),
            sequence_length=6,
            future_start=4,
        )
        output = backbone(
            x=torch.randn(2, 6, 6),
            x_pad_mask=torch.ones(2, 6, dtype=torch.bool),
            text_feat=torch.randn(2, 3, 10),
            text_feat_pad_mask=torch.ones(2, 3, dtype=torch.bool),
            timesteps=torch.tensor([1, 2]),
            first_heading_angle=torch.zeros(2),
            control_visual_tokens=root_tokens,
            hint_fuser=controlnet.root_hint_fusion,
            future_start=4,
        )
        self.assertEqual(tuple(output.shape), (2, 6, 5))

    def test_backbone_accepts_sparse_hint_mapping(self):
        backbone = self._backbone()
        output = backbone(
            x=torch.randn(2, 6, 6),
            x_pad_mask=torch.ones(2, 6, dtype=torch.bool),
            text_feat=torch.randn(2, 3, 10),
            text_feat_pad_mask=torch.ones(2, 3, dtype=torch.bool),
            timesteps=torch.tensor([1, 2]),
            first_heading_angle=torch.zeros(2),
            hints={1: torch.randn(2, 6, 8), 3: torch.randn(2, 6, 8)},
        )
        self.assertEqual(tuple(output.shape), (2, 6, 5))

    def test_dynamic_fusion_requires_complete_arguments(self):
        backbone = self._backbone()
        common = dict(
            x=torch.randn(1, 6, 6),
            x_pad_mask=torch.ones(1, 6, dtype=torch.bool),
            text_feat=torch.randn(1, 3, 10),
            text_feat_pad_mask=torch.ones(1, 3, dtype=torch.bool),
            timesteps=torch.tensor([1]),
            first_heading_angle=torch.zeros(1),
        )
        with self.assertRaisesRegex(ValueError, "must either both be provided"):
            backbone(
                **common,
                control_visual_tokens={1: torch.randn(1, 4, 8)},
            )

    def test_old_learnable_query_architecture_is_absent(self):
        controlnet = ControlNet(
            _Denoiser(),
            image_feat_dim=6,
            image_token_count=4,
            motion_token_count=6,
            num_control_layers=2,
            future_token_count=2,
            token_mlp_hidden_dims=(6, 4),
        )
        self.assertFalse(hasattr(controlnet, "root_future_query_resampler"))
        self.assertFalse(hasattr(controlnet, "body_future_query_resampler"))


if __name__ == "__main__":
    unittest.main()
