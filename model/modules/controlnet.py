import copy

import torch
from torch import nn

from .backbone import TimestepEmbedder


def _zero_linear(dim: int) -> nn.Linear:
    layer = nn.Linear(dim, dim)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class _SharedCrossAttention(nn.Module):
    """Read visual tokens using the current future-motion hidden tokens as queries."""

    def __init__(self, latent_dim: int, num_heads: int):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(
                f"latent_dim={latent_dim} must be divisible by num_heads={num_heads}"
            )
        self.latent_dim = int(latent_dim)
        self.query_norm = nn.LayerNorm(latent_dim)
        self.context_norm = nn.LayerNorm(latent_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(
        self,
        future_motion_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if future_motion_tokens.ndim != 3 or visual_tokens.ndim != 3:
            raise ValueError(
                "Expected future motion and visual tokens to be rank-3 tensors, got "
                f"{tuple(future_motion_tokens.shape)} and {tuple(visual_tokens.shape)}"
            )
        if future_motion_tokens.shape[0] != visual_tokens.shape[0]:
            raise ValueError(
                "Future motion and visual tokens must share the batch dimension, got "
                f"{future_motion_tokens.shape[0]} and {visual_tokens.shape[0]}"
            )
        if (
            future_motion_tokens.shape[-1] != self.latent_dim
            or visual_tokens.shape[-1] != self.latent_dim
        ):
            raise ValueError(
                f"Expected latent dim {self.latent_dim}, got "
                f"{future_motion_tokens.shape[-1]} and {visual_tokens.shape[-1]}"
            )

        query = self.query_norm(future_motion_tokens)
        context = self.context_norm(visual_tokens)
        attended, _ = self.attention(
            query=query,
            key=context,
            value=context,
            need_weights=False,
        )
        return attended


class _VisualTokenMLP(nn.Module):
    """Compress spatial visual tokens into future-position visual tokens."""

    def __init__(
        self,
        image_token_count: int,
        future_token_count: int,
        hidden_token_dims: tuple[int, ...],
    ):
        super().__init__()
        token_dims = (
            int(image_token_count),
            *(int(dim) for dim in hidden_token_dims),
            int(future_token_count),
        )
        if any(dim <= 0 for dim in token_dims):
            raise ValueError(f"All visual token MLP dimensions must be positive, got {token_dims}")

        layers = []
        for layer_index, (input_dim, output_dim) in enumerate(
            zip(token_dims[:-1], token_dims[1:])
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if layer_index < len(token_dims) - 2:
                layers.append(nn.GELU())
        self.network = nn.Sequential(*layers)
        self.image_token_count = token_dims[0]
        self.future_token_count = token_dims[-1]

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if visual_tokens.ndim != 3:
            raise ValueError(
                "Expected visual tokens [B, image_tokens, latent_dim], got "
                f"{tuple(visual_tokens.shape)}"
            )
        if visual_tokens.shape[1] != self.image_token_count:
            raise ValueError(
                f"Expected {self.image_token_count} visual tokens, "
                f"got {visual_tokens.shape[1]}"
            )

        # Apply the MLP over the token axis while preserving every latent channel.
        future_tokens = self.network(visual_tokens.transpose(1, 2)).transpose(1, 2)
        if future_tokens.shape[1] != self.future_token_count:
            raise RuntimeError(
                f"Visual token MLP produced {future_tokens.shape[1]} tokens, "
                f"expected {self.future_token_count}"
            )
        return future_tokens


class _LayerHintProjection(nn.Module):
    """Fuse two visual paths and produce one zero-initialized future-motion hint."""

    def __init__(
        self,
        image_token_count: int,
        future_token_count: int,
        latent_dim: int,
        token_mlp_hidden_dims: tuple[int, ...],
    ):
        super().__init__()
        self.future_token_count = int(future_token_count)
        self.latent_dim = int(latent_dim)
        self.visual_token_mlp = _VisualTokenMLP(
            image_token_count=image_token_count,
            future_token_count=self.future_token_count,
            hidden_token_dims=token_mlp_hidden_dims,
        )
        self.fusion_zero_projection = _zero_linear(latent_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        future_motion_tokens: torch.Tensor,
        shared_cross_attention: _SharedCrossAttention,
    ) -> torch.Tensor:
        if future_motion_tokens.ndim != 3:
            raise ValueError(
                "Expected future motion tokens [B, future_tokens, latent_dim], got "
                f"{tuple(future_motion_tokens.shape)}"
            )
        if future_motion_tokens.shape[1:] != (
            self.future_token_count,
            self.latent_dim,
        ):
            raise ValueError(
                "Expected future motion token shape "
                f"[B, {self.future_token_count}, {self.latent_dim}], got "
                f"{tuple(future_motion_tokens.shape)}"
            )

        mlp_future_tokens = self.visual_token_mlp(visual_tokens)
        attended_visual_tokens = shared_cross_attention(
            future_motion_tokens,
            visual_tokens,
        )
        if mlp_future_tokens.shape != attended_visual_tokens.shape:
            raise RuntimeError(
                "Visual MLP and shared cross-attention outputs must have identical shapes, "
                f"got {tuple(mlp_future_tokens.shape)} and "
                f"{tuple(attended_visual_tokens.shape)}"
            )

        fused_future_tokens = mlp_future_tokens + attended_visual_tokens
        return self.fusion_zero_projection(fused_future_tokens)


class _StageHintFusion(nn.Module):
    """Fuse per-layer visual tokens with motion queries using one shared attention."""

    def __init__(
        self,
        injection_layers: tuple[int, ...],
        image_token_count: int,
        future_token_count: int,
        latent_dim: int,
        num_heads: int,
        token_mlp_hidden_dims: tuple[int, ...],
    ):
        super().__init__()
        self.injection_layers = tuple(int(layer) for layer in injection_layers)
        self.shared_cross_attention = _SharedCrossAttention(
            latent_dim=latent_dim,
            num_heads=num_heads,
        )
        self.projections = nn.ModuleList(
            _LayerHintProjection(
                image_token_count=image_token_count,
                future_token_count=future_token_count,
                latent_dim=latent_dim,
                token_mlp_hidden_dims=token_mlp_hidden_dims,
            )
            for _ in self.injection_layers
        )

    def forward(
        self,
        layer_index: int,
        visual_tokens: torch.Tensor,
        future_motion_tokens: torch.Tensor,
    ) -> torch.Tensor:
        try:
            projection_index = self.injection_layers.index(int(layer_index))
        except ValueError as error:
            raise ValueError(
                f"Layer {layer_index} is not a configured injection layer "
                f"{self.injection_layers}"
            ) from error
        return self.projections[projection_index](
            visual_tokens,
            future_motion_tokens,
            self.shared_cross_attention,
        )


class ControlNet(nn.Module):
    """Sequential sparse visual ControlNet for the frozen two-stage Kimodo denoiser."""

    def __init__(
        self,
        denoiser,
        image_feat_dim: int,
        image_token_count: int = 196,
        motion_token_count: int = 150,
        num_control_layers: int = 8,
        future_token_count: int | None = None,
        token_mlp_hidden_dims: tuple[int, ...] = (256, 128),
    ):
        super().__init__()
        root_model = denoiser.root_model
        body_model = denoiser.body_model
        if root_model.latent_dim != body_model.latent_dim:
            raise ValueError("Root and body ControlNet stages require the same latent dimension")

        latent_dim = root_model.latent_dim
        self.image_token_count = int(image_token_count)
        self.motion_token_count = int(motion_token_count)
        self.future_token_count = int(
            motion_token_count if future_token_count is None else future_token_count
        )
        if self.future_token_count <= 0 or self.future_token_count > self.motion_token_count:
            raise ValueError(
                "future_token_count must be in "
                f"[1, {self.motion_token_count}], got {self.future_token_count}"
            )
        self.token_mlp_hidden_dims = tuple(int(dim) for dim in token_mlp_hidden_dims)
        if any(dim <= 0 for dim in self.token_mlp_hidden_dims):
            raise ValueError(
                "token_mlp_hidden_dims must contain only positive dimensions, got "
                f"{self.token_mlp_hidden_dims}"
            )
        root_num_heads = int(getattr(root_model, "num_heads", 8))
        body_num_heads = int(getattr(body_model, "num_heads", root_num_heads))
        self.root_injection_layers = self._injection_layers(
            len(root_model.seqTransEncoder.layers), num_control_layers
        )
        self.body_injection_layers = self._injection_layers(
            len(body_model.seqTransEncoder.layers), num_control_layers
        )
        self.sequence_pos_encoder = copy.deepcopy(root_model.sequence_pos_encoder)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)
        self.image_projection = nn.Linear(image_feat_dim, latent_dim)

        self.root_layers = nn.ModuleList(
            copy.deepcopy(root_model.seqTransEncoder.layers[layer_index])
            for layer_index in self.root_injection_layers
        )
        self.body_layers = nn.ModuleList(
            copy.deepcopy(body_model.seqTransEncoder.layers[layer_index])
            for layer_index in self.body_injection_layers
        )
        self.root_hint_fusion = _StageHintFusion(
            injection_layers=self.root_injection_layers,
            image_token_count=self.image_token_count,
            future_token_count=self.future_token_count,
            latent_dim=latent_dim,
            num_heads=root_num_heads,
            token_mlp_hidden_dims=self.token_mlp_hidden_dims,
        )
        self.body_hint_fusion = _StageHintFusion(
            injection_layers=self.body_injection_layers,
            image_token_count=self.image_token_count,
            future_token_count=self.future_token_count,
            latent_dim=latent_dim,
            num_heads=body_num_heads,
            token_mlp_hidden_dims=self.token_mlp_hidden_dims,
        )

    @staticmethod
    def _injection_layers(backbone_depth: int, num_control_layers: int) -> tuple[int, ...]:
        backbone_depth = int(backbone_depth)
        num_control_layers = int(num_control_layers)
        if num_control_layers <= 0 or num_control_layers > backbone_depth:
            raise ValueError(
                f"num_control_layers must be in [1, {backbone_depth}], got {num_control_layers}"
            )
        if backbone_depth % num_control_layers != 0:
            raise ValueError(
                f"Backbone depth {backbone_depth} must be divisible by "
                f"num_control_layers={num_control_layers}"
            )
        stride = backbone_depth // num_control_layers
        return tuple(range(stride - 1, backbone_depth, stride))

    def _collect_visual_tokens(
        self,
        xseq: torch.Tensor,
        layers: nn.ModuleList,
        injection_layers: tuple[int, ...],
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        visual_tokens_by_layer = {}
        for layer_index, layer in zip(injection_layers, layers):
            xseq = layer(xseq, src_key_padding_mask=padding_mask)
            visual_tokens_by_layer[layer_index] = xseq[:, 1:]
        return xseq, visual_tokens_by_layer

    def forward(
        self,
        timesteps: torch.Tensor,
        image_features: torch.Tensor,
        sequence_length: int,
        future_start: int,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        if image_features.shape[1] != self.image_token_count:
            raise ValueError(
                f"Expected {self.image_token_count} image tokens, got {image_features.shape[1]}"
            )
        if sequence_length != self.motion_token_count:
            raise ValueError(
                f"Expected motion length {self.motion_token_count}, got {sequence_length}"
            )
        if future_start < 0 or future_start > sequence_length:
            raise ValueError(f"Invalid future_start={future_start} for length {sequence_length}")
        available_future_length = sequence_length - future_start
        if available_future_length != self.future_token_count:
            raise ValueError(
                "The configured future token count must exactly match the future motion "
                f"range: configured={self.future_token_count}, "
                f"available={available_future_length}"
            )

        time_token = self.embed_timestep(timesteps)
        image_tokens = self.image_projection(image_features)
        xseq = self.sequence_pos_encoder(torch.cat([time_token, image_tokens], dim=1))
        padding_mask = torch.zeros(
            xseq.shape[:2], dtype=torch.bool, device=xseq.device
        )

        xseq, root_visual_tokens = self._collect_visual_tokens(
            xseq,
            self.root_layers,
            self.root_injection_layers,
            padding_mask,
        )
        _, body_visual_tokens = self._collect_visual_tokens(
            xseq,
            self.body_layers,
            self.body_injection_layers,
            padding_mask,
        )
        return root_visual_tokens, body_visual_tokens
