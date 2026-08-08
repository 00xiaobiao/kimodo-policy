import math
import os
import torch
from torch import nn
from omegaconf import OmegaConf
from typing import List, Optional

from .modules.denoiser import TwostageDenoiser
from .modules.dinov3_wrapper import DINOv3Encoder
from .modules.diffusion import DDIMSampler, Diffusion
from .modules.controlnet import ControlNet
from .modules.hand_control import HandTokenDiffusionDenoiser
from skeleton.definitions import G1Skeleton34
from motion.representation.kimodo_motionrep import KimodoMotionRep
from dataclasses import dataclass


@dataclass
class KimodoPolicyConfig:
    fps: int = 30
    motion_mask_mode: str = "concat"
    dinov3_model_name: str = "dinov3-vitl16-pretrain-lvd1689m"
    dinov3_checkpoint: Optional[str] = None
    action_chunk: int = 50    # 预测的未来帧数
    action_history: int = 100 # 作为约束的历史帧数
    load_text_encoder: bool = True
    controlnet_num_layers: int = 8
    root_loss_weight: float = 2.0
    body_loss_weight: float = 1.0
    enable_hand_head: bool = False
    hand_hidden_dim: int = 256
    hand_num_layers: int = 4
    hand_num_heads: int = 4
    hand_ffn_dim: int = 1024
    hand_loss_weight: float = 1.0
    hand_transition_loss_weight: float = 0.2
    hand_init_seed: int = 3407


class KimodoPolicy(nn.Module):
    def __init__(self, config: KimodoPolicyConfig = None):
        super().__init__()
        self.config = config or KimodoPolicyConfig()
        config = self.config
        self.fps = config.fps
        checkpoint_path = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "checkpoints")
        )
        model_dir = os.path.join(checkpoint_path, "Kimodo-G1-RP-v1")
        image_checkpoint = config.dinov3_checkpoint or os.path.join(
            checkpoint_path, config.dinov3_model_name
        )
        if config.dinov3_checkpoint is None and not os.path.isdir(image_checkpoint):
            raise FileNotFoundError(
                f"DINOv3 checkpoint not found at {image_checkpoint}. Set model.dinov3_checkpoint "
                "to a local Hugging Face checkpoint directory or model id."
            )
        # 1. G1 机器人骨骼结构
        self.g1_skeleton_34 = G1Skeleton34()
        # 2. motion 转化器
        self.representation = KimodoMotionRep(skeleton=self.g1_skeleton_34, fps=self.fps, stats_path=os.path.join(model_dir, "stats/motion"))
        # 3. 去噪网络
        self.denoiser = TwostageDenoiser(motion_rep=self.representation, motion_mask_mode=config.motion_mask_mode, ckpt_path=model_dir)
        # 4. 去噪器
        self.diffusion = Diffusion(num_base_steps=OmegaConf.load(os.path.join(model_dir, "config.yaml")).num_base_steps)
        self.sampler = DDIMSampler(self.diffusion)
        # 5. text encoder & 冻结。训练可直接使用预计算 embedding，避免 8B LLM 常驻显存。
        self.text_encoder = None
        if config.load_text_encoder:
            from .modules.llm2vec.llm2vec_wreapper import LLM2VecEncoder

            self.text_encoder = LLM2VecEncoder(checkpoint_path=checkpoint_path)
            for p in self.text_encoder.model.parameters():
                p.requires_grad = False
        # 6. 冻结 Kimodo 主体并训练 ControlNet
        for p in self.denoiser.parameters():
            p.requires_grad = False
        self.image_encoder = DINOv3Encoder(checkpoint_path=image_checkpoint)
        for p in self.image_encoder.model.parameters():
            p.requires_grad = False
        self.controlnet = ControlNet(
            self.denoiser,
            image_feat_dim=self.image_encoder.output_dim,
            motion_token_count=config.action_history + config.action_chunk,
            num_control_layers=config.controlnet_num_layers,
            future_token_count=config.action_chunk,
        )
        for p in self.controlnet.parameters():
            p.requires_grad = True
        self.hand_head = None
        if config.enable_hand_head:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config.hand_init_seed)
                self.hand_head = HandTokenDiffusionDenoiser(
                    input_dim=self.denoiser.body_model.latent_dim,
                    hidden_dim=config.hand_hidden_dim,
                    future_token_count=config.action_chunk,
                    image_token_count=self.controlnet.image_token_count,
                    num_layers=config.hand_num_layers,
                    num_heads=config.hand_num_heads,
                    ffn_dim=config.hand_ffn_dim,
                    max_diffusion_steps=self.diffusion.num_base_steps,
                )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def train(self, mode: bool = True):
        super().train(mode)
        self.denoiser.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        self.image_encoder.eval()
        return self

    @staticmethod
    def _length_mask(lengths, max_length: int, device: torch.device) -> torch.Tensor:
        lengths = torch.as_tensor(lengths, device=device)
        return torch.arange(max_length, device=device).unsqueeze(0) < lengths.unsqueeze(1)

    @staticmethod
    def _first_history_heading(
        history_motion: torch.Tensor,
        history_mask: torch.Tensor,
        heading_slice: slice,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if history_motion.shape[:2] != history_mask.shape:
            raise ValueError(
                "history_motion and history_mask must share batch and time dimensions"
            )
        first_valid_index = history_mask.long().argmax(dim=1)
        heading_cos_sin = history_motion[
            torch.arange(history_motion.shape[0], device=history_motion.device),
            first_valid_index,
            heading_slice,
        ].float()
        first_heading = torch.atan2(heading_cos_sin[:, 1], heading_cos_sin[:, 0])
        has_history = history_mask.any(dim=1)
        first_heading = torch.where(
            has_history, first_heading, torch.zeros_like(first_heading)
        )
        if dtype is not None:
            first_heading = first_heading.to(dtype=dtype)
        return first_heading

    @staticmethod
    def _last_valid_hand_state(
        hand_history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        current_hand, _ = KimodoPolicy._last_valid_hand_state_and_mask(
            hand_history, history_mask
        )
        return current_hand

    @staticmethod
    def _last_valid_hand_state_and_mask(
        hand_history: torch.Tensor,
        history_mask: torch.Tensor,
        hand_valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hand_history.ndim != 3 or hand_history.shape[-1] != 2:
            raise ValueError(
                f"Expected hand_history shape [B, T, 2], got {tuple(hand_history.shape)}"
            )
        if hand_history.shape[:2] != history_mask.shape:
            raise ValueError(
                "hand_history and history_mask must share batch and time dimensions"
            )
        batch_size, history_length, _ = hand_history.shape
        if hand_valid_mask is None:
            hand_valid_mask = history_mask.unsqueeze(-1).expand_as(hand_history)
        else:
            hand_valid_mask = torch.as_tensor(
                hand_valid_mask, device=hand_history.device, dtype=torch.bool
            )
            if hand_valid_mask.shape != hand_history.shape:
                raise ValueError(
                    f"Expected hand_valid_mask shape {tuple(hand_history.shape)}, "
                    f"got {tuple(hand_valid_mask.shape)}"
                )
            hand_valid_mask = hand_valid_mask & history_mask.unsqueeze(-1)
        if history_length == 0:
            return (
                torch.zeros(
                    batch_size,
                    2,
                    device=hand_history.device,
                    dtype=hand_history.dtype,
                ),
                torch.zeros(batch_size, 2, device=hand_history.device, dtype=torch.bool),
            )
        time_indices = torch.arange(history_length, device=hand_history.device)
        last_indices = torch.where(
            hand_valid_mask,
            time_indices.reshape(1, -1, 1),
            torch.full(
                (1, history_length, 1),
                -1,
                device=hand_history.device,
                dtype=time_indices.dtype,
            ),
        ).max(dim=1).values
        safe_indices = last_indices.clamp_min(0)
        batch_indices = torch.arange(batch_size, device=hand_history.device).unsqueeze(-1)
        hand_indices = torch.arange(2, device=hand_history.device).unsqueeze(0)
        current_hand = hand_history[batch_indices, safe_indices, hand_indices]
        current_valid = last_indices >= 0
        return (
            torch.where(current_valid, current_hand, torch.zeros_like(current_hand)),
            current_valid,
        )

    @staticmethod
    def _rtc_temporal_weights(
        overlap_frames: int,
        frozen_frames: int,
        ramp_power: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return old-trajectory influence for DDIM real-time chunking.

        Frozen frames keep the previous plan exactly.  The remaining overlap
        uses a cosine decay whose endpoints are deliberately excluded: the
        first soft frame is close to the old plan, while the final soft frame
        still provides a small continuity prior without becoming another hard
        constraint.
        """
        overlap_frames = int(overlap_frames)
        frozen_frames = int(frozen_frames)
        ramp_power = float(ramp_power)
        if overlap_frames <= 0:
            raise ValueError("rtc overlap_frames must be positive")
        if not 0 <= frozen_frames <= overlap_frames:
            raise ValueError(
                "rtc frozen_frames must be between 0 and overlap_frames, got "
                f"{frozen_frames} and {overlap_frames}"
            )
        if not math.isfinite(ramp_power) or ramp_power <= 0:
            raise ValueError(f"rtc ramp_power must be finite and positive, got {ramp_power}")

        weights = torch.ones(overlap_frames, device=device, dtype=torch.float32)
        soft_frames = overlap_frames - frozen_frames
        if soft_frames:
            progress = torch.arange(
                1, soft_frames + 1, device=device, dtype=torch.float32
            ) / float(soft_frames + 1)
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.pi * progress))
            weights[frozen_frames:] = cosine_decay.pow(ramp_power)
        return weights.to(dtype=dtype)

    @staticmethod
    def _batched_rtc_reference(
        reference: torch.Tensor,
        *,
        batch_size: int,
        feature_dim: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        reference = torch.as_tensor(reference, device=device)
        if reference.ndim == 2:
            reference = reference.unsqueeze(0)
        if reference.ndim != 3 or tuple(reference.shape[::2]) != (
            batch_size,
            feature_dim,
        ):
            raise ValueError(
                f"Expected {name} shape [B, T, {feature_dim}] with B={batch_size}, "
                f"got {tuple(reference.shape)}"
            )
        if reference.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one frame")
        if not torch.is_floating_point(reference):
            reference = reference.float()
        if not torch.isfinite(reference).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return reference

    def forward(
        self,
        instruction: List[str],
        egoview: torch.Tensor,
        gt_motion: torch.Tensor,
        gt_mask: torch.Tensor,
        condition_motion: Optional[torch.Tensor] = None,
        condition_motion_mask: Optional[torch.Tensor] = None,
        gt_hand: Optional[torch.Tensor] = None,
        gt_hand_mask: Optional[torch.Tensor] = None,
        text_feat: Optional[torch.Tensor] = None,
        text_length: Optional[torch.Tensor] = None,
    ):
        return self.training_kimodo_policy_controlnet(
            instruction=instruction,
            egoview=egoview,
            gt_motion=gt_motion,
            gt_mask=gt_mask,
            condition_motion=condition_motion,
            condition_motion_mask=condition_motion_mask,
            gt_hand=gt_hand,
            gt_hand_mask=gt_hand_mask,
            text_feat=text_feat,
            text_length=text_length,
        )

    def training_kimodo_policy_controlnet(
        self,
        instruction: List[str],
        egoview: torch.Tensor, # [B, 3, 480, 640]
        gt_motion: torch.Tensor, # [B, T, 417], T = action_history + action_chunk
        gt_mask: torch.Tensor,   # [B, T] bool, T = action_history + action_chunk
        condition_motion: Optional[torch.Tensor] = None,
        condition_motion_mask: Optional[torch.Tensor] = None,
        gt_hand: Optional[torch.Tensor] = None, # [B, T, 2], binary open/closed state
        gt_hand_mask: Optional[torch.Tensor] = None, # [B, T, 2], valid hand supervision
        text_feat: Optional[torch.Tensor] = None,
        text_length: Optional[torch.Tensor] = None,
    ):
        # 0. 基本配置
        B, T, _ = gt_motion.shape
        device = gt_motion.device
        H = self.config.action_history  # 100
        C = self.config.action_chunk    # 50
        if T != H + C:
            raise ValueError(f"Expected {H + C} motion frames, got {T}")
        if not gt_mask[:, H:].any(dim=1).all():
            raise ValueError("Every sample must contain at least one valid future frame")
        if condition_motion is None:
            condition_motion = gt_motion
        condition_motion = torch.as_tensor(
            condition_motion, device=device, dtype=gt_motion.dtype
        )
        if tuple(condition_motion.shape) != tuple(gt_motion.shape):
            raise ValueError(
                f"Expected condition_motion shape {tuple(gt_motion.shape)}, got "
                f"{tuple(condition_motion.shape)}"
            )
        if condition_motion_mask is None:
            condition_motion_mask = gt_mask.unsqueeze(-1).expand_as(gt_motion)
        else:
            condition_motion_mask = torch.as_tensor(
                condition_motion_mask, device=device, dtype=torch.bool
            )
            if tuple(condition_motion_mask.shape) != tuple(gt_motion.shape):
                raise ValueError(
                    "Expected condition_motion_mask shape "
                    f"{tuple(gt_motion.shape)}, got "
                    f"{tuple(condition_motion_mask.shape)}"
                )
            condition_motion_mask = (
                condition_motion_mask & gt_mask.unsqueeze(-1)
            )
        valid_target = gt_motion.masked_select(gt_mask.unsqueeze(-1))
        valid_condition = condition_motion.masked_select(condition_motion_mask)
        if not torch.isfinite(valid_target).all():
            raise ValueError("gt_motion contains NaN or Inf in valid frames")
        if not torch.isfinite(valid_condition).all():
            raise ValueError("condition_motion contains NaN or Inf in valid features")
        if self.hand_head is not None:
            if gt_hand is None:
                raise ValueError("gt_hand is required when enable_hand_head=True")
            gt_hand = torch.as_tensor(gt_hand, device=device, dtype=gt_motion.dtype)
            if tuple(gt_hand.shape) != (B, T, 2):
                raise ValueError(f"Expected gt_hand shape {(B, T, 2)}, got {tuple(gt_hand.shape)}")
            if gt_hand_mask is None:
                gt_hand_mask = gt_mask.unsqueeze(-1).expand_as(gt_hand)
            else:
                gt_hand_mask = torch.as_tensor(
                    gt_hand_mask, device=device, dtype=torch.bool
                )
                if tuple(gt_hand_mask.shape) != (B, T, 2):
                    raise ValueError(
                        f"Expected gt_hand_mask shape {(B, T, 2)}, "
                        f"got {tuple(gt_hand_mask.shape)}"
                    )
                gt_hand_mask = gt_hand_mask & gt_mask.unsqueeze(-1)
            valid_hand = gt_hand.masked_select(gt_hand_mask)
            if not torch.isfinite(valid_hand).all():
                raise ValueError("gt_hand contains NaN or Inf")
            if not torch.logical_or(valid_hand == 0, valid_hand == 1).all():
                raise ValueError("gt_hand must contain only binary 0/1 values")
        # 1. 标准化完整 motion
        x_start_full = self.representation.normalize(gt_motion)  # [B, T, 417]
        normalized_condition = self.representation.normalize(condition_motion)
        # 3. 时间步采样
        t = torch.randint(0, self.diffusion.num_base_steps, (B,), device=device)
        # 4. 完整序列加噪；历史帧由 Kimodo motion constraint 硬覆盖
        x_t = self.diffusion.q_sample(x_start_full, t, torch.randn_like(x_start_full))
        history_mask = gt_mask[:, :H]   # [B, H]
        x_pad_mask_full = gt_mask.clone()  # history + chunk 都按 gt_mask 有效性参与 attention
        # 5. 组织 observed_motion 和 motion_mask（前 H 帧有约束，后 C 帧无约束）
        observed_motion = torch.zeros_like(x_start_full)
        observed_motion[:, :H] = normalized_condition[:, :H]
        motion_mask = torch.zeros_like(x_start_full)
        motion_mask[:, :H] = condition_motion_mask[:, :H].to(
            dtype=x_start_full.dtype
        )
        observed_motion = observed_motion.masked_fill(motion_mask == 0, 0)
        # 6. 文本编码。训练优先使用 Dataset 返回的预计算 embedding。
        if text_feat is None:
            if self.text_encoder is None:
                raise RuntimeError("Text encoder is disabled; provide precomputed text_feat")
            text_feat, text_length = self.text_encoder(instruction)
        text_feat = text_feat.to(device=device, dtype=x_start_full.dtype)
        if text_length is None:
            text_length = torch.full(
                (B,), text_feat.shape[1], dtype=torch.long, device=device
            )
        maxlen = text_feat.shape[1]
        text_pad_mask = self._length_mask(text_length, maxlen, device)
        # 7. 图像编码 (DINOv3 冻结)
        with torch.no_grad():
            image_feat = self.image_encoder(egoview)
        image_feat = image_feat.to(dtype=next(self.controlnet.parameters()).dtype)
        # 8. ControlNet 前向（可训练）
        root_control_tokens, body_control_tokens = self.controlnet(t, image_feat, T, H)
        # 9. 冻结的 denoiser 前向。每个注入层用当前 future motion hidden
        # 作为 Query 读取对应的 visual tokens，梯度回传到 ControlNet。
        heading_slice = self.representation.slice_dict["global_root_heading"]
        heading_history_mask = (
            history_mask
            & condition_motion_mask[:, :H, heading_slice].all(dim=-1)
        )
        first_heading_angle = self._first_history_heading(
            condition_motion[:, :H],
            heading_history_mask,
            heading_slice,
            dtype=x_start_full.dtype,
        )
        denoiser_output = self.denoiser(
            x=x_t,
            x_pad_mask=x_pad_mask_full,
            text_feat=text_feat,
            text_feat_pad_mask=text_pad_mask,
            timesteps=t,
            first_heading_angle=first_heading_angle,
            observed_motion=observed_motion,
            motion_mask=motion_mask,
            root_control_tokens=root_control_tokens,
            body_control_tokens=body_control_tokens,
            root_hint_fuser=self.controlnet.root_hint_fusion,
            body_hint_fuser=self.controlnet.body_hint_fusion,
            control_future_start=H,
            detach_root_for_body=True,
            return_body_hidden=self.hand_head is not None,
        )
        if self.hand_head is not None:
            pred_clean, body_hidden = denoiser_output
        else:
            pred_clean = denoiser_output
        # 10. 计算 loss（只对 chunk 的有效帧）
        pred_chunk = pred_clean[:, H:]
        x_start_chunk = x_start_full[:, H:]
        chunk_valid = gt_mask[:, H:]
        root_slice = self.representation.root_slice
        body_slice = self.representation.body_slice
        root_valid = chunk_valid.unsqueeze(-1).expand_as(pred_chunk[..., root_slice])
        body_valid = chunk_valid.unsqueeze(-1).expand_as(pred_chunk[..., body_slice])
        root_loss = nn.functional.mse_loss(
            pred_chunk[..., root_slice], x_start_chunk[..., root_slice], reduction="none"
        ).masked_select(root_valid).mean()
        body_loss = nn.functional.mse_loss(
            pred_chunk[..., body_slice], x_start_chunk[..., body_slice], reduction="none"
        ).masked_select(body_valid).mean()
        motion_loss = (
            self.config.root_loss_weight * root_loss
            + self.config.body_loss_weight * body_loss
        )
        loss = motion_loss
        output = {
            "loss": loss,
            "motion_loss": motion_loss,
            "root_loss": root_loss,
            "body_loss": body_loss,
        }
        if self.hand_head is not None:
            current_hand, current_hand_valid = self._last_valid_hand_state_and_mask(
                gt_hand[:, :H], history_mask, gt_hand_mask[:, :H]
            )
            current_hand = current_hand * 2.0 - 1.0
            clean_hand = gt_hand[:, H:] * 2.0 - 1.0
            noisy_hand = self.diffusion.q_sample(
                clean_hand,
                t,
                torch.randn_like(clean_hand),
            )
            final_visual_tokens = body_control_tokens[
                max(self.controlnet.body_injection_layers)
            ]
            predicted_clean_hand = self.hand_head(
                visual_tokens=final_visual_tokens,
                body_future_tokens=body_hidden[:, H:],
                noisy_future_hand=noisy_hand,
                current_hand_state=current_hand,
                timesteps=t,
                future_mask=chunk_valid,
            )
            hand_valid = gt_hand_mask[:, H:] & chunk_valid.unsqueeze(-1)
            hand_state_error = nn.functional.mse_loss(
                predicted_clean_hand.float(),
                clean_hand.float(),
                reduction="none",
            )
            hand_state_loss = (
                hand_state_error * hand_valid.float()
            ).sum() / hand_valid.sum().clamp_min(1)
            predicted_sequence = torch.cat(
                (current_hand.unsqueeze(1), predicted_clean_hand), dim=1
            )
            target_sequence = torch.cat(
                (current_hand.unsqueeze(1), clean_hand), dim=1
            )
            hand_transition_error = nn.functional.l1_loss(
                predicted_sequence[:, 1:].float() - predicted_sequence[:, :-1].float(),
                target_sequence[:, 1:].float() - target_sequence[:, :-1].float(),
                reduction="none",
            )
            previous_valid = torch.cat(
                (current_hand_valid.unsqueeze(1), hand_valid[:, :-1]), dim=1
            )
            transition_valid = hand_valid & previous_valid
            hand_transition_loss = (
                hand_transition_error * transition_valid.float()
            ).sum() / transition_valid.sum().clamp_min(1)
            hand_loss = (
                hand_state_loss
                + self.config.hand_transition_loss_weight * hand_transition_loss
            )
            loss = motion_loss + self.config.hand_loss_weight * hand_loss
            output.update(
                {
                    "loss": loss,
                    "hand_loss": hand_loss,
                    "hand_state_loss": hand_state_loss,
                    "hand_transition_loss": hand_transition_loss,
                }
            )
        return output

    @torch.inference_mode()
    def predict_future(
        self,
        instruction: List[str] | str,
        egoview: torch.Tensor,
        history_motion: torch.Tensor,
        diffusion_steps: int = 50,
        squeeze_batch: bool = True,
        history_mask: Optional[torch.Tensor] = None,
        history_feature_mask: Optional[torch.Tensor] = None,
        text_feat: Optional[torch.Tensor] = None,
        text_length: Optional[torch.Tensor] = None,
        hand_history: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        rtc_motion_reference: Optional[torch.Tensor] = None,
        rtc_hand_reference: Optional[torch.Tensor] = None,
        rtc_overlap_frames: int = 0,
        rtc_frozen_frames: int = 0,
        rtc_ramp_power: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if isinstance(instruction, str):
            instruction = [instruction]
        if history_motion.ndim == 2:
            history_motion = history_motion.unsqueeze(0)
        batch_size, provided_history_length, motion_dim = history_motion.shape
        denoiser_dtype = next(self.denoiser.parameters()).dtype
        history_motion = history_motion.to(device=self.device)
        if not torch.is_floating_point(history_motion):
            history_motion = history_motion.float()
        if egoview.ndim == 3:
            egoview = egoview.unsqueeze(0)
        if egoview.ndim != 4 or egoview.shape[0] != batch_size:
            raise ValueError(
                f"Expected egoview shape [B, C, H, W] with B={batch_size}, "
                f"got {tuple(egoview.shape)}"
            )
        egoview = egoview.to(self.device)
        history_length = self.config.action_history
        if provided_history_length > history_length:
            raise ValueError(
                f"Expected at most {history_length} history frames, got {provided_history_length}"
            )
        if motion_dim != self.representation.motion_rep_dim:
            raise ValueError(f"Expected motion dim {self.representation.motion_rep_dim}, got {motion_dim}")
        rtc_motion_clean = None
        rtc_hand_clean = None
        rtc_motion_weights = None
        rtc_hand_weights = None
        effective_rtc_overlap = 0
        if rtc_motion_reference is None:
            if rtc_hand_reference is not None:
                raise ValueError("rtc_hand_reference requires rtc_motion_reference")
        else:
            requested_overlap = int(rtc_overlap_frames)
            requested_frozen = int(rtc_frozen_frames)
            if requested_overlap <= 0:
                raise ValueError(
                    "rtc_overlap_frames must be positive when rtc_motion_reference is provided"
                )
            if not 0 <= requested_frozen <= requested_overlap:
                raise ValueError(
                    "rtc_frozen_frames must be between 0 and rtc_overlap_frames"
                )
            rtc_motion_reference = self._batched_rtc_reference(
                rtc_motion_reference,
                batch_size=batch_size,
                feature_dim=motion_dim,
                device=self.device,
                name="rtc_motion_reference",
            )
            effective_rtc_overlap = min(
                requested_overlap,
                int(rtc_motion_reference.shape[1]),
                int(self.config.action_chunk),
            )
            effective_frozen = min(requested_frozen, effective_rtc_overlap)
            rtc_time_weights = self._rtc_temporal_weights(
                effective_rtc_overlap,
                effective_frozen,
                rtc_ramp_power,
                device=self.device,
                dtype=denoiser_dtype,
            )
            rtc_motion_clean = self.representation.normalize(
                rtc_motion_reference[:, :effective_rtc_overlap].float()
            ).to(dtype=denoiser_dtype)
            rtc_motion_weights = rtc_time_weights.reshape(1, -1, 1).expand(
                batch_size, effective_rtc_overlap, motion_dim
            ).clone()

            # Each training window is translated independently.  Arena state
            # does not observe root x/z, so those two features cannot be moved
            # reliably from the previous window's planar gauge into the new
            # one.  All heading, rotation, body, velocity and contact features
            # remain compatible and retain RTC continuity.
            smooth_root_slice = self.representation.slice_dict.get("smooth_root_pos")
            if smooth_root_slice is not None:
                smooth_root_start = 0 if smooth_root_slice.start is None else smooth_root_slice.start
                smooth_root_stop = motion_dim if smooth_root_slice.stop is None else smooth_root_slice.stop
                if smooth_root_stop - smooth_root_start < 3:
                    raise ValueError("smooth_root_pos must contain x, y and z features")
                rtc_motion_weights[..., smooth_root_start] = 0
                rtc_motion_weights[..., smooth_root_start + 2] = 0

            if rtc_hand_reference is not None:
                if self.hand_head is None:
                    raise ValueError("rtc_hand_reference requires an enabled hand head")
                rtc_hand_reference = self._batched_rtc_reference(
                    rtc_hand_reference,
                    batch_size=batch_size,
                    feature_dim=2,
                    device=self.device,
                    name="rtc_hand_reference",
                )
                if rtc_hand_reference.shape[1] < effective_rtc_overlap:
                    raise ValueError(
                        "rtc_hand_reference is shorter than the effective motion overlap"
                    )
                if (rtc_hand_reference.abs() > 1.0001).any():
                    raise ValueError("rtc_hand_reference must be in the clean [-1, 1] space")
                rtc_hand_clean = rtc_hand_reference[
                    :, :effective_rtc_overlap
                ].clamp(-1.0, 1.0).to(dtype=denoiser_dtype)
                rtc_hand_weights = rtc_time_weights.reshape(1, -1, 1).expand_as(
                    rtc_hand_clean
                )
        if history_mask is None:
            history_mask = torch.ones(
                batch_size, provided_history_length, dtype=torch.bool, device=self.device
            )
        else:
            history_mask = torch.as_tensor(history_mask, device=self.device, dtype=torch.bool)
            if history_mask.shape != (batch_size, provided_history_length):
                raise ValueError(
                    f"Expected history_mask shape {(batch_size, provided_history_length)}, "
                    f"got {tuple(history_mask.shape)}"
                )
        if history_feature_mask is None:
            history_feature_mask = history_mask.unsqueeze(-1).expand(
                batch_size, provided_history_length, motion_dim
            )
        else:
            history_feature_mask = torch.as_tensor(
                history_feature_mask, device=self.device, dtype=torch.bool
            )
            if history_feature_mask.ndim == 2 and batch_size == 1:
                history_feature_mask = history_feature_mask.unsqueeze(0)
            if tuple(history_feature_mask.shape) != (
                batch_size,
                provided_history_length,
                motion_dim,
            ):
                raise ValueError(
                    "Expected history_feature_mask shape "
                    f"{(batch_size, provided_history_length, motion_dim)}, got "
                    f"{tuple(history_feature_mask.shape)}"
                )
            history_feature_mask = (
                history_feature_mask & history_mask.unsqueeze(-1)
            )
        current_hand_binary = None
        if self.hand_head is not None:
            if hand_history is None:
                hand_history = torch.zeros(
                    batch_size,
                    provided_history_length,
                    2,
                    device=self.device,
                    dtype=history_motion.dtype,
                )
            else:
                hand_history = torch.as_tensor(
                    hand_history, device=self.device, dtype=history_motion.dtype
                )
                if hand_history.ndim == 2:
                    hand_history = hand_history.unsqueeze(0)
                if tuple(hand_history.shape) != (
                    batch_size,
                    provided_history_length,
                    2,
                ):
                    raise ValueError(
                        "Expected hand_history shape "
                        f"{(batch_size, provided_history_length, 2)}, got "
                        f"{tuple(hand_history.shape)}"
                    )
            current_hand_binary = self._last_valid_hand_state(
                hand_history, history_mask
            )
            if not torch.isfinite(current_hand_binary).all():
                raise ValueError("hand_history contains NaN or Inf")
            if not torch.logical_or(
                current_hand_binary == 0, current_hand_binary == 1
            ).all():
                raise ValueError("hand_history must contain only binary 0/1 values")
        if provided_history_length < history_length:
            left_padding = history_length - provided_history_length
            padded_history = torch.zeros(
                batch_size, history_length, motion_dim,
                device=self.device, dtype=history_motion.dtype,
            )
            padded_history[:, left_padding:] = history_motion
            padded_mask = torch.zeros(
                batch_size, history_length, dtype=torch.bool, device=self.device
            )
            padded_mask[:, left_padding:] = history_mask
            padded_feature_mask = torch.zeros(
                batch_size,
                history_length,
                motion_dim,
                dtype=torch.bool,
                device=self.device,
            )
            padded_feature_mask[:, left_padding:] = history_feature_mask
            history_motion = padded_history
            history_mask = padded_mask
            history_feature_mask = padded_feature_mask
        total_length = history_length + self.config.action_chunk
        normalized_history = self.representation.normalize(history_motion.float()).to(
            dtype=denoiser_dtype
        )
        normalized_history = normalized_history.masked_fill(
            ~history_feature_mask, 0
        )
        observed_motion = torch.zeros(
            batch_size, total_length, motion_dim, device=self.device, dtype=normalized_history.dtype
        )
        observed_motion[:, :history_length] = normalized_history
        motion_mask = torch.zeros_like(observed_motion)
        motion_mask[:, :history_length] = history_feature_mask.to(
            dtype=motion_mask.dtype
        )
        motion_pad_mask = torch.ones(
            batch_size, total_length, dtype=torch.bool, device=self.device
        )
        motion_pad_mask[:, :history_length] = history_mask

        if text_feat is None:
            if self.text_encoder is None:
                raise RuntimeError(
                    "predict_future requires load_text_encoder=True or precomputed text_feat"
                )
            if len(instruction) == 1 and batch_size > 1:
                instruction = instruction * batch_size
            elif len(instruction) != batch_size:
                raise ValueError(
                    f"Expected {batch_size} instruction(s), got {len(instruction)}"
                )
            text_features, text_lengths = self.text_encoder(instruction)
        else:
            text_features = torch.as_tensor(text_feat)
            if text_features.ndim == 2:
                text_features = text_features.unsqueeze(0)
            if text_features.ndim != 3 or text_features.shape[0] != batch_size:
                raise ValueError(
                    f"Expected text_feat shape [B, L, D] with B={batch_size}, "
                    f"got {tuple(text_features.shape)}"
                )
            if text_length is None:
                text_lengths = torch.full(
                    (batch_size,), text_features.shape[1], dtype=torch.long
                )
            else:
                text_lengths = torch.as_tensor(text_length, dtype=torch.long).reshape(-1)
                if text_lengths.shape != (batch_size,):
                    raise ValueError(
                        f"Expected text_length shape {(batch_size,)}, got {tuple(text_lengths.shape)}"
                    )
        text_features = text_features.to(device=self.device, dtype=denoiser_dtype)
        text_pad_mask = self._length_mask(text_lengths, text_features.shape[1], self.device)
        image_features = self.image_encoder(egoview).to(
            dtype=next(self.controlnet.parameters()).dtype
        )
        heading_slice = self.representation.slice_dict["global_root_heading"]
        heading_history_mask = (
            history_mask
            & history_feature_mask[..., heading_slice].all(dim=-1)
        )
        first_heading = self._first_history_heading(
            history_motion,
            heading_history_mask,
            heading_slice,
            dtype=denoiser_dtype,
        )

        use_timesteps, timestep_map = self.diffusion.space_timesteps(diffusion_steps)
        self.diffusion.calc_diffusion_vars(use_timesteps)
        current_motion = torch.randn(
            observed_motion.shape,
            device=observed_motion.device,
            dtype=observed_motion.dtype,
            generator=generator,
        )
        noisy_hand = None
        current_hand = None
        hand_future_mask = None
        if self.hand_head is not None:
            current_hand = current_hand_binary.to(dtype=denoiser_dtype) * 2.0 - 1.0
            noisy_hand = torch.randn(
                batch_size,
                self.config.action_chunk,
                2,
                device=self.device,
                dtype=denoiser_dtype,
                generator=generator,
            )
            hand_future_mask = torch.ones(
                batch_size,
                self.config.action_chunk,
                device=self.device,
                dtype=torch.bool,
            )
        rtc_motion_noise = None
        rtc_hand_noise = None
        if rtc_motion_clean is not None:
            rtc_motion_noise = torch.randn(
                rtc_motion_clean.shape,
                device=self.device,
                dtype=denoiser_dtype,
                generator=generator,
            )
        if rtc_hand_clean is not None:
            rtc_hand_noise = torch.randn(
                rtc_hand_clean.shape,
                device=self.device,
                dtype=denoiser_dtype,
                generator=generator,
            )
        for step in range(diffusion_steps - 1, -1, -1):
            current_history = current_motion[:, :history_length]
            current_motion[:, :history_length] = torch.where(
                history_feature_mask,
                normalized_history,
                current_history,
            )
            sampler_timestep = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            model_timestep = timestep_map[sampler_timestep]
            if rtc_motion_clean is not None:
                rtc_motion_at_t = self.diffusion.q_sample(
                    rtc_motion_clean, model_timestep, rtc_motion_noise
                )
                future_overlap = current_motion[
                    :, history_length : history_length + effective_rtc_overlap
                ]
                current_motion[
                    :, history_length : history_length + effective_rtc_overlap
                ] = torch.lerp(future_overlap, rtc_motion_at_t, rtc_motion_weights)
            if rtc_hand_clean is not None:
                rtc_hand_at_t = self.diffusion.q_sample(
                    rtc_hand_clean, model_timestep, rtc_hand_noise
                )
                noisy_hand[:, :effective_rtc_overlap] = torch.lerp(
                    noisy_hand[:, :effective_rtc_overlap],
                    rtc_hand_at_t,
                    rtc_hand_weights,
                )
            root_control_tokens, body_control_tokens = self.controlnet(
                model_timestep, image_features, total_length, history_length
            )
            denoiser_output = self.denoiser(
                x=current_motion,
                x_pad_mask=motion_pad_mask,
                text_feat=text_features,
                text_feat_pad_mask=text_pad_mask,
                timesteps=model_timestep,
                first_heading_angle=first_heading,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                root_control_tokens=root_control_tokens,
                body_control_tokens=body_control_tokens,
                root_hint_fuser=self.controlnet.root_hint_fusion,
                body_hint_fuser=self.controlnet.body_hint_fusion,
                control_future_start=history_length,
                detach_root_for_body=False,
                return_body_hidden=self.hand_head is not None,
            )
            if self.hand_head is not None:
                predicted_clean, body_hidden = denoiser_output
                body_visual_tokens = body_control_tokens[
                    max(self.controlnet.body_injection_layers)
                ]
                predicted_clean_hand = self.hand_head(
                    visual_tokens=body_visual_tokens,
                    body_future_tokens=body_hidden[:, history_length:],
                    noisy_future_hand=noisy_hand,
                    current_hand_state=current_hand,
                    timesteps=model_timestep,
                    future_mask=hand_future_mask,
                )
            else:
                predicted_clean = denoiser_output
            current_motion = self.sampler(use_timesteps, current_motion, predicted_clean, sampler_timestep)
            if self.hand_head is not None:
                noisy_hand = self.sampler(
                    use_timesteps,
                    noisy_hand,
                    predicted_clean_hand,
                    sampler_timestep,
                )

        # The final DDIM step returns clean x0.  Apply the same soft overlap one
        # last time in clean space; this makes frozen frames exact and keeps the
        # soft region consistent with the trajectory seen throughout denoising.
        if rtc_motion_clean is not None:
            future_overlap = current_motion[
                :, history_length : history_length + effective_rtc_overlap
            ]
            current_motion[
                :, history_length : history_length + effective_rtc_overlap
            ] = torch.lerp(future_overlap, rtc_motion_clean, rtc_motion_weights)
        if rtc_hand_clean is not None:
            noisy_hand[:, :effective_rtc_overlap] = torch.lerp(
                noisy_hand[:, :effective_rtc_overlap],
                rtc_hand_clean,
                rtc_hand_weights,
            )

        decoded_motion = current_motion.float()
        decoded = self.representation.inverse(
            decoded_motion,
            is_normalized=True,
            return_numpy=False,
        )
        # The future root is expressed in the generated window's local planar
        # gauge. Preserve its same-window boundary so action conversion never
        # subtracts an endpoint from a previous window's gauge. At startup the
        # entire history is padding, so anchor to the first future root itself;
        # this matches cut=0 training windows whose first planar delta is zero.
        decoded_root_positions = decoded["root_positions"]
        history_indices = torch.arange(history_length, device=self.device)
        last_valid_history_index = torch.where(
            history_mask,
            history_indices.unsqueeze(0),
            torch.full(
                (batch_size, history_length),
                -1,
                device=self.device,
                dtype=history_indices.dtype,
            ),
        ).max(dim=1).values
        boundary_indices = torch.where(
            last_valid_history_index >= 0,
            last_valid_history_index,
            torch.full_like(last_valid_history_index, history_length),
        )
        history_last_root_position = decoded_root_positions[
            torch.arange(batch_size, device=self.device), boundary_indices
        ].clone()
        future_motion_features = self.representation.unnormalize(decoded_motion)[
            :, history_length:
        ]
        output = {
            key: value[:, history_length:] if torch.is_tensor(value) and value.ndim >= 2 else value
            for key, value in decoded.items()
        }
        output["motion_features"] = future_motion_features
        output["history_last_root_position"] = history_last_root_position
        if self.hand_head is not None:
            hand_clean = noisy_hand.clamp(-1.0, 1.0)
            output["hand_clean"] = hand_clean
            output["hand_probability"] = ((hand_clean + 1.0) * 0.5).clamp(0.0, 1.0)
            output["hand_binary"] = (hand_clean >= 0).to(dtype=hand_clean.dtype)
        if squeeze_batch and batch_size == 1:
            output = {
                key: value[0] if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1 else value
                for key, value in output.items()
            }
        return output

    def load_controlnet_checkpoint(self, checkpoint_path: str, strict: bool = True) -> int:
        checkpoint_path = os.path.expanduser(checkpoint_path)
        state_path = (
            os.path.join(checkpoint_path, "training_state.pt")
            if os.path.isdir(checkpoint_path)
            else checkpoint_path
        )
        payload = torch.load(state_path, map_location="cpu", weights_only=True)
        state_dict = payload.get("model", payload)
        controlnet_state = {}
        hand_state = {}
        for name, value in state_dict.items():
            normalized_name = name.removeprefix("_orig_mod.")
            if normalized_name.startswith("controlnet."):
                controlnet_state[normalized_name.removeprefix("controlnet.")] = value
            elif normalized_name.startswith("hand_head."):
                hand_state[normalized_name.removeprefix("hand_head.")] = value
        if not controlnet_state:
            raise RuntimeError(f"No ControlNet weights found in {state_path}")
        incompatible = self.controlnet.load_state_dict(controlnet_state, strict=False)
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise RuntimeError(
                "ControlNet checkpoint mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        hand_head = getattr(self, "hand_head", None)
        if hand_head is not None:
            if not hand_state:
                if strict:
                    raise RuntimeError(f"No hand-head weights found in {state_path}")
            else:
                hand_incompatible = hand_head.load_state_dict(hand_state, strict=False)
                if strict and (
                    hand_incompatible.missing_keys or hand_incompatible.unexpected_keys
                ):
                    raise RuntimeError(
                        "Hand-head checkpoint mismatch: "
                        f"missing={hand_incompatible.missing_keys}, "
                        f"unexpected={hand_incompatible.unexpected_keys}"
                    )
        return int(payload.get("global_step", -1))
