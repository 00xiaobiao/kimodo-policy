import argparse
import base64
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if os.environ.get("KIMODO_DETERMINISTIC_EVAL", "0").strip() == "1":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.kimodo_policy import KimodoPolicy, KimodoPolicyConfig
from motion.g1_reference import (
    CANONICAL_G1_JOINT_NAMES_29,
    HumanoidArenaActionDecoder,
    resample_hand_binary_chunk,
    resample_motion,
    resample_motion_chunk,
    rot6d_row_to_matrix,
)
from motion.representation.kimodo_motionrep import KimodoMotionRep
from skeleton.definitions import G1Skeleton34


G1_XML_PATH = PROJECT_ROOT / "skeleton/assets/g1skel34/xml/g1.xml"
DEFAULT_TEXT_EMBEDDING_CACHE = PROJECT_ROOT / "data/cache/HumanoidArena"
_SEED_MASK = (1 << 63) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _derive_inference_seed(episode_seed: int, inference_index: int) -> int:
    value = (
        (int(episode_seed) & 0xFFFFFFFFFFFFFFFF)
        + _SPLITMIX_INCREMENT * (int(inference_index) + 1)
    ) & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) & _SEED_MASK


def _configure_deterministic_torch() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _resolve_execution_frames(requested_frames: int, prediction_frames: int) -> int:
    requested_frames = int(requested_frames)
    prediction_frames = int(prediction_frames)
    if prediction_frames <= 0:
        raise ValueError(f"prediction_frames must be positive, got {prediction_frames}")
    if requested_frames < 0:
        raise ValueError(f"execution_frames must be non-negative, got {requested_frames}")
    if requested_frames == 0:
        return prediction_frames
    if requested_frames > prediction_frames:
        raise ValueError(
            f"execution_frames ({requested_frames}) cannot exceed the checkpoint "
            f"action_chunk ({prediction_frames})"
        )
    return requested_frames


def _resolve_rtc_parameters(
    enabled: bool,
    *,
    prediction_frames: int,
    execution_frames: int,
    overlap_frames: int,
    frozen_frames: int,
    ramp_power: float,
) -> tuple[bool, int, int, float]:
    """Validate RTC settings and clamp overlap to the unexecuted chunk tail."""
    prediction_frames = int(prediction_frames)
    execution_frames = int(execution_frames)
    overlap_frames = int(overlap_frames)
    frozen_frames = int(frozen_frames)
    ramp_power = float(ramp_power)
    if overlap_frames < 0:
        raise ValueError(f"rtc_overlap_frames must be non-negative, got {overlap_frames}")
    if frozen_frames < 0:
        raise ValueError(f"rtc_frozen_frames must be non-negative, got {frozen_frames}")
    if frozen_frames > overlap_frames:
        raise ValueError(
            "rtc_frozen_frames cannot exceed rtc_overlap_frames, got "
            f"{frozen_frames} and {overlap_frames}"
        )
    if not np.isfinite(ramp_power) or ramp_power <= 0:
        raise ValueError(f"rtc_ramp_power must be finite and positive, got {ramp_power}")
    available_tail = max(0, prediction_frames - execution_frames)
    effective_overlap = min(overlap_frames, available_tail)
    effective_frozen = min(frozen_frames, effective_overlap)
    active = bool(enabled) and effective_overlap > 0
    return active, effective_overlap, effective_frozen, ramp_power


def _select_execution_prefix(
    future_features: torch.Tensor,
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    execution_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = {
        int(future_features.shape[0]),
        int(local_rot_mats.shape[0]),
        int(root_positions.shape[0]),
    }
    if len(lengths) != 1:
        raise ValueError(f"Prediction outputs have inconsistent frame counts: {sorted(lengths)}")
    available_frames = lengths.pop()
    if execution_frames > available_frames:
        raise ValueError(
            f"execution_frames ({execution_frames}) exceeds prediction output ({available_frames})"
        )
    return (
        future_features[:execution_frames],
        local_rot_mats[:execution_frames],
        root_positions[:execution_frames],
    )


def _select_hand_execution_prefix(
    hand_binary: torch.Tensor,
    execution_frames: int,
) -> torch.Tensor:
    if hand_binary.ndim != 2 or hand_binary.shape[-1] != 2:
        raise ValueError(
            f"Expected predicted hand shape [T, 2], got {tuple(hand_binary.shape)}"
        )
    if execution_frames > hand_binary.shape[0]:
        raise ValueError(
            f"execution_frames ({execution_frames}) exceeds hand prediction "
            f"({hand_binary.shape[0]})"
        )
    return hand_binary[:execution_frames]


def _align_hand_history(
    hand_history: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Align recurrent hand state length to a separately sampled motion history."""
    target_length = int(target_length)
    if target_length < 0:
        raise ValueError("target_length must be non-negative")
    if hand_history.ndim != 3 or hand_history.shape[0] != 1 or hand_history.shape[2] != 2:
        raise ValueError(
            f"Expected hand_history shape [1, T, 2], got {tuple(hand_history.shape)}"
        )
    if hand_history.shape[1] >= target_length:
        return hand_history[:, -target_length:] if target_length else hand_history[:, :0]
    padding = torch.zeros(
        1,
        target_length - hand_history.shape[1],
        2,
        device=hand_history.device,
        dtype=hand_history.dtype,
    )
    return torch.cat((padding, hand_history), dim=1)


def _kimodo_features_from_motion(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    representation: KimodoMotionRep,
) -> torch.Tensor:
    """Build Kimodo features, padding only the short startup solver input."""
    motion_frame_count = local_rot_mats.shape[0]
    representation_local_rot_mats = local_rot_mats
    representation_root_positions = root_positions
    if motion_frame_count < 5:
        # Kimodo's smooth-root solver needs at least three samples at its
        # coarsest level. Startup requests can legitimately contain one to four
        # frames, so extend only for feature construction and discard the
        # synthetic tail immediately afterwards.
        padding = 5 - motion_frame_count
        representation_local_rot_mats = torch.cat(
            (
                local_rot_mats,
                local_rot_mats[-1:].expand(padding, -1, -1, -1),
            ),
            dim=0,
        )
        representation_root_positions = torch.cat(
            (
                root_positions,
                root_positions[-1:].expand(padding, -1),
            ),
            dim=0,
        )
    return representation(
        representation_local_rot_mats,
        representation_root_positions,
        to_normalize=False,
    ).cpu()[:motion_frame_count]


def _pad_configuration_for_decoder(
    joint_q: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad short configurations around decoder-internal smooth-root features."""
    frame_count = joint_q.shape[0]
    if frame_count >= 5:
        return joint_q, root_positions, root_rotations, frame_count
    padding = 5 - frame_count
    return (
        torch.cat((joint_q, joint_q[-1:].expand(padding, -1)), dim=0),
        torch.cat(
            (root_positions, root_positions[-1:].expand(padding, -1)), dim=0
        ),
        torch.cat(
            (
                root_rotations,
                root_rotations[-1:].expand(padding, -1, -1),
            ),
            dim=0,
        ),
        frame_count,
    )


def _partial_arena_state_to_motion(
    state: np.ndarray | torch.Tensor,
    *,
    decoder: HumanoidArenaActionDecoder,
    representation: KimodoMotionRep,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert one or more 64D Arena observations into partial Kimodo constraints."""
    states = torch.as_tensor(state, dtype=torch.float32)
    if states.ndim == 1:
        states = states.unsqueeze(0)
    if states.ndim != 2 or states.shape[1] != 64 or states.shape[0] == 0:
        raise ValueError(f"Expected HumanoidArena state shape [T, 64], got {tuple(states.shape)}")
    if not torch.isfinite(states).all():
        raise ValueError("HumanoidArena observation.state contains NaN or Inf")
    joint_q, root_positions, root_rotations, frame_count = (
        _pad_configuration_for_decoder(
            states[:, 6:35],
            torch.zeros(states.shape[0], 3, dtype=torch.float32),
            rot6d_row_to_matrix(states[:, :6]),
        )
    )
    decoded = decoder.decode_joint_configuration(
        joint_q,
        root_positions,
        root_rotation_matrices=root_rotations,
        joint_names=CANONICAL_G1_JOINT_NAMES_29,
    )
    decoded = {
        key: value[:frame_count] if isinstance(value, torch.Tensor) else value
        for key, value in decoded.items()
    }
    motion = _kimodo_features_from_motion(
        decoded["local_rot_mats"], decoded["root_positions"], representation
    )
    feature_mask = torch.zeros_like(motion, dtype=torch.bool)
    feature_mask[:, representation.slice_dict["global_root_heading"]] = True
    feature_mask[:, representation.slice_dict["global_rot_data"]] = True
    return (
        motion,
        feature_mask,
        decoded["local_rot_mats"],
        decoded["root_positions"],
    )


def _partial_arena_state_history_to_motion(
    state_history: np.ndarray | torch.Tensor,
    *,
    decoder: HumanoidArenaActionDecoder,
    representation: KimodoMotionRep,
    source_fps: float,
    target_fps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert 50 Hz Arena observations exactly like the training adapter."""
    states = torch.as_tensor(state_history, dtype=torch.float32)
    if states.ndim != 2 or states.shape[1] != 64 or states.shape[0] == 0:
        raise ValueError(
            f"Expected observation state history shape [T, 64], got {tuple(states.shape)}"
        )
    if not torch.isfinite(states).all():
        raise ValueError("observation state history contains NaN or Inf")

    joint_q, root_positions, root_rotations, source_frame_count = (
        _pad_configuration_for_decoder(
            states[:, 6:35],
            torch.zeros(states.shape[0], 3, dtype=torch.float32),
            rot6d_row_to_matrix(states[:, :6]),
        )
    )
    decoded = decoder.decode_joint_configuration(
        joint_q,
        root_positions,
        root_rotation_matrices=root_rotations,
        joint_names=CANONICAL_G1_JOINT_NAMES_29,
    )
    local_rot_mats = decoded["local_rot_mats"][:source_frame_count]
    root_positions = decoded["root_positions"][:source_frame_count]
    local_rot_mats, root_positions = resample_motion(
        local_rot_mats,
        root_positions,
        source_fps=source_fps,
        target_fps=target_fps,
    )
    motion = _kimodo_features_from_motion(
        local_rot_mats, root_positions, representation
    )
    feature_mask = torch.zeros_like(motion, dtype=torch.bool)
    feature_mask[:, representation.slice_dict["global_root_heading"]] = True
    feature_mask[:, representation.slice_dict["global_rot_data"]] = True
    return motion, feature_mask, local_rot_mats, root_positions


def _validated_text_embedding(value, *, cache_path: Path, task_id: str) -> torch.Tensor:
    embedding = torch.as_tensor(value).detach().cpu().contiguous()
    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError(
            f"Expected text embedding [1, D] for task {task_id!r} in {cache_path}, "
            f"got {tuple(embedding.shape)}"
        )
    if embedding.shape[1] <= 0 or not torch.isfinite(embedding).all():
        raise ValueError(
            f"Text embedding for task {task_id!r} in {cache_path} is empty or non-finite"
        )
    return embedding


def _register_task_alias(aliases: dict[str, str], alias: str, task_id: str) -> None:
    alias = str(alias).strip()
    if not alias:
        return
    existing = aliases.get(alias)
    if existing is not None and existing != task_id:
        raise ValueError(
            f"Text cache alias {alias!r} maps to both {existing!r} and {task_id!r}"
        )
    aliases[alias] = task_id


def _load_text_embedding_cache(
    cache_location: str | os.PathLike,
) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, str]]:
    cache_path = Path(cache_location).expanduser().resolve()
    if cache_path.is_dir():
        cache_files = sorted(cache_path.glob("*.pt"))
        if not cache_files:
            raise RuntimeError(f"No text embedding .pt files found below {cache_path}")
    elif cache_path.is_file():
        cache_files = [cache_path]
    else:
        raise FileNotFoundError(f"Text embedding cache does not exist: {cache_path}")

    embeddings: dict[str, torch.Tensor] = {}
    instructions: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for file_path in cache_files:
        payload = torch.load(file_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError(f"Text embedding cache must contain a dictionary: {file_path}")

        # Legacy evaluation caches stored every task in one file.
        if "embeddings" in payload:
            legacy_embeddings = payload["embeddings"]
            legacy_instructions = payload.get("instructions", {})
            if not isinstance(legacy_embeddings, dict) or not isinstance(
                legacy_instructions, dict
            ):
                raise TypeError(f"Invalid legacy text embedding cache: {file_path}")
            if set(legacy_embeddings) != set(legacy_instructions):
                raise ValueError(
                    f"Legacy text cache task IDs and instructions do not match: {file_path}"
                )
            entries = [
                (str(task_id), str(instruction).strip(), embedding, None)
                for task_id, embedding in legacy_embeddings.items()
                for instruction in [legacy_instructions[task_id]]
            ]
        else:
            task_id = str(payload.get("task_id", "")).strip()
            instruction = str(payload.get("instruction", "")).strip()
            if not task_id or not instruction or "embedding" not in payload:
                raise ValueError(
                    f"New text cache requires task_id, instruction, and embedding: {file_path}"
                )
            source = str(payload.get("source", "")).strip()
            if source and source != "HumanoidArena":
                raise ValueError(
                    f"Expected a HumanoidArena text cache, got source={source!r}: {file_path}"
                )
            entries = [
                (
                    task_id,
                    instruction,
                    payload["embedding"],
                    str(payload.get("task_name", "")).strip(),
                )
            ]

        for task_id, instruction, embedding_value, task_name in entries:
            embedding = _validated_text_embedding(
                embedding_value, cache_path=file_path, task_id=task_id
            )
            if task_id in embeddings:
                if instructions[task_id] != instruction or not torch.equal(
                    embeddings[task_id], embedding
                ):
                    raise ValueError(
                        f"Conflicting text embedding cache entries for task {task_id!r}"
                    )
                continue
            embeddings[task_id] = embedding
            instructions[task_id] = instruction
            _register_task_alias(aliases, task_id, task_id)
            if task_id.startswith("HumanoidArena::"):
                _register_task_alias(
                    aliases, task_id.removeprefix("HumanoidArena::"), task_id
                )
            if task_name:
                _register_task_alias(aliases, task_name, task_id)

    if not embeddings:
        raise RuntimeError(f"No text embeddings found in {cache_path}")
    return embeddings, instructions, aliases


def _resolve_cached_task(task: str, aliases: dict[str, str]) -> str:
    task = str(task).strip()
    if task in aliases:
        return aliases[task]
    prefixed = f"HumanoidArena::{task}"
    if prefixed in aliases:
        return aliases[prefixed]
    available = ", ".join(sorted(aliases))
    raise KeyError(f"No cached embedding for task {task!r}; available aliases: {available}")


class KimodoHumanoidArenaRuntime:
    def __init__(self, args: argparse.Namespace):
        self.deterministic_eval = _env_flag("KIMODO_DETERMINISTIC_EVAL")
        if self.deterministic_eval:
            _configure_deterministic_torch()
        checkpoint_dir = Path(args.checkpoint).expanduser().resolve()
        checkpoint_config = json.loads((checkpoint_dir / "config.json").read_text())
        model_config = checkpoint_config["model"]
        main_config = checkpoint_config["main"]
        config = KimodoPolicyConfig(
            fps=int(model_config["fps"]),
            motion_mask_mode=str(model_config["motion_mask_mode"]),
            dinov3_model_name=str(model_config["dinov3_model_name"]),
            dinov3_checkpoint=model_config.get("dinov3_checkpoint"),
            action_chunk=int(main_config["action_chunk"]),
            action_history=int(main_config["action_history"]),
            load_text_encoder=False,
            controlnet_num_layers=int(model_config.get("controlnet_num_layers", 8)),
            root_loss_weight=float(checkpoint_config["training"]["loss"]["root_weight"]),
            body_loss_weight=float(checkpoint_config["training"]["loss"]["body_weight"]),
            enable_hand_head=bool(model_config.get("enable_hand_head", False)),
            hand_hidden_dim=int(model_config.get("hand_hidden_dim", 256)),
            hand_num_layers=int(model_config.get("hand_num_layers", 4)),
            hand_num_heads=int(model_config.get("hand_num_heads", 4)),
            hand_ffn_dim=int(model_config.get("hand_ffn_dim", 1024)),
            hand_loss_weight=float(
                checkpoint_config["training"]["loss"].get("hand_weight", 1.0)
            ),
            hand_transition_loss_weight=float(
                checkpoint_config["training"]["loss"].get(
                    "hand_transition_weight", 0.2
                )
            ),
            hand_init_seed=int(model_config.get("hand_init_seed", 3407)),
        )
        self.device = torch.device(args.device)
        self.dtype = _dtype_from_name(args.dtype)
        self.diffusion_steps = int(args.diffusion_steps)
        self.control_fps = float(args.control_fps)
        self.execution_frames = _resolve_execution_frames(
            args.execution_frames, config.action_chunk
        )
        (
            self.rtc_enabled,
            self.rtc_overlap_frames,
            self.rtc_frozen_frames,
            self.rtc_ramp_power,
        ) = _resolve_rtc_parameters(
            bool(args.rtc),
            prediction_frames=config.action_chunk,
            execution_frames=self.execution_frames,
            overlap_frames=args.rtc_overlap_frames,
            frozen_frames=args.rtc_frozen_frames,
            ramp_power=args.rtc_ramp_power,
        )
        (
            self.text_embeddings,
            self.task_instructions,
            self.task_aliases,
        ) = _load_text_embedding_cache(args.text_embedding_cache)
        self.model = KimodoPolicy(config)
        loaded_step = self.model.load_controlnet_checkpoint(str(checkpoint_dir))
        self.model.to(device=self.device, dtype=self.dtype).eval()
        self.action_codec = HumanoidArenaActionDecoder(
            G1Skeleton34(), G1_XML_PATH, fps=self.control_fps
        )
        self.state_motion_representation = KimodoMotionRep(
            G1Skeleton34(), fps=float(self.model.fps), stats_path=None
        )
        self.observation_state_history = np.empty((0, 64), dtype=np.float32)
        self.history_motion = torch.empty(
            1,
            0,
            self.model.representation.motion_rep_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.history_hand = torch.empty(
            1,
            0,
            2,
            device=self.device,
            dtype=torch.float32,
        )
        self.previous_local_rot_mat = None
        self.previous_root_position = None
        self.rtc_motion_tail = None
        self.rtc_hand_tail = None
        self.rtc_task_id = None
        self.episode_seed: int | None = None
        self.inference_index = 0
        self.lock = threading.Lock()
        print(
            f"Loaded checkpoint step={loaded_step} on {self.device} "
            f"dtype={self.dtype} diffusion_steps={self.diffusion_steps} "
            f"synchronized_hand_diffusion={config.enable_hand_head} "
            f"prediction_frames={config.action_chunk} "
            f"execution_frames={self.execution_frames} "
            f"rtc_enabled={self.rtc_enabled} "
            f"rtc_overlap_frames={self.rtc_overlap_frames} "
            f"rtc_frozen_frames={self.rtc_frozen_frames} "
            f"rtc_ramp_power={self.rtc_ramp_power:g} "
            f"deterministic_eval={self.deterministic_eval} "
            f"text_tasks={sorted(self.text_embeddings)}",
            flush=True,
        )

    def reset(self, seed: int | None = None) -> None:
        with self.lock:
            if seed is not None:
                seed = int(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                self.episode_seed = seed
            self.inference_index = 0
            self.observation_state_history = np.empty((0, 64), dtype=np.float32)
            self.history_motion = torch.empty(
                1,
                0,
                self.model.representation.motion_rep_dim,
                device=self.device,
                dtype=torch.float32,
            )
            self.history_hand = torch.empty(
                1,
                0,
                2,
                device=self.device,
                dtype=torch.float32,
            )
            self.previous_local_rot_mat = None
            self.previous_root_position = None
            self.rtc_motion_tail = None
            self.rtc_hand_tail = None
            self.rtc_task_id = None

    @staticmethod
    def _decode_rgb(image_payload: dict) -> torch.Tensor:
        shape = tuple(int(value) for value in image_payload["shape"])
        dtype = np.dtype(image_payload["dtype"])
        raw = base64.b64decode(image_payload["data_b64"])
        rgb = np.frombuffer(raw, dtype=dtype).reshape(shape)
        if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
            raise ValueError(f"Expected HWC RGB/RGBA image, got {rgb.shape}")
        rgb = np.array(rgb[..., :3], copy=True, order="C")
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    def infer(self, payload: dict) -> np.ndarray:
        observation = payload["observation"]
        task = str(payload.get("task", "")).strip()
        task_id = _resolve_cached_task(
            task,
            getattr(
                self,
                "task_aliases",
                {task_id: task_id for task_id in self.text_embeddings},
            ),
        )
        image = self._decode_rgb(observation["images"]["front"]).to(self.device)
        state_history_segment = observation.get("state_history")
        text_feature = self.text_embeddings[task_id].to(dtype=self.dtype)
        instruction = self.task_instructions[task_id]

        with self.lock, torch.inference_mode():
            # Never expose the committed recurrent tensors to prediction,
            # resampling, or encoding code.  The current implementations are
            # read-only, but snapshots also preserve transactionality if a future
            # implementation performs an in-place operation before raising.
            history_motion_snapshot = self.history_motion.clone()
            history_hand_snapshot = self.history_hand.clone()
            model_hand_history_snapshot = history_hand_snapshot
            observation_state_history_snapshot = np.array(
                self.observation_state_history, dtype=np.float32, copy=True
            )
            previous_local_rot_mat_snapshot = (
                None
                if self.previous_local_rot_mat is None
                else self.previous_local_rot_mat.clone()
            )
            previous_root_position_snapshot = (
                None
                if self.previous_root_position is None
                else self.previous_root_position.clone()
            )
            rtc_motion_tail_snapshot = (
                None
                if getattr(self, "rtc_motion_tail", None) is None
                else self.rtc_motion_tail.clone()
            )
            rtc_hand_tail_snapshot = (
                None
                if getattr(self, "rtc_hand_tail", None) is None
                else self.rtc_hand_tail.clone()
            )
            rtc_task_id_snapshot = getattr(self, "rtc_task_id", None)
            use_rtc_reference = (
                bool(getattr(self, "rtc_enabled", False))
                and rtc_motion_tail_snapshot is not None
                and rtc_task_id_snapshot == task_id
            )
            history_feature_mask_snapshot = None
            using_state_history = False
            next_observation_state_history = observation_state_history_snapshot
            motion_previous_local_rot_mat = previous_local_rot_mat_snapshot
            motion_previous_root_position = previous_root_position_snapshot
            if state_history_segment is not None:
                using_state_history = True
                segment = np.asarray(state_history_segment, dtype=np.float32)
                if segment.ndim == 1:
                    segment = segment.reshape(1, -1)
                if segment.ndim != 2 or segment.shape[1] != 64 or segment.shape[0] == 0:
                    raise ValueError(
                        "observation.state_history must have shape [T, 64]"
                    )
                if not np.isfinite(segment).all():
                    raise ValueError("observation.state_history contains NaN or Inf")
                next_observation_state_history = np.concatenate(
                    (observation_state_history_snapshot, segment), axis=0
                )
                (
                    state_history_motion,
                    state_history_feature_mask,
                    state_local_rot_mats,
                    state_root_positions,
                ) = _partial_arena_state_history_to_motion(
                    next_observation_state_history,
                    decoder=self.action_codec,
                    representation=self.state_motion_representation,
                    source_fps=self.control_fps,
                    target_fps=float(self.model.fps),
                )
                # The request image and newest state are at the prediction cut.
                # Training conditions on [cut-history, cut), so keep the newest
                # state for the next request but exclude it from this prediction.
                condition_history_motion = state_history_motion[:-1]
                condition_history_feature_mask = state_history_feature_mask[:-1]
                condition_history_motion = condition_history_motion[
                    -self.model.config.action_history :
                ]
                history_motion_snapshot = condition_history_motion.unsqueeze(0).to(
                    device=self.device, dtype=torch.float32
                )
                history_feature_mask_snapshot = condition_history_feature_mask[
                    -self.model.config.action_history :
                ].unsqueeze(0).to(device=self.device)
                model_hand_history_snapshot = _align_hand_history(
                    history_hand_snapshot, history_motion_snapshot.shape[1]
                )
                motion_previous_local_rot_mat = state_local_rot_mats[-1].clone()
                if motion_previous_root_position is None:
                    motion_previous_root_position = state_root_positions[-1].clone()
            elif observation.get("state") is not None:
                raise ValueError(
                    "Single-frame observation.state inference is unsupported because "
                    "it does not match training history. Update HumanoidArena to send "
                    "observation.state_history at every replanning request."
                )
            # Existing checkpoints were trained without valid observed hand state,
            # so the hand head always saw the default fully-open state.  Preserve
            # that condition at inference while keeping history_hand_snapshot intact
            # for continuous hand control across replanning boundaries.
            model_hand_history_snapshot = torch.zeros_like(
                model_hand_history_snapshot
            )
            generator = None
            noise_seed = None
            if self.deterministic_eval:
                if self.episode_seed is None:
                    raise RuntimeError("model server was not reset with an episode seed")
                noise_seed = _derive_inference_seed(
                    self.episode_seed, self.inference_index
                )
                generator = torch.Generator(device=self.device)
                generator.manual_seed(noise_seed)
            output = self.model.predict_future(
                instruction=instruction,
                egoview=image,
                history_motion=history_motion_snapshot,
                history_feature_mask=history_feature_mask_snapshot,
                diffusion_steps=self.diffusion_steps,
                squeeze_batch=True,
                text_feat=text_feature,
                hand_history=model_hand_history_snapshot,
                generator=generator,
                rtc_motion_reference=(
                    rtc_motion_tail_snapshot if use_rtc_reference else None
                ),
                rtc_hand_reference=(
                    rtc_hand_tail_snapshot if use_rtc_reference else None
                ),
                rtc_overlap_frames=int(getattr(self, "rtc_overlap_frames", 0)),
                rtc_frozen_frames=int(getattr(self, "rtc_frozen_frames", 0)),
                rtc_ramp_power=float(getattr(self, "rtc_ramp_power", 1.0)),
            )
            full_future_features = output["motion_features"].detach().to(
                device=self.device, dtype=torch.float32
            )
            full_source_local_rot_mats = (
                output["local_rot_mats"].detach().float().cpu()
            )
            full_source_root_positions = (
                output["root_positions"].detach().float().cpu()
            )
            if not (
                full_future_features.ndim == 2
                and full_source_local_rot_mats.shape[0]
                == full_source_root_positions.shape[0]
                == full_future_features.shape[0]
            ):
                raise ValueError("Model returned inconsistent full prediction shapes")

            next_rtc_motion_tail = None
            next_rtc_hand_tail = None
            next_rtc_task_id = None
            if bool(getattr(self, "rtc_enabled", False)):
                rtc_tail_start = self.execution_frames
                rtc_tail_end = min(
                    full_future_features.shape[0],
                    rtc_tail_start + int(self.rtc_overlap_frames),
                )
                if rtc_tail_end > rtc_tail_start:
                    next_rtc_motion_tail = full_future_features[
                        rtc_tail_start:rtc_tail_end
                    ].detach().clone()
                    next_rtc_task_id = task_id
                    full_hand_clean = output.get("hand_clean")
                    if full_hand_clean is not None:
                        full_hand_clean = (
                            full_hand_clean.detach().to(
                                device=self.device, dtype=torch.float32
                            )
                        )
                        if (
                            full_hand_clean.ndim != 2
                            or full_hand_clean.shape[1] != 2
                            or full_hand_clean.shape[0]
                            != full_future_features.shape[0]
                        ):
                            raise ValueError(
                                "Model returned inconsistent clean hand prediction shape"
                            )
                        next_rtc_hand_tail = full_hand_clean[
                            rtc_tail_start:rtc_tail_end
                        ].detach().clone()

            future_features = full_future_features
            source_local_rot_mats = full_source_local_rot_mats
            source_root_positions = full_source_root_positions
            history_last_root_position = output.get("history_last_root_position")
            if history_last_root_position is not None:
                history_last_root_position = (
                    history_last_root_position.detach().float().cpu().reshape(3)
                )
                if not torch.isfinite(history_last_root_position).all():
                    raise ValueError(
                        "Predicted history boundary root contains NaN or Inf"
                    )
                # Root x/z is window-relative.  The first future delta must be
                # measured from the history endpoint generated in this same
                # diffusion window, never from the previous window's gauge.
                motion_previous_root_position = history_last_root_position
            future_features, source_local_rot_mats, source_root_positions = (
                _select_execution_prefix(
                    future_features,
                    source_local_rot_mats,
                    source_root_positions,
                    self.execution_frames,
                )
            )
            source_hand_binary = output.get("hand_binary")
            previous_hand_binary = (
                history_hand_snapshot[0, -1].detach().float().cpu().clone()
                if history_hand_snapshot.shape[1] > 0
                else torch.zeros(2, dtype=torch.float32)
            )
            if source_hand_binary is not None:
                source_hand_binary = _select_hand_execution_prefix(
                    source_hand_binary.detach().float().cpu(),
                    self.execution_frames,
                )
            # Build the next recurrent state without publishing it yet.  Resampling
            # and action encoding can still fail; in that case the client receives
            # no action and the server must continue to represent the last action
            # chunk that was actually returned successfully.
            if not using_state_history:
                next_history_motion = torch.cat(
                    (history_motion_snapshot, future_features.unsqueeze(0)), dim=1
                )[:, -self.model.config.action_history :]
            else:
                next_history_motion = history_motion_snapshot
            next_history_hand = history_hand_snapshot
            if source_hand_binary is not None:
                next_history_hand = torch.cat(
                    (
                        history_hand_snapshot,
                        source_hand_binary.to(self.device).unsqueeze(0),
                    ),
                    dim=1,
                )[:, -self.model.config.action_history :]

            local_rot_mats, root_positions = resample_motion_chunk(
                source_local_rot_mats,
                source_root_positions,
                source_fps=float(self.model.fps),
                target_fps=self.control_fps,
                previous_local_rot_mat=motion_previous_local_rot_mat,
                previous_root_position=motion_previous_root_position,
            )
            control_hand_binary = None
            if source_hand_binary is not None:
                control_hand_binary = resample_hand_binary_chunk(
                    source_hand_binary,
                    source_fps=float(self.model.fps),
                    target_fps=self.control_fps,
                    previous_hand_binary=previous_hand_binary,
                )
                if control_hand_binary.shape[0] != local_rot_mats.shape[0]:
                    raise RuntimeError(
                        "Motion and hand control chunks have different lengths: "
                        f"{local_rot_mats.shape[0]} and {control_hand_binary.shape[0]}"
                    )
            action_chunk = self.action_codec.encode(
                local_rot_mats,
                root_positions,
                previous_root_position=motion_previous_root_position,
                hand_binary=control_hand_binary,
            )
            # Materialize the response before committing any recurrent state.  This
            # also keeps conversion failures transactional.
            action_chunk_numpy = action_chunk.detach().cpu().numpy().astype(np.float32)
            next_previous_local_rot_mat = source_local_rot_mats[-1].clone()
            next_previous_root_position = source_root_positions[-1].clone()
            if self.deterministic_eval:
                print(
                    "[deterministic_inference] "
                    f"episode_seed={self.episode_seed} "
                    f"inference_index={self.inference_index} "
                    f"noise_seed={noise_seed} "
                    f"rtc_reference_used={use_rtc_reference}",
                    flush=True,
                )

            # Commit all state together only after prediction, resampling, encoding,
            # response conversion, and deterministic logging have succeeded.
            self.observation_state_history = next_observation_state_history
            self.history_motion = next_history_motion
            self.history_hand = next_history_hand
            self.previous_local_rot_mat = next_previous_local_rot_mat
            self.previous_root_position = next_previous_root_position
            self.rtc_motion_tail = next_rtc_motion_tail
            self.rtc_hand_tail = next_rtc_hand_tail
            self.rtc_task_id = next_rtc_task_id
            if self.deterministic_eval:
                self.inference_index += 1
        return action_chunk_numpy


class RequestHandler(BaseHTTPRequestHandler):
    runtime: KimodoHumanoidArenaRuntime = None

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in {"", "/health"}:
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path.rstrip("/") == "/reset":
                self.runtime.reset(payload.get("seed"))
                self._send_json(200, {"status": "ok"})
                return
            if self.path.rstrip("/") == "/infer":
                action_chunk = self.runtime.infer(payload)
                self._send_json(200, {"action_chunk": action_chunk.tolist()})
                return
            self._send_json(404, {"error": "not found"})
        except Exception as exc:
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kimodo policy server for HumanoidArena")
    parser.add_argument("--checkpoint", "--policy-path", dest="checkpoint", required=True)
    parser.add_argument(
        "--text-embedding-cache",
        default=os.environ.get(
            "KIMODO_TEXT_EMBEDDING_CACHE", str(DEFAULT_TEXT_EMBEDDING_CACHE)
        ),
        help="HumanoidArena per-task cache directory or a legacy aggregate cache file",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp32"),
        default=os.environ.get("KIMODO_DTYPE", "fp32"),
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=int(os.environ.get("KIMODO_DIFFUSION_STEPS", "10")),
    )
    parser.add_argument(
        "--execution-frames",
        type=int,
        default=int(os.environ.get("KIMODO_EXECUTION_FRAMES", "0")),
        help="Execute only this prediction prefix; 0 executes the full action chunk",
    )
    parser.add_argument(
        "--rtc",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("KIMODO_RTC", "1")),
        help="Enable DDIM real-time chunking when an unexecuted tail exists",
    )
    parser.add_argument(
        "--rtc-overlap-frames",
        type=int,
        default=int(os.environ.get("KIMODO_RTC_OVERLAP_FRAMES", "12")),
        help="Previous 30 Hz motion frames used as the next chunk's RTC prior",
    )
    parser.add_argument(
        "--rtc-frozen-frames",
        type=int,
        default=int(os.environ.get("KIMODO_RTC_FROZEN_FRAMES", "1")),
        help="Leading RTC frames preserved exactly before the cosine soft ramp",
    )
    parser.add_argument(
        "--rtc-ramp-power",
        type=float,
        default=float(os.environ.get("KIMODO_RTC_RAMP_POWER", "1.0")),
        help="Positive exponent controlling how quickly old-trajectory influence decays",
    )
    parser.add_argument("--control-fps", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RequestHandler.runtime = KimodoHumanoidArenaRuntime(args)
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"Serving on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
