import torch
import random
import logging
import os
import json
import time
import argparse
import gc
import hashlib
import re
import torch.distributed as dist
import numpy as np
import wandb
import math
from functools import partial
from torch.utils.data import DataLoader, Sampler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from datetime import datetime
from accelerate.utils import InitProcessGroupKwargs
from accelerate.utils import set_seed
from datetime import timedelta
from data.multisource_dataset import MultiSourceG1Dataset
from data.motion_cache import (
    PRETRAIN_MOTION_CACHE_SOURCES,
    MOTION_CACHE_VERSION,
    motion_cache_signature,
    prepare_motion_cache,
)
from model.kimodo_policy import KimodoPolicy, KimodoPolicyConfig
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)
logging.getLogger("accelerate").setLevel(logging.ERROR)

_MOTION_COMPONENT_LOSS_KEYS = (
    "root_position_loss",
    "root_heading_loss",
    "joint_position_loss",
    "joint_velocity_loss",
    "joint_rotation_loss",
    "foot_contact_loss",
    "fk_loss",
)
_DEFAULT_KIMODO_SMOOTH_L1_WEIGHT_CONFIG = {
    config_key: KimodoPolicy.KIMODO_SMOOTH_L1_WEIGHTS[output_key]
    for config_key, output_key in KimodoPolicy.KIMODO_SMOOTH_L1_WEIGHT_CONFIG_KEYS.items()
}
_DEFAULT_MSE_WEIGHT_CONFIG = {
    "root": 2.0,
    "body": 1.0,
    "hand": 1.0,
    "hand_transition": 0.2,
}
_LEGACY_MSE_WEIGHT_KEYS = {
    "root_weight": "root",
    "body_weight": "body",
    "hand_weight": "hand",
    "hand_transition_weight": "hand_transition",
}


def build_controlnet_param_groups(model, config):
    backbone_lr = float(config.training.optimizer.control_backbone_lr)
    adapter_lr = float(config.training.optimizer.control_adapter_lr)
    hand_lr = float(config.training.optimizer.get("hand_lr", adapter_lr))
    weight_decay = float(config.training.optimizer.weight_decay)
    grouped = {
        "backbone_decay": {"params": [], "lr": backbone_lr, "weight_decay": weight_decay},
        "backbone_no_decay": {"params": [], "lr": backbone_lr, "weight_decay": 0.0},
        "adapter_decay": {"params": [], "lr": adapter_lr, "weight_decay": weight_decay},
        "adapter_no_decay": {"params": [], "lr": adapter_lr, "weight_decay": 0.0},
    }
    if model.hand_head is not None:
        grouped.update(
            {
                "hand_decay": {
                    "params": [], "lr": hand_lr, "weight_decay": weight_decay
                },
                "hand_no_decay": {
                    "params": [], "lr": hand_lr, "weight_decay": 0.0
                },
            }
        )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_hand = name.startswith("hand_head.")
        is_backbone = name.startswith("controlnet.root_layers.") or name.startswith(
            "controlnet.body_layers."
        )
        use_decay = parameter.ndim >= 2 and not name.endswith("bias")
        family = "hand" if is_hand else ("backbone" if is_backbone else "adapter")
        decay = "decay" if use_decay else "no_decay"
        grouped[f"{family}_{decay}"]["params"].append(parameter)
    if not all(group["params"] for group in grouped.values()):
        empty_groups = [name for name, group in grouped.items() if not group["params"]]
        raise RuntimeError(f"Empty optimizer parameter groups: {empty_groups}")
    return list(grouped.values())


def setup_logging(rank, save_path):
    logging.basicConfig(level=logging.INFO, format=f'[Rank {rank}] %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter(f'[Rank {rank}] %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if rank == 0:
        log_file = os.path.join(save_path, 'training.log')
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter) 
        logging.getLogger().addHandler(file_handler) 


def build_model_and_optimizer(config):
    # 1. 创建模型
    loss_config = config.training.loss
    mse_weights = loss_config.get("mse_weights", {}) or {}
    kimodo_smooth_l1_weights = config.training.loss.get(
        "kimodo_smooth_l1_weights", None
    )
    if kimodo_smooth_l1_weights is not None:
        kimodo_smooth_l1_weights = (
            OmegaConf.to_container(kimodo_smooth_l1_weights, resolve=True)
            if OmegaConf.is_config(kimodo_smooth_l1_weights)
            else dict(kimodo_smooth_l1_weights)
        )
    model_config = KimodoPolicyConfig(
        fps              = config.model.fps,
        motion_mask_mode = config.model.motion_mask_mode,
        dinov3_model_name= config.model.dinov3_model_name,
        dinov3_checkpoint= config.model.get("dinov3_checkpoint", None),
        action_chunk     = config.main.action_chunk,
        action_history   = config.main.action_history,
        load_text_encoder = not config.main.get("precompute_text_embeddings", True),
        controlnet_num_layers = config.model.get("controlnet_num_layers", 8),
        detach_root_control_for_body = config.model.get(
            "detach_root_control_for_body", False
        ),
        motion_loss_type = config.training.loss.get("motion_loss_type", "mse"),
        kimodo_smooth_l1_weights = kimodo_smooth_l1_weights,
        root_loss_weight = mse_weights.get(
            "root", loss_config.get("root_weight", 2.0)
        ),
        body_loss_weight = mse_weights.get(
            "body", loss_config.get("body_weight", 1.0)
        ),
        hand_hidden_dim = config.model.get("hand_hidden_dim", 256),
        hand_num_layers = config.model.get("hand_num_layers", 4),
        hand_num_heads = config.model.get("hand_num_heads", 4),
        hand_ffn_dim = config.model.get("hand_ffn_dim", 1024),
        hand_loss_weight = mse_weights.get(
            "hand", loss_config.get("hand_weight", 1.0)
        ),
        hand_transition_loss_weight = mse_weights.get(
            "hand_transition", loss_config.get("hand_transition_weight", 0.2)
        ),
        hand_init_seed = config.model.get("hand_init_seed", 3407),
    )
    model = KimodoPolicy(config=model_config)
    optimizer = torch.optim.AdamW(
        build_controlnet_param_groups(model, config),
        betas=tuple(config.training.optimizer.betas),
        eps=config.training.optimizer.eps,
        fused=config.training.optimizer.get("fused", True),
    )
    if config.model.get("compile", False):
        model = torch.compile(
            model,
            mode=config.model.get("compile_mode", "reduce-overhead"),
            dynamic=False,
        )
    # 2. 创建 scheduler（cosine with warmup，支持最小 lr）
    sig_gpu_max_training_steps = config.main.max_steps
    min_lr_ratio  = config.training.scheduler.get("min_lr_ratio", 0.0)
    num_cycles    = config.training.scheduler.num_cycles
    warmup_steps  = int(sig_gpu_max_training_steps * config.training.scheduler.warmup_ratio)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, sig_gpu_max_training_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    scheduler = LambdaLR(optimizer, lr_lambda)
    return model, optimizer, scheduler, sig_gpu_max_training_steps


def build_dataset(config):
    dataset_roots = config.main.get("dataset_roots", None)
    if dataset_roots is not None:
        dataset_roots = OmegaConf.to_container(dataset_roots, resolve=True)
    sampling = config.main.get("sampling", None)
    if sampling is not None:
        sampling = OmegaConf.to_container(sampling, resolve=True)
    return MultiSourceG1Dataset(
        dataset_root=config.main.get("data_root", None),
        dataset_roots=dataset_roots,
        action_history=config.main.action_history,
        action_chunk=config.main.action_chunk,
        sample_stride=config.main.get("sample_stride", 1),
        episode_cache_size=config.main.get("episode_cache_size", 2),
        video_cache_size=config.main.get("video_cache_size", 32),
        dataset_selection=OmegaConf.to_container(
            config.main.get("dataset_selection", {}), resolve=True
        ),
        sampling=sampling,
        target_fps=config.model.fps,
        sampling_seed=int(config.main.seed),
    )


def prepare_pretrain_motion_cache(config, dataset, accelerator, config_path):
    """Build/reuse the optional cache for non-Arena pre-training sources."""

    # Cache generation is opt-in from the pre-training YAML. Fine-tuning
    # configs intentionally omit this key and keep the live loader.
    enabled = bool(config.main.get("precompute_motion_cache", False))
    if not enabled:
        return
    selected_sources = set(dataset.adapters) & set(PRETRAIN_MOTION_CACHE_SOURCES)
    if not selected_sources:
        if accelerator.is_main_process:
            logger.info(
                "precompute_motion_cache=true but no supported pre-training source "
                "is selected; HumanoidArena is intentionally left on the live loader"
            )
        return

    dataset_selection = OmegaConf.to_container(
        config.main.get("dataset_selection", {}), resolve=True
    )
    dataset_roots = OmegaConf.to_container(
        config.main.get("dataset_roots", {}), resolve=True
    )
    signature_payload = {
        "cache_version": MOTION_CACHE_VERSION,
        "action_chunk": int(config.main.action_chunk),
        "sample_stride": int(config.main.get("sample_stride", 1)),
        "target_fps": float(config.model.fps),
        "dataset_roots": dataset_roots,
        "dataset_selection": dataset_selection,
        "selected_sources": sorted(selected_sources),
    }
    signature = motion_cache_signature(signature_payload)
    config_name = os.path.splitext(os.path.basename(str(config_path)))[0]
    cache_root = os.path.join(
        os.path.dirname(__file__),
        "data",
        "cache",
        f"pretrain_motion_{config_name}",
    )

    result = [None, None]
    if accelerator.is_main_process:
        try:
            cache_dir = prepare_motion_cache(
                dataset,
                cache_root,
                signature,
            )
            result[0] = str(cache_dir)
        except Exception as error:  # propagate a useful error to every rank
            result[1] = f"{type(error).__name__}: {error}"
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    if result[1] is not None:
        raise RuntimeError(result[1])
    dataset.attach_motion_cache(result[0], signature)
    if accelerator.is_main_process:
        logger.info(
            "Pre-training motion cache enabled: root=%s signature=%s",
            result[0],
            signature[:16],
        )


class ResumableOrdinalSampler(Sampler[int]):
    """Yield absolute sample ordinals so resumed data does not restart at zero."""

    def __init__(self, start: int, stop: int) -> None:
        self.start = int(start)
        self.stop = int(stop)
        if self.start < 0 or self.stop < self.start:
            raise ValueError(
                f"Invalid ordinal range: start={self.start}, stop={self.stop}"
            )

    def __iter__(self):
        return iter(range(self.start, self.stop))

    def __len__(self) -> int:
        return self.stop - self.start


def _config_get(config, key, default=None):
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _gradient_clip_statistics(total_norm, max_norm):
    """Return the pre-clip norm, clipping indicator, and applied scale."""
    max_norm = float(max_norm)
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError(f"Gradient clip max_norm must be finite and positive, got {max_norm}")
    if torch.is_tensor(total_norm):
        total_norm = total_norm.detach().float().item()
    total_norm = float(total_norm)
    if not math.isfinite(total_norm):
        return total_norm, 1.0, 0.0
    was_clipped = float(total_norm > max_norm)
    clip_scale = min(1.0, max_norm / (total_norm + 1e-12))
    return total_norm, was_clipped, clip_scale


def _samples_per_optimizer_step(config, world_size: int) -> int:
    gradient = _config_get(config.main, "gradient", None)
    accumulation_steps = int(
        _config_get(gradient, "grad_accumulation_steps", 1)
    )
    return (
        int(config.main.batch_size)
        * max(1, int(world_size))
        * accumulation_steps
    )


def _sample_ordinal_range(config, resume_step: int, world_size: int):
    max_steps = _config_get(config.main, "max_steps", None)
    if max_steps is None:
        return None
    samples_per_step = _samples_per_optimizer_step(config, world_size)
    start = int(resume_step) * samples_per_step
    stop = int(max_steps) * samples_per_step
    if start > stop:
        raise ValueError(
            f"Resume step {resume_step} exceeds configured max_steps={max_steps}"
        )
    return start, stop


def _process_seed(base_seed, global_step, world_size):
    return int(base_seed) + int(global_step) * max(1, int(world_size))


def _worker_seed(base_seed, global_step, world_size, workers_per_process, process_index, worker_id):
    workers_per_process = max(1, int(workers_per_process))
    stream_offset = (
        int(global_step) * max(1, int(world_size)) * workers_per_process
        + int(process_index) * workers_per_process
        + int(worker_id)
    )
    return int(base_seed) + stream_offset


def _initialize_dataloader_worker(
    worker_id,
    *,
    base_seed,
    global_step,
    world_size,
    workers_per_process,
    process_index,
):
    worker_seed = _worker_seed(
        base_seed,
        global_step,
        world_size,
        workers_per_process,
        process_index,
        worker_id,
    )
    np.random.seed(worker_seed % (2 ** 32))
    random.seed(worker_seed)
    torch.manual_seed(worker_seed % (2 ** 63 - 1))
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def build_dataloader(
    config,
    train_dataset=None,
    process_index=0,
    world_size=1,
    resume_step=0,
):
    train_dataset = train_dataset or build_dataset(config)
    worker_count = int(config.main.cpu_workers_num)
    batch_size = int(config.main.batch_size)
    windows_per_episode = int(
        getattr(train_dataset, "windows_per_episode", 1)
    )
    if batch_size % windows_per_episode != 0:
        raise ValueError(
            f"main.batch_size={batch_size} must be divisible by "
            "sampling.windows_per_episode="
            f"{windows_per_episode} so episode groups stay within one batch"
        )
    if int(process_index) == 0:
        logger.info(
            "Data pipeline: workers_per_rank=%d total_workers=%d "
            "video_cache_size=%d windows_per_episode=%d",
            worker_count,
            worker_count * max(1, int(world_size)),
            int(getattr(train_dataset, "video_cache_size", 0)),
            windows_per_episode,
        )
    workers_per_process = max(1, worker_count)
    worker_init_fn = partial(
        _initialize_dataloader_worker,
        base_seed=int(config.main.seed),
        global_step=int(resume_step),
        world_size=int(world_size),
        workers_per_process=workers_per_process,
        process_index=int(process_index),
    )
    ordinal_range = _sample_ordinal_range(config, resume_step, world_size)
    sampler = (
        ResumableOrdinalSampler(*ordinal_range)
        if ordinal_range is not None
        else None
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = False,
        sampler = sampler,
        num_workers = worker_count,
        pin_memory = True,
        drop_last = True,
        worker_init_fn = worker_init_fn,
        persistent_workers=worker_count > 0,
        prefetch_factor=1 if worker_count > 0 else None,
        multiprocessing_context="spawn" if worker_count > 0 else None,
    )

    return train_dataloader


_TEXT_EMBEDDING_CACHE_VERSION = 3
_TEXT_EMBEDDING_CACHE_ROOT = os.path.join(os.path.dirname(__file__), "data", "cache")


def _cache_filename_component(value, fallback):
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    component = component.strip("._-")
    return (component or fallback)[:96]


def _task_text_embedding_cache_path(cache_root, task_id, task_name):
    source, separator, _ = str(task_id).partition("::")
    if not separator:
        raise ValueError(f"Text embedding task_id has no source prefix: {task_id!r}")
    source_dir = _cache_filename_component(source, "unknown_source")
    task_stem = _cache_filename_component(task_name, "task")
    task_hash = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()[:12]
    return os.path.join(cache_root, source_dir, f"{task_stem}__{task_hash}.pt")


def _load_task_text_embedding(
    cache_path,
    task_id,
    task_name,
    instruction,
    feature_dim,
):
    if not os.path.isfile(cache_path):
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except Exception as error:
        logger.warning("Ignoring unreadable text embedding cache %s: %s", cache_path, error)
        return None
    if not isinstance(payload, dict):
        return None
    expected_source = str(task_id).split("::", 1)[0]
    if (
        payload.get("version") != _TEXT_EMBEDDING_CACHE_VERSION
        or payload.get("source") != expected_source
        or payload.get("task_id") != task_id
        or payload.get("task_name") != task_name
        or payload.get("instruction") != instruction
    ):
        return None
    embedding = payload.get("embedding")
    if not isinstance(embedding, torch.Tensor):
        return None
    if embedding.ndim != 2 or embedding.shape != (1, feature_dim):
        return None
    if not torch.isfinite(embedding).all():
        return None
    return embedding


def _load_text_embedding_cache(
    cache_root,
    task_instructions,
    task_cache_names,
    feature_dim,
):
    if set(task_instructions) != set(task_cache_names):
        raise ValueError("Text embedding task instructions and cache names do not match")
    embeddings = {}
    missing_task_ids = []
    for task_id in sorted(task_instructions):
        cache_path = _task_text_embedding_cache_path(
            cache_root, task_id, task_cache_names[task_id]
        )
        embedding = _load_task_text_embedding(
            cache_path,
            task_id,
            task_cache_names[task_id],
            task_instructions[task_id],
            feature_dim,
        )
        if embedding is None:
            missing_task_ids.append(task_id)
        else:
            embeddings[task_id] = embedding
    return embeddings, missing_task_ids


def _save_task_text_embedding(
    cache_root,
    task_id,
    task_name,
    instruction,
    embedding,
):
    cache_path = _task_text_embedding_cache_path(cache_root, task_id, task_name)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    temporary_path = f"{cache_path}.tmp-{os.getpid()}"
    torch.save(
        {
            "version": _TEXT_EMBEDDING_CACHE_VERSION,
            "source": str(task_id).split("::", 1)[0],
            "task_id": task_id,
            "task_name": task_name,
            "instruction": instruction,
            "embedding": embedding,
        },
        temporary_path,
    )
    os.replace(temporary_path, cache_path)


def prepare_text_embeddings(config, dataset, accelerator):
    task_instructions = dataset.task_instructions
    task_cache_names = dataset.task_cache_names
    feature_dim = int(config.model.get("text_feature_dim", 4096))
    cache_root = _TEXT_EMBEDDING_CACHE_ROOT
    embeddings, missing_task_ids = _load_text_embedding_cache(
        cache_root, task_instructions, task_cache_names, feature_dim
    )
    if missing_task_ids and accelerator.is_main_process:
        missing_instructions = [
            task_instructions[task_id] for task_id in missing_task_ids
        ]
        logger.info(
            "Precomputing %d missing natural-language embedding(s) below %s: %s",
            len(missing_task_ids),
            cache_root,
            missing_instructions,
        )
        from model.modules.llm2vec.llm2vec_wreapper import LLM2VecEncoder

        checkpoint_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "checkpoints")
        )
        encoder = LLM2VecEncoder(checkpoint_path=checkpoint_path).to(accelerator.device)
        encoder.eval()
        with torch.inference_mode():
            encoded, _ = encoder(missing_instructions)
        for index, task_id in enumerate(missing_task_ids):
            embedding = (
                encoded[index]
                .detach()
                .to(device="cpu", dtype=torch.bfloat16)
                .contiguous()
            )
            _save_task_text_embedding(
                cache_root,
                task_id,
                task_cache_names[task_id],
                task_instructions[task_id],
                embedding,
            )
        del encoder, encoded
        gc.collect()
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    embeddings, missing_task_ids = _load_text_embedding_cache(
        cache_root, task_instructions, task_cache_names, feature_dim
    )
    if missing_task_ids:
        missing_paths = [
            _task_text_embedding_cache_path(
                cache_root, task_id, task_cache_names[task_id]
            )
            for task_id in missing_task_ids
        ]
        raise RuntimeError(
            f"Text embedding cache is invalid after generation: {missing_paths}"
        )
    dataset.set_text_embeddings(embeddings)
    if accelerator.is_main_process:
        logger.info(
            "Loaded %d per-task text embedding cache file(s); training model will not load LLM2Vec",
            len(embeddings),
        )


_RESUME_CONFIG_FIELDS = (
    "main.batch_size",
    "main.max_steps",
    "main.action_chunk",
    "main.action_history",
    "main.seed",
    "main.dtype",
    "main.data_root",
    "main.dataset_roots",
    "main.dataset_selection",
    "main.sampling",
    "main.sample_stride",
    "main.precompute_text_embeddings",
    "main.precompute_motion_cache",
    "main.gradient.grad_clip_norm",
    "main.gradient.hand_grad_clip_norm",
    "main.gradient.grad_accumulation_steps",
    "model.fps",
    "model.motion_mask_mode",
    "model.dinov3_model_name",
    "model.dinov3_checkpoint",
    "model.text_feature_dim",
    "model.controlnet_num_layers",
    "model.detach_root_control_for_body",
    "model.enable_hand_head",
    "model.hand_hidden_dim",
    "model.hand_num_layers",
    "model.hand_num_heads",
    "model.hand_ffn_dim",
    "model.hand_init_seed",
    "training.loss",
    "training.optimizer",
    "training.scheduler",
)
_INIT_CHECKPOINT_CONFIG_FIELDS = (
    "main.action_chunk",
    "main.action_history",
    "model.fps",
    "model.motion_mask_mode",
    "model.dinov3_model_name",
    "model.dinov3_checkpoint",
    "model.text_feature_dim",
    "model.controlnet_num_layers",
    "model.enable_hand_head",
    "model.hand_hidden_dim",
    "model.hand_num_layers",
    "model.hand_num_heads",
    "model.hand_ffn_dim",
    "model.hand_init_seed",
)
_MISSING_CONFIG_VALUE = object()
_RNG_STATE_VERSION = 1


def _config_value(config, path):
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return _MISSING_CONFIG_VALUE
        value = value[key]
    return value


def _normalize_resume_config_value(path, value):
    """Fill newly introduced no-op defaults when comparing old checkpoints."""
    if path == "main.precompute_motion_cache":
        return False if value is _MISSING_CONFIG_VALUE else value
    if path == "model.enable_hand_head":
        return True if value is _MISSING_CONFIG_VALUE else value
    if path == "model.detach_root_control_for_body":
        return False if value is _MISSING_CONFIG_VALUE else value
    if (
        path != "training.loss"
        or value is _MISSING_CONFIG_VALUE
        or not isinstance(value, dict)
    ):
        return value
    normalized = dict(value)
    normalized.setdefault("motion_loss_type", "mse")
    configured_mse_weights = normalized.get("mse_weights", {})
    if isinstance(configured_mse_weights, dict):
        merged_mse_weights = dict(_DEFAULT_MSE_WEIGHT_CONFIG)
        for legacy_key, config_key in _LEGACY_MSE_WEIGHT_KEYS.items():
            if legacy_key in normalized and config_key not in configured_mse_weights:
                merged_mse_weights[config_key] = normalized[legacy_key]
            normalized.pop(legacy_key, None)
        merged_mse_weights.update(configured_mse_weights)
        normalized["mse_weights"] = merged_mse_weights
    configured_weights = normalized.get("kimodo_smooth_l1_weights", {})
    if isinstance(configured_weights, dict):
        merged_weights = dict(_DEFAULT_KIMODO_SMOOTH_L1_WEIGHT_CONFIG)
        merged_weights.update(configured_weights)
        legacy_fk_enabled = normalized.pop("kimodo_fk_loss_enabled", None)
        if legacy_fk_enabled is False and "fk" not in configured_weights:
            merged_weights["fk"] = 0.0
        normalized["kimodo_smooth_l1_weights"] = merged_weights
    return normalized


def _normalize_resume_checkpoint(checkpoint_path):
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint directory does not exist: {checkpoint_path}")
    for filename in ("training_state.pt", "config.json"):
        file_path = os.path.join(checkpoint_path, filename)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Resume checkpoint is missing {filename}: {file_path}")
    return checkpoint_path


def _normalize_init_checkpoint(checkpoint_path):
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(
            f"Initialization checkpoint directory does not exist: {checkpoint_path}"
        )
    for filename in ("training_state.pt", "config.json"):
        file_path = os.path.join(checkpoint_path, filename)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Initialization checkpoint is missing {filename}: {file_path}"
            )
    return checkpoint_path


def _validate_resume_config(config, checkpoint_path):
    with open(os.path.join(checkpoint_path, "config.json"), "r", encoding="utf-8") as file:
        checkpoint_config = json.load(file)
    current_config = OmegaConf.to_container(config, resolve=True)
    mismatches = []
    for path in _RESUME_CONFIG_FIELDS:
        current_value = _normalize_resume_config_value(
            path, _config_value(current_config, path)
        )
        checkpoint_value = _normalize_resume_config_value(
            path, _config_value(checkpoint_config, path)
        )
        if current_value != checkpoint_value:
            mismatches.append(
                f"{path}: current={current_value!r}, checkpoint={checkpoint_value!r}"
            )
    if mismatches:
        details = "\n  - ".join(mismatches)
        raise ValueError(
            "Resume config is incompatible with the checkpoint:\n"
            f"  - {details}"
        )


def _validate_init_checkpoint_config(config, checkpoint_path):
    """Require model compatibility while allowing a new dataset/training schedule."""
    with open(os.path.join(checkpoint_path, "config.json"), "r", encoding="utf-8") as file:
        checkpoint_config = json.load(file)
    current_config = OmegaConf.to_container(config, resolve=True)
    mismatches = []
    for path in _INIT_CHECKPOINT_CONFIG_FIELDS:
        current_value = _normalize_resume_config_value(
            path, _config_value(current_config, path)
        )
        checkpoint_value = _normalize_resume_config_value(
            path, _config_value(checkpoint_config, path)
        )
        if current_value != checkpoint_value:
            mismatches.append(
                f"{path}: current={current_value!r}, checkpoint={checkpoint_value!r}"
            )
    if mismatches:
        details = "\n  - ".join(mismatches)
        raise ValueError(
            "Initialization checkpoint model config is incompatible:\n"
            f"  - {details}"
        )


def _checkpoint_model_state(payload, checkpoint_path):
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(
            f"Checkpoint has no model state: {checkpoint_path}"
        )
    model_state = payload["model"]
    if not isinstance(model_state, dict):
        raise RuntimeError(
            f"Checkpoint model state is invalid: {checkpoint_path}"
        )
    return {
        name.removeprefix("_orig_mod."): value
        for name, value in model_state.items()
    }


def _validate_trainable_model_state(target_model, checkpoint_model, checkpoint_path):
    trainable_parameters = {
        name: parameter
        for name, parameter in target_model.named_parameters()
        if parameter.requires_grad
    }
    trainable_keys = set(trainable_parameters)
    checkpoint_keys = set(checkpoint_model)
    missing_trainable_keys = sorted(trainable_keys - checkpoint_keys)
    unexpected_model_keys = sorted(checkpoint_keys - trainable_keys)
    shape_mismatches = sorted(
        (
            name,
            tuple(checkpoint_model[name].shape),
            tuple(trainable_parameters[name].shape),
        )
        for name in trainable_keys & checkpoint_keys
        if tuple(checkpoint_model[name].shape)
        != tuple(trainable_parameters[name].shape)
    )
    if missing_trainable_keys or unexpected_model_keys or shape_mismatches:
        raise RuntimeError(
            f"Checkpoint trainable model state is incompatible ({checkpoint_path}): "
            f"missing={missing_trainable_keys}, unexpected={unexpected_model_keys}, "
            f"shape_mismatches={shape_mismatches}"
        )


def _load_model_initialization_checkpoint(model, checkpoint_path):
    """Load trainable weights only; deliberately leave optimizer/scheduler untouched."""
    state_path = os.path.join(checkpoint_path, "training_state.pt")
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    checkpoint_model = _checkpoint_model_state(payload, checkpoint_path)
    target_model = getattr(model, "_orig_mod", model)
    _validate_trainable_model_state(target_model, checkpoint_model, checkpoint_path)
    target_model.load_state_dict(checkpoint_model, strict=False)

    source_config = {}
    with open(os.path.join(checkpoint_path, "config.json"), "r", encoding="utf-8") as file:
        source_config = json.load(file)
    metadata = {
        "mode": "init_checkpoint",
        "source_checkpoint": os.path.abspath(checkpoint_path),
        "source_global_step": int(payload.get("global_step", -1)),
    }
    if payload.get("world_size") is not None:
        metadata["source_world_size"] = int(payload["world_size"])
    if source_config.get("initialization") is not None:
        metadata["source_initialization"] = source_config["initialization"]
    return metadata


def _load_training_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_path,
    expected_world_size=None,
):
    state_path = os.path.join(checkpoint_path, "training_state.pt")
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    required_keys = {"global_step", "model", "optimizer", "scheduler"}
    missing_payload_keys = sorted(required_keys - set(payload))
    if missing_payload_keys:
        raise RuntimeError(f"Checkpoint is missing training state: {missing_payload_keys}")

    global_step = int(payload["global_step"])
    if global_step < 0:
        raise RuntimeError(f"Checkpoint global_step must be non-negative, got {global_step}")
    checkpoint_name = os.path.basename(checkpoint_path)
    if checkpoint_name.startswith("checkpoint_"):
        try:
            directory_step = int(checkpoint_name.removeprefix("checkpoint_"))
        except ValueError:
            directory_step = None
        if directory_step is not None and directory_step != global_step:
            raise RuntimeError(
                f"Checkpoint directory step {directory_step} does not match payload step {global_step}"
            )
    checkpoint_world_size = payload.get("world_size")
    if (
        expected_world_size is not None
        and checkpoint_world_size is not None
        and int(checkpoint_world_size) != int(expected_world_size)
    ):
        raise RuntimeError(
            f"Checkpoint world_size={checkpoint_world_size} does not match current "
            f"world_size={expected_world_size}"
        )

    rng_state_version = payload.get("rng_state_version")
    if rng_state_version is not None:
        if int(rng_state_version) != _RNG_STATE_VERSION:
            raise RuntimeError(
                f"Unsupported checkpoint RNG state version {rng_state_version!r}"
            )
        rng_world_size = int(
            checkpoint_world_size
            if checkpoint_world_size is not None
            else expected_world_size or 1
        )
        missing_rng_files = [
            _rng_state_path(checkpoint_path, rank)
            for rank in range(rng_world_size)
            if not os.path.isfile(_rng_state_path(checkpoint_path, rank))
        ]
        if missing_rng_files:
            raise RuntimeError(
                "Checkpoint declares exact RNG state but is incomplete; missing "
                f"files: {missing_rng_files}"
            )

    data_state = payload.get("data_state")
    if data_state is not None:
        required_data_keys = {
            "sampling_version",
            "next_sample_ordinal",
            "batch_size",
            "gradient_accumulation_steps",
            "world_size",
        }
        missing_data_keys = sorted(required_data_keys - set(data_state))
        if missing_data_keys:
            raise RuntimeError(
                f"Checkpoint data state is incomplete: missing={missing_data_keys}"
            )
        if int(data_state["sampling_version"]) != 1:
            raise RuntimeError(
                "Unsupported checkpoint data sampling version "
                f"{data_state['sampling_version']!r}"
            )
        data_batch_size = int(data_state["batch_size"])
        data_accumulation = int(data_state["gradient_accumulation_steps"])
        data_world_size = int(data_state["world_size"])
        if min(data_batch_size, data_accumulation, data_world_size) <= 0:
            raise RuntimeError(f"Invalid checkpoint data state: {data_state}")
        expected_ordinal = (
            global_step
            * data_batch_size
            * data_accumulation
            * data_world_size
        )
        if int(data_state["next_sample_ordinal"]) != expected_ordinal:
            raise RuntimeError(
                "Checkpoint next_sample_ordinal="
                f"{data_state['next_sample_ordinal']} does not match global_step-derived "
                f"ordinal {expected_ordinal}"
            )
        if (
            checkpoint_world_size is not None
            and data_world_size != int(checkpoint_world_size)
        ):
            raise RuntimeError(
                f"Checkpoint data world_size={data_world_size} does not match "
                f"training world_size={checkpoint_world_size}"
            )

    target_model = getattr(model, "_orig_mod", model)
    checkpoint_model = _checkpoint_model_state(payload, checkpoint_path)
    _validate_trainable_model_state(target_model, checkpoint_model, checkpoint_path)
    target_model.load_state_dict(checkpoint_model, strict=False)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if int(scheduler.last_epoch) != global_step:
        raise RuntimeError(
            f"Scheduler last_epoch={scheduler.last_epoch} does not match global_step={global_step}"
        )
    return global_step


def _capture_rng_state(rank: int, world_size: int) -> dict:
    numpy_state = np.random.get_state()
    return {
        "version": _RNG_STATE_VERSION,
        "rank": int(rank),
        "world_size": int(world_size),
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": (
            torch.cuda.get_rng_state(torch.cuda.current_device()).cpu()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng_state(payload: dict, expected_rank=None, expected_world_size=None) -> None:
    if int(payload.get("version", -1)) != _RNG_STATE_VERSION:
        raise RuntimeError(
            f"Unsupported RNG checkpoint version {payload.get('version')!r}"
        )
    saved_rank = int(payload.get("rank", -1))
    saved_world_size = int(payload.get("world_size", -1))
    if expected_rank is not None and saved_rank != int(expected_rank):
        raise RuntimeError(
            f"RNG checkpoint rank={saved_rank} does not match current rank={expected_rank}"
        )
    if expected_world_size is not None and saved_world_size != int(expected_world_size):
        raise RuntimeError(
            "RNG checkpoint world_size="
            f"{saved_world_size} does not match current world_size={expected_world_size}"
        )

    numpy_state = payload.get("numpy")
    required_keys = {
        "python", "numpy", "torch_cpu", "torch_cuda"
    }
    missing = sorted(required_keys - set(payload))
    if missing or not isinstance(numpy_state, dict):
        raise RuntimeError(f"RNG checkpoint is incomplete: missing={missing}")

    random.setstate(payload["python"])
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            torch.as_tensor(numpy_state["keys"], dtype=torch.uint32).cpu().numpy(),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch.as_tensor(payload["torch_cpu"], dtype=torch.uint8).cpu())

    cuda_state = payload["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG state, but CUDA is unavailable"
            )
        torch.cuda.set_rng_state(
            torch.as_tensor(cuda_state, dtype=torch.uint8).cpu(),
            device=torch.cuda.current_device(),
        )


def _rng_state_path(checkpoint_path: str, rank: int) -> str:
    return os.path.join(checkpoint_path, f"rng_state_rank_{int(rank):05d}.pt")


def _save_rank_rng_state(
    checkpoint_path: str, rank: int, world_size: int
) -> None:
    state_path = _rng_state_path(checkpoint_path, rank)
    temporary_path = f"{state_path}.tmp.{os.getpid()}"
    torch.save(_capture_rng_state(rank, world_size), temporary_path)
    os.replace(temporary_path, state_path)


def _load_rank_rng_state(
    checkpoint_path: str, rank: int, expected_world_size: int
):
    state_path = _rng_state_path(checkpoint_path, rank)
    if not os.path.isfile(state_path):
        return None
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid RNG checkpoint payload: {state_path}")
    if int(payload.get("rank", -1)) != int(rank):
        raise RuntimeError(
            f"RNG checkpoint {state_path} belongs to rank {payload.get('rank')}, "
            f"not rank {rank}"
        )
    if int(payload.get("world_size", -1)) != int(expected_world_size):
        raise RuntimeError(
            f"RNG checkpoint world_size={payload.get('world_size')} does not match "
            f"current world_size={expected_world_size}"
        )
    return payload


def learning(config_path=None, resume=None, init_checkpoint=None):
    # 1. 加载配置参数 & 加载保存根目录
    config_path = config_path or os.path.join(os.path.dirname(__file__), "train.yaml")
    config = OmegaConf.load(config_path)
    if not os.path.isabs(config.main.save_root):
        config.main.save_root = os.path.join(os.path.dirname(__file__), config.main.save_root)
    os.makedirs(config.main.save_root, exist_ok=True)
    if resume is not None and init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    resume = _normalize_resume_checkpoint(resume) if resume is not None else None
    init_checkpoint = (
        _normalize_init_checkpoint(init_checkpoint)
        if init_checkpoint is not None
        else None
    )
    if resume:
        _validate_resume_config(config, resume)
    if init_checkpoint:
        _validate_init_checkpoint_config(config, init_checkpoint)
    # 2. 配置分布式
    distributed_timeout_seconds = int(
        config.main.get("distributed_timeout_seconds", 3600)
    )
    if config.main.get("precompute_motion_cache", False):
        # Rank 0 builds the cache before broadcasting its location.  Allow a
        # long one-time build without letting the other ranks time out.
        distributed_timeout_seconds = max(distributed_timeout_seconds, 21600)
    accelerator = Accelerator(
        gradient_accumulation_steps = config.main.gradient.grad_accumulation_steps,
        mixed_precision = config.main.dtype,
        step_scheduler_with_optimizer = False,
        project_dir = config.main.save_root,
        project_config = ProjectConfiguration(total_limit= 20),
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=distributed_timeout_seconds)),
                        DistributedDataParallelKwargs(
                            find_unused_parameters=False,
                            gradient_as_bucket_view=True,
                            static_graph=True,
                        )])
    set_seed(config.main.seed, device_specific=True)
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    if resume:
        save_path = os.path.dirname(resume)
    else:
        run_name = [
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if accelerator.is_main_process
            else None
        ]
        if dist.is_initialized():
            dist.broadcast_object_list(run_name, src=0)
        save_path = os.path.join(config.main.save_root, run_name[0])
    os.makedirs(save_path, exist_ok=True)
    setup_logging(rank, save_path)
    # 3. 加载数据
    train_dataset = build_dataset(config)
    prepare_pretrain_motion_cache(config, train_dataset, accelerator, config_path)
    if config.main.get("precompute_text_embeddings", True):
        prepare_text_embeddings(config, train_dataset, accelerator)
    # 4. 加载模型和优化器
    set_seed(config.main.seed, device_specific=False)
    model, optimizer, scheduler, max_training_steps = build_model_and_optimizer(config)
    initialization_metadata = None
    if init_checkpoint:
        initialization_metadata = _load_model_initialization_checkpoint(
            model, init_checkpoint
        )
        if rank == 0:
            logging.info(
                "Initialized trainable model weights from %s at source step %d; "
                "optimizer, scheduler, global step, data order, and RNG start fresh",
                initialization_metadata["source_checkpoint"],
                initialization_metadata["source_global_step"],
            )
    elif resume:
        with open(os.path.join(resume, "config.json"), "r", encoding="utf-8") as file:
            initialization_metadata = json.load(file).get("initialization")
    if initialization_metadata is not None:
        config.initialization = initialization_metadata
    set_seed(config.main.seed, device_specific=True)
    global_step = 0
    resume_rng_state = None
    if resume:
        global_step = _load_training_checkpoint(
            model,
            optimizer,
            scheduler,
            resume,
            expected_world_size=world_size,
        )
        if global_step >= max_training_steps:
            raise RuntimeError(
                f"Checkpoint step {global_step} has already reached max_steps={max_training_steps}"
            )
        resume_process_seed = _process_seed(config.main.seed, global_step, world_size)
        set_seed(resume_process_seed % (2 ** 32), device_specific=True)
        resume_rng_state = _load_rank_rng_state(
            resume, rank, expected_world_size=world_size
        )
        if rank == 0:
            logging.info(
                "Resumed model, optimizer, scheduler, and global step from %s at step %d",
                resume,
                global_step,
            )
            logging.info(
                "Loaded exact per-rank RNG state from checkpoint"
                if resume_rng_state is not None
                else (
                    "Checkpoint has no per-rank RNG state; using backward-compatible "
                    "step-offset seed (exact model RNG replay is unavailable)"
                ),
            )
    train_dataloader = build_dataloader(
        config,
        train_dataset,
        process_index=rank,
        world_size=world_size,
        resume_step=global_step,
    )
    # 5. 分布式分发（Accelerate 负责 dataloader shard）
    model, optimizer, scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader
    )
    model.train()
    # 7. 初始化 wandb (仅在主进程)
    if rank == 0:
        if config.main.wandb == "offline":
            wandb.init(
                project=config.main.get("wandb_project", "Kimodo-Policy"),
                name=config.main.get("wandb_run_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
                config=OmegaConf.to_container(config, resolve=True),
                dir=save_path,
                mode='offline'
            )
        else: 
            wandb.init(
                project=config.main.get("wandb_project", "Kimodo-Policy"),
                name=config.main.get("wandb_run_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
                config=OmegaConf.to_container(config, resolve=True),
                dir=save_path
            )
    torch.cuda.empty_cache()
    # 8. 训练
    if rank == 0:
        print("Start Training ...")
    initial_step = global_step
    start_time = time.time()
    data_iter = iter(train_dataloader)
    if resume_rng_state is not None:
        _restore_rng_state(
            resume_rng_state,
            expected_rank=rank,
            expected_world_size=world_size,
        )
    accumulated_loss = 0.0
    accumulated_motion_loss = 0.0
    accumulated_root_loss = 0.0
    accumulated_body_loss = 0.0
    accumulated_motion_component_losses = {}
    accumulated_hand_loss = 0.0
    accumulated_hand_state_loss = 0.0
    accumulated_hand_transition_loss = 0.0
    accumulation_count = 0
    while global_step < max_training_steps:
        ## 8.1 加载一个 batch
        data_start_time = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        data_time = time.perf_counter() - data_start_time
        ## 8.2 预处理 batch
        device = accelerator.device
        egoview   = batch["egoview"].to(device, non_blocking=True)
        gt_motion = batch["gt_motion"].to(device, non_blocking=True)
        condition_motion = batch.get("condition_motion")
        if condition_motion is not None:
            condition_motion = condition_motion.to(device, non_blocking=True)
        condition_motion_mask = batch.get("condition_motion_mask")
        if condition_motion_mask is not None:
            condition_motion_mask = condition_motion_mask.to(device, non_blocking=True)
        gt_hand   = batch["gt_hand"].to(device, non_blocking=True)
        gt_hand_mask = batch.get("gt_hand_mask")
        if gt_hand_mask is not None:
            gt_hand_mask = gt_hand_mask.to(device, non_blocking=True)
        gt_mask   = batch["gt_mask"].to(device, non_blocking=True)
        instruction = batch["instruction"]
        text_embedding = batch.get("text_embedding")
        text_length = batch.get("text_length")
        if text_embedding is not None:
            text_embedding = text_embedding.to(device, non_blocking=True)
        if text_length is not None:
            text_length = text_length.to(device, non_blocking=True)
        ## 8.3 前向传播 + 反向传播
        compute_start_time = time.perf_counter()
        control_grad_statistics = None
        hand_grad_statistics = None
        with accelerator.accumulate(model):
            loss_dict = model(
                instruction,
                egoview,
                gt_motion,
                gt_mask,
                condition_motion=condition_motion,
                condition_motion_mask=condition_motion_mask,
                gt_hand=gt_hand,
                gt_hand_mask=gt_hand_mask,
                text_feat=text_embedding,
                text_length=text_length,
            )
            loss = loss_dict["loss"]
            accumulated_loss += loss.detach()
            accumulated_motion_loss += loss_dict["motion_loss"].detach()
            accumulated_root_loss += loss_dict["root_loss"].detach()
            accumulated_body_loss += loss_dict["body_loss"].detach()
            for component_key in _MOTION_COMPONENT_LOSS_KEYS:
                if component_key in loss_dict:
                    accumulated_motion_component_losses[component_key] = (
                        accumulated_motion_component_losses.get(component_key, 0.0)
                        + loss_dict[component_key].detach()
                    )
            if "hand_loss" in loss_dict:
                accumulated_hand_loss += loss_dict["hand_loss"].detach()
                accumulated_hand_state_loss += loss_dict["hand_state_loss"].detach()
                accumulated_hand_transition_loss += loss_dict[
                    "hand_transition_loss"
                ].detach()
            accumulation_count += 1
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                named_trainable = [
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                ]
                hand_parameters = [
                    parameter
                    for name, parameter in named_trainable
                    if "hand_head." in name
                ]
                control_parameters = [
                    parameter
                    for name, parameter in named_trainable
                    if "hand_head." not in name
                ]
                if control_parameters:
                    control_clip_norm = float(
                        config.main.gradient.grad_clip_norm
                    )
                    control_total_norm = accelerator.clip_grad_norm_(
                        control_parameters, control_clip_norm
                    )
                    control_grad_statistics = _gradient_clip_statistics(
                        control_total_norm, control_clip_norm
                    )
                if hand_parameters:
                    hand_clip_norm = float(
                        config.main.gradient.get("hand_grad_clip_norm", 1.0)
                    )
                    hand_total_norm = accelerator.clip_grad_norm_(
                        hand_parameters,
                        hand_clip_norm,
                    )
                    hand_grad_statistics = _gradient_clip_statistics(
                        hand_total_norm, hand_clip_norm
                    )
            optimizer.step()
            optimizer.zero_grad()
            if accelerator.sync_gradients:
                scheduler.step()
                global_step += 1
        compute_time = time.perf_counter() - compute_start_time
        ## 8.4 记录日志
        if accelerator.sync_gradients:
            avg_loss = accumulated_loss / accumulation_count
            avg_motion_loss = accumulated_motion_loss / accumulation_count
            avg_root_loss = accumulated_root_loss / accumulation_count
            avg_body_loss = accumulated_body_loss / accumulation_count
            avg_motion_component_losses = {
                key: value / accumulation_count
                for key, value in accumulated_motion_component_losses.items()
            }
            has_hand_head = "hand_loss" in loss_dict
            avg_hand_loss = accumulated_hand_loss / accumulation_count
            avg_hand_state_loss = accumulated_hand_state_loss / accumulation_count
            avg_hand_transition_loss = (
                accumulated_hand_transition_loss / accumulation_count
            )
            if rank == 0:
                elapsed_time = time.time() - start_time
                completed_this_run = max(1, global_step - initial_step)
                avg_time_per_step = elapsed_time / completed_this_run
                eta_seconds = avg_time_per_step * (max_training_steps - global_step)
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                hand_log = (
                    f"Hand: {avg_hand_loss.item():.4f} | " if has_hand_head else ""
                )
                control_grad_log = (
                    f"Grad-Control: {control_grad_statistics[0]:.3f} | "
                    if control_grad_statistics is not None
                    else ""
                )
                hand_grad_log = (
                    f"Grad-Hand: {hand_grad_statistics[0]:.3f} | "
                    if hand_grad_statistics is not None
                    else ""
                )
                motion_component_log = "".join(
                    f"{key.removesuffix('_loss')}: {value.item():.4f} | "
                    for key, value in avg_motion_component_losses.items()
                )
                logging.info(
                    f"Step: {global_step}/{max_training_steps} | "
                    f"Loss: {avg_loss.item():.4f} | Motion: {avg_motion_loss.item():.4f} | "
                    f"Root: {avg_root_loss.item():.4f} | "
                    f"Body: {avg_body_loss.item():.4f} | "
                    f"{motion_component_log}"
                    f"{hand_log}"
                    f"{control_grad_log}"
                    f"{hand_grad_log}"
                    f"Data: {data_time:.2f}s | Compute: {compute_time:.2f}s | "
                    f"LR-Backbone: {scheduler.get_last_lr()[0]:.2e} | "
                    f"LR-Adapter: {scheduler.get_last_lr()[2]:.2e} | "
                    f"ETA: {eta_hours}h {eta_minutes}m"
                )
                metrics = {
                    "train/loss": avg_loss.item(),
                    "train/motion_loss": avg_motion_loss.item(),
                    "train/root_loss": avg_root_loss.item(),
                    "train/body_loss": avg_body_loss.item(),
                    "perf/data_time": data_time,
                    "perf/compute_time": compute_time,
                    "train/lr_control_backbone": scheduler.get_last_lr()[0],
                    "train/lr_control_adapter": scheduler.get_last_lr()[2],
                }
                metrics.update(
                    {
                        f"train/{key}": value.item()
                        for key, value in avg_motion_component_losses.items()
                    }
                )
                if control_grad_statistics is not None:
                    metrics.update(
                        {
                            "train/control_grad_norm": control_grad_statistics[0],
                            "train/control_was_clipped": control_grad_statistics[1],
                            "train/control_clip_scale": control_grad_statistics[2],
                        }
                    )
                if has_hand_head:
                    metrics.update(
                        {
                            "train/hand_loss": avg_hand_loss.item(),
                            "train/hand_state_loss": avg_hand_state_loss.item(),
                            "train/hand_transition_loss": avg_hand_transition_loss.item(),
                            "train/lr_hand": scheduler.get_last_lr()[4],
                        }
                    )
                if hand_grad_statistics is not None:
                    metrics.update(
                        {
                            "train/hand_grad_norm": hand_grad_statistics[0],
                            "train/hand_was_clipped": hand_grad_statistics[1],
                            "train/hand_clip_scale": hand_grad_statistics[2],
                        }
                    )
                wandb.log(metrics, step=global_step)
            accumulated_loss = 0.0
            accumulated_motion_loss = 0.0
            accumulated_root_loss = 0.0
            accumulated_body_loss = 0.0
            accumulated_motion_component_losses = {}
            accumulated_hand_loss = 0.0
            accumulated_hand_state_loss = 0.0
            accumulated_hand_transition_loss = 0.0
            accumulation_count = 0
        ## 8.5 按步数保存 checkpoint
        if global_step > 0 and global_step % config.main.save_steps == 0 and accelerator.sync_gradients:
            if rank == 0:
                logging.info(f"Saving checkpoint at step {global_step}...")
            checkpoint_dir = os.path.join(save_path, f"checkpoint_{global_step}")
            if accelerator.is_main_process:
                os.makedirs(checkpoint_dir, exist_ok=True)
            accelerator.wait_for_everyone()
            _save_rank_rng_state(checkpoint_dir, rank, world_size)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                trainable_state = {
                    name.removeprefix("_orig_mod."): parameter.detach().cpu()
                    for name, parameter in unwrapped_model.named_parameters()
                    if parameter.requires_grad
                }
                state_path = os.path.join(checkpoint_dir, "training_state.pt")
                temporary_state_path = f"{state_path}.tmp"
                checkpoint_payload = {
                    "global_step": global_step,
                    "model": trainable_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "world_size": world_size,
                    "rng_state_version": _RNG_STATE_VERSION,
                    "data_state": {
                        "sampling_version": 1,
                        "next_sample_ordinal": global_step
                        * _samples_per_optimizer_step(config, world_size),
                        "batch_size": int(config.main.batch_size),
                        "gradient_accumulation_steps": int(
                            config.main.gradient.grad_accumulation_steps
                        ),
                        "world_size": world_size,
                    },
                    "seed_metadata": {
                        "base_seed": int(config.main.seed),
                        "global_step": global_step,
                        "workers_per_process": int(config.main.cpu_workers_num),
                    },
                }
                if initialization_metadata is not None:
                    checkpoint_payload["initialization"] = initialization_metadata
                accelerator.save(
                    checkpoint_payload,
                    temporary_state_path,
                )
                os.replace(temporary_state_path, state_path)
                config_path = os.path.join(checkpoint_dir, "config.json")
                temporary_config_path = f"{config_path}.tmp"
                cfg = OmegaConf.to_container(config, resolve=True)
                if initialization_metadata is not None:
                    cfg["initialization"] = initialization_metadata
                with open(temporary_config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
                os.replace(temporary_config_path, config_path)
            accelerator.wait_for_everyone()

    # 9. 关闭 wandb
    if rank == 0:
        wandb.finish()
    accelerator.end_training()



def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to a training YAML file")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        default=None,
        help="Resume the exact same run, including optimizer/scheduler/RNG/data state",
    )
    checkpoint_group.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize model weights only for a fresh fine-tuning run",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    learning(args.config, args.resume, args.init_checkpoint)
