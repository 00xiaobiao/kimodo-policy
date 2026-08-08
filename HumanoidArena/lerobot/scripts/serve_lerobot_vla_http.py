#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import random
import signal
import ssl
import sys
import tempfile
import threading
import traceback
from contextlib import nullcontext
from copy import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch


_SERVER_STOP_REASON = "unknown"
_LEGACY_CHECKPOINT_ROOT = Path("/mnt/workspace/users/xujunzhe/yunhengwang/lerobot/lerobot/checkpoints")
_COMPAT_CHECKPOINT_ROOT = Path("/ai/Yichi/taowen/ckpts/checkpoints")

TASK_LANGUAGE_INSTRUCTIONS = {
    "HOI_double_desk": "Put the hammer from the right table into the basket on the left table.",
    "HOI_football": "Kick the soccer ball into the goal.",
    "HOI_pp_box": "Move the box from the table onto the shelf.",
    "HSI_vision_navi": "Avoid obstacles and move to the yellow marked area.",
    "HSI_open_door": "Open the door.",
    "HSI_sit_sofa": "Sit on the sofa.",
    "HOI_grap_cup": "Place the orange drink can on the table into the basket on the tea table.",
    "HSI_boxing": "Strike the green markers on the punching bag.",
}

_TASK_NAME_ALIASES = (
    ("hoidoubledesk", "HOI_double_desk"),
    ("doubledesk", "HOI_double_desk"),
    ("double_desk", "HOI_double_desk"),
    ("hoifootball", "HOI_football"),
    ("football", "HOI_football"),
    ("hoippbox", "HOI_pp_box"),
    ("ppbox", "HOI_pp_box"),
    ("pp_box", "HOI_pp_box"),
    ("hsivisionnavi", "HSI_vision_navi"),
    ("visionnavi", "HSI_vision_navi"),
    ("vision_navi", "HSI_vision_navi"),
    ("hsiopendoor", "HSI_open_door"),
    ("opendoor", "HSI_open_door"),
    ("open_door", "HSI_open_door"),
    ("hsisitsofa", "HSI_sit_sofa"),
    ("sitsofa", "HSI_sit_sofa"),
    ("sit_sofa", "HSI_sit_sofa"),
    ("hoigrapcup", "HOI_grap_cup"),
    ("hoigrabcup", "HOI_grap_cup"),
    ("grapcup", "HOI_grap_cup"),
    ("grabcup", "HOI_grap_cup"),
    ("grap_cup", "HOI_grap_cup"),
    ("grab_cup", "HOI_grap_cup"),
    ("hsiboxing", "HSI_boxing"),
    ("boxing", "HSI_boxing"),
)

_HTTP_INFERENCE_IMAGE_TRANSFORM_TYPES = {"Resize", "Pad"}


def _remap_legacy_checkpoint_ref(value):
    if not isinstance(value, str):
        return value, False

    try:
        path = Path(value)
    except Exception:
        return value, False

    if not path.is_absolute():
        return value, False
    if path.exists():
        return value, False

    try:
        path.relative_to(_LEGACY_CHECKPOINT_ROOT)
    except Exception:
        return value, False

    compat_candidate = _COMPAT_CHECKPOINT_ROOT / path.name
    if compat_candidate.exists():
        print(
            f"[lerobot_vla_server] remap legacy checkpoint ref {value} -> {compat_candidate}",
            flush=True,
        )
        return str(compat_candidate), True

    print(
        f"[lerobot_vla_server] unresolved legacy checkpoint ref {value}; compat candidate missing: {compat_candidate}",
        flush=True,
    )
    return value, False


def _remap_json_tree(obj):
    if isinstance(obj, dict):
        changed = False
        remapped = {}
        for key, value in obj.items():
            remapped_value, child_changed = _remap_json_tree(value)
            remapped[key] = remapped_value
            changed = changed or child_changed
        return remapped, changed

    if isinstance(obj, list):
        changed = False
        remapped = []
        for value in obj:
            remapped_value, child_changed = _remap_json_tree(value)
            remapped.append(remapped_value)
            changed = changed or child_changed
        return remapped, changed

    return _remap_legacy_checkpoint_ref(obj)


def _prepare_compat_policy_dir(policy_dir: Path):
    temp_dir = tempfile.TemporaryDirectory(prefix="lerobot_policy_compat_")
    compat_dir = Path(temp_dir.name)

    for child in policy_dir.iterdir():
        target = compat_dir / child.name
        target.symlink_to(child, target_is_directory=child.is_dir())

    changed_files = []
    for json_path in sorted(policy_dir.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text())
        except Exception:
            continue

        remapped_payload, changed = _remap_json_tree(payload)
        if not changed:
            continue

        compat_json_path = compat_dir / json_path.name
        if compat_json_path.exists() or compat_json_path.is_symlink():
            compat_json_path.unlink()
        compat_json_path.write_text(json.dumps(remapped_payload, indent=2) + "\n")
        changed_files.append(json_path.name)

    if changed_files:
        print(
            "[lerobot_vla_server] prepared compat policy dir for "
            f"{policy_dir} with remapped files: " + ", ".join(changed_files),
            flush=True,
        )
    else:
        print(f"[lerobot_vla_server] compat policy dir not needed for {policy_dir}", flush=True)

    return temp_dir, compat_dir


def _interrupt_handler(signum, frame):
    global _SERVER_STOP_REASON
    _SERVER_STOP_REASON = signal.Signals(signum).name
    raise KeyboardInterrupt


def _install_interrupt_handlers():
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _interrupt_handler)


def _feature_shape_dim(feature) -> tuple[int, ...] | None:
    if feature is None:
        return None
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    if shape is None:
        return None
    return tuple(int(v) for v in shape)


def _normalize_task_lookup_key(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _canonical_task_name(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw in TASK_LANGUAGE_INSTRUCTIONS:
        return raw

    normalized = _normalize_task_lookup_key(raw)
    for task_name, instruction in TASK_LANGUAGE_INSTRUCTIONS.items():
        if normalized == _normalize_task_lookup_key(instruction):
            return task_name
    for alias, task_name in _TASK_NAME_ALIASES:
        if _normalize_task_lookup_key(alias) in normalized:
            return task_name
    return None


def _resolve_task_instruction(value: str | None) -> tuple[str | None, str | None]:
    task_name = _canonical_task_name(value)
    if task_name is not None:
        return task_name, TASK_LANGUAGE_INSTRUCTIONS[task_name]
    if value is None:
        return None, None
    raw = str(value).strip()
    return (None, raw) if raw else (None, None)


def _infer_task_instruction_from_policy_path(policy_dir: Path) -> tuple[str | None, str | None]:
    return _resolve_task_instruction(str(policy_dir))


class ServerImageTransform:
    def __init__(self, transform_specs: list[dict], summary: str):
        self.transform_specs = transform_specs
        self.summary = summary

    @staticmethod
    def _apply_resize(tensor: torch.Tensor, kwargs: dict) -> torch.Tensor:
        import torch.nn.functional as F

        size = kwargs.get("size")
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError(f"Resize requires size=[height, width], got {size}")
        height, width = (int(size[0]), int(size[1]))
        mode = str(kwargs.get("interpolation", "bilinear"))
        interpolate_kwargs = {"size": (height, width), "mode": mode}
        if mode in {"bilinear", "bicubic"}:
            interpolate_kwargs["align_corners"] = False
            interpolate_kwargs["antialias"] = True
        try:
            return F.interpolate(tensor.unsqueeze(0), **interpolate_kwargs).squeeze(0)
        except TypeError:
            interpolate_kwargs.pop("antialias", None)
            return F.interpolate(tensor.unsqueeze(0), **interpolate_kwargs).squeeze(0)

    @staticmethod
    def _apply_pad(tensor: torch.Tensor, kwargs: dict) -> torch.Tensor:
        import torch.nn.functional as F

        padding = kwargs.get("padding")
        if isinstance(padding, int):
            left = top = right = bottom = int(padding)
        elif isinstance(padding, (list, tuple)) and len(padding) == 2:
            left = right = int(padding[0])
            top = bottom = int(padding[1])
        elif isinstance(padding, (list, tuple)) and len(padding) == 4:
            left, top, right, bottom = (int(v) for v in padding)
        else:
            raise ValueError(f"Pad requires int, [x, y], or [left, top, right, bottom], got {padding}")
        fill = float(kwargs.get("fill", 0))
        return F.pad(tensor, (left, right, top, bottom), mode="constant", value=fill)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(f"Expected HWC image with 3 or 4 channels, got {image.shape}")

        rgb = np.ascontiguousarray(image[..., :3])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32) / 255.0
        with torch.no_grad():
            for spec in self.transform_specs:
                tf_type = spec["type"]
                kwargs = spec.get("kwargs", {})
                if tf_type == "Resize":
                    tensor = self._apply_resize(tensor, kwargs)
                elif tf_type == "Pad":
                    tensor = self._apply_pad(tensor, kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP image transform type: {tf_type}")

        if tensor.ndim != 3 or tensor.shape[0] not in (1, 3):
            raise ValueError(f"Image transform returned unexpected shape {tuple(tensor.shape)}")

        transformed = tensor.detach().cpu().clamp(0.0, 1.0)
        transformed = transformed.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()
        if transformed.shape[-1] == 1:
            transformed = np.repeat(transformed, 3, axis=-1)
        return transformed


def _load_server_image_transform(policy_dir: Path) -> ServerImageTransform | None:
    train_config_path = policy_dir / "train_config.json"
    if not train_config_path.is_file():
        print(
            f"[lerobot_vla_server] no train_config.json in {policy_dir}; http image transform disabled",
            flush=True,
        )
        return None

    try:
        train_config = json.loads(train_config_path.read_text())
    except Exception as exc:
        print(
            f"[lerobot_vla_server] failed to read {train_config_path}; http image transform disabled: {exc}",
            flush=True,
        )
        return None

    image_transform_cfg = train_config.get("dataset", {}).get("image_transforms", {})
    if not bool(image_transform_cfg.get("enable", False)):
        print("[lerobot_vla_server] http image transform disabled by train_config", flush=True)
        return None

    tfs_cfg = image_transform_cfg.get("tfs") or {}
    if not tfs_cfg:
        print("[lerobot_vla_server] empty image transform config; http image transform disabled", flush=True)
        return None

    max_num_transforms = int(image_transform_cfg.get("max_num_transforms", len(tfs_cfg)))
    random_order = bool(image_transform_cfg.get("random_order", False))

    transform_specs = []
    active_tfs_cfg = {}
    skipped_transform_types = []
    for name, tf_cfg in tfs_cfg.items():
        weight = float(tf_cfg.get("weight", 1.0))
        if weight <= 0.0:
            continue
        tf_type = str(tf_cfg.get("type", ""))
        if tf_type not in _HTTP_INFERENCE_IMAGE_TRANSFORM_TYPES:
            skipped_transform_types.append(f"{name}:{tf_type}")
            continue
        transform_specs.append({"name": name, "type": tf_type, "kwargs": dict(tf_cfg.get("kwargs", {}))})
        active_tfs_cfg[name] = tf_cfg

    if skipped_transform_types:
        print(
            "[lerobot_vla_server] skipped train-only image transforms for HTTP inference: "
            + ", ".join(skipped_transform_types),
            flush=True,
        )
    if not transform_specs:
        print("[lerobot_vla_server] no deterministic HTTP image transforms; http image transform disabled", flush=True)
        return None
    if random_order:
        raise ValueError("HTTP image transform only supports random_order=false for deterministic inference")
    if max_num_transforms < len(transform_specs):
        raise ValueError("HTTP image transform requires max_num_transforms to include all configured transforms")

    summary = json.dumps(
        {
            "max_num_transforms": max_num_transforms,
            "random_order": random_order,
            "order": [spec["name"] for spec in transform_specs],
            "tfs": active_tfs_cfg,
            "skipped_train_only_transforms": skipped_transform_types,
        },
    )
    print(f"[lerobot_vla_server] http image transform enabled: {summary}", flush=True)
    return ServerImageTransform(transform_specs=transform_specs, summary=summary)


def _load_policy(policy_dir: Path, device_name: str):
    lerobot_src = Path(__file__).resolve().parents[1] / "src"
    if not lerobot_src.is_dir():
        raise FileNotFoundError(f"LeRobot src directory not found: {lerobot_src}")
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import _reconnect_relative_absolute_steps, get_policy_class
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    from lerobot.utils.control_utils import predict_action

    compat_dir_ctx, effective_policy_dir = _prepare_compat_policy_dir(policy_dir)

    config = PreTrainedConfig.from_pretrained(effective_policy_dir)
    config.device = device_name
    policy_cls = get_policy_class(config.type)

    policy = policy_cls.from_pretrained(effective_policy_dir, config=config)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        effective_policy_dir,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        effective_policy_dir,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    _reconnect_relative_absolute_steps(preprocessor, postprocessor)
    return (
        config,
        policy,
        preprocessor,
        postprocessor,
        predict_action,
        prepare_observation_for_inference,
        compat_dir_ctx,
    )


class LeRobotServerState:
    def __init__(self, policy_dir: Path, device_name: str):
        (
            self.config,
            self.policy,
            self.preprocessor,
            self.postprocessor,
            self.predict_action,
            self.prepare_observation_for_inference,
            self._compat_policy_dir_ctx,
        ) = _load_policy(policy_dir, device_name)
        self.expected_state_shape = _feature_shape_dim(self.config.input_features.get("observation.state"))
        self.expected_action_shape = _feature_shape_dim(self.config.output_features.get("action"))
        self.http_image_transform = _load_server_image_transform(policy_dir)
        self.default_task_name, self.default_task_instruction = _infer_task_instruction_from_policy_path(policy_dir)
        if self.default_task_instruction:
            print(
                f"[lerobot_vla_server] default task route {self.default_task_name}: "
                f"{self.default_task_instruction}",
                flush=True,
            )
        else:
            print(
                f"[lerobot_vla_server] no default task route inferred from policy path: {policy_dir}",
                flush=True,
            )
        self.device = torch.device(device_name)
        self._use_amp = bool(getattr(self.config, "use_amp", False))
        self._amp_dtype = self._resolve_amp_dtype()
        amp_label = "off" if not self._use_amp or self.device.type != "cuda" else str(self._amp_dtype or "default")
        config_type = getattr(self.config, "type", "unknown")
        config_dtype = getattr(self.config, "dtype", None)
        print(
            "[lerobot_vla_server] precision policy type={} config_dtype={} use_amp={} amp={}".format(
                config_type, config_dtype, self._use_amp, amp_label
            ),
            flush=True,
        )
        self.lock = threading.Lock()
        self.reset_count = 0
        self.infer_count = 0
        self.current_seed: int | None = None
        self.reset()

    def _seed_runtime(self, seed: int) -> None:
        normalized_seed = int(seed) & 0xFFFFFFFF
        random.seed(normalized_seed)
        np.random.seed(normalized_seed)
        torch.manual_seed(normalized_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(normalized_seed)

    def reset(self, seed: int | None = None):
        with self.lock:
            if seed is not None:
                self.current_seed = int(seed) & 0xFFFFFFFFFFFFFFFF
                self._seed_runtime(self.current_seed)
            self.policy.reset()
            self.preprocessor.reset()
            self.postprocessor.reset()
            self.reset_count += 1
            self.infer_count = 0
            return self.reset_count

    def _resolve_amp_dtype(self):
        dtype_name = str(getattr(self.config, "dtype", "") or "").strip().lower()
        if dtype_name in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype_name in {"float16", "fp16", "half"}:
            return torch.float16
        if dtype_name in {"float32", "fp32", "float"}:
            return torch.float32
        return None

    def _amp_context(self):
        if self.device.type != "cuda" or not self._use_amp:
            return nullcontext()
        if self._amp_dtype is torch.float32:
            return nullcontext()
        if self._amp_dtype is None:
            return torch.autocast(device_type=self.device.type)
        return torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)

    def _get_action_stats(self) -> dict[str, torch.Tensor]:
        for step in getattr(self.postprocessor, "steps", []):
            tensor_stats = getattr(step, "_tensor_stats", None)
            if isinstance(tensor_stats, dict) and "action" in tensor_stats:
                return tensor_stats["action"]
        return {}

    def _get_state_stats(self) -> dict[str, torch.Tensor]:
        for step in getattr(self.preprocessor, "steps", []):
            tensor_stats = getattr(step, "_tensor_stats", None)
            if isinstance(tensor_stats, dict) and "observation.state" in tensor_stats:
                return tensor_stats["observation.state"]
        return {}

    def _state_dim_name(self, dim: int) -> str:
        feature = self.config.input_features.get("observation.state")
        names = getattr(feature, "names", None)
        if names is None and isinstance(feature, dict):
            names = feature.get("names")
        if isinstance(names, list) and 0 <= dim < len(names):
            return str(names[dim])
        return f"dim{dim}"

    def _action_dim_name(self, dim: int) -> str:
        names = getattr(self.config, "action_feature_names", None)
        if isinstance(names, list) and 0 <= dim < len(names):
            return str(names[dim])
        return f"dim{dim}"

    def _should_log_action_debug(self) -> bool:
        infer_index = self.infer_count + 1
        return infer_index <= 5 or infer_index % 20 == 0

    def _log_normalized_action_chunk(self, action_chunk: torch.Tensor) -> None:
        if not self._should_log_action_debug():
            return
        arr = action_chunk.detach().cpu().to(torch.float32)
        flat = arr.reshape(-1, arr.shape[-1])
        over = flat.abs() > 1.0
        ratios = over.float().mean(dim=0)
        top_count = min(8, flat.shape[-1])
        top_vals, top_idx = torch.topk(ratios, k=top_count)
        parts = []
        for ratio, dim_tensor in zip(top_vals.tolist(), top_idx.tolist()):
            if ratio <= 0:
                continue
            dim = int(dim_tensor)
            dim_vals = flat[:, dim]
            parts.append(
                f"d{dim}:{self._action_dim_name(dim)} "
                f"over={ratio * 100:.1f}% norm_rng=({dim_vals.min().item():+.2f},{dim_vals.max().item():+.2f})"
            )
        print(
            f"[lerobot_vla_server][action_debug] infer={self.infer_count + 1} "
            f"normalized_chunk shape={tuple(arr.shape)} "
            f"mean_abs={flat.abs().mean().item():.3f} max_abs={flat.abs().max().item():.3f} "
            f"first={flat[0].tolist()}",
            flush=True,
        )
        if parts:
            print(
                f"[lerobot_vla_server][action_debug] infer={self.infer_count + 1} "
                "normalized_outside_unit_top " + " | ".join(parts),
                flush=True,
            )

    def _log_raw_state(self, state_tensor: torch.Tensor) -> None:
        if not self._should_log_action_debug():
            return
        arr = state_tensor.detach().cpu().to(torch.float32).reshape(-1)
        stats = self._get_state_stats()
        print(
            f"[lerobot_vla_server][state_debug] infer={self.infer_count + 1} "
            f"raw_state shape={tuple(state_tensor.shape)} mean={arr.mean().item():+.4f} "
            f"std={arr.std(unbiased=False).item():.4f} min={arr.min().item():+.4f} max={arr.max().item():+.4f} "
            f"first={arr.tolist()}",
            flush=True,
        )
        q01 = stats.get("q01")
        q99 = stats.get("q99")
        min_val = stats.get("min")
        max_val = stats.get("max")
        if q01 is None or q99 is None:
            return
        q01 = q01.detach().cpu().to(torch.float32).reshape(-1)
        q99 = q99.detach().cpu().to(torch.float32).reshape(-1)
        q_out = (arr < q01) | (arr > q99)
        if min_val is not None and max_val is not None:
            min_val = min_val.detach().cpu().to(torch.float32).reshape(-1)
            max_val = max_val.detach().cpu().to(torch.float32).reshape(-1)
            mm_out = (arr < min_val) | (arr > max_val)
            rank = q_out.to(torch.float32) + mm_out.to(torch.float32) * 2.0
        else:
            mm_out = torch.zeros_like(q_out)
            rank = q_out.to(torch.float32)
        denom = torch.where((q99 - q01).abs() < 1e-8, torch.ones_like(q99), q99 - q01)
        norm = 2.0 * (arr - q01) / denom - 1.0
        top_count = min(10, arr.numel())
        _, top_idx = torch.topk(rank, k=top_count)
        parts = []
        for dim_tensor in top_idx.tolist():
            dim = int(dim_tensor)
            if not bool(q_out[dim].item()) and not bool(mm_out[dim].item()):
                continue
            parts.append(
                f"d{dim}:{self._state_dim_name(dim)} raw={arr[dim].item():+.4f} "
                f"q01={q01[dim].item():+.4f} q99={q99[dim].item():+.4f} "
                f"min={min_val[dim].item():+.4f} max={max_val[dim].item():+.4f} "
                f"norm={norm[dim].item():+.2f}"
            )
        if parts:
            print(
                f"[lerobot_vla_server][state_debug] infer={self.infer_count + 1} "
                "raw_state_outside_stats_top " + " | ".join(parts),
                flush=True,
            )

    def _log_processed_state(self, processed_observation: dict) -> None:
        if not self._should_log_action_debug():
            return
        state = processed_observation.get("observation.state")
        if state is None:
            return
        arr = state.detach().cpu().to(torch.float32).reshape(-1)
        bins = np.digitize(arr.numpy(), bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        over = arr.abs() > 1.0
        print(
            f"[lerobot_vla_server][state_debug] infer={self.infer_count + 1} "
            f"normalized_state shape={tuple(state.shape)} mean={arr.mean().item():+.4f} "
            f"std={arr.std(unbiased=False).item():.4f} min={arr.min().item():+.4f} max={arr.max().item():+.4f} "
            f"outside_unit={over.float().mean().item() * 100:.1f}% bin_min={int(bins.min())} bin_max={int(bins.max())} "
            f"first={arr.tolist()}",
            flush=True,
        )
        if bool(over.any().item()):
            vals, idx = torch.topk(arr.abs(), k=min(10, arr.numel()))
            parts = []
            for _, dim_tensor in zip(vals.tolist(), idx.tolist()):
                dim = int(dim_tensor)
                if not bool(over[dim].item()):
                    continue
                parts.append(
                    f"d{dim}:{self._state_dim_name(dim)} norm={arr[dim].item():+.2f} bin={int(bins[dim])}"
                )
            if parts:
                print(
                    f"[lerobot_vla_server][state_debug] infer={self.infer_count + 1} "
                    "normalized_state_outside_unit_top " + " | ".join(parts),
                    flush=True,
                )

    def _log_raw_action_chunk(self, action_chunk: torch.Tensor) -> None:
        if not self._should_log_action_debug():
            return
        arr = action_chunk.detach().cpu().to(torch.float32)
        flat = arr.reshape(-1, arr.shape[-1])
        stats = self._get_action_stats()
        q01 = stats.get("q01")
        q99 = stats.get("q99")
        min_val = stats.get("min")
        max_val = stats.get("max")
        print(
            f"[lerobot_vla_server][action_debug] infer={self.infer_count + 1} "
            f"raw_chunk shape={tuple(arr.shape)} "
            f"mean={flat.mean().item():+.4f} std={flat.std(unbiased=False).item():.4f} "
            f"min={flat.min().item():+.4f} max={flat.max().item():+.4f} "
            f"first={flat[0].tolist()}",
            flush=True,
        )
        if q01 is None or q99 is None:
            return
        q01 = q01.detach().cpu().to(torch.float32).reshape(-1)
        q99 = q99.detach().cpu().to(torch.float32).reshape(-1)
        q_over = ((flat < q01) | (flat > q99)).float().mean(dim=0)
        if min_val is not None and max_val is not None:
            min_val = min_val.detach().cpu().to(torch.float32).reshape(-1)
            max_val = max_val.detach().cpu().to(torch.float32).reshape(-1)
            mm_over = ((flat < min_val) | (flat > max_val)).float().mean(dim=0)
            rank = mm_over * 2.0 + q_over
        else:
            mm_over = torch.zeros_like(q_over)
            rank = q_over
        top_count = min(8, flat.shape[-1])
        _, top_idx = torch.topk(rank, k=top_count)
        parts = []
        denom = torch.where((q99 - q01).abs() < 1e-8, torch.ones_like(q99), q99 - q01)
        norm_flat = 2.0 * (flat - q01) / denom - 1.0
        for dim_tensor in top_idx.tolist():
            dim = int(dim_tensor)
            if q_over[dim].item() <= 0 and mm_over[dim].item() <= 0:
                continue
            dim_vals = flat[:, dim]
            dim_norm = norm_flat[:, dim]
            parts.append(
                f"d{dim}:{self._action_dim_name(dim)} "
                f"q={q_over[dim].item() * 100:.1f}% mm={mm_over[dim].item() * 100:.1f}% "
                f"raw_rng=({dim_vals.min().item():+.3f},{dim_vals.max().item():+.3f}) "
                f"norm_rng=({dim_norm.min().item():+.2f},{dim_norm.max().item():+.2f})"
            )
        if parts:
            print(
                f"[lerobot_vla_server][action_debug] infer={self.infer_count + 1} "
                "raw_outside_stats_top " + " | ".join(parts),
                flush=True,
            )

    def resolve_task_instruction(self, task_value: str | None) -> tuple[str | None, str | None]:
        task_name, instruction = _resolve_task_instruction(task_value)
        if instruction:
            return task_name, instruction
        return self.default_task_name, self.default_task_instruction

    def _prepare_observation(self, observation: dict, robot_type: str, task: str | None):
        prepared = self.prepare_observation_for_inference(copy(observation), self.device, task, robot_type)
        raw_state = prepared.get("observation.state")
        if raw_state is not None:
            self._log_raw_state(raw_state)
        processed = self.preprocessor(prepared)
        self._log_processed_state(processed)
        return processed

    def infer(self, observation: dict, robot_type: str, task: str | None):
        with self.lock:
            with torch.inference_mode(), self._amp_context():
                processed_observation = self._prepare_observation(observation, robot_type, task)
                action = self.policy.select_action(processed_observation)
                return self.postprocessor(action)

    def infer_chunk(self, observation: dict, robot_type: str, task: str | None) -> np.ndarray:
        with self.lock:
            with torch.inference_mode(), self._amp_context():
                processed_observation = self._prepare_observation(observation, robot_type, task)
                normalized_action_chunk = self.policy.predict_action_chunk(processed_observation)
                if normalized_action_chunk.ndim != 3:
                    normalized_action_chunk = normalized_action_chunk.unsqueeze(0)
                self._log_normalized_action_chunk(normalized_action_chunk)

                processed_actions = []
                _, chunk_size, _ = normalized_action_chunk.shape
                for i in range(chunk_size):
                    processed_actions.append(self.postprocessor(normalized_action_chunk[:, i, :]))

                raw_action_chunk = torch.stack(processed_actions, dim=1).squeeze(0)
                self._log_raw_action_chunk(raw_action_chunk)
                return raw_action_chunk.detach().cpu().to(torch.float32).numpy()


def _decode_image(image_payload: dict) -> np.ndarray:
    shape = tuple(int(v) for v in image_payload["shape"])
    dtype = np.dtype(image_payload["dtype"])
    raw = base64.b64decode(image_payload["data_b64"].encode("ascii"))
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def make_handler(state: LeRobotServerState):
    class Handler(BaseHTTPRequestHandler):
        def _read_json(self):
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def _send_json(self, status_code: int, payload: dict):
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            try:
                if self.path == "/reset":
                    payload = self._read_json()
                    reset_seed = payload.get("seed")
                    reset_count = state.reset(seed=reset_seed)
                    print(
                        f"[lerobot_vla_server] reset count={reset_count} seed={state.current_seed} peer={self.client_address}",
                        flush=True,
                    )
                    self._send_json(200, {"ok": True})
                    return

                if self.path != "/infer":
                    self._send_json(404, {"error": f"unknown path: {self.path}"})
                    return

                payload = self._read_json()
                image = _decode_image(payload["observation"]["images"]["front"])
                raw_image_shape = tuple(image.shape)
                if state.http_image_transform is not None:
                    image = state.http_image_transform(image)
                observation_state = np.asarray(payload["observation"]["state"], dtype=np.float32).copy()
                if state.expected_state_shape is not None and observation_state.shape != state.expected_state_shape:
                    raise ValueError(
                        f"Expected observation.state shape {state.expected_state_shape}, got {observation_state.shape}"
                    )
                robot_type = payload.get("robot_type", "g129")
                task_value = payload.get("task", payload.get("task_name"))
                task_name, task_instruction = state.resolve_task_instruction(task_value)
                observation = {
                    "observation.images.front": image,
                    "observation.state": observation_state,
                }
                infer_index = state.infer_count + 1
                if infer_index == 1:
                    print(
                        f"[lerobot_vla_server] first_infer peer={self.client_address} robot_type={robot_type} "
                        f"task_name={task_name} task={task_instruction!r} "
                        f"state_shape={tuple(observation_state.shape)} raw_image_shape={raw_image_shape} "
                        f"image_shape={tuple(image.shape)}",
                        flush=True,
                    )

                if bool(payload.get("return_chunk", False)):
                    action_chunk = np.asarray(state.infer_chunk(observation, robot_type, task_instruction), dtype=np.float32)
                    if action_chunk.ndim == 1:
                        action_chunk = action_chunk.reshape(1, -1)
                    if action_chunk.ndim != 2:
                        raise ValueError(f"Expected 2D action chunk, got {action_chunk.shape}")
                    first_action = action_chunk[0]
                    response = {
                        "action": first_action.tolist(),
                        "action_chunk": action_chunk.tolist(),
                        "chunk_size": int(action_chunk.shape[0]),
                    }
                    log_suffix = f"chunk_size={action_chunk.shape[0]} first_action={first_action.tolist()}"
                else:
                    action = state.infer(observation, robot_type, task_instruction)
                    if not isinstance(action, torch.Tensor):
                        action = torch.as_tensor(action)
                    action = action.detach().cpu().to(torch.float32).reshape(-1)
                    if state.expected_action_shape is not None and tuple(action.shape) != state.expected_action_shape:
                        raise ValueError(
                            f"Expected action shape {state.expected_action_shape}, got {tuple(action.shape)}"
                        )
                    response = {"action": action.tolist()}
                    log_suffix = f"action={action.tolist()}"

                state.infer_count = infer_index
                print(f"[lerobot_vla_server] infer count={infer_index} {log_suffix}", flush=True)
                self._send_json(200, response)
            except BrokenPipeError:
                print(
                    f"[lerobot_vla_server] client disconnected while responding "
                    f"path={self.path} peer={self.client_address}",
                    flush=True,
                )
            except Exception as exc:
                traceback.print_exc()
                try:
                    self._send_json(500, {"error": str(exc)})
                except BrokenPipeError:
                    print(
                        f"[lerobot_vla_server] client disconnected before error response "
                        f"path={self.path} peer={self.client_address} error={exc}",
                        flush=True,
                    )

        def log_message(self, format: str, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Serve LeRobot VLA policy over HTTP(S)")
    parser.add_argument("--policy-path", required=True, help="Path to LeRobot pretrained_model directory")
    parser.add_argument("--device", default="cuda:0", help="LeRobot inference device")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8443, help="Bind port")
    parser.add_argument("--tls-cert-file", default="", help="Optional TLS certificate file")
    parser.add_argument("--tls-key-file", default="", help="Optional TLS private key file")
    args = parser.parse_args()

    policy_dir = Path(args.policy_path).expanduser().resolve()
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"Policy directory not found: {policy_dir}")

    _install_interrupt_handlers()

    server = None
    try:
        state = LeRobotServerState(policy_dir, args.device)
        server = ThreadingHTTPServer((args.host, args.port), make_handler(state))

        if args.tls_cert_file and args.tls_key_file:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile=args.tls_cert_file, keyfile=args.tls_key_file)
            server.socket = context.wrap_socket(server.socket, server_side=True)
            scheme = "https"
        else:
            scheme = "http"

        print(f"[lerobot_vla_server] Serving on {scheme}://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"[lerobot_vla_server] interrupted stop_reason={_SERVER_STOP_REASON}")
    finally:
        if server is not None:
            server.server_close()
            print("[lerobot_vla_server] server_closed")


if __name__ == "__main__":
    main()
