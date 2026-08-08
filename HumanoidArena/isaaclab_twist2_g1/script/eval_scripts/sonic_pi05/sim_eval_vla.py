#!/usr/bin/env python3

import argparse
import gc
import json
import math
import os
import random
import signal
import sys
import time
import uuid
from pathlib import Path
from task_runtime_profiles import apply_task_runtime_profile

ISAACLAB_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = str(ISAACLAB_ROOT)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher


def _load_simple_video_recorder():
    try:
        from utils.video_recorder import SimpleVideoRecorder
        return SimpleVideoRecorder
    except ModuleNotFoundError:
        import importlib.util

        module_path = Path(PROJECT_ROOT) / "utils" / "video_recorder.py"
        spec = importlib.util.spec_from_file_location("fallback_video_recorder", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.SimpleVideoRecorder


TASK_FOOTBALL_SINGLE = "Isaac-Move-Football-Single-G129-Dex3-Wholebody"
_INTERRUPT_REASON = "unknown"


def _interrupt_handler(signum, frame):
    global _INTERRUPT_REASON
    _INTERRUPT_REASON = signal.Signals(signum).name
    raise KeyboardInterrupt


def _install_interrupt_handlers():
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _interrupt_handler)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-episode or seed-batched VLA evaluation for football-single")
    parser.add_argument("--task", type=str, default=TASK_FOOTBALL_SINGLE)
    parser.add_argument("--env_config_yaml", type=str, default="tasks/common_test_config/base_test/football_single_sonic_test.yaml", help="YAML file with env config overrides")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeat_idx", type=int, default=0)
    parser.add_argument("--episode_seed", type=int, default=None)
    parser.add_argument("--episode_batch_json", type=str, default="")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--model_path", type=str, default="", help="Compatibility argument (unused in SONIC VLA eval)")
    parser.add_argument("--sonic_encoder_path", type=str, required=True, help="SONIC encoder ONNX path (29DoF controller)")
    parser.add_argument("--sonic_decoder_path", type=str, required=True, help="SONIC decoder ONNX path (29DoF controller)")
    parser.add_argument(
        "--sonic_vla_root_rot6d_layout",
        type=str,
        default="row",
        choices=["auto", "row", "col"],
        help="Root rot6d decode layout for VLA action (auto=row/col continuity selection).",
    )
    parser.add_argument(
        "--sonic_vla_root_max_delta_deg",
        type=float,
        default=26.0,
        help="Clamp max root orientation delta per step (degrees). <=0 to disable.",
    )
    parser.add_argument("--sonic_effort_control", action="store_true", default=False)
    parser.add_argument("--sonic_debug", action="store_true", default=False)
    parser.add_argument("--sonic_log_every", type=int, default=50)
    parser.add_argument("--lerobot_server_url", type=str, required=True)
    parser.add_argument("--lerobot_server_timeout", type=float, default=5.0)
    parser.add_argument("--lerobot_server_verify_ssl", action="store_true", default=False)
    parser.add_argument("--lerobot_gripper_threshold", type=float, default=0.5)
    parser.add_argument("--robot_type", type=str, default="unitree_g1_refpose_v3_1")
    parser.add_argument("--result_json", type=str, default="")
    parser.add_argument("--success_video_dir", type=str, default="")
    parser.add_argument("--failure_video_dir", type=str, default="")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--post_termination_record_steps", type=int, default=0)
    parser.add_argument("--record_video_every_n", type=int, default=1)
    parser.add_argument("--step_log_every_n", type=int, default=0)
    parser.add_argument("--verbose_startup", action="store_true", default=False)
    parser.add_argument("--disable_fall_detection", action="store_true", default=False)
    parser.add_argument("--fall_tilt_deg", type=float, default=60.0)
    parser.add_argument("--fall_hard_tilt_deg", type=float, default=75.0)
    parser.add_argument("--fall_contact_force_threshold", type=float, default=50.0)
    parser.add_argument("--fall_confirm_steps", type=int, default=5)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--model_label", type=str, default="")
    parser.add_argument("--eval_model_path", type=str, default="")
    parser.add_argument("--recording_save_dir", type=str, default="")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _is_verbose_startup(args) -> bool:
    return bool(getattr(args, "verbose_startup", False))


def _log_verbose(args, message: str):
    if _is_verbose_startup(args):
        print(message)


def _should_record_video(args, spec: dict) -> bool:
    every = max(0, int(getattr(args, "record_video_every_n", 1)))
    return every > 0 and int(spec["episode_index"]) % every == 0


def _should_log_reward_step(args, step_idx: int) -> bool:
    every = int(getattr(args, "step_log_every_n", 0))
    if every <= 0:
        return False
    return step_idx == 1 or step_idx % every == 0


def _cuda_device_index(device: str) -> int | None:
    token = str(device or "").strip().lower()
    if token.isdigit():
        return int(token)
    if token.startswith("cuda:"):
        gpu_idx = token.split(":", 1)[1]
        if gpu_idx.isdigit():
            return int(gpu_idx)
    return None


def _append_unique_kit_arg(args, kit_arg: str) -> None:
    existing_kit_args = (getattr(args, "kit_args", "") or "").split()
    if kit_arg not in existing_kit_args:
        args.kit_args = " ".join([*existing_kit_args, kit_arg]).strip()


def _configure_renderer_gpu(args) -> None:
    gpu_idx = _cuda_device_index(getattr(args, "device", ""))
    if gpu_idx is None:
        return
    _append_unique_kit_arg(args, f"--/renderer/activeGpu={gpu_idx}")
    _append_unique_kit_arg(args, f"--/physics/cudaDevice={gpu_idx}")


def _ensure_unique_multi_image_shm_name(args) -> str:
    existing = os.environ.get("ISAAC_MULTI_IMAGE_SHM_NAME", "").strip()
    if existing:
        return existing

    unique_suffix = f"{os.getpid()}_{args.seed}_{args.episode_index}_{uuid.uuid4().hex[:8]}"
    shm_name = f"isaac_multi_image_shm_{unique_suffix}"
    os.environ["ISAAC_MULTI_IMAGE_SHM_NAME"] = shm_name
    _log_verbose(args, f"[sim_eval_vla] shm_name={shm_name}")
    return shm_name


def _normalize_control_routing(args):
    args.input_source = "vla"
    args.gmt_backend = "sonic"
    args.action_source = "sonic_wholebody"
    args.enable_wholebody_dds = True
    args.enable_dex1_dds = False
    args.enable_dex3_dds = True
    args.enable_inspire_dds = False
    args.replay_file = ""
    args.replay_mode = "inference_replay"
    args.replay_loop = False
    args.replay_data = False
    args.language_instruction = ""
    args.lerobot_policy_path = ""
    args.lerobot_policy_device = ""
    args.sonic_pose_source = "redis"
    args.enable_world_camera = False

    if not args.recording_save_dir:
        if args.result_json:
            args.recording_save_dir = str(Path(args.result_json).resolve().parent / "recordings")
        else:
            args.recording_save_dir = str(Path.cwd() / "recordings")


def _initialize_task_scene(env, env_cfg, args, apply_optional_runtime_augments):
    try:
        initialize_task_scene = getattr(env_cfg, "initialize_task_scene", None)
        if callable(initialize_task_scene):
            initialize_task_scene(env, args)
            return
        apply_optional_runtime_augments(args)
        legacy_runtime_setup = getattr(env_cfg, "apply_runtime_setup", None)
        if callable(legacy_runtime_setup):
            legacy_runtime_setup(env, args)
    except Exception as exc:
        print(f"[eval_vla] init setup failed: {exc}")


def _trigger_task_reset_event(env_cfg, event_name, env):
    event_manager = getattr(env_cfg, "event_manager", None)
    if event_manager is None:
        return False
    event_manager.trigger(event_name, env)
    return True


def _disable_action_provider_internal_recording(action_provider):
    if action_provider is None:
        return
    setattr(action_provider, "_disable_eval_recording", True)
    if hasattr(action_provider, "_should_start_recording_on_first_call"):
        action_provider._should_start_recording_on_first_call = False
    if hasattr(action_provider, "_recording_command"):
        action_provider._recording_command = "none"
    if hasattr(action_provider, "_recording_active"):
        action_provider._recording_active = False
    if hasattr(action_provider, "_recording_display_state"):
        action_provider._recording_display_state = "idle"
    if hasattr(action_provider, "_save_in_progress"):
        action_provider._save_in_progress = False
    if hasattr(action_provider, "_pending_save_jobs"):
        action_provider._pending_save_jobs = 0
    recording_manager = getattr(action_provider, "recording_manager", None)
    if recording_manager is not None:
        try:
            recording_manager.cancel_recording()
        except Exception:
            pass


def _notify_action_provider_env_reset(action_provider):
    if action_provider is None:
        return
    for method_name in ("on_env_reset", "_reset_internal_buffers"):
        method = getattr(action_provider, method_name, None)
        if callable(method):
            method()
            break
    _disable_action_provider_internal_recording(action_provider)


def _capture_front_camera_rgb(env):
    try:
        if "front_camera" not in env.scene.keys():
            return None
        camera = env.scene["front_camera"]
        rgb = camera.data.output.get("rgb")
        if rgb is None:
            return None
        frame = rgb[0].detach().cpu().numpy()
        if frame.ndim != 3:
            return None
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != "uint8":
            frame = frame.clip(0, 255).astype("uint8")
        return frame
    except Exception:
        return None


def _extract_reward_info(env) -> dict:
    import torch

    reward_manager = getattr(env, "reward_manager", None)
    if reward_manager is not None:
        dt = getattr(env, "step_dt", None)
        if dt is None:
            dt = getattr(env, "physics_dt", None)
        if dt is None:
            raise RuntimeError("env.step_dt and env.physics_dt are both unavailable")
        scaled_reward = reward_manager.compute(dt=dt)
        if isinstance(scaled_reward, torch.Tensor):
            scaled_value = float(scaled_reward.detach().reshape(-1)[0].item())
        else:
            scaled_value = float(scaled_reward[0])

        raw_terms = []
        raw_total = 0.0
        get_terms = getattr(reward_manager, "get_active_iterable_terms", None)
        if callable(get_terms):
            for entry in get_terms(0):
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                term_name = str(entry[0])
                term_values = entry[1]
                if isinstance(term_values, (list, tuple)) and term_values:
                    term_value = float(term_values[0])
                else:
                    term_value = float(term_values)
                raw_terms.append((term_name, term_value))
                raw_total += term_value
        else:
            raw_total = scaled_value / dt

        return {
            "scaled_total": scaled_value,
            "raw_total": raw_total,
            "dt": float(dt),
            "raw_terms": raw_terms,
        }

    reward_buf = getattr(env, "reward_buf", None)
    if reward_buf is not None:
        if isinstance(reward_buf, torch.Tensor):
            value = float(reward_buf.detach().reshape(-1)[0].item())
        else:
            value = float(reward_buf[0])
        return {
            "scaled_total": value,
            "raw_total": value,
            "dt": 1.0,
            "raw_terms": [],
        }

    raise RuntimeError("env.reward_manager and env.reward_buf are both unavailable")


_FALL_CRITICAL_BODY_TOKENS = (
    "pelvis",
    "torso",
    "waist",
    "hip",
    "thigh",
    "knee",
    "calf",
    "shin",
    "head",
)
_FALL_SUPPORTED_BODY_TOKENS = (
    "ankle",
    "foot",
)


def _root_up_alignment_from_quat_wxyz(quat_wxyz) -> float:
    x = float(quat_wxyz[1])
    y = float(quat_wxyz[2])
    return 1.0 - 2.0 * (x * x + y * y)


def _build_fall_detector(env, args_cli):
    if getattr(args_cli, "disable_fall_detection", False):
        _log_verbose(args_cli, "[sim_eval_vla] fall detection disabled")
        return None

    robot = env.scene["robot"]
    body_names = [str(name) for name in (getattr(robot.data, "body_names", []) or [])]
    critical_body_indices = []
    critical_body_names = []
    for idx, name in enumerate(body_names):
        lowered = name.lower()
        if any(token in lowered for token in _FALL_SUPPORTED_BODY_TOKENS):
            continue
        if any(token in lowered for token in _FALL_CRITICAL_BODY_TOKENS):
            critical_body_indices.append(idx)
            critical_body_names.append(name)

    detector = {
        "soft_up_alignment": math.cos(math.radians(float(getattr(args_cli, "fall_tilt_deg", 60.0)))),
        "hard_up_alignment": math.cos(math.radians(float(getattr(args_cli, "fall_hard_tilt_deg", 75.0)))),
        "contact_force_threshold": float(getattr(args_cli, "fall_contact_force_threshold", 50.0)),
        "confirm_steps": max(1, int(getattr(args_cli, "fall_confirm_steps", 5))),
        "critical_body_indices": critical_body_indices,
        "critical_body_names": critical_body_names,
    }
    _log_verbose(
        args_cli,
        "[sim_eval_vla] fall detector configured "
        f"soft_tilt_deg={float(getattr(args_cli, 'fall_tilt_deg', 60.0)):.1f} "
        f"hard_tilt_deg={float(getattr(args_cli, 'fall_hard_tilt_deg', 75.0)):.1f} "
        f"contact_force_threshold={detector['contact_force_threshold']:.1f}N "
        f"confirm_steps={detector['confirm_steps']} "
        f"critical_bodies={detector['critical_body_names']}"
    )
    return detector


def _detect_robot_fall(env, detector):
    if detector is None:
        return False, ""

    robot = env.scene["robot"]
    up_alignment = None

    projected_gravity = getattr(robot.data, "projected_gravity_b", None)
    if projected_gravity is not None:
        try:
            up_alignment = float((-projected_gravity[0, 2]).detach().cpu().item())
        except Exception:
            up_alignment = None

    if up_alignment is None:
        root_state = getattr(robot.data, "root_state_w", None)
        if root_state is None:
            return False, ""
        quat_wxyz = root_state[0, 3:7].detach().cpu().tolist()
        up_alignment = _root_up_alignment_from_quat_wxyz(quat_wxyz)

    up_alignment = max(-1.0, min(1.0, float(up_alignment)))
    soft_tilt = up_alignment < float(detector["soft_up_alignment"])
    hard_tilt = up_alignment < float(detector["hard_up_alignment"])

    critical_contacts = []
    contact_forces = getattr(robot.data, "body_net_contact_force_w", None)
    if contact_forces is not None and detector["critical_body_indices"]:
        try:
            import torch

            magnitudes = torch.linalg.vector_norm(
                contact_forces[0, detector["critical_body_indices"]], dim=-1
            ).detach().cpu().tolist()
            for name, magnitude in zip(detector["critical_body_names"], magnitudes):
                force_value = float(magnitude)
                if force_value >= float(detector["contact_force_threshold"]):
                    critical_contacts.append(f"{name}:{force_value:.1f}N")
        except Exception:
            critical_contacts = []

    if not hard_tilt and not (soft_tilt and critical_contacts):
        return False, ""

    tilt_deg = math.degrees(math.acos(up_alignment))
    detail_parts = [
        f"tilt_deg={tilt_deg:.1f}",
        f"up_alignment={up_alignment:.3f}",
        "mode=hard_tilt" if hard_tilt else "mode=tilt_plus_contact",
    ]
    if critical_contacts:
        detail_parts.append("contacts=" + ",".join(critical_contacts[:4]))
    return True, " ".join(detail_parts)


def _write_result(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def _finalize_video(recorder, target_dir: Path, episode_name: str, suffix: str) -> str:
    if recorder is None:
        return ""
    target_dir.mkdir(parents=True, exist_ok=True)
    final_path = target_dir / f"{episode_name}__{suffix}.mp4"
    has_frames = bool(getattr(recorder, 'frame_count', 0) > 0 or getattr(recorder, 'frames', None))
    if has_frames:
        recorder.save(output_path=final_path)
        return str(final_path)
    return ""


def _get_process_rss_mb() -> float:
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0
    except Exception:
        pass
    return -1.0


def _cleanup_episode_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, 'ipc_collect', None)
            if callable(ipc_collect):
                ipc_collect()
    except Exception:
        pass


def _seed_runtime_rngs(seed: int):
    normalized_seed = int(seed) & 0xFFFFFFFF
    random.seed(normalized_seed)
    try:
        import numpy as np
        np.random.seed(normalized_seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(normalized_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(normalized_seed)
    except Exception:
        pass


def _configure_episode_seed_state(env_cfg, episode_seed: int) -> int:
    normalized_seed = int(episode_seed) & 0x7FFFFFFF
    env_cfg.seed = normalized_seed
    env_cfg.object_reset_seed_source = "env_seed"
    setattr(env_cfg, "_episode_runtime_seed", normalized_seed)
    setattr(env_cfg, "_episode_object_seed_counter", 0)
    setattr(env_cfg, "_current_episode_object_seed", None)
    setattr(env_cfg, "_current_episode_object_seed_source", "")
    return normalized_seed


def _build_single_episode_spec(args_cli) -> dict:
    if not args_cli.result_json:
        raise ValueError("single-episode mode requires --result_json")
    episode_seed = int(args_cli.episode_seed if args_cli.episode_seed is not None else args_cli.seed)
    return {
        "seed": int(args_cli.seed),
        "repeat_idx": int(args_cli.repeat_idx),
        "episode_seed": episode_seed,
        "episode_index": int(args_cli.episode_index),
        "result_json": args_cli.result_json,
        "success_video_dir": args_cli.success_video_dir,
        "failure_video_dir": args_cli.failure_video_dir,
        "video_fps": int(args_cli.video_fps),
        "post_termination_record_steps": int(args_cli.post_termination_record_steps),
        "model_label": args_cli.model_label,
        "eval_model_path": args_cli.eval_model_path,
        "recording_save_dir": args_cli.recording_save_dir,
        "max_steps": int(args_cli.max_steps),
    }


def _load_episode_specs(args_cli) -> list[dict]:
    if not args_cli.episode_batch_json:
        return [_build_single_episode_spec(args_cli)]

    batch_path = Path(args_cli.episode_batch_json).expanduser().resolve()
    raw = json.loads(batch_path.read_text())
    episodes = raw.get("episodes", [])
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"episode batch must contain a non-empty episodes list: {batch_path}")

    normalized = []
    for entry in episodes:
        normalized.append(
            {
                "seed": int(entry.get("seed", args_cli.seed)),
                "repeat_idx": int(entry["repeat_idx"]),
                "episode_seed": int(entry["episode_seed"]),
                "episode_index": int(entry["episode_index"]),
                "result_json": str(entry["result_json"]),
                "success_video_dir": str(entry["success_video_dir"]),
                "failure_video_dir": str(entry["failure_video_dir"]),
                "video_fps": int(entry.get("video_fps", args_cli.video_fps)),
                "post_termination_record_steps": int(entry.get("post_termination_record_steps", args_cli.post_termination_record_steps)),
                "model_label": str(entry.get("model_label", args_cli.model_label)),
                "eval_model_path": str(entry.get("eval_model_path", args_cli.eval_model_path)),
                "recording_save_dir": str(entry.get("recording_save_dir", args_cli.recording_save_dir)),
                "max_steps": int(entry.get("max_steps", args_cli.max_steps)),
            }
        )
    return normalized


def _reset_environment_for_episode(env, env_cfg, episode_seed: int):
    _seed_runtime_rngs(episode_seed)
    _configure_episode_seed_state(env_cfg, episode_seed)
    runtime_cfg = getattr(env, "cfg", env_cfg)
    if runtime_cfg is not env_cfg:
        _configure_episode_seed_state(runtime_cfg, episode_seed)
    env.sim.reset()
    env.reset()
    vis_cfg_json = os.getenv("VISION_RANDOMIZATION")
    if vis_cfg_json:
        try:
            import json
            from tasks.common_runtime import apply_vision_light_randomization
            vis_cfg = json.loads(vis_cfg_json)
            apply_vision_light_randomization(
                prim_path=vis_cfg.get("prim_path", "/World/light"),
                episode_seed=episode_seed,
                rotation_ranges=vis_cfg.get("rotation"),
                intensity_range=vis_cfg.get("intensity_range"),
                color_range=vis_cfg.get("color_range"),
                position_range=vis_cfg.get("position_range"),
            )
        except Exception as exc:
            print(f"[vision_test] per-episode randomization failed: {exc}")
    if getattr(env_cfg, "startup_task_reset_enabled", True):
        _trigger_task_reset_event(env_cfg, "reset_all_self", env)


def _reset_lerobot_runtime_seed(action_provider, episode_seed: int):
    client = getattr(action_provider, "_lerobot_http_client", None)
    if client is not None:
        client.reset(seed=int(episode_seed))



def _build_result_payload(args_cli, spec: dict, model_label: str, server_url: str, *, success: bool, failure_reason: str, step_idx: int, terminal_step_idx: int, final_reward: float, final_reward_scaled: float, max_reward: float, max_reward_scaled: float, video_path: str, started_at: float, error: str = "", terminal_detail: str = "", env=None) -> dict:
    from common_env_objects import get_current_episode_object_seed_info

    payload = {
        "task": args_cli.task,
        "model_path": spec.get("eval_model_path", ""),
        "model_label": model_label,
        "seed": int(spec["seed"]),
        "repeat_idx": int(spec["repeat_idx"]),
        "episode_seed": int(spec["episode_seed"]),
        "episode_index": int(spec["episode_index"]),
        "success": bool(success),
        "failure_reason": failure_reason,
        "episode_steps": int(terminal_step_idx or step_idx),
        "max_steps": int(spec["max_steps"]),
        "final_reward": float(final_reward),
        "final_reward_scaled": float(final_reward_scaled),
        "max_reward": 0.0 if max_reward == float("-inf") else float(max_reward),
        "max_reward_scaled": 0.0 if max_reward_scaled == float("-inf") else float(max_reward_scaled),
        "video_path": video_path,
        "video_recorded": bool(video_path),
        "server_url": server_url,
        "started_at": started_at,
        "finished_at": time.time(),
        "duration_sec": time.time() - started_at,
        "episode_object_seed": get_current_episode_object_seed_info(env.cfg).get("seed") if env is not None else None,
        "episode_object_seed_source": get_current_episode_object_seed_info(env.cfg).get("source") if env is not None else "",
    }
    if error:
        payload["error"] = error
    if terminal_detail:
        payload["terminal_detail"] = terminal_detail
    return payload


def _run_episode_once(simulation_app, env, env_cfg, action_provider, controller, args_cli, spec: dict, SimpleVideoRecorder):
    model_label = spec.get("model_label") or Path(spec.get("eval_model_path") or getattr(args_cli, "model_path", "model")).stem
    episode_name = f"{model_label}__seed_{spec['seed']}__repeat_{spec['repeat_idx']}__episode_{spec['episode_index']}"
    result_path = Path(spec["result_json"]).expanduser().resolve()
    success_video_dir = Path(spec["success_video_dir"]).expanduser().resolve()
    failure_video_dir = Path(spec["failure_video_dir"]).expanduser().resolve()
    record_video = _should_record_video(args_cli, spec)
    if record_video:
        success_video_dir.mkdir(parents=True, exist_ok=True)
        failure_video_dir.mkdir(parents=True, exist_ok=True)
        temp_video_path = result_path.parent / f"{episode_name}__tmp.mp4"
        if temp_video_path.exists():
            temp_video_path.unlink()
        recorder = SimpleVideoRecorder(str(temp_video_path), fps=int(spec["video_fps"]))
    else:
        recorder = None
    started_at = time.time()
    step_idx = 0
    max_reward = float("-inf")
    max_reward_scaled = float("-inf")
    final_reward = 0.0
    final_reward_scaled = 0.0
    terminal_step_idx = 0
    success = False
    failure_reason = "unknown"
    terminal_detail = ""
    video_path = ""
    post_termination_steps_remaining = 0
    record_post_termination_steps = int(spec["post_termination_record_steps"]) if recorder is not None else 0
    fall_detector = _build_fall_detector(env, args_cli)
    fall_streak = 0

    try:
        _reset_environment_for_episode(env, env_cfg, int(spec["episode_seed"]))
        _reset_lerobot_runtime_seed(action_provider, int(spec["episode_seed"]))
        _notify_action_provider_env_reset(action_provider)
        controller.start()

        if recorder is not None:
            initial_frame = _capture_front_camera_rgb(env)
            if initial_frame is not None:
                recorder.add_frame(initial_frame)

        while simulation_app.is_running() and controller.is_running:
            controller.step()
            step_idx += 1

            if recorder is not None:
                frame = _capture_front_camera_rgb(env)
                if frame is not None:
                    recorder.add_frame(frame)

            if post_termination_steps_remaining > 0:
                post_termination_steps_remaining -= 1
                if env.sim.is_stopped():
                    _log_verbose(
                        args_cli,
                        f"[sim_eval_vla] post-termination recording interrupted at control_step={step_idx} because simulation stopped"
                    )
                    break
                if post_termination_steps_remaining == 0:
                    _log_verbose(args_cli, f"[sim_eval_vla] post-termination recording finished at control_step={step_idx}")
                    break
                continue

            reward_info = _extract_reward_info(env)
            final_reward = reward_info["raw_total"]
            final_reward_scaled = reward_info["scaled_total"]
            if final_reward > max_reward:
                max_reward = final_reward
            if final_reward_scaled > max_reward_scaled:
                max_reward_scaled = final_reward_scaled

            if _should_log_reward_step(args_cli, step_idx):
                terms_str = ", ".join(f"{name}={value:.4f}" for name, value in reward_info["raw_terms"])
                print(
                    f"[sim_eval_vla] reward step={step_idx} raw_total={final_reward:.4f} scaled_total={final_reward_scaled:.4f}"
                    + (f" | {terms_str}" if terms_str else "")
                )

            if final_reward >= 1.0:
                success = True
                failure_reason = "success"
                terminal_step_idx = step_idx
                post_termination_steps_remaining = record_post_termination_steps
                if post_termination_steps_remaining <= 0:
                    break
                continue

            fall_detected, fall_detail = _detect_robot_fall(env, fall_detector)
            if fall_detected:
                fall_streak += 1
                if _is_verbose_startup(args_cli):
                    print(
                        f"[sim_eval_vla] fall_candidate step={step_idx} streak={fall_streak}/{fall_detector['confirm_steps']} {fall_detail}"
                    )
            else:
                fall_streak = 0

            if fall_detected and fall_streak >= int(fall_detector["confirm_steps"]):
                failure_reason = "fall"
                terminal_detail = fall_detail
                terminal_step_idx = step_idx
                print(f"[sim_eval_vla] fall_terminated step={step_idx} {terminal_detail}")
                post_termination_steps_remaining = record_post_termination_steps
                if post_termination_steps_remaining <= 0:
                    break
                continue

            if step_idx >= int(spec["max_steps"]):
                failure_reason = "timeout"
                terminal_step_idx = step_idx
                post_termination_steps_remaining = record_post_termination_steps
                if post_termination_steps_remaining <= 0:
                    break
                continue

            if env.sim.is_stopped():
                failure_reason = "sim_stopped"
                terminal_step_idx = step_idx
                break

        target_dir = success_video_dir if success else failure_video_dir
        video_suffix = "success" if success else failure_reason
        video_path = _finalize_video(recorder, target_dir, episode_name, video_suffix)
        payload = _build_result_payload(
            args_cli,
            spec,
            model_label,
            args_cli.lerobot_server_url,
            success=success,
            failure_reason=failure_reason,
            step_idx=step_idx,
            terminal_step_idx=terminal_step_idx,
            final_reward=final_reward,
            final_reward_scaled=final_reward_scaled,
            max_reward=max_reward,
            max_reward_scaled=max_reward_scaled,
            video_path=video_path,
            started_at=started_at,
            terminal_detail=terminal_detail,
            env=env,
        )
        _write_result(result_path, payload)
        return payload
    except KeyboardInterrupt:
        reason = _INTERRUPT_REASON or "KeyboardInterrupt"
        print(f"[sim_eval_vla] interrupted stop_reason={reason} control_step={step_idx}")
        video_path = _finalize_video(recorder, failure_video_dir, episode_name, "interrupted")
        payload = _build_result_payload(
            args_cli,
            spec,
            model_label,
            args_cli.lerobot_server_url,
            success=False,
            failure_reason="interrupted",
            step_idx=step_idx,
            terminal_step_idx=step_idx,
            final_reward=final_reward,
            final_reward_scaled=final_reward_scaled,
            max_reward=max_reward,
            max_reward_scaled=max_reward_scaled,
            video_path=video_path,
            started_at=started_at,
            error=reason,
            env=env,
        )
        _write_result(result_path, payload)
        raise
    except Exception as exc:
        print(f"[sim_eval_vla] episode failed: {exc}")
        video_path = _finalize_video(recorder, failure_video_dir, episode_name, "sim_error")
        payload = _build_result_payload(
            args_cli,
            spec,
            model_label,
            args_cli.lerobot_server_url,
            success=False,
            failure_reason="sim_error",
            step_idx=step_idx,
            terminal_step_idx=step_idx,
            final_reward=final_reward,
            final_reward_scaled=final_reward_scaled,
            max_reward=max_reward,
            max_reward_scaled=max_reward_scaled,
            video_path=video_path,
            started_at=started_at,
            error=str(exc),
            env=env,
        )
        _write_result(result_path, payload)
        return payload
    finally:
        try:
            controller.stop()
        except Exception as exc:
            print(f"[sim_eval_vla] controller stop failed: {exc}")
        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:
                print(f"[sim_eval_vla] recorder close failed: {exc}")
            try:
                recorder.clear()
            except Exception as exc:
                print(f"[sim_eval_vla] recorder clear failed: {exc}")
        _cleanup_episode_memory()


def main() -> int:
    parser = _build_parser()
    args_cli = parser.parse_args()
    _install_interrupt_handlers()
    _ensure_unique_multi_image_shm_name(args_cli)
    args_cli.enable_cameras = True
    args_cli.multi_gpu = False
    _append_unique_kit_arg(args_cli, "--/renderer/multiGpu/enabled=False")
    _configure_renderer_gpu(args_cli)
    _normalize_control_routing(args_cli)
    apply_task_runtime_profile(args_cli)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import tasks
    from action_provider.create_action_provider import create_action_provider
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from layeredcontrol.robot_control_system import ControlConfig, RobotController
    from tasks.common_env_config import apply_env_config_yaml
    from tasks.common_runtime import apply_optional_runtime_augments, setup_vision_test_light
    SimpleVideoRecorder = _load_simple_video_recorder()

    episode_specs = _load_episode_specs(args_cli)
    if not episode_specs:
        raise ValueError("No episode specs resolved")

    env = None
    controller = None
    action_provider = None
    exit_code = 0

    try:
        first_spec = episode_specs[0]
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task
        apply_env_config_yaml(
            env_cfg,
            args_cli.env_config_yaml,
            task_name=args_cli.task,
            route_name=args_cli.gmt_backend or args_cli.action_source,
        )
        _seed_runtime_rngs(int(first_spec["episode_seed"]))
        _configure_episode_seed_state(env_cfg, int(first_spec["episode_seed"]))
        print(
            f"[sim_eval_vla] startup seed={first_spec['seed']} repeat_idx={first_spec['repeat_idx']} episode_seed={first_spec['episode_seed']}"
        )

        try:
            _log_verbose(args_cli, "[sim_eval_vla] startup checkpoint=before_gym_make")
            env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
            _log_verbose(args_cli, "[sim_eval_vla] startup checkpoint=after_gym_make")
            _initialize_task_scene(env, env_cfg, args_cli, apply_optional_runtime_augments)
            _log_verbose(args_cli, "[sim_eval_vla] startup checkpoint=after_initialize_task_scene")
            if os.getenv("VISION_RANDOMIZATION"):
                setup_vision_test_light()
            _reset_environment_for_episode(env, env_cfg, int(first_spec["episode_seed"]))
            _log_verbose(args_cli, "[sim_eval_vla] startup checkpoint=after_initial_reset")

            physics_dt = getattr(env, "physics_dt", None) or env_cfg.sim.dt
            control_hz = int(round(1.0 / (physics_dt * env_cfg.decimation)))
            control_config = ControlConfig(step_hz=control_hz, replay_mode=False, use_rl_action_mode=True)
            print(
                f"[sim_eval_vla] task={args_cli.task} control_hz={control_hz} episodes={len(episode_specs)} "
                f"lerobot_server={args_cli.lerobot_server_url} record_video_every_n={args_cli.record_video_every_n} "
                f"step_log_every_n={args_cli.step_log_every_n}"
            )

            _log_verbose(args_cli, "[sim_eval_vla] startup checkpoint=before_create_action_provider")
            action_provider = create_action_provider(env, args_cli)
            _log_verbose(args_cli, "[sim_eval_vla] startup checkpoint=after_create_action_provider")
            if action_provider is None:
                raise RuntimeError("failed to create action provider")
            _disable_action_provider_internal_recording(action_provider)
            env.action_provider = action_provider

            controller = RobotController(env, control_config)
            controller.set_action_provider(action_provider)
            _log_verbose(args_cli, "[sim_eval_vla] startup checkpoint=after_controller_setup")

            for spec in episode_specs:
                rss_before_mb = _get_process_rss_mb()
                record_video = _should_record_video(args_cli, spec)
                print(
                    f"[sim_eval_vla] episode_start seed={spec['seed']} repeat_idx={spec['repeat_idx']} "
                    f"episode_seed={spec['episode_seed']} rss_mb={rss_before_mb:.1f} record_video={int(record_video)}"
                )
                try:
                    _run_episode_once(simulation_app, env, env_cfg, action_provider, controller, args_cli, spec, SimpleVideoRecorder)
                except KeyboardInterrupt:
                    exit_code = 130
                    break
                finally:
                    rss_after_mb = _get_process_rss_mb()
                    print(
                        f"[sim_eval_vla] episode_end seed={spec['seed']} repeat_idx={spec['repeat_idx']} "
                        f"episode_seed={spec['episode_seed']} rss_mb={rss_after_mb:.1f}"
                    )
        except BaseException as exc:
            print(f"[sim_eval_vla] startup failed: {type(exc).__name__}: {exc}")
            raise
    finally:
        try:
            if controller is not None:
                controller.cleanup()
        except Exception as exc:
            print(f"[sim_eval_vla] controller cleanup failed: {exc}")
        try:
            if env is not None:
                env.close()
        except Exception as exc:
            print(f"[sim_eval_vla] env close failed: {exc}")
        try:
            simulation_app.close()
        except Exception as exc:
            print(f"[sim_eval_vla] simulation_app close failed: {exc}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
