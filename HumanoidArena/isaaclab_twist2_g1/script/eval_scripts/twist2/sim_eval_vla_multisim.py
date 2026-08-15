#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from task_runtime_profiles import apply_task_runtime_profile

ISAACLAB_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = str(ISAACLAB_ROOT)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher

from sim_eval_vla import (
    TASK_FOOTBALL_SINGLE,
    _cleanup_episode_memory,
    _ensure_unique_multi_image_shm_name,
    _finalize_video,
    _get_process_rss_mb,
    _initialize_task_scene,
    _install_interrupt_handlers,
    _load_simple_video_recorder,
    _normalize_control_routing,
    _seed_runtime_rngs,
    _write_result,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-env VLA evaluation with one IsaacLab app")
    parser.add_argument("--task", type=str, default=TASK_FOOTBALL_SINGLE)
    parser.add_argument(
        "--env_config_yaml",
        type=str,
        default="tasks/common_test_config/base_test/football_single_twist2_test.yaml",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode_batch_json", type=str, required=True)
    parser.add_argument("--server_urls_json", type=str, required=True)
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lerobot_server_timeout", type=float, default=5.0)
    parser.add_argument("--lerobot_server_verify_ssl", action="store_true", default=False)
    parser.add_argument("--lerobot_gripper_threshold", type=float, default=0.5)
    parser.add_argument("--robot_type", type=str, default="unitree_g1_refpose_v3_1")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--post_termination_record_steps", type=int, default=0)
    parser.add_argument("--recording_save_dir", type=str, default="")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _load_episode_specs(args_cli) -> list[dict]:
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
                "post_termination_record_steps": int(
                    entry.get("post_termination_record_steps", args_cli.post_termination_record_steps)
                ),
                "model_label": str(entry.get("model_label", Path(args_cli.model_path).stem)),
                "eval_model_path": str(entry.get("eval_model_path", args_cli.model_path)),
                "recording_save_dir": str(entry.get("recording_save_dir", args_cli.recording_save_dir)),
                "max_steps": int(entry.get("max_steps", args_cli.max_steps)),
            }
        )
    return normalized


def _load_server_urls(args_cli) -> list[str]:
    server_urls = json.loads(Path(args_cli.server_urls_json).expanduser().resolve().read_text())
    if not isinstance(server_urls, list) or not server_urls:
        raise ValueError("server_urls_json must contain a non-empty JSON list")
    normalized = [str(url).strip().rstrip("/") for url in server_urls if str(url).strip()]
    if len(normalized) < int(args_cli.num_envs):
        raise ValueError(
            f"num_envs={args_cli.num_envs} requires at least that many server URLs, got {len(normalized)}"
        )
    return normalized[: int(args_cli.num_envs)]


def _capture_front_camera_rgb(env, env_id: int):
    try:
        if "front_camera" not in env.scene.keys():
            return None
        camera = env.scene["front_camera"]
        rgb = camera.data.output.get("rgb")
        if rgb is None:
            return None
        frame = rgb[int(env_id)].detach().cpu().numpy()
        if frame.ndim != 3:
            return None
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != "uint8":
            frame = frame.clip(0, 255).astype("uint8")
        return frame
    except Exception:
        return None


def _extract_reward_info_for_env(env, env_id: int) -> dict:
    import torch

    reward_manager = getattr(env, "reward_manager", None)
    raw_terms = []
    raw_total = 0.0
    if reward_manager is not None:
        get_terms = getattr(reward_manager, "get_active_iterable_terms", None)
        if callable(get_terms):
            for entry in get_terms(int(env_id)):
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                term_name = str(entry[0])
                term_values = entry[1]
                if isinstance(term_values, torch.Tensor):
                    term_value = float(term_values.detach().reshape(-1)[0].item())
                elif isinstance(term_values, (list, tuple)) and term_values:
                    term_value = float(term_values[0])
                else:
                    term_value = float(term_values)
                raw_terms.append((term_name, term_value))
                raw_total += term_value

    reward_buf = getattr(env, "reward_buf", None)
    if isinstance(reward_buf, torch.Tensor):
        scaled_total = float(reward_buf[int(env_id)].detach().item())
    elif reward_buf is not None:
        scaled_total = float(reward_buf[int(env_id)])
    else:
        dt = getattr(env, "step_dt", None)
        if dt is None:
            dt = getattr(env, "physics_dt", 1.0) * int(getattr(env.cfg, "decimation", 1))
        scaled_total = raw_total * float(dt)
    return {
        "scaled_total": scaled_total,
        "raw_total": raw_total,
        "raw_terms": raw_terms,
    }


def _build_result_payload(
    spec: dict,
    server_url: str,
    *,
    success: bool,
    failure_reason: str,
    step_idx: int,
    terminal_step_idx: int,
    final_reward: float,
    final_reward_scaled: float,
    max_reward: float,
    max_reward_scaled: float,
    video_path: str,
    started_at: float,
    error: str = "",
) -> dict:
    payload = {
        "task": TASK_FOOTBALL_SINGLE,
        "model_path": spec.get("eval_model_path", ""),
        "model_label": spec.get("model_label", "model"),
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
        "server_url": server_url,
        "started_at": started_at,
        "finished_at": time.time(),
        "duration_sec": time.time() - started_at,
        "episode_object_seed": None,
        "episode_object_seed_source": "",
    }
    if error:
        payload["error"] = error
    return payload


@dataclass
class SlotState:
    env_id: int
    server_url: str
    current_spec: dict | None = None
    recorder: object | None = None
    temp_video_path: Path | None = None
    started_at: float = 0.0
    step_idx: int = 0
    terminal_step_idx: int = 0
    final_reward: float = 0.0
    final_reward_scaled: float = 0.0
    max_reward: float = float("-inf")
    max_reward_scaled: float = float("-inf")
    success: bool = False
    failure_reason: str = ""
    post_remaining: int = 0

    def is_active(self) -> bool:
        return self.current_spec is not None

    def reset_metrics(self):
        self.started_at = time.time()
        self.step_idx = 0
        self.terminal_step_idx = 0
        self.final_reward = 0.0
        self.final_reward_scaled = 0.0
        self.max_reward = float("-inf")
        self.max_reward_scaled = float("-inf")
        self.success = False
        self.failure_reason = ""
        self.post_remaining = 0


class MultiEnvVLATwist2Runtime:
    def __init__(self, env, args_cli, server_urls):
        import numpy as np
        import onnxruntime as ort
        import torch
        from action_provider.lerobot_vla_http_client import LeRobotVLAHttpClient
        from action_provider.vla_robot_current_local_runtime_v3 import (
            VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM,
            UnifiedRobotCurrentLocalActionRuntimeV3,
            build_twist2_mimic_obs_v3,
            build_vla_rotlocal_v3_observation_state,
        )
        from action_provider.vla_smpl_runtime import (
            TWIST2_G1_JOINT_NAMES_29,
            reorder_twist2_to_canonical_29,
        )
        from pico_server.data_utils.params import DEFAULT_HAND_POSE

        self.np = np
        self.torch = torch
        self.ort = ort
        self.env = env
        self.args_cli = args_cli
        self.num_envs = int(args_cli.num_envs)
        self.device = env.device
        self.server_urls = list(server_urls)
        self.vla_action_dim = VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM
        self.build_twist2_mimic_obs_v3 = build_twist2_mimic_obs_v3
        self.build_vla_rotlocal_v3_observation_state = build_vla_rotlocal_v3_observation_state
        self.reorder_twist2_to_canonical_29 = reorder_twist2_to_canonical_29
        self.DEFAULT_HAND_POSE = DEFAULT_HAND_POSE
        self.control_dt = float(env.physics_dt) * int(getattr(env.cfg, "decimation", 1))
        self.history_len = 10
        self.obs_single_dim = 127
        self.gripper_threshold = float(getattr(args_cli, "lerobot_gripper_threshold", 0.5))
        self.enable_robot = args_cli.robot_type
        if self.enable_robot != "unitree_g1_refpose_v3_1":
            raise ValueError(
                "Multi-env VLA v3.1 runtime requires robot_type=unitree_g1_refpose_v3_1; "
                f"got {self.enable_robot!r}"
            )

        all_joint_names = env.scene["robot"].data.joint_names
        self.joint_to_index = {name: i for i, name in enumerate(all_joint_names)}
        missing = [n for n in TWIST2_G1_JOINT_NAMES_29 if n not in self.joint_to_index]
        if missing:
            raise ValueError(f"TWIST2 joints missing in Isaac asset: {missing}")
        self.twist2_action_indices = [self.joint_to_index[n] for n in TWIST2_G1_JOINT_NAMES_29]
        self.twist2_default_pos = env.scene["robot"].data.default_joint_pos[:, self.twist2_action_indices].clone()
        self.default_joint_pos = env.scene["robot"].data.default_joint_pos.clone()
        self._twist2_last_action = torch.zeros(self.num_envs, 29, device=self.device, dtype=torch.float32)
        self._twist2_history = torch.zeros(
            self.num_envs, self.history_len, self.obs_single_dim, device=self.device, dtype=torch.float32
        )
        self._vla_gripper_binary = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._chunk_queues = [deque() for _ in range(self.num_envs)]
        self._runtimes = [UnifiedRobotCurrentLocalActionRuntimeV3() for _ in range(self.num_envs)]
        self._vla_initial_robot_quat_wxyz = [None for _ in range(self.num_envs)]
        self._clients = [
            LeRobotVLAHttpClient(
                base_url=url,
                timeout_s=float(args_cli.lerobot_server_timeout),
                verify_ssl=bool(args_cli.lerobot_server_verify_ssl),
            )
            for url in self.server_urls
        ]
        self._left_hand_target_idx_t = None
        self._right_hand_target_idx_t = None
        left_joints = [
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
        ]
        right_joints = [
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
        ]
        if all(name in self.joint_to_index for name in left_joints + right_joints):
            self._left_hand_target_idx_t = torch.tensor(
                [self.joint_to_index[name] for name in left_joints], dtype=torch.long, device=self.device
            )
            self._right_hand_target_idx_t = torch.tensor(
                [self.joint_to_index[name] for name in right_joints], dtype=torch.long, device=self.device
            )
        self._twist2_ankle_idx = [4, 5, 10, 11]
        self.policy = self._load_onnx_policy(args_cli.model_path, str(getattr(args_cli, "device", "")))

    def _resolve_onnx_device_id(self, requested_device: str) -> int:
        value = (requested_device or "").strip().lower()
        if value.startswith("cuda:"):
            try:
                return int(value.split(":", 1)[1])
            except Exception:
                return 0
        if value.startswith("cuda"):
            return 0
        if value.isdigit():
            try:
                return int(value)
            except Exception:
                return 0
        return 0

    def _load_onnx_policy(self, path: str, requested_device: str):
        available = []
        try:
            available = self.ort.get_available_providers()
        except Exception:
            available = []
        providers = []
        expected_gpu_providers = []
        device_id = self._resolve_onnx_device_id(requested_device)
        if str(requested_device).startswith("cuda") or str(requested_device).isdigit():
            if "TensorrtExecutionProvider" in available:
                providers.append(("TensorrtExecutionProvider", {"device_id": device_id}))
                expected_gpu_providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in available:
                providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
                expected_gpu_providers.append("CUDAExecutionProvider")
            if not expected_gpu_providers:
                raise RuntimeError(
                    f"requested GPU ONNX session on {requested_device}, but available providers are {available}"
                )
        providers.append("CPUExecutionProvider")
        sess_options = self.ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
        model = self.ort.InferenceSession(path, sess_options=sess_options, providers=providers)
        loaded_providers = model.get_providers()
        if expected_gpu_providers and not any(name in loaded_providers for name in expected_gpu_providers):
            raise RuntimeError(
                f"ONNX model loaded on CPU instead of {requested_device}; providers={loaded_providers}"
            )
        input_name = model.get_inputs()[0].name
        print(f"[sim_eval_vla_multisim] ONNX providers={loaded_providers}")

        def run_inference(input_tensor):
            ort_inputs = {input_name: input_tensor.detach().cpu().numpy()}
            ort_outs = model.run(None, ort_inputs)
            return self.torch.tensor(ort_outs[0], device=self.env.device, dtype=self.torch.float32)

        return run_inference

    def reset_env_slot(self, env_id: int, episode_seed: int) -> None:
        self._chunk_queues[env_id].clear()
        root_quat_wxyz, root_xy_world = self._get_current_robot_root_pose(env_id)
        self._vla_initial_robot_quat_wxyz[env_id] = root_quat_wxyz.copy()
        self._runtimes[env_id].reset(
            body_xy_world=root_xy_world,
            target_root_quat_wxyz=root_quat_wxyz,
        )
        self._twist2_last_action[env_id].zero_()
        self._twist2_history[env_id].zero_()
        self._vla_gripper_binary[env_id].zero_()
        self._clients[env_id].reset(seed=int(episode_seed))

    def _twist2_roll_pitch_from_quaternion(self, quat):
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = self.torch.atan2(t0, t1)
        t2 = 2.0 * (w * y - z * x)
        t2 = self.torch.clamp(t2, -1.0, 1.0)
        pitch = self.torch.asin(t2)
        return self.torch.stack([roll, pitch], dim=-1)

    def _front_rgb(self, env_id: int):
        rgb = self.env.scene["front_camera"].data.output["rgb"][env_id].detach().cpu().numpy()
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != self.np.uint8:
            rgb = self.np.clip(rgb, 0, 255).astype("uint8")
        return rgb

    def _get_current_robot_root_pose(self, env_id: int):
        robot = self.env.scene["robot"].data
        root_state = robot.root_state_w[env_id].detach().cpu().numpy().astype(self.np.float32)
        return root_state[3:7].copy(), root_state[0:2].copy()

    def _build_vla_state(self, env_id: int):
        robot = self.env.scene["robot"].data
        joint_pos_twist2 = (
            robot.joint_pos[env_id, self.twist2_action_indices].detach().cpu().numpy().astype(self.np.float32)
        )
        joint_vel_twist2 = (
            robot.joint_vel[env_id, self.twist2_action_indices].detach().cpu().numpy().astype(self.np.float32)
        )
        joint_pos_canonical = self.reorder_twist2_to_canonical_29(joint_pos_twist2)
        joint_vel_canonical = self.reorder_twist2_to_canonical_29(joint_vel_twist2)
        base_quat_wxyz, _ = self._get_current_robot_root_pose(env_id)
        if self._vla_initial_robot_quat_wxyz[env_id] is None:
            self._vla_initial_robot_quat_wxyz[env_id] = base_quat_wxyz.copy()
        return self.build_vla_rotlocal_v3_observation_state(
            initial_robot_orientation_wxyz=self._vla_initial_robot_quat_wxyz[env_id],
            root_orientation_wxyz=base_quat_wxyz,
            joint_pos_canonical_29=joint_pos_canonical,
            joint_vel_canonical_29=joint_vel_canonical,
        )

    def fetch_missing_chunks(self, env_ids: list[int]) -> None:
        if not env_ids:
            return

        def _fetch_one(env_id: int):
            return env_id, self._clients[env_id].infer_chunk(
                front_rgb=self._front_rgb(env_id),
                observation_state=self._build_vla_state(env_id),
                robot_type=self.enable_robot,
            )

        with ThreadPoolExecutor(max_workers=len(env_ids)) as pool:
            for env_id, action_chunk in pool.map(_fetch_one, env_ids):
                chunk = self.np.asarray(action_chunk, dtype=self.np.float32)
                if chunk.ndim == 1:
                    chunk = chunk.reshape(1, -1)
                if chunk.ndim != 2 or chunk.shape[1] != self.vla_action_dim:
                    raise ValueError(
                        f"Expected refpose v3.1 VLA action chunk shape [N, {self.vla_action_dim}], got {chunk.shape}"
                    )
                for action in chunk:
                    self._chunk_queues[env_id].append(self.np.asarray(action, dtype=self.np.float32).copy())

    def _build_action_mimic_batch(self, active_env_ids: list[int]):
        missing = [env_id for env_id in active_env_ids if not self._chunk_queues[env_id]]
        self.fetch_missing_chunks(missing)

        action_mimic = self.torch.zeros(self.num_envs, 35, device=self.device, dtype=self.torch.float32)
        for env_id in active_env_ids:
            action_np = self.np.asarray(self._chunk_queues[env_id].popleft(), dtype=self.np.float32).reshape(-1)
            current_robot_quat_wxyz, current_robot_xy_world = self._get_current_robot_root_pose(env_id)
            runtime_frame = self._runtimes[env_id].step(
                action_np,
                current_robot_quat_wxyz=current_robot_quat_wxyz,
                current_robot_xy_world=current_robot_xy_world,
            )
            self._vla_gripper_binary[env_id].copy_(
                self.torch.from_numpy(self.np.clip(runtime_frame.hand_binary, 0.0, 1.0)).to(
                    self.device, dtype=self.torch.float32
                )
            )
            mimic_obs = self.build_twist2_mimic_obs_v3(runtime_frame=runtime_frame, control_dt=self.control_dt)
            action_mimic[env_id] = self.torch.from_numpy(mimic_obs).to(self.device, dtype=self.torch.float32)
        return action_mimic

    def compute_full_action(self, active_mask):
        active_env_ids = [idx for idx, active in enumerate(active_mask) if active]
        full_action = self.default_joint_pos.clone()
        if not active_env_ids:
            return full_action

        root_state = self.env.scene["robot"].data.root_state_w
        ang_vel = self.env.scene["robot"].data.root_ang_vel_b
        joint_pos = self.env.scene["robot"].data.joint_pos
        joint_vel = self.env.scene["robot"].data.joint_vel

        quat = root_state[:, 3:7]
        rp = self._twist2_roll_pitch_from_quaternion(quat)
        dof_pos = joint_pos[:, self.twist2_action_indices]
        dof_vel = joint_vel[:, self.twist2_action_indices]
        dof_pos_delta = dof_pos - self.twist2_default_pos
        dof_vel = dof_vel.clone()
        dof_vel[:, self._twist2_ankle_idx] = 0.0

        obs_proprio = self.torch.cat(
            [
                ang_vel * 0.25,
                rp,
                dof_pos_delta,
                dof_vel * 0.05,
                self._twist2_last_action,
            ],
            dim=-1,
        )
        action_mimic = self._build_action_mimic_batch(active_env_ids)
        obs_full = self.torch.cat([action_mimic, obs_proprio], dim=-1)
        obs_hist = self._twist2_history.reshape(self.num_envs, -1)
        obs_buf = self.torch.cat([obs_full, obs_hist, action_mimic], dim=-1)

        action = self.policy(obs_buf)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        raw_action = self.torch.clip(action.to(self.device, dtype=self.torch.float32), -10.0, 10.0)
        target_29 = raw_action * 0.5 + self.twist2_default_pos
        self._twist2_last_action.copy_(raw_action)
        full_action[:, self.twist2_action_indices] = target_29

        if self._left_hand_target_idx_t is not None:
            pose_robot_name = "unitree_g1_with_hands"
            for env_id in active_env_ids:
                left_closed = bool(self._vla_gripper_binary[env_id, 0].item() >= self.gripper_threshold)
                right_closed = bool(self._vla_gripper_binary[env_id, 1].item() >= self.gripper_threshold)
                left_pose = self.DEFAULT_HAND_POSE[pose_robot_name]["left"]["close" if left_closed else "open"]
                right_pose = self.DEFAULT_HAND_POSE[pose_robot_name]["right"]["close" if right_closed else "open"]
                full_action[env_id, self._left_hand_target_idx_t] = self.torch.as_tensor(
                    left_pose[: len(self._left_hand_target_idx_t)], dtype=self.torch.float32, device=self.device
                )
                full_action[env_id, self._right_hand_target_idx_t] = self.torch.as_tensor(
                    right_pose[: len(self._right_hand_target_idx_t)], dtype=self.torch.float32, device=self.device
                )

        self._twist2_history = self.torch.roll(self._twist2_history, shifts=-1, dims=1)
        self._twist2_history[:, -1, :].copy_(obs_full)
        return full_action

    def step_sim(self, full_action, *, render: bool):
        from tools.get_reward import sync_reward_after_physics_step

        decimation = int(getattr(self.env.cfg, "decimation", 1))
        for idx in range(decimation):
            is_last_step = idx == decimation - 1
            self.env.scene["robot"].set_joint_position_target(full_action)
            self.env.scene.write_data_to_sim()
            self.env.sim.step(render=bool(render and is_last_step))
            self.env.scene.update(dt=self.env.physics_dt)
        sync_reward_after_physics_step(self.env)
        self.env.observation_manager.compute()


class MultiSimEvaluator:
    def __init__(self, simulation_app, env, env_cfg, args_cli, episode_specs, server_urls, SimpleVideoRecorder):
        self.simulation_app = simulation_app
        self.env = env
        self.env_cfg = env_cfg
        self.args_cli = args_cli
        self.pending_specs = list(episode_specs)
        self.finished_results = []
        self.SimpleVideoRecorder = SimpleVideoRecorder
        self.runtime = MultiEnvVLATwist2Runtime(env, args_cli, server_urls)
        self.slots = [SlotState(env_id=idx, server_url=server_urls[idx]) for idx in range(int(args_cli.num_envs))]
        for slot in self.slots:
            self._assign_next_job(slot)

    def _reset_env_slot(self, env_id: int, episode_seed: int):
        seed = int(episode_seed) & 0x7FFFFFFF
        _seed_runtime_rngs(seed)
        setattr(self.env_cfg, "seed", seed)
        setattr(self.env_cfg, "object_reset_seed_source", "env_seed")
        setattr(self.env_cfg, "_episode_runtime_seed", seed)
        setattr(self.env_cfg, "_episode_object_seed_counter", 0)
        setattr(self.env_cfg, "_current_episode_object_seed", None)
        setattr(self.env_cfg, "_current_episode_object_seed_source", "")
        runtime_cfg = getattr(self.env, "cfg", self.env_cfg)
        if runtime_cfg is not self.env_cfg:
            setattr(runtime_cfg, "seed", seed)
            setattr(runtime_cfg, "object_reset_seed_source", "env_seed")
            setattr(runtime_cfg, "_episode_runtime_seed", seed)
            setattr(runtime_cfg, "_episode_object_seed_counter", 0)
            setattr(runtime_cfg, "_current_episode_object_seed", None)
            setattr(runtime_cfg, "_current_episode_object_seed_source", "")
        self.env.reset(seed=seed, env_ids=[int(env_id)])

    def _assign_next_job(self, slot: SlotState):
        if not self.pending_specs:
            slot.current_spec = None
            slot.recorder = None
            slot.temp_video_path = None
            return

        spec = self.pending_specs.pop(0)
        if not spec.get("model_label"):
            spec["model_label"] = Path(spec.get("eval_model_path") or self.args_cli.model_path).stem
        slot.current_spec = spec
        slot.reset_metrics()
        self._reset_env_slot(slot.env_id, int(spec["episode_seed"]))
        self.runtime.reset_env_slot(slot.env_id, int(spec["episode_seed"]))

        result_path = Path(spec["result_json"]).expanduser().resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temp_video_path = result_path.parent / (
            f"slot_{slot.env_id}__{spec['model_label']}__seed_{spec['seed']}__repeat_{spec['repeat_idx']}__episode_{spec['episode_index']}__tmp.mp4"
        )
        if temp_video_path.exists():
            temp_video_path.unlink()
        slot.temp_video_path = temp_video_path
        slot.recorder = self.SimpleVideoRecorder(str(temp_video_path), fps=int(spec["video_fps"]))
        initial_frame = _capture_front_camera_rgb(self.env, slot.env_id)
        if initial_frame is not None:
            slot.recorder.add_frame(initial_frame)
        print(
            f"[sim_eval_vla_multisim] slot={slot.env_id} start seed={spec['seed']} repeat_idx={spec['repeat_idx']} "
            f"episode_seed={spec['episode_seed']} rss_mb={_get_process_rss_mb():.1f}"
        )

    def _finalize_slot(self, slot: SlotState, *, error: str = ""):
        spec = slot.current_spec
        if spec is None:
            return
        success_video_dir = Path(spec["success_video_dir"]).expanduser().resolve()
        failure_video_dir = Path(spec["failure_video_dir"]).expanduser().resolve()
        episode_name = (
            f"{spec['model_label']}__seed_{spec['seed']}__repeat_{spec['repeat_idx']}__episode_{spec['episode_index']}"
        )
        target_dir = success_video_dir if slot.success else failure_video_dir
        suffix = "success" if slot.success else slot.failure_reason
        video_path = _finalize_video(slot.recorder, target_dir, episode_name, suffix)
        payload = _build_result_payload(
            spec,
            slot.server_url,
            success=slot.success,
            failure_reason=slot.failure_reason,
            step_idx=slot.step_idx,
            terminal_step_idx=slot.terminal_step_idx,
            final_reward=slot.final_reward,
            final_reward_scaled=slot.final_reward_scaled,
            max_reward=slot.max_reward,
            max_reward_scaled=slot.max_reward_scaled,
            video_path=video_path,
            started_at=slot.started_at,
            error=error,
        )
        _write_result(Path(spec["result_json"]).expanduser().resolve(), payload)
        self.finished_results.append(payload)
        print(
            f"[sim_eval_vla_multisim] slot={slot.env_id} end seed={spec['seed']} repeat_idx={spec['repeat_idx']} "
            f"success={slot.success} reason={slot.failure_reason} rss_mb={_get_process_rss_mb():.1f}"
        )
        try:
            slot.recorder.close()
        except Exception:
            pass
        try:
            slot.recorder.clear()
        except Exception:
            pass
        self._assign_next_job(slot)
        _cleanup_episode_memory()

    def run(self) -> int:
        try:
            while self.simulation_app.is_running():
                active_mask = [slot.is_active() for slot in self.slots]
                if not any(active_mask):
                    break

                should_render = any(slot.is_active() and not self.runtime._chunk_queues[slot.env_id] for slot in self.slots)
                full_action = self.runtime.compute_full_action(active_mask)
                self.runtime.step_sim(full_action, render=should_render)

                for slot in self.slots:
                    if not slot.is_active():
                        continue
                    slot.step_idx += 1
                    frame = _capture_front_camera_rgb(self.env, slot.env_id)
                    if frame is not None:
                        slot.recorder.add_frame(frame)

                    if slot.post_remaining > 0:
                        slot.post_remaining -= 1
                        if slot.post_remaining == 0:
                            self._finalize_slot(slot)
                        continue

                    reward_info = _extract_reward_info_for_env(self.env, slot.env_id)
                    slot.final_reward = reward_info["raw_total"]
                    slot.final_reward_scaled = reward_info["scaled_total"]
                    slot.max_reward = max(slot.max_reward, slot.final_reward)
                    slot.max_reward_scaled = max(slot.max_reward_scaled, slot.final_reward_scaled)

                    if slot.final_reward >= 1.0:
                        slot.success = True
                        slot.failure_reason = "success"
                        slot.terminal_step_idx = slot.step_idx
                        slot.post_remaining = int(slot.current_spec["post_termination_record_steps"])
                        if slot.post_remaining <= 0:
                            self._finalize_slot(slot)
                        continue

                    if slot.step_idx >= int(slot.current_spec["max_steps"]):
                        slot.success = False
                        slot.failure_reason = "timeout"
                        slot.terminal_step_idx = slot.step_idx
                        slot.post_remaining = int(slot.current_spec["post_termination_record_steps"])
                        if slot.post_remaining <= 0:
                            self._finalize_slot(slot)
                        continue

                if self.env.sim.is_stopped():
                    for slot in self.slots:
                        if slot.is_active():
                            slot.success = False
                            slot.failure_reason = "sim_stopped"
                            slot.terminal_step_idx = slot.step_idx
                            self._finalize_slot(slot)
                    break
            return 0
        except KeyboardInterrupt:
            for slot in self.slots:
                if slot.is_active():
                    slot.success = False
                    slot.failure_reason = "interrupted"
                    slot.terminal_step_idx = slot.step_idx
                    self._finalize_slot(slot, error="KeyboardInterrupt")
            return 130
        except Exception as exc:
            for slot in self.slots:
                if slot.is_active():
                    slot.success = False
                    slot.failure_reason = "sim_error"
                    slot.terminal_step_idx = slot.step_idx
                    self._finalize_slot(slot, error=str(exc))
            raise


def main() -> int:
    parser = _build_parser()
    args_cli = parser.parse_args()
    _install_interrupt_handlers()
    _ensure_unique_multi_image_shm_name(args_cli)
    args_cli.enable_cameras = True
    args_cli.multi_gpu = False
    disable_multi_gpu_arg = "--/renderer/multiGpu/enabled=False"
    existing_kit_args = (getattr(args_cli, "kit_args", "") or "").split()
    if disable_multi_gpu_arg not in existing_kit_args:
        args_cli.kit_args = " ".join([*existing_kit_args, disable_multi_gpu_arg]).strip()
    _normalize_control_routing(args_cli)
    apply_task_runtime_profile(args_cli)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import tasks
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tasks.common_env_config import apply_env_config_yaml
    from tasks.common_runtime import apply_optional_runtime_augments

    SimpleVideoRecorder = _load_simple_video_recorder()

    episode_specs = _load_episode_specs(args_cli)
    server_urls = _load_server_urls(args_cli)
    env = None
    exit_code = 0

    try:
        first_spec = episode_specs[0]
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=int(args_cli.num_envs))
        env_cfg.env_name = args_cli.task
        apply_env_config_yaml(
            env_cfg,
            args_cli.env_config_yaml,
            task_name=args_cli.task,
            route_name=args_cli.gmt_backend or args_cli.action_source,
        )
        _seed_runtime_rngs(int(first_spec["episode_seed"]))
        setattr(env_cfg, "seed", int(first_spec["episode_seed"]) & 0x7FFFFFFF)
        setattr(env_cfg, "object_reset_seed_source", "env_seed")
        print(
            f"[sim_eval_vla_multisim] startup seed={first_spec['seed']} episodes={len(episode_specs)} num_envs={args_cli.num_envs}"
        )
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        _initialize_task_scene(env, env_cfg, args_cli, apply_optional_runtime_augments)
        env.reset(seed=int(first_spec["episode_seed"]))
        evaluator = MultiSimEvaluator(
            simulation_app,
            env,
            env_cfg,
            args_cli,
            episode_specs,
            server_urls,
            SimpleVideoRecorder,
        )
        exit_code = evaluator.run()
    finally:
        try:
            if env is not None:
                env.close()
        except Exception as exc:
            print(f"[sim_eval_vla_multisim] env close failed: {exc}")
        try:
            simulation_app.close()
        except Exception as exc:
            print(f"[sim_eval_vla_multisim] simulation_app close failed: {exc}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
