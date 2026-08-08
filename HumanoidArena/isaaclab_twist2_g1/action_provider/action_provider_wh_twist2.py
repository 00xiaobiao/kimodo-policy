# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
from action_provider.action_base import ActionProvider, ReplayComplete
from action_provider.reset_control import get_input_ready_key
from collections import deque
from typing import Optional
import torch
import os
import json
import redis
import math
import onnxruntime as ort
from dds.dds_master import dds_manager
from dds.sharedmemorymanager import SharedMemoryManager
import time
import threading
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
import ast
import queue
import copy
import numpy as np
from pathlib import Path
from action_provider.lerobot_vla_http_client import LeRobotVLAHttpClient
from action_provider.vision_video import write_rgb_video_mp4
from action_provider.vla_robot_current_local_runtime_v3 import (
    VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM,
    VLA_ROBOT_CURRENT_LOCAL_V3_STATE_DIM,
    UnifiedRobotCurrentLocalActionRuntimeV3,
    build_twist2_mimic_obs_v3,
    build_vla_rotlocal_v3_observation_state,
)
from action_provider.vla_smpl_runtime import (
    Twist2ActionMimicRecorder,
    build_vla_observation_state,
    reorder_canonical_to_twist2_29,
    reorder_twist2_to_canonical_29,
)
from common_env_objects import (
    add_env_object_frame_arrays,
    add_episode_init_env_object_fields,
    collect_recordable_env_object_states,
    get_current_episode_object_seed_info,
    resolve_env_object_scene_key,
)
from pico_server.data_utils.params import DEFAULT_HAND_POSE
from tools.get_reward import get_step_reward_value

project_root = os.environ.get("PROJECT_ROOT")


def _extract_controller_binary_signals(controller_data: dict | None) -> dict[str, bool]:
    def _get_side_data(side_name: str) -> dict:
        if not isinstance(controller_data, dict):
            return {}
        side_data = controller_data.get(side_name, {})
        if not isinstance(side_data, dict):
            return {}
        return side_data

    def _get_grip_binary(side_name: str) -> bool:
        side_data = _get_side_data(side_name)
        if "grip_binary" in side_data:
            return bool(side_data.get("grip_binary"))
        try:
            if float(side_data.get("index_trig", 0.0)) > 0.5:
                return True
            if float(side_data.get("grip", 0.0)) > 0.5:
                return False
        except Exception:
            pass
        return False

    def _get_close_trigger_binary(side_name: str) -> bool:
        side_data = _get_side_data(side_name)
        if "close_trigger_binary" in side_data:
            return bool(side_data.get("close_trigger_binary"))
        try:
            return bool(float(side_data.get("index_trig", 0.0)) > 0.5)
        except Exception:
            return False

    def _get_open_trigger_binary(side_name: str) -> bool:
        side_data = _get_side_data(side_name)
        if "open_trigger_binary" in side_data:
            return bool(side_data.get("open_trigger_binary"))
        try:
            return bool(float(side_data.get("grip", 0.0)) > 0.5)
        except Exception:
            return False

    return {
        "left_grip_binary": _get_grip_binary("LeftController"),
        "right_grip_binary": _get_grip_binary("RightController"),
        "left_close_trigger_binary": _get_close_trigger_binary("LeftController"),
        "right_close_trigger_binary": _get_close_trigger_binary("RightController"),
        "left_open_trigger_binary": _get_open_trigger_binary("LeftController"),
        "right_open_trigger_binary": _get_open_trigger_binary("RightController"),
    }


def _decode_npz_scalar(value):
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def _decode_json_list_field(value) -> list:
    decoded = _decode_npz_scalar(value)
    if decoded in (None, ""):
        return []
    if isinstance(decoded, (bytes, bytearray)):
        decoded = decoded.decode("utf-8")
    if isinstance(decoded, str):
        return json.loads(decoded)
    if isinstance(decoded, list):
        return decoded
    raise TypeError(f"Unsupported JSON list payload type: {type(decoded)}")


def _reward_scalar(env) -> float:
    try:
        reward = get_step_reward_value(env)
        if isinstance(reward, torch.Tensor):
            if reward.numel() == 0:
                return 0.0
            return float(reward.detach().reshape(-1)[0].item())
        return float(reward)
    except Exception:
        return 0.0


def _reward_success_flag(reward: float, tol: float = 1e-6) -> bool:
    try:
        return float(reward) > float(tol)
    except Exception:
        return False


def _build_replay_controller_data(controller_binary: dict[str, bool]) -> dict[str, dict[str, float | bool]]:
    left_close = bool(controller_binary.get("left_close_trigger_binary", False))
    right_close = bool(controller_binary.get("right_close_trigger_binary", False))
    left_open = bool(controller_binary.get("left_open_trigger_binary", False))
    right_open = bool(controller_binary.get("right_open_trigger_binary", False))
    return {
        "LeftController": {
            "grip_binary": bool(controller_binary.get("left_grip_binary", False)),
            "close_trigger_binary": left_close,
            "open_trigger_binary": left_open,
            "index_trig": 1.0 if left_close else 0.0,
            "grip": 1.0 if left_open else 0.0,
        },
        "RightController": {
            "grip_binary": bool(controller_binary.get("right_grip_binary", False)),
            "close_trigger_binary": right_close,
            "open_trigger_binary": right_open,
            "index_trig": 1.0 if right_close else 0.0,
            "grip": 1.0 if right_open else 0.0,
        },
    }


def _store_optional_camera_stream(
    organized: dict,
    *,
    save_dir: str,
    task_name: str,
    timestamp_us: int,
    fps: float,
    frames: list[np.ndarray],
    depth_frames: list[np.ndarray],
    frame_indices: list[int],
    rgb_key: str,
    depth_key: str,
    indices_key: str,
    video_path_key: str,
    video_fps_key: str,
    video_num_frames_key: str,
    video_suffix: str,
) -> bool:
    if not frames:
        return False
    video_basename = f"{task_name}_{timestamp_us}_{video_suffix}.mp4"
    video_relpath = os.path.join("videos", video_basename)
    video_abspath = os.path.join(save_dir, video_relpath)
    written = write_rgb_video_mp4(frames, video_abspath, fps=fps)
    organized[video_path_key] = np.array(video_relpath)
    organized[video_fps_key] = np.array(fps, dtype=np.float32)
    organized[video_num_frames_key] = np.array(written, dtype=np.int32)
    organized[indices_key] = np.array(frame_indices, dtype=np.int32)
    if depth_frames:
        organized[depth_key] = np.asarray(depth_frames, dtype=np.float16)
    return True


class RecordingManager:
    """Manages recording data collection and asynchronous saving to disk.

    This class handles:
    - Recording state management (start/save/cancel)
    - Data buffering during recording
    - Asynchronous saving to disk using a background thread
    - File naming with timestamps
    """

    def __init__(
        self,
        save_dir: str,
        task_name: str,
        max_frames: int = 10000,
        max_save_workers: int = 1,
        max_queue_size: int = 10,
    ):
        """Initialize the recording manager.

        Args:
            save_dir: Directory to save recording files
            task_name: Name of the task (used in filename)
            max_frames: Maximum number of frames to record (default: 10000, ~5min @ 30Hz)
        """
        self.save_dir = save_dir
        self.task_name = task_name
        self.max_frames = max_frames
        self.max_save_workers = max(1, int(max_save_workers))

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Recording state
        self.is_recording = False
        self.recording_buffer = []
        self.frame_count = 0

        # Async save queue and thread
        self.save_queue = queue.Queue(maxsize=max(1, int(max_queue_size)))  # Limit queue size to prevent memory overflow
        self.save_threads = []
        self.thread_running = False
        self._state_lock = threading.Lock()
        self._active_workers = 0

        # Start the save worker thread
        self._start_save_worker()

        print(f"[RecordingManager] Initialized with save_dir={save_dir}, task={task_name}")

    def _start_save_worker(self):
        """Start the background save worker thread."""
        self.thread_running = True
        for worker_idx in range(self.max_save_workers):
            save_thread = threading.Thread(
                target=self._save_worker,
                name=f"twist2-save-{worker_idx}",
                daemon=False,
            )
            save_thread.start()
            self.save_threads.append(save_thread)
        print(f"[RecordingManager] Save worker threads started: {len(self.save_threads)}")

    def _save_worker(self):
        """Background worker thread that saves data to disk."""
        while self.thread_running:
            try:
                # Wait for save task with timeout to allow checking thread_running
                task = self.save_queue.get(timeout=1.0)

                if task is None:  # Poison pill to stop thread
                    break

                # Unpack task
                data_buffer, timestamp_us, callback = task

                with self._state_lock:
                    self._active_workers += 1
                # Save to disk
                success = self._save_to_disk(data_buffer, timestamp_us)

                # Call callback to notify completion
                if callback is not None:
                    callback(success)

                # Mark task as done
                with self._state_lock:
                    self._active_workers = max(0, self._active_workers - 1)
                self.save_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[RecordingManager] Error in save worker: {e}")
                import traceback
                traceback.print_exc()

    def _save_to_disk(self, data_buffer: list, timestamp_us: int) -> bool:
        """Save recording data to disk.

        Args:
            data_buffer: List of recording data dictionaries
            timestamp_us: Timestamp in microseconds for filename

        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Generate filename with timestamp
            filename = f"{self.task_name}_{timestamp_us}.npz"
            # Note: np.savez_compressed automatically adds .npz extension, so use base name for temp
            temp_basename = f"{self.task_name}_{timestamp_us}_temp"
            filepath = os.path.join(self.save_dir, filename)
            temp_filepath = os.path.join(self.save_dir, temp_basename)  # Will become temp_basename.npz

            print(f"[RecordingManager] 💾 Saving {len(data_buffer)} frames to {filename}...")

            save_start = time.time()

            # Organize data for npz format
            # Convert list of dicts to dict of lists
            print(f"[RecordingManager] Organizing data...")
            organized_data = self._organize_data_for_save(data_buffer, timestamp_us)

            # Save to temporary file first
            # np.savez_compressed will automatically add .npz extension
            print(f"[RecordingManager] Writing to temporary file: {temp_basename}.npz")
            try:
                np.savez_compressed(temp_filepath, **organized_data)
                print(f"[RecordingManager] Temporary file written successfully")
            except Exception as e:
                print(f"[RecordingManager] ❌ Failed to write npz file: {e}")
                import traceback
                traceback.print_exc()
                raise

            # The actual temp file will have .npz extension
            actual_temp_filepath = temp_filepath + ".npz"

            # Check if temp file exists
            if not os.path.exists(actual_temp_filepath):
                raise FileNotFoundError(f"Temporary file was not created: {actual_temp_filepath}")

            # Rename to final filename (atomic operation)
            print(f"[RecordingManager] Renaming to final file: {filename}")
            os.rename(actual_temp_filepath, filepath)

            save_time = time.time() - save_start
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

            print(f"[RecordingManager] ✅ Saved successfully!")
            print(f"  - File: {filename}")
            print(f"  - Frames: {len(data_buffer)}")
            print(f"  - Size: {file_size_mb:.2f} MB")
            print(f"  - Time: {save_time:.2f}s")

            return True

        except Exception as e:
            print(f"[RecordingManager] ❌ Failed to save recording: {e}")
            import traceback
            traceback.print_exc()

            # Clean up temp file if it exists
            actual_temp_filepath = temp_filepath + ".npz"
            if os.path.exists(actual_temp_filepath):
                try:
                    os.remove(actual_temp_filepath)
                    print(f"[RecordingManager] Cleaned up temporary file")
                except:
                    pass

            return False

    def _organize_data_for_save(self, data_buffer: list, timestamp_us: int) -> dict:
        """Organize list of frame data into arrays for npz format.

        Args:
            data_buffer: List of recording data dictionaries (one per frame)

        Returns:
            Dictionary with organized arrays suitable for np.savez
        """
        num_frames = len(data_buffer)
        print(f"[RecordingManager] Organizing {num_frames} frames...")

        # Validate first frame structure
        if num_frames == 0:
            raise ValueError("Data buffer is empty")

        first_frame = data_buffer[0]
        last_frame = data_buffer[-1]
        print(f"[RecordingManager] First frame keys: {first_frame.keys()}")
        print(f"[RecordingManager] Task: {first_frame.get('task', 'N/A')}")

        # Initialize storage
        organized = {
            'schema_version': np.array("twist2_episode_v2"),
            'task': data_buffer[0]['task'],  # Task name (scalar)
            'num_frames': num_frames,

            # Human data
            'human_hand_left': np.zeros((num_frames, 7), dtype=np.float32),
            'human_hand_right': np.zeros((num_frames, 7), dtype=np.float32),
            'human_neck': np.zeros((num_frames, 2), dtype=np.float32),
            'pico_left_grip_binary': np.zeros(num_frames, dtype=np.bool_),
            'pico_right_grip_binary': np.zeros(num_frames, dtype=np.bool_),
            'pico_left_close_trigger_binary': np.zeros(num_frames, dtype=np.bool_),
            'pico_right_close_trigger_binary': np.zeros(num_frames, dtype=np.bool_),
            'pico_left_open_trigger_binary': np.zeros(num_frames, dtype=np.bool_),
            'pico_right_open_trigger_binary': np.zeros(num_frames, dtype=np.bool_),

            # Robot data
            'robot_qpos_before_decimation': np.zeros((num_frames, 29), dtype=np.float32),
            'robot_qvel_before_decimation': np.zeros((num_frames, 29), dtype=np.float32),
            'robot_root_position': np.zeros((num_frames, 3), dtype=np.float32),
            'robot_root_orientation': np.zeros((num_frames, 4), dtype=np.float32),
            # Root velocities - both local and world frame
            'robot_root_lin_vel_local': np.zeros((num_frames, 3), dtype=np.float32),
            'robot_root_ang_vel_local': np.zeros((num_frames, 3), dtype=np.float32),
            'robot_root_lin_vel_world': np.zeros((num_frames, 3), dtype=np.float32),
            'robot_root_ang_vel_world': np.zeros((num_frames, 3), dtype=np.float32),
            'robot_twist2_inference_qpos': np.zeros((num_frames, 29), dtype=np.float32),
            'robot_action_mimic': np.zeros((num_frames, 35), dtype=np.float32),
            'robot_obs_buf': np.zeros((num_frames, 1432), dtype=np.float32),  # 127*11+35 = 1432
            'vla_state': np.zeros((num_frames, VLA_ROBOT_CURRENT_LOCAL_V3_STATE_DIM), dtype=np.float32),
            'vla_state_root_rot6d': np.zeros((num_frames, 6), dtype=np.float32),
            'vla_state_dof_pos_29': np.zeros((num_frames, 29), dtype=np.float32),
            'vla_state_dof_vel_29': np.zeros((num_frames, 29), dtype=np.float32),
            'vla_action': np.zeros((num_frames, VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM), dtype=np.float32),
            'vla_action_root_xy_delta': np.zeros((num_frames, 2), dtype=np.float32),
            'vla_action_root_z': np.zeros((num_frames, 1), dtype=np.float32),
            'vla_action_root_rot6d': np.zeros((num_frames, 6), dtype=np.float32),
            'vla_action_joint_pos_29': np.zeros((num_frames, 29), dtype=np.float32),
            'vla_action_hand_binary': np.zeros((num_frames, 2), dtype=np.float32),
            'vla_action_hand_binary_2': np.zeros((num_frames, 2), dtype=np.float32),

            # System data
            'system_control_frequency': np.zeros(num_frames, dtype=np.float32),
            'system_decimation': np.zeros(num_frames, dtype=np.int32),
            'system_physics_dt': np.zeros(num_frames, dtype=np.float32),
            'system_timestamp': np.zeros(num_frames, dtype=np.float64),
        }
        episode_init_meta = first_frame.get("episode_init_meta", {})
        if isinstance(episode_init_meta, dict):
            seed_value = episode_init_meta.get("object_seed")
            if seed_value is not None:
                organized["episode_object_seed"] = np.array(int(seed_value), dtype=np.int64)
            seed_source = episode_init_meta.get("object_seed_source")
            if seed_source:
                organized["episode_object_seed_source"] = np.array(str(seed_source))
        rerecord_final_reward = last_frame.get("rerecord_final_reward")
        if rerecord_final_reward is not None:
            organized["rerecord_final_reward"] = np.array(float(rerecord_final_reward), dtype=np.float32)
        rerecord_max_reward = last_frame.get("rerecord_max_reward")
        if rerecord_max_reward is not None:
            organized["rerecord_max_reward"] = np.array(float(rerecord_max_reward), dtype=np.float32)
        rerecord_any_success = last_frame.get("rerecord_any_success")
        if rerecord_any_success is not None:
            organized["rerecord_any_success"] = np.array(bool(rerecord_any_success), dtype=np.bool_)

        first_robot = first_frame.get('robot', {})
        first_torque = first_robot.get('applied_torque_before_decimation')
        if first_torque is not None:
            organized['robot_applied_torque_before_decimation'] = np.zeros(
                (num_frames, len(first_torque)), dtype=np.float32
            )
        first_contact_forces = first_robot.get('body_net_contact_forces')
        if first_contact_forces is not None:
            organized['robot_body_net_contact_forces'] = np.zeros(
                (num_frames,) + np.asarray(first_contact_forces).shape, dtype=np.float32
            )

        # Store observation semantics (same for all frames, store once)
        organized['observation_semantics'] = json.dumps(data_buffer[0]['robot']['observation']['semantics'])

        # Lists for variable-size data
        human_smplx_list = []
        human_info_list = []
        vision_indices = list(range(num_frames))
        front_rgb_list = []
        front_depth_list = []
        front_frame_indices = []
        left_wrist_rgb_list = []
        left_wrist_depth_list = []
        left_wrist_frame_indices = []
        right_wrist_rgb_list = []
        right_wrist_depth_list = []
        right_wrist_frame_indices = []

        # Fill arrays frame by frame
        for i, frame_data in enumerate(data_buffer):
            # Human data
            organized['human_hand_left'][i] = frame_data['human']['hand_control']['left']
            organized['human_hand_right'][i] = frame_data['human']['hand_control']['right']
            organized['human_neck'][i] = frame_data['human']['hand_control']['neck']
            controller_binary = frame_data['human'].get('controller_binary', {})
            organized['pico_left_grip_binary'][i] = bool(controller_binary.get('left_grip_binary', False))
            organized['pico_right_grip_binary'][i] = bool(controller_binary.get('right_grip_binary', False))
            organized['pico_left_close_trigger_binary'][i] = bool(controller_binary.get('left_close_trigger_binary', False))
            organized['pico_right_close_trigger_binary'][i] = bool(controller_binary.get('right_close_trigger_binary', False))
            organized['pico_left_open_trigger_binary'][i] = bool(controller_binary.get('left_open_trigger_binary', False))
            organized['pico_right_open_trigger_binary'][i] = bool(controller_binary.get('right_open_trigger_binary', False))

            # Store SMPLX data (variable size, store as list)
            human_smplx_list.append(frame_data['human']['smplx_data_before_gmr'])
            human_info_list.append(frame_data['human']['human_info'])

            # Robot data
            organized['robot_qpos_before_decimation'][i] = frame_data['robot']['qpos_before_decimation']
            organized['robot_qvel_before_decimation'][i] = frame_data['robot']['qvel_before_decimation']
            organized['robot_root_position'][i] = frame_data['robot']['root_position']
            organized['robot_root_orientation'][i] = frame_data['robot']['root_orientation']
            # Root velocities - both local and world frame
            organized['robot_root_lin_vel_local'][i] = frame_data['robot']['root_lin_vel_local']
            organized['robot_root_ang_vel_local'][i] = frame_data['robot']['root_ang_vel_local']
            organized['robot_root_lin_vel_world'][i] = frame_data['robot']['root_lin_vel_world']
            organized['robot_root_ang_vel_world'][i] = frame_data['robot']['root_ang_vel_world']
            organized['robot_twist2_inference_qpos'][i] = frame_data['robot']['twist2_inference_qpos']
            organized['robot_action_mimic'][i] = frame_data['robot']['action_mimic']
            organized['robot_obs_buf'][i] = frame_data['robot']['observation']['obs_buf']
            organized['vla_state'][i] = frame_data['robot']['vla_state']
            organized['vla_state_root_rot6d'][i] = frame_data['robot']['vla_state'][:6]
            organized['vla_state_dof_pos_29'][i] = frame_data['robot']['vla_state'][6:35]
            organized['vla_state_dof_vel_29'][i] = frame_data['robot']['vla_state'][35:64]
            organized['vla_action'][i] = frame_data['robot']['vla_action']
            organized['vla_action_root_xy_delta'][i] = frame_data['robot']['vla_action'][:2]
            organized['vla_action_root_z'][i] = frame_data['robot']['vla_action'][2:3]
            organized['vla_action_root_rot6d'][i] = frame_data['robot']['vla_action'][3:9]
            organized['vla_action_joint_pos_29'][i] = frame_data['robot']['vla_action'][9:38]
            organized['vla_action_hand_binary'][i] = frame_data['robot']['vla_action'][38:40]
            organized['vla_action_hand_binary_2'][i] = frame_data['robot']['vla_action'][38:40]
            if 'robot_applied_torque_before_decimation' in organized:
                torque = frame_data['robot'].get('applied_torque_before_decimation')
                if torque is not None:
                    organized['robot_applied_torque_before_decimation'][i] = torque
            if 'robot_body_net_contact_forces' in organized:
                contact_forces = frame_data['robot'].get('body_net_contact_forces')
                if contact_forces is not None:
                    organized['robot_body_net_contact_forces'][i] = contact_forces

            # Vision data (only store selected frames)
            if i in vision_indices:
                vision = frame_data['robot']['vision']

                rgb = vision.get('rgb')
                depth = vision.get('depth')
                if rgb is not None:
                    front_rgb_list.append(rgb)
                    front_frame_indices.append(i)
                if depth is not None:
                    front_depth_list.append(depth)

                left_rgb = vision.get('left_wrist_rgb')
                left_depth = vision.get('left_wrist_depth')
                if left_rgb is not None:
                    left_wrist_rgb_list.append(left_rgb)
                    left_wrist_frame_indices.append(i)
                if left_depth is not None:
                    left_wrist_depth_list.append(left_depth)

                right_rgb = vision.get('right_wrist_rgb')
                right_depth = vision.get('right_wrist_depth')
                if right_rgb is not None:
                    right_wrist_rgb_list.append(right_rgb)
                    right_wrist_frame_indices.append(i)
                if right_depth is not None:
                    right_wrist_depth_list.append(right_depth)

            # System data
            organized['system_control_frequency'][i] = frame_data['system']['control_frequency']
            organized['system_decimation'][i] = frame_data['system']['decimation']
            organized['system_physics_dt'][i] = frame_data['system']['physics_dt']
            organized['system_timestamp'][i] = frame_data['system']['timestamp']

        # Add variable-size data
        organized['human_smplx_data'] = json.dumps(human_smplx_list)
        organized['human_info_data'] = json.dumps(human_info_list)
        add_env_object_frame_arrays(organized, data_buffer)
        add_episode_init_env_object_fields(organized, first_frame.get("episode_init_env"))

        # Add vision data
        if front_rgb_list:
            fps = float(np.clip(np.nanmedian(organized["system_control_frequency"]), 1.0, 240.0))
            organized['vision_storage_format'] = np.array("video_v1")
            _store_optional_camera_stream(
                organized,
                save_dir=self.save_dir,
                task_name=self.task_name,
                timestamp_us=timestamp_us,
                fps=fps,
                frames=front_rgb_list,
                depth_frames=front_depth_list,
                frame_indices=front_frame_indices,
                rgb_key='vision_rgb',
                depth_key='vision_depth',
                indices_key='vision_frame_indices',
                video_path_key='vision_rgb_video_path',
                video_fps_key='vision_rgb_video_fps',
                video_num_frames_key='vision_rgb_video_num_frames',
                video_suffix='front_rgb',
            )
            if _store_optional_camera_stream(
                organized,
                save_dir=self.save_dir,
                task_name=self.task_name,
                timestamp_us=timestamp_us,
                fps=fps,
                frames=left_wrist_rgb_list,
                depth_frames=left_wrist_depth_list,
                frame_indices=left_wrist_frame_indices,
                rgb_key='vision_left_wrist_rgb',
                depth_key='vision_left_wrist_depth',
                indices_key='vision_left_wrist_frame_indices',
                video_path_key='vision_left_wrist_rgb_video_path',
                video_fps_key='vision_left_wrist_rgb_video_fps',
                video_num_frames_key='vision_left_wrist_rgb_video_num_frames',
                video_suffix='left_wrist_rgb',
            ):
                organized['schema_version'] = np.array("twist2_episode_v3_multicam")
            if _store_optional_camera_stream(
                organized,
                save_dir=self.save_dir,
                task_name=self.task_name,
                timestamp_us=timestamp_us,
                fps=fps,
                frames=right_wrist_rgb_list,
                depth_frames=right_wrist_depth_list,
                frame_indices=right_wrist_frame_indices,
                rgb_key='vision_right_wrist_rgb',
                depth_key='vision_right_wrist_depth',
                indices_key='vision_right_wrist_frame_indices',
                video_path_key='vision_right_wrist_rgb_video_path',
                video_fps_key='vision_right_wrist_rgb_video_fps',
                video_num_frames_key='vision_right_wrist_rgb_video_num_frames',
                video_suffix='right_wrist_rgb',
            ):
                organized['schema_version'] = np.array("twist2_episode_v3_multicam")

        return organized

    def start_recording(self):
        """Start a new recording session."""
        if self.is_recording:
            print(f"[RecordingManager] ⚠️ Already recording, ignoring start command")
            return

        self.is_recording = True
        self.recording_buffer = []
        self.frame_count = 0
        print(f"[RecordingManager] 🔴 Recording started")

    def add_frame(self, frame_data: dict):
        """Add a frame to the recording buffer.

        Args:
            frame_data: Dictionary containing all recording data for this frame
        """
        if not self.is_recording:
            return

        # Check frame limit
        if self.frame_count >= self.max_frames:
            print(f"[RecordingManager] ⚠️ Max frames ({self.max_frames}) reached, stopping recording")
            self.save_recording()
            return

        # Deep copy to avoid data corruption from subsequent modifications
        frame_copy = copy.deepcopy(frame_data)
        self.recording_buffer.append(frame_copy)
        self.frame_count += 1

    def save_recording(self, completion_callback=None):
        """Save the current recording and stop recording.

        Args:
            completion_callback: Optional callback function(success: bool) called when save completes
        """
        if not self.is_recording:
            print(f"[RecordingManager] ⚠️ Not recording, nothing to save")
            return

        if len(self.recording_buffer) == 0:
            print(f"[RecordingManager] ⚠️ Recording buffer is empty, nothing to save")
            self.is_recording = False
            return

        # Stop recording
        self.is_recording = False

        # Generate timestamp (microseconds for uniqueness)
        timestamp_us = int(time.time() * 1_000_000)

        # Check queue size
        if self.save_queue.full():
            print(f"[RecordingManager] ⚠️ Save queue is full, waiting for previous saves to complete...")

        # Queue the save task (this will block if queue is full)
        print(f"[RecordingManager] 📦 Queuing {len(self.recording_buffer)} frames for save...")
        self.save_queue.put((self.recording_buffer, timestamp_us, completion_callback))

        # Clear buffer
        self.recording_buffer = []
        self.frame_count = 0

        print(f"[RecordingManager] 💾 Recording queued for save (timestamp: {timestamp_us})")

    def cancel_recording(self):
        """Cancel the current recording without saving."""
        if not self.is_recording:
            print(f"[RecordingManager] ⚠️ Not recording, nothing to cancel")
            return

        self.is_recording = False
        frame_count = len(self.recording_buffer)
        self.recording_buffer = []
        self.frame_count = 0

        print(f"[RecordingManager] ❌ Recording cancelled ({frame_count} frames discarded)")

    def get_active_worker_count(self) -> int:
        with self._state_lock:
            return int(self._active_workers)

    def get_pending_save_count(self) -> int:
        return int(self.save_queue.qsize() + self.get_active_worker_count())

    def shutdown(self):
        """Shutdown the recording manager and wait for pending saves."""
        print(f"[RecordingManager] Shutting down...")

        # Stop recording if active
        if self.is_recording:
            print(f"[RecordingManager] Recording in progress, cancelling...")
            self.cancel_recording()

        # Wait for queue to empty
        if not self.save_queue.empty():
            print(f"[RecordingManager] Waiting for {self.save_queue.qsize()} pending saves...")
            self.save_queue.join()

        # Stop worker thread
        self.thread_running = False
        for _ in self.save_threads:
            self.save_queue.put(None)  # Poison pill

        for save_thread in self.save_threads:
            if save_thread.is_alive():
                save_thread.join(timeout=10.0)
                if save_thread.is_alive():
                    print(f"[RecordingManager] ⚠️ Save thread did not stop gracefully")

        print(f"[RecordingManager] Shutdown complete")


class TWIST2ActionProvider(ActionProvider):
    """Action provider based on DDS"""

    def __init__(self, env, args_cli):
        super().__init__("TWIST2ActionProvider")

        # Set random seed for reproducibility
        if hasattr(args_cli, 'seed') and args_cli.seed is not None:
            import random
            import numpy as np
            seed = args_cli.seed
            print(f"[{self.name}] Setting random seed: {seed}")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            # Enable deterministic mode for PyTorch
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Store seed for ONNX Runtime configuration
            self.onnx_seed = seed
        else:
            self.onnx_seed = None
            print(f"[{self.name}] No seed specified, using non-deterministic mode")

        self.enable_robot = args_cli.robot_type
        self.enable_gripper = args_cli.enable_dex1_dds
        self.enable_dex3 = args_cli.enable_dex3_dds
        self.enable_inspire = args_cli.enable_inspire_dds
        self.wh = args_cli.enable_wholebody_dds
        self._replay_file = getattr(args_cli, "replay_file", "")
        self._replay_mode = self._normalize_replay_mode(getattr(args_cli, "replay_mode", "inference_replay"))
        self._replay_loop = bool(getattr(args_cli, "replay_loop", False))
        self._replay_enabled = bool(self._replay_file)
        self._record_during_replay = bool(getattr(args_cli, "record_during_replay", False))
        self._exit_when_replay_complete = bool(getattr(args_cli, "exit_when_replay_complete", False))
        self._input_source = getattr(args_cli, "input_source", "") or ""
        self._use_lerobot_vla = self._input_source == "vla"
        self._lerobot_server_url = getattr(args_cli, "lerobot_server_url", "") or ""
        self._lerobot_server_timeout = float(getattr(args_cli, "lerobot_server_timeout", 5.0))
        self._lerobot_server_verify_ssl = bool(getattr(args_cli, "lerobot_server_verify_ssl", False))
        self._lerobot_policy = None
        self._lerobot_preprocessor = None
        self._lerobot_postprocessor = None
        self._lerobot_predict_action = None
        self._lerobot_device = None
        self._lerobot_http_client = None
        self._vla_max_root_delta_deg = float(getattr(args_cli, "vla_max_root_delta_deg", 0.0) or 0.0)
        self._lerobot_vla_runtime = UnifiedRobotCurrentLocalActionRuntimeV3(
            max_root_delta_deg=self._vla_max_root_delta_deg if self._vla_max_root_delta_deg > 0 else None
        )
        if self._vla_max_root_delta_deg > 0:
            print(f"[{self.name}] VLA max_root_delta_deg={self._vla_max_root_delta_deg}")
        self._vla_initial_robot_quat_wxyz: np.ndarray | None = None
        self._lerobot_gripper_threshold = float(getattr(args_cli, "lerobot_gripper_threshold", 0.5))
        self._latest_vla_action = None
        self._lerobot_action_chunk_queue = deque()
        self._lerobot_observation_state_segment = deque()
        self._lerobot_observation_state_last_step = -1
        self._twist2_onnx_requested_device = str(getattr(args_cli, "device", "") or "")
        self._twist2_onnx_device_id = self._resolve_onnx_device_id(self._twist2_onnx_requested_device)
        print(
            f"[{self.name}] TWIST2 ONNX requested_device={self._twist2_onnx_requested_device or 'auto'} "
            f"resolved_cuda_device_id={self._twist2_onnx_device_id}"
        )
        self.policy_path = self._resolve_policy_path(args_cli.model_path)
        if (not self._replay_enabled or self._replay_mode != "direct_replay") and not os.path.exists(self.policy_path):
            raise FileNotFoundError(f"[{self.name}] Policy file not found: {self.policy_path}")
        if self._use_lerobot_vla and self._replay_enabled:
            raise ValueError(f"[{self.name}] input_source=vla and replay_file are mutually exclusive")
        if self._record_during_replay and not self._replay_enabled:
            raise ValueError(f"[{self.name}] record_during_replay requires replay_file")
        self.env = env
        self.task_name = args_cli.task  # Store task name for recording

        # Initialize RecordingManager
        self.recording_manager = RecordingManager(
            save_dir=args_cli.recording_save_dir,
            task_name=args_cli.task,
            max_frames=10000,  # ~5 minutes @ 30Hz
            max_save_workers=int(getattr(args_cli, "recording_save_workers", 1)),
            max_queue_size=int(getattr(args_cli, "recording_save_queue_size", 10)),
        )

        # Recording will start on first get_action() call (after env.reset())
        # This ensures Frame 0 captures real physics state, not default values
        self._should_start_recording_on_first_call = (not self._replay_enabled) or self._record_during_replay
        if self._replay_enabled and not self._record_during_replay:
            print(f"[{self.name}] 🎬 Recording auto-start disabled during replay")
        else:
            print(f"[{self.name}] 🔴 AUTO-START RECORDING ENABLED")
            print(f"[{self.name}] Recording will start on first get_action() call")
            print(f"[{self.name}] This ensures Frame 0 captures real physics state")
        if self._record_during_replay:
            print(f"[{self.name}] 🔁 Replay rerecord enabled")

        # Debug: Fix root in air for PID tuning
        # Set to True to fix robot root at a fixed height, preventing falls during teleop debugging
        # NOTE: Disabled because without ground contact, robot lacks damping force and may move erratically
        self._debug_fix_root_in_air = False
        if self._debug_fix_root_in_air:
            # Fixed position: [x, y, z] in world frame (z=0.9m keeps robot suspended in air)
            self._debug_fixed_root_position = torch.tensor([0.0, 0.0, 0.9], device=self.env.device, dtype=torch.float32)
            # Fixed orientation: [w, x, y, z] quaternion (identity = upright)
            self._debug_fixed_root_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.env.device, dtype=torch.float32)
            print(f"[{self.name}] 🔧 DEBUG MODE: Root fixed in air at z={self._debug_fixed_root_position[2]:.2f}m")
            print(f"[{self.name}] 🔧 This prevents falling during PID tuning and teleop debugging")
            print(f"[{self.name}] 🔧 Set self._debug_fix_root_in_air = False to disable")

        # Initialize DDS communication
        self.robot_dds = None
        self.gripper_dds = None
        self.dex3_dds = None
        self.inspire_dds = None
        self.run_command = None
        self._setup_dds()
        self._setup_joint_mapping()

        # --- TWIST2 (Redis teleop / motion tracker) quick integration ---
        self.redis_client = None
        self.redis_pipeline = None
        self._redis_state_publish_disabled = False
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_pipeline = self.redis_client.pipeline()
        except Exception as e:
            print(f"[{self.name}] Redis init failed: {e}")
        self._input_ready_key = get_input_ready_key("twist2")
        self._input_ready_timestamp_ms = 0
        self._input_ready_epoch_id = -1
        self._stale_input_drop_logged_epoch = -1

        # TWIST2 observation sizes (from server_low_level_g1_sim.py)
        self.n_mimic_obs = 35
        self.n_obs_single = 127  # 35 + 92
        self.history_len = 10
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs  # 127*11+35 = 1432

        # Buffers
        self._twist2_history = torch.zeros(self.history_len, self.n_obs_single, device=self.env.device,
                                           dtype=torch.float32)
        self._twist2_last_action = torch.zeros(1, 29, device=self.env.device, dtype=torch.float32)
        self._twist2_obs_buf = torch.zeros(1, self.total_obs_size, device=self.env.device, dtype=torch.float32)
        self._twist2_hand_dim = 7
        self._twist2_neck_dim = 2
        self._twist2_action_hand_left = torch.zeros(1, self._twist2_hand_dim, device=self.env.device,
                                                    dtype=torch.float32)
        self._twist2_action_hand_right = torch.zeros(1, self._twist2_hand_dim, device=self.env.device,
                                                     dtype=torch.float32)
        self._twist2_action_neck = torch.zeros(1, self._twist2_neck_dim, device=self.env.device, dtype=torch.float32)
        self._twist2_hand_valid = False
        self._vla_gripper_binary = torch.zeros(1, 2, device=self.env.device, dtype=torch.float32)

        # Human SMPLX data (before GMR retargeting) storage
        self._twist2_human_smplx_data = None
        self._twist2_human_smplx_valid = False

        # Human info (height, etc.) storage
        self._twist2_human_info = None
        self._twist2_human_info_valid = False
        self._twist2_controller_data = None

        # Recording control state
        self._recording_active = False
        self._recording_command = "none"  # "none", "start", "save", "cancel"
        self._latest_recording_control = None
        self._latest_recording_control_sequence = -1

        # Display state for overlay (persists until save completes)
        self._recording_display_state = "idle"  # "idle", "recording", "saving", "saved", "discard"
        self._recording_display_counter = 0  # Counter for how long to show saved/discard after completion
        self._recording_display_duration = 10  # Show saved/discard for 60 frames (~2 seconds @ 30Hz)
        self._save_in_progress = False  # Track if save is currently in progress

        # Thread-safe flag for save completion (set by background thread, read by main thread)
        self._save_completion_state = None  # None, "success", or "failure"
        self._pending_save_jobs = 0

        self._replay_data_qpos = None
        self._replay_recorded_joint_pos = None
        self._replay_data_obs = None
        self._replay_data_hand_left = None
        self._replay_data_hand_right = None
        self._replay_data_neck = None
        self._replay_human_smplx_data = []
        self._replay_human_info_data = []
        self._replay_pico_left_grip_binary = None
        self._replay_pico_right_grip_binary = None
        self._replay_pico_left_close_trigger_binary = None
        self._replay_pico_right_close_trigger_binary = None
        self._replay_pico_left_open_trigger_binary = None
        self._replay_pico_right_open_trigger_binary = None
        self._replay_object_states = {}
        self._replay_num_frames = 0
        self._replay_cursor = 0
        self._replay_joint_mae_sum = 0.0
        self._replay_joint_mae_count = 0
        self._replay_joint_err_log_interval = 10
        self._replay_object_err_sums = {}
        self._replay_object_err_counts = {}
        self._replay_completion_requested = False
        self._replay_reward_max = None
        self._replay_any_success = False
        self._episode_init_env_state = self._collect_env_state()

        if self._replay_enabled:
            self._setup_local_replay()
            print(
                f"[{self.name}] 🎬 Local replay enabled  "
                f"file={self._replay_file}  mode={self._replay_mode}  loop={self._replay_loop}"
            )

        # Reset control state
        self._reset_requested = False  # Flag to indicate reset is requested
        self._waiting_for_reset_complete = False  # Flag to indicate waiting for reset completion
        self._reset_complete_received = False  # Flag set when reset complete signal received

        # Debug control
        self._debug_smpl_data = False  # Set to True to enable SMPL data debug output
        self._debug_counter = 0
        self._debug_interval = 100  # Print debug info every N steps

        # Default 35D mimic_obs when no Redis data available (prevents falling)
        # Structure: [xy_vel(2), z_pos(1), roll_pitch(2), yaw_vel(1), joints(29)] = 35D
        self._default_mimic_obs = torch.tensor([
            0.0, 0.0,  # xy velocity
            0.8,       # z position
            0.0, 0.0,  # roll/pitch
            0.0,       # yaw angular velocity
            # 29 DOF joint positions (matching TWIST2 default_dof_pos)
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
            0.0, 0.0, 0.0,                   # waist (3)
            # 0.0, 0.4, 0.0, 0.05, 0.0, 0.0, 0.0,  # left arm (7)
            0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,  # left arm (7)
            # 0.0, -0.4, 0.0, 0.05, 0.0, 0.0, 0.0, # right arm (7)
            0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0, # right arm (7)
        ], device=self.env.device, dtype=torch.float32).unsqueeze(0)  # [1, 35]

        # Indices used in TWIST2
        self._twist2_ankle_idx = [4, 5, 10, 11]

        self.policy = None
        if not self._replay_enabled or self._replay_mode != "direct_replay":
            self.policy = self.load_policy(self.policy_path)
        if self._use_lerobot_vla:
            self._setup_lerobot_vla(args_cli)

        # 预计算索引张量与复用缓冲
        device = self.env.device
        if hasattr(self, "arm_joint_mapping") and self.arm_joint_mapping:
            self._arm_target_indices = [self.joint_to_index[name] for name in self.arm_joint_mapping.keys()]
            self._arm_source_indices = [idx + 15 for idx in self.arm_joint_mapping.values()]
            self._arm_target_idx_t = torch.tensor(self._arm_target_indices, dtype=torch.long, device=device)
            self._arm_source_idx_t = torch.tensor(self._arm_source_indices, dtype=torch.long, device=device)
        if self.enable_gripper:
            self._gripper_target_indices = [self.joint_to_index[name] for name in self.gripper_joint_mapping.keys()]
            self._gripper_source_indices = [idx for idx in self.gripper_joint_mapping.values()]
            self._gripper_target_idx_t = torch.tensor(self._gripper_target_indices, dtype=torch.long, device=device)
            self._gripper_source_idx_t = torch.tensor(self._gripper_source_indices, dtype=torch.long, device=device)
        if self.enable_dex3:
            self._left_hand_target_indices = [self.joint_to_index[name] for name in self.left_hand_joint_mapping.keys()]
            self._left_hand_source_indices = [idx for idx in self.left_hand_joint_mapping.values()]
            self._right_hand_target_indices = [self.joint_to_index[name] for name in
                                               self.right_hand_joint_mapping.keys()]
            self._right_hand_source_indices = [idx for idx in self.right_hand_joint_mapping.values()]
            self._left_hand_target_idx_t = torch.tensor(self._left_hand_target_indices, dtype=torch.long, device=device)
            self._left_hand_source_idx_t = torch.tensor(self._left_hand_source_indices, dtype=torch.long, device=device)
            self._right_hand_target_idx_t = torch.tensor(self._right_hand_target_indices, dtype=torch.long,
                                                         device=device)
            self._right_hand_source_idx_t = torch.tensor(self._right_hand_source_indices, dtype=torch.long,
                                                         device=device)
        if self.enable_inspire:
            self._inspire_target_indices = [self.joint_to_index[name] for name in
                                            self.inspire_hand_joint_mapping.keys()]
            self._inspire_source_indices = [idx for idx in self.inspire_hand_joint_mapping.values()]
            self._inspire_special_target_indices = [self.joint_to_index[name] for name in
                                                    self.special_joint_mapping.keys()]
            self._inspire_special_source_indices = [spec[0] for spec in self.special_joint_mapping.values()]
            self._inspire_special_scales = torch.tensor([spec[1] for spec in self.special_joint_mapping.values()],
                                                        dtype=torch.float32)
            self._inspire_target_idx_t = torch.tensor(self._inspire_target_indices, dtype=torch.long, device=device)
            self._inspire_source_idx_t = torch.tensor(self._inspire_source_indices, dtype=torch.long, device=device)
            self._inspire_special_target_idx_t = torch.tensor(self._inspire_special_target_indices, dtype=torch.long,
                                                              device=device)
            self._inspire_special_source_idx_t = torch.tensor(self._inspire_special_source_indices, dtype=torch.long,
                                                              device=device)
            self._inspire_special_scales_t = self._inspire_special_scales.to(device)
        if hasattr(self, "twist2_action_indices"):
            self._twist2_action_idx_t = torch.tensor(self.twist2_action_indices, dtype=torch.long, device=device)

        self._full_action_buf = torch.zeros(len(self.all_joint_names), device=device, dtype=torch.float32)
        self._positions_buf = torch.empty(29, device=device, dtype=torch.float32)
        if self.enable_gripper:
            self._gripper_buf = torch.empty(2, device=device, dtype=torch.float32)
        if self.enable_dex3:
            self._left_hand_buf = torch.empty(len(self._left_hand_source_indices), device=device, dtype=torch.float32)
            self._right_hand_buf = torch.empty(len(self._right_hand_source_indices), device=device, dtype=torch.float32)
        if self.enable_inspire:
            self._inspire_buf = torch.empty(12, device=device, dtype=torch.float32)

    def _setup_dds(self):
        """Setup DDS communication"""
        print(f"enable_robot: {self.enable_robot}")
        print(f"enable_gripper: {self.enable_gripper}")
        print(f"enable_dex3: {self.enable_dex3}")
        try:
            if self.enable_robot == "g129":
                self.robot_dds = dds_manager.get_object("g129")
            if self.enable_gripper:
                self.gripper_dds = dds_manager.get_object("dex1")
            elif self.enable_dex3:
                self.dex3_dds = dds_manager.get_object("dex3")
            elif self.enable_inspire:
                self.inspire_dds = dds_manager.get_object("inspire")
            if self.wh:
                self.run_command_dds = dds_manager.get_object("run_command")
            print(f"[{self.name}] DDS communication initialized")
        except Exception as e:
            print(f"[{self.name}] DDS initialization failed: {e}")

    def _setup_joint_mapping(self):
        """Setup joint mapping"""
        if self.wh:
            self.action_joint_names = [
                'left_hip_pitch_joint',
                'right_hip_pitch_joint',
                'left_hip_roll_joint',
                'right_hip_roll_joint',
                'left_hip_yaw_joint',
                'right_hip_yaw_joint',
                'left_knee_joint',
                'right_knee_joint',
                'left_ankle_pitch_joint',
                'right_ankle_pitch_joint',
                'left_ankle_roll_joint',
                'right_ankle_roll_joint'
            ]
            self.waist_joint_mapping = [
                'waist_yaw_joint',
                'waist_roll_joint',
                'waist_pitch_joint',
            ]
            self.arm_joint_names = [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                # right arm (7)
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]

            # TWIST2 29-dof action order (MuJoCo actuator order)
            self.twist2_action_joint_names = [
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]

            self.old_action_joints_names = [
                'left_hip_pitch_joint',
                'right_hip_pitch_joint',
                'waist_yaw_joint',
                'left_hip_roll_joint',
                'right_hip_roll_joint',
                'waist_roll_joint',
                'left_hip_yaw_joint',
                'right_hip_yaw_joint',
                'waist_pitch_joint',
                'left_knee_joint',
                'right_knee_joint',
                'left_shoulder_pitch_joint',
                'right_shoulder_pitch_joint',
                'left_ankle_pitch_joint',
                'right_ankle_pitch_joint',
                'left_shoulder_roll_joint',
                'right_shoulder_roll_joint',
                'left_ankle_roll_joint',
                'right_ankle_roll_joint',
                'left_shoulder_yaw_joint',
                'right_shoulder_yaw_joint',
                'left_elbow_joint',
                'right_elbow_joint',
                'left_wrist_roll_joint',
                'right_wrist_roll_joint',
                'left_wrist_pitch_joint',
                'right_wrist_pitch_joint',
                'left_wrist_yaw_joint',
                'right_wrist_yaw_joint', ]
        if self.enable_robot == "g129":
            self.arm_joint_mapping = {
                "left_shoulder_pitch_joint": 0,
                "left_shoulder_roll_joint": 1,
                "left_shoulder_yaw_joint": 2,
                "left_elbow_joint": 3,
                "left_wrist_roll_joint": 4,
                "left_wrist_pitch_joint": 5,
                "left_wrist_yaw_joint": 6,
                "right_shoulder_pitch_joint": 7,
                "right_shoulder_roll_joint": 8,
                "right_shoulder_yaw_joint": 9,
                "right_elbow_joint": 10,
                "right_wrist_roll_joint": 11,
                "right_wrist_pitch_joint": 12,
                "right_wrist_yaw_joint": 13
            }
        if self.enable_gripper:
            self.gripper_joint_mapping = {
                "left_hand_Joint1_1": 1,
                "left_hand_Joint2_1": 1,
                "right_hand_Joint1_1": 0,
                "right_hand_Joint2_1": 0,
            }
        if self.enable_dex3:
            self.left_hand_joint_mapping = {
                "left_hand_thumb_0_joint": 0,
                "left_hand_thumb_1_joint": 1,
                "left_hand_thumb_2_joint": 2,
                "left_hand_middle_0_joint": 3,
                "left_hand_middle_1_joint": 4,
                "left_hand_index_0_joint": 5,
                "left_hand_index_1_joint": 6}
            self.right_hand_joint_mapping = {
                "right_hand_thumb_0_joint": 0,
                "right_hand_thumb_1_joint": 1,
                "right_hand_thumb_2_joint": 2,
                "right_hand_middle_0_joint": 3,
                "right_hand_middle_1_joint": 4,
                "right_hand_index_0_joint": 5,
                "right_hand_index_1_joint": 6}
        if self.enable_inspire:
            self.inspire_hand_joint_mapping = {
                "R_pinky_proximal_joint": 0,
                "R_ring_proximal_joint": 1,
                "R_middle_proximal_joint": 2,
                "R_index_proximal_joint": 3,
                "R_thumb_proximal_pitch_joint": 4,
                "R_thumb_proximal_yaw_joint": 5,
                "L_pinky_proximal_joint": 6,
                "L_ring_proximal_joint": 7,
                "L_middle_proximal_joint": 8,
                "L_index_proximal_joint": 9,
                "L_thumb_proximal_pitch_joint": 10,
                "L_thumb_proximal_yaw_joint": 11,
            }
            self.special_joint_mapping = {
                "L_index_intermediate_joint": [9, 1],
                "L_middle_intermediate_joint": [8, 1],
                "L_pinky_intermediate_joint": [6, 1],
                "L_ring_intermediate_joint": [7, 1],
                "L_thumb_intermediate_joint": [10, 1.5],
                "L_thumb_distal_joint": [10, 2.4],

                "R_index_intermediate_joint": [3, 1],
                "R_middle_intermediate_joint": [2, 1],
                "R_pinky_intermediate_joint": [0, 1],
                "R_ring_intermediate_joint": [1, 1],
                "R_thumb_intermediate_joint": [4, 1.5],
                "R_thumb_distal_joint": [4, 2.4],
            }
        self.all_joint_names = self.env.scene["robot"].data.joint_names
        self.joint_to_index = {name: i for i, name in enumerate(self.all_joint_names)}

        # Debug: Print joint order and PD gains
        # print("\n" + "="*80)
        # print("🔍 JOINT ORDER AND PD GAINS DEBUG")
        # print("="*80)
        # print(f"Total joints: {len(self.all_joint_names)}")
        # print("\nJoint order:")
        # for i, name in enumerate(self.all_joint_names):
        #     print(f"  [{i:2d}] {name}")
        # print("\nStiffness (Kp):")
        # print(f"  {self.env.scene['robot'].data.default_joint_stiffness}")
        # print("\nDamping (Kd):")
        # print(f"  {self.env.scene['robot'].data.default_joint_damping}")
        # print("="*80 + "\n")

        # Precompute Isaac indices for TWIST2 29-dof order
        if hasattr(self, "twist2_action_joint_names"):
            missing = [n for n in self.twist2_action_joint_names if n not in self.joint_to_index]
            if missing:
                raise ValueError(f"TWIST2 joints missing in Isaac asset: {missing}")
            self.twist2_action_indices = [self.joint_to_index[n] for n in self.twist2_action_joint_names]
            self.twist2_default_pos = self.env.scene["robot"].data.default_joint_pos[:, self.twist2_action_indices]
        if hasattr(self, "arm_joint_mapping") and self.arm_joint_mapping:
            self.arm_action_pose = [self.joint_to_index[name] for name in self.arm_joint_mapping.keys()]
            self.arm_action_pose_indices = [self.arm_joint_mapping[name] for name in self.arm_joint_mapping.keys()]
        else:
            self.arm_action_pose = []
            self.arm_action_pose_indices = []
        self.action_to_indices = []
        for action_joint in self.action_joint_names:
            if action_joint in self.all_joint_names:
                self.action_to_indices.append(self.all_joint_names.index(action_joint))
            else:
                raise ValueError(f"action joint '{action_joint}' not in all joint list")
        self.waist_to_all_indices = []
        for waist_joint in self.waist_joint_mapping:
            if waist_joint in self.all_joint_names:
                self.waist_to_all_indices.append(self.all_joint_names.index(waist_joint))
            else:
                raise ValueError(f"waist joint '{waist_joint}' not in all joint list")

        self.arm_to_all_indices = []
        for arm_joint in self.arm_joint_names:
            if arm_joint in self.all_joint_names:
                self.arm_to_all_indices.append(self.all_joint_names.index(arm_joint))
            else:
                raise ValueError(f"arm joint '{arm_joint}' not in all joint list")
        self.default_waist_positions = self.env.scene["robot"].data.default_joint_pos[:, self.waist_to_all_indices]
        self.default_action_positions = self.env.scene["robot"].data.default_joint_pos
        self.default_action_velocities = self.env.scene["robot"].data.default_joint_vel
        self.all_obs_indices = self.action_to_indices + self.arm_to_all_indices
        self.old_action_indices = []
        for old_action_joint in self.old_action_joints_names:
            if old_action_joint in self.all_joint_names:
                self.old_action_indices.append(self.all_joint_names.index(old_action_joint))
            else:
                raise ValueError(f"action joint '{old_action_joint}' not in all joint list")
        self.arm_action = []
        self.obs_scales = {"ang_vel": 1.0, "projected_gravity": 1.0, "commands": 1.0,
                           "joint_pos": 1.0, "joint_vel": 1.0, "actions": 1.0}
        self.ang_vel = self.env.scene["robot"].data.root_ang_vel_b
        self.projected_gravity = self.env.scene["robot"].data.projected_gravity_b
        self.joint_pos = self.env.scene["robot"].data.joint_pos
        self.joint_vel = self.env.scene["robot"].data.joint_vel
        self.actor_obs_buffer = CircularBuffer(
            max_len=10, batch_size=1, device=self.env.device
        )
        self.num_envs = 1
        self.clip_obs = 100
        self.num_actions_all = self.env.scene["robot"].data.default_joint_pos[:, self.old_action_indices].shape[1]
        self.action_buffer = DelayBuffer(
            5, self.num_envs, device=self.env.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions_all, dtype=torch.float, device=self.env.device,
                        requires_grad=False)
        )
        self.clip_actions = 100
        self.action_scale = 0.25
        self.sim_step_counter = 0
        cfg = getattr(self.env, "cfg", None)
        self._twist2_decimation = int(getattr(cfg, "decimation", 4))
        self._canonical_action_recorder = Twist2ActionMimicRecorder(
            control_dt=float(self._twist2_decimation * self.env.physics_dt)
        )
        # self._twist2_decimation = 1

        # Render control: only render when camera needs update
        self._render_counter = 0
        self._render_interval = 1  # Render every 3 control steps (30Hz camera for 100Hz control)

        # Observation update control: reduce observation computation frequency
        self._obs_counter = 0
        self._obs_interval = 1  # Update observations every 3 steps (30Hz camera updates)

        # Performance profiling
        self._perf_stats = {
            'redis_fetch': [],
            'policy_inference': [],
            'action_preparation': [],
            'recording_data_collection': [],
            'physics_step': [],
            'scene_update': [],
            'render_time': [],
            'observation_compute': [],
            'total_get_action': []
        }
        self._perf_report_interval = 100  # Report every 100 steps

    def _resolve_policy_path(self, model_path: str) -> str:
        if os.path.isabs(model_path):
            return model_path
        if project_root:
            return os.path.join(project_root, model_path)
        return model_path

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

    def load_policy(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".onnx":
            return self.load_onnx_policy(path)
        elif ext == ".pt":
            return self.load_jit_pt_policy(path)

    def load_jit_pt_policy(self, path):
        return torch.jit.load(path)

    def load_onnx_policy(self, path):
        available = []
        try:
            available = ort.get_available_providers()
        except Exception:
            available = []

        requested_device = self._twist2_onnx_requested_device or 'auto'
        providers = []
        expected_gpu_providers = []
        if str(requested_device).startswith('cuda') or str(requested_device).isdigit():
            if "TensorrtExecutionProvider" in available:
                providers.append(("TensorrtExecutionProvider", {"device_id": self._twist2_onnx_device_id}))
                expected_gpu_providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in available:
                providers.append(("CUDAExecutionProvider", {"device_id": self._twist2_onnx_device_id}))
                expected_gpu_providers.append("CUDAExecutionProvider")
            if not expected_gpu_providers:
                raise RuntimeError(
                    f"[{self.name}] requested GPU ONNX session on {requested_device}, "
                    f"but available providers are {available}"
                )
        providers.append("CPUExecutionProvider")
        selected_provider_names = [item[0] if isinstance(item, tuple) else item for item in providers]
        print(f"[{self.name}] ONNX available providers: {available}")
        print(
            f"[{self.name}] ONNX selected providers: {selected_provider_names} "
            f"(requested_device={requested_device}, "
            f"cuda_device_id={self._twist2_onnx_device_id})"
        )

        # Configure session options for deterministic inference
        sess_options = ort.SessionOptions()
        if self.onnx_seed is not None:
            # Enable deterministic compute
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            print(f"[{self.name}] ONNX Runtime configured for deterministic inference (seed={self.onnx_seed})")

        model = ort.InferenceSession(path, sess_options=sess_options, providers=providers)
        loaded_providers = model.get_providers()
        if expected_gpu_providers and not any(name in loaded_providers for name in expected_gpu_providers):
            raise RuntimeError(
                f"[{self.name}] ONNX model loaded on CPU instead of {requested_device}; "
                f"providers={loaded_providers}"
            )
        input_name = model.get_inputs()[0].name

        def run_inference(input_tensor: torch.Tensor):
            # Keep it simple: ORT expects numpy
            ort_inputs = {input_name: input_tensor.detach().cpu().numpy()}
            ort_outs = model.run(None, ort_inputs)
            return torch.tensor(ort_outs[0], device=self.env.device, dtype=torch.float32)

        print(f"[{self.name}] ONNX policy loaded with providers: {loaded_providers}")
        return run_inference

    def _normalize_replay_mode(self, replay_mode: str) -> str:
        if replay_mode in ("direct", "direct_replay"):
            return "direct_replay"
        if replay_mode in ("inference", "inference_replay"):
            return "inference_replay"
        raise ValueError(f"[{self.name}] Unsupported replay_mode: {replay_mode}")

    def _setup_lerobot_vla(self, args_cli) -> None:
        if self.enable_robot != "unitree_g1_refpose_v3_1":
            raise ValueError(
                f"[{self.name}] VLA v3.1 runtime requires robot_type=unitree_g1_refpose_v3_1; "
                f"got {self.enable_robot!r}"
            )
        if self._lerobot_server_url:
            self._lerobot_http_client = LeRobotVLAHttpClient(
                base_url=self._lerobot_server_url,
                timeout_s=self._lerobot_server_timeout,
                verify_ssl=self._lerobot_server_verify_ssl,
            )
            print(
                f"[{self.name}] LeRobot VLA remote client enabled: "
                f"url={self._lerobot_server_url} verify_ssl={self._lerobot_server_verify_ssl}"
            )
            return

        lerobot_policy_path = Path(getattr(args_cli, "lerobot_policy_path", "")).expanduser().resolve()
        if not lerobot_policy_path.is_dir():
            raise FileNotFoundError(f"[{self.name}] LeRobot policy directory not found: {lerobot_policy_path}")

        isaaclab_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        lerobot_src = isaaclab_root.parent / "lerobot" / "src"
        if not lerobot_src.is_dir():
            raise FileNotFoundError(f"[{self.name}] LeRobot src directory not found: {lerobot_src}")

        import sys

        if str(lerobot_src) not in sys.path:
            sys.path.insert(0, str(lerobot_src))

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import _reconnect_relative_absolute_steps, get_policy_class
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        from lerobot.utils.constants import (
            POLICY_POSTPROCESSOR_DEFAULT_NAME,
            POLICY_PREPROCESSOR_DEFAULT_NAME,
        )
        from lerobot.utils.control_utils import predict_action

        device_name = getattr(args_cli, "lerobot_policy_device", "") or getattr(args_cli, "device", "cuda:0")
        config = PreTrainedConfig.from_pretrained(lerobot_policy_path)
        config.device = device_name
        state_feature = (config.input_features or {}).get("observation.state")
        action_feature = (config.output_features or {}).get("action")
        state_shape = tuple(getattr(state_feature, "shape", ()) or ())
        action_shape = tuple(getattr(action_feature, "shape", ()) or ())
        if state_shape and state_shape != (VLA_ROBOT_CURRENT_LOCAL_V3_STATE_DIM,):
            raise ValueError(
                f"[{self.name}] VLA v3.1 policy must use observation.state shape {(VLA_ROBOT_CURRENT_LOCAL_V3_STATE_DIM,)}, "
                f"got {state_shape}"
            )
        if action_shape and action_shape != (VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM,):
            raise ValueError(
                f"[{self.name}] VLA v3.1 policy must use action shape {(VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM,)}, "
                f"got {action_shape}"
            )
        policy_cls = get_policy_class(config.type)

        self._lerobot_policy = policy_cls.from_pretrained(lerobot_policy_path, config=config)
        self._lerobot_preprocessor = PolicyProcessorPipeline.from_pretrained(
            lerobot_policy_path,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        )
        self._lerobot_postprocessor = PolicyProcessorPipeline.from_pretrained(
            lerobot_policy_path,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        _reconnect_relative_absolute_steps(self._lerobot_preprocessor, self._lerobot_postprocessor)

        self._lerobot_predict_action = predict_action
        self._lerobot_device = torch.device(device_name)
        self._lerobot_policy.reset()
        self._lerobot_preprocessor.reset()
        self._lerobot_postprocessor.reset()

        print(
            f"[{self.name}] LeRobot VLA enabled: "
            f"path={lerobot_policy_path} type={config.type} device={self._lerobot_device}"
        )

    def _get_front_camera_rgb_for_vla(self) -> np.ndarray:
        if "front_camera" not in self.env.scene.keys():
            raise RuntimeError(f"[{self.name}] front_camera not found in scene for LeRobot VLA inference")

        camera = self.env.scene["front_camera"]
        if "rgb" not in camera.data.output:
            raise RuntimeError(f"[{self.name}] front_camera has no rgb output for LeRobot VLA inference")

        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.ndim != 3:
            raise RuntimeError(f"[{self.name}] Unexpected front_camera rgb shape: {rgb.shape}")
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _get_current_robot_root_pose_for_vla(self) -> tuple[np.ndarray, np.ndarray]:
        robot = self.env.scene["robot"].data
        root_state = robot.root_state_w[0].detach().cpu().numpy().astype(np.float32)
        root_xy_world = root_state[0:2].astype(np.float32, copy=True)
        root_quat_wxyz = root_state[3:7].astype(np.float32, copy=True)
        return root_quat_wxyz, root_xy_world

    def _build_lerobot_vla_observation_state(self) -> np.ndarray:
        robot = self.env.scene["robot"].data
        joint_pos_twist2 = robot.joint_pos[0, self.twist2_action_indices].detach().cpu().numpy().astype(np.float32)
        joint_vel_twist2 = robot.joint_vel[0, self.twist2_action_indices].detach().cpu().numpy().astype(np.float32)
        joint_pos_canonical = reorder_twist2_to_canonical_29(joint_pos_twist2)
        joint_vel_canonical = reorder_twist2_to_canonical_29(joint_vel_twist2)
        base_quat_wxyz, _ = self._get_current_robot_root_pose_for_vla()
        if self._vla_initial_robot_quat_wxyz is None:
            self._vla_initial_robot_quat_wxyz = base_quat_wxyz.copy()
        state = build_vla_rotlocal_v3_observation_state(
            initial_robot_orientation_wxyz=self._vla_initial_robot_quat_wxyz,
            root_orientation_wxyz=base_quat_wxyz,
            joint_pos_canonical_29=joint_pos_canonical,
            joint_vel_canonical_29=joint_vel_canonical,
        )
        if state.shape != (VLA_ROBOT_CURRENT_LOCAL_V3_STATE_DIM,):
            raise RuntimeError(f"[{self.name}] Expected canonical VLA state shape {(VLA_ROBOT_CURRENT_LOCAL_V3_STATE_DIM,)}, got {state.shape}")
        return state

    def _record_lerobot_observation_state(self) -> np.ndarray:
        proprio = self._build_lerobot_vla_observation_state()
        step_index = int(self.sim_step_counter)
        if (
            self._lerobot_http_client is not None
            and step_index != self._lerobot_observation_state_last_step
        ):
            self._lerobot_observation_state_segment.append(proprio.copy())
            self._lerobot_observation_state_last_step = step_index
        return proprio

    def _fetch_lerobot_action_chunk(self, proprio: np.ndarray) -> np.ndarray:
        rgb = self._get_front_camera_rgb_for_vla()

        if self._lerobot_http_client is not None:
            if not self._lerobot_observation_state_segment:
                raise RuntimeError("No executed observation state is available for VLA inference")
            pending_count = len(self._lerobot_observation_state_segment)
            state_history = np.stack(
                list(self._lerobot_observation_state_segment), axis=0
            ).astype(np.float32, copy=False)
            action_chunk = self._lerobot_http_client.infer_chunk(
                front_rgb=rgb,
                observation_state=proprio,
                robot_type=self.enable_robot,
                task=self.task_name,
                observation_state_history=state_history,
            )
        else:
            pending_count = 0
            if self._lerobot_policy is None or self._lerobot_predict_action is None:
                raise RuntimeError(f"[{self.name}] LeRobot VLA requested before initialization")

            observation = {
                "observation.images.front": rgb,
                "observation.state": proprio,
            }

            action = self._lerobot_predict_action(
                observation=observation,
                policy=self._lerobot_policy,
                device=self._lerobot_device,
                preprocessor=self._lerobot_preprocessor,
                postprocessor=self._lerobot_postprocessor,
                use_amp=self._lerobot_device.type == "cuda",
                task=self.task_name,
                robot_type=self.enable_robot,
            )
            if isinstance(action, torch.Tensor):
                action_chunk = action.detach().cpu().numpy().astype(np.float32)
            else:
                action_chunk = np.asarray(action, dtype=np.float32)

        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk.reshape(1, -1)
        if action_chunk.ndim != 2 or action_chunk.shape[1] != VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM:
            raise ValueError(
                f"[{self.name}] Expected canonical VLA action chunk shape [N, {VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM}], got {action_chunk.shape}"
            )
        for _ in range(pending_count):
            self._lerobot_observation_state_segment.popleft()
        return action_chunk

    def _pop_lerobot_action(self, proprio: np.ndarray) -> np.ndarray:
        if not self._lerobot_action_chunk_queue:
            for action in self._fetch_lerobot_action_chunk(proprio):
                self._lerobot_action_chunk_queue.append(np.asarray(action, dtype=np.float32).copy())
        action_np = np.asarray(self._lerobot_action_chunk_queue.popleft(), dtype=np.float32).reshape(-1)
        if action_np.shape != (VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM,):
            raise ValueError(
                f"[{self.name}] Expected canonical VLA action dim {VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM}, got {action_np.shape}"
            )
        return action_np

    def _should_refresh_lerobot_visuals_next_step(self) -> bool:
        return (not self._use_lerobot_vla) or (len(self._lerobot_action_chunk_queue) == 0)

    def _infer_lerobot_high_level_command(self) -> torch.Tensor:
        proprio = self._record_lerobot_observation_state()
        action_np = self._pop_lerobot_action(proprio)
        self._latest_vla_action = action_np.copy()
        current_robot_quat_wxyz, current_robot_xy_world = self._get_current_robot_root_pose_for_vla()
        runtime_frame = self._lerobot_vla_runtime.step(
            action_np,
            current_robot_quat_wxyz=current_robot_quat_wxyz,
            current_robot_xy_world=current_robot_xy_world,
        )
        self._vla_gripper_binary.copy_(
            torch.from_numpy(np.clip(runtime_frame.hand_binary, 0.0, 1.0)).to(
                self.env.device, dtype=torch.float32
            ).unsqueeze(0)
        )
        control_dt = float(self._twist2_decimation * self.env.physics_dt)
        mimic_obs = build_twist2_mimic_obs_v3(
            runtime_frame=runtime_frame,
            control_dt=control_dt,
        )
        return torch.from_numpy(mimic_obs).to(self.env.device, dtype=torch.float32).unsqueeze(0)

    def _apply_vla_gripper_command(self, full_action: torch.Tensor) -> bool:
        if not self._use_lerobot_vla:
            return False

        left_closed = bool(self._vla_gripper_binary[0, 0].item() >= self._lerobot_gripper_threshold)
        right_closed = bool(self._vla_gripper_binary[0, 1].item() >= self._lerobot_gripper_threshold)
        if self.enable_gripper and hasattr(self, "_gripper_source_idx_t"):
            left_position = 0.03 if left_closed else -0.02
            right_position = 0.03 if right_closed else -0.02
            self._gripper_buf.copy_(
                torch.tensor([right_position, left_position], dtype=torch.float32, device=self.env.device)
            )
            gp_vals = self._gripper_buf.index_select(0, self._gripper_source_idx_t)
            full_action.index_copy_(0, self._gripper_target_idx_t, gp_vals)
            return True

        if self.enable_dex3 and hasattr(self, "_left_hand_target_idx_t"):
            pose_robot_name = "unitree_g1_with_hands"
            left_pose = DEFAULT_HAND_POSE[pose_robot_name]["left"]["close" if left_closed else "open"]
            right_pose = DEFAULT_HAND_POSE[pose_robot_name]["right"]["close" if right_closed else "open"]
            self._left_hand_buf.copy_(
                torch.as_tensor(left_pose[: len(self._left_hand_buf)], dtype=torch.float32, device=self.env.device)
            )
            self._right_hand_buf.copy_(
                torch.as_tensor(right_pose[: len(self._right_hand_buf)], dtype=torch.float32, device=self.env.device)
            )
            l_vals = self._left_hand_buf.index_select(0, self._left_hand_source_idx_t)
            r_vals = self._right_hand_buf.index_select(0, self._right_hand_source_idx_t)
            full_action.index_copy_(0, self._left_hand_target_idx_t, l_vals)
            full_action.index_copy_(0, self._right_hand_target_idx_t, r_vals)
            return True

        return False

    def _load_replay_array(self, replay_data, *keys, required=False):
        for key in keys:
            if key in replay_data:
                return np.asarray(replay_data[key]).copy()
        if required:
            raise KeyError(f"[{self.name}] replay npz missing keys: {keys}")
        return None

    def _setup_local_replay(self):
        replay_path = Path(self._replay_file).expanduser().resolve()
        if not replay_path.is_file():
            raise FileNotFoundError(f"[{self.name}] replay file not found: {replay_path}")

        with np.load(replay_path, allow_pickle=True) as replay_data:
            self._replay_data_qpos = None
            if "robot_twist2_inference_qpos" in replay_data:
                self._replay_data_qpos = np.asarray(replay_data["robot_twist2_inference_qpos"]).copy()
            elif "vla_action_joint_pos_29" in replay_data:
                self._replay_data_qpos = reorder_canonical_to_twist2_29(
                    np.asarray(replay_data["vla_action_joint_pos_29"], dtype=np.float32)
                )
            elif "vla_action" in replay_data:
                vla_action = np.asarray(replay_data["vla_action"], dtype=np.float32)
                if vla_action.ndim == 2 and vla_action.shape[-1] == VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM:
                    self._replay_data_qpos = reorder_canonical_to_twist2_29(vla_action[:, 9:38])
            if self._replay_data_qpos is None:
                raise KeyError(
                    f"[{self.name}] replay npz missing keys: "
                    f"('robot_twist2_inference_qpos', 'vla_action_joint_pos_29', 'vla_action')"
                )
            self._replay_recorded_joint_pos = self._load_replay_array(
                replay_data,
                "robot_qpos_before_decimation",
            )
            self._replay_data_obs = self._load_replay_array(
                replay_data,
                "robot_obs_buf",
                required=self._replay_mode == "inference_replay",
            )
            self._replay_data_hand_left = self._load_replay_array(replay_data, "human_hand_left")
            self._replay_data_hand_right = self._load_replay_array(replay_data, "human_hand_right")
            if (
                self._replay_data_hand_left is None
                and self._replay_data_hand_right is None
                and (
                    "vla_action_hand_binary_2" in replay_data
                    or "vla_action_hand_binary" in replay_data
                    or "vla_action" in replay_data
                )
            ):
                hand_binary = None
                if "vla_action_hand_binary_2" in replay_data:
                    hand_binary = np.asarray(replay_data["vla_action_hand_binary_2"], dtype=np.float32)
                elif "vla_action" in replay_data:
                    vla_action = np.asarray(replay_data["vla_action"], dtype=np.float32)
                    if vla_action.ndim == 2 and vla_action.shape[-1] == VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM:
                        hand_binary = vla_action[:, 38:40]
                elif "vla_action_hand_binary" in replay_data:
                    hand_binary = np.asarray(replay_data["vla_action_hand_binary"], dtype=np.float32)
                if hand_binary is not None and hand_binary.ndim == 2 and hand_binary.shape[-1] == 2:
                    pose_robot_name = "unitree_g1_with_hands"
                    left_open = np.asarray(DEFAULT_HAND_POSE[pose_robot_name]["left"]["open"], dtype=np.float32)
                    left_close = np.asarray(DEFAULT_HAND_POSE[pose_robot_name]["left"]["close"], dtype=np.float32)
                    right_open = np.asarray(DEFAULT_HAND_POSE[pose_robot_name]["right"]["open"], dtype=np.float32)
                    right_close = np.asarray(DEFAULT_HAND_POSE[pose_robot_name]["right"]["close"], dtype=np.float32)
                    self._replay_data_hand_left = np.stack(
                        [left_close if row[0] >= 0.5 else left_open for row in hand_binary], axis=0
                    ).astype(np.float32)
                    self._replay_data_hand_right = np.stack(
                        [right_close if row[1] >= 0.5 else right_open for row in hand_binary], axis=0
                    ).astype(np.float32)
            self._replay_data_neck = self._load_replay_array(replay_data, "human_neck")
            self._replay_human_smplx_data = _decode_json_list_field(replay_data.get("human_smplx_data"))
            self._replay_human_info_data = _decode_json_list_field(replay_data.get("human_info_data"))
            self._replay_pico_left_grip_binary = self._load_replay_array(replay_data, "pico_left_grip_binary")
            self._replay_pico_right_grip_binary = self._load_replay_array(replay_data, "pico_right_grip_binary")
            self._replay_pico_left_close_trigger_binary = self._load_replay_array(
                replay_data, "pico_left_close_trigger_binary"
            )
            self._replay_pico_right_close_trigger_binary = self._load_replay_array(
                replay_data, "pico_right_close_trigger_binary"
            )
            self._replay_pico_left_open_trigger_binary = self._load_replay_array(
                replay_data, "pico_left_open_trigger_binary"
            )
            self._replay_pico_right_open_trigger_binary = self._load_replay_array(
                replay_data, "pico_right_open_trigger_binary"
            )
            self._replay_object_states = {}
            replay_object_suffixes = {
                "_position": "position",
                "_linear_velocity": "linear_velocity",
                "_angular_velocity": "angular_velocity",
            }
            for key in replay_data.files:
                if not key.startswith("env_obj_"):
                    continue
                for suffix, field_name in replay_object_suffixes.items():
                    if key.endswith(suffix):
                        object_name = key[len("env_obj_") : -len(suffix)]
                        self._replay_object_states.setdefault(object_name, {})[field_name] = np.asarray(
                            replay_data[key]
                        ).copy()
                        break

        candidate_arrays = [
            self._replay_data_qpos,
            self._replay_data_obs,
            self._replay_data_hand_left,
            self._replay_data_hand_right,
        ]
        candidate_arrays = [arr for arr in candidate_arrays if arr is not None]
        if not candidate_arrays:
            raise ValueError(f"[{self.name}] replay npz contains no usable frame arrays")

        self._replay_num_frames = int(candidate_arrays[0].shape[0])
        self._replay_file = str(replay_path)
        self._replay_cursor = 0

    def _log_replay_joint_position_error(self, frame_idx: int) -> None:
        if (
            not self._replay_enabled
            or self._replay_recorded_joint_pos is None
            or len(self._replay_recorded_joint_pos) == 0
        ):
            return
        compare_idx = min(frame_idx + 1, len(self._replay_recorded_joint_pos) - 1)
        recorded = np.asarray(self._replay_recorded_joint_pos[compare_idx], dtype=np.float32).reshape(-1)
        current = (
            self.env.scene["robot"].data.joint_pos[0, self._twist2_action_idx_t]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .reshape(-1)
        )
        dims = min(current.shape[0], recorded.shape[0])
        if dims <= 0:
            return
        err = np.abs(current[:dims] - recorded[:dims])
        mae = float(err.mean())
        max_err = float(err.max())
        self._replay_joint_mae_sum += mae
        self._replay_joint_mae_count += 1
        running_mae = self._replay_joint_mae_sum / max(1, self._replay_joint_mae_count)
        if frame_idx < 3 or frame_idx % self._replay_joint_err_log_interval == 0:
            print(
                f"[{self.name}] Replay joint err: frame={frame_idx} "
                f"compare_to_recorded_frame={compare_idx} "
                f"mae={mae:.6f} rad max={max_err:.6f} rad "
                f"running_mae={running_mae:.6f} rad"
            )

    def _resolve_replay_object_scene_key(self, object_name: str) -> str | None:
        return resolve_env_object_scene_key(self.env, self.env.cfg, object_name)

    def _get_current_replay_object_state(self, object_name: str) -> dict | None:
        scene_key = self._resolve_replay_object_scene_key(object_name)
        if scene_key is None:
            return None
        try:
            obj = self.env.scene[scene_key]
            root_state = obj.data.root_state_w
            return {
                "position": root_state[0, 0:3].detach().cpu().numpy().astype(np.float32),
                "linear_velocity": root_state[0, 7:10].detach().cpu().numpy().astype(np.float32),
                "angular_velocity": root_state[0, 10:13].detach().cpu().numpy().astype(np.float32),
            }
        except Exception:
            return None

    def _log_replay_object_state_error(self, frame_idx: int) -> None:
        if not self._replay_enabled or not self._replay_object_states:
            return
        if not (frame_idx < 3 or frame_idx % self._replay_joint_err_log_interval == 0):
            return

        field_units = {
            "position": "m",
            "linear_velocity": "m/s",
            "angular_velocity": "rad/s",
        }
        field_labels = {
            "position": "pos",
            "linear_velocity": "lin_vel",
            "angular_velocity": "ang_vel",
        }

        for object_name, recorded_fields in self._replay_object_states.items():
            current_state = self._get_current_replay_object_state(object_name)
            if current_state is None:
                continue

            parts = []
            object_compare_idx = None
            for field_name in ("position", "linear_velocity", "angular_velocity"):
                recorded_series = recorded_fields.get(field_name)
                if recorded_series is None or len(recorded_series) == 0:
                    continue
                compare_idx = min(frame_idx + 1, len(recorded_series) - 1)
                object_compare_idx = compare_idx
                recorded = np.asarray(recorded_series[compare_idx], dtype=np.float32).reshape(-1)
                current = np.asarray(current_state[field_name], dtype=np.float32).reshape(-1)
                dims = min(current.shape[0], recorded.shape[0])
                if dims <= 0:
                    continue
                err = np.abs(current[:dims] - recorded[:dims])
                mae = float(err.mean())
                max_err = float(err.max())
                stat_key = (object_name, field_name)
                self._replay_object_err_sums[stat_key] = self._replay_object_err_sums.get(stat_key, 0.0) + mae
                self._replay_object_err_counts[stat_key] = self._replay_object_err_counts.get(stat_key, 0) + 1
                running_mae = self._replay_object_err_sums[stat_key] / self._replay_object_err_counts[stat_key]
                unit = field_units[field_name]
                parts.append(
                    f"{field_labels[field_name]}_mae={mae:.6f} {unit} "
                    f"max={max_err:.6f} {unit} running={running_mae:.6f} {unit}"
                )

            if parts:
                print(
                    f"[{self.name}] Replay env err: frame={frame_idx} "
                    f"compare_to_recorded_frame={object_compare_idx if object_compare_idx is not None else frame_idx} "
                    f"object={object_name} "
                    + " ".join(parts)
                )

    def _collect_env_state(self) -> dict:
        return collect_recordable_env_object_states(self.env, self.env.cfg)

    def _next_replay_frame_idx(self) -> int | None:
        if not self._replay_enabled or self._replay_num_frames <= 0:
            return None
        if self._replay_cursor >= self._replay_num_frames:
            if not self._replay_loop:
                return None
            self._replay_cursor = 0
        frame_idx = int(self._replay_cursor)
        self._replay_cursor += 1
        return frame_idx

    def _get_replay_controller_binary(self, frame_idx: int) -> dict[str, bool]:
        def _frame_bool(series) -> bool:
            if series is None or frame_idx >= len(series):
                return False
            return bool(np.asarray(series[frame_idx]).reshape(-1)[-1])

        return {
            "left_grip_binary": _frame_bool(self._replay_pico_left_grip_binary),
            "right_grip_binary": _frame_bool(self._replay_pico_right_grip_binary),
            "left_close_trigger_binary": _frame_bool(self._replay_pico_left_close_trigger_binary),
            "right_close_trigger_binary": _frame_bool(self._replay_pico_right_close_trigger_binary),
            "left_open_trigger_binary": _frame_bool(self._replay_pico_left_open_trigger_binary),
            "right_open_trigger_binary": _frame_bool(self._replay_pico_right_open_trigger_binary),
        }

    def _apply_replay_human_context(self, frame_idx: int) -> None:
        if frame_idx < len(self._replay_human_smplx_data):
            self._twist2_human_smplx_data = copy.deepcopy(self._replay_human_smplx_data[frame_idx])
            self._twist2_human_smplx_valid = self._twist2_human_smplx_data is not None
        else:
            self._twist2_human_smplx_data = None
            self._twist2_human_smplx_valid = False

        if frame_idx < len(self._replay_human_info_data):
            self._twist2_human_info = copy.deepcopy(self._replay_human_info_data[frame_idx])
            self._twist2_human_info_valid = self._twist2_human_info is not None
        else:
            self._twist2_human_info = None
            self._twist2_human_info_valid = False

        controller_binary = self._get_replay_controller_binary(frame_idx)
        self._twist2_controller_data = _build_replay_controller_data(controller_binary)

    def _finalize_replay_if_needed(self) -> None:
        if self._replay_completion_requested:
            return
        self._replay_completion_requested = True
        try:
            setattr(self.env, "_request_main_loop_exit", True)
            print(f"[{self.name}] Marked env for main-loop exit after replay completion")
        except Exception:
            pass
        if self._record_during_replay and self.recording_manager.is_recording:
            self._update_replay_reward_stats()
            final_reward = _reward_scalar(self.env)
            max_reward = final_reward if self._replay_reward_max is None else max(self._replay_reward_max, final_reward)
            any_success = bool(self._replay_any_success or _reward_success_flag(max_reward))
            if self.recording_manager.recording_buffer:
                last_frame = self.recording_manager.recording_buffer[-1]
                last_frame["rerecord_final_reward"] = final_reward
                last_frame["rerecord_max_reward"] = max_reward
                last_frame["rerecord_any_success"] = any_success
            print(f"[{self.name}] Final rerecord reward={final_reward:.4f}")
            print(f"[{self.name}] Max rerecord reward={max_reward:.4f} any_success={str(any_success).lower()}")
            print(f"[{self.name}] 💾 Finalizing replay rerecord before exit")
            self.recording_manager.save_recording()
        if self._exit_when_replay_complete:
            sim = getattr(self.env, "sim", None)
            stop_fn = getattr(sim, "stop", None)
            if callable(stop_fn):
                try:
                    stop_fn()
                    print(f"[{self.name}] Requested env.sim.stop() after replay completion")
                except Exception as exc:
                    print(f"[{self.name}] env.sim.stop() after replay completion failed: {exc}")

    def should_exit_after_replay_complete(self) -> bool:
        if not self._exit_when_replay_complete:
            return False
        return bool(self._replay_completion_requested)

    def _set_replay_hand_targets(self, frame_idx: int) -> None:
        self._apply_replay_human_context(frame_idx)
        self._twist2_hand_valid = False
        self._twist2_action_hand_left.zero_()
        self._twist2_action_hand_right.zero_()
        self._twist2_action_neck.zero_()

        left = None
        right = None
        if self._replay_data_hand_left is not None and frame_idx < len(self._replay_data_hand_left):
            left = np.asarray(self._replay_data_hand_left[frame_idx], dtype=np.float32).reshape(-1)
            self._twist2_action_hand_left[0, : min(self._twist2_hand_dim, left.shape[0])] = torch.as_tensor(
                left[: self._twist2_hand_dim], device=self.env.device, dtype=torch.float32
            )
        if self._replay_data_hand_right is not None and frame_idx < len(self._replay_data_hand_right):
            right = np.asarray(self._replay_data_hand_right[frame_idx], dtype=np.float32).reshape(-1)
            self._twist2_action_hand_right[0, : min(self._twist2_hand_dim, right.shape[0])] = torch.as_tensor(
                right[: self._twist2_hand_dim], device=self.env.device, dtype=torch.float32
            )
        if self._replay_data_neck is not None and frame_idx < len(self._replay_data_neck):
            neck = np.asarray(self._replay_data_neck[frame_idx], dtype=np.float32).reshape(-1)
            self._twist2_action_neck[0, : min(self._twist2_neck_dim, neck.shape[0])] = torch.as_tensor(
                neck[: self._twist2_neck_dim], device=self.env.device, dtype=torch.float32
            )
        self._twist2_hand_valid = left is not None and right is not None

    def _run_replay_policy(self, frame_idx: int):
        self._set_replay_hand_targets(frame_idx)

        if self._replay_mode == "inference_replay":
            obs_buf = torch.from_numpy(self._replay_data_obs[frame_idx]).unsqueeze(0).to(
                self.env.device, dtype=torch.float32
            )
            with torch.no_grad():
                action = self.policy(obs_buf)
            if isinstance(action, torch.Tensor) and action.dim() == 1:
                action = action.unsqueeze(0)
            if isinstance(action, torch.Tensor) and action.shape[-1] == 29:
                self._twist2_last_action.copy_(action.to(self.env.device, dtype=torch.float32))
            raw_action = torch.clip(action.to(self.env.device, dtype=torch.float32), -10.0, 10.0)
            target_29 = raw_action * 0.5 + self.twist2_default_pos
            return target_29, obs_buf

        target_29 = torch.from_numpy(self._replay_data_qpos[frame_idx]).unsqueeze(0).to(
            self.env.device, dtype=torch.float32
        )
        if self._replay_data_obs is not None and frame_idx < len(self._replay_data_obs):
            obs_buf = torch.from_numpy(self._replay_data_obs[frame_idx]).unsqueeze(0).to(
                self.env.device, dtype=torch.float32
            )
        else:
            obs_buf = torch.zeros((1, self.total_obs_size), device=self.env.device, dtype=torch.float32)
        raw_action = (target_29 - self.twist2_default_pos) / 0.5
        self._twist2_last_action.copy_(raw_action.to(self.env.device, dtype=torch.float32))
        return target_29, obs_buf

    def _twist2_parse_list(self, value, expected_len: int) -> list:
        if value is None:
            return [0.0] * expected_len
        try:
            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8")
            data = json.loads(value)
        except Exception:
            return [0.0] * expected_len
        if not isinstance(data, list):
            return [0.0] * expected_len
        if len(data) < expected_len:
            data = data + [0.0] * (expected_len - len(data))
        elif len(data) > expected_len:
            data = data[:expected_len]
        return data

    def _update_input_ready_guard(self, raw_guard) -> None:
        guard = None
        if raw_guard is not None:
            try:
                if isinstance(raw_guard, (bytes, bytearray)):
                    raw_guard = raw_guard.decode("utf-8")
                parsed = json.loads(raw_guard)
                if isinstance(parsed, dict):
                    guard = parsed
            except Exception:
                guard = None
        new_epoch = int(guard.get("epoch_id", -1)) if guard is not None else -1
        new_timestamp_ms = int(guard.get("ready_timestamp_ms", 0)) if guard is not None else 0
        if new_epoch != self._input_ready_epoch_id or new_timestamp_ms != self._input_ready_timestamp_ms:
            self._stale_input_drop_logged_epoch = -1
        self._input_ready_epoch_id = new_epoch
        self._input_ready_timestamp_ms = new_timestamp_ms

    def _parse_action_timestamp_ms(self, raw_value) -> int | None:
        if raw_value is None:
            return None
        try:
            if isinstance(raw_value, (bytes, bytearray)):
                raw_value = raw_value.decode("utf-8")
            return int(float(raw_value))
        except Exception:
            return None

    def _clear_stale_live_inputs(self) -> None:
        self._twist2_action_hand_left.zero_()
        self._twist2_action_hand_right.zero_()
        self._twist2_action_neck.zero_()
        self._twist2_hand_valid = False
        self._twist2_human_smplx_data = None
        self._twist2_human_smplx_valid = False
        self._twist2_human_info = None
        self._twist2_human_info_valid = False
        self._twist2_controller_data = None
        self._recording_command = "none"

    def _should_drop_stale_twist2_action(self, action_timestamp_ms: int | None) -> bool:
        if self._input_ready_timestamp_ms <= 0:
            return False
        if action_timestamp_ms is None:
            return True
        return action_timestamp_ms <= self._input_ready_timestamp_ms

    def _twist2_fetch_actions(self) -> torch.Tensor:
        """Fetch TWIST2 actions (body + hand + neck) from Redis."""
        if self.redis_pipeline is None:
            # print("[XY_DEBUG] ⚠️ Redis pipeline is None!")
            self._twist2_hand_valid = False
            self._twist2_human_smplx_valid = False
            self._twist2_human_info_valid = False
            self._twist2_controller_data = None
            # Return default 35D mimic_obs to maintain stable standing pose
            return self._default_mimic_obs.clone()
        try:
            keys = [
                "action_body_unitree_g1_with_hands",
                "action_hand_left_unitree_g1_with_hands",
                "action_hand_right_unitree_g1_with_hands",
                "action_neck_unitree_g1_with_hands",
                "human_smplx_data_unitree_g1_with_hands",
                "human_info_unitree_g1_with_hands",
                "controller_data",
                "recording_control_unitree_g1_with_hands",
                "t_action",
                self._input_ready_key,
            ]
            for key in keys:
                self.redis_pipeline.get(key)
            res = self.redis_pipeline.execute()
            action_body_raw = res[0] if len(res) > 0 else None

            # # 🔍 调试点1：检查Redis原始数据
            # if action_body_raw is None:
            #     print("[XY_DEBUG] ⚠️ action_body_raw is None - no data in Redis")
            # else:
            #     print(f"[XY_DEBUG] ✓ Redis data received: {len(action_body_raw)} bytes")

            action_left_raw = res[1] if len(res) > 1 else None
            action_right_raw = res[2] if len(res) > 2 else None
            action_neck_raw = res[3] if len(res) > 3 else None
            human_smplx_data_raw = res[4] if len(res) > 4 else None
            human_info_raw = res[5] if len(res) > 5 else None
            controller_data_raw = res[6] if len(res) > 6 else None
            recording_control_raw = res[7] if len(res) > 7 else None
            action_timestamp_raw = res[8] if len(res) > 8 else None
            input_ready_raw = res[9] if len(res) > 9 else None

            self._update_input_ready_guard(input_ready_raw)
            action_timestamp_ms = self._parse_action_timestamp_ms(action_timestamp_raw)
            if self._should_drop_stale_twist2_action(action_timestamp_ms):
                self._clear_stale_live_inputs()
                if self._input_ready_epoch_id != self._stale_input_drop_logged_epoch:
                    print(
                        f"[{self.name}] Ignoring stale TWIST2 input: "
                        f"t_action={action_timestamp_ms} ready_timestamp_ms={self._input_ready_timestamp_ms}"
                    )
                    self._stale_input_drop_logged_epoch = self._input_ready_epoch_id
                return self._default_mimic_obs.clone()

            action_body = self._twist2_parse_list(action_body_raw, self.n_mimic_obs)
            action_left = self._twist2_parse_list(action_left_raw, self._twist2_hand_dim)
            action_right = self._twist2_parse_list(action_right_raw, self._twist2_hand_dim)
            action_neck = self._twist2_parse_list(action_neck_raw, self._twist2_neck_dim)
            self._twist2_controller_data = None
            if controller_data_raw:
                try:
                    if isinstance(controller_data_raw, (bytes, bytearray)):
                        controller_data_raw = controller_data_raw.decode("utf-8")
                    self._twist2_controller_data = json.loads(controller_data_raw)
                except Exception:
                    self._twist2_controller_data = None


            # action_body[0] = max(0, min(2, action_body[0]))
            # action_body[1] = max(0, min(2, action_body[1]))
            # action_body[5] = max(0, min(2, action_body[5]))

            # 🔍 调试点2：检查解析后的数据
            if len(action_body) >= 6:
                xy_vel = action_body[0:2]
                z_pos = action_body[2]
                yaw_vel = action_body[5]

                import math
                xy_speed = math.sqrt(xy_vel[0]**2 + xy_vel[1]**2)

                # print(f"[ISAAC_XY_DEBUG] Isaac Lab接收到的数据:")
                # print(f"  XY vel: [{xy_vel[0]:.6f}, {xy_vel[1]:.6f}] m/s (speed: {xy_speed:.6f})")
                # print(f"  Z pos: {z_pos:.4f} m, Yaw vel: {yaw_vel:.6f} rad/s")
                # print(f"  完整action_body前6维: {action_body[0:6]}")

                # 分析速度方向
                # if xy_speed > 0.1:
                #     angle_deg = math.degrees(math.atan2(xy_vel[1], xy_vel[0]))
                #     print(f"  速度方向: {angle_deg:.2f}° (0°=+X前方, 90°=+Y左侧)")
                #     if abs(xy_vel[0]) > abs(xy_vel[1]) * 2:
                #         print(f"  ✓ 主要沿X方向（前方）")
                #     elif abs(xy_vel[1]) > abs(xy_vel[0]) * 2:
                #         print(f"  ⚠️ 主要沿Y方向（左侧）")
                #     else:
                #         print(f"  ⚠️ X和Y速度相近，可能有偏移")

            # Check if action_body is all zeros (no valid data from Redis)
            # This prevents robot from falling when Redis is empty
            # 🔍 调试点3：检查全0判断逻辑
            will_reject = action_body_raw is None or all(x == 0.0 for x in action_body)
            if will_reject:
                self._clear_stale_live_inputs()
                # print("[XY_DEBUG] ⚠️ Data REJECTED by all-zero check! Returning default.")
                # print(f"  - action_body_raw is None: {action_body_raw is None}")
                # if action_body_raw is not None:
                #     print(f"  - all zeros: {all(x == 0.0 for x in action_body)}")
                # No valid teleop data, return default standing pose
                return self._default_mimic_obs.clone()

            # Parse human SMPLX data (before GMR retargeting)
            if human_smplx_data_raw is not None:
                try:
                    if isinstance(human_smplx_data_raw, (bytes, bytearray)):
                        human_smplx_data_raw = human_smplx_data_raw.decode("utf-8")
                    self._twist2_human_smplx_data = json.loads(human_smplx_data_raw)
                    self._twist2_human_smplx_valid = True
                except Exception as e:
                    # print(f"[{self.name}] Failed to parse human SMPLX data: {e}")
                    self._twist2_human_smplx_data = None
                    self._twist2_human_smplx_valid = False
            else:
                self._twist2_human_smplx_data = None
                self._twist2_human_smplx_valid = False

            # Parse human info (height, etc.)
            if human_info_raw is not None:
                try:
                    if isinstance(human_info_raw, (bytes, bytearray)):
                        human_info_raw = human_info_raw.decode("utf-8")
                    self._twist2_human_info = json.loads(human_info_raw)
                    self._twist2_human_info_valid = True
                except Exception as e:
                    # print(f"[{self.name}] Failed to parse human info: {e}")
                    self._twist2_human_info = None
                    self._twist2_human_info_valid = False
            else:
                self._twist2_human_info = None
                self._twist2_human_info_valid = False

            # Parse recording control state
            # Skip reading new commands if we're waiting for reset to complete
            if recording_control_raw is not None and not self._waiting_for_reset_complete:
                try:
                    if isinstance(recording_control_raw, (bytes, bytearray)):
                        recording_control_raw = recording_control_raw.decode("utf-8")
                    recording_control = json.loads(recording_control_raw)
                    self._latest_recording_control = recording_control
                    new_recording_state = bool(recording_control.get("active", False))
                    new_recording_command = str(recording_control.get("command", "none"))
                    sequence = int(recording_control.get("sequence", -1))

                    # Debug: print when the teleop-side state changes.
                    if (
                        new_recording_state != self._recording_active
                        or (
                            new_recording_command != "none"
                            and sequence != self._latest_recording_control_sequence
                        )
                    ):
                        print(
                            f"[{self.name}] 🔄 Recording state changed: "
                            f"active={new_recording_state}, command={new_recording_command}, seq={sequence}"
                        )

                    self._recording_active = new_recording_state

                    # Recording commands are level-held in Redis for ~30 frames.
                    # Only consume a command once per sequence edge, otherwise a
                    # single button press repeatedly saves tiny 1-frame episodes.
                    if (
                        new_recording_command != "none"
                        and sequence != self._latest_recording_control_sequence
                    ):
                        self._latest_recording_control_sequence = sequence
                        self._recording_command = new_recording_command

                        if new_recording_command == "start":
                            print(f"[{self.name}] 🔴 Recording started")
                        elif new_recording_command == "save":
                            print(f"[{self.name}] 💾 Recording saved and stopped")
                        elif new_recording_command == "cancel":
                            print(f"[{self.name}] ❌ Recording cancelled (not saved)")

                except Exception as e:
                    print(f"[{self.name}] Failed to parse recording control: {e}")
                    pass
            elif self._waiting_for_reset_complete:
                # While waiting for reset, ignore new commands from Redis
                pass

            self._twist2_hand_valid = action_left_raw is not None and action_right_raw is not None
            self._twist2_action_hand_left.copy_(
                torch.tensor(action_left, device=self.env.device, dtype=torch.float32).unsqueeze(0))
            self._twist2_action_hand_right.copy_(
                torch.tensor(action_right, device=self.env.device, dtype=torch.float32).unsqueeze(0))
            self._twist2_action_neck.copy_(
                torch.tensor(action_neck, device=self.env.device, dtype=torch.float32).unsqueeze(0))

            result = torch.tensor(action_body, device=self.env.device, dtype=torch.float32).unsqueeze(0)

            # 🔍 调试点4：检查返回的tensor
            # print(f"[XY_DEBUG] ✓ Returning tensor with xy_vel: {result[0, 0:2].cpu().numpy()}")

            return result
        except Exception as e:
            # print(f"[{self.name}] Redis action fetch failed: {e}")
            self._twist2_hand_valid = False
            self._twist2_human_smplx_valid = False
            self._twist2_human_info_valid = False
            self._twist2_controller_data = None
            # Return default 35D mimic_obs on error to maintain stable standing pose
            return self._default_mimic_obs.clone()

    def _twist2_publish_state(self, state_body, state_hand_left, state_hand_right, state_neck) -> None:
        if self.redis_pipeline is None or self._redis_state_publish_disabled:
            return
        try:
            self.redis_pipeline.set("state_body_unitree_g1_with_hands", json.dumps(state_body))
            self.redis_pipeline.set("state_hand_left_unitree_g1_with_hands", json.dumps(state_hand_left))
            self.redis_pipeline.set("state_hand_right_unitree_g1_with_hands", json.dumps(state_hand_right))
            self.redis_pipeline.set("state_neck_unitree_g1_with_hands", json.dumps(state_neck))
            self.redis_pipeline.set("t_state", int(time.time() * 1000))
            self.redis_pipeline.execute()
        except Exception as e:
            print(f"[{self.name}] Redis state publish failed once, disabling Redis state publish for this run: {e}")
            self._redis_state_publish_disabled = True
            self.redis_pipeline = None
            self.redis_client = None

    def get_human_smplx_data(self):
        """Get the most recent human SMPLX data (before GMR retargeting) from Redis.

        Returns:
            dict or None: Human SMPLX data dictionary if available and valid, None otherwise.
        """
        if self._twist2_human_smplx_valid:
            return self._twist2_human_smplx_data
        return None

    def is_human_smplx_data_valid(self):
        """Check if human SMPLX data is valid and available.

        Returns:
            bool: True if human SMPLX data is available and valid, False otherwise.
        """
        return self._twist2_human_smplx_valid

    def get_human_info(self):
        """Get the human information (height, etc.) from Redis.

        Returns:
            dict or None: Human info dictionary if available and valid, None otherwise.
                         Expected keys: 'height', 'neck_retarget_scale'
        """
        if self._twist2_human_info_valid:
            return self._twist2_human_info
        return None

    def is_human_info_valid(self):
        """Check if human info is valid and available.

        Returns:
            bool: True if human info is available and valid, False otherwise.
        """
        return self._twist2_human_info_valid

    def is_recording_active(self):
        """Check if recording is currently active.

        Returns:
            bool: True if recording is active, False otherwise.
        """
        return self._recording_active

    def get_recording_command(self):
        """Get the current recording command.

        Returns:
            str: Recording command - "none", "start", "save", or "cancel"
        """
        return self._recording_command

    def get_pending_save_jobs(self) -> int:
        return int(self._pending_save_jobs)

    def _twist2_roll_pitch_from_quaternion(self, quat: torch.Tensor) -> torch.Tensor:
        """Compute (roll, pitch) from quaternion (w, x, y, z)."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(t0, t1)
        t2 = 2.0 * (w * y - z * x)
        t2 = torch.clamp(t2, -1.0, 1.0)
        pitch = torch.asin(t2)
        return torch.stack([roll, pitch], dim=-1)

    def compute_current_observations(self):
        # Proprio from Isaac
        root_state = self.env.scene["robot"].data.root_state_w
        # 使用局部坐标系的角速度（与其他action provider一致）
        self.ang_vel = self.env.scene["robot"].data.root_ang_vel_b  # [1,3] 局部坐标系
        quat = root_state[:, 3:7]
        self.joint_pos = self.env.scene["robot"].data.joint_pos
        self.joint_vel = self.env.scene["robot"].data.joint_vel

        # roll/pitch from quaternion (TWIST2 convention)
        rp = self._twist2_roll_pitch_from_quaternion(quat)

        # TWIST2 29-dof vectors in the correct order
        idx = self.twist2_action_indices
        dof_pos = self.joint_pos[:, idx]
        dof_vel = self.joint_vel[:, idx]
        dof_pos_delta = dof_pos - self.twist2_default_pos

        # Publish state to Redis for teleop/data record
        state_body = torch.cat([self.ang_vel, rp, dof_pos], dim=-1)
        if self.enable_dex3 and hasattr(self, "_left_hand_target_idx_t"):
            left_state = self.joint_pos.index_select(1, self._left_hand_target_idx_t)
            right_state = self.joint_pos.index_select(1, self._right_hand_target_idx_t)
            state_hand_left = left_state.squeeze(0).tolist()
            state_hand_right = right_state.squeeze(0).tolist()
        else:
            state_hand_left = [0.0] * self._twist2_hand_dim
            state_hand_right = [0.0] * self._twist2_hand_dim
        state_neck = [0.0] * self._twist2_neck_dim
        self._twist2_publish_state(state_body.squeeze(0).tolist(), state_hand_left, state_hand_right, state_neck)

        # zero ankle velocities (TWIST2 convention)
        if len(self._twist2_ankle_idx) > 0:
            dof_vel = dof_vel.clone()
            dof_vel[:, self._twist2_ankle_idx] = 0.0

        obs_proprio = torch.cat(
            [
                self.ang_vel * 0.25,
                rp,
                dof_pos_delta,
                dof_vel * 0.05,
                self._twist2_last_action,
            ],
            dim=-1,
        )  # [1,92]

        if self._use_lerobot_vla:
            action_mimic = self._infer_lerobot_high_level_command()  # [1,35]
        else:
            action_mimic = self._twist2_fetch_actions()  # [1,35]

        velocity_amplification = 1.0
        action_mimic[:, 0:2] = action_mimic[:, 0:2] * velocity_amplification
        action_mimic[:, 5:6] = action_mimic[:, 5:6] * velocity_amplification

        obs_full = torch.cat([action_mimic, obs_proprio], dim=-1)  # [1,127]

        # Debug: Print observation components every 30 steps
        # if self.sim_step_counter % 30 == 0:
        #     print(f"\n[OBS_DEBUG] Step {self.sim_step_counter}")
        #     print(f"  action_mimic (XY vel): {action_mimic[0, :2].cpu().numpy()}")
        #     print(f"  action_mimic (joints 6-11): {action_mimic[0, 6:12].cpu().numpy()}")
        #     print(f"  ang_vel (local): {self.ang_vel[0].cpu().numpy()}")
        #     print(f"  dof_pos_delta (legs 0-5): {dof_pos_delta[0, :6].cpu().numpy()}")

        # Get history BEFORE updating (matches real robot server_low_level_g1_real.py line 256-257)
        obs_hist = self._twist2_history.reshape(1, -1)

        # Update history immediately after getting it (matches real robot timing)
        self._twist2_history = torch.roll(self._twist2_history, shifts=-1, dims=0)
        self._twist2_history[-1].copy_(obs_full.squeeze(0))

        # Future: current mimic (35)
        future_obs = action_mimic

        # Construct final observation buffer [obs_full(127) + obs_hist(1270) + future_obs(35)] = 1432
        obs_buf = torch.cat([obs_full, obs_hist, future_obs], dim=-1)  # [1,1402]

        return obs_buf

    def compute_observations(self):
        obs = self.compute_current_observations()
        obs = torch.clip(obs, -self.clip_obs, self.clip_obs)
        return obs

    def run_policy(self):
        import time
        redis_start = time.perf_counter()
        obs = self.compute_observations()
        redis_time = time.perf_counter() - redis_start

        # Store Redis fetch time (included in observation computation)
        if hasattr(self, '_perf_stats'):
            self._perf_stats['redis_fetch'].append(redis_time * 1000)

        with torch.no_grad():
            action = self.policy(obs)
        # Ensure shape [1,29]
        if isinstance(action, torch.Tensor) and action.dim() == 1:
            action = action.unsqueeze(0)
        # Update last_action (TWIST2)
        if isinstance(action, torch.Tensor) and action.shape[-1] == 29:
            self._twist2_last_action.copy_(action.to(self.env.device, dtype=torch.float32))
        return action, obs  # Return both action and observation

    def get_action(self, env) -> Optional[torch.Tensor]:
        """Get action from DDS"""
        import time
        total_start = time.perf_counter()

        recording_enabled = not getattr(self, "_disable_eval_recording", False)

        # Auto-start recording on first call (after env.reset() has been called)
        if recording_enabled and hasattr(self, '_should_start_recording_on_first_call') and self._should_start_recording_on_first_call:
            print(f"[{self.name}] 🔴 Starting recording on first get_action() call")
            print(f"[{self.name}] This ensures Frame 0 captures real physics state")
            self.recording_manager.start_recording()
            self._should_start_recording_on_first_call = False

        # Timing variables
        policy_time = 0.0
        action_prep_time = 0.0
        recording_data_time = 0.0
        physics_time = 0.0
        scene_update_time = 0.0
        obs_time = 0.0
        render_time = 0.0

        try:
            self._update_replay_reward_stats()
            full_action = self._full_action_buf
            full_action.zero_()
            replay_frame_idx = None

            # 1. Policy inference
            policy_start = time.perf_counter()
            if self._replay_enabled:
                replay_frame_idx = self._next_replay_frame_idx()
                if replay_frame_idx is None:
                    print(f"[{self.name}] Replay finished")
                    self._finalize_replay_if_needed()
                    if self._exit_when_replay_complete:
                        raise ReplayComplete(f"{self.name} replay rerecord complete")
                    return None
                target_29, obs_buf = self._run_replay_policy(replay_frame_idx)
            else:
                action_data, obs_buf = self.run_policy()  # Get both action and observation
            policy_time = time.perf_counter() - policy_start

            # 2. Action preparation
            action_prep_start = time.perf_counter()
            if self._replay_enabled:
                if replay_frame_idx % 100 == 0:
                    print(
                        f"[{self.name}] Replay progress: {replay_frame_idx}/{self._replay_num_frames} "
                        f"mode={self._replay_mode}"
                    )
            else:
                # --- TWIST2 full-body action (29-dof, MuJoCo actuator order) ---
                raw_action = torch.clip(action_data.to(self.env.device, dtype=torch.float32), -10.0, 10.0)

                # Server uses per-joint action_scale=0.5; keep it simple for quick dev
                target_29 = raw_action * 0.5 + self.twist2_default_pos  # [1,29]

            # Fill defaults first
            full_action.copy_(self.env.scene["robot"].data.default_joint_pos.squeeze(0))
            # Overwrite the 29 controlled joints
            if hasattr(self, "_twist2_action_idx_t"):
                full_action.index_copy_(0, self._twist2_action_idx_t, target_29.squeeze(0))
            else:
                full_action.index_copy_(0, torch.tensor(self.twist2_action_indices, device=self.env.device,
                                                        dtype=torch.long), target_29.squeeze(0))

            # 夹爪/手指（若有）
            hand_or_gripper_applied = False
            if self._apply_vla_gripper_command(full_action):
                hand_or_gripper_applied = True
            elif self.enable_dex3 and self._twist2_hand_valid and hasattr(self, "_left_hand_source_idx_t"):
                self._left_hand_buf.copy_(self._twist2_action_hand_left.squeeze(0))
                self._right_hand_buf.copy_(self._twist2_action_hand_right.squeeze(0))
                l_vals = self._left_hand_buf.index_select(0, self._left_hand_source_idx_t)
                r_vals = self._right_hand_buf.index_select(0, self._right_hand_source_idx_t)
                full_action.index_copy_(0, self._left_hand_target_idx_t, l_vals)
                full_action.index_copy_(0, self._right_hand_target_idx_t, r_vals)
                hand_or_gripper_applied = True

            if not hand_or_gripper_applied:
                if self.gripper_dds and hasattr(self, "_gripper_source_idx_t"):
                    gripper_cmd = self.gripper_dds.get_gripper_command()
                    if gripper_cmd:
                        left_gripper_cmd = gripper_cmd.get('left_gripper_cmd', {})
                        right_gripper_cmd = gripper_cmd.get('right_gripper_cmd', {})
                        left_gripper_positions = left_gripper_cmd.get('positions', [])
                        right_gripper_positions = right_gripper_cmd.get('positions', [])
                        gripper_positions = right_gripper_positions + left_gripper_positions
                        if len(gripper_positions) >= 2:
                            self._gripper_buf.copy_(
                                torch.tensor(gripper_positions[:2], dtype=torch.float32, device=self.env.device))
                            gp_vals = self._gripper_buf.index_select(0, self._gripper_source_idx_t)
                            full_action.index_copy_(0, self._gripper_target_idx_t, gp_vals)
                elif self.dex3_dds and hasattr(self, "_left_hand_source_idx_t"):
                    hand_cmds = self.dex3_dds.get_hand_commands()
                    if hand_cmds:
                        left_hand_cmd = hand_cmds.get('left_hand_cmd', {})
                        right_hand_cmd = hand_cmds.get('right_hand_cmd', {})
                        if left_hand_cmd and right_hand_cmd:
                            left_positions = left_hand_cmd.get('positions', [])
                            right_positions = right_hand_cmd.get('positions', [])
                            if len(left_positions) >= len(self._left_hand_buf) and len(right_positions) >= len(
                                    self._right_hand_buf):
                                self._left_hand_buf.copy_(
                                    torch.tensor(left_positions[:len(self._left_hand_buf)], dtype=torch.float32,
                                                 device=self.env.device))
                                self._right_hand_buf.copy_(
                                    torch.tensor(right_positions[:len(self._right_hand_buf)], dtype=torch.float32,
                                                 device=self.env.device))
                                l_vals = self._left_hand_buf.index_select(0, self._left_hand_source_idx_t)
                                r_vals = self._right_hand_buf.index_select(0, self._right_hand_source_idx_t)
                                full_action.index_copy_(0, self._left_hand_target_idx_t, l_vals)
                                full_action.index_copy_(0, self._right_hand_target_idx_t, r_vals)
                elif self.inspire_dds and hasattr(self, "_inspire_source_idx_t"):
                    inspire_cmds = self.inspire_dds.get_inspire_hand_command()
                    if inspire_cmds and 'positions' in inspire_cmds:
                        inspire_cmds_positions = inspire_cmds['positions']
                        if len(inspire_cmds_positions) >= 12:
                            self._inspire_buf.copy_(
                                torch.tensor(inspire_cmds_positions[:12], dtype=torch.float32, device=self.env.device))
                            base_vals = self._inspire_buf.index_select(0, self._inspire_source_idx_t)
                            full_action.index_copy_(0, self._inspire_target_idx_t, base_vals)
                            special_vals = self._inspire_buf.index_select(0,
                                                                          self._inspire_special_source_idx_t) * self._inspire_special_scales_t
                            full_action.index_copy_(0, self._inspire_special_target_idx_t, special_vals)
            action_prep_time = time.perf_counter() - action_prep_start

            # 2.5. Collect recording data (before decimation loop)
            # This captures the state before physics simulation steps
            recording_data = None
            if recording_enabled:
                recording_data_start = time.perf_counter()
                recording_data = self.collect_recording_data(obs_buf, target_29)
                recording_data_time = time.perf_counter() - recording_data_start

            # 2.6. Recording state machine
            # Debug: Print recording command state
            if self._recording_command != "none":
                print(f"[{self.name}] 🔍 Recording command received: {self._recording_command}")
                print(f"[{self.name}] 🔍 Recording manager state: is_recording={self.recording_manager.is_recording}, buffer_size={len(self.recording_manager.recording_buffer)}")

            # Teleop server emits a "start" edge on init and after every ready epoch.
            # If recording is already running locally, that edge is only an ack and
            # must not be reinterpreted as "save", otherwise we immediately flush a
            # 1-frame episode and break save_and_reset semantics.
            if recording_enabled and self._recording_command == "start" and self.recording_manager.is_recording:
                print(f"[{self.name}] 🔄 Already recording: ignoring duplicate 'start' edge")
                self._recording_command = "none"

            # Handle recording commands from Redis
            if recording_enabled and self._recording_command == "start" and not self.recording_manager.is_recording:
                self.recording_manager.start_recording()
                self._recording_display_state = "recording"
                self._recording_display_counter = 0
                self._save_in_progress = False
                # Reset command after processing (one-time trigger)
                self._recording_command = "none"

            elif recording_enabled and self._recording_command == "save":
                # Define callback to update display state when save completes
                def on_save_complete(success: bool):
                    # Set completion flag (thread-safe: single write operation)
                    self._save_completion_state = "success" if success else "failure"
                    print(f"[{self.name}] 📝 Save completed: {self._save_completion_state}")

                # Start save with callback
                print(f"[{self.name}] 💾 Starting save, setting state to 'saving'")
                self.recording_manager.save_recording(completion_callback=on_save_complete)
                self._recording_display_state = "saving"  # Show "saving" while in progress
                self._save_in_progress = True
                self._pending_save_jobs = self.recording_manager.get_pending_save_count()
                self._save_completion_state = None  # Reset completion state
                print(
                    f"[{self.name}] 🔍 After save command: display_state={self._recording_display_state}, "
                    f"save_in_progress={self._save_in_progress}, pending={self._pending_save_jobs}"
                )
                # Reset command after processing
                self._recording_command = "none"

            elif recording_enabled and self._recording_command == "cancel":
                self.recording_manager.cancel_recording()
                self._recording_display_state = "discard"
                self._recording_display_counter = 0
                self._pending_save_jobs = self.recording_manager.get_pending_save_count()
                self._save_in_progress = self._pending_save_jobs > 0
                # Reset command after processing
                self._recording_command = "none"

            elif recording_enabled and self._recording_command == "save_and_reset":
                # Save recording asynchronously, then trigger reset immediately
                print(f"[{self.name}] 💾 save_and_reset command received")

                def on_save_complete(success: bool):
                    self._save_completion_state = "success" if success else "failure"
                    print(f"[{self.name}] 📝 save_and_reset completed: {self._save_completion_state}")

                print(f"[{self.name}] 💾 Saving recording...")
                self.recording_manager.save_recording(completion_callback=on_save_complete)
                self._recording_display_state = "saving"
                self._save_in_progress = True
                self._pending_save_jobs = self.recording_manager.get_pending_save_count()
                print(f"[{self.name}] 📦 Save queued, pending={self._pending_save_jobs}")

                # Trigger complete reset via Redis
                print(f"[{self.name}] 🔄 Triggering complete reset...")
                self._trigger_complete_reset()

                # Wait for reset complete signal
                print(f"[{self.name}] ⏳ Waiting for reset to complete...")
                self._waiting_for_reset_complete = True
                self._reset_complete_received = False

                # Reset command after processing
                self._recording_command = "none"

            elif recording_enabled and self._recording_command == "discard_and_reset":
                # Discard recording, then trigger reset
                print(f"[{self.name}] ❌ discard_and_reset command received")

                # Discard recording (immediate)
                print(f"[{self.name}] ❌ Discarding recording...")
                self.recording_manager.cancel_recording()
                print(f"[{self.name}] ✅ Recording discarded")
                self._pending_save_jobs = self.recording_manager.get_pending_save_count()
                self._save_in_progress = self._pending_save_jobs > 0

                # Trigger complete reset via Redis
                print(f"[{self.name}] 🔄 Triggering complete reset...")
                self._trigger_complete_reset()

                # Wait for reset complete signal
                print(f"[{self.name}] ⏳ Waiting for reset to complete...")
                self._waiting_for_reset_complete = True
                self._reset_complete_received = False

                # Reset command after processing
                self._recording_command = "none"

            # Check for reset complete signal from sim_main
            if recording_enabled and self._waiting_for_reset_complete:
                reset_complete = self._check_reset_complete()
                if reset_complete:
                    print(f"[{self.name}] ✅ Reset complete signal received")
                    self._waiting_for_reset_complete = False
                    self._reset_complete_received = True

                    # Reset internal buffers after environment reset
                    print(f"[{self.name}] 🔄 Resetting internal buffers...")
                    self._reset_internal_buffers()

                    # Start new recording immediately after reset
                    print(f"[{self.name}] 🔴 Starting new recording after reset...")
                    self.recording_manager.start_recording()
                    self._recording_display_state = "recording"
                    print(f"[{self.name}] ✅ New recording started")

            # Update display state counter and transitions
            # Priority order: check save completion > saving > saved/discard countdown > recording > idle

            # Debug: print state before transitions
            if recording_enabled and self.sim_step_counter % 10 == 0 and (self._save_in_progress or self._recording_display_state != "idle"):
                print(f"[{self.name}] 🔍 State check (step {self.sim_step_counter}): display={self._recording_display_state}, save_in_progress={self._save_in_progress}, is_recording={self.recording_manager.is_recording}, completion={self._save_completion_state}")

            # Check if save completed (set by background thread callback)
            if recording_enabled:
                self._pending_save_jobs = self.recording_manager.get_pending_save_count()
                self._save_in_progress = self._pending_save_jobs > 0
            else:
                self._pending_save_jobs = 0
                self._save_in_progress = False
                self._recording_display_state = "idle"
            if recording_enabled and self._save_completion_state is not None:
                if self._save_completion_state == "success":
                    self._recording_display_state = "saved"
                else:  # "failure"
                    self._recording_display_state = "discard"
                self._recording_display_counter = 0
                self._save_completion_state = None  # Clear the flag
                print(f"[{self.name}] 🔄 Display state updated to: {self._recording_display_state}")
            # Normal state transitions (only if save didn't just complete)
            elif recording_enabled and self._save_in_progress:
                # Keep showing "saving" while save is in progress
                # State will be updated when _save_completion_state is set
                # Don't change _recording_display_state here
                pass
            elif recording_enabled and self._recording_display_state in ["saved", "discard"]:
                # Count down the display duration for saved/discard states
                self._recording_display_counter += 1
                if self._recording_display_counter >= self._recording_display_duration:
                    self._recording_display_state = "idle"
                    self._recording_display_counter = 0
            elif recording_enabled and self.recording_manager.is_recording:
                # Currently recording
                self._recording_display_state = "recording"
            elif recording_enabled and not self.recording_manager.is_recording and self._recording_display_state == "recording":
                # Recording stopped but no save/cancel command yet (shouldn't happen normally)
                # Only transition to idle if we're not in the middle of saving
                if not self._save_in_progress:
                    print(f"[{self.name}] ⚠️ Recording stopped without save/cancel, transitioning to idle")
                    self._recording_display_state = "idle"

            # Add frame to recording buffer if recording is active
            if recording_enabled and self.recording_manager.is_recording and recording_data is not None:
                self.recording_manager.add_frame(recording_data)

            # 3. Physics simulation loop
            physics_total = 0.0
            scene_update_total = 0.0
            render_total = 0.0

            for i in range(self._twist2_decimation):
                # Set joint targets and write to sim
                step_start = time.perf_counter()
                self.env.scene["robot"].set_joint_position_target(full_action)
                self.env.scene.write_data_to_sim()

                # Physics step with optional rendering
                is_last_step = (i == self._twist2_decimation - 1)
                should_render = is_last_step and (self._should_refresh_lerobot_visuals_next_step() if self._use_lerobot_vla else (self._render_counter % self._render_interval == 0))

                if should_render:
                    render_start = time.perf_counter()
                    self.env.sim.step(render=True)
                    render_total += time.perf_counter() - render_start
                else:
                    self.env.sim.step(render=False)

                physics_total += time.perf_counter() - step_start - (render_total if should_render else 0)

                # Scene update
                update_start = time.perf_counter()
                self.env.scene.update(dt=self.env.physics_dt)
                scene_update_total += time.perf_counter() - update_start

                # Debug: Fix root in air (after scene update to override physics)
                if self._debug_fix_root_in_air:
                    # Get current root state [1, 13]: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
                    root_state = self.env.scene["robot"].data.root_state_w.clone()
                    # Fix position to target height
                    root_state[0, 0:3] = self._debug_fixed_root_position
                    # Fix orientation to upright
                    root_state[0, 3:7] = self._debug_fixed_root_orientation
                    # Zero out all velocities (prevents drift)
                    root_state[0, 7:13] = 0.0
                    # Write back to simulation
                    self.env.scene["robot"].write_root_state_to_sim(root_state, env_ids=torch.tensor([0], device=self.env.device))

            physics_time = physics_total
            scene_update_time = scene_update_total
            render_time = render_total
            self._render_counter += 1

            if self._replay_enabled and replay_frame_idx is not None:
                self._log_replay_joint_position_error(replay_frame_idx)
                self._log_replay_object_state_error(replay_frame_idx)

            # 4. Observation computation
            obs_start = time.perf_counter()
            self._obs_counter += 1
            should_compute_obs = self._should_refresh_lerobot_visuals_next_step() if self._use_lerobot_vla else (self._obs_counter % self._obs_interval == 0)
            if should_compute_obs:
                self.env.observation_manager.compute()
                obs_time = time.perf_counter() - obs_start

            # Record performance stats
            total_time = time.perf_counter() - total_start
            self._perf_stats['policy_inference'].append(policy_time * 1000)
            self._perf_stats['action_preparation'].append(action_prep_time * 1000)
            self._perf_stats['recording_data_collection'].append(recording_data_time * 1000)
            self._perf_stats['physics_step'].append(physics_time * 1000)
            self._perf_stats['scene_update'].append(scene_update_time * 1000)
            self._perf_stats['observation_compute'].append(obs_time * 1000)
            self._perf_stats['render_time'].append(render_time * 1000)
            self._perf_stats['total_get_action'].append(total_time * 1000)

            # Report performance statistics
            self.sim_step_counter += 1
            if self.sim_step_counter % self._perf_report_interval == 0:
                self._print_performance_report()

            return full_action
        except ReplayComplete:
            raise
        except Exception as e:
            print(f"[{self.name}] Get DDS action failed: {e}")
            return None

    def _print_performance_report(self):
        """Print detailed performance statistics"""
        print("\n" + "="*80)
        print(f"🔍 PERFORMANCE ANALYSIS (last {self._perf_report_interval} steps)")
        print("="*80)

        # Define display order and labels
        stat_labels = {
            'redis_fetch': 'Redis Fetch (in obs)',
            'policy_inference': 'Policy Inference',
            'action_preparation': 'Action Preparation',
            'recording_data_collection': 'Recording Data Collection',
            'physics_step': 'Physics Step',
            'scene_update': 'Scene Update',
            'render_time': 'Render Time',
            'observation_compute': 'Observation Compute',
            'total_get_action': 'Total get_action'
        }

        for key in stat_labels.keys():
            values = self._perf_stats.get(key, [])
            if not values:
                continue

            avg = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)

            # Calculate percentage of total time
            if key != 'total_get_action':
                total_avg = sum(self._perf_stats['total_get_action']) / len(self._perf_stats['total_get_action'])
                percentage = (avg / total_avg * 100) if total_avg > 0 else 0
                print(f"  {stat_labels[key]:30s}: avg={avg:6.2f}ms  min={min_val:6.2f}ms  max={max_val:6.2f}ms  ({percentage:5.1f}%)")
            else:
                print(f"  {stat_labels[key]:30s}: avg={avg:6.2f}ms  min={min_val:6.2f}ms  max={max_val:6.2f}ms")

        # Calculate theoretical max frequency
        total_avg = sum(self._perf_stats['total_get_action']) / len(self._perf_stats['total_get_action'])
        max_freq = 1000.0 / total_avg if total_avg > 0 else 0
        print(f"\n  Theoretical max frequency: {max_freq:.1f} Hz")
        print(f"  Render interval: {self._render_interval}, Obs interval: {self._obs_interval}")
        print(f"  Recording active: {self._recording_active}, Command: {self._recording_command}")
        print("="*80 + "\n")

        # Clear stats for next interval
        for key in self._perf_stats:
            self._perf_stats[key].clear()

    def _get_observation_semantics(self) -> dict:
        """Get detailed semantics for observation buffer dimensions.

        Returns:
            dict: Detailed semantic description of observation buffer structure
        """
        semantics = {
            "total_dims": self.total_obs_size,  # 1432
            "structure": {
                "obs_full": {
                    "dims": [0, 127],
                    "description": "Current full observation",
                    "components": {
                        "action_mimic": {
                            "dims": [0, 35],
                            "description": "Mimic action from Redis teleop",
                            "components": {
                                "xy_vel": {
                                    "dims": [0, 2],
                                    "description": "XY velocity command",
                                    "unit": "m/s"
                                },
                                "z_pos": {
                                    "dims": [2, 3],
                                    "description": "Z position target",
                                    "unit": "m"
                                },
                                "roll_pitch": {
                                    "dims": [3, 5],
                                    "description": "Roll and pitch angles",
                                    "unit": "rad"
                                },
                                "yaw_vel": {
                                    "dims": [5, 6],
                                    "description": "Yaw angular velocity",
                                    "unit": "rad/s"
                                },
                                "joint_targets": {
                                    "dims": [6, 35],
                                    "description": "29 DOF joint position targets",
                                    "unit": "rad"
                                }
                            }
                        },
                        "obs_proprio": {
                            "dims": [35, 127],
                            "description": "Proprioceptive observations",
                            "components": {
                                "ang_vel_scaled": {
                                    "dims": [35, 38],
                                    "description": "Angular velocity * 0.25",
                                    "unit": "rad/s"
                                },
                                "roll_pitch": {
                                    "dims": [38, 40],
                                    "description": "Roll and pitch from quaternion",
                                    "unit": "rad"
                                },
                                "dof_pos_delta": {
                                    "dims": [40, 69],
                                    "description": "Joint position - default position (29 DOF)",
                                    "unit": "rad"
                                },
                                "dof_vel_scaled": {
                                    "dims": [69, 98],
                                    "description": "Joint velocity * 0.05 (29 DOF)",
                                    "unit": "rad/s"
                                },
                                "last_action": {
                                    "dims": [98, 127],
                                    "description": "Previous action output (29 DOF)",
                                    "unit": "rad"
                                }
                            }
                        }
                    }
                },
                "obs_hist": {
                    "dims": [127, 1397],
                    "description": "10 frames of historical obs_full (127*10=1270)",
                    "unit": "various"
                },
                "future_obs": {
                    "dims": [1397, 1432],
                    "description": "Current action_mimic (35D)",
                    "unit": "various"
                }
            }
        }
        return semantics

    def _add_recording_status_overlay(self, rgb_image):
        """Add recording status overlay to RGB image for display.

        Args:
            rgb_image: numpy array [H, W, 3] in range [0, 1] (float) or [0, 255] (uint8)

        Returns:
            numpy array with recording status overlay
        """
        import cv2
        import numpy as np

        # Convert to uint8 if needed
        if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
            img = (rgb_image * 255).astype(np.uint8)
        else:
            img = rgb_image.copy()

        # Get image dimensions
        h, w = img.shape[:2]

        # Define overlay parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        padding = 10
        circle_radius = 15

        # Determine status based on recording display state
        # Priority order: saving > saved > discard > recording > idle
        if self._recording_display_state == "saving":
            # Currently saving - orange dot
            text = "SAVING..."
            color = (0, 165, 255)  # Orange in BGR (B=0, G=165, R=255)
        elif self._recording_display_state == "saved":
            # Just saved - green dot
            text = "SAVED"
            color = (0, 255, 0)  # Green in BGR
        elif self._recording_display_state == "discard":
            # Just cancelled - red dot
            text = "DISCARD"
            color = (0, 0, 255)  # Red in BGR
        elif self._recording_display_state == "recording":
            # Currently recording - yellow dot
            text = "RECORDING"
            color = (0, 255, 255)  # Yellow in BGR (B=0, G=255, R=255)
        else:  # idle
            # Idle - green dot
            text = "IDLE"
            color = (0, 255, 0)  # Green in BGR

        # Draw filled circle (status indicator)
        circle_center = (padding + circle_radius, padding + circle_radius)
        cv2.circle(img, circle_center, circle_radius, color, -1)

        # Draw text next to circle
        text_pos = (padding + circle_radius * 2 + 10, padding + circle_radius + 8)
        cv2.putText(img, text, text_pos, font, font_scale, color, thickness, cv2.LINE_AA)

        # Convert back to float if original was float
        if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
            img = img.astype(np.float32) / 255.0

        return img

    def collect_recording_data(self, obs_buf: torch.Tensor, target_29: torch.Tensor) -> dict:
        """Collect all data needed for recording.

        This method collects comprehensive data for recording, including:
        - Human data (SMPLX, hand control)
        - Environment objects (football position/velocity)
        - Robot state (qpos, qvel, root state, vision)
        - System info (control frequency, task name)

        Args:
            obs_buf: Observation buffer [1, 1402] from compute_observations()
            target_29: Target joint positions [1, 29] from policy inference

        Returns:
            dict: Organized recording data with keys: human, env_obj, robot, system, task
        """
        import numpy as np
        import time

        recording_data = {}

        # ===== HUMAN DATA =====
        human_data = {}

        # Human SMPLX data before GMR retargeting
        if self._twist2_human_smplx_valid and self._twist2_human_smplx_data is not None:
            human_data["smplx_data_before_gmr"] = self._twist2_human_smplx_data
        else:
            human_data["smplx_data_before_gmr"] = None

        # Human info (height, etc.)
        if self._twist2_human_info_valid and self._twist2_human_info is not None:
            human_data["human_info"] = self._twist2_human_info
        else:
            human_data["human_info"] = None

        human_data["controller_binary"] = _extract_controller_binary_signals(self._twist2_controller_data)

        # Hand control data
        human_data["hand_control"] = {
            "left": self._twist2_action_hand_left.cpu().numpy().squeeze(0),  # [7]
            "right": self._twist2_action_hand_right.cpu().numpy().squeeze(0),  # [7]
            "neck": self._twist2_action_neck.cpu().numpy().squeeze(0)  # [2]
        }

        recording_data["human"] = human_data

        # ===== ENVIRONMENT OBJECTS =====
        env_obj_data = self._collect_env_state()
        recording_data["env_obj"] = env_obj_data
        recording_data["episode_init_env"] = self._episode_init_env_state
        object_seed_info = get_current_episode_object_seed_info(self.env.cfg)
        recording_data["episode_init_meta"] = {
            "object_seed": object_seed_info.get("seed"),
            "object_seed_source": object_seed_info.get("source"),
        }

        # ===== ROBOT DATA =====
        robot_data = {}

        # Get robot state (before decimation)
        root_state = self.env.scene["robot"].data.root_state_w  # [1, 13]

        # Joint positions and velocities (29 DOF, before decimation)
        idx = self.twist2_action_indices
        robot_data["qpos_before_decimation"] = self.joint_pos[0, idx].cpu().numpy()  # [29]
        robot_data["qvel_before_decimation"] = self.joint_vel[0, idx].cpu().numpy()  # [29]

        # Root state - Complete velocity information for replay
        # Position and orientation
        robot_data["root_position"] = root_state[0, 0:3].cpu().numpy()  # [3]
        robot_data["root_orientation"] = root_state[0, 3:7].cpu().numpy()  # [4] quaternion (w,x,y,z)

        # Velocities - Record both local and world frame for maximum compatibility
        robot_data["root_lin_vel_local"] = self.env.scene["robot"].data.root_lin_vel_b[0].cpu().numpy()  # [3] local frame
        robot_data["root_ang_vel_local"] = self.env.scene["robot"].data.root_ang_vel_b[0].cpu().numpy()  # [3] local frame
        robot_data["root_lin_vel_world"] = root_state[0, 7:10].cpu().numpy()  # [3] world frame
        robot_data["root_ang_vel_world"] = root_state[0, 10:13].cpu().numpy()  # [3] world frame

        # TWIST2 inference output
        robot_data["twist2_inference_qpos"] = target_29.cpu().numpy().squeeze(0)  # [29]

        try:
            robot_data["applied_torque_before_decimation"] = (
                self.env.scene["robot"].data.applied_torque[0, idx].cpu().numpy()
            )
        except Exception:
            robot_data["applied_torque_before_decimation"] = None

        try:
            robot_data["body_net_contact_forces"] = (
                self.env.scene["robot"].data.body_net_contact_force_w[0].cpu().numpy()
            )
        except Exception:
            robot_data["body_net_contact_forces"] = None

        # Observation data
        robot_data["observation"] = {
            "obs_buf": obs_buf.cpu().numpy().squeeze(0),  # [1402]
            "semantics": self._get_observation_semantics()
        }

        # Vision data (from previous control cycle's last rendered frame)
        vision_data = {}
        try:
            if "front_camera" in self.env.scene.keys():
                camera = self.env.scene["front_camera"]

                # RGB image
                if "rgb" in camera.data.output:
                    rgb_tensor = camera.data.output["rgb"][0]  # [H, W, 3]
                    rgb_array = rgb_tensor.cpu().numpy()

                    # Store original image for saving (without overlay)
                    vision_data["rgb"] = rgb_array.copy()

                    # Create a copy with recording status overlay for Redis display
                    vision_data["rgb_display"] = self._add_recording_status_overlay(rgb_array.copy())
                else:
                    vision_data["rgb"] = None
                    vision_data["rgb_display"] = None
                    print(f"[{self.name}] Warning: RGB data not available in front_camera")

                # Depth image
                if "distance_to_image_plane" in camera.data.output:
                    depth_tensor = camera.data.output["distance_to_image_plane"][0]  # [H, W]
                    vision_data["depth"] = depth_tensor.cpu().numpy()
                else:
                    vision_data["depth"] = None
                    print(f"[{self.name}] Warning: Depth data not available in front_camera")
            else:
                vision_data["rgb"] = None
                vision_data["rgb_display"] = None
                vision_data["depth"] = None
                print(f"[{self.name}] Warning: front_camera not found in scene")

            if "left_wrist_camera" in self.env.scene.keys():
                left_camera = self.env.scene["left_wrist_camera"]
                if "rgb" in left_camera.data.output:
                    vision_data["left_wrist_rgb"] = left_camera.data.output["rgb"][0].cpu().numpy().copy()
                else:
                    vision_data["left_wrist_rgb"] = None
                if "distance_to_image_plane" in left_camera.data.output:
                    vision_data["left_wrist_depth"] = left_camera.data.output["distance_to_image_plane"][0].cpu().numpy().copy()
                else:
                    vision_data["left_wrist_depth"] = None
            else:
                vision_data["left_wrist_rgb"] = None
                vision_data["left_wrist_depth"] = None

            if "right_wrist_camera" in self.env.scene.keys():
                right_camera = self.env.scene["right_wrist_camera"]
                if "rgb" in right_camera.data.output:
                    vision_data["right_wrist_rgb"] = right_camera.data.output["rgb"][0].cpu().numpy().copy()
                else:
                    vision_data["right_wrist_rgb"] = None
                if "distance_to_image_plane" in right_camera.data.output:
                    vision_data["right_wrist_depth"] = right_camera.data.output["distance_to_image_plane"][0].cpu().numpy().copy()
                else:
                    vision_data["right_wrist_depth"] = None
            else:
                vision_data["right_wrist_rgb"] = None
                vision_data["right_wrist_depth"] = None
        except Exception as e:
            print(f"[{self.name}] Failed to get camera data: {e}")
            vision_data["rgb"] = None
            vision_data["rgb_display"] = None
            vision_data["depth"] = None
            vision_data["left_wrist_rgb"] = None
            vision_data["left_wrist_depth"] = None
            vision_data["right_wrist_rgb"] = None
            vision_data["right_wrist_depth"] = None

        robot_data["vision"] = vision_data

        action_mimic = obs_buf[0, :35].detach().cpu().numpy().astype(np.float32)
        joint_pos_canonical = reorder_twist2_to_canonical_29(robot_data["qpos_before_decimation"])
        joint_vel_canonical = reorder_twist2_to_canonical_29(robot_data["qvel_before_decimation"])
        canonical_state = build_vla_observation_state(
            root_orientation_wxyz=robot_data["root_orientation"],
            joint_pos_canonical_29=joint_pos_canonical,
            joint_vel_canonical_29=joint_vel_canonical,
        )
        if self._use_lerobot_vla and self._latest_vla_action is not None:
            canonical_action = np.asarray(self._latest_vla_action, dtype=np.float32).copy()
        else:
            controller_binary = human_data["controller_binary"]
            canonical_action = self._canonical_action_recorder.step(
                mimic_obs=action_mimic,
                hand_binary=np.array(
                    [
                        float(controller_binary.get("left_grip_binary", False)),
                        float(controller_binary.get("right_grip_binary", False)),
                    ],
                    dtype=np.float32,
                ),
            )

        robot_data["action_mimic"] = action_mimic
        robot_data["vla_state"] = canonical_state
        robot_data["vla_action"] = canonical_action
        recording_data["robot"] = robot_data

        # ===== SYSTEM DATA =====
        system_data = {
            "control_frequency": 1.0 / (self._twist2_decimation * self.env.physics_dt),  # Hz
            "decimation": self._twist2_decimation,
            "physics_dt": self.env.physics_dt,
            "timestamp": time.time()
        }

        recording_data["system"] = system_data

        # ===== TASK NAME =====
        recording_data["task"] = self.task_name

        return recording_data

    def _trigger_complete_reset(self):
        """Trigger complete reset (including PhysX) via Redis."""
        import redis
        import json
        import time

        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=0)

            # Send complete reset command (category 3 = complete reset including PhysX)
            reset_command = {
                "reset_category": "3",  # Complete reset (PhysX + all entities)
                "timestamp": int(time.time() * 1000)
            }

            redis_client.set("isaac_reset_trigger", json.dumps(reset_command))
            redis_client.expire("isaac_reset_trigger", 5)  # Auto-expire after 5 seconds

            print(f"[{self.name}] ✅ Complete reset command sent via Redis")

        except Exception as e:
            print(f"[{self.name}] ❌ Failed to send reset command: {e}")
            import traceback
            traceback.print_exc()

    def _check_reset_complete(self) -> bool:
        """Check if reset complete signal has been received from sim_main."""
        try:
            import redis
            import json

            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

            # Check for reset complete signal
            reset_complete_signal = redis_client.get("isaac_reset_complete_unitree_g1_with_hands")

            if reset_complete_signal:
                reset_data = json.loads(reset_complete_signal)
                status = reset_data.get("status", "")

                if status == "complete":
                    # Clear the signal
                    redis_client.delete("isaac_reset_complete_unitree_g1_with_hands")
                    return True

            return False

        except Exception as e:
            print(f"[{self.name}] ❌ Failed to check reset complete: {e}")
            return False

    def _reset_internal_buffers(self):
        """Reset internal buffers after environment reset to ensure clean state."""
        try:
            self._episode_init_env_state = self._collect_env_state()

            # Reset TWIST2 history buffer
            self._twist2_history.zero_()

            # Reset last action buffer
            self._twist2_last_action.zero_()

            # Reset observation buffer
            self._twist2_obs_buf.zero_()

            # Reset hand action buffers
            self._twist2_action_hand_left.zero_()
            self._twist2_action_hand_right.zero_()
            self._twist2_action_neck.zero_()
            self._vla_gripper_binary.zero_()

            # Reset hand validity flag
            self._twist2_hand_valid = False
            self._latest_vla_action = None
            self._canonical_action_recorder.reset()

            # Reset SMPLX data validity flags
            self._twist2_smplx_valid = False
            self._twist2_human_smplx_valid = False
            self._twist2_human_info_valid = False
            self._twist2_controller_data = None

            # Reset recording command state
            self._recording_command = "none"
            self._recording_active = False
            if getattr(self, "_disable_eval_recording", False):
                self._should_start_recording_on_first_call = False
                self._recording_display_state = "idle"
                self._save_in_progress = False
                self._pending_save_jobs = 0
                try:
                    self.recording_manager.cancel_recording()
                except Exception:
                    pass
            self._input_ready_timestamp_ms = 0
            self._input_ready_epoch_id = -1
            self._stale_input_drop_logged_epoch = -1
            self._replay_joint_mae_sum = 0.0
            self._replay_joint_mae_count = 0
            self._replay_object_err_sums = {}
            self._replay_object_err_counts = {}
            self._replay_completion_requested = False
            self._replay_reward_max = None
            self._replay_any_success = False
            if self._use_lerobot_vla:
                self._lerobot_action_chunk_queue.clear()
                self._lerobot_observation_state_segment.clear()
                self._lerobot_observation_state_last_step = -1
                try:
                    root_quat_wxyz, root_xy_world = self._get_current_robot_root_pose_for_vla()
                    self._vla_initial_robot_quat_wxyz = root_quat_wxyz.copy()
                    self._lerobot_vla_runtime.reset(
                        body_xy_world=root_xy_world,
                        target_root_quat_wxyz=root_quat_wxyz,
                    )
                except Exception:
                    self._vla_initial_robot_quat_wxyz = None
                    self._lerobot_vla_runtime.reset()
                if self._lerobot_http_client is not None:
                    self._lerobot_http_client.reset()
                if self._lerobot_policy is not None:
                    self._lerobot_policy.reset()
                if self._lerobot_preprocessor is not None:
                    self._lerobot_preprocessor.reset()
                if self._lerobot_postprocessor is not None:
                    self._lerobot_postprocessor.reset()

            print(f"[{self.name}] ✅ Internal buffers reset successfully")

        except Exception as e:
            print(f"[{self.name}] ❌ Failed to reset internal buffers: {e}")
            import traceback
            traceback.print_exc()

    def on_env_objects_reset(self):
        self._episode_init_env_state = self._collect_env_state()

    def _update_replay_reward_stats(self) -> None:
        if not (self._replay_enabled and self._record_during_replay):
            return
        if self._replay_cursor <= 0:
            return
        reward = _reward_scalar(self.env)
        if self._replay_reward_max is None:
            self._replay_reward_max = reward
        else:
            self._replay_reward_max = max(self._replay_reward_max, reward)
        if _reward_success_flag(reward):
            self._replay_any_success = True

    def _convert_to_joint_range(self, value):
        """Convert gripper control value to joint angle"""
        input_min, input_max = 0.0, 5.6
        output_min, output_max = 0.03, -0.02
        value = max(input_min, min(input_max, value))
        return output_min + (output_max - output_min) * (value - input_min) / (input_max - input_min)

    def cleanup(self):
        """Clean up DDS resources and recording manager"""
        try:
            # Shutdown recording manager first (wait for pending saves)
            if hasattr(self, 'recording_manager'):
                self.recording_manager.shutdown()

            # Clean up DDS resources
            if self.robot_dds:
                self.robot_dds.stop_communication()
            if self.gripper_dds:
                self.gripper_dds.stop_communication()
            if self.dex3_dds:
                self.dex3_dds.stop_communication()
            if self.inspire_dds:
                self.inspire_dds.stop_communication()
        except Exception as e:
            print(f"[{self.name}] Clean up resources failed: {e}")
