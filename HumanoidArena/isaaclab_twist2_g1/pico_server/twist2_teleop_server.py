"""
conda activate gmr
sudo ufw disable
python xrobot_teleop_to_robot_w_hand.py --robot unitree_g1

State Machine Controls:
- Right controller key_one: Cycle through idle -> teleop -> pause -> teleop...
- Left controller key_one: Exit program from any state
- Left controller axis_click: Emergency stop - kills sim2real.sh process
- Left controller axis: Control root xy velocity and yaw velocity
- Right controller axis: Fine-tune root xy velocity and yaw velocity
- Auto-transition: idle -> teleop when motion data is available

States:
- idle: Waiting for input or data
- teleop: Processing motion retargeting with velocity control
- pause: Data received but not processing
- exit: Program will terminate

Whole-Body Teleop Features:
- Sends whole-body mode information to Redis
- 35-dimensional mimic observations
- Uses retargeted motion directly from the teleoperation stream
"""
import argparse
import json
import pathlib
import os
import subprocess
import sys
import time

def _prefer_workspace_gmr():
    """Prefer the checked-out GMR package when the installed egg misses assets."""
    candidate_roots = []

    gmr_root_env = os.environ.get("GMR_ROOT")
    if gmr_root_env:
        candidate_roots.append(pathlib.Path(gmr_root_env).expanduser())

    candidate_roots.append(pathlib.Path(__file__).resolve().parents[2] / "GMR")

    for gmr_root in candidate_roots:
        package_dir = gmr_root / "general_motion_retargeting"
        unitree_g1_xml = gmr_root / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
        if package_dir.is_dir() and unitree_g1_xml.is_file():
            gmr_root_str = str(gmr_root)
            if gmr_root_str not in sys.path:
                sys.path.insert(0, gmr_root_str)
            os.environ.setdefault("GMR_ROOT", gmr_root_str)
            return


_prefer_workspace_gmr()
project_root = pathlib.Path(__file__).resolve().parents[1]
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R
import torch
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import draw_frame
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT
from general_motion_retargeting import human_head_to_robot_neck
from rich import print
from tqdm import tqdm
import cv2
import redis
from rich import print
from general_motion_retargeting import XRobotStreamer
from action_provider.reset_control import (
    GMR_BODY_POS_KEY,
    GMR_BODY_QUAT_W_KEY,
    GMR_FRAME_INDEX_KEY,
    GMR_FULL_QPOS_KEY,
    GMR_JOINT_POS_KEY,
    GMR_JOINT_VEL_KEY,
    get_input_ready_key,
)

from data_utils.params import (
    DEFAULT_HAND_POSE,
    DEFAULT_MIMIC_OBS,
    HAND_MOVEMENT_STEP,
)
from data_utils.rot_utils import euler_from_quaternion_np, quat_diff_np, quat_rotate_inverse_np
from data_utils.fps_monitor import FPSMonitor
from pico_server.sonic_tools.trl.utils.torch_transform import angle_axis_to_quaternion
from pico_server.sonic_tools.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up


TWIST2_MUJOCO_JOINT_ORDER = [
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

SONIC_ISAACLAB_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

TWIST2_TO_SONIC_JOINT_INDICES = [
    TWIST2_MUJOCO_JOINT_ORDER.index(name) for name in SONIC_ISAACLAB_JOINT_ORDER
]

# Match pico_server_pose_only live teleop cadence so SONIC sees reference motion
# histories and finite-difference joint velocities in the same temporal spacing.
SONIC_JOINT29_TARGET_FPS = 50

SONIC_JOINT29_FALLBACK_BODY_QUAT_WXYZ = np.array(
    [0.7071, 0.0, 0.0, 0.7071], dtype=np.float32
)
SONIC_JOINT29_BODY_QUAT_EMA_ALPHA = 0.35
SONIC_JOINT29_JOINT_VEL_EMA_ALPHA = 0.2
SONIC_JOINT29_JOINT_VEL_CLIP = 8.0

def _quat_lerp_normalized_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Linearly interpolate two wxyz quaternions and renormalize."""
    q0 = np.asarray(q0, dtype=np.float32).reshape(4)
    q1 = np.asarray(q1, dtype=np.float32).reshape(4)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    q = (1.0 - float(alpha)) * q0 + float(alpha) * q1
    norm = float(np.linalg.norm(q))
    if norm > 1e-6:
        return (q / norm).astype(np.float32)
    return q1.copy()


def _quat_heading_wxyz(quat: np.ndarray) -> np.ndarray:
    """Extract z-up heading-only quaternion in wxyz format."""
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm > 1e-6:
        quat = quat / norm
    w, x, y, z = quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half_yaw = 0.5 * float(yaw)
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float32)


def _quat_conjugate_wxyz_np(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4).copy()
    quat[1:] *= -1.0
    return quat


def start_interpolation(state_machine, start_obs, end_obs, duration=1.0):
    """Start interpolation from start_obs to end_obs over given duration"""
    state_machine.is_interpolating = True
    state_machine.interpolation_start_time = time.time()
    state_machine.interpolation_duration = duration
    state_machine.interpolation_start_obs = start_obs.copy() if start_obs is not None else None
    state_machine.interpolation_target_obs = end_obs.copy() if end_obs is not None else None


def get_interpolated_obs(state_machine):
    """Get current interpolated observation, returns None if interpolation complete"""
    if (not state_machine.is_interpolating or
        state_machine.interpolation_start_obs is None or
        state_machine.interpolation_target_obs is None or
        state_machine.interpolation_start_time is None):
        return None
    elapsed_time = time.time() - state_machine.interpolation_start_time
    progress = min(elapsed_time / state_machine.interpolation_duration, 1.0)

    # Linear interpolation
    interp_obs = state_machine.interpolation_start_obs + (state_machine.interpolation_target_obs - state_machine.interpolation_start_obs) * progress

    # Check if interpolation is complete
    if progress >= 1.0:
        state_machine.is_interpolating = False
        return state_machine.interpolation_target_obs

    return interp_obs

def extract_mimic_obs_whole_body(qpos, last_qpos, dt=1/30):
    """Extract whole body mimic observations from robot joint positions (35 dims)"""
    root_pos, last_root_pos = qpos[0:3], last_qpos[0:3]
    root_quat, last_root_quat = qpos[3:7], last_qpos[3:7]
    robot_joints = qpos[7:].copy()  # Make a copy to avoid modifying original
    base_vel = (root_pos - last_root_pos) / dt
    base_ang_vel = quat_diff_np(last_root_quat, root_quat, scalar_first=True) / dt
    roll, pitch, yaw = euler_from_quaternion_np(root_quat.reshape(1, -1), scalar_first=True)
    # convert root vel to local frame
    base_vel_local = quat_rotate_inverse_np(root_quat, base_vel, scalar_first=True)
    base_ang_vel_local = quat_rotate_inverse_np(root_quat, base_ang_vel, scalar_first=True)

    # Standard mimic observation (35 dims)
    height = root_pos[2:3]
    # print("height: ", height)
    mimic_obs = np.concatenate([
        base_vel_local[:2],  # xy velocity (2 dims)
        root_pos[2:3],       # z position (1 dim)
        roll, pitch,         # roll, pitch (2 dims)
        base_ang_vel_local[2:3],  # yaw angular velocity (1 dim)
        robot_joints         # joint positions (29 dims)
    ])

    return mimic_obs



class StateMachine:
    def __init__(self, enable_smooth=False, smooth_window_size=5, use_pinch=False):
        """
        State process for teleoperation:
        idle -> teleop -> pause -> teleop ... -> idle -> exit
        """
        self.state = "idle"
        self.previous_state = "idle"
        self.right_key_one_was_pressed = False
        self.left_key_one_was_pressed = False
        self.left_key_two_was_pressed = False
        self.left_axis_click_was_pressed = False

        # Recording control state - Start recording immediately on initialization
        self.recording_active = True  # Changed: Start recording immediately
        self.recording_command = "start"  # Changed: Set initial command to start
        self.recording_command_frame_count = 0  # Counter to keep command visible for a few frames
        print("🔴 Recording will start immediately on initialization")
        # Interpolation state
        self.is_interpolating = False
        self.interpolation_start_time = None
        self.interpolation_duration = 2.0  # seconds
        self.interpolation_start_obs = None
        self.interpolation_target_obs = None
        self.current_mimic_obs = None
        self.last_mimic_obs = None
        self.current_neck_data = None
        self.last_neck_data = None

        # Hand state - interpolation values (0.0 = open, 1.0 = closed)
        self.hand_left_position = 0.0  # 0.0 = fully open, 1.0 = fully closed
        self.hand_right_position = 0.0
        self.use_pinch = use_pinch
        # Hand control parameters
        self.hand_movement_step = HAND_MOVEMENT_STEP

        # Velocity commands from joystick
        self.velocity_commands = np.array([0.0, 0.0, 0.0])  # [vx, vy, vyaw]

        # Smooth filtering
        self.enable_smooth = enable_smooth
        self.smooth_window_size = smooth_window_size
        self.smooth_history = []  # Store recent observations for sliding window

    def reset_for_ready_epoch(self):
        """Reset teleop-side state when Isaac starts a new clean episode."""
        self.state = "idle"
        self.previous_state = "idle"
        self.right_key_one_was_pressed = False
        self.left_key_one_was_pressed = False
        self.left_key_two_was_pressed = False
        self.left_axis_click_was_pressed = False
        self.is_interpolating = False
        self.interpolation_start_time = None
        self.interpolation_start_obs = None
        self.interpolation_target_obs = None
        self.current_mimic_obs = None
        self.last_mimic_obs = None
        self.current_neck_data = None
        self.last_neck_data = None
        self.recording_active = True
        self.recording_command = "start"
        self.recording_command_frame_count = 0
        self.hand_left_position = 0.0
        self.hand_right_position = 0.0
        self.velocity_commands[:] = 0.0
        self.smooth_history.clear()

    def update(self, controller_data):
        """Update state machine with controller data"""
        # Store previous state
        self.previous_state = self.state

        # Get current button states
        right_key_current = controller_data.get('RightController', {}).get('key_one', False)
        left_key_current = controller_data.get('LeftController', {}).get('key_one', False)
        left_key_two_current = controller_data.get('LeftController', {}).get('key_two', False)

        # Hand control - index_trig for close, grip for open
        right_index_trig_current = controller_data.get('RightController', {}).get('index_trig', False)
        left_index_trig_current = controller_data.get('LeftController', {}).get('index_trig', False)
        right_grip_current = controller_data.get('RightController', {}).get('grip', False)
        left_grip_current = controller_data.get('LeftController', {}).get('grip', False)

        # Emergency stop - left controller axis_click
        left_axis_click_current = controller_data.get('LeftController', {}).get('axis_click', False)

        # Detect button presses
        right_key_just_pressed = right_key_current and not self.right_key_one_was_pressed
        left_key_just_pressed = left_key_current and not self.left_key_one_was_pressed
        left_key_two_just_pressed = left_key_two_current and not self.left_key_two_was_pressed
        left_axis_click_just_pressed = left_axis_click_current and not self.left_axis_click_was_pressed

        # Debug: print button states
        if left_key_current or right_key_current:
            print(f"[DEBUG] Left key: {left_key_current}, Right key: {right_key_current}")
            print(f"[DEBUG] Left just pressed: {left_key_just_pressed}, Right just pressed: {right_key_just_pressed}")

        # Handle left axis click - recording cancel (when recording) or emergency stop (when not recording)
        if left_axis_click_just_pressed:
            if self.recording_active:
                # Cancel recording without saving
                self.recording_active = False
                self.recording_command = "cancel"
                self.recording_command_frame_count = 0
                print("❌ Recording cancelled (not saved)")
            else:
                # Emergency stop
                self._emergency_stop()

        # Handle left key press - Save recording and trigger complete reset
        if left_key_just_pressed:
            print("💾 Left key pressed - Save recording and trigger complete reset...")
            # Set recording command to save_and_reset
            self.recording_active = False
            self.recording_command = "save_and_reset"
            self.recording_command_frame_count = 0
            print("💾 Recording will be saved, then environment will be completely reset")

        # Handle left key_two press - Discard recording and trigger complete reset
        if left_key_two_just_pressed:
            print("❌ Left key_two pressed - Discard recording and trigger complete reset...")
            # Set recording command to discard_and_reset
            self.recording_active = False
            self.recording_command = "discard_and_reset"
            self.recording_command_frame_count = 0
            print("❌ Recording will be discarded, then environment will be completely reset")

        # Handle right key press - cycle between idle, teleop, pause
        if right_key_just_pressed:
            if self.state == "idle":
                self.state = "teleop"
            elif self.state == "teleop":
                self.state = "pause"
            elif self.state == "pause":
                self.state = "teleop"

        # Handle hand control - continuous interpolation
        # Right hand control
        if right_index_trig_current:  # Close right hand
            new_position = min(1.0, self.hand_right_position + self.hand_movement_step)
            if new_position != self.hand_right_position:
                self.hand_right_position = new_position
                print(f"Right hand closing: {self.hand_right_position:.1f}")
        elif right_grip_current:  # Open right hand
            new_position = max(0.0, self.hand_right_position - self.hand_movement_step)
            if new_position != self.hand_right_position:
                self.hand_right_position = new_position
                print(f"Right hand opening: {self.hand_right_position:.1f}")

        # Left hand control
        if left_index_trig_current:  # Close left hand
            new_position = min(1.0, self.hand_left_position + self.hand_movement_step)
            if new_position != self.hand_left_position:
                self.hand_left_position = new_position
                print(f"Left hand closing: {self.hand_left_position:.1f}")
        elif left_grip_current:  # Open left hand
            new_position = max(0.0, self.hand_left_position - self.hand_movement_step)
            if new_position != self.hand_left_position:
                self.hand_left_position = new_position
                print(f"Left hand opening: {self.hand_left_position:.1f}")

        # Extract velocity commands from controller axes
        self._update_velocity_commands(controller_data)

        # Update button state tracking
        self.right_key_one_was_pressed = right_key_current
        self.left_key_one_was_pressed = left_key_current
        self.left_key_two_was_pressed = left_key_two_current
        self.left_axis_click_was_pressed = left_axis_click_current

    def _update_velocity_commands(self, controller_data):
        """Update velocity commands from controller axes"""
        left_axis = controller_data.get('LeftController', {}).get('axis', [0.0, 0.0])
        right_axis = controller_data.get('RightController', {}).get('axis', [0.0, 0.0])

        # Use left stick for xy movement, right stick for yaw rotation
        if len(left_axis) >= 2 and len(right_axis) >= 2:
            # Scale factors for velocity commands
            xy_scale = 2.0  # m/s
            yaw_scale = 3.0  # rad/s

            self.velocity_commands[0] = left_axis[1] * xy_scale   # forward/backward (y axis inverted)
            self.velocity_commands[1] = -left_axis[0] * xy_scale  # left/right (x axis inverted)
            self.velocity_commands[2] = -right_axis[0] * yaw_scale  # yaw rotation (x axis inverted)

    def has_state_changed(self):
        """Check if state has changed since last update"""
        return self.state != self.previous_state


    def set_current_mimic_obs(self, mimic_obs):
        """Update current mimic obs"""
        self.current_mimic_obs = mimic_obs.copy() if mimic_obs is not None else None

    def set_last_mimic_obs(self, mimic_obs):
        """Update last mimic obs (used when entering pause)"""
        self.last_mimic_obs = mimic_obs.copy() if mimic_obs is not None else None

    def set_last_neck_data(self, neck_data):
        """Update last neck data (used when entering pause)"""
        self.last_neck_data = neck_data[:] if neck_data is not None else None

    def set_current_neck_data(self, neck_data):
        """Update current neck data"""
        self.current_neck_data = neck_data[:] if neck_data is not None else None

    def get_current_state(self):
        return self.state


    def get_velocity_commands(self):
        return self.velocity_commands.copy()

    def is_teleop_active(self):
        """Return True if currently in teleop state"""
        return self.state == "teleop"

    def should_exit(self):
        """Return True if should exit the program"""
        return self.state == "exit"

    def should_process_data(self):
        """Return True if should process motion data"""
        return self.state == "teleop" and not self.is_interpolating

    def get_hand_state(self):
        return self.hand_left_position, self.hand_right_position

    def get_hand_pose(self, robot_name):
        """Get interpolated hand poses based on current hand positions"""
        use_pinch = self.use_pinch
        # Get open and closed poses

        if not use_pinch:
            left_open = DEFAULT_HAND_POSE[robot_name]['left']['open']
            left_closed = DEFAULT_HAND_POSE[robot_name]['left']['close']
            right_open = DEFAULT_HAND_POSE[robot_name]['right']['open']
            right_closed = DEFAULT_HAND_POSE[robot_name]['right']['close']
        else:
            left_fully_open = DEFAULT_HAND_POSE[robot_name]['left']['open_pinch']
            left_fully_closed = DEFAULT_HAND_POSE[robot_name]['left']['close_pinch']
            right_fully_open = DEFAULT_HAND_POSE[robot_name]['right']['open_pinch']
            right_fully_closed = DEFAULT_HAND_POSE[robot_name]['right']['close_pinch']

            # compute the intermediate poses to shortern the distance betwen open and close
            # ratio * open + (1 - ratio) * closed
            ratio_open = 0.8
            ratio_closed = 0.0
            left_open =  left_fully_open * ratio_open + (1 - ratio_open) * left_fully_closed
            left_closed = left_fully_open * ratio_closed + (1 - ratio_closed) * left_fully_closed
            right_open = right_fully_open * ratio_open + (1 - ratio_open) * right_fully_closed
            right_closed = right_fully_open * ratio_closed + (1 - ratio_closed) * right_fully_closed

        # Interpolate between open and closed poses
        left_pose = left_open + (left_closed - left_open) * self.hand_left_position
        right_pose = right_open + (right_closed - right_open) * self.hand_right_position

        return left_pose, right_pose

    def apply_smooth(self, mimic_obs):
        """Apply sliding window smoothing to mimic observations"""
        if not self.enable_smooth or mimic_obs is None:
            return mimic_obs

        # Convert to numpy array if needed
        obs_array = np.array(mimic_obs) if not isinstance(mimic_obs, np.ndarray) else mimic_obs.copy()

        # Add current observation to history
        self.smooth_history.append(obs_array)

        # Keep only the recent window_size observations
        if len(self.smooth_history) > self.smooth_window_size:
            self.smooth_history.pop(0)

        # Apply sliding window average
        if len(self.smooth_history) >= 2:  # Need at least 2 observations for smoothing
            # Stack all observations in history
            history_stack = np.stack(self.smooth_history, axis=0)  # Shape: (history_len, obs_dim)
            # Compute mean across the time dimension
            smoothed_obs = np.mean(history_stack, axis=0)
            return smoothed_obs
        else:
            # Not enough history, return original observation
            return obs_array

    def reset_smooth_history(self):
        """Reset smooth history (call when transitioning states)"""
        self.smooth_history = []

    def _emergency_stop(self):
        """Emergency stop: kill sim2real.sh process (server_low_level_g1_real_future.py)"""
        try:
            print("[EMERGENCY STOP] Killing sim2real.sh process...")
            # Kill sim2real.sh which contains server_low_level_g1_real_future.py
            result = subprocess.run(['pkill', '-f', 'sim2real.sh'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("[EMERGENCY STOP] Successfully killed sim2real.sh process")
            else:
                print(f"[EMERGENCY STOP] pkill returned code {result.returncode}")

            # Also try to kill the specific server script directly as backup
            result2 = subprocess.run(['pkill', '-f', 'server_low_level_g1_real_future.py'],
                                   capture_output=True, text=True, timeout=5)
            if result2.returncode == 0:
                print("[EMERGENCY STOP] Successfully killed server_low_level_g1_real_future.py process")
            else:
                print(f"[EMERGENCY STOP] pkill for server script returned code {result2.returncode}")

        except subprocess.TimeoutExpired:
            print("[EMERGENCY STOP] pkill command timed out")
        except Exception as e:
            print(f"[EMERGENCY STOP] Error executing pkill: {e}")

    def _trigger_isaac_reset(self):
        """Trigger Isaac Lab reset via Redis (simpler than shared memory)"""
        print("[DEBUG] _trigger_isaac_reset() called")
        try:
            import redis
            import json
            import time

            print("[DEBUG] Connecting to Redis...")
            redis_client = redis.Redis(host='localhost', port=6379, db=0)

            # Test connection
            redis_client.ping()
            print("[DEBUG] Redis connection OK")

            # Send reset command via Redis
            # Isaac Lab will read this and trigger reset
            reset_command = {
                "reset_category": "2",  # Reset all (robot + objects)
                "timestamp": int(time.time() * 1000)
            }

            json_str = json.dumps(reset_command)
            print(f"[DEBUG] Writing to Redis: {json_str}")

            # Use a simple Redis key that Isaac Lab can check
            result = redis_client.set("isaac_reset_trigger", json_str)
            print(f"[DEBUG] Redis SET result: {result}")

            # Verify it was written
            verify = redis_client.get("isaac_reset_trigger")
            print(f"[DEBUG] Redis GET verify: {verify}")

            redis_client.expire("isaac_reset_trigger", 5)  # Auto-expire after 5 seconds

            print("✅ [RESET] Isaac Lab reset command sent via Redis")

            # Also reset state machine to idle
            self.state = "idle"
            self.velocity_commands = np.array([0.0, 0.0, 0.0])
            self.hand_left_position = 0.0
            self.hand_right_position = 0.0

        except redis.ConnectionError as e:
            print(f"❌ [RESET] Redis connection failed: {e}")
        except Exception as e:
            print(f"❌ [RESET] Failed to send reset command: {e}")
            import traceback
            traceback.print_exc()

class XRobotTeleopToRobot:
    def __init__(self, args):
        self.args = args
        self.robot_name = args.robot
        self.xml_file = ROBOT_XML_DICT[args.robot]
        self.robot_base = ROBOT_BASE_DICT[args.robot]
        self.target_backend = str(getattr(args, "target_backend", "twist2"))

        print(f"Pinch mode: {self.args.pinch_mode}")
        # Initialize state tracking
        self.last_qpos = None
        self.last_time = time.time()
        requested_target_fps = int(args.target_fps)
        if self.target_backend == "sonic_joint29" and requested_target_fps != SONIC_JOINT29_TARGET_FPS:
            print(
                f"[sonic_joint29] overriding target_fps {requested_target_fps} -> "
                f"{SONIC_JOINT29_TARGET_FPS} to match SONIC live teleop cadence"
            )
            requested_target_fps = SONIC_JOINT29_TARGET_FPS
        self.target_fps = requested_target_fps
        self.measured_dt = 1 / self.target_fps  # default fallback dt

        # Initialize components
        self.teleop_data_streamer = None
        self.redis_client = None
        self._input_ready_key = get_input_ready_key(self.target_backend)
        self._input_ready_epoch_id = -1
        self._ready_for_live_stream = False
        self._missing_ready_key_logged = False
        self._recording_control_sequence = 0
        self._last_recording_command_sent = "none"
        self._last_sent_gmr_qpos = None
        self._last_sent_gmr_time = None
        self._last_sent_body_quat_w = None
        self._last_sent_joint_vel_mujoco = None
        self._sonic_joint29_prev_source_ts_ns = None
        self._sonic_joint29_prev_qpos = None
        self._sonic_joint29_prev_body_quat_w = None
        self._sonic_joint29_next_target_ns = None
        self._default_full_qpos = None
        self.retarget = None
        self.model = None
        self.data = None
        self.state_machine = StateMachine(
            enable_smooth=args.smooth,
            smooth_window_size=args.smooth_window_size,
            use_pinch=args.pinch_mode
        )
        self.rate = None

        # Video recording
        self.video_writer = None
        self.renderer = None

        # FPS monitoring
        self.fps_monitor = FPSMonitor(
            enable_detailed_stats=args.measure_fps,
            quick_print_interval=100,
            detailed_print_interval=1000,
            expected_fps=self.target_fps,
            name="Teleop Loop"
        )
        print(f"Teleop target backend: {self.target_backend}")


    def setup_teleop_data_streamer(self):
        """Initialize and start the teleop data streamer"""
        self.teleop_data_streamer = XRobotStreamer()
        print("Teleop data streamer initialized")

    def setup_redis_connection(self):
        """Setup Redis connection"""
        redis_ip = self.args.redis_ip
        self.redis_client = redis.Redis(host=redis_ip, port=6379, db=0)
        self.redis_pipeline = self.redis_client.pipeline()
        self.redis_client.ping()
        print("Redis connected successfully")

    def setup_retargeting_system(self):
        """Initialize the motion retargeting system"""
        self.retarget = GMR(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            actual_human_height=self.args.actual_human_height,
        )
        print("Retargeting system initialized")

    def setup_mujoco_simulation(self):
        """Setup MuJoCo model and data"""
        self.model = mj.MjModel.from_xml_path(str(self.xml_file))
        self.data = mj.MjData(self.model)
        mj.mj_forward(self.model, self.data)
        self._default_full_qpos = self.data.qpos.copy()
        print("MuJoCo simulation initialized")

    def setup_video_recording(self):
        """Setup video recording if requested"""
        if not self.args.record_video:
            return

        self.video_writer = cv2.VideoWriter(
            'output.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,
            (640, 480)
        )
        width, height = 640, 480
        self.renderer = mj.Renderer(self.model, height=height, width=width)
        print("Video recording setup completed")

    def setup_rate_limiter(self):
        """Setup rate limiter for consistent FPS"""
        self.rate = RateLimiter(frequency=self.target_fps, warn=False)
        print(f"Rate limiter setup for {self.target_fps} FPS")

    def get_teleop_data(self):
        """Get current teleop data from streamer"""
        if self.teleop_data_streamer is not None:
            return self.teleop_data_streamer.get_current_frame()
        return None, None, None, None, None

    def _refresh_input_ready_epoch(self) -> bool:
        if self.redis_client is None:
            return True
        try:
            raw_guard = self.redis_client.get(self._input_ready_key)
        except Exception as exc:
            print(f"[TWIST2_PICO] Failed to read input ready key: {exc}")
            if not self._ready_for_live_stream:
                self._ready_for_live_stream = True
            return self._ready_for_live_stream
        if raw_guard is None:
            if not self._ready_for_live_stream:
                self._ready_for_live_stream = True
            if not self._missing_ready_key_logged:
                print(
                    f"[TWIST2_PICO] Input ready key not found yet: {self._input_ready_key}. "
                    f"Continuing local preview and Redis publishing without guard."
                )
                self._missing_ready_key_logged = True
            return self._ready_for_live_stream
        try:
            if isinstance(raw_guard, (bytes, bytearray)):
                raw_guard = raw_guard.decode("utf-8")
            payload = json.loads(raw_guard)
            epoch_id = int(payload.get("epoch_id", -1))
        except Exception:
            return self._ready_for_live_stream
        if epoch_id != self._input_ready_epoch_id:
            self._input_ready_epoch_id = epoch_id
            self._ready_for_live_stream = True
            self._missing_ready_key_logged = False
            self.state_machine.reset_for_ready_epoch()
            self._last_sent_gmr_qpos = None
            self._last_sent_body_quat_w = None
            self._last_sent_gmr_time = None
            self._last_sent_joint_vel_mujoco = None
            self._sonic_joint29_prev_source_ts_ns = None
            self._sonic_joint29_prev_qpos = None
            self._sonic_joint29_prev_body_quat_w = None
            self._sonic_joint29_next_target_ns = None
            self._last_recording_command_sent = "none"
            print(f"[TWIST2_PICO] Input ready epoch updated: {epoch_id}")
            return False
        return self._ready_for_live_stream

    def process_retargeting(self, smplx_data):
        """Process motion retargeting and return observations"""
        if smplx_data is None or self.retarget is None:
            return None, None

        # Measure dt between retarget calls
        current_time = time.time()
        self.measured_dt = current_time - self.last_time
        self.last_time = current_time

        # Retarget till convergence
        qpos = self.retarget.retarget(smplx_data, offset_to_ground=True)

        # Create mimic obs from retargeting
        if self.last_qpos is not None:
            current_retarget_obs = extract_mimic_obs_whole_body(qpos, self.last_qpos, dt=self.measured_dt)
        else:
            current_retarget_obs = DEFAULT_MIMIC_OBS[self.robot_name]

        self.last_qpos = qpos.copy()
        return qpos, current_retarget_obs

    def update_visualization(self, qpos, smplx_data, viewer):
        """Update MuJoCo visualization"""
        if qpos is None:
            return

        # Clean custom geometry
        if hasattr(viewer, 'user_scn') and viewer.user_scn is not None:
            viewer.user_scn.ngeom = 0

        # Draw the task targets for reference
        if smplx_data is not None and self.retarget is not None:
            for robot_link, ik_data in self.retarget.ik_match_table1.items():
                body_name = ik_data[0]
                if body_name not in smplx_data:
                    continue
                draw_frame(
                    self.retarget.scaled_human_data[body_name][0] - self.retarget.ground,
                    R.from_quat(smplx_data[body_name][1]).as_matrix(),
                    viewer,
                    0.1,
                    orientation_correction=R.from_quat(ik_data[-1]),
                )

        # Update the simulation
        if qpos is not None:
            self.data.qpos[:] = qpos.copy()
            mj.mj_forward(self.model, self.data)

            # Camera follow the pelvis
            self._update_camera_position(viewer)

    def _update_camera_position(self, viewer):
        """Update camera to follow the robot"""
        FOLLOW_CAMERA = True
        if FOLLOW_CAMERA:
            robot_base_pos = self.data.xpos[self.model.body(self.robot_base).id]
            viewer.cam.lookat = robot_base_pos
            viewer.cam.distance = 3.0

    def handle_state_transitions(self, current_retarget_obs):
        """Handle state machine transitions and interpolations"""
        if not self.state_machine.has_state_changed():
            return

        current_state = self.state_machine.get_current_state()
        previous_state = self.state_machine.previous_state

        print(f"State changed: {previous_state} -> {current_state}")

        if current_state == "teleop":
            self._handle_enter_teleop(previous_state, current_retarget_obs)
        elif current_state == "pause":
            self._handle_enter_pause()

    def _handle_enter_teleop(self, previous_state, current_retarget_obs):
        """Handle entering teleop state"""
        if previous_state in ["idle", "pause"]:
            self.state_machine.reset_smooth_history()
            print("Reset smooth history on entering teleop")

        if previous_state == "idle":
            if current_retarget_obs is not None:
                default_obs = DEFAULT_MIMIC_OBS[self.robot_name]
                start_interpolation(self.state_machine, default_obs, current_retarget_obs[:35])
                print("Interpolating from default to teleop...")
        elif previous_state == "pause":
            if (current_retarget_obs is not None and
                self.state_machine.last_mimic_obs is not None):
                last_obs_35d = self.state_machine.last_mimic_obs[:35] if len(self.state_machine.last_mimic_obs) > 35 else self.state_machine.last_mimic_obs
                start_interpolation(self.state_machine, last_obs_35d, current_retarget_obs[:35])
                print("Interpolating from pause to teleop...")
    def _handle_enter_pause(self):
        """Handle entering pause state"""
        if self.state_machine.current_mimic_obs is not None:
            self.state_machine.set_last_mimic_obs(self.state_machine.current_mimic_obs)
            print("Entered pause mode, storing last obs")
        if self.state_machine.current_neck_data is not None:
            self.state_machine.set_last_neck_data(self.state_machine.current_neck_data)
            print("Entered pause mode, storing last neck data")

    def determine_mimic_obs_to_send(self, current_retarget_obs):
        """Determine which mimic observation to send based on current state"""
        current_state = self.state_machine.get_current_state()

        if current_state == "idle":
            obs = DEFAULT_MIMIC_OBS[self.robot_name]
        elif current_state == "pause":
            if self.state_machine.last_mimic_obs is not None:
                obs = self.state_machine.last_mimic_obs[:35] if len(self.state_machine.last_mimic_obs) > 35 else self.state_machine.last_mimic_obs
            else:
                obs = DEFAULT_MIMIC_OBS[self.robot_name]
        elif current_state == "teleop":
            obs = self._get_teleop_mimic_obs(current_retarget_obs)
            obs = self.state_machine.apply_smooth(obs)
        else:
            obs = DEFAULT_MIMIC_OBS[self.robot_name]

        return obs

    def _get_teleop_mimic_obs(self, current_retarget_obs):
        """Get mimic obs for teleop state, handling interpolation"""
        if self.state_machine.is_interpolating:
            interp_obs = get_interpolated_obs(self.state_machine)
            if interp_obs is not None:
                self.state_machine.set_current_mimic_obs(interp_obs)
                return interp_obs
            return DEFAULT_MIMIC_OBS[self.robot_name]

        if current_retarget_obs is not None:
            obs_35d = current_retarget_obs[:35] if len(current_retarget_obs) > 35 else current_retarget_obs
            self.state_machine.set_current_mimic_obs(obs_35d)
            return obs_35d

        return DEFAULT_MIMIC_OBS[self.robot_name]

    def determine_neck_data_to_send(self, smplx_data):
        """Determine which neck data to send based on current state"""

        current_state = self.state_machine.get_current_state()

        # In non-teleop states, send default neck position [0, 0]
        if current_state in ["idle"]:
            return [0.0, 0.0]

        if current_state == "pause":
            # return [0.0, 0.0]
            # use last neck data
            if self.state_machine.last_neck_data is not None:
                return self.state_machine.last_neck_data
            else:
                return [0.0, 0.0]

        # In teleop state, extract neck data from smplx_data
        elif current_state == "teleop" and smplx_data is not None:
            scale = self.args.neck_retarget_scale
            neck_yaw, neck_pitch = human_head_to_robot_neck(smplx_data)
            return [neck_yaw * scale, neck_pitch * scale]

        # Default fallback
        return [0.0, 0.0]

    def _convert_smplx_to_serializable(self, smplx_data):
        """Convert SMPLX data with numpy arrays to JSON-serializable format"""
        if smplx_data is None:
            return None

        serializable_data = {}
        for key, value in smplx_data.items():
            if isinstance(value, (list, tuple)):
                # Convert each element in the list/tuple
                serializable_value = []
                for item in value:
                    if isinstance(item, np.ndarray):
                        serializable_value.append(item.tolist())
                    else:
                        serializable_value.append(item)
                serializable_data[key] = serializable_value
            elif isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            else:
                serializable_data[key] = value

        return serializable_data

    def _select_gmr_qpos_to_send(self, current_qpos):
        """Select the full-body GMR qpos to publish for the current teleop state."""
        current_state = self.state_machine.get_current_state()
        if current_state == "teleop" and current_qpos is not None:
            qpos_to_send = np.asarray(current_qpos, dtype=np.float32)
        elif current_state == "teleop" and self._last_sent_gmr_qpos is not None:
            qpos_to_send = self._last_sent_gmr_qpos.copy()
        elif current_state == "pause" and self._last_sent_gmr_qpos is not None:
            qpos_to_send = self._last_sent_gmr_qpos.copy()
        elif self._default_full_qpos is not None:
            qpos_to_send = np.asarray(self._default_full_qpos, dtype=np.float32)
        else:
            qpos_to_send = None
        return qpos_to_send

    def _compute_sonic_joint29_body_quat_w(self, smplx_data=None, qpos=None):
        if qpos is not None:
            try:
                qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
                if qpos.size >= 7:
                    root_quat_wxyz = qpos[3:7].astype(np.float32, copy=False)
                    norm = float(np.linalg.norm(root_quat_wxyz))
                    if norm > 1e-6:
                        return (root_quat_wxyz / norm).astype(np.float32, copy=False)
            except Exception:
                pass

        if smplx_data is None:
            if self._last_sent_body_quat_w is not None:
                return self._last_sent_body_quat_w.copy()
            return SONIC_JOINT29_FALLBACK_BODY_QUAT_WXYZ.copy()

        try:
            pelvis = smplx_data.get("Pelvis")
            if pelvis is None or len(pelvis) < 2:
                raise ValueError("Pelvis entry missing")
            processed_pelvis_quat_wxyz = np.asarray(pelvis[1], dtype=np.float32).reshape(4)
            unity_to_right_hand_rot = np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            )
            unity_to_right_hand_quat_wxyz = R.from_matrix(unity_to_right_hand_rot).as_quat(
                scalar_first=True
            ).astype(np.float32)
            raw_pelvis_quat_wxyz = quat_mul_np(
                _quat_conjugate_wxyz_np(unity_to_right_hand_quat_wxyz),
                processed_pelvis_quat_wxyz,
                scalar_first=True,
            ).astype(np.float32)
            root_rot = R.from_quat(raw_pelvis_quat_wxyz, scalar_first=True)
            root_rot = root_rot * R.from_euler("y", 180.0, degrees=True)
            global_orient_quat = torch.from_numpy(
                root_rot.as_quat(scalar_first=True).astype(np.float32)
            ).reshape(1, 4)
            global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
            global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)
            body_quat_wxyz = global_orient_quat.detach().cpu().numpy().reshape(4).astype(np.float32)
            norm = float(np.linalg.norm(body_quat_wxyz))
            if norm > 1e-6:
                body_quat_wxyz /= norm
            else:
                body_quat_wxyz = SONIC_JOINT29_FALLBACK_BODY_QUAT_WXYZ.copy()
            return body_quat_wxyz
        except Exception:
            if self._last_sent_body_quat_w is not None:
                return self._last_sent_body_quat_w.copy()
            return SONIC_JOINT29_FALLBACK_BODY_QUAT_WXYZ.copy()

    def _maybe_interpolate_sonic_joint29_reference(self, qpos, body_quat_w, source_timestamp_ns):
        """Resample the live GMR stream onto a steady 50 Hz publish grid."""
        if qpos is None or body_quat_w is None:
            return qpos, body_quat_w

        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        body_quat_w = np.asarray(body_quat_w, dtype=np.float32).reshape(4)
        try:
            source_ts_ns = int(source_timestamp_ns)
        except (TypeError, ValueError):
            source_ts_ns = 0

        if source_ts_ns <= 0:
            return qpos, body_quat_w

        step_ns = int(1e9 / max(1, self.target_fps))
        prev_ts_ns = self._sonic_joint29_prev_source_ts_ns
        prev_qpos = self._sonic_joint29_prev_qpos
        prev_body_quat_w = self._sonic_joint29_prev_body_quat_w

        if prev_ts_ns is None or prev_qpos is None or prev_body_quat_w is None:
            self._sonic_joint29_prev_source_ts_ns = source_ts_ns
            self._sonic_joint29_prev_qpos = qpos.copy()
            self._sonic_joint29_prev_body_quat_w = body_quat_w.copy()
            self._sonic_joint29_next_target_ns = source_ts_ns
            return qpos, body_quat_w

        if source_ts_ns <= prev_ts_ns:
            if self._last_sent_gmr_qpos is not None and self._last_sent_body_quat_w is not None:
                return self._last_sent_gmr_qpos.copy(), self._last_sent_body_quat_w.copy()
            return qpos, body_quat_w

        if self._sonic_joint29_next_target_ns is None:
            self._sonic_joint29_next_target_ns = prev_ts_ns + step_ns
        if self._sonic_joint29_next_target_ns < prev_ts_ns:
            self._sonic_joint29_next_target_ns = prev_ts_ns

        if self._sonic_joint29_next_target_ns > source_ts_ns:
            # Not enough source horizon yet. Use the freshest reference to minimize lag,
            # but keep the target grid unchanged so the next publish can catch up cleanly.
            self._sonic_joint29_prev_source_ts_ns = source_ts_ns
            self._sonic_joint29_prev_qpos = qpos.copy()
            self._sonic_joint29_prev_body_quat_w = body_quat_w.copy()
            return qpos, body_quat_w

        denom = float(source_ts_ns - prev_ts_ns)
        alpha = (
            float(self._sonic_joint29_next_target_ns - prev_ts_ns) / denom
            if denom > 0.0
            else 1.0
        )
        alpha = float(np.clip(alpha, 0.0, 1.0))
        use_qpos = ((1.0 - alpha) * prev_qpos + alpha * qpos).astype(np.float32)
        use_body_quat_w = _quat_lerp_normalized_wxyz(prev_body_quat_w, body_quat_w, alpha)

        self._sonic_joint29_next_target_ns += step_ns
        self._sonic_joint29_prev_source_ts_ns = source_ts_ns
        self._sonic_joint29_prev_qpos = qpos.copy()
        self._sonic_joint29_prev_body_quat_w = body_quat_w.copy()
        return use_qpos, use_body_quat_w

    def _build_gmr_joint29_payload(
        self, qpos, publish_dt=None, smplx_data=None, body_quat_w_override=None
    ):
        """Convert full MuJoCo qpos into the Redis payload expected by sonic_joint29."""
        if qpos is None:
            return None
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if qpos.size < 36:
            return None

        joint_pos_mujoco = qpos[7:36]
        if joint_pos_mujoco.shape[0] != 29:
            return None

        joint_pos_sonic = joint_pos_mujoco[TWIST2_TO_SONIC_JOINT_INDICES]
        if self._last_sent_gmr_qpos is None:
            joint_vel_mujoco = np.zeros_like(joint_pos_mujoco)
        else:
            dt = max(
                float(publish_dt) if publish_dt is not None else float(self.measured_dt),
                1e-3,
            )
            last_joint_pos_mujoco = np.asarray(self._last_sent_gmr_qpos, dtype=np.float32).reshape(-1)[7:36]
            joint_vel_mujoco = (joint_pos_mujoco - last_joint_pos_mujoco) / dt
            if self._last_sent_joint_vel_mujoco is not None:
                last_joint_vel = np.asarray(self._last_sent_joint_vel_mujoco, dtype=np.float32).reshape(29)
                alpha = float(np.clip(SONIC_JOINT29_JOINT_VEL_EMA_ALPHA, 0.0, 1.0))
                joint_vel_mujoco = alpha * joint_vel_mujoco + (1.0 - alpha) * last_joint_vel
        joint_vel_mujoco = np.clip(
            joint_vel_mujoco,
            -SONIC_JOINT29_JOINT_VEL_CLIP,
            SONIC_JOINT29_JOINT_VEL_CLIP,
        ).astype(np.float32, copy=False)
        joint_vel_sonic = joint_vel_mujoco[TWIST2_TO_SONIC_JOINT_INDICES]
        if body_quat_w_override is not None:
            body_quat_wxyz = np.asarray(body_quat_w_override, dtype=np.float32).reshape(4)
        else:
            body_quat_wxyz = self._compute_sonic_joint29_body_quat_w(smplx_data=smplx_data, qpos=qpos)
        if self._last_sent_body_quat_w is not None:
            body_quat_wxyz = _quat_lerp_normalized_wxyz(
                self._last_sent_body_quat_w,
                body_quat_wxyz,
                SONIC_JOINT29_BODY_QUAT_EMA_ALPHA,
            )

        return {
            "full_qpos": qpos.copy(),
            "joint_pos": joint_pos_sonic.astype(np.float32, copy=False),
            "joint_vel": joint_vel_sonic.astype(np.float32, copy=False),
            "joint_vel_mujoco": joint_vel_mujoco.copy(),
            "body_pos": qpos[0:3].astype(np.float32, copy=False),
            "body_quat_w": body_quat_wxyz,
        }

    def send_to_redis(
        self,
        mimic_obs,
        neck_data=None,
        smplx_data=None,
        recording_active=False,
        recording_command="none",
        qpos=None,
        source_timestamp_ns=None,
    ):
        """Send mimic observations to Redis"""

        if self.redis_client is not None and mimic_obs is not None:
            # Expect 35D mimic observations
            assert len(mimic_obs) == 35, f"Expected 35 mimic obs dims, got {len(mimic_obs)}"
            # Send to both keys for compatibility
            self.redis_pipeline.set("action_body_unitree_g1_with_hands", json.dumps(mimic_obs.tolist()))

        # Send hand action to redis
        if self.redis_client is not None:
            hand_left_pose, hand_right_pose = self.state_machine.get_hand_pose(self.robot_name)
            self.redis_pipeline.set("action_hand_left_unitree_g1_with_hands", json.dumps(hand_left_pose.tolist()))
            self.redis_pipeline.set("action_hand_right_unitree_g1_with_hands", json.dumps(hand_right_pose.tolist()))

        # Send neck data to redis
        if neck_data is not None:
            self.redis_pipeline.set("action_neck_unitree_g1_with_hands", json.dumps(neck_data))

        # Send human SMPLX data (before GMR retargeting) to redis
        if smplx_data is not None:
            serializable_smplx = self._convert_smplx_to_serializable(smplx_data)
            self.redis_pipeline.set("human_smplx_data_unitree_g1_with_hands", json.dumps(serializable_smplx))

        # Send human height information to redis
        if hasattr(self.args, 'actual_human_height'):
            human_info = {
                'height': self.args.actual_human_height,
                'neck_retarget_scale': self.args.neck_retarget_scale
            }
            self.redis_pipeline.set("human_info_unitree_g1_with_hands", json.dumps(human_info))

        if self.target_backend == "sonic_joint29":
            gmr_qpos_to_send = self._select_gmr_qpos_to_send(qpos)
            body_quat_override = None
            if (
                self.state_machine.get_current_state() == "teleop"
                and qpos is not None
            ):
                current_body_quat_w = self._compute_sonic_joint29_body_quat_w(
                    smplx_data=smplx_data,
                    qpos=gmr_qpos_to_send,
                )
                gmr_qpos_to_send, body_quat_override = self._maybe_interpolate_sonic_joint29_reference(
                    gmr_qpos_to_send,
                    current_body_quat_w,
                    source_timestamp_ns,
                )
            now_monotonic = time.monotonic()
            publish_dt = None
            if self._last_sent_gmr_time is not None:
                publish_dt = now_monotonic - self._last_sent_gmr_time
            gmr_payload = self._build_gmr_joint29_payload(
                gmr_qpos_to_send,
                publish_dt=publish_dt,
                smplx_data=smplx_data,
                body_quat_w_override=body_quat_override,
            )
            if gmr_payload is not None:
                frame_index = self._send_counter if hasattr(self, "_send_counter") else 0
                self.redis_pipeline.set(GMR_FULL_QPOS_KEY, json.dumps(gmr_payload["full_qpos"].tolist()))
                self.redis_pipeline.set(GMR_JOINT_POS_KEY, json.dumps(gmr_payload["joint_pos"].tolist()))
                self.redis_pipeline.set(GMR_JOINT_VEL_KEY, json.dumps(gmr_payload["joint_vel"].tolist()))
                self.redis_pipeline.set(GMR_BODY_POS_KEY, json.dumps(gmr_payload["body_pos"].tolist()))
                self.redis_pipeline.set(GMR_BODY_QUAT_W_KEY, json.dumps(gmr_payload["body_quat_w"].tolist()))
                self.redis_pipeline.set(GMR_FRAME_INDEX_KEY, frame_index)
                self._last_sent_gmr_qpos = gmr_payload["full_qpos"].copy()
                self._last_sent_gmr_time = now_monotonic
                self._last_sent_joint_vel_mujoco = gmr_payload["joint_vel_mujoco"].copy()
                self._last_sent_body_quat_w = gmr_payload["body_quat_w"].copy()

        # Send recording control state to redis
        if recording_command != "none" and recording_command != self._last_recording_command_sent:
            self._recording_control_sequence += 1
        self._last_recording_command_sent = recording_command
        timestamp_ms = int(time.time() * 1000)
        recording_control_data = {
            "active": recording_active,
            "command": recording_command,
            "sequence": self._recording_control_sequence,
            "timestamp_ms": timestamp_ms,
            "source": "twist2_teleop_server",
        }
        self.redis_pipeline.set("recording_control_unitree_g1_with_hands", json.dumps(recording_control_data))

        # Debug: print recording state every 100 frames
        if not hasattr(self, '_send_counter'):
            self._send_counter = 0
        self._send_counter += 1
        if self._send_counter % 100 == 0:
            print(f"[SEND_TO_REDIS DEBUG] Recording state: active={recording_active}, command={recording_command}")

        # Send timestamp to redis
        t_action = timestamp_ms
        self.redis_pipeline.set("t_action", t_action)

        # execute the pipeline once
        self.redis_pipeline.execute()


    def send_controller_data_to_redis(self, controller_data):
        """Send controller data to Redis"""
        if self.redis_client is not None and controller_data is not None:
            controller_payload = dict(controller_data)
            left_controller = dict(controller_payload.get("LeftController", {}))
            right_controller = dict(controller_payload.get("RightController", {}))

            def _compute_grip_binary(side_data: dict, current_hand_position: float) -> bool:
                try:
                    index_trig = float(side_data.get("index_trig", 0.0))
                    grip = float(side_data.get("grip", 0.0))
                except Exception:
                    return bool(current_hand_position > 0.0)
                if index_trig > 0.5:
                    return True
                if grip > 0.5:
                    return False
                return bool(current_hand_position > 0.0)

            left_controller["close_trigger_binary"] = bool(float(left_controller.get("index_trig", 0.0)) > 0.5)
            left_controller["open_trigger_binary"] = bool(float(left_controller.get("grip", 0.0)) > 0.5)
            right_controller["close_trigger_binary"] = bool(float(right_controller.get("index_trig", 0.0)) > 0.5)
            right_controller["open_trigger_binary"] = bool(float(right_controller.get("grip", 0.0)) > 0.5)
            left_controller["grip_binary"] = _compute_grip_binary(
                left_controller, self.state_machine.hand_left_position
            )
            right_controller["grip_binary"] = _compute_grip_binary(
                right_controller, self.state_machine.hand_right_position
            )
            left_controller.pop("index_trig_binary", None)
            right_controller.pop("index_trig_binary", None)
            controller_payload["LeftController"] = left_controller
            controller_payload["RightController"] = right_controller
            self.redis_client.set(
                f"controller_data",
                json.dumps(controller_payload)
            )


    def record_video_frame(self, viewer):
        """Record current frame to video if recording is enabled"""
        if not self.args.record_video or self.renderer is None:
            return

        self.renderer.update_scene(self.data, camera=viewer.cam)
        pixels = self.renderer.render()

        # Convert from RGB to BGR (OpenCV uses BGR)
        frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)

    def handle_exit_sequence(self, viewer):
        """Handle graceful exit with interpolation to default pose"""
        if self.state_machine.current_mimic_obs is not None:
            default_obs = DEFAULT_MIMIC_OBS[self.robot_name]
            current_obs = self.state_machine.current_mimic_obs[:35] if len(self.state_machine.current_mimic_obs) > 35 else self.state_machine.current_mimic_obs
            start_interpolation(self.state_machine, current_obs, default_obs)
            print("Interpolating to default pose before exit...")

            # Wait for interpolation to complete
            while self.state_machine.is_interpolating:
                interp_obs = get_interpolated_obs(self.state_machine)
                if interp_obs is not None:
                    # During exit sequence, send default neck position [0, 0]
                    neck_data_to_send = self.determine_neck_data_to_send(None)
                    self.send_to_redis(interp_obs, neck_data_to_send, smplx_data=None, recording_active=False)
                viewer.sync()
                self.rate.sleep()



    def initialize_all_systems(self):
        """Initialize all required systems"""
        print("Initializing teleop systems...")
        self.setup_teleop_data_streamer()
        self.setup_redis_connection()
        self.setup_retargeting_system()
        self.setup_mujoco_simulation()
        self.setup_video_recording()
        self.setup_rate_limiter()

        print("Teleop state machine initialized. Controls:")
        print("- Right controller key_one: Cycle through idle -> teleop -> pause -> teleop...")
        print("- Left controller key_one: Exit program")
        print("- Left controller axis_click: Emergency stop - kills sim2real.sh process")
        print("- Left controller axis: Control root xy velocity")
        print("- Right controller axis: Control yaw velocity")
        print("- Publishes 35-dimensional mimic observations")
        print(f"Starting in state: {self.state_machine.get_current_state()}")

        if self.state_machine.enable_smooth:
            print(f"- Smooth filtering: ENABLED (window size: {self.state_machine.smooth_window_size} frames)")
        else:
            print("- Smooth filtering: DISABLED")

        if self.fps_monitor.enable_detailed_stats:
            print(f"- FPS measurement: ENABLED (detailed stats every {self.fps_monitor.detailed_print_interval} steps)")
        else:
            print(f"- FPS measurement: Quick stats only (every {self.fps_monitor.quick_print_interval} steps)")

        print("Ready to receive teleop data.")

    def run(self):
        """Main execution loop"""
        self.initialize_all_systems()

        # Start the viewer
        with mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False
        ) as viewer:
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1

            while viewer.is_running():
                # Get current teleop data
                smplx_data, left_hand_data, right_hand_data, controller_data, headset_data = self.get_teleop_data()
                ready_for_live_stream = self._refresh_input_ready_epoch()

                # Update state machine
                if ready_for_live_stream and controller_data is not None:
                    self.state_machine.update(controller_data)
                    self.send_controller_data_to_redis(controller_data)

                # Check if we should exit
                if self.state_machine.should_exit():
                    print("Exit requested via controller")
                    self.handle_exit_sequence(viewer)
                    break

                # Process retargeting if we have data
                qpos, current_retarget_obs = None, None
                if ready_for_live_stream and smplx_data is not None:
                    qpos, current_retarget_obs = self.process_retargeting(smplx_data)
                    self.update_visualization(qpos, smplx_data, viewer)

                # Handle state transitions
                if ready_for_live_stream:
                    self.handle_state_transitions(current_retarget_obs)

                # Determine and send mimic observations
                if ready_for_live_stream:
                    mimic_obs_to_send = self.determine_mimic_obs_to_send(current_retarget_obs)
                    neck_data_to_send = self.determine_neck_data_to_send(smplx_data)
                else:
                    mimic_obs_to_send = DEFAULT_MIMIC_OBS[self.robot_name]
                    neck_data_to_send = [0.0, 0.0]
                    smplx_data = None

                # Store current neck data in state machine for pause state handling
                if ready_for_live_stream and neck_data_to_send is not None:
                    self.state_machine.set_current_neck_data(neck_data_to_send)

                # Send data with recording state and command
                self.send_to_redis(
                    mimic_obs_to_send,
                    neck_data_to_send,
                    smplx_data,
                    self.state_machine.recording_active if ready_for_live_stream else False,
                    self.state_machine.recording_command if ready_for_live_stream else "none",
                    qpos=qpos if ready_for_live_stream else None,
                    source_timestamp_ns=(
                        controller_data.get("timestamp")
                        if ready_for_live_stream and controller_data is not None
                        else None
                    ),
                )

                # Manage recording command visibility
                # Keep "start", "save", "cancel", "save_and_reset", "discard_and_reset" commands visible for 30 frames (~1 second at 30Hz)
                if self.state_machine.recording_command in ["start", "save", "cancel", "save_and_reset", "discard_and_reset"]:
                    self.state_machine.recording_command_frame_count += 1
                    if self.state_machine.recording_command_frame_count >= 30:
                        self.state_machine.recording_command = "none"
                        self.state_machine.recording_command_frame_count = 0
                else:
                    self.state_machine.recording_command_frame_count = 0

                # Update visualization and record video
                viewer.sync()
                self.record_video_frame(viewer)

                # FPS monitoring
                self.fps_monitor.tick()

                self.rate.sleep()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands"],
        default="unitree_g1",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Whether to record the video.",
    )
    parser.add_argument(
        "--pinch_mode",
        action="store_true",
        help="Whether to use pinch mode for hand control.",
        default=False,
    )
    parser.add_argument(
        "--redis_ip",
        type=str,
        default="localhost",
        help="Redis IP",
    )
    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=1.5,
        help="Actual human height for retargeting.",
    )
    parser.add_argument(
        "--neck_retarget_scale",
        type=float,
        default=1.5,
        help="Scale factor for neck data.",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Enable smooth filtering for mimic observations in teleop mode.",
    )
    parser.add_argument(
        "--smooth_window_size",
        type=int,
        default=5,
        help="Window size for sliding window smoothing (default: 5 frames).",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=100,
        help="Target FPS for the teleop system.",
    )
    parser.add_argument(
        "--measure_fps",
        type=int,
        default=0,
        help="Measure and print detailed FPS statistics (0=disabled, 1=enabled).",
    )
    parser.add_argument(
        "--target_backend",
        choices=["twist2", "sonic_joint29"],
        default="twist2",
        help="Select which Isaac input-ready/backend route this teleop publisher should target.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    teleop_robot = XRobotTeleopToRobot(args)
    teleop_robot.run()
