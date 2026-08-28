#!/usr/bin/env python3
"""Replay one SIMPLE LeRobot episode and export an Arena/Kimodo-compatible set.

The default ``mujoco_isaac`` mode keeps MuJoCo as the fast physics/WBC engine
and drives Isaac Sim as the synchronized renderer.  This is important for
SIMPLE episodes: the camera image is rendered from the recorded HSSD room,
instead of MuJoCo's plain fallback floor/table.  Passing ``--sim-mode mujoco``
is still available as a lightweight diagnostic.

The source SIMPLE archive contains a 32-D observation and 36-D WBC command. The
exporter samples the *measured* MuJoCo state after each replay step, while the
action root is reconstructed from the recorded reference command, and writes
the 64-D/40-D Arena protocol:

    state = root_rot6d_relative_to_episode_heading + q29 + dq29
    action = reference_root_local_xy_delta + reference_root_z
             + reference_root_rot6d + target_q29 + hand_binary

The output is a normal LeRobot-like directory (one parquet file and one MP4
for this prototype) and can be extended to shard multiple episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Keep Isaac's auxiliary wheels (pyarrow/transforms3d/python-fcl) *after* the
# active environment's site-packages.  The data-disk bundle also contains a
# newer MuJoCo wheel; putting it first changes MjSpec.attach naming semantics
# and breaks SIMPLE's robot model.  Appending preserves the IsaacLab env's
# pinned MuJoCo while still making the missing pure/binary wheels available.
_EXTRA_PYTHON = os.environ.get("SIMPLE_ISAAC_EXTRA_PYTHON", "").strip()
if _EXTRA_PYTHON:
    sys.path.append(_EXTRA_PYTHON)

# Select the headless renderer before importing packages that may initialize
# MuJoCo's OpenGL backend.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import gymnasium as gym
import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]  # controlnet_v1.2
SIMPLE_ROOT = PROJECT_ROOT / "SIMPLE"
DATA_ROOT = Path("/data/local-data/data/Humanoid/psi-data")
TORCH_EXTENSIONS_ROOT = DATA_ROOT / "torch-extensions"
ISAAC_CACHE_ROOT = DATA_ROOT / "isaac-cache"
DEFAULT_SOURCE = DATA_ROOT / "simple-extracted/G1WholebodyBendHandoverTeleop-v0"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/playback/output/G1WholebodyBendHandoverTeleop-v0"

BODY_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
# HumanoidArena's protocol order (the project calls this canonical): legs and
# waist are interleaved by side, followed by the interleaved arm joints.
CANONICAL_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)
LEFT_HAND_NAMES = (
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
)
RIGHT_HAND_NAMES = (
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)

SOURCE_ACTION_DIM = 36
SOURCE_BASE_HEIGHT_INDEX = 31
SOURCE_LOCAL_XY_SLICE = slice(32, 34)
SOURCE_TARGET_YAW_INDEX = 35
REFERENCE_ROOT_SOURCE = "simple_synced_command_integrated"
REFERENCE_ROOT_ORIENTATION = "yaw_only_target"


class _WholebodyReplayObservationAdapter:
    """Expose the Sonic proprioception schema without changing G1Wholebody.

    The legacy MP robot predates ``prepare_obs``.  Replay still uses the
    decoupled WBC policy, so build the small observation dictionary from the
    robot's existing MuJoCo handles at the replay boundary.
    """

    def __init__(self, robot):
        self.robot = robot

    def prepare_obs(self) -> dict:
        robot = self.robot
        data, model = robot.mjdata, robot.mjmodel

        def joint_values(names, field):
            return np.asarray(
                [float(getattr(robot.joints[name], field)[0]) for name in names],
                dtype=np.float32,
            )

        body_names = robot.joints_names[:29]
        left_hand_names = robot.joints_names[29:36]
        right_hand_names = robot.joints_names[36:43]
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        torso_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, torso_id, torso_velocity, 1
        )
        return {
            "floating_base_pose": np.asarray(data.qpos[:7], dtype=np.float32).copy(),
            "floating_base_vel": np.asarray(data.qvel[:6], dtype=np.float32).copy(),
            "floating_base_acc": np.asarray(data.qacc[:6], dtype=np.float32).copy(),
            "body_q": joint_values(body_names, "qpos"),
            "body_dq": joint_values(body_names, "qvel"),
            "body_ddq": joint_values(body_names, "qacc"),
            "body_tau_est": np.asarray(
                [float(robot.actuators[name].force[0]) for name in body_names],
                dtype=np.float32,
            ),
            "left_hand_q": joint_values(left_hand_names, "qpos"),
            "left_hand_dq": joint_values(left_hand_names, "qvel"),
            "left_hand_ddq": joint_values(left_hand_names, "qacc"),
            "left_hand_tau_est": np.asarray(
                [float(robot.actuators[name].force[0]) for name in left_hand_names],
                dtype=np.float32,
            ),
            "right_hand_q": joint_values(right_hand_names, "qpos"),
            "right_hand_dq": joint_values(right_hand_names, "qvel"),
            "right_hand_ddq": joint_values(right_hand_names, "qacc"),
            "right_hand_tau_est": np.asarray(
                [float(robot.actuators[name].force[0]) for name in right_hand_names],
                dtype=np.float32,
            ),
            "secondary_imu_quat": np.asarray(data.xquat[torso_id], dtype=np.float32).copy(),
            # mj_objectVelocity returns angular then linear velocity.
            "secondary_imu_vel": np.concatenate(
                (torso_velocity[3:6], torso_velocity[:3])
            ).astype(np.float32),
            "time": float(data.time),
        }


def _legacy_replay_action(action_cmd, robot):
    """Convert a WBC target into G1Wholebody's existing replay command."""
    from simple.core.action import ActionCmd

    if action_cmd.type != "decoupled_wbc":
        raise ValueError(
            f"legacy replay expected a decoupled_wbc action, got {action_cmd.type!r}"
        )
    target_q = np.asarray(action_cmd["target_q"], dtype=np.float32).reshape(29)
    target_qpos = {
        name: float(value)
        for name, value in zip(robot.joints_names[:29], target_q)
    }
    for names, values in (
        (robot.joints_names[29:36], action_cmd["left_hand_q"]),
        (robot.joints_names[36:43], action_cmd["right_hand_q"]),
    ):
        if values is not None:
            target_qpos.update(
                {name: float(value) for name, value in zip(names, np.asarray(values).reshape(7))}
            )
    return ActionCmd("replay_move_actuators", target_qpos=target_qpos)


def configure_replay_boundary(robot):
    """Select the observation/action bridge supported by the concrete robot.

    ``G1Sonic`` natively exposes ``prepare_obs`` and consumes the
    ``decoupled_wbc`` ActionCmd returned by ReplayDecoupledAgent. Older
    G1Wholebody implementations predate that interface and instead need the
    observation adapter plus ``replay_move_actuators`` conversion.
    """
    from simple.robots.g1_sonic import G1Sonic

    if isinstance(robot, G1Sonic):
        if not callable(getattr(robot, "prepare_obs", None)):
            raise TypeError("G1Sonic replay requires robot.prepare_obs()")

        def native_action(action_cmd):
            if action_cmd.type != "decoupled_wbc":
                raise ValueError(
                    f"G1Sonic replay expected decoupled_wbc, got {action_cmd.type!r}"
                )
            return action_cmd

        return robot, native_action, "g1_sonic:decoupled_wbc"

    # Legacy G1Wholebody variants use plural/lowercase MuJoCo handle names.
    # Accept newer aliases as well so archived MP tasks remain replayable.
    if not hasattr(robot, "joints_names") and hasattr(robot, "joint_names"):
        robot.joints_names = robot.joint_names
    if not hasattr(robot, "mjdata") and hasattr(robot, "mjData"):
        robot.mjdata = robot.mjData
    if not hasattr(robot, "mjmodel") and hasattr(robot, "mjModel"):
        robot.mjmodel = robot.mjModel

    missing = [
        name for name in ("joints_names", "mjdata", "mjmodel", "joints", "actuators")
        if not hasattr(robot, name)
    ]
    if missing:
        raise TypeError(
            f"unsupported replay robot {type(robot).__module__}.{type(robot).__name__}; "
            f"missing legacy attributes: {', '.join(missing)}"
        )

    def legacy_action(action_cmd):
        return _legacy_replay_action(action_cmd, robot)

    return _WholebodyReplayObservationAdapter(robot), legacy_action, (
        f"{type(robot).__name__}:replay_move_actuators"
    )


def assert_isaac_gpu_ready(simulation_app) -> dict:
    """Fail before replay when Kit has no usable Vulkan/CUDA device.

    Isaac can report ``app ready`` even when RTX/GPU Foundation failed to
    initialize.  Continuing in that state produces empty camera buffers (or
    hangs in ``rep.orchestrator.step``), which is worse than a clear error for
    a dataset export.  The factory interface is available after
    ``SimulationApp`` startup and gives us a version-independent device count.
    """
    try:
        from omni.gpu_foundation_factory import get_gpu_foundation_factory_interface

        factory = get_gpu_foundation_factory_interface()
        count = int(factory.get_device_count())
        names = [str(factory.get_device_name(i)) for i in range(count)]
    except Exception as exc:  # pragma: no cover - depends on Kit extensions
        raise RuntimeError(
            "Isaac Sim GPU Foundation interface is unavailable; RTX camera "
            "capture cannot be trusted. See the Kit log for Vulkan/CUDA errors."
        ) from exc
    # On some multi-GPU Isaac 4.5 installations the CUDA-facing Foundation
    # interface reports zero devices even though the Vulkan renderer has an
    # active GPU (the Kit log prints the selected card and RTX can render).
    # In that mode the camera validator below is the authoritative check: it
    # rejects an all-zero frame before anything is written.  Keep the strict
    # behaviour by default, with an explicit opt-in for this known Vulkan-only
    # configuration.
    allow_vulkan_only = os.environ.get("SIMPLE_ISAAC_ALLOW_ZERO_GPU_COUNT", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if count <= 0 and not allow_vulkan_only:
        raise RuntimeError(
            "Isaac Sim started without a usable GPU device (GPU Foundation "
            "device_count=0). The H20 host currently reports CUDA/Vulkan "
            "initialization failure; refusing to write blank scene frames. "
            "Try a matching Vulkan/CUDA GPU mapping or restart the node."
        )
    return {"device_count": count, "devices": names,
            "vulkan_only_fallback": bool(count <= 0 and allow_vulkan_only)}


def validate_camera_frame(image: np.ndarray, frame_index: int) -> np.ndarray:
    """Validate an Isaac RGB frame before it enters the exported MP4."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise RuntimeError(f"Isaac camera frame {frame_index} has invalid shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    # A valid rendered room has non-zero dynamic range.  This catches the
    # all-zero/all-black buffer returned when RTX has no foundation device.
    if int(image.max()) == 0 or int(image.min()) == int(image.max()):
        raise RuntimeError(
            f"Isaac camera frame {frame_index} is blank (min=max={int(image.min())}); "
            "scene rendering is unavailable"
        )
    return image


def fixed_list(values: np.ndarray, width: int, dtype=None) -> pa.FixedSizeListArray:
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"expected matrix (*,{width}), got {values.shape}")
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float32() if dtype is None else dtype), width
    )


def rot6d_from_matrix(mats: np.ndarray) -> np.ndarray:
    # Arena stores row-major values of the first two matrix columns.
    return np.asarray(mats[:, :, :2].reshape(len(mats), 6), dtype=np.float32)


def matrix_from_rot6d(rot6d: np.ndarray) -> np.ndarray:
    """Decode Arena's row-major first-two-columns rotation representation."""
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    column0 = rot6d[:, (0, 2, 4)]
    column1 = rot6d[:, (1, 3, 5)]
    column0 /= np.linalg.norm(column0, axis=1, keepdims=True)
    column1 -= np.sum(column0 * column1, axis=1, keepdims=True) * column0
    column1 /= np.linalg.norm(column1, axis=1, keepdims=True)
    column2 = np.cross(column0, column1)
    return np.stack((column0, column1, column2), axis=-1)


def reference_root_from_source_action(source_action: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """Build an episode-local reference root from SIMPLE's synchronized command.

    SIMPLE's processed 36-D action stores base height at index 31 and the
    synchronized ``[vx, vy, turning_flag, target_yaw]`` navigation command at
    indices 32:36.  This mirrors HumanoidArena's TWIST2 reference-pose
    conversion: local XY velocity becomes a per-frame local displacement,
    while target yaw defines the episode-local reference orientation.

    The source does not contain a commanded pelvis roll/pitch, so the reference
    root orientation is intentionally yaw-only.  ``root_p_relative`` and
    ``root_q_relative`` are relative to the first reference pose; ``height`` is
    kept absolute for Arena action[2].
    """
    source_action = np.asarray(source_action, dtype=np.float32)
    if source_action.ndim != 2 or source_action.shape[1] != SOURCE_ACTION_DIM:
        raise ValueError(
            f"expected SIMPLE source action shape (T, {SOURCE_ACTION_DIM}), "
            f"got {source_action.shape}"
        )
    if len(source_action) == 0:
        raise ValueError("cannot construct a reference root for an empty episode")
    if not np.isfinite(source_action).all():
        raise ValueError("SIMPLE source action contains NaN or Inf")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid reference-root fps: {fps}")

    local_xy_delta = np.zeros((len(source_action), 2), dtype=np.float32)
    if len(source_action) > 1:
        local_xy_delta[1:] = source_action[1:, SOURCE_LOCAL_XY_SLICE] / fps

    height = source_action[:, SOURCE_BASE_HEIGHT_INDEX].astype(np.float32, copy=True)
    target_yaw = np.unwrap(
        source_action[:, SOURCE_TARGET_YAW_INDEX].astype(np.float64)
    )
    target_yaw -= target_yaw[0]
    root_rotations = Rotation.from_euler("z", target_yaw).as_matrix().astype(np.float32)
    quaternion_xyzw = Rotation.from_matrix(root_rotations).as_quat().astype(np.float32)
    root_q_relative = quaternion_xyzw[:, (3, 0, 1, 2)]

    root_p_relative = np.zeros((len(source_action), 3), dtype=np.float32)
    root_p_relative[:, 2] = height - height[0]
    for frame_index in range(1, len(source_action)):
        local_step = np.array(
            [local_xy_delta[frame_index, 0], local_xy_delta[frame_index, 1], 0.0],
            dtype=np.float32,
        )
        root_p_relative[frame_index, :2] = (
            root_p_relative[frame_index - 1, :2]
            + (root_rotations[frame_index] @ local_step)[:2]
        )

    return {
        "local_xy_delta": local_xy_delta,
        "height": height,
        "rotation_matrices": root_rotations,
        "root_p_relative": root_p_relative,
        "root_q_relative": root_q_relative,
    }


def reference_protocol_metadata() -> dict:
    """Return the complete V3.1 semantics used by replay exports."""
    return {
        "schema": "unitree_g1_gmt_refpose_v3_1",
        "version": "3.1",
        "action_dim": 40,
        "action_layout": "root_xy_delta_z_rot6d_joints29_hands2",
        "rotation_6d_layout": "row",
        "action_semantics": "reference_pose_not_robot_current_residual",
        "root_xy_delta_frame": "current_reference_base_frame",
        "root_rotation_frame": "episode_reference_frame",
        "fps": 50.0,
        "control_dt": 0.02,
        "source": "SIMPLE MuJoCo replay",
        "reference_root_source": REFERENCE_ROOT_SOURCE,
        "reference_root_orientation": REFERENCE_ROOT_ORIENTATION,
        "measured_root_fields": ["observation.root_p", "observation.root_q"],
    }


def add_reference_features(features: dict) -> None:
    """Declare the explicit audit/reference columns stored beside 64D/40D."""
    features.update(
        {
            "observation.root_p": {
                "dtype": "float32", "shape": [3], "names": ["x", "y", "z"]
            },
            "observation.root_q": {
                "dtype": "float32", "shape": [4], "names": ["w", "x", "y", "z"]
            },
            "observation.joint_q": {
                "dtype": "float32", "shape": [29], "names": list(CANONICAL_NAMES)
            },
            "observation.joint_dq": {
                "dtype": "float32", "shape": [29], "names": list(CANONICAL_NAMES)
            },
            "observation.hand_q": {
                "dtype": "float32", "shape": [14]
            },
            "action.target_joint_q": {
                "dtype": "float32", "shape": [29], "names": list(CANONICAL_NAMES)
            },
            "action.reference_root_p": {
                "dtype": "float32", "shape": [3], "names": ["x", "y", "z"],
                "semantics": "episode_first_reference_pose_relative",
            },
            "action.reference_root_q": {
                "dtype": "float32", "shape": [4], "names": ["w", "x", "y", "z"],
                "semantics": "episode_first_reference_pose_relative_yaw_only",
            },
            "source.states": {"dtype": "float32", "shape": [32]},
            "source.action": {"dtype": "float32", "shape": [36]},
        }
    )


def root_matrix_from_qpos(qpos: np.ndarray) -> np.ndarray:
    q = np.asarray(qpos[3:7], dtype=np.float64)
    q /= np.linalg.norm(q)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix().astype(np.float32)


def joint_vector(mj_data, names: tuple[str, ...], field: str) -> np.ndarray:
    out = []
    for name in names:
        joint = mj_data.joint(name)
        out.append(float(getattr(joint, field)[0]))
    return np.asarray(out, dtype=np.float32)


def hand_binary(command: np.ndarray) -> np.ndarray:
    """Map SIMPLE's 7-DoF absolute hand targets to Arena's binary labels."""
    command = np.asarray(command, dtype=np.float32).reshape(-1)[:14]
    if command.size != 14:
        raise ValueError(f"expected at least 14 hand command values, got {command.size}")
    return np.asarray(
        [float(np.max(np.abs(command[:7])) > 0.10),
         float(np.max(np.abs(command[7:])) > 0.10)],
        dtype=np.float32,
    )


def load_episode(source_root: Path, episode_index: int):
    files = sorted((source_root / "data").rglob(f"episode_{episode_index:06d}.parquet"))
    if not files:
        raise FileNotFoundError(f"episode {episode_index} parquet not found below {source_root / 'data'}")
    table = pq.read_table(files[0])
    return table.to_pandas(), files[0]


def load_env_config(source_root: Path, episode_index: int) -> dict:
    path = source_root / "meta/episodes.jsonl"
    with path.open() as f:
        for line in f:
            item = json.loads(line)
            if int(item.get("episode_index", -1)) == episode_index:
                raw = item.get("environment_config")
                if not raw:
                    raise ValueError(f"episode {episode_index} has no environment_config")
                return json.loads(raw)
    raise KeyError(f"episode {episode_index} missing from {path}")


def make_wbc_config():
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    config = SimLoopConfig()
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    # The archived teleop metadata uses 200 Hz physics and 50 Hz control.
    sonic_config["SIMULATE_DT"] = 0.005
    # The G1 MJCF exposes a free root in qpos/qvel but has actuators only for
    # the 43 body/hand joints.  Setting FREE_BASE=False prevents the upstream
    # robot adapter from prepending six non-existent base actuators to ctrl;
    # root motion is still measured from qpos below.
    sonic_config["FREE_BASE"] = False
    sonic_config["ENABLE_ELASTIC_BAND"] = False
    return sonic_config


def write_output(
    output_root: Path,
    episode_index: int,
    task_text: str,
    env_conf: dict,
    frames: dict[str, np.ndarray],
    images: list[np.ndarray],
    fps: int,
) -> None:
    data_dir = output_root / "data/chunk-000"
    video_dir = output_root / "videos/observation.images.front/chunk-000"
    episode_meta_dir = output_root / "meta/episodes"
    for p in (data_dir, video_dir, episode_meta_dir):
        p.mkdir(parents=True, exist_ok=True)

    parquet_path = data_dir / "file-000.parquet"
    n = len(frames["state"])
    columns = {
        "observation.state": fixed_list(frames["state"], 64),
        "action": fixed_list(frames["action"], 40),
        "observation.root_p": fixed_list(frames["root_p"], 3),
        "observation.root_q": fixed_list(frames["root_q"], 4),
        "observation.joint_q": fixed_list(frames["joint_q"], 29),
        "observation.joint_dq": fixed_list(frames["joint_dq"], 29),
        "observation.hand_q": fixed_list(frames["hand_q"], 14),
        "action.target_joint_q": fixed_list(frames["target_joint_q"], 29),
        "action.reference_root_p": fixed_list(frames["reference_root_p"], 3),
        "action.reference_root_q": fixed_list(frames["reference_root_q"], 4),
        "source.states": fixed_list(frames["source_states"], 32),
        "source.action": fixed_list(frames["source_action"], 36),
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
        "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64)),
        "timestamp": pa.array(np.arange(n, dtype=np.float32) / fps),
        "next.done": pa.array(np.asarray(frames["done"], dtype=bool)),
        "task_index": pa.array(np.zeros(n, dtype=np.int64)),
    }
    pq.write_table(pa.table(columns), parquet_path)

    video_path = video_dir / "file-000.mp4"
    h, w = images[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {video_path}")
    try:
        for image in images:
            writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    tasks_table = pa.table({"task_index": pa.array([0], type=pa.int64()), "task": pa.array([task_text])})
    pq.write_table(tasks_table, output_root / "meta/tasks.parquet")
    episode_table = pa.table({
        "episode_index": pa.array([episode_index], type=pa.int64()),
        "tasks": pa.array([[0]], type=pa.list_(pa.int64())),
        "length": pa.array([n], type=pa.int64()),
        "dataset_from_index": pa.array([0], type=pa.int64()),
        "dataset_to_index": pa.array([n - 1], type=pa.int64()),
        "data/chunk_index": pa.array([0], type=pa.int64()),
        "data/file_index": pa.array([0], type=pa.int64()),
        "videos/observation.images.front/chunk_index": pa.array([0], type=pa.int64()),
        "videos/observation.images.front/file_index": pa.array([0], type=pa.int64()),
        "videos/observation.images.front/from_timestamp": pa.array([0.0], type=pa.float32()),
    })
    pq.write_table(episode_table, episode_meta_dir / "chunk-000.parquet")
    with (output_root / "meta/episodes.jsonl").open("w") as f:
        f.write(json.dumps({"episode_index": episode_index, "length": n,
                            "environment_config": json.dumps(env_conf),
                            "success": bool(frames["success"][0])}) + "\n")

    info = {
        "codebase_version": "simple-replay-arena-v1-reference-root",
        "robot_type": "unitree_g1_refpose_v3_1",
        "total_episodes": 1,
        "total_frames": n,
        "total_tasks": 1,
        "total_videos": 1,
        "total_chunks": 1,
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/file-{file_index:03d}.mp4",
        "vla_protocol": reference_protocol_metadata(),
        "features": {
            "observation.images.front": {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channel"]},
            "observation.state": {
                "dtype": "float32", "shape": [64],
                "names": [f"state.root_heading_canonical_rot6d.{i}" for i in range(6)]
                + [f"state.dof_pos.{name}" for name in CANONICAL_NAMES]
                + [f"state.dof_vel.{name}" for name in CANONICAL_NAMES],
            },
            "action": {
                "dtype": "float32", "shape": [40],
                "names": ["action.root_ref_base_local_xy_delta.x", "action.root_ref_base_local_xy_delta.y",
                          "action.root_z"]
                + [f"action.root_ref_rot6d.{i}" for i in range(6)]
                + [f"action.joint_pos.{name}" for name in CANONICAL_NAMES]
                + ["action.hand_binary.left", "action.hand_binary.right"],
            },
        },
    }
    add_reference_features(info["features"])
    (output_root / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")


def validate_output(output_root: Path, fps: int, *, validate_root_motion: bool = True) -> dict:
    """Run Arena shape checks and construct the 417-D Kimodo representation."""
    table = pq.read_table(output_root / "data/chunk-000/file-000.parquet")
    state = np.asarray(table["observation.state"].combine_chunks().values).reshape(-1, 64).astype(np.float32)
    action = np.asarray(table["action"].combine_chunks().values).reshape(-1, 40).astype(np.float32)
    root_p = np.asarray(table["observation.root_p"].combine_chunks().values).reshape(-1, 3).astype(np.float32)
    root_q = np.asarray(table["observation.root_q"].combine_chunks().values).reshape(-1, 4).astype(np.float32)
    joint_q = np.asarray(table["observation.joint_q"].combine_chunks().values).reshape(-1, 29).astype(np.float32)
    target_joint_q = np.asarray(table["action.target_joint_q"].combine_chunks().values).reshape(-1, 29).astype(np.float32)
    reference_root_p = np.asarray(
        table["action.reference_root_p"].combine_chunks().values
    ).reshape(-1, 3).astype(np.float32)
    reference_root_q = np.asarray(
        table["action.reference_root_q"].combine_chunks().values
    ).reshape(-1, 4).astype(np.float32)
    source_action = np.asarray(table["source.action"].combine_chunks().values).reshape(-1, 36).astype(np.float32)
    assert state.shape[1:] == (64,) and action.shape[1:] == (40,)
    assert all(
        np.isfinite(values).all()
        for values in (
            state,
            action,
            root_p,
            root_q,
            joint_q,
            target_joint_q,
            reference_root_p,
            reference_root_q,
            source_action,
        )
    )
    assert np.unique(action[:, 38:40]).tolist() <= [0.0, 1.0]
    if len(action) > 1:
        assert np.max(np.linalg.norm(np.diff(root_p, axis=0), axis=1)) < 0.2, "root jump >20cm/frame"

    expected_reference = reference_root_from_source_action(source_action, fps)
    expected_root_action = np.concatenate(
        (
            expected_reference["local_xy_delta"],
            expected_reference["height"][:, None],
            rot6d_from_matrix(expected_reference["rotation_matrices"]),
        ),
        axis=1,
    )
    reference_action_error = float(np.max(np.abs(action[:, :9] - expected_root_action)))
    reference_position_field_error = float(
        np.max(np.abs(reference_root_p - expected_reference["root_p_relative"]))
    )
    reference_quaternion_field_error = float(
        np.max(np.abs(reference_root_q - expected_reference["root_q_relative"]))
    )
    if reference_action_error > 1e-5:
        raise ValueError(
            f"action root differs from SIMPLE reference command by {reference_action_error:.3e}"
        )
    if reference_position_field_error > 1e-5 or reference_quaternion_field_error > 1e-5:
        raise ValueError(
            "explicit reference root fields differ from the source command: "
            f"position={reference_position_field_error:.3e}, "
            f"quaternion={reference_quaternion_field_error:.3e}"
        )

    action_root_rotations = matrix_from_rot6d(action[:, 3:9])
    decoded_reference_p = np.zeros_like(reference_root_p)
    decoded_reference_p[:, 2] = action[:, 2]
    for frame_index in range(1, len(action)):
        local_delta = np.zeros(3, dtype=np.float64)
        local_delta[:2] = action[frame_index, :2]
        rotation = action_root_rotations[frame_index]
        delta_z = float(action[frame_index, 2] - action[frame_index - 1, 2])
        if abs(rotation[2, 2]) <= 1e-8:
            raise ValueError("reference root rotation cannot reconstruct the commanded height")
        local_delta[2] = (
            delta_z
            - rotation[2, 0] * local_delta[0]
            - rotation[2, 1] * local_delta[1]
        ) / rotation[2, 2]
        decoded_reference_p[frame_index, :2] = (
            decoded_reference_p[frame_index - 1, :2]
            + (rotation @ local_delta)[:2]
        )
    decoded_reference_p[:, 2] -= decoded_reference_p[0, 2]
    reference_decode_error = float(
        np.max(np.linalg.norm(decoded_reference_p - reference_root_p, axis=1))
    )
    if reference_decode_error > 1e-5:
        raise ValueError(
            f"Arena action decoder root differs from explicit reference by {reference_decode_error:.3e} m"
        )

    # Feed both measured configuration and root trajectory through the same
    # decoder used by controlnet_v1.2's HumanoidArena adapter.
    sys.path.insert(0, str(PROJECT_ROOT))
    # Isaac/Omniverse may preload a top-level ``utils`` package.  The motion
    # decoder intentionally imports this repository's ``utils.geometry``;
    # evict the unrelated module so the protocol validation is deterministic.
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            sys.modules.pop(name, None)
    import types
    repo_utils = types.ModuleType("utils")
    repo_utils.__path__ = [str(PROJECT_ROOT / "utils")]
    sys.modules["utils"] = repo_utils
    from motion.g1_reference import (
        CANONICAL_G1_JOINT_NAMES_29,
        HumanoidArenaActionDecoder,
        rot6d_row_to_matrix,
    )
    from motion.representation.kimodo_motionrep import KimodoMotionRep
    from skeleton.definitions import G1Skeleton34

    skeleton = G1Skeleton34()
    xml_path = PROJECT_ROOT / "skeleton/assets/g1skel34/xml/g1.xml"
    decoder = HumanoidArenaActionDecoder(skeleton, xml_path, fps)
    observed_root_rot = rot6d_row_to_matrix(__import__("torch").from_numpy(state[:, :6]))
    pose = decoder.decode_joint_configuration_pose(
        # Arena observation.state intentionally omits absolute root position;
        # the current training loader decodes observed motion at an episode
        # local planar origin (zeros), just as it does for native Arena data.
        joint_q, np.zeros_like(root_p), root_rotation_matrices=observed_root_rot,
        joint_names=CANONICAL_G1_JOINT_NAMES_29,
    )
    rep = KimodoMotionRep(skeleton=skeleton, fps=fps, stats_path=None)
    features = rep(pose["local_rot_mats"].unsqueeze(0), pose["root_positions"].unsqueeze(0), to_normalize=False)
    assert tuple(features.shape) == (1, len(state), 417), tuple(features.shape)
    target_pose = decoder.decode_action_pose(action)
    target_features = rep(
        target_pose["local_rot_mats"].unsqueeze(0),
        target_pose["root_positions"].unsqueeze(0),
        to_normalize=False,
    )
    assert tuple(target_features.shape) == (1, len(action), 417), tuple(target_features.shape)
    tracking_error = target_joint_q - joint_q
    root_steps = np.linalg.norm(np.diff(root_p, axis=0), axis=1) if len(root_p) > 1 else np.zeros(0)
    planar_steps = np.linalg.norm(np.diff(root_p[:, :2], axis=0), axis=1) if len(root_p) > 1 else np.zeros(0)
    tracking_rmse = float(np.sqrt(np.mean(tracking_error ** 2)))
    root_step_max = float(np.max(root_steps)) if len(root_steps) else 0.0
    root_path_length = float(np.sum(planar_steps))
    source_xy_command_peak = float(np.max(np.linalg.norm(source_action[:, 32:34], axis=1)))
    root_height_min = float(np.min(root_p[:, 2]))
    reference_planar_steps = (
        np.linalg.norm(np.diff(reference_root_p[:, :2], axis=0), axis=1)
        if len(reference_root_p) > 1
        else np.zeros(0)
    )
    reference_path_length = float(np.sum(reference_planar_steps))
    reference_quaternion_norm_error = float(
        np.max(np.abs(np.linalg.norm(reference_root_q, axis=1) - 1.0))
    )

    quality_errors = []
    if tracking_rmse > 0.20:
        quality_errors.append(f"joint tracking RMSE {tracking_rmse:.4f} rad exceeds 0.20")
    # G1Sonic uses the free-joint root at qpos[:7], so its height and planar
    # path are meaningful quality signals.  The archived G1Wholebody model
    # uses a different qpos layout; keep exporting its historical root fields
    # for compatibility, but do not reject an otherwise well-tracked legacy
    # replay based on coordinates that are not the floating base.
    if validate_root_motion:
        if root_height_min < 0.35:
            quality_errors.append(f"root height dropped to {root_height_min:.4f} m")
        if source_xy_command_peak > 0.05 and root_path_length < 0.02:
            quality_errors.append(
                "non-zero navigation command produced less than 0.02 m planar root motion"
            )

    return {"frames": int(len(state)), "state_dim": int(state.shape[1]), "action_dim": int(action.shape[1]),
            "kimodo_dim": int(features.shape[-1]), "target_kimodo_dim": int(target_features.shape[-1]),
            "root_step_max_m": root_step_max,
            "root_planar_path_m": root_path_length,
            "root_height_min_m": root_height_min,
            "source_xy_command_peak": source_xy_command_peak,
            "action_semantics": "reference_pose_not_robot_current_residual",
            "reference_root_source": REFERENCE_ROOT_SOURCE,
            "reference_root_orientation": REFERENCE_ROOT_ORIENTATION,
            "reference_root_planar_path_m": reference_path_length,
            "reference_root_decode_max_error_m": reference_decode_error,
            "reference_root_action_max_error": reference_action_error,
            "reference_root_quaternion_norm_max_error": reference_quaternion_norm_error,
            "joint_tracking_rmse_rad": tracking_rmse,
            "joint_tracking_max_rad": float(np.max(np.abs(tracking_error))),
            "quality_passed": not quality_errors,
            "quality_errors": quality_errors,
            "canonical_joint_order": list(CANONICAL_NAMES)}


def run(args) -> dict:
    os.environ.setdefault("SIMPLE_DATA_DIR", str(DATA_ROOT))
    # Keep any on-demand SIMPLE asset lookup on the fast Hugging Face mirror.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # Use MuJoCo's EGL backend on the headless H20 worker (no X11 DISPLAY).
    os.environ.setdefault("MUJOCO_GL", "egl")
    # CuRobo JIT-builds a small CUDA/C++ kinematics extension on first import.
    # Keep that cache on the data volume; the home NFS quota is too restrictive
    # for the compiler's temporary files.
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(TORCH_EXTENSIONS_ROOT))
    # Isaac Sim/Warp otherwise defaults to ~/.cache/warp.  The home filesystem
    # has a restrictive per-user quota, while the data volume has ample room.
    # Set these before importing SIMPLE (BaseDualSim preloads torch/curobo and
    # then starts Isaac's SimulationApp).
    os.environ.setdefault("WARP_CACHE_PATH", str(ISAAC_CACHE_ROOT / "warp"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ISAAC_CACHE_ROOT / "xdg"))
    os.environ.setdefault("TMPDIR", str(ISAAC_CACHE_ROOT / "tmp"))
    # Isaac Sim can otherwise probe every visible GPU and create one context
    # per card.  The physical GPU can be overridden with SIMPLE_ISAAC_GPU;
    # the legacy CUDA mask remaps it to ordinal 0 for both Isaac and PyTorch.
    # On RTX 4090 + Isaac 4.5, however, CUDA_VISIBLE_DEVICES makes GPU
    # Foundation report zero devices even while Vulkan is healthy.  Disable
    # that mask explicitly for the Vulkan-only fallback used by the remote
    # renderer.
    isaac_gpu = os.environ.get("SIMPLE_ISAAC_GPU", "6").strip() or "6"
    no_cuda_mask = os.environ.get("SIMPLE_ISAAC_NO_CUDA_MASK", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not no_cuda_mask:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", isaac_gpu)
    os.environ.setdefault("SIMPLE_ISAAC_ACTIVE_GPU", "0")
    os.environ.setdefault("SIMPLE_ISAAC_PHYSICS_GPU", "0")
    os.environ.setdefault("SIMPLE_ISAAC_MAX_GPU_COUNT", "1")
    os.environ.setdefault("SIMPLE_ISAAC_PORTABLE_ROOT", str(ISAAC_CACHE_ROOT / "portable"))
    for cache_dir in (ISAAC_CACHE_ROOT / "warp", ISAAC_CACHE_ROOT / "xdg", ISAAC_CACHE_ROOT / "tmp"):
        cache_dir.mkdir(parents=True, exist_ok=True)
    # H20 is compute capability 9.0; compiling only this target avoids
    # producing binaries for every visible GPU architecture.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    # The uv environment's ninja executable is not globally installed, while
    # torch's extension loader discovers it through PATH rather than import.
    os.environ["PATH"] = str(SIMPLE_ROOT / ".venv/bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("PYTHONPATH", str(SIMPLE_ROOT / "src"))
    sys.path.insert(0, str(SIMPLE_ROOT / "src"))
    # CuRobo is vendored as a source checkout.  The upstream SIMPLE editable
    # install exposes ``third_party`` (for its other vendored packages), while
    # CuRobo itself uses a ``src`` layout; add that directory explicitly so
    # MuJoCo task registration can import ``curobo.types`` without installing
    # the optional CUDA extension package.
    sys.path.insert(0, str(SIMPLE_ROOT / "third_party/curobo/src"))
    # unitree-sdk2py is a non-src-layout vendored checkout.
    sys.path.insert(0, str(SIMPLE_ROOT / "third_party/unitree_sdk2_python"))
    import simple.envs as _  # noqa: F401
    from simple.agents.replay_decoupled_agent import ReplayDecoupledAgent

    source_root = Path(args.source).resolve()
    output_root = Path(args.output).resolve()
    episode, source_path = load_episode(source_root, args.episode)
    env_conf = load_env_config(source_root, args.episode)
    task_text = json.loads((source_root / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
    sonic_config = make_wbc_config()
    env = gym.make(args.env_id, sim_mode=args.sim_mode, render_hz=50, headless=True,
                   max_episode_steps=max(len(episode) + 200, 1000), physics_dt=args.physics_dt,
                   sonic_config=sonic_config)
    raw_env = env.unwrapped
    mujoco_sim = raw_env.mujoco
    isaac_gpu = None
    if "isaac" in args.sim_mode:
        isaac_gpu = assert_isaac_gpu_ready(raw_env.simulation_app)
    observation, info = env.reset(options={"state_dict": env_conf, "task_id": f"episode_{args.episode}"})
    robot = raw_env.task.robot
    agent = ReplayDecoupledAgent(robot, sonic_config)
    replay_robot, adapt_action, replay_backend = configure_replay_boundary(robot)
    agent.robot = replay_robot
    print(
        f"[replay] robot={type(robot).__module__}.{type(robot).__name__} "
        f"backend={replay_backend}"
    )
    agent.load_episode(episode)
    if hasattr(agent._wbc_policy, "lower_body_policy"):
        agent._wbc_policy.lower_body_policy.use_policy_action = True

    # Stabilize without recording the transient phase.  Sonic robots expose a
    # latched ``stabilized`` property whose upstream threshold is intentionally
    # extremely strict (1e-4 m/s).  That threshold is useful for interactive
    # teleop, but can keep an offline replay in its warm-up loop for tens of
    # minutes because tiny contact noise never reaches zero.  For replay we
    # accept the same practical floating-base velocity criterion used by the
    # legacy G1Wholebody path; the strict latch still wins whenever it fires.
    has_stabilized_flag = hasattr(robot, "stabilized")
    stabilized = False
    min_fallback_steps = min(args.max_stabilize_steps, 30)
    replay_velocity_threshold = float(
        os.environ.get("SIMPLE_REPLAY_STABILIZE_VEL_THRESHOLD", "0.05")
    )
    if not np.isfinite(replay_velocity_threshold) or replay_velocity_threshold <= 0:
        raise ValueError(
            "SIMPLE_REPLAY_STABILIZE_VEL_THRESHOLD must be a positive finite number"
        )
    for stabilize_step in range(args.max_stabilize_steps):
        if has_stabilized_flag:
            strict_stabilized = bool(robot.stabilized)
            qvel = np.asarray(mujoco_sim.mjData.qvel[:6], dtype=np.float32)
            practical_stabilized = (
                stabilize_step >= min_fallback_steps
                and np.max(np.abs(qvel)) < replay_velocity_threshold
            )
            stabilized = strict_stabilized or practical_stabilized
        elif stabilize_step >= min_fallback_steps:
            qvel = np.asarray(mujoco_sim.mjData.qvel[:6], dtype=np.float32)
            stabilized = bool(np.max(np.abs(qvel)) < replay_velocity_threshold)
        if stabilized:
            break
        stabilize_action = adapt_action(agent.get_stabilize_action(observation))
        observation, _, _, _, info = env.step(stabilize_action)
    if not stabilized:
        print(
            "[replay] practical stabilization threshold not reached; "
            "continuing so validation can decide whether this replay is usable"
        )

    states, actions, root_ps, root_qs, joint_qs, joint_dqs, hand_qs, target_joint_qs = [], [], [], [], [], [], [], []
    reference_root_ps, reference_root_qs = [], []
    source_states, source_actions, dones, images = [], [], [], []
    success = []
    first_heading = None
    body_name_to_index = {name: i for i, name in enumerate(BODY_NAMES)}
    frame_limit = len(episode) if args.max_frames is None else min(len(episode), max(1, args.max_frames))
    source_action_matrix = np.stack(
        [np.asarray(value, dtype=np.float32) for value in episode["action"].iloc[:frame_limit]],
        axis=0,
    )
    reference_root = reference_root_from_source_action(source_action_matrix, 50)
    for i in range(frame_limit):
        action_cmd = agent.get_action(observation)
        target_q_body = np.asarray(action_cmd["target_q"], dtype=np.float32).reshape(29)
        target_q = target_q_body[[body_name_to_index[name] for name in CANONICAL_NAMES]]
        source_row = episode.iloc[i]
        replay_action = adapt_action(action_cmd)
        observation, _, terminated, truncated, info = env.step(replay_action)
        qpos = np.asarray(mujoco_sim.mjData.qpos[:7], dtype=np.float32).copy()
        p = qpos[:3].copy()
        r_world = root_matrix_from_qpos(qpos)
        if first_heading is None:
            yaw = float(np.arctan2(r_world[1, 0], r_world[0, 0]))
            first_heading = Rotation.from_euler("z", yaw).as_matrix().astype(np.float32)
        r_rel = first_heading.T @ r_world
        # Export the protocol's canonical order, not the MuJoCo XML/Unitree
        # contiguous-leg order used internally by the WBC controller.
        q = joint_vector(mujoco_sim.mjData, CANONICAL_NAMES, "qpos")
        dq = joint_vector(mujoco_sim.mjData, CANONICAL_NAMES, "qvel")
        hq = np.concatenate([joint_vector(mujoco_sim.mjData, LEFT_HAND_NAMES, "qpos"),
                             joint_vector(mujoco_sim.mjData, RIGHT_HAND_NAMES, "qpos")])
        source_command = source_action_matrix[i]
        reference_rotation = reference_root["rotation_matrices"][i]
        state = np.concatenate([rot6d_from_matrix(r_rel[None])[0], q, dq]).astype(np.float32)
        arena_action = np.concatenate(
            [
                reference_root["local_xy_delta"][i],
                [reference_root["height"][i]],
                rot6d_from_matrix(reference_rotation[None])[0],
                target_q,
                hand_binary(source_command),
            ]
        ).astype(np.float32)
        states.append(state); actions.append(arena_action); root_ps.append(p); root_qs.append(qpos[3:7]);
        joint_qs.append(q); joint_dqs.append(dq); hand_qs.append(hq)
        # Keep the WBC target separately for auditing measured-vs-reference
        # tracking while exposing it in the protocol action[9:38].
        target_joint_qs.append(target_q)
        reference_root_ps.append(reference_root["root_p_relative"][i])
        reference_root_qs.append(reference_root["root_q_relative"][i])
        source_states.append(np.asarray(source_row["states"], dtype=np.float32))
        source_actions.append(source_command)
        image = np.asarray(observation["head_stereo_left"], dtype=np.uint8).copy()
        if "isaac" in args.sim_mode:
            image = validate_camera_frame(image, i)
        images.append(image)
        dones.append(bool(terminated or truncated)); success.append(bool(getattr(raw_env, "_success", False)))
        if terminated or truncated:
            break
    frames = {"state": np.stack(states), "action": np.stack(actions), "root_p": np.stack(root_ps),
              "root_q": np.stack(root_qs), "joint_q": np.stack(joint_qs), "joint_dq": np.stack(joint_dqs),
              "hand_q": np.stack(hand_qs), "source_states": np.stack(source_states),
              "target_joint_q": np.stack(target_joint_qs), "source_action": np.stack(source_actions),
              "reference_root_p": np.stack(reference_root_ps),
              "reference_root_q": np.stack(reference_root_qs),
              "done": np.asarray(dones), "success": np.asarray(success)}
    write_output(output_root, args.episode, task_text, env_conf, frames, images, 50)
    report = validate_output(
        output_root,
        50,
        validate_root_motion=replay_backend.startswith("g1_sonic:"),
    )
    report.update({"source": str(source_path), "output": str(output_root), "success": bool(success[-1]),
                   "replay_backend": replay_backend,
                   "recorded_frames": len(states), "source_frames": len(episode)})
    if isaac_gpu is not None:
        report["isaac_gpu"] = isaac_gpu
    (output_root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    # Isaac Sim's close path may tear down Kit's Python runtime on some 4.5
    # builds.  Persist the replay and validation first so a successful camera
    # capture is never lost during renderer shutdown.
    env.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--env-id", default="simple/G1WholebodyBendHandoverTeleop-v0")
    parser.add_argument("--sim-mode", choices=("mujoco_isaac", "mujoco"), default="mujoco_isaac",
                        help="MuJoCo physics only, or synchronized Isaac Sim HSSD rendering (default).")
    parser.add_argument("--max-stabilize-steps", type=int, default=600)
    parser.add_argument("--physics-dt", type=float, default=0.005,
                        help="Physics timestep passed to the SIMPLE task. BendPickMP requires 0.002.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional limit for a quick renderer smoke test.")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
