#!/usr/bin/env python3
"""Replay one SIMPLE LeRobot episode and export an Arena/Kimodo-compatible set.

The default ``mujoco_isaac`` mode keeps MuJoCo as the fast physics/WBC engine
and drives Isaac Sim as the synchronized renderer.  This is important for
SIMPLE episodes: the camera image is rendered from the recorded HSSD room,
instead of MuJoCo's plain fallback floor/table.  Passing ``--sim-mode mujoco``
is still available as a lightweight diagnostic.

The source SIMPLE archive contains a 32-D observation and 36-D WBC command, so
the exporter samples the *measured* MuJoCo state after each replay step and
writes the 64-D/40-D Arena protocol:

    state = root_rot6d_relative_to_episode_heading + q29 + dq29
    action = root_local_xy_delta + root_z + root_rot6d + q29 + hand_binary

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
    if count <= 0:
        raise RuntimeError(
            "Isaac Sim started without a usable GPU device (GPU Foundation "
            "device_count=0). The H20 host currently reports CUDA/Vulkan "
            "initialization failure; refusing to write blank scene frames. "
            "Try a matching Vulkan/CUDA GPU mapping or restart the node."
        )
    return {"device_count": count, "devices": names}


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
        "codebase_version": "simple-replay-arena-v0",
        "robot_type": "unitree_g1_refpose_v3_1",
        "total_episodes": 1,
        "total_frames": n,
        "total_tasks": 1,
        "total_videos": 1,
        "total_chunks": 1,
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/file-{file_index:03d}.mp4",
        "vla_protocol": {"schema": "unitree_g1_gmt_refpose_v3_1", "version": "3.1", "source": "SIMPLE MuJoCo replay"},
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
    (output_root / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")


def validate_output(output_root: Path, fps: int) -> dict:
    """Run Arena shape checks and construct the 417-D Kimodo representation."""
    table = pq.read_table(output_root / "data/chunk-000/file-000.parquet")
    state = np.asarray(table["observation.state"].combine_chunks().values).reshape(-1, 64).astype(np.float32)
    action = np.asarray(table["action"].combine_chunks().values).reshape(-1, 40).astype(np.float32)
    root_p = np.asarray(table["observation.root_p"].combine_chunks().values).reshape(-1, 3).astype(np.float32)
    root_q = np.asarray(table["observation.root_q"].combine_chunks().values).reshape(-1, 4).astype(np.float32)
    joint_q = np.asarray(table["observation.joint_q"].combine_chunks().values).reshape(-1, 29).astype(np.float32)
    target_joint_q = np.asarray(table["action.target_joint_q"].combine_chunks().values).reshape(-1, 29).astype(np.float32)
    assert state.shape[1:] == (64,) and action.shape[1:] == (40,)
    assert np.isfinite(state).all() and np.isfinite(action).all()
    assert np.unique(action[:, 38:40]).tolist() <= [0.0, 1.0]
    if len(action) > 1:
        assert np.max(np.linalg.norm(np.diff(root_p, axis=0), axis=1)) < 0.2, "root jump >20cm/frame"

    # Feed both measured configuration and root trajectory through the same
    # decoder used by controlnet_v1.2's HumanoidArena adapter.
    sys.path.insert(0, str(PROJECT_ROOT))
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
    return {"frames": int(len(state)), "state_dim": int(state.shape[1]), "action_dim": int(action.shape[1]),
            "kimodo_dim": int(features.shape[-1]), "target_kimodo_dim": int(target_features.shape[-1]),
            "root_step_max_m": float(np.max(np.linalg.norm(np.diff(root_p, axis=0), axis=1))) if len(root_p) > 1 else 0.0,
            "joint_tracking_rmse_rad": float(np.sqrt(np.mean(tracking_error ** 2))),
            "joint_tracking_max_rad": float(np.max(np.abs(tracking_error))),
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
    # per card.  A single idle H20 is sufficient for this one-episode
    # renderer.  The physical GPU can be overridden with
    # SIMPLE_ISAAC_GPU; CUDA_VISIBLE_DEVICES then remaps it to ordinal 0 for
    # both Isaac and PyTorch.
    isaac_gpu = os.environ.get("SIMPLE_ISAAC_GPU", "6").strip() or "6"
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
                   max_episode_steps=max(len(episode) + 200, 1000), physics_dt=0.005,
                   sonic_config=sonic_config)
    raw_env = env.unwrapped
    isaac_gpu = None
    if "isaac" in args.sim_mode:
        isaac_gpu = assert_isaac_gpu_ready(raw_env.simulation_app)
    observation, info = env.reset(options={"state_dict": env_conf, "task_id": f"episode_{args.episode}"})
    robot = raw_env.task.robot
    agent = ReplayDecoupledAgent(robot, sonic_config)
    agent.load_episode(episode)
    if hasattr(agent._wbc_policy, "lower_body_policy"):
        agent._wbc_policy.lower_body_policy.use_policy_action = True

    # Stabilize without recording the transient phase.
    for _ in range(args.max_stabilize_steps):
        if robot.stabilized:
            break
        observation, _, _, _, info = env.step(agent.get_stabilize_action(observation))
    if not robot.stabilized:
        raise RuntimeError("robot did not stabilize; increase --max-stabilize-steps")

    states, actions, root_ps, root_qs, joint_qs, joint_dqs, hand_qs, target_joint_qs = [], [], [], [], [], [], [], []
    source_states, source_actions, dones, images = [], [], [], []
    success = []
    prev_p = None
    first_heading = None
    body_name_to_index = {name: i for i, name in enumerate(BODY_NAMES)}
    for i in range(len(episode)):
        action_cmd = agent.get_action(observation)
        target_q_body = np.asarray(action_cmd["target_q"], dtype=np.float32).reshape(29)
        target_q = target_q_body[[body_name_to_index[name] for name in CANONICAL_NAMES]]
        source_row = episode.iloc[i]
        observation, _, terminated, truncated, info = env.step(action_cmd)
        qpos = np.asarray(raw_env.mjData.qpos[:7], dtype=np.float32).copy()
        p = qpos[:3].copy()
        r_world = root_matrix_from_qpos(qpos)
        if first_heading is None:
            yaw = float(np.arctan2(r_world[1, 0], r_world[0, 0]))
            first_heading = Rotation.from_euler("z", yaw).as_matrix().astype(np.float32)
        r_rel = first_heading.T @ r_world
        # Export the protocol's canonical order, not the MuJoCo XML/Unitree
        # contiguous-leg order used internally by the WBC controller.
        q = joint_vector(raw_env.mjData, CANONICAL_NAMES, "qpos")
        dq = joint_vector(raw_env.mjData, CANONICAL_NAMES, "qvel")
        hq = np.concatenate([joint_vector(raw_env.mjData, LEFT_HAND_NAMES, "qpos"),
                             joint_vector(raw_env.mjData, RIGHT_HAND_NAMES, "qpos")])
        delta = np.zeros(3, dtype=np.float32) if prev_p is None else p - prev_p
        local_delta = r_world.T @ delta
        state = np.concatenate([rot6d_from_matrix(r_rel[None])[0], q, dq]).astype(np.float32)
        arena_action = np.concatenate([local_delta[:2], [p[2]], rot6d_from_matrix(r_rel[None])[0], target_q,
                                        hand_binary(np.asarray(source_row["action"]))]).astype(np.float32)
        states.append(state); actions.append(arena_action); root_ps.append(p); root_qs.append(qpos[3:7]);
        joint_qs.append(q); joint_dqs.append(dq); hand_qs.append(hq)
        # Keep the WBC target separately for auditing measured-vs-reference
        # tracking while exposing it in the protocol action[9:38].
        target_joint_qs.append(target_q)
        source_states.append(np.asarray(source_row["states"], dtype=np.float32))
        source_actions.append(np.asarray(source_row["action"], dtype=np.float32))
        image = np.asarray(observation["head_stereo_left"], dtype=np.uint8).copy()
        if "isaac" in args.sim_mode:
            image = validate_camera_frame(image, i)
        images.append(image)
        dones.append(bool(terminated or truncated)); success.append(bool(getattr(raw_env, "_success", False)))
        prev_p = p
        if terminated or truncated:
            break
    env.close()
    frames = {"state": np.stack(states), "action": np.stack(actions), "root_p": np.stack(root_ps),
              "root_q": np.stack(root_qs), "joint_q": np.stack(joint_qs), "joint_dq": np.stack(joint_dqs),
              "hand_q": np.stack(hand_qs), "source_states": np.stack(source_states),
              "target_joint_q": np.stack(target_joint_qs), "source_action": np.stack(source_actions),
              "done": np.asarray(dones), "success": np.asarray(success)}
    write_output(output_root, args.episode, task_text, env_conf, frames, images, 50)
    report = validate_output(output_root, 50)
    report.update({"source": str(source_path), "output": str(output_root), "success": bool(success[-1]),
                   "recorded_frames": len(states), "source_frames": len(episode)})
    if isaac_gpu is not None:
        report["isaac_gpu"] = isaac_gpu
    (output_root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
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
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
