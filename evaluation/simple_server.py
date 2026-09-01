#!/usr/bin/env python3
"""SIMPLE evaluation bridge for Kimodo checkpoints.

The replay exporter and the Kimodo dataloader use the Arena-compatible
``state=(64,)``/``action=(40,)`` contract.  SIMPLE's historical HTTP agents
send a smaller 32-D WBC state and expect a 36-D command, so this module keeps
that legacy protocol available while the evaluator itself uses the complete
SIMPLE ``info["proprio"]`` observation and the native Sonic WBC agent.

Run as a server (for clients that provide a 64-D state history)::

    python evaluation/simple_server.py --checkpoint CHECKPOINT --device cuda:0

Run the self-contained evaluator (the mode used by
``simple_eval_signal_task_sonic.sh``)::

    python evaluation/simple_server.py --eval --task G1WholebodyCloseDoorTeleop-v0 \
        --checkpoint CHECKPOINT --device cuda:0 --headless
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMPLE_ROOT = PROJECT_ROOT / "SIMPLE"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# MuJoCo/SIMPLE use this order, while Kimodo's protocol uses the interleaved
# canonical order.  Keeping the names here makes the adapter auditable and
# avoids depending on incidental XML ordering.
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

# SIMPLE's 36-D WBC command stores arms first, then left thumb/middle/index and
# right hand, followed by waist roll/pitch/yaw, height and navigation.
LEFT_HAND_CLOSE = np.asarray(
    [0.3523, -0.0964, 0.2790, -0.5058, -1.1950, -0.5389, -0.9835],
    dtype=np.float32,
)
RIGHT_HAND_CLOSE = np.asarray(
    [0.02331954, -0.02398408, -0.22170663, 0.25662386, 1.3371105, 0.3085137, 0.9805285],
    dtype=np.float32,
)

# Hand labels in the Kimodo checkpoint are discrete states, not velocities.
# The SIMPLE WBC nevertheless needs the corresponding target to be present for
# multiple 50 Hz control cycles.  These parameters implement a small
# transition controller on the SIMPLE side only.  A two-frame confirmation is
# about 67 ms at the checkpoint's 30 Hz prediction rate and filters isolated
# diffusion threshold noise without delaying a real event appreciably.
_SIMPLE_HAND_EVENT_CONFIRM_FRAMES = 2
_SIMPLE_HAND_MIN_HOLD_SECONDS = 0.4
_SIMPLE_HAND_MAX_HOLD_SECONDS = 2.0
_SIMPLE_HAND_OPEN_POSITION_TOL = 0.08
_SIMPLE_HAND_VELOCITY_TOL = 0.20
_SIMPLE_HAND_CLOSE_PROGRESS = 0.45
_SIMPLE_HAND_TORQUE_FRACTION = 0.75


def _mjcf_hand_to_wbc(values: Any) -> np.ndarray:
    """Convert SIMPLE/MuJoCo hand order to decoupled-WBC order.

    The G1 MuJoCo model declares each hand as thumb, middle, index, whereas
    the Pinocchio/decoupled-WBC model exposes thumb, index, middle.  The
    arrays returned by ``G1Sonic.prepare_obs`` therefore need this explicit
    name-based permutation before being fed back to WBC.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size != 7:
        raise ValueError(f"expected 7 hand values, got {values.size}")
    return np.concatenate((values[:3], values[5:7], values[3:5])).astype(
        np.float32, copy=False
    )


def _wbc_hand_to_mjcf(values: Any) -> np.ndarray:
    """Convert decoupled-WBC hand order back to the MuJoCo actuator order."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size != 7:
        raise ValueError(f"expected 7 hand values, got {values.size}")
    return np.concatenate((values[:3], values[5:7], values[3:5])).astype(
        np.float32, copy=False
    )


def _numpy_serialize(value: Any) -> Any:
    """Serialize numpy values using SIMPLE's official JSON wire format."""
    from numpy.lib.format import dtype_to_descr

    if isinstance(value, np.ndarray):
        raw = value.tobytes(order="C")
        return {
            "__numpy__": base64.b64encode(raw).decode("ascii"),
            "dtype": dtype_to_descr(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return _numpy_serialize(np.asarray(value))
    if isinstance(value, dict):
        return {str(k): _numpy_serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numpy_serialize(v) for v in value]
    return value


def _numpy_deserialize(value: Any) -> Any:
    from numpy.lib.format import descr_to_dtype

    if isinstance(value, dict) and "__numpy__" in value:
        raw = base64.b64decode(value["__numpy__"])
        array = np.frombuffer(raw, dtype=descr_to_dtype(value["dtype"]))
        shape = tuple(int(v) for v in value.get("shape", ()))
        return array.reshape(shape) if shape else array[0]
    if isinstance(value, dict):
        return {k: _numpy_deserialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_numpy_deserialize(v) for v in value]
    return value


def _rot6d_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[..., :, :2].reshape(-1, 6)[0], dtype=np.float32)


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("floating-base quaternion is empty or non-finite")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _as_named_vector(values: Any, names: tuple[str, ...], label: str) -> np.ndarray:
    if isinstance(values, dict):
        try:
            result = np.asarray([values[name] for name in names], dtype=np.float32)
        except KeyError as exc:
            raise ValueError(f"{label} is missing joint {exc.args[0]!r}") from exc
    else:
        result = np.asarray(values, dtype=np.float32).reshape(-1)
        if result.size != len(names):
            raise ValueError(f"{label} must have {len(names)} values, got {result.size}")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains NaN or Inf")
    return result


def build_arena_state(proprio: dict[str, Any], first_heading: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Convert SIMPLE ``G1Sonic.prepare_obs`` to the replay 64-D state."""
    pose = np.asarray(proprio.get("floating_base_pose"), dtype=np.float32).reshape(-1)
    if pose.size < 7:
        raise ValueError("SIMPLE proprio must expose floating_base_pose with 7 values")
    root_rotation = _quat_wxyz_to_matrix(pose[3:7])
    if first_heading is None:
        first_heading = _yaw_matrix(float(np.arctan2(root_rotation[1, 0], root_rotation[0, 0])))
    relative_rotation = first_heading.T @ root_rotation
    q_body = _as_named_vector(proprio.get("body_q"), BODY_NAMES, "body_q")
    dq_body = _as_named_vector(proprio.get("body_dq"), BODY_NAMES, "body_dq")
    q = np.asarray([q_body[BODY_NAMES.index(name)] for name in CANONICAL_NAMES], dtype=np.float32)
    dq = np.asarray([dq_body[BODY_NAMES.index(name)] for name in CANONICAL_NAMES], dtype=np.float32)
    state = np.concatenate((_rot6d_from_matrix(relative_rotation[None]), q, dq)).astype(np.float32)
    if state.shape != (64,):
        raise AssertionError(f"internal state adapter produced {state.shape}, expected (64,)")
    return state, first_heading


def append_state_history(
    state_buffer: list[np.ndarray],
    proprio: dict[str, Any],
    first_heading: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Append one 50 Hz SIMPLE proprio frame to a pending inference segment."""
    state, first_heading = build_arena_state(proprio, first_heading)
    state_buffer.append(state)
    return state, first_heading


def arena_action_to_simple(
    action_chunk: np.ndarray,
    control_fps: float = 50.0,
) -> np.ndarray:
    """Convert Kimodo's 40-D reference action to SIMPLE's 36-D WBC command."""
    actions = np.asarray(action_chunk, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != 40:
        raise ValueError(f"expected Kimodo action shape (T, 40), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Kimodo action contains NaN or Inf")

    canonical_index = {name: i for i, name in enumerate(CANONICAL_NAMES)}
    body_index = {name: i for i, name in enumerate(BODY_NAMES)}
    source = np.zeros((len(actions), 36), dtype=np.float32)
    for frame, action in enumerate(actions):
        q_canonical = action[9:38]
        q_body = np.asarray([q_canonical[canonical_index[name]] for name in BODY_NAMES], dtype=np.float32)
        arms = q_body[body_index["left_shoulder_pitch_joint"] : body_index["right_wrist_yaw_joint"] + 1]
        # BODY_NAMES has the two arm blocks contiguous (14 values).
        source[frame, 14:28] = arms
        source[frame, 28] = q_canonical[canonical_index["waist_roll_joint"]]
        source[frame, 29] = q_canonical[canonical_index["waist_pitch_joint"]]
        source[frame, 30] = q_canonical[canonical_index["waist_yaw_joint"]]

        left_hand = LEFT_HAND_CLOSE if action[38] >= 0.5 else np.zeros(7, dtype=np.float32)
        right_hand = RIGHT_HAND_CLOSE if action[39] >= 0.5 else np.zeros(7, dtype=np.float32)
        source[frame, 0:3] = left_hand[:3]
        source[frame, 3:5] = left_hand[5:7]
        source[frame, 5:7] = left_hand[3:5]
        source[frame, 7:14] = right_hand

        # Kimodo's planar canonicalization translates feature x/z only.  In the
        # MuJoCo convention used by the 40-D action, height is z (Kimodo y), so
        # action[2] remains an absolute pelvis-height command.
        source[frame, 31] = action[2]
        source[frame, 32] = action[0] * float(control_fps)
        source[frame, 33] = action[1] * float(control_fps)
        source[frame, 34] = 0.0  # turning flag is not represented by the 40-D contract
        # The action root is episode-relative, matching SIMPLE's target-yaw frame.
        try:
            from motion.g1_reference import rot6d_row_to_matrix
            import torch

            rotation = rot6d_row_to_matrix(torch.from_numpy(action[3:9]).reshape(1, 6))[0].numpy()
        except Exception as exc:  # pragma: no cover - only reached in broken environments
            raise RuntimeError("could not decode Kimodo root rotation") from exc
        source[frame, 35] = np.arctan2(rotation[1, 0], rotation[0, 0])
    return source


def _simple_image(observation: dict[str, Any]) -> np.ndarray:
    for key in ("head_stereo_left", "front_stereo_left", "rgb_head_stereo_left", "front"):
        if key in observation:
            image = np.asarray(observation[key])
            if image.ndim == 3 and image.shape[-1] in (3, 4):
                image = image[..., :3]
                if image.dtype != np.uint8:
                    image = np.clip(image, 0, 255).astype(np.uint8)
                return np.ascontiguousarray(image)
    raise ValueError("SIMPLE observation has no HWC RGB camera frame")


def _export_ego_video(episode_dir: Path) -> Path | None:
    """Expose SIMPLE's head camera recording under Arena's ``ego_view`` name."""
    source_candidates = (
        episode_dir / "head_stereo_left_success.mp4",
        episode_dir / "head_stereo_left_failed.mp4",
        episode_dir / "head_stereo_left.mp4",
    )
    source = next((path for path in source_candidates if path.exists()), None)
    if source is None:
        return None
    suffix = source.name.removeprefix("head_stereo_left")
    target = episode_dir / f"ego_view{suffix}"
    if target != source:
        shutil.copy2(source, target)
    return target


def _robot_proprio(robot: Any) -> dict[str, Any] | None:
    """Return the common proprio schema for Sonic and legacy G1Wholebody."""
    if hasattr(robot, "prepare_obs"):
        value = robot.prepare_obs()
        if isinstance(value, dict):
            return value
    joints = getattr(robot, "joints", None)
    if isinstance(joints, dict) and all(name in joints for name in BODY_NAMES):
        body_q = np.asarray([joints[name].qpos[0] for name in BODY_NAMES], dtype=np.float32)
        body_dq = np.asarray([joints[name].qvel[0] for name in BODY_NAMES], dtype=np.float32)
        data = getattr(robot, "mjdata", None)
        if data is None:
            data = getattr(robot, "mjData", None)
        pose = np.asarray(data.qpos[:7], dtype=np.float32) if data is not None else np.asarray([0, 0, 0.75, 1, 0, 0, 0], dtype=np.float32)
        return {"floating_base_pose": pose, "body_q": body_q, "body_dq": body_dq}
    return None


def _reset_episode_step_counters(env: Any) -> None:
    """Reset Gym TimeLimit counters after Sonic's stabilization pre-roll.

    ``gym.make(..., max_episode_steps=...)`` wraps the environment in a
    ``TimeLimit``.  Sonic stabilization advances that wrapper before the
    recorded episode starts, so clearing only the raw SIMPLE environment's
    ``step_count`` would make the first policy episode truncate early.
    Traverse the wrapper chain and reset each elapsed-step counter explicitly.
    """
    current = env
    while current is not None:
        if hasattr(current, "_elapsed_steps"):
            current._elapsed_steps = 0
        current = getattr(current, "env", None)


def _make_sonic_config() -> dict[str, Any]:
    """Load the same Sonic WBC config used by SIMPLE replay/datagen."""
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    config = SimLoopConfig().load_wbc_yaml()
    config["ENV_NAME"] = "simple"
    config["SIMULATE_DT"] = 0.005  # 200 Hz physics, 50 Hz control
    config["FREE_BASE"] = False  # G1 MJCF has no six base actuators
    config["ENABLE_ELASTIC_BAND"] = False
    return config


def _runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=args.checkpoint,
        text_embedding_cache=args.text_embedding_cache,
        device=args.device,
        dtype=args.dtype,
        diffusion_steps=args.diffusion_steps,
        execution_frames=args.execution_frames,
        rtc=args.rtc,
        rtc_overlap_frames=args.rtc_overlap_frames,
        rtc_frozen_frames=args.rtc_frozen_frames,
        rtc_ramp_power=args.rtc_ramp_power,
        control_fps=args.control_fps,
        max_navigation_speed=args.max_navigation_speed,
    )


def _add_model_dependency_paths() -> None:
    """Make the shared training environment visible from SIMPLE/.venv.

    SIMPLE's environment intentionally omits a few training-only packages
    (notably ``safetensors`` and ``transformers``).  On this workstation those
    wheels live in the patch-policy environment; an explicit override keeps
    the bridge portable to another installation.
    """
    # Pin torch/torchvision from the active environment before adding training
    # wheels.  The vendored curobo extension is ABI-sensitive; importing a
    # second torch build from a training environment makes it fail to load.
    try:
        import torch  # noqa: F401
        try:
            import torchvision  # noqa: F401
        except Exception:
            pass
        # Some remote wheels omit torchvision's compiled NMS operator while
        # torchvision still unconditionally registers its fake implementation.
        # Define the operator schema before transformers lazily imports
        # torchvision transforms (used by the DINOv3 image processor).
        try:
            torch.ops.torchvision.nms
        except (AttributeError, RuntimeError):
            try:
                _torchvision_library = torch.library.Library("torchvision", "DEF")
                _torchvision_library.define(
                    "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
                )
                globals()["_TORCHVISION_LIBRARY"] = _torchvision_library
            except Exception:
                pass
    except Exception:
        pass

    # Transformers/accelerate are loaded from a separate training overlay on
    # some machines.  Keep SIMPLE's already-imported torch/numpy ahead of that
    # overlay: putting a whole conda site-packages directory at sys.path[0]
    # can shadow the stdlib (for example with an old ``uuid.py``) or replace
    # the CUDA build used by SIMPLE.
    try:
        import numpy as _np
        import numpy.core.multiarray as _multiarray

        # accelerate >=1.0 detects NumPy >=2 from its own distribution
        # metadata.  When it runs with SIMPLE's NumPy 1.x, provide the
        # compatible private alias expected by its safe-loader registration.
        if not hasattr(_np, "_core"):
            _np._core = _np.core
        _np._core.multiarray = _multiarray
        # NumPy 1.x exposes ``numpy.core`` while newer accelerate releases
        # import the private ``numpy._core`` package directly.  Importing the
        # latter creates a distinct module without the compatibility alias,
        # so install the alias there as well before accelerate is imported.
        try:
            import numpy._core as _np_core

            _np_core.multiarray = _multiarray
            sys.modules["numpy._core.multiarray"] = _multiarray
        except Exception:
            pass
    except Exception:
        pass

    candidates = []
    configured = os.environ.get("KIMODO_MODEL_SITE_PACKAGES", "").strip()
    if configured:
        candidates.append(configured)
    candidates.append("/data/local-data/data/conda_envs/patch-policy-ddt/lib/python3.10/site-packages")
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_dir() and str(path) not in sys.path:
            # The SIMPLE venv may already contain an incompatible newer
            # huggingface_hub.  Put the explicitly selected training overlay
            # ahead of that site-packages directory so its transformers,
            # huggingface_hub, and safetensors are resolved as one set.  The
            # active torch/numpy modules were imported above and remain pinned
            # in sys.modules, so this does not replace SIMPLE's CUDA stack.
            site_index = next(
                (index for index, entry in enumerate(sys.path) if "site-packages" in entry),
                len(sys.path),
            )
            sys.path.insert(site_index, str(path))

    # Resolve transformers/safetensors after the active torch modules have
    # been pinned.  This also pulls a compatible huggingface_hub version from
    # the model environment when SIMPLE's optional dependency is newer.
    try:
        import safetensors  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Kimodo inference requires safetensors and transformers. Set "
            "KIMODO_MODEL_SITE_PACKAGES to the site-packages directory used "
            "for training."
        ) from exc


def _add_simple_dependency_paths() -> None:
    """Expose vendored SIMPLE extras that are optional in its uv profile."""
    for candidate in (
        SIMPLE_ROOT / "src",
        SIMPLE_ROOT / "third_party/curobo/src",
        SIMPLE_ROOT / "third_party/unitree_sdk2_python",
        SIMPLE_ROOT / "third_party",
    ):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _resolve_future_root_boundary(root_positions):
    """Estimate the root immediately before a predicted SIMPLE chunk."""
    import torch

    roots = torch.as_tensor(root_positions, dtype=torch.float32)
    if roots.ndim != 2 or roots.shape[1] != 3 or roots.shape[0] == 0:
        raise ValueError(
            f"Expected future root positions [T, 3], got {tuple(roots.shape)}"
        )
    if not torch.isfinite(roots).all():
        raise ValueError("Future root positions contain NaN or Inf")
    if roots.shape[0] == 1:
        return roots[0].clone()
    return (2.0 * roots[0] - roots[1]).clone()


class _SimpleRuntimeMixin:
    """SIMPLE-only policy layered on top of the unchanged Arena runtime."""

    def __init__(self, args):
        self.max_navigation_speed = float(
            getattr(args, "max_navigation_speed", 1.5)
        )
        if (
            not np.isfinite(self.max_navigation_speed)
            or self.max_navigation_speed <= 0
            ):
            raise ValueError(
                "max_navigation_speed must be finite and positive, got "
                f"{self.max_navigation_speed}"
            )
        self._init_simple_hand_fsm()
        super().__init__(args)

    def _init_simple_hand_fsm(self) -> None:
        """Initialize the SIMPLE-only per-hand transition controller.

        The method is deliberately lazy-safe: several transaction tests build
        the runtime with ``__new__`` and therefore bypass ``__init__``.
        """
        self.simple_hand_fsm_enabled = True
        self.simple_hand_state = np.zeros(2, dtype=bool)
        self.simple_hand_phase = ["stable_open", "stable_open"]
        self.simple_hand_hold_steps = np.zeros(2, dtype=np.int64)
        self.simple_hand_elapsed_steps = np.zeros(2, dtype=np.int64)
        self.simple_hand_stable_votes = np.zeros(2, dtype=np.int64)
        self.simple_hand_candidate_state = np.zeros(2, dtype=bool)
        self.simple_hand_candidate_count = np.zeros(2, dtype=np.int64)
        self.simple_hand_last_chunk_steps = 0
        self.simple_hand_transition_indices = np.full(2, -1, dtype=np.int64)

    def _ensure_simple_hand_fsm(self) -> None:
        if not hasattr(self, "simple_hand_fsm_enabled"):
            # Keep hand behavior unchanged for lightweight legacy test doubles
            # that do not opt into the SIMPLE evaluator controller.
            self.simple_hand_fsm_enabled = False
        if not hasattr(self, "simple_hand_state"):
            self.simple_hand_state = np.zeros(2, dtype=bool)
        if not hasattr(self, "simple_hand_phase"):
            self.simple_hand_phase = ["stable_open", "stable_open"]
        if not hasattr(self, "simple_hand_hold_steps"):
            self.simple_hand_hold_steps = np.zeros(2, dtype=np.int64)
        if not hasattr(self, "simple_hand_elapsed_steps"):
            self.simple_hand_elapsed_steps = np.zeros(2, dtype=np.int64)
        if not hasattr(self, "simple_hand_stable_votes"):
            self.simple_hand_stable_votes = np.zeros(2, dtype=np.int64)
        if not hasattr(self, "simple_hand_candidate_state"):
            self.simple_hand_candidate_state = np.zeros(2, dtype=bool)
        if not hasattr(self, "simple_hand_candidate_count"):
            self.simple_hand_candidate_count = np.zeros(2, dtype=np.int64)
        if not hasattr(self, "simple_hand_last_chunk_steps"):
            self.simple_hand_last_chunk_steps = 0
        if not hasattr(self, "simple_hand_transition_indices"):
            self.simple_hand_transition_indices = np.full(2, -1, dtype=np.int64)

    def _simple_hand_target_pose(self, side: int, closed: bool) -> np.ndarray:
        if not closed:
            return np.zeros(7, dtype=np.float32)
        # ``prepare_obs`` exposes MuJoCo/XML order (thumb, middle, index),
        # while the constants and WBC interface use (thumb, index, middle).
        close = LEFT_HAND_CLOSE if side == 0 else RIGHT_HAND_CLOSE
        return _wbc_hand_to_mjcf(close)

    def _simple_hand_motion_complete(
        self, side: int, closed: bool, hand_observation: dict[str, Any] | None
    ) -> bool:
        """Return whether a SIMPLE hand has reached/settled at its target.

        A grasped object can stop the fingers before the nominal close pose, so
        close completion accepts either substantial closure progress or a
        high actuator effort while the hand velocity is small.  The caller
        also enforces a minimum hold time and a finite timeout.
        """
        if not isinstance(hand_observation, dict):
            return False
        prefix = "left" if side == 0 else "right"
        q = hand_observation.get(f"{prefix}_hand_q")
        dq = hand_observation.get(f"{prefix}_hand_dq")
        if q is None or dq is None:
            return False
        try:
            q = np.asarray(q, dtype=np.float32).reshape(-1)
            dq = np.asarray(dq, dtype=np.float32).reshape(-1)
        except Exception:
            return False
        if q.size != 7 or dq.size != 7 or not np.isfinite(q).all() or not np.isfinite(dq).all():
            return False
        if float(np.max(np.abs(dq))) > _SIMPLE_HAND_VELOCITY_TOL:
            return False

        if not closed:
            return bool(float(np.max(np.abs(q))) <= _SIMPLE_HAND_OPEN_POSITION_TOL)

        open_pose = np.zeros(7, dtype=np.float32)
        close_pose = self._simple_hand_target_pose(side, closed=True)
        span = np.abs(close_pose - open_pose)
        progress = float(np.mean(np.abs(q - open_pose) / np.maximum(span, 1e-3)))
        tau = hand_observation.get(f"{prefix}_hand_tau_est")
        effort_high = False
        if tau is not None:
            try:
                tau = np.asarray(tau, dtype=np.float32).reshape(-1)
                limits = np.asarray([2.45] + [0.7] * 6, dtype=np.float32)
                effort_high = bool(
                    tau.size == 7
                    and np.isfinite(tau).all()
                    and np.max(np.abs(tau) / limits) >= _SIMPLE_HAND_TORQUE_FRACTION
                )
            except Exception:
                effort_high = False
        return bool(progress >= _SIMPLE_HAND_CLOSE_PROGRESS or effort_high)

    def _advance_simple_hand_phase(self, hand_observation: dict[str, Any] | None) -> None:
        """Advance timers and settle transitions using the latest proprioception."""
        self._ensure_simple_hand_fsm()
        elapsed = int(getattr(self, "simple_hand_last_chunk_steps", 0))
        if elapsed <= 0:
            return
        self.simple_hand_last_chunk_steps = 0
        for side, phase in enumerate(self.simple_hand_phase):
            if phase not in ("closing", "opening"):
                continue
            self.simple_hand_elapsed_steps[side] += elapsed
            self.simple_hand_hold_steps[side] = max(
                0, int(self.simple_hand_hold_steps[side]) - elapsed
            )
            min_done = self.simple_hand_hold_steps[side] == 0
            settled = self._simple_hand_motion_complete(
                side, bool(self.simple_hand_state[side]), hand_observation
            )
            if min_done and settled:
                self.simple_hand_stable_votes[side] += 1
            else:
                self.simple_hand_stable_votes[side] = 0
            control_fps = max(float(getattr(self, "control_fps", 50.0)), 1.0)
            max_steps = int(round(_SIMPLE_HAND_MAX_HOLD_SECONDS * control_fps))
            if self.simple_hand_stable_votes[side] >= 1 or self.simple_hand_elapsed_steps[side] >= max_steps:
                self.simple_hand_phase[side] = (
                    "stable_closed" if self.simple_hand_state[side] else "stable_open"
                )
                self.simple_hand_stable_votes[side] = 0
                self.simple_hand_candidate_count[side] = 0

    def _resolve_simple_hand_prefix(self, source_hand: Any) -> tuple[Any, np.ndarray]:
        """Apply event locking to the executed source-rate hand prefix.

        Only this prefix is inspected.  Predictions in the unexecuted tail of
        a 50-frame diffusion chunk must never trigger a current close/open
        action.
        """
        import torch

        hand = torch.as_tensor(source_hand, dtype=torch.float32).cpu()
        if hand.ndim != 2 or hand.shape[1] != 2:
            raise ValueError(f"Expected source hand prefix [T,2], got {tuple(hand.shape)}")
        effective = torch.zeros_like(hand)
        transitions = np.full(2, -1, dtype=np.int64)
        confirm = _SIMPLE_HAND_EVENT_CONFIRM_FRAMES
        control_fps = max(float(getattr(self, "control_fps", 50.0)), 1.0)
        min_hold = max(1, int(round(_SIMPLE_HAND_MIN_HOLD_SECONDS * control_fps)))
        for side in range(2):
            phase = self.simple_hand_phase[side]
            current = bool(self.simple_hand_state[side])
            if phase in ("closing", "opening"):
                effective[:, side] = float(current)
                continue

            candidate_count = int(self.simple_hand_candidate_count[side])
            candidate_state = bool(self.simple_hand_candidate_state[side])
            transition_idx = -1
            new_state = current
            for frame_idx, value in enumerate(hand[:, side].tolist()):
                sample = bool(float(value) >= 0.5)
                if sample == current:
                    candidate_count = 0
                    candidate_state = current
                    continue
                if candidate_count == 0 or candidate_state != sample:
                    candidate_state = sample
                    candidate_count = 1
                else:
                    candidate_count += 1
                if candidate_count >= confirm:
                    transition_idx = frame_idx - confirm + 1
                    new_state = sample
                    break

            if transition_idx >= 0:
                self.simple_hand_state[side] = new_state
                self.simple_hand_phase[side] = (
                    "closing" if new_state else "opening"
                )
                self.simple_hand_hold_steps[side] = min_hold
                self.simple_hand_elapsed_steps[side] = 0
                self.simple_hand_stable_votes[side] = 0
                self.simple_hand_candidate_count[side] = 0
                transitions[side] = transition_idx
                effective[:transition_idx, side] = float(current)
                effective[transition_idx:, side] = float(new_state)
            else:
                self.simple_hand_candidate_state[side] = candidate_state
                self.simple_hand_candidate_count[side] = candidate_count
                effective[:, side] = float(current)
        self.simple_hand_transition_indices = transitions
        return effective, transitions

    @staticmethod
    def _snapshot_state(runtime: "_SimpleRuntimeMixin") -> dict[str, object]:
        """Copy mutable recurrent state so safety checks remain transactional."""
        runtime._ensure_simple_hand_fsm()
        return {
            "observation_state_history": np.array(
                runtime.observation_state_history, dtype=np.float32, copy=True
            ),
            "history_motion": runtime.history_motion.clone(),
            "history_hand": runtime.history_hand.clone(),
            "previous_local_rot_mat": (
                None
                if runtime.previous_local_rot_mat is None
                else runtime.previous_local_rot_mat.clone()
            ),
            "previous_root_position": (
                None
                if runtime.previous_root_position is None
                else runtime.previous_root_position.clone()
            ),
            "rtc_motion_tail": (
                None
                if runtime.rtc_motion_tail is None
                else runtime.rtc_motion_tail.clone()
            ),
            "rtc_hand_tail": (
                None
                if runtime.rtc_hand_tail is None
                else runtime.rtc_hand_tail.clone()
            ),
            "rtc_task_id": runtime.rtc_task_id,
            "inference_index": runtime.inference_index,
            "simple_hand_fsm_enabled": bool(runtime.simple_hand_fsm_enabled),
            "simple_hand_state": runtime.simple_hand_state.copy(),
            "simple_hand_phase": list(runtime.simple_hand_phase),
            "simple_hand_hold_steps": runtime.simple_hand_hold_steps.copy(),
            "simple_hand_elapsed_steps": runtime.simple_hand_elapsed_steps.copy(),
            "simple_hand_stable_votes": runtime.simple_hand_stable_votes.copy(),
            "simple_hand_candidate_state": runtime.simple_hand_candidate_state.copy(),
            "simple_hand_candidate_count": runtime.simple_hand_candidate_count.copy(),
            "simple_hand_last_chunk_steps": int(runtime.simple_hand_last_chunk_steps),
            "simple_hand_transition_indices": runtime.simple_hand_transition_indices.copy(),
        }

    def _restore_state(self, snapshot: dict[str, object]) -> None:
        self.observation_state_history = snapshot["observation_state_history"]
        self.history_motion = snapshot["history_motion"]
        self.history_hand = snapshot["history_hand"]
        self.previous_local_rot_mat = snapshot["previous_local_rot_mat"]
        self.previous_root_position = snapshot["previous_root_position"]
        self.rtc_motion_tail = snapshot["rtc_motion_tail"]
        self.rtc_hand_tail = snapshot["rtc_hand_tail"]
        self.rtc_task_id = snapshot["rtc_task_id"]
        self.inference_index = snapshot["inference_index"]
        self.simple_hand_fsm_enabled = bool(snapshot["simple_hand_fsm_enabled"])
        self.simple_hand_state = snapshot["simple_hand_state"].copy()
        self.simple_hand_phase = list(snapshot["simple_hand_phase"])
        self.simple_hand_hold_steps = snapshot["simple_hand_hold_steps"].copy()
        self.simple_hand_elapsed_steps = snapshot["simple_hand_elapsed_steps"].copy()
        self.simple_hand_stable_votes = snapshot["simple_hand_stable_votes"].copy()
        self.simple_hand_candidate_state = snapshot["simple_hand_candidate_state"].copy()
        self.simple_hand_candidate_count = snapshot["simple_hand_candidate_count"].copy()
        self.simple_hand_last_chunk_steps = int(snapshot["simple_hand_last_chunk_steps"])
        self.simple_hand_transition_indices = snapshot["simple_hand_transition_indices"].copy()

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._init_simple_hand_fsm()

    def infer(self, payload: dict) -> np.ndarray:
        """Run inference with SIMPLE's root-boundary policy and diagnostics.

        The Arena base class is deliberately not modified.  We adapt only the
        model output seen by that class: SIMPLE's state omits planar root
        position, so the model's reconstructed history root is an arbitrary
        gauge.  The wrapper replaces that field with a one-step future-root
        extrapolation, then restores the model method after the transaction.
        """
        import torch

        original_predict_future = self.model.predict_future
        captured: dict[str, torch.Tensor] = {}

        def predict_future_for_simple(*args, **kwargs):
            # RTC remains useful for the body, but blending a stale hand tail
            # would bypass the event controller below.  The hand branch is a
            # discrete state machine, so always let it start from the current
            # recurrent state and apply its own prefix override.
            if getattr(self, "simple_hand_fsm_enabled", False):
                kwargs = dict(kwargs)
                kwargs["rtc_hand_reference"] = None
            output = original_predict_future(*args, **kwargs)
            if not isinstance(output, dict):
                return output
            roots = output.get("root_positions")
            if roots is None:
                raise ValueError("Model output is missing root_positions")
            roots_cpu = torch.as_tensor(roots).detach().float().cpu()
            boundary = _resolve_future_root_boundary(roots_cpu)
            captured["root_positions"] = roots_cpu
            # Keep the hand branch observable in SIMPLE runs.  The public
            # 40-D action only contains the thresholded binary state, so
            # capturing both the continuous denoiser output and the binary
            # prefix here lets us distinguish a model prediction problem from
            # a downstream WBC/robot-joint mapping problem.
            hand_binary = output.get("hand_binary")
            if hand_binary is not None:
                captured["hand_binary"] = (
                    torch.as_tensor(hand_binary).detach().float().cpu()
                )
            hand_clean = output.get("hand_clean")
            if hand_clean is not None:
                captured["hand_clean"] = (
                    torch.as_tensor(hand_clean).detach().float().cpu()
                )
            local_rot_mats = output.get("local_rot_mats")
            if local_rot_mats is not None:
                captured["local_rot_mats"] = (
                    torch.as_tensor(local_rot_mats).detach().float().cpu()
                )
            adapted = dict(output)
            adapted["history_last_root_position"] = boundary
            return adapted

        # Assigning an instance attribute avoids changing the Arena module or
        # its class-level method.  The base runtime calls this while holding its
        # own lock; restore it in ``finally`` even when prediction fails.
        self.model.predict_future = predict_future_for_simple
        snapshot = self._snapshot_state(self)
        self._advance_simple_hand_phase(payload.get("hand_observation"))
        previous_local_rot_mat = snapshot["previous_local_rot_mat"]
        state_history_segment = payload.get("observation", {}).get("state_history")
        try:
            action_chunk = super().infer(payload)
        except Exception:
            # ``_advance_simple_hand_phase`` is part of the same transaction as
            # model inference; restore it if prediction or decoding fails.
            self._restore_state(snapshot)
            raise
        finally:
            self.model.predict_future = original_predict_future

        hand_transition_indices = np.full(2, -1, dtype=np.int64)
        full_hand_binary = captured.get("hand_binary")
        try:
            if (
                getattr(self, "simple_hand_fsm_enabled", False)
                and full_hand_binary is not None
            ):
                if full_hand_binary.ndim == 3 and full_hand_binary.shape[0] == 1:
                    full_hand_binary = full_hand_binary[0]
                if full_hand_binary.ndim == 2 and full_hand_binary.shape[-1] == 2:
                    source_hand = full_hand_binary[: self.execution_frames].float().cpu()
                    effective_hand, hand_transition_indices = self._resolve_simple_hand_prefix(
                        source_hand
                    )
                    previous_hand = (
                        snapshot["history_hand"][0, -1].detach().float().cpu()
                        if snapshot["history_hand"].shape[1] > 0
                        else torch.zeros(2, dtype=torch.float32)
                    )
                    from motion.g1_reference import resample_hand_binary_chunk

                    control_hand = resample_hand_binary_chunk(
                        effective_hand,
                        source_fps=float(self.model.fps),
                        target_fps=float(self.control_fps),
                        previous_hand_binary=previous_hand,
                    )
                    action_chunk = np.asarray(action_chunk, dtype=np.float32).copy()
                    if control_hand.shape[0] != action_chunk.shape[0]:
                        raise RuntimeError(
                            "SIMPLE hand controller produced a chunk with a different length: "
                            f"{control_hand.shape[0]} vs {action_chunk.shape[0]}"
                        )
                    action_chunk[:, 38:40] = control_hand.numpy()
                    # The model must be conditioned on the state that was actually
                    # sent to SIMPLE, not the unmodified diffusion prefix.
                    self.history_hand = torch.cat(
                        (
                            snapshot["history_hand"],
                            effective_hand.to(snapshot["history_hand"].device).unsqueeze(0),
                        ),
                        dim=1,
                    )[:, -self.model.config.action_history :]
                    # Do not let the base runtime's raw hand tail re-enter through
                    # RTC on the next replan.  Motion RTC remains untouched.
                    self.rtc_hand_tail = None
        except Exception:
            self._restore_state(snapshot)
            raise

        action_chunk_tensor = torch.as_tensor(action_chunk, dtype=torch.float32)
        navigation_speed = torch.linalg.vector_norm(
            action_chunk_tensor[:, :2], dim=1
        ) * float(self.control_fps)
        max_navigation_speed = float(navigation_speed.max().item())
        first_navigation_speed = float(navigation_speed[0].item())
        if max_navigation_speed > self.max_navigation_speed:
            self._restore_state(snapshot)
            raise ValueError(
                "Predicted navigation speed exceeds safety threshold: "
                f"first={first_navigation_speed:.3f} m/s "
                f"max={max_navigation_speed:.3f} m/s "
                f"threshold={self.max_navigation_speed:.3f} m/s"
            )

        source_root_positions = captured.get("root_positions")
        if source_root_positions is None:
            raise RuntimeError("SIMPLE model output did not expose root_positions")
        source_local_rot_mats = captured.get("local_rot_mats")
        if source_local_rot_mats is None:
            max_joint_boundary_delta = 0.0
        elif previous_local_rot_mat is None:
            max_joint_boundary_delta = 0.0
        else:
            max_joint_boundary_delta = float(
                (
                    source_local_rot_mats[0].float()
                    - torch.as_tensor(previous_local_rot_mat).float()
                )
                .abs()
                .max()
                .item()
            )
        height_values = source_root_positions[:, 1].float()
        source_hand_binary = captured.get("hand_binary")
        source_hand_clean = captured.get("hand_clean")
        hand_diag = ""
        if source_hand_binary is not None:
            binary = source_hand_binary
            if binary.ndim == 3 and binary.shape[0] == 1:
                binary = binary[0]
            if binary.ndim == 2 and binary.shape[-1] == 2:
                hand_diag = (
                    f" model_hand_active={binary.sum(dim=0).to(torch.int64).tolist()}"
                    f"/{binary.shape[0]}"
                )
        if source_hand_clean is not None:
            clean = source_hand_clean
            if clean.ndim == 3 and clean.shape[0] == 1:
                clean = clean[0]
            if clean.ndim == 2 and clean.shape[-1] == 2:
                hand_diag += (
                    f" model_hand_clean_mean="
                    f"{clean.mean(dim=0).tolist()}"
                    f" model_hand_clean_min="
                    f"{clean.min(dim=0).values.tolist()}"
                    f" model_hand_clean_max="
                    f"{clean.max(dim=0).values.tolist()}"
                )
        encoded_hand = action_chunk_tensor[:, 38:40]
        if encoded_hand.numel():
            hand_diag += (
                f" control_hand_active="
                f"{encoded_hand.sum(dim=0).to(torch.int64).tolist()}"
                f"/{encoded_hand.shape[0]}"
            )
        if getattr(self, "simple_hand_fsm_enabled", False):
            hand_diag += (
                f" hand_state={self.simple_hand_state.astype(int).tolist()}"
                f" hand_phase={list(self.simple_hand_phase)}"
                f" hand_transition_idx={hand_transition_indices.tolist()}"
                f" hand_hold_steps={self.simple_hand_hold_steps.tolist()}"
            )
        if state_history_segment is None:
            state_segment_frames = 0
        else:
            segment = np.asarray(state_history_segment)
            state_segment_frames = 1 if segment.ndim == 1 else int(segment.shape[0])
        print(
            "[chunk_diagnostics] "
            f"state_segment_frames={state_segment_frames} "
            f"first_navigation_speed={first_navigation_speed:.3f}m/s "
            f"max_navigation_speed={max_navigation_speed:.3f}m/s "
            f"height_range=[{float(height_values.min()):.4f},{float(height_values.max()):.4f}] "
            f"max_joint_boundary_delta={max_joint_boundary_delta:.6f}"
            f"{hand_diag}",
            flush=True,
        )
        if getattr(self, "simple_hand_fsm_enabled", False):
            # This chunk is now committed and will have elapsed by the next
            # replan, where ``_advance_simple_hand_phase`` consumes it.
            self.simple_hand_last_chunk_steps = int(action_chunk.shape[0])
        return action_chunk


def _make_simple_runtime_class(arena_server):
    """Create the SIMPLE runtime class without importing model code at --help."""
    return type(
        "KimodoSimpleHumanoidArenaRuntime",
        (_SimpleRuntimeMixin, arena_server.KimodoHumanoidArenaRuntime),
        {},
    )


class KimodoSimpleRuntime:
    """Load the existing Kimodo runtime while accepting Simple text caches."""

    def __init__(self, args: argparse.Namespace):
        _add_model_dependency_paths()
        from evaluation import humanoidarena_server as arena_server

        simple_runtime_class = _make_simple_runtime_class(arena_server)

        original_loader = arena_server._load_text_embedding_cache

        def load_cache(path):
            # The implementation is intentionally local so HumanoidArena's
            # strict source check remains unchanged for its original server.
            cache_path = Path(path).expanduser().resolve()
            files = sorted(cache_path.glob("*.pt")) if cache_path.is_dir() else [cache_path]
            if not files or not all(file.is_file() for file in files):
                raise FileNotFoundError(f"Text embedding cache does not exist: {cache_path}")
            embeddings, instructions, aliases = {}, {}, {}
            for file_path in files:
                import torch

                payload = torch.load(file_path, map_location="cpu", weights_only=True)
                task_id = str(payload.get("task_id", "")).strip()
                task_name = str(payload.get("task_name", "")).strip()
                instruction = str(payload.get("instruction", "")).strip()
                if not task_id or not instruction or "embedding" not in payload:
                    raise ValueError(f"Invalid Simple text cache entry: {file_path}")
                embedding = arena_server._validated_text_embedding(
                    payload["embedding"], cache_path=file_path, task_id=task_id
                )
                embeddings[task_id] = embedding
                instructions[task_id] = instruction
                aliases[task_id] = task_id
                if task_name:
                    aliases[task_name] = task_id
            if not embeddings:
                raise RuntimeError(f"No text embeddings found in {cache_path}")
            return embeddings, instructions, aliases

        arena_server._load_text_embedding_cache = load_cache
        try:
            self.runtime = simple_runtime_class(_runtime_args(args))
        finally:
            arena_server._load_text_embedding_cache = original_loader
        self.lock = threading.Lock()

    def reset(self, seed: int | None = None) -> None:
        self.runtime.reset(seed)

    def infer40(
        self,
        task: str,
        image: np.ndarray,
        state_history: np.ndarray,
        hand_observation: dict[str, Any] | None = None,
    ) -> np.ndarray:
        image = np.ascontiguousarray(image, dtype=np.uint8)
        state_history = np.asarray(state_history, dtype=np.float32)
        if state_history.ndim == 1:
            state_history = state_history.reshape(1, -1)
        if state_history.ndim != 2 or state_history.shape[1] != 64:
            raise ValueError(f"state_history must have shape (T, 64), got {state_history.shape}")
        payload = {
            "task": str(task).removeprefix("simple/"),
            "observation": {
                "images": {
                    "front": {
                        "shape": list(image.shape),
                        "dtype": str(image.dtype),
                        "data_b64": base64.b64encode(image.tobytes()).decode("ascii"),
                    }
                },
                "state_history": state_history.tolist(),
            },
        }
        if hand_observation is not None:
            # Keep the hand telemetry outside the 64-D model state.  It is
            # consumed only by the SIMPLE-side transition controller and is
            # intentionally not fed into the checkpoint.
            payload["hand_observation"] = {
                key: np.asarray(value, dtype=np.float32).reshape(-1).tolist()
                for key, value in hand_observation.items()
                if value is not None
            }
        with self.lock:
            return np.asarray(self.runtime.infer(payload), dtype=np.float32)


def _task_from_instruction(instruction: str, runtime: KimodoSimpleRuntime) -> str:
    text = str(instruction or "").strip()
    for task_id, cached_instruction in runtime.runtime.task_instructions.items():
        if cached_instruction.strip().lower() == text.lower():
            return task_id
    raise KeyError(
        f"No Simple text cache matches instruction {text!r}; pass a task name or regenerate data/cache/Simple"
    )


def _load_simple_eval_configs(data_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load the fixed environment configurations from a SIMPLE eval split.

    Official SIMPLE evaluation stores one serialized ``environment_config`` per
    episode in ``meta/episodes.jsonl``.  Reusing those configs is important:
    the Level 0/1/2 benchmark is defined by these fixed scenes, not by drawing
    a fresh random layout at reset time.
    """
    root = Path(data_dir).expanduser().resolve()
    metadata_path = root / "meta" / "episodes.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"SIMPLE eval data must contain {metadata_path}"
        )

    configs: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        metadata_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            episode = json.loads(line)
            raw_config = episode.get("environment_config")
            if isinstance(raw_config, str):
                config = json.loads(raw_config)
            elif isinstance(raw_config, dict):
                config = raw_config
            else:
                raise ValueError("missing environment_config")
        except Exception as exc:
            raise ValueError(
                f"Invalid environment_config in {metadata_path}:{line_number}"
            ) from exc
        if not isinstance(config, dict) or "uid" not in config or "dr_state_dict" not in config:
            raise ValueError(
                f"Environment config in {metadata_path}:{line_number} is not a SIMPLE task state"
            )

        # Match SIMPLE's LeRobot loader.  Older exports refer to the scene by
        # its HSSD numeric id; the local scene archive exposes the corresponding
        # stable alias as ``scene3``.
        state = config.get("dr_state_dict")
        if isinstance(state, dict) and isinstance(state.get("scene"), dict):
            uid = state["scene"].get("uid")
            if isinstance(uid, str):
                state["scene"]["uid"] = uid.replace("102344280", "scene3")
        configs.append(config)

    if not configs:
        raise ValueError(f"No episode configs found in {metadata_path}")
    return configs


def _extract_64_state(payload: dict[str, Any]) -> np.ndarray:
    state = payload.get("state", {})
    candidates = []
    if isinstance(state, dict):
        candidates.extend((state.get("state_history"), state.get("observation_state_history"), state.get("states")))
    candidates.extend((payload.get("state_history"), payload.get("observation_state_history")))
    for candidate in candidates:
        if candidate is None:
            continue
        array = np.asarray(candidate, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim == 2 and array.shape[1] == 64:
            return array
    raise ValueError(
        "SIMPLE /act requires state_dict['state_history'] with shape (T,64). "
        "The legacy 32-D psi0 payload cannot be losslessly adapted to this checkpoint."
    )


def _extract_act_image(payload: dict[str, Any]) -> np.ndarray:
    image_dict = payload.get("image") or payload.get("images") or {}
    if isinstance(image_dict, np.ndarray):
        return _simple_image({"front": image_dict})
    if isinstance(image_dict, dict):
        for value in image_dict.values():
            if isinstance(value, np.ndarray):
                return _simple_image({"front": value})
    raise ValueError("SIMPLE /act payload has no numpy RGB image")


class _HttpHandler:
    runtime: KimodoSimpleRuntime | None = None


def serve(args: argparse.Namespace) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    service_runtime = KimodoSimpleRuntime(args)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(_numpy_serialize(payload)).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in {"", "/health", "/healthz"}:
                self._json(200, {"status": "ok", "protocol": "simple", "state_dim": 64, "action_dim": 40})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                size = int(self.headers.get("Content-Length", "0"))
                payload = _numpy_deserialize(json.loads(self.rfile.read(size) or b"{}"))
                path = self.path.rstrip("/")
                if path == "/reset":
                    service_runtime.reset(payload.get("seed"))
                    self._json(200, {"status": "ok"})
                    return
                if path == "/infer":
                    self._json(200, {"action_chunk": service_runtime.runtime.infer(payload).tolist()})
                    return
                if path == "/act":
                    state_history = _extract_64_state(payload)
                    image = _extract_act_image(payload)
                    task = str(payload.get("task", "")).strip() or _task_from_instruction(
                        payload.get("instruction", ""), service_runtime
                    )
                    actions40 = service_runtime.infer40(task, image, state_history)
                    actions36 = arena_action_to_simple(actions40, args.control_fps)
                    self._json(200, {"action": actions36, "err": 0.0, "traj_image": np.zeros((1, 1, 3), dtype=np.uint8)})
                    return
                self._json(404, {"error": "not found"})
            except Exception as exc:  # keep the client-facing error structured
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt: str, *values: Any) -> None:
            print(f"[{self.log_date_time_string()}] {fmt % values}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving SIMPLE Kimodo policy on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


@dataclass
class _EpisodeContext:
    first_heading: np.ndarray | None = None
    observation: dict[str, Any] | None = None
    info: dict[str, Any] | None = None
    state_buffer: list[np.ndarray] = field(default_factory=list)


def _make_action_from_40(action: np.ndarray, robot: Any, current_height: float) -> Any:
    """Fallback command for non-Sonic G1Wholebody environments."""
    from simple.core.action import ActionCmd

    action = np.asarray(action, dtype=np.float32).reshape(40)
    canonical_index = {name: i for i, name in enumerate(CANONICAL_NAMES)}
    q = action[9:38]
    target_qpos = {name: float(q[canonical_index[name]]) for name in CANONICAL_NAMES}
    left = LEFT_HAND_CLOSE if action[38] >= 0.5 else np.zeros(7, dtype=np.float32)
    right = RIGHT_HAND_CLOSE if action[39] >= 0.5 else np.zeros(7, dtype=np.float32)
    target_qpos.update(dict(zip(LEFT_HAND_NAMES, left)))
    target_qpos.update(dict(zip(RIGHT_HAND_NAMES, right)))
    yaw = float(np.arctan2(action[5], action[3]))
    command = [float(action[0] * 50), yaw, float(action[1] * 50), float(action[2] - current_height), yaw, 0.0, 0.0, 0.0]
    waist = {
        "waist_yaw_joint": target_qpos["waist_yaw_joint"],
        "waist_roll_joint": target_qpos["waist_roll_joint"],
        "waist_pitch_joint": target_qpos["waist_pitch_joint"],
    }
    return ActionCmd("eval_move_actuators", target_qpos=target_qpos, action_command=command, waist_qpos=waist)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Run SIMPLE's gym/reset/step/video loop with a local Kimodo policy."""
    if str(args.task).startswith("simple/"):
        env_id = str(args.task)
        task_name = env_id.removeprefix("simple/")
    else:
        task_name = str(args.task)
        env_id = f"simple/{task_name}"

    # Importing SIMPLE is deliberately delayed so --help and --dry-run work in
    # lightweight environments without Isaac Sim libraries.  Load Kimodo first
    # so its training environment selects a matching torch/torchvision pair
    # before SIMPLE's optional robotics imports touch torch.
    runtime = KimodoSimpleRuntime(args)
    if str(SIMPLE_ROOT) not in sys.path:
        sys.path.insert(0, str(SIMPLE_ROOT))
    _add_simple_dependency_paths()
    import gymnasium as gym
    import simple.envs as _  # noqa: F401
    from simple.envs.wrappers import VideoRecorder
    output_root = Path(args.results_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds]
    episode_index = 0
    results: list[dict[str, Any]] = []

    eval_configs: list[dict[str, Any]] | None = None
    if args.eval_data_dir:
        eval_configs = _load_simple_eval_configs(args.eval_data_dir)
        requested = int(args.num_episodes)
        if requested <= 0:
            requested = len(eval_configs)
        if requested > len(eval_configs):
            raise ValueError(
                f"Requested {requested} episodes from {args.eval_data_dir}, "
                f"but only {len(eval_configs)} environment configs are available"
            )
        eval_configs = eval_configs[:requested]

    env_kwargs = {
        "sim_mode": args.sim_mode,
        "headless": args.headless,
        "render_hz": args.render_hz,
    }
    # Registered SIMPLE teleop environments require this positional config;
    # passing it to MP environments is harmless because their constructors
    # forward unknown kwargs to the base task.
    env_kwargs["sonic_config"] = _make_sonic_config()
    env_kwargs["physics_dt"] = float(env_kwargs["sonic_config"]["SIMULATE_DT"])
    env = gym.make(env_id, **env_kwargs)
    task = env.unwrapped.task
    robot = task.robot
    # Match SIMPLE's official evaluator: an omitted limit uses the task's
    # metadata value.  Apply TimeLimit after constructing the raw environment
    # so the limit is explicit and can be reset after Sonic stabilization.
    episode_limit = args.max_episode_steps
    if episode_limit is None:
        episode_limit = task.metadata.get("max_episode_steps")
    if episode_limit is None:
        episode_limit = 15000
    episode_limit = int(episode_limit)
    if episode_limit <= 0:
        raise ValueError(f"max_episode_steps must be positive, got {episode_limit}")
    from gymnasium.wrappers import TimeLimit

    env = TimeLimit(env, max_episode_steps=episode_limit)

    requested_episodes = int(args.num_episodes)
    if eval_configs is not None:
        total_episodes = len(eval_configs)
    else:
        total_episodes = requested_episodes if requested_episodes > 0 else len(seeds) * int(args.repeats)
    for episode_offset in range(total_episodes):
            seed = seeds[episode_offset % len(seeds)]
            repeat = episode_offset // len(seeds)
            episode_seed = seed + repeat
            runtime.reset(episode_seed)
            context = _EpisodeContext()
            reset_options = None
            if eval_configs is not None:
                # The state dict carries the exact Level 0/1/2 scene, object,
                # lighting, material and spatial configuration from SIMPLE's
                # official eval export.
                reset_options = {"state_dict": eval_configs[episode_offset]}
            observation, info = env.reset(seed=episode_seed, options=reset_options)
            # SIMPLE's teleop environments expose full proprioception in info.
            # Keep a direct fallback for environments whose info omits it.
            if not isinstance(info.get("proprio"), dict):
                proprio = _robot_proprio(robot)
                if proprio is None:
                    raise RuntimeError("SIMPLE environment did not expose info['proprio'] and robot has no prepare_obs()")
                info = dict(info, proprio=proprio)
            context.info = info
            context.observation = observation

            if getattr(robot, "uid", "") == "g1_sonic":
                # Psi0DecoupledWbcAgent is SIMPLE's established adapter for a
                # 36-D high-level policy command.  It both requests a fresh
                # action chunk when its queue is empty and converts each
                # command through the decoupled WBC.  The plain
                # SonicDecoupledWbcAgent only consumes an already-populated
                # queue, so using it directly would stop at the first step.
                from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent

                class SimplePsi0DecoupledWbcAgent(Psi0DecoupledWbcAgent):
                    """SIMPLE-only WBC adapter with explicit hand-order conversion.

                    ``G1Sonic.prepare_obs`` exposes MuJoCo/XML order
                    (thumb, middle, index), while the decoupled-WBC model and
                    ``from_psi0_upper_joints`` use named natural order
                    (thumb, index, middle).  Keep this conversion local to
                    the SIMPLE evaluator; Arena's runtime is untouched.
                    """

                    _HAND_KEYS = (
                        "left_hand_q", "right_hand_q",
                        "left_hand_dq", "right_hand_dq",
                        "left_hand_ddq", "right_hand_ddq",
                        "left_hand_tau_est", "right_hand_tau_est",
                    )

                    def _build_wbc_observation(self, sim_obs: dict) -> dict:
                        adapted = dict(sim_obs)
                        for key in self._HAND_KEYS:
                            value = adapted.get(key)
                            if value is not None:
                                adapted[key] = _mjcf_hand_to_wbc(value)
                        return super()._build_wbc_observation(adapted)

                    @staticmethod
                    def _to_mjcf_action(action_cmd):
                        if action_cmd is None or action_cmd.type != "decoupled_wbc":
                            return action_cmd
                        for key in ("left_hand_q", "right_hand_q"):
                            value = action_cmd[key]
                            if value is not None:
                                action_cmd.parameters[key] = _wbc_hand_to_mjcf(value)
                        return action_cmd

                    def get_stabilize_action(self, observation):
                        return self._to_mjcf_action(
                            super().get_stabilize_action(observation)
                        )

                    def get_action(self, observation, *args, **kwargs):
                        return self._to_mjcf_action(
                            super().get_action(observation, *args, **kwargs)
                        )

                class LocalClient:
                    def query_action(self, *unused, **kwargs):
                        assert context.observation is not None and context.info is not None
                        proprio = context.info["proprio"]
                        if not context.state_buffer:
                            _state, context.first_heading = append_state_history(
                                context.state_buffer,
                                proprio,
                                context.first_heading,
                            )
                        state_history = np.stack(context.state_buffer, axis=0)
                        image = _simple_image(context.observation)
                        hand_observation = {
                            key: proprio.get(key)
                            for key in (
                                "left_hand_q", "right_hand_q",
                                "left_hand_dq", "right_hand_dq",
                                "left_hand_tau_est", "right_hand_tau_est",
                            )
                            if proprio.get(key) is not None
                        }
                        actions40 = runtime.infer40(
                            task_name,
                            image,
                            state_history,
                            hand_observation=hand_observation,
                        )
                        # The segment has been consumed by the runtime. Future
                        # 50 Hz frames are collected after each env.step.
                        context.state_buffer.clear()
                        actions36 = arena_action_to_simple(
                            actions40,
                            args.control_fps,
                        )
                        return actions36, 0.0, None

                # Host/port are unused after replacing ``client`` below, but
                # keeping the normal constructor preserves SIMPLE's exact WBC
                # setup and reset semantics.
                agent = SimplePsi0DecoupledWbcAgent(
                    robot,
                    "127.0.0.1",
                    0,
                    sonic_config=robot.sonic_config,
                )
                agent.client = LocalClient()
                agent._wbc_policy.lower_body_policy.use_policy_action = True

                # Match SIMPLE's evaluator pre-roll: drive the Sonic WBC to
                # its default pose until the robot's velocity-based
                # stabilization latch engages (at most five control seconds).
                # The generic SIMPLE stand wrapper emits ``loco_command``,
                # which is not a valid G1Sonic action type.
                # Reset before warm-up, as SIMPLE's official evaluator does;
                # this arms the WBC's smooth default-pose interpolation.
                agent.reset()
                stabilize_steps = 0
                while not robot.stabilized and stabilize_steps < 300:
                    observation, _reward, _term, _trunc, info = env.step(
                        agent.get_stabilize_action(observation)
                    )
                    stabilize_steps += 1
                print(f"Robot stabilized after {stabilize_steps} simulation steps.", flush=True)
                if hasattr(env.unwrapped, "step_count"):
                    env.unwrapped.step_count = 0
                if hasattr(env.unwrapped, "_success"):
                    env.unwrapped._success = False
                _reset_episode_step_counters(env)
                # Warm-up advances both image and proprioception.  The first
                # model request must use that post-stabilization observation,
                # otherwise the robot and policy are evaluated at different
                # states.
                context.observation = observation
                if not isinstance(info.get("proprio"), dict):
                    proprio = _robot_proprio(robot)
                    if proprio is None:
                        raise RuntimeError(
                            "SIMPLE environment did not expose usable robot proprioception"
                        )
                    info = dict(info, proprio=proprio)
                context.info = info
            else:
                agent = None

            video_env = env
            recorder = None
            if args.save_video:
                recorder = VideoRecorder(
                    env=env,
                    video_folder=str(output_root),
                    name_prefix=f"episode_{episode_index:06d}",
                    framerate=args.render_hz,
                    write_png=False,
                )
                # The environment was reset before Sonic's stabilization
                # warm-up, so the wrapper cannot rely on reset() to create its
                # writers.  Start recording explicitly from the post-warm-up
                # observation that is also sent to the policy.
                recorder._init_writers(observation)
                video_env = recorder

            terminated = truncated = False
            termination_reason = None
            steps = 0
            started = time.perf_counter()
            while not (terminated or truncated):
                context.observation = observation
                context.info = info
                if agent is not None:
                    action = agent.get_action(observation, instruction=task.instruction, info=info)
                else:
                    proprio = info.get("proprio")
                    if not isinstance(proprio, dict):
                        proprio = _robot_proprio(robot)
                    if proprio is None:
                        raise RuntimeError("SIMPLE environment did not expose usable robot proprioception")
                    state, context.first_heading = build_arena_state(proprio, context.first_heading)
                    image = _simple_image(observation)
                    predicted = runtime.infer40(task_name, image, state)
                    current_height = float(np.asarray(proprio["floating_base_pose"])[2])
                    action = _make_action_from_40(predicted[0], robot, current_height)
                observation, reward, terminated, truncated, info = video_env.step(action)
                if terminated:
                    termination_reason = (
                        "task_success"
                        if bool(getattr(env.unwrapped, "_success", False))
                        else "terminated"
                    )
                elif truncated:
                    termination_reason = "timeout"
                if not isinstance(info.get("proprio"), dict):
                    proprio = _robot_proprio(robot)
                    if proprio is not None:
                        info = dict(info, proprio=proprio)
                if isinstance(info.get("proprio"), dict):
                    _state, context.first_heading = append_state_history(
                        context.state_buffer,
                        info["proprio"],
                        context.first_heading,
                    )
                steps += 1
                if steps >= episode_limit and not terminated:
                    truncated = True
                    termination_reason = "timeout"
            if recorder is not None:
                recorder.release()
                _export_ego_video(output_root / f"episode_{episode_index:06d}")
            success = bool(getattr(env.unwrapped, "_success", False))
            item = {
                "episode": episode_index,
                "eval_episode": episode_offset if eval_configs is not None else None,
                "seed": episode_seed,
                "success": success,
                "steps": steps,
                "max_episode_steps": episode_limit,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": termination_reason or "unknown",
                "duration_seconds": time.perf_counter() - started,
            }
            if args.level is not None:
                item["level"] = int(args.level)
            results.append(item)
            (output_root / f"episode_{episode_index:06d}.json").write_text(json.dumps(item, indent=2) + "\n")
            print(f"episode={episode_index} seed={episode_seed} success={success} steps={steps}", flush=True)
            episode_index += 1
    summary = {
        "task": task_name,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "level": int(args.level) if args.level is not None else None,
        "eval_data_dir": str(Path(args.eval_data_dir).expanduser().resolve()) if args.eval_data_dir else None,
        "official_eval_split": bool(eval_configs is not None),
        "max_episode_steps": episode_limit,
        "episodes": len(results),
        "successes": sum(bool(item["success"]) for item in results),
        "success_rate": (
            sum(bool(item["success"]) for item in results) / len(results) if results else 0.0
        ),
        "results": results,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Persist the result before Isaac Sim teardown.  Some Kit builds terminate
    # Python while closing the SimulationApp, which would otherwise discard a
    # completed episode summary and make the shell wrapper report a false
    # failure.
    env.close()
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kimodo SIMPLE policy server/evaluator")
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory containing config.json and training_state.pt")
    parser.add_argument("--text-embedding-cache", default=str(PROJECT_ROOT / "data/cache/Simple"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--execution-frames", type=int, default=0)
    parser.add_argument("--rtc", type=int, choices=(0, 1), default=1)
    parser.add_argument("--rtc-overlap-frames", type=int, default=12)
    parser.add_argument("--rtc-frozen-frames", type=int, default=1)
    parser.add_argument("--rtc-ramp-power", type=float, default=1.0)
    parser.add_argument("--control-fps", type=float, default=50.0)
    parser.add_argument(
        "--max-navigation-speed",
        type=float,
        default=1.5,
        help="Fail fast when predicted planar speed exceeds this m/s threshold",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=22085)
    parser.add_argument("--eval", action="store_true", help="run the local SIMPLE gym evaluator instead of serving HTTP")
    parser.add_argument("--task", default="G1WholebodyCloseDoorTeleop-v0")
    parser.add_argument("--sim-mode", default="mujoco_isaac")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render-hz", type=int, default=50)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="TimeLimit per episode; omitted uses the task metadata value",
    )
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--eval-data-dir",
        default="",
        help=(
            "SIMPLE fixed eval split containing meta/episodes.jsonl. "
            "When set, each episode is reset from its serialized environment_config."
        ),
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Official SIMPLE evaluation level associated with --eval-data-dir.",
    )
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "eval_results/simple"))
    parser.add_argument("--save-video", dest="save_video", action="store_true", default=True)
    parser.add_argument("--no-save-video", dest="save_video", action="store_false")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.eval:
        run_eval(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
