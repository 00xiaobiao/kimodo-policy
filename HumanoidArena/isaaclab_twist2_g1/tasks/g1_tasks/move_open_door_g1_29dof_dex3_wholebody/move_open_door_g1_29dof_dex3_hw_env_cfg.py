import os
import math
from typing import Any
import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
import numpy as np

from . import mdp
from common_env_objects import apply_deterministic_object_resets
from tasks.common_config import CameraPresets, G1RobotPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_open_door import OpenDoorSceneCfg
import numpy as np

from . import mdp

try:
    from isaaclab.managers import EventTermCfg
except ImportError:
    EventTermCfg = None

ROBOT_INIT_POS = (-1.6, 0.2, 0.8)
ROBOT_INIT_ROT = (0.70711, 0.0, 0.0, 0.70711)
DOOR_LEAF_JOINT_STATIC_FRICTION = 0.0
DOOR_LEAF_JOINT_DYNAMIC_FRICTION = 0.0
DOOR_LEAF_JOINT_VISCOUS_FRICTION = 0.0
DOOR_HANDLE_JOINT_STATIC_FRICTION = 2.0
DOOR_HANDLE_JOINT_DYNAMIC_FRICTION = 1.5
DOOR_HANDLE_JOINT_VISCOUS_FRICTION = 0.5
DOOR_LEAF_JOINT_REL_PATH = "E_leaf_2/RevoluteJoint_door001"
DOOR_HANDLE_JOINT_REL_PATH = "E_handle_4/RevoluteJoint"
DOOR_LEAF_BODY_REL_PATH = "E_leaf_2"
DOOR_HANDLE_BODY_REL_PATH = "E_handle_4"
DOOR_DC_ARTICULATION_REL_PATHS = ("", "E_bodyM1_1", "E_leaf_2")
DOOR_LEAF_DEFAULT_DRIVE = {
    "drive_type": "acceleration",
    "stiffness": 5.0,
    "damping": 20.0,
    "max_force": 5.0,
    "target_position": 0.0,
    "target_velocity": 0.0,
    "joint_friction": 0.0,
}

_OPEN_DOOR_DEBUG_PRIM_PATHS = (
    "/World/envs/env_0/Door",
    "/World/envs/env_0/Door/E_bodyM1_1",
    "/World/envs/env_0/Door/E_leaf_2",
    "/World/envs/env_0/Door/E_handle_4",
    "/World/envs/env_0/Door/E_handle_4/E_Component201_5",
    "/World/envs/env_0/Door/E_handle_4/E_Component3_6",
    "/World/envs/env_0/Door/E_handle_4/E_Component31_7",
    "/World/envs/env_0/Door/gate",
)
_OPEN_DOOR_DIAGNOSTIC_VERSION = "rigid_pose_v1"
_OPEN_DOOR_JOINT_SAMPLE_STEPS = (
    1,
    5,
    10,
    20,
    30,
    40,
    50,
    75,
    100,
    150,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    850,
    900,
    950,
    1000,
    1100,
    1200,
    1300,
    1400,
    1450,
    1500,
    1550,
    1600,
    1650,
    1700,
    1750,
    1800,
)
_OPEN_DOOR_EVENT_SMOKE_MODE = os.environ.get("OPEN_DOOR_EVENT_SMOKE_MODE", "startup").strip().lower()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in {"", "0", "false", "no", "off"}


def _env_flag_default(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[open_door_latch] invalid {name}={value!r}; using {default}")
        return default


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.3f}"


def _first_index(value) -> int | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().reshape(-1).tolist()
        elif hasattr(value, "reshape") and hasattr(value, "tolist"):
            value = value.reshape(-1).tolist()
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return int(value[0])
    try:
        return int(value)
    except Exception:
        return None


def _open_door_event_smoke(env, env_ids) -> None:
    if not _env_flag("OPEN_DOOR_EVENT_SMOKE_DEBUG"):
        return
    if isinstance(env_ids, torch.Tensor):
        env_ids_repr = env_ids.detach().cpu().reshape(-1).tolist()
    else:
        env_ids_repr = env_ids
    print(f"[open_door_event_smoke] triggered env_ids={env_ids_repr}")


def _open_door_latch_startup_event(env, env_ids) -> None:
    env_cfg = getattr(env, "cfg", None)
    setup = getattr(env_cfg, "setup_open_door_latch", None)
    if callable(setup):
        setup(env, env_ids=env_ids, reason="event_startup")


def _open_door_latch_reset_event(env, env_ids) -> None:
    env_cfg = getattr(env, "cfg", None)
    reset = getattr(env_cfg, "reset_open_door_latch", None)
    if callable(reset):
        reset(env, env_ids=env_ids, reason="event_reset")


def _open_door_latch_interval_event(env, env_ids) -> None:
    env_cfg = getattr(env, "cfg", None)
    poll = getattr(env_cfg, "apply_open_door_latch_interval", None)
    if callable(poll):
        poll(env, env_ids=env_ids, reason="event_interval")


def _xformable_local_matrix(xformable):
    result = xformable.GetLocalTransformation()
    if isinstance(result, tuple):
        return result[0]
    return result


def _matrix_cell(matrix, row: int, col: int) -> float:
    try:
        return float(matrix[row][col])
    except Exception:
        pass
    try:
        return float(matrix[row, col])
    except Exception:
        pass
    try:
        return float(matrix[row * 4 + col])
    except Exception:
        pass
    return float(matrix.GetRow(row)[col])


def _matrix_to_rows(matrix) -> list[list[float]]:
    return [[_matrix_cell(matrix, row, col) for col in range(4)] for row in range(4)]


def _quat_wxyz_from_matrix(matrix) -> np.ndarray:
    quat = matrix.ExtractRotationQuat()
    imag = quat.GetImaginary()
    return np.array(
        [
            float(quat.GetReal()),
            float(imag[0]),
            float(imag[1]),
            float(imag[2]),
        ],
        dtype=np.float64,
    )


def _quat_angle_deg_from_matrix(matrix) -> float:
    quat = _quat_wxyz_from_matrix(matrix)
    quat_norm = np.linalg.norm(quat)
    if quat_norm <= 1e-12:
        return 0.0
    quat = quat / quat_norm
    w = float(np.clip(quat[0], -1.0, 1.0))
    return float(math.degrees(2.0 * math.acos(abs(w))))


def _signed_axis_angle_deg_from_matrix(matrix, axis: str) -> float:
    axis = axis.upper()
    if axis == "X":
        radians = math.atan2(float(matrix[2][1]), float(matrix[1][1]))
    elif axis == "Y":
        radians = math.atan2(float(matrix[0][2]), float(matrix[0][0]))
    elif axis == "Z":
        radians = math.atan2(float(matrix[1][0]), float(matrix[0][0]))
    else:
        raise ValueError(f"unsupported rotation axis: {axis}")
    return float(math.degrees(radians))


def _prim_rotate_zyx_deg(prim) -> list[float] | None:
    attr = prim.GetAttribute("xformOp:rotateZYX")
    if not attr or not attr.IsValid():
        return None
    value = attr.Get()
    if value is None:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return None


def _first_target_path(prim, rel_name: str) -> str:
    rel = prim.GetRelationship(rel_name)
    if rel is None or not rel.IsValid():
        return ""
    targets = rel.GetTargets()
    if not targets:
        return ""
    return str(targets[0])


def _classify_revolute_joint(body0_path: str, body1_path: str) -> str:
    endpoints = {body0_path.split("/")[-1], body1_path.split("/")[-1]}
    if endpoints == {"E_bodyM1_1", "E_leaf_2"}:
        return "door_leaf_joint"
    if endpoints == {"E_leaf_2", "E_handle_4"}:
        return "handle_joint"
    return "other"


@configclass
class OpenDoorTerrainSceneCfg(OpenDoorSceneCfg):
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_wholebody(
        init_pos=ROBOT_INIT_POS,
        init_rot=ROBOT_INIT_ROT,
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=10,
        track_air_time=True,
        debug_vis=False,
    )

    front_camera = CameraPresets.g1_front_camera()


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_gipper_state = ObsTerm(func=mdp.get_robot_dex3_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    pass


@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward_open_door, weight=1.0)


@configclass
class EventCfg:
    if _env_flag("OPEN_DOOR_EVENT_SMOKE") and EventTermCfg is not None:
        if _OPEN_DOOR_EVENT_SMOKE_MODE in {"startup", "all"}:
            open_door_event_smoke_startup = EventTermCfg(
                func=_open_door_event_smoke,
                mode="startup",
            )
        if _OPEN_DOOR_EVENT_SMOKE_MODE in {"reset", "all"}:
            open_door_event_smoke_reset = EventTermCfg(
                func=_open_door_event_smoke,
                mode="reset",
            )
        if _OPEN_DOOR_EVENT_SMOKE_MODE in {"interval", "all"}:
            open_door_event_smoke_interval = EventTermCfg(
                func=_open_door_event_smoke,
                mode="interval",
                interval_range_s=(0.25, 0.25),
                is_global_time=False,
            )
    if _env_flag_default("OPEN_DOOR_LATCH_ENABLE", True) and not _env_flag("OPEN_DOOR_LATCH_DISABLE") and EventTermCfg is not None:
        open_door_latch_startup = EventTermCfg(
            func=_open_door_latch_startup_event,
            mode="startup",
        )
        open_door_latch_reset = EventTermCfg(
            func=_open_door_latch_reset_event,
            mode="reset",
        )
        open_door_latch_poll = EventTermCfg(
            func=_open_door_latch_interval_event,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            is_global_time=False,
        )


@configclass
class MoveOpenDoorG129Dex3WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    scene: OpenDoorTerrainSceneCfg = OpenDoorTerrainSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        # Allow YAML override via tasks/common_env_config/*.yaml.
        self.object_reset_seed_source = "env_seed"
        self.deterministic_object_resets = []
        self._replay_initial_env_state_active = False
        self._open_door_transform_debug_enabled = _env_flag("OPEN_DOOR_TRANSFORM_DEBUG")
        self._open_door_joint_debug_enabled = _env_flag("OPEN_DOOR_JOINT_DEBUG")
        self._open_door_joint_catalog_logged = False
        self._open_door_latch_enabled = _env_flag_default("OPEN_DOOR_LATCH_ENABLE", True) and not _env_flag(
            "OPEN_DOOR_LATCH_DISABLE"
        )
        self._open_door_latch_debug_enabled = _env_flag("OPEN_DOOR_LATCH_DEBUG")
        self._open_door_tensor_proxy_enabled = _env_flag_default("OPEN_DOOR_TENSOR_PROXY_ENABLE", True) and not _env_flag(
            "OPEN_DOOR_TENSOR_PROXY_DISABLE"
        )
        self._open_door_dc_angle_debug_enabled = _env_flag("OPEN_DOOR_DC_ANGLE_DEBUG")
        self._open_door_handle_unlock_angle_deg = _env_float("OPEN_DOOR_HANDLE_UNLOCK_ANGLE_DEG", -20.0)
        self._open_door_leaf_lock_drive_type = os.environ.get("OPEN_DOOR_LEAF_LOCK_DRIVE_TYPE", "force").strip()
        self._open_door_leaf_lock_stiffness = _env_float("OPEN_DOOR_LEAF_LOCK_STIFFNESS", 100000.0)
        self._open_door_leaf_lock_damping = _env_float("OPEN_DOOR_LEAF_LOCK_DAMPING", 50000.0)
        self._open_door_leaf_lock_max_force = _env_float("OPEN_DOOR_LEAF_LOCK_MAX_FORCE", 1000000.0)
        self._open_door_leaf_lock_joint_friction = _env_float("OPEN_DOOR_LEAF_LOCK_JOINT_FRICTION", 10000.0)
        self._open_door_latch_poll_log_interval = max(
            1,
            int(_env_float("OPEN_DOOR_LATCH_POLL_LOG_INTERVAL", 25.0)),
        )
        self._open_door_latch_defaults: dict[str, dict[str, Any]] = {}
        self._open_door_latch_articulation_defaults: dict[int, dict[str, Any]] = {}
        self._open_door_latch_dc_defaults: dict[int, dict[str, Any]] = {}
        self._open_door_latch_reference: dict[int, Any] = {}
        self._open_door_latch_tensor_proxy_reference: dict[int, float] = {}
        self._open_door_leaf_tensor_proxy_reference: dict[int, float] = {}
        self._open_door_latch_articulation_reference: dict[int, float] = {}
        self._open_door_latch_dc_reference: dict[int, float] = {}
        self._open_door_latch_angle_source_by_env: dict[int, str] = {}
        self._open_door_latch_unlocked_env_ids: set[int] = set()
        self._open_door_latch_missing_logged: set[str] = set()
        self._open_door_latch_poll_counter = 0
        self._open_door_dc = None
        self._open_door_dc_module = None
        self._open_door_dc_cache: dict[int, dict[str, Any]] = {}
        self._open_door_tensor_proxy_cache: dict[str, Any] = {}
        self._open_door_articulation_cache: dict[str, Any] = {}
        self._reset_open_door_runtime_debug_state()

        self.sim.dt = 0.005
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"

        self.event_manager = SimpleEventManager()
        self.event_manager.register(
            "reset_object_self",
            SimpleEvent(func=lambda env: self._reset_object_self(env)),
        )
        self.event_manager.register(
            "reset_all_self",
            SimpleEvent(func=lambda env: self._reset_all_self(env)),
        )

    def initialize_task_scene(self, env, args_cli=None):
        self._replay_initial_env_state_active = bool(getattr(args_cli, "replay_file", "")) if args_cli else False
        self._disable_overlapping_room_gate_collisions()
        self._configure_door_joint_physics(env)
        self._debug_transform_phase("initialize_task_scene")
        self._log_door_joint_catalog_once()

    def _reset_object_self(self, env):
        self._reset_open_door_runtime_debug_state()
        applied = apply_deterministic_object_resets(
            self,
            env,
            selected_record_names={"door"},
        )
        if applied:
            print("[object_reset] " + ", ".join(applied))
        self._debug_transform_phase("after_reset_object_self")

    def _reset_all_self(self, env):
        self._reset_open_door_runtime_debug_state()
        base_mdp.reset_scene_to_default(
            env,
            torch.arange(env.num_envs, device=env.device),
        )
        applied = apply_deterministic_object_resets(
            self,
            env,
            selected_record_names={"door"},
        )
        if applied:
            print("[object_reset] " + ", ".join(applied))
        self._debug_transform_phase("after_reset_all_self")

    def debug_after_startup_reset(self, env, args_cli=None):
        self._debug_transform_phase("after_startup_reset")
        self._log_door_joint_catalog_once()

    def debug_after_first_control_step(self, env, args_cli=None):
        if self._open_door_first_control_step_debug_done:
            return
        self._open_door_first_control_step_debug_done = True
        self._debug_transform_phase("after_first_control_step")

    def debug_joint_runtime_step(self, env):
        self._open_door_runtime_step_counter += 1
        step = int(self._open_door_runtime_step_counter)

        if not self._open_door_first_control_step_debug_done:
            self.debug_after_first_control_step(env)

        if self._open_door_joint_debug_enabled and step in _OPEN_DOOR_JOINT_SAMPLE_STEPS:
            if step not in self._open_door_joint_runtime_logged_steps:
                self._open_door_joint_runtime_logged_steps.add(step)
                self._log_joint_runtime_state(step)

    def _reset_open_door_runtime_debug_state(self):
        self._open_door_joint_runtime_logged_steps = set()
        self._open_door_runtime_step_counter = 0
        self._open_door_first_control_step_debug_done = False

    def _reset_object_self(self, env):
        applied = apply_deterministic_object_resets(
            self,
            env,
            selected_record_names={"door"},
        )
        if applied:
            print("[object_reset] " + ", ".join(applied))
        self.reset_open_door_latch(env, reason="reset_object_self")

    def _reset_all_self(self, env):
        base_mdp.reset_scene_to_default(
            env,
            torch.arange(env.num_envs, device=env.device),
        )
        applied = apply_deterministic_object_resets(
            self,
            env,
            selected_record_names={"door"},
        )
        if applied:
            print("[object_reset] " + ", ".join(applied))
        self.reset_open_door_latch(env, reason="reset_all_self")

    def setup_open_door_latch(self, env, env_ids=None, reason: str = "setup"):
        self.reset_open_door_latch(env, env_ids=env_ids, reason=reason)

    def reset_open_door_latch(self, env, env_ids=None, reason: str = "reset"):
        if not self._open_door_latch_enabled:
            return
        try:
            stage = self._get_open_door_stage(env)
            if stage is None:
                self._log_open_door_latch_missing_once("stage", "stage unavailable; latch reset skipped")
                return

            for env_id in self._normalize_open_door_env_ids(env, env_ids):
                prims = self._get_open_door_prims(stage, env_id)
                leaf_joint = prims["leaf_joint"]
                if not leaf_joint or not leaf_joint.IsValid():
                    self._log_open_door_latch_missing_once(
                        f"leaf_joint:{env_id}",
                        f"env={env_id} leaf joint missing at {prims['leaf_joint_path']}; latch reset skipped",
                    )
                    continue

                self._capture_open_door_latch_defaults(prims)
                self._capture_open_door_handle_reference(prims, env_id)
                self._open_door_leaf_tensor_proxy_reference.pop(env_id, None)
                self._capture_open_door_handle_tensor_proxy_reference(env, env_id)
                self._capture_open_door_dc_defaults(env_id)
                self._capture_open_door_handle_dc_reference(env_id)
                self._apply_open_door_leaf_lock(leaf_joint)
                self._capture_open_door_articulation_defaults(env, env_id)
                self._capture_open_door_handle_articulation_reference(env, env_id)
                self._apply_open_door_leaf_lock_articulation(env, env_id)
                self._apply_open_door_leaf_lock_dynamic(env_id)
                self._open_door_latch_unlocked_env_ids.discard(env_id)

                if self._open_door_latch_debug_enabled:
                    print(
                        "[open_door_latch] "
                        f"phase=lock env={env_id} reason={reason} "
                        f"handle_threshold_deg={self._open_door_handle_unlock_angle_deg:.3f} "
                        f"leaf_lock={self._format_open_door_leaf_lock_drive()}"
                    )
        except Exception as exc:
            print(f"[open_door_latch] failed to reset latch: {exc}")

    def apply_open_door_latch_interval(self, env, env_ids=None, reason: str = "interval"):
        if not self._open_door_latch_enabled:
            return
        try:
            stage = self._get_open_door_stage(env)
            if stage is None:
                self._log_open_door_latch_missing_once("stage_interval", "stage unavailable; latch poll skipped")
                return

            self._open_door_latch_poll_counter += 1
            should_log_poll = (
                self._open_door_latch_debug_enabled
                and self._open_door_latch_poll_counter % self._open_door_latch_poll_log_interval == 0
            )

            for env_id in self._normalize_open_door_env_ids(env, env_ids):
                prims = self._get_open_door_prims(stage, env_id)
                leaf_joint = prims["leaf_joint"]
                if not leaf_joint or not leaf_joint.IsValid():
                    continue

                self._capture_open_door_latch_defaults(prims)
                handle_angle_deg = self._get_open_door_handle_angle_deg(env, prims, env_id)
                if handle_angle_deg is None:
                    continue
                angle_source = self._open_door_latch_angle_source_by_env.get(env_id, "unknown")
                tensor_proxy_angle_deg = self._get_open_door_handle_angle_deg_tensor_proxy(env, env_id)
                articulation_angle_deg = self._get_open_door_handle_angle_deg_articulation(env, env_id)
                dynamic_angle_deg = (
                    self._get_open_door_handle_angle_deg_dynamic(env_id)
                    if self._open_door_dc_angle_debug_enabled
                    else None
                )
                usd_xform_angle_deg = self._get_open_door_handle_angle_deg_usd_xform(prims, env_id)
                diagnostic_angles = (
                    f"tensor_proxy_handle_delta_deg={_format_optional_float(tensor_proxy_angle_deg)} "
                    f"articulation_handle_delta_deg={_format_optional_float(articulation_angle_deg)} "
                    f"dc_handle_delta_deg={_format_optional_float(dynamic_angle_deg)} "
                    f"usd_xform_handle_delta_deg={_format_optional_float(usd_xform_angle_deg)}"
                )

                if env_id not in self._open_door_latch_unlocked_env_ids:
                    if self._open_door_latch_unlock_threshold_met(handle_angle_deg):
                        self._restore_open_door_leaf_defaults(leaf_joint)
                        self._restore_open_door_leaf_defaults_articulation(env, env_id)
                        self._restore_open_door_leaf_defaults_dynamic(env_id)
                        self._open_door_latch_unlocked_env_ids.add(env_id)
                        print(
                            "[open_door_latch] "
                            f"phase=unlock env={env_id} reason={reason} "
                            f"angle_source={angle_source} "
                            f"handle_angle_deg={handle_angle_deg:.3f} "
                            f"threshold_deg={self._open_door_handle_unlock_angle_deg:.3f} "
                            f"{diagnostic_angles} "
                            f"leaf_restored={self._format_open_door_leaf_defaults(leaf_joint)}"
                        )
                    elif should_log_poll:
                        print(
                            "[open_door_latch] "
                            f"phase=poll env={env_id} state=locked reason={reason} "
                            f"angle_source={angle_source} "
                            f"handle_angle_deg={handle_angle_deg:.3f} "
                            f"threshold_deg={self._open_door_handle_unlock_angle_deg:.3f} "
                            f"{diagnostic_angles}"
                        )
                elif should_log_poll:
                    print(
                        "[open_door_latch] "
                        f"phase=poll env={env_id} state=unlocked reason={reason} "
                        f"angle_source={angle_source} "
                        f"handle_angle_deg={handle_angle_deg:.3f} "
                        f"{diagnostic_angles}"
                    )
        except Exception as exc:
            print(f"[open_door_latch] failed during latch poll: {exc}")

    def _get_open_door_stage(self, env=None):
        try:
            sim = getattr(env, "sim", None)
            stage = getattr(sim, "stage", None)
            if stage is not None:
                return stage
        except Exception:
            pass
        import omni.usd

        return omni.usd.get_context().get_stage()

    def _normalize_open_door_env_ids(self, env, env_ids=None) -> list[int]:
        if env_ids is None or isinstance(env_ids, slice):
            num_envs = int(getattr(env, "num_envs", 1) or 1)
            return list(range(num_envs))
        if isinstance(env_ids, torch.Tensor):
            return [int(item) for item in env_ids.detach().cpu().reshape(-1).tolist()]
        if isinstance(env_ids, np.ndarray):
            return [int(item) for item in env_ids.reshape(-1).tolist()]
        if isinstance(env_ids, (list, tuple, set)):
            return [int(item) for item in env_ids]
        return [int(env_ids)]

    def _get_open_door_prims(self, stage, env_id: int) -> dict[str, Any]:
        root_path = f"/World/envs/env_{env_id}/Door"
        leaf_joint_path = f"{root_path}/{DOOR_LEAF_JOINT_REL_PATH}"
        handle_joint_path = f"{root_path}/{DOOR_HANDLE_JOINT_REL_PATH}"
        return {
            "root_path": root_path,
            "leaf_joint_path": leaf_joint_path,
            "handle_joint_path": handle_joint_path,
            "leaf": stage.GetPrimAtPath(f"{root_path}/{DOOR_LEAF_BODY_REL_PATH}"),
            "handle": stage.GetPrimAtPath(f"{root_path}/{DOOR_HANDLE_BODY_REL_PATH}"),
            "leaf_joint": stage.GetPrimAtPath(leaf_joint_path),
            "handle_joint": stage.GetPrimAtPath(handle_joint_path),
        }

    def _log_open_door_latch_missing_once(self, key: str, message: str):
        if key in self._open_door_latch_missing_logged:
            return
        self._open_door_latch_missing_logged.add(key)
        print(f"[open_door_latch] {message}")

    def _get_open_door_dc(self):
        if self._open_door_dc is not None and self._open_door_dc_module is not None:
            return self._open_door_dc, self._open_door_dc_module
        try:
            from omni.isaac.dynamic_control import _dynamic_control

            self._open_door_dc = _dynamic_control.acquire_dynamic_control_interface()
            self._open_door_dc_module = _dynamic_control
            return self._open_door_dc, self._open_door_dc_module
        except Exception as exc:
            self._log_open_door_latch_missing_once(
                "dynamic_control_import",
                f"dynamic_control unavailable; falling back to USD xform handle angle: {exc}",
            )
            return None, None

    def _get_open_door_articulation_handles(self, env) -> dict[str, Any] | None:
        cached = self._open_door_articulation_cache.get("door")
        if cached:
            return cached
        try:
            door_asset = env.scene["door"]
        except Exception as exc:
            self._log_open_door_latch_missing_once(
                "articulation_door_asset",
                f"door articulation unavailable from env.scene: {exc}",
            )
            return None

        if not hasattr(door_asset, "data") or not hasattr(door_asset.data, "joint_pos"):
            self._log_open_door_latch_missing_once(
                "articulation_joint_pos",
                "door scene asset has no IsaacLab articulation joint_pos tensor; articulation angle source disabled",
            )
            return None

        leaf_id = None
        handle_id = None
        if hasattr(door_asset, "find_joints"):
            try:
                leaf_id = _first_index(door_asset.find_joints(["RevoluteJoint_door001"], preserve_order=True)[0])
            except Exception:
                leaf_id = None
            for pattern in ("E_handle_4/RevoluteJoint", "RevoluteJoint$"):
                try:
                    handle_id = _first_index(door_asset.find_joints([pattern], preserve_order=True)[0])
                except Exception:
                    handle_id = None
                if handle_id is not None:
                    break

        joint_names = list(getattr(door_asset, "joint_names", []) or getattr(door_asset.data, "joint_names", []) or [])
        if leaf_id is None:
            for index, name in enumerate(joint_names):
                if str(name) == "RevoluteJoint_door001" or str(name).endswith("/E_leaf_2/RevoluteJoint_door001"):
                    leaf_id = index
                    break
        if handle_id is None:
            for index, name in enumerate(joint_names):
                if str(name) == "RevoluteJoint" or str(name).endswith("/E_handle_4/RevoluteJoint"):
                    handle_id = index
                    break

        if handle_id is None:
            self._log_open_door_latch_missing_once(
                "articulation_handle_joint",
                f"door articulation handle joint not found; joint_names={joint_names}",
            )
            return None

        handles = {
            "door_asset": door_asset,
            "leaf_id": leaf_id,
            "handle_id": handle_id,
            "joint_names": joint_names,
        }
        self._open_door_articulation_cache["door"] = handles
        if self._open_door_latch_debug_enabled:
            print(
                "[open_door_latch] "
                f"phase=articulation_catalog leaf_id={leaf_id} handle_id={handle_id} joint_names={joint_names}"
            )
        return handles

    def _resolve_open_door_tensor_proxy_root_expr(self) -> str | None:
        try:
            import isaaclab.sim as sim_utils
            from pxr import UsdPhysics
        except Exception as exc:
            self._log_open_door_latch_missing_once(
                "tensor_proxy_import",
                f"tensor proxy unavailable; failed to import IsaacLab/PXR helpers: {exc}",
            )
            return None

        prim_expr = "/World/envs/env_.*/Door"
        try:
            first_env_matching_prim = sim_utils.find_first_matching_prim(prim_expr)
        except Exception as exc:
            self._log_open_door_latch_missing_once(
                "tensor_proxy_find_door",
                f"tensor proxy failed to resolve {prim_expr}: {exc}",
            )
            return None
        if first_env_matching_prim is None:
            self._log_open_door_latch_missing_once(
                "tensor_proxy_find_door",
                f"tensor proxy failed to find prim for {prim_expr}",
            )
            return None

        first_env_matching_path = first_env_matching_prim.GetPath().pathString
        try:
            if first_env_matching_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                root_prims = [first_env_matching_prim]
            else:
                root_prims = sim_utils.get_all_matching_child_prims(
                    first_env_matching_path,
                    predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
                )
        except Exception as exc:
            self._log_open_door_latch_missing_once(
                "tensor_proxy_find_root",
                f"tensor proxy failed to find articulation root under {first_env_matching_path}: {exc}",
            )
            return None

        if len(root_prims) == 0:
            self._log_open_door_latch_missing_once(
                "tensor_proxy_find_root",
                f"tensor proxy found no ArticulationRootAPI under {first_env_matching_path}",
            )
            return None
        if len(root_prims) > 1:
            self._log_open_door_latch_missing_once(
                "tensor_proxy_find_root",
                f"tensor proxy found multiple articulation roots under {first_env_matching_path}: {root_prims}",
            )
            return None

        root_path = root_prims[0].GetPath().pathString
        relative_root_path = root_path[len(first_env_matching_path) :]
        return prim_expr + relative_root_path

    def _get_open_door_tensor_proxy_handles(self, env) -> dict[str, Any] | None:
        if not self._open_door_tensor_proxy_enabled:
            return None
        cached = self._open_door_tensor_proxy_cache.get("door")
        if cached:
            return cached

        root_expr = self._resolve_open_door_tensor_proxy_root_expr()
        if not root_expr:
            return None

        try:
            from isaacsim.core.simulation_manager import SimulationManager

            physics_sim_view = SimulationManager.get_physics_sim_view()
            view = physics_sim_view.create_articulation_view(root_expr.replace(".*", "*"))
        except Exception as exc:
            self._log_open_door_latch_missing_once(
                "tensor_proxy_create_view",
                f"tensor proxy failed to create articulation view at {root_expr}: {exc}",
            )
            return None

        if getattr(view, "_backend", None) is None:
            self._log_open_door_latch_missing_once(
                "tensor_proxy_create_view",
                f"tensor proxy articulation view has no backend at {root_expr}",
            )
            return None

        joint_names = list(getattr(view.shared_metatype, "dof_names", []) or [])
        leaf_id = None
        handle_id = None
        for index, name in enumerate(joint_names):
            name = str(name)
            if leaf_id is None and (name == "RevoluteJoint_door001" or name.endswith("/E_leaf_2/RevoluteJoint_door001")):
                leaf_id = index
            if handle_id is None and (name == "RevoluteJoint" or name.endswith("/E_handle_4/RevoluteJoint")):
                handle_id = index

        if handle_id is None:
            self._log_open_door_latch_missing_once(
                "tensor_proxy_handle_joint",
                f"tensor proxy handle joint not found; root_expr={root_expr} joint_names={joint_names}",
            )
            return None

        handles = {
            "view": view,
            "root_expr": root_expr,
            "leaf_id": leaf_id,
            "handle_id": handle_id,
            "joint_names": joint_names,
        }
        self._open_door_tensor_proxy_cache["door"] = handles
        if self._open_door_latch_debug_enabled:
            print(
                "[open_door_latch] "
                f"phase=tensor_proxy_catalog root_expr={root_expr} "
                f"leaf_id={leaf_id} handle_id={handle_id} joint_names={joint_names}"
            )
        return handles

    def _capture_open_door_handle_tensor_proxy_reference(self, env, env_id: int):
        handles = self._get_open_door_tensor_proxy_handles(env)
        if not handles:
            return
        pos = self._read_open_door_tensor_proxy_handle_pos(handles, env_id, log_failures=False)
        if pos is None:
            handles = self._rebuild_open_door_tensor_proxy_handles(env, env_id, "reference")
            pos = self._read_open_door_tensor_proxy_handle_pos(handles, env_id) if handles else None
        if pos is None:
            return
        self._open_door_latch_tensor_proxy_reference[env_id] = pos
        if self._open_door_latch_debug_enabled:
            print(
                "[open_door_latch] "
                f"phase=tensor_proxy_reference env={env_id} "
                f"handle_pos_rad={pos:.6f} handle_pos_deg={math.degrees(pos):.3f}"
            )

    def _rebuild_open_door_tensor_proxy_handles(self, env, env_id: int, reason: str) -> dict[str, Any] | None:
        self._open_door_tensor_proxy_cache.pop("door", None)
        handles = self._get_open_door_tensor_proxy_handles(env)
        if handles and self._open_door_latch_debug_enabled:
            print(f"[open_door_latch] phase=tensor_proxy_rebuild env={env_id} reason={reason}")
        return handles

    def _read_open_door_tensor_proxy_handle_pos(
        self, handles: dict[str, Any], env_id: int, log_failures: bool = True
    ) -> float | None:
        return self._read_open_door_tensor_proxy_dof_pos(handles, env_id, handles["handle_id"], "handle", log_failures)

    def _read_open_door_tensor_proxy_dof_pos(
        self,
        handles: dict[str, Any],
        env_id: int,
        dof_id: int | None,
        label: str,
        log_failures: bool = True,
    ) -> float | None:
        if dof_id is None:
            if log_failures:
                self._log_open_door_latch_missing_once(
                    f"tensor_proxy_{label}_joint",
                    f"env={env_id} tensor proxy {label} joint unavailable",
                )
            return None
        view = handles["view"]
        try:
            joint_pos = view.get_dof_positions()
        except Exception as exc:
            if log_failures:
                self._log_open_door_latch_missing_once(
                    f"tensor_proxy_positions:{env_id}",
                    f"env={env_id} tensor proxy failed to read dof positions: {exc}",
                )
            return None
        try:
            return float(joint_pos[env_id, dof_id].item())
        except Exception:
            try:
                return float(joint_pos[0, dof_id].item())
            except Exception as exc:
                self._log_open_door_latch_missing_once(
                    f"tensor_proxy_angle:{label}:{env_id}",
                    f"env={env_id} tensor proxy {label} angle unavailable: {exc}",
                )
                return None

    def _get_open_door_handle_angle_deg_tensor_proxy(self, env, env_id: int) -> float | None:
        handles = self._get_open_door_tensor_proxy_handles(env)
        if not handles:
            return None
        pos = self._read_open_door_tensor_proxy_handle_pos(handles, env_id, log_failures=False)
        if pos is None:
            handles = self._rebuild_open_door_tensor_proxy_handles(env, env_id, "read_failure")
            pos = self._read_open_door_tensor_proxy_handle_pos(handles, env_id) if handles else None
        if pos is None:
            return None
        reference = self._open_door_latch_tensor_proxy_reference.setdefault(env_id, pos)
        return math.degrees(pos - reference)

    def _get_open_door_leaf_angle_deg_tensor_proxy(self, env, env_id: int) -> float | None:
        handles = self._get_open_door_tensor_proxy_handles(env)
        if not handles:
            return None
        leaf_id = handles.get("leaf_id")
        if leaf_id is None:
            return None
        pos = self._read_open_door_tensor_proxy_dof_pos(handles, env_id, leaf_id, "leaf", log_failures=False)
        if pos is None:
            handles = self._rebuild_open_door_tensor_proxy_handles(env, env_id, "leaf_read_failure")
            if handles:
                pos = self._read_open_door_tensor_proxy_dof_pos(
                    handles, env_id, handles.get("leaf_id"), "leaf"
                )
        if pos is None:
            return None
        reference = self._open_door_leaf_tensor_proxy_reference.setdefault(env_id, pos)
        return math.degrees(pos - reference)

    def _capture_open_door_articulation_defaults(self, env, env_id: int):
        handles = self._get_open_door_articulation_handles(env)
        if not handles or env_id in self._open_door_latch_articulation_defaults:
            return
        door_asset = handles["door_asset"]
        leaf_id = handles.get("leaf_id")
        if leaf_id is None:
            return
        defaults = {}
        data = getattr(door_asset, "data", None)
        for attr_name, key in (
            ("joint_stiffness", "stiffness"),
            ("joint_damping", "damping"),
            ("joint_friction_coeff", "friction"),
        ):
            values = getattr(data, attr_name, None)
            if values is None:
                continue
            try:
                defaults[key] = float(values[env_id, leaf_id].item())
            except Exception:
                try:
                    defaults[key] = float(values[0, leaf_id].item())
                except Exception:
                    pass
        self._open_door_latch_articulation_defaults[env_id] = defaults
        if self._open_door_latch_debug_enabled:
            print(f"[open_door_latch] phase=articulation_defaults env={env_id} leaf_defaults={defaults}")

    def _capture_open_door_handle_articulation_reference(self, env, env_id: int):
        handles = self._get_open_door_articulation_handles(env)
        if not handles:
            return
        door_asset = handles["door_asset"]
        handle_id = handles["handle_id"]
        try:
            pos = float(door_asset.data.joint_pos[env_id, handle_id].item())
        except Exception:
            try:
                pos = float(door_asset.data.joint_pos[0, handle_id].item())
            except Exception as exc:
                self._log_open_door_latch_missing_once(
                    f"articulation_reference:{env_id}",
                    f"env={env_id} articulation handle reference unavailable: {exc}",
                )
                return
        self._open_door_latch_articulation_reference[env_id] = pos
        if self._open_door_latch_debug_enabled:
            print(
                "[open_door_latch] "
                f"phase=articulation_reference env={env_id} "
                f"handle_pos_rad={pos:.6f} handle_pos_deg={math.degrees(pos):.3f}"
            )

    def _write_open_door_leaf_articulation_props(self, env, env_id: int, values: dict[str, Any]) -> bool:
        handles = self._get_open_door_articulation_handles(env)
        if not handles:
            return False
        door_asset = handles["door_asset"]
        leaf_id = handles.get("leaf_id")
        if leaf_id is None:
            return False
        joint_ids = [leaf_id]
        try:
            if values.get("stiffness") is not None and hasattr(door_asset, "write_joint_stiffness_to_sim"):
                door_asset.write_joint_stiffness_to_sim(float(values["stiffness"]), joint_ids=joint_ids)
            if values.get("damping") is not None and hasattr(door_asset, "write_joint_damping_to_sim"):
                door_asset.write_joint_damping_to_sim(float(values["damping"]), joint_ids=joint_ids)
            if values.get("friction") is not None and hasattr(door_asset, "write_joint_friction_coefficient_to_sim"):
                door_asset.write_joint_friction_coefficient_to_sim(float(values["friction"]), joint_ids=joint_ids)
            if self._open_door_latch_debug_enabled:
                print(
                    "[open_door_latch] "
                    f"phase=articulation_leaf_props env={env_id} leaf_id={leaf_id} values={values}"
                )
            return True
        except Exception as exc:
            self._log_open_door_latch_missing_once(
                f"articulation_leaf_props:{env_id}:{values}",
                f"env={env_id} failed to write articulation leaf props: {exc}",
            )
            return False

    def _apply_open_door_leaf_lock_articulation(self, env, env_id: int) -> bool:
        return self._write_open_door_leaf_articulation_props(
            env,
            env_id,
            {
                "stiffness": self._open_door_leaf_lock_stiffness,
                "damping": self._open_door_leaf_lock_damping,
                "friction": self._open_door_leaf_lock_joint_friction,
            },
        )

    def _restore_open_door_leaf_defaults_articulation(self, env, env_id: int) -> bool:
        defaults = self._open_door_latch_articulation_defaults.get(env_id)
        if not defaults:
            return False
        return self._write_open_door_leaf_articulation_props(env, env_id, defaults)

    def _get_open_door_dc_handles(self, env_id: int) -> dict[str, Any] | None:
        dc, dc_mod = self._get_open_door_dc()
        if dc is None or dc_mod is None:
            return None

        cached = self._open_door_dc_cache.get(env_id)
        if cached and cached.get("art") != dc_mod.INVALID_HANDLE:
            return cached

        root_path = f"/World/envs/env_{env_id}/Door"
        candidate_paths = [
            root_path if rel_path == "" else f"{root_path}/{rel_path}" for rel_path in DOOR_DC_ARTICULATION_REL_PATHS
        ]
        for art_path in candidate_paths:
            try:
                art = dc.get_articulation(art_path)
            except Exception:
                art = dc_mod.INVALID_HANDLE
            if art == dc_mod.INVALID_HANDLE:
                continue

            dof_infos = []
            leaf_dof = dc_mod.INVALID_HANDLE
            handle_dof = dc_mod.INVALID_HANDLE
            leaf_index = None
            handle_index = None
            try:
                dof_count = int(dc.get_articulation_dof_count(art))
            except Exception:
                dof_count = 0

            for index in range(dof_count):
                dof = dc.get_articulation_dof(art, index)
                if dof == dc_mod.INVALID_HANDLE:
                    continue
                try:
                    name = str(dc.get_dof_name(dof))
                except Exception:
                    name = ""
                try:
                    path = str(dc.get_dof_path(dof))
                except Exception:
                    path = ""
                dof_infos.append(f"{index}:{name}:{path}")
                if name == "RevoluteJoint_door001" or path.endswith(DOOR_LEAF_JOINT_REL_PATH):
                    leaf_dof = dof
                    leaf_index = index
                if name == "RevoluteJoint" or path.endswith(DOOR_HANDLE_JOINT_REL_PATH):
                    handle_dof = dof
                    handle_index = index

            handles = {
                "dc": dc,
                "dc_mod": dc_mod,
                "art": art,
                "art_path": art_path,
                "leaf_dof": leaf_dof,
                "leaf_index": leaf_index,
                "handle_dof": handle_dof,
                "handle_index": handle_index,
                "dof_infos": dof_infos,
            }
            self._open_door_dc_cache[env_id] = handles
            if self._open_door_latch_debug_enabled:
                print(
                    "[open_door_latch] "
                    f"phase=dc_catalog env={env_id} art_path={art_path} "
                    f"leaf_index={leaf_index} handle_index={handle_index} dofs={dof_infos}"
                )
            return handles

        self._log_open_door_latch_missing_once(
            f"dynamic_control_articulation:{env_id}",
            f"env={env_id} dynamic_control articulation not found; candidates={candidate_paths}",
        )
        return None

    def _capture_open_door_dc_defaults(self, env_id: int):
        handles = self._get_open_door_dc_handles(env_id)
        if not handles:
            return
        leaf_index = handles.get("leaf_index")
        if leaf_index is None or env_id in self._open_door_latch_dc_defaults:
            return
        dc = handles["dc"]
        props = dc.get_articulation_dof_properties(handles["art"])
        if props is None:
            return
        fields = list(getattr(props, "dtype", ()).names or [])
        defaults = {"fields": fields}
        for field in fields:
            try:
                defaults[field] = props[leaf_index][field].item()
            except Exception:
                try:
                    defaults[field] = props[leaf_index][field]
                except Exception:
                    pass
        self._open_door_latch_dc_defaults[env_id] = defaults
        if self._open_door_latch_debug_enabled:
            print(f"[open_door_latch] phase=dc_defaults env={env_id} leaf_defaults={defaults}")

    def _set_open_door_dynamic_leaf_props(self, env_id: int, values: dict[str, Any]):
        handles = self._get_open_door_dc_handles(env_id)
        if not handles:
            return False
        leaf_index = handles.get("leaf_index")
        if leaf_index is None:
            return False
        dc = handles["dc"]
        props = dc.get_articulation_dof_properties(handles["art"])
        if props is None:
            return False
        fields = set(getattr(props, "dtype", ()).names or [])
        if "driveMode" in fields and values.get("drive_mode") is not None:
            props[leaf_index]["driveMode"] = values["drive_mode"]
        if "stiffness" in fields and values.get("stiffness") is not None:
            props[leaf_index]["stiffness"] = float(values["stiffness"])
        if "damping" in fields and values.get("damping") is not None:
            props[leaf_index]["damping"] = float(values["damping"])
        for field in ("max_effort", "maxEffort", "max_force", "maxForce"):
            if field in fields and values.get("max_force") is not None:
                props[leaf_index][field] = float(values["max_force"])
        dc.set_articulation_dof_properties(handles["art"], props)
        dc.wake_up_articulation(handles["art"])
        return True

    def _apply_open_door_leaf_lock_dynamic(self, env_id: int):
        dc, dc_mod = self._get_open_door_dc()
        if dc is None or dc_mod is None:
            return
        drive_mode = getattr(dc_mod, "DRIVE_FORCE", None)
        applied = self._set_open_door_dynamic_leaf_props(
            env_id,
            {
                "drive_mode": drive_mode,
                "stiffness": self._open_door_leaf_lock_stiffness,
                "damping": self._open_door_leaf_lock_damping,
                "max_force": self._open_door_leaf_lock_max_force,
            },
        )
        if applied and self._open_door_latch_debug_enabled:
            print(f"[open_door_latch] phase=dc_lock env={env_id}")

    def _restore_open_door_leaf_defaults_dynamic(self, env_id: int):
        defaults = self._open_door_latch_dc_defaults.get(env_id)
        if not defaults:
            return
        values = {}
        for key in ("driveMode", "stiffness", "damping"):
            if key in defaults:
                values["drive_mode" if key == "driveMode" else key] = defaults[key]
        for field in ("max_effort", "maxEffort", "max_force", "maxForce"):
            if field in defaults:
                values["max_force"] = defaults[field]
                break
        applied = self._set_open_door_dynamic_leaf_props(env_id, values)
        if applied and self._open_door_latch_debug_enabled:
            print(f"[open_door_latch] phase=dc_restore env={env_id} values={values}")

    def _capture_open_door_latch_defaults(self, prims: dict[str, Any]):
        leaf_joint = prims.get("leaf_joint")
        if leaf_joint and leaf_joint.IsValid():
            leaf_path = str(leaf_joint.GetPath())
            self._open_door_latch_defaults.setdefault(
                leaf_path,
                self._read_open_door_joint_drive(leaf_joint, DOOR_LEAF_DEFAULT_DRIVE),
            )

    def _read_open_door_joint_drive(self, joint_prim, fallback: dict[str, Any]) -> dict[str, Any]:
        from pxr import PhysxSchema, UsdPhysics

        values = dict(fallback)
        drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        if drive:
            attr_map = {
                "drive_type": drive.GetTypeAttr(),
                "stiffness": drive.GetStiffnessAttr(),
                "damping": drive.GetDampingAttr(),
                "max_force": drive.GetMaxForceAttr(),
                "target_position": drive.GetTargetPositionAttr(),
                "target_velocity": drive.GetTargetVelocityAttr(),
            }
            for key, attr in attr_map.items():
                if attr and attr.IsValid():
                    value = attr.Get()
                    if value is not None:
                        values[key] = value

        joint_api = PhysxSchema.PhysxJointAPI(joint_prim)
        friction_attr = joint_api.GetJointFrictionAttr()
        if friction_attr and friction_attr.IsValid():
            friction = friction_attr.Get()
            if friction is not None:
                values["joint_friction"] = friction
        return values

    def _write_open_door_joint_drive(self, joint_prim, values: dict[str, Any]):
        from pxr import PhysxSchema, UsdPhysics

        drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")

        attr_map = {
            "drive_type": drive.GetTypeAttr(),
            "stiffness": drive.GetStiffnessAttr(),
            "damping": drive.GetDampingAttr(),
            "max_force": drive.GetMaxForceAttr(),
            "target_position": drive.GetTargetPositionAttr(),
            "target_velocity": drive.GetTargetVelocityAttr(),
        }
        for key, attr in attr_map.items():
            if key in values and values[key] is not None:
                attr.Set(values[key])

        if values.get("joint_friction") is not None:
            joint_api = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
            joint_api.GetJointFrictionAttr().Set(float(values["joint_friction"]))

    def _apply_open_door_leaf_lock(self, leaf_joint):
        self._write_open_door_joint_drive(
            leaf_joint,
            {
                "drive_type": self._open_door_leaf_lock_drive_type,
                "stiffness": self._open_door_leaf_lock_stiffness,
                "damping": self._open_door_leaf_lock_damping,
                "max_force": self._open_door_leaf_lock_max_force,
                "target_position": 0.0,
                "target_velocity": 0.0,
                "joint_friction": self._open_door_leaf_lock_joint_friction,
            },
        )

    def _restore_open_door_leaf_defaults(self, leaf_joint):
        leaf_path = str(leaf_joint.GetPath())
        default_values = self._open_door_latch_defaults.get(leaf_path, DOOR_LEAF_DEFAULT_DRIVE)
        self._write_open_door_joint_drive(leaf_joint, default_values)

    def _capture_open_door_handle_reference(self, prims: dict[str, Any], env_id: int):
        from pxr import Usd, UsdGeom

        leaf_prim = prims.get("leaf")
        handle_prim = prims.get("handle")
        if not leaf_prim or not leaf_prim.IsValid() or not handle_prim or not handle_prim.IsValid():
            self._log_open_door_latch_missing_once(
                f"handle_reference:{env_id}",
                f"env={env_id} leaf/handle prim missing; handle angle reference unavailable",
            )
            return

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        leaf_world = cache.GetLocalToWorldTransform(leaf_prim)
        handle_world = cache.GetLocalToWorldTransform(handle_prim)
        self._open_door_latch_reference[env_id] = leaf_world.GetInverse() * handle_world

    def _capture_open_door_handle_dc_reference(self, env_id: int):
        handles = self._get_open_door_dc_handles(env_id)
        if not handles:
            return
        handle_dof = handles.get("handle_dof")
        dc_mod = handles.get("dc_mod")
        if handle_dof == dc_mod.INVALID_HANDLE:
            return
        state = handles["dc"].get_dof_state(handle_dof, dc_mod.STATE_ALL)
        if state is None:
            return
        self._open_door_latch_dc_reference[env_id] = float(state.pos)
        if self._open_door_latch_debug_enabled:
            print(
                "[open_door_latch] "
                f"phase=dc_reference env={env_id} "
                f"handle_pos_rad={float(state.pos):.6f} handle_pos_deg={math.degrees(float(state.pos)):.3f}"
            )

    def _get_open_door_handle_angle_deg(self, env, prims: dict[str, Any], env_id: int) -> float | None:
        tensor_proxy_angle = self._get_open_door_handle_angle_deg_tensor_proxy(env, env_id)
        if tensor_proxy_angle is not None:
            self._open_door_latch_angle_source_by_env[env_id] = "tensor_proxy"
            return tensor_proxy_angle

        articulation_angle = self._get_open_door_handle_angle_deg_articulation(env, env_id)
        if articulation_angle is not None:
            self._open_door_latch_angle_source_by_env[env_id] = "articulation_tensor"
            return articulation_angle

        if self._open_door_dc_angle_debug_enabled:
            dynamic_angle = self._get_open_door_handle_angle_deg_dynamic(env_id)
            if dynamic_angle is not None:
                self._open_door_latch_angle_source_by_env[env_id] = "dynamic_control"
                return dynamic_angle

        from pxr import Usd, UsdGeom

        if env_id not in self._open_door_latch_reference:
            self._capture_open_door_handle_reference(prims, env_id)
        reference = self._open_door_latch_reference.get(env_id)
        if reference is None:
            return None

        leaf_prim = prims.get("leaf")
        handle_prim = prims.get("handle")
        if not leaf_prim or not leaf_prim.IsValid() or not handle_prim or not handle_prim.IsValid():
            return None

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        leaf_world = cache.GetLocalToWorldTransform(leaf_prim)
        handle_world = cache.GetLocalToWorldTransform(handle_prim)
        current = leaf_world.GetInverse() * handle_world
        delta = reference.GetInverse() * current
        self._open_door_latch_angle_source_by_env[env_id] = "usd_xform"
        return _signed_axis_angle_deg_from_matrix(delta, "Y")

    def _get_open_door_handle_angle_deg_usd_xform(self, prims: dict[str, Any], env_id: int) -> float | None:
        from pxr import Usd, UsdGeom

        if env_id not in self._open_door_latch_reference:
            self._capture_open_door_handle_reference(prims, env_id)
        reference = self._open_door_latch_reference.get(env_id)
        if reference is None:
            return None

        leaf_prim = prims.get("leaf")
        handle_prim = prims.get("handle")
        if not leaf_prim or not leaf_prim.IsValid() or not handle_prim or not handle_prim.IsValid():
            return None

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        leaf_world = cache.GetLocalToWorldTransform(leaf_prim)
        handle_world = cache.GetLocalToWorldTransform(handle_prim)
        current = leaf_world.GetInverse() * handle_world
        delta = reference.GetInverse() * current
        return _signed_axis_angle_deg_from_matrix(delta, "Y")

    def _get_open_door_handle_angle_deg_articulation(self, env, env_id: int) -> float | None:
        handles = self._get_open_door_articulation_handles(env)
        if not handles:
            return None
        door_asset = handles["door_asset"]
        handle_id = handles["handle_id"]
        try:
            pos = float(door_asset.data.joint_pos[env_id, handle_id].item())
        except Exception:
            try:
                pos = float(door_asset.data.joint_pos[0, handle_id].item())
            except Exception as exc:
                self._log_open_door_latch_missing_once(
                    f"articulation_angle:{env_id}",
                    f"env={env_id} articulation handle angle unavailable: {exc}",
                )
                return None
        reference = self._open_door_latch_articulation_reference.setdefault(env_id, pos)
        return math.degrees(pos - reference)

    def _get_open_door_handle_angle_deg_dynamic(self, env_id: int) -> float | None:
        handles = self._get_open_door_dc_handles(env_id)
        if not handles:
            return None
        handle_dof = handles.get("handle_dof")
        dc_mod = handles.get("dc_mod")
        if handle_dof == dc_mod.INVALID_HANDLE:
            self._log_open_door_latch_missing_once(
                f"dynamic_control_handle_dof:{env_id}",
                f"env={env_id} dynamic_control handle DOF not found; dofs={handles.get('dof_infos')}",
            )
            return None
        state = handles["dc"].get_dof_state(handle_dof, dc_mod.STATE_ALL)
        if state is None:
            return None
        pos = float(state.pos)
        reference = self._open_door_latch_dc_reference.setdefault(env_id, pos)
        return math.degrees(pos - reference)

    def _open_door_latch_unlock_threshold_met(self, handle_angle_deg: float) -> bool:
        threshold = self._open_door_handle_unlock_angle_deg
        if threshold >= 0.0:
            return handle_angle_deg >= threshold
        return handle_angle_deg <= threshold

    def _format_open_door_leaf_lock_drive(self) -> str:
        return (
            f"type={self._open_door_leaf_lock_drive_type},"
            f"stiffness={self._open_door_leaf_lock_stiffness:g},"
            f"damping={self._open_door_leaf_lock_damping:g},"
            f"max_force={self._open_door_leaf_lock_max_force:g},"
            f"joint_friction={self._open_door_leaf_lock_joint_friction:g}"
        )

    def _format_open_door_leaf_defaults(self, leaf_joint) -> str:
        values = self._open_door_latch_defaults.get(str(leaf_joint.GetPath()), DOOR_LEAF_DEFAULT_DRIVE)
        return (
            f"type={values.get('drive_type')},"
            f"stiffness={float(values.get('stiffness', 0.0)):g},"
            f"damping={float(values.get('damping', 0.0)):g},"
            f"max_force={float(values.get('max_force', 0.0)):g},"
            f"joint_friction={float(values.get('joint_friction', 0.0)):g}"
        )

    def get_open_door_reward_diagnostics(self, env) -> dict[str, Any]:
        latch_unlocked = torch.full(
            (env.num_envs,),
            not self._open_door_latch_enabled,
            device=env.device,
            dtype=torch.bool,
        )
        handle_angle_deg = torch.full((env.num_envs,), float("nan"), device=env.device, dtype=torch.float32)
        leaf_angle_deg = torch.full((env.num_envs,), float("nan"), device=env.device, dtype=torch.float32)
        angle_source: list[str] = []

        for env_id in range(int(env.num_envs)):
            if self._open_door_latch_enabled:
                latch_unlocked[env_id] = env_id in self._open_door_latch_unlocked_env_ids

            handle_angle = self._get_open_door_handle_angle_deg_tensor_proxy(env, env_id)
            if handle_angle is not None:
                handle_angle_deg[env_id] = float(handle_angle)

            leaf_angle = self._get_open_door_leaf_angle_deg_tensor_proxy(env, env_id)
            if leaf_angle is not None:
                leaf_angle_deg[env_id] = float(leaf_angle)

            angle_source.append(self._open_door_latch_angle_source_by_env.get(env_id, "tensor_proxy_reward_diag"))

        return {
            "latch_enabled": self._open_door_latch_enabled,
            "latch_unlocked": latch_unlocked,
            "handle_angle_deg": handle_angle_deg,
            "leaf_angle_deg": leaf_angle_deg,
            "angle_source": angle_source,
        }

    def _disable_overlapping_room_gate_collisions(self):
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            room_root = stage.GetPrimAtPath("/World/envs/env_0/Room")
            if not room_root or not room_root.IsValid():
                print("[open_door] room root not found; skipping gate collision override")
                return

            gate_roots = []
            stack = [room_root]
            while stack:
                current = stack.pop()
                if current.GetName().lower() == "gate":
                    gate_roots.append(current)
                stack.extend(list(current.GetChildren()))

            if not gate_roots:
                print("[open_door] no gate subtree found under room root; skipping gate collision override")
                return

            disabled_count = 0
            gate_paths = [str(prim.GetPath()) for prim in gate_roots]
            for gate_root in gate_roots:
                stack = [gate_root]
                while stack:
                    current = stack.pop()
                    collision_attr = current.GetAttribute("physics:collisionEnabled")
                    if collision_attr.IsValid() and collision_attr.Get() is not False:
                        collision_attr.Set(False)
                        disabled_count += 1
                    stack.extend(list(current.GetChildren()))

            print(
                "[open_door] disabled collision on "
                f"{disabled_count} prims under gate subtrees: {gate_paths}"
            )
        except Exception as exc:
            print(f"[open_door] failed to disable room gate collisions: {exc}")

    def _configure_door_joint_physics(self, env):
        try:
            import omni.usd
            from pxr import PhysxSchema, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            door_asset = env.scene["door"]
            if not hasattr(door_asset, "find_joints"):
                print(
                    "[open_door] scheme_a_assetbase active: door scene asset has no find_joints(); "
                    "skip articulation joint runtime overrides"
                )
                return

            door_joint_ids, _ = door_asset.find_joints(["RevoluteJoint_door001"], preserve_order=True)
            handle_joint_ids, _ = door_asset.find_joints(["RevoluteJoint$"], preserve_order=True)
            door_joint = stage.GetPrimAtPath(
                "/World/envs/env_0/Door/E_leaf_2/RevoluteJoint_door001"
            )
            handle_joint = stage.GetPrimAtPath("/World/envs/env_0/Door/E_handle_4/RevoluteJoint")
            if not door_joint or not door_joint.IsValid():
                print("[open_door] door leaf joint not found; skipping drive override")
                return
            if not handle_joint or not handle_joint.IsValid():
                print("[open_door] handle joint not found; skipping handle friction override")

            angular_drive = UsdPhysics.DriveAPI.Get(door_joint, "angular")
            if not angular_drive:
                angular_drive = UsdPhysics.DriveAPI.Apply(door_joint, "angular")

            angular_drive.GetTypeAttr().Set("force")
            angular_drive.GetStiffnessAttr().Set(0.0)
            angular_drive.GetDampingAttr().Set(0.0)
            angular_drive.GetMaxForceAttr().Set(0.0)
            angular_drive.GetTargetPositionAttr().Set(0.0)
            angular_drive.GetTargetVelocityAttr().Set(0.0)

            door_asset.write_joint_stiffness_to_sim(0.0, joint_ids=door_joint_ids)
            door_asset.write_joint_damping_to_sim(0.0, joint_ids=door_joint_ids)
            door_asset.write_joint_friction_coefficient_to_sim(
                DOOR_LEAF_JOINT_STATIC_FRICTION,
                joint_dynamic_friction_coeff=DOOR_LEAF_JOINT_DYNAMIC_FRICTION,
                joint_viscous_friction_coeff=DOOR_LEAF_JOINT_VISCOUS_FRICTION,
                joint_ids=door_joint_ids,
            )

            leaf_joint_api = PhysxSchema.PhysxJointAPI.Apply(door_joint)
            handle_friction_value = None
            handle_drive_stiffness = None
            handle_drive_damping = None
            handle_drive_max_force = None
            handle_drive_target_position = None
            if handle_joint and handle_joint.IsValid():
                handle_drive = UsdPhysics.DriveAPI.Get(handle_joint, "angular")
                handle_joint_api = PhysxSchema.PhysxJointAPI.Apply(handle_joint)
                handle_friction_value = handle_joint_api.GetJointFrictionAttr().Get()
                if handle_drive:
                    handle_drive_stiffness = handle_drive.GetStiffnessAttr().Get()
                    handle_drive_damping = handle_drive.GetDampingAttr().Get()
                    handle_drive_max_force = handle_drive.GetMaxForceAttr().Get()
                    handle_drive_target_position = handle_drive.GetTargetPositionAttr().Get()

            leaf_stiffness = door_asset.data.joint_stiffness[0, door_joint_ids[0]].item()
            leaf_damping = door_asset.data.joint_damping[0, door_joint_ids[0]].item()
            leaf_static_friction = door_asset.data.joint_friction_coeff[0, door_joint_ids[0]].item()
            if len(handle_joint_ids) > 0:
                handle_stiffness = door_asset.data.joint_stiffness[0, handle_joint_ids[0]].item()
                handle_damping = door_asset.data.joint_damping[0, handle_joint_ids[0]].item()
                handle_static_friction = door_asset.data.joint_friction_coeff[0, handle_joint_ids[0]].item()
            else:
                handle_stiffness = None
                handle_damping = None
                handle_static_friction = None

            joint_names = None
            try:
                joint_names = list(env.scene["door"].data.joint_names)
            except Exception:
                joint_names = None

            print(
                "[open_door] configured door joints: "
                f"joint_names={joint_names}, "
                f"leaf_drive=(stiffness={angular_drive.GetStiffnessAttr().Get()}, "
                f"damping={angular_drive.GetDampingAttr().Get()}, "
                f"max_force={angular_drive.GetMaxForceAttr().Get()}), "
                f"leaf_runtime=(stiffness={leaf_stiffness}, damping={leaf_damping}, "
                f"static_friction={leaf_static_friction}), "
                f"leaf_joint_friction_attr={leaf_joint_api.GetJointFrictionAttr().Get()}, "
                f"handle_runtime=(stiffness={handle_stiffness}, damping={handle_damping}, "
                f"static_friction={handle_static_friction}), "
                f"handle_joint_friction_attr={handle_friction_value}, "
                f"handle_drive=(target={handle_drive_target_position}, stiffness={handle_drive_stiffness}, "
                f"damping={handle_drive_damping}, max_force={handle_drive_max_force})"
            )
        except Exception as exc:
            print(f"[open_door] failed to configure door joint physics: {exc}")

    def _debug_transform_phase(self, phase: str):
        if not self._open_door_transform_debug_enabled:
            return
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(f"[open_door_transform_debug] phase={phase} stage unavailable")
                return

            print(
                "[open_door_transform_debug] "
                f"phase={phase} stage_metadata="
                f"root={stage.GetPseudoRoot().GetPath()} default_prim={stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else '<none>'}"
            )
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            for path in _OPEN_DOOR_DEBUG_PRIM_PATHS:
                prim = stage.GetPrimAtPath(path)
                if prim is None or not prim.IsValid():
                    print(
                        f"[open_door_transform_debug] phase={phase} path={path} "
                        "status=missing"
                    )
                    continue
                local_matrix = _xformable_local_matrix(UsdGeom.Xformable(prim))
                world_matrix = cache.GetLocalToWorldTransform(prim)
                print(
                    f"[open_door_transform_debug] phase={phase} path={path} "
                    f"local_matrix={_matrix_to_rows(local_matrix)} "
                    f"world_matrix={_matrix_to_rows(world_matrix)}"
                )
        except Exception as exc:
            print(f"[open_door_transform_debug] phase={phase} failed: {exc}")

    def _iter_door_revolute_joints(self) -> list[dict[str, Any]]:
        try:
            import omni.usd
            from pxr import PhysxSchema, Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return []

            door_root = stage.GetPrimAtPath("/World/envs/env_0/Door")
            if door_root is None or not door_root.IsValid():
                return []

            joints: list[dict[str, Any]] = []
            for prim in Usd.PrimRange(door_root):
                type_name = prim.GetTypeName()
                if type_name not in {"PhysicsRevoluteJoint", "PhysicsJoint"} and not prim.GetName().startswith(
                    "RevoluteJoint"
                ):
                    continue
                body0_path = _first_target_path(prim, "physics:body0")
                body1_path = _first_target_path(prim, "physics:body1")
                classification = _classify_revolute_joint(body0_path, body1_path)
                angular_drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                joint_api = PhysxSchema.PhysxJointAPI(prim)
                joints.append(
                    {
                        "path": str(prim.GetPath()),
                        "classification": classification,
                        "body0": body0_path,
                        "body1": body1_path,
                        "axis": prim.GetAttribute("physics:axis").Get(),
                        "lower": prim.GetAttribute("physics:lowerLimit").Get(),
                        "upper": prim.GetAttribute("physics:upperLimit").Get(),
                        "drive_type": angular_drive.GetTypeAttr().Get() if angular_drive else None,
                        "drive_target_position": angular_drive.GetTargetPositionAttr().Get()
                        if angular_drive
                        else None,
                        "drive_stiffness": angular_drive.GetStiffnessAttr().Get() if angular_drive else None,
                        "drive_damping": angular_drive.GetDampingAttr().Get() if angular_drive else None,
                        "drive_max_force": angular_drive.GetMaxForceAttr().Get() if angular_drive else None,
                        "joint_friction": joint_api.GetJointFrictionAttr().Get()
                        if joint_api and joint_api.GetJointFrictionAttr().IsValid()
                        else None,
                    }
                )
            return joints
        except Exception as exc:
            print(f"[open_door_joint_debug] phase=joint_catalog failed: {exc}")
            return []

    def _log_door_joint_catalog_once(self):
        if not self._open_door_joint_debug_enabled or self._open_door_joint_catalog_logged:
            return
        joints = self._iter_door_revolute_joints()
        self._open_door_joint_catalog_logged = True
        for joint in joints:
            print(
                "[open_door_joint_debug] phase=joint_catalog "
                f"path={joint['path']} classification={joint['classification']} "
                f"body0={joint['body0']} body1={joint['body1']} axis={joint['axis']} "
                f"lower={joint['lower']} upper={joint['upper']} "
                f"drive_type={joint['drive_type']} drive_target_position={joint['drive_target_position']} "
                f"drive_stiffness={joint['drive_stiffness']} drive_damping={joint['drive_damping']} "
                f"drive_max_force={joint['drive_max_force']} joint_friction={joint['joint_friction']}"
            )

    def _log_joint_runtime_state(self, step: int):
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(f"[open_door_joint_debug] phase=runtime_sample step={step} stage unavailable")
                return

            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            body_prim = stage.GetPrimAtPath("/World/envs/env_0/Door/E_bodyM1_1")
            leaf_prim = stage.GetPrimAtPath("/World/envs/env_0/Door/E_leaf_2")
            handle_prim = stage.GetPrimAtPath("/World/envs/env_0/Door/E_handle_4")
            if not body_prim or not body_prim.IsValid() or not leaf_prim or not leaf_prim.IsValid():
                print(f"[open_door_joint_debug] phase=runtime_sample step={step} missing door prims")
                return

            leaf_rotate = _prim_rotate_zyx_deg(leaf_prim)
            handle_rotate = _prim_rotate_zyx_deg(handle_prim) if handle_prim and handle_prim.IsValid() else None
            body_world = cache.GetLocalToWorldTransform(body_prim)
            leaf_world = cache.GetLocalToWorldTransform(leaf_prim)
            handle_world = cache.GetLocalToWorldTransform(handle_prim) if handle_prim and handle_prim.IsValid() else None

            body_to_leaf = body_world.GetInverse() * leaf_world
            leaf_to_handle_angle = None
            if handle_world is not None:
                leaf_to_handle = leaf_world.GetInverse() * handle_world
                leaf_to_handle_angle = _quat_angle_deg_from_matrix(leaf_to_handle)

            dc_leaf_pos_deg = None
            dc_handle_pos_deg = None
            dc_handle_delta_deg = None
            dc = None
            dc_mod = None
            dc_handles = self._get_open_door_dc_handles(0)
            if dc_handles:
                dc = dc_handles.get("dc")
                dc_mod = dc_handles.get("dc_mod")
                leaf_dof = dc_handles.get("leaf_dof")
                handle_dof = dc_handles.get("handle_dof")
                if dc is not None and dc_mod is not None:
                    if leaf_dof != dc_mod.INVALID_HANDLE:
                        leaf_state = dc.get_dof_state(leaf_dof, dc_mod.STATE_ALL)
                        if leaf_state is not None:
                            dc_leaf_pos_deg = math.degrees(float(leaf_state.pos))
                    if handle_dof != dc_mod.INVALID_HANDLE:
                        handle_state = dc.get_dof_state(handle_dof, dc_mod.STATE_ALL)
                        if handle_state is not None:
                            handle_pos = float(handle_state.pos)
                            dc_handle_pos_deg = math.degrees(handle_pos)
                            reference = self._open_door_latch_dc_reference.get(0, handle_pos)
                            dc_handle_delta_deg = math.degrees(handle_pos - reference)

            print(
                "[open_door_joint_debug] phase=runtime_sample "
                f"version={_OPEN_DOOR_DIAGNOSTIC_VERSION} "
                f"step={step} leaf_rotateZYX_deg={leaf_rotate} handle_rotateZYX_deg={handle_rotate} "
                f"body_to_leaf_angle_deg={_quat_angle_deg_from_matrix(body_to_leaf):.4f} "
                f"leaf_to_handle_angle_deg={leaf_to_handle_angle} "
                f"dc_leaf_pos_deg={dc_leaf_pos_deg} "
                f"dc_handle_pos_deg={dc_handle_pos_deg} "
                f"dc_handle_delta_deg={dc_handle_delta_deg}"
            )
            rigid_debug_paths = [
                path
                for path in _OPEN_DOOR_DEBUG_PRIM_PATHS
                if "/E_leaf_2" in path or "/E_handle_4" in path
            ]
            print(
                "[open_door_joint_debug] phase=runtime_rigid_pose_probe "
                f"version={_OPEN_DOOR_DIAGNOSTIC_VERSION} "
                f"step={step} "
                f"dc_handles_present={dc_handles is not None} "
                f"dc_present={dc is not None} "
                f"dc_mod_present={dc_mod is not None} "
                f"get_rigid_body_api={hasattr(dc, 'get_rigid_body') if dc is not None else False} "
                f"get_rigid_body_pose_api={hasattr(dc, 'get_rigid_body_pose') if dc is not None else False} "
                f"path_count={len(rigid_debug_paths)}"
            )
            for path in _OPEN_DOOR_DEBUG_PRIM_PATHS:
                if "/E_handle_4" not in path:
                    continue
                try:
                    prim = stage.GetPrimAtPath(path)
                    if prim is None or not prim.IsValid():
                        print(
                            "[open_door_joint_debug] phase=runtime_component "
                            f"step={step} path={path} status=missing"
                        )
                        continue
                    local_matrix = _xformable_local_matrix(UsdGeom.Xformable(prim))
                    world_matrix = cache.GetLocalToWorldTransform(prim)
                    leaf_relative = leaf_world.GetInverse() * world_matrix
                    print(
                        "[open_door_joint_debug] phase=runtime_component "
                        f"step={step} path={path} "
                        f"leaf_relative_angle_deg={_quat_angle_deg_from_matrix(leaf_relative):.4f} "
                        f"local_matrix={_matrix_to_rows(local_matrix)} "
                        f"world_matrix={_matrix_to_rows(world_matrix)}"
                    )
                except Exception as exc:
                    print(
                        "[open_door_joint_debug] phase=runtime_component "
                        f"step={step} path={path} status=failed error={exc}"
                    )
            if not rigid_debug_paths:
                print(
                    "[open_door_joint_debug] phase=runtime_rigid_pose "
                    f"version={_OPEN_DOOR_DIAGNOSTIC_VERSION} "
                    f"step={step} status=no_debug_paths"
                )
                return
            if dc_handles is None:
                for path in rigid_debug_paths:
                    print(
                        "[open_door_joint_debug] phase=runtime_rigid_pose "
                        f"version={_OPEN_DOOR_DIAGNOSTIC_VERSION} "
                        f"step={step} path={path} status=no_dc_handles"
                    )
                return
            if dc is None or dc_mod is None:
                for path in rigid_debug_paths:
                    print(
                        "[open_door_joint_debug] phase=runtime_rigid_pose "
                        f"version={_OPEN_DOOR_DIAGNOSTIC_VERSION} "
                        f"step={step} path={path} status=no_dc_interface "
                        f"dc_present={dc is not None} dc_mod_present={dc_mod is not None}"
                    )
                return
            if not hasattr(dc, "get_rigid_body") or not hasattr(dc, "get_rigid_body_pose"):
                for path in rigid_debug_paths:
                    print(
                        "[open_door_joint_debug] phase=runtime_rigid_pose "
                        f"version={_OPEN_DOOR_DIAGNOSTIC_VERSION} "
                        f"step={step} path={path} status=no_rigid_body_api "
                        f"get_rigid_body_api={hasattr(dc, 'get_rigid_body')} "
                        f"get_rigid_body_pose_api={hasattr(dc, 'get_rigid_body_pose')}"
                    )
                return
            if dc is not None and dc_mod is not None:
                invalid_handle = getattr(dc_mod, "INVALID_HANDLE", None)
                for path in rigid_debug_paths:
                    try:
                        rigid_body = dc.get_rigid_body(path)
                    except Exception as exc:
                        print(
                            "[open_door_joint_debug] phase=runtime_rigid_pose "
                            f"version={_OPEN_DOOR_DIAGNOSTIC_VERSION} "
                            f"step={step} path={path} status=get_rigid_body_failed error={exc}"
                        )
                        continue
                    if rigid_body == invalid_handle:
                        print(
                            "[open_door_joint_debug] phase=runtime_rigid_pose "
                            f"version={_OPEN_DOOR_DIAGNOSTIC_VERSION} "
                            f"step={step} path={path} status=missing"
                        )
                        continue
                    try:
                        pose = dc.get_rigid_body_pose(rigid_body)
                        pos = getattr(pose, "p", None)
                        rot = getattr(pose, "r", None)
                        pos_tuple = (
                            (float(pos.x), float(pos.y), float(pos.z))
                            if pos is not None and all(hasattr(pos, axis) for axis in ("x", "y", "z"))
                            else None
                        )
                        rot_tuple = (
                            (float(rot.w), float(rot.x), float(rot.y), float(rot.z))
                            if rot is not None and all(hasattr(rot, axis) for axis in ("w", "x", "y", "z"))
                            else None
                        )
                        print(
                            "[open_door_joint_debug] phase=runtime_rigid_pose "
                            f"version={_OPEN_DOOR_DIAGNOSTIC_VERSION} "
                            f"step={step} path={path} status=ok "
                            f"pos={pos_tuple} quat_wxyz={rot_tuple}"
                        )
                    except Exception as exc:
                        print(
                            "[open_door_joint_debug] phase=runtime_rigid_pose "
                            f"version={_OPEN_DOOR_DIAGNOSTIC_VERSION} "
                            f"step={step} path={path} status=get_pose_failed error={exc}"
                        )
        except Exception as exc:
            print(f"[open_door_joint_debug] phase=runtime_sample step={step} failed: {exc}")
