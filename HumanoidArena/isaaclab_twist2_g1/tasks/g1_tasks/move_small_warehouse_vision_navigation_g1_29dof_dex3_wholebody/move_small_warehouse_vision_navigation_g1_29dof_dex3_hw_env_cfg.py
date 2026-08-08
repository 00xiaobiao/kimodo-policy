import numpy as np
import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from common_env_objects import (
    apply_deterministic_object_resets_with_seed,
    apply_explicit_env_object_states,
    begin_new_episode_object_seed,
    collect_recordable_env_object_states,
    resolve_env_object_scene_key,
    set_current_episode_object_seed,
)
from tasks.common_config import CameraPresets, G1RobotPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_small_warehouse_vision_navigation import (
    SmallWarehouseVisionNavigationSceneCfg,
)

from . import mdp
from .obstacle_layout import (
    OBSTACLE_RECORD_NAMES,
    build_obstacle_layout_states,
    should_randomize_target_sign,
)

ROBOT_INIT_POS = (-1.90, -5.20, 0.80)
ROBOT_INIT_ROT = (0.7071, 0.0, 0.0, 0.7071)
TARGET_SIGN_POSE_RANGE = {
    "x": [-3.5, -0.5],
    "y": [-0.4, -0.1],
    "z": [0.0, 0.0],
    "yaw": [-0.523599, 0.523599],
}
OBSTACLE_01_A_LAYOUT_Z = 0.16
OBSTACLE_01_B_LAYOUT_Z = 0.36
OBSTACLE_01_C_LAYOUT_Z = 0.56
OBSTACLE_02_A_LAYOUT_Z = 0.76
OBSTACLE_02_B_LAYOUT_Z = 0.96
OBSTACLE_02_C_LAYOUT_Z = 1.16
OBSTACLE_LAYOUT_Z_ATTRS = {
    "obstacle_01_a": "obstacle_01_a_layout_z",
    "obstacle_01_b": "obstacle_01_b_layout_z",
    "obstacle_01_c": "obstacle_01_c_layout_z",
    "obstacle_02_a": "obstacle_02_a_layout_z",
    "obstacle_02_b": "obstacle_02_b_layout_z",
    "obstacle_02_c": "obstacle_02_c_layout_z",
}
OBSTACLE_01_A_LAYOUT_X_RANGE = (-2.2, 2.5)
OBSTACLE_01_B_LAYOUT_X_RANGE = (-2.2, 2.5)
OBSTACLE_01_C_LAYOUT_X_RANGE = (-2.2, 2.5)
OBSTACLE_02_A_LAYOUT_X_RANGE = (-2.2, 2.5)
OBSTACLE_02_B_LAYOUT_X_RANGE = (-2.2, 2.5)
OBSTACLE_02_C_LAYOUT_X_RANGE = (-2.2, 2.5)
OBSTACLE_LAYOUT_X_RANGE_ATTRS = {
    "obstacle_01_a": "obstacle_01_a_layout_x_range",
    "obstacle_01_b": "obstacle_01_b_layout_x_range",
    "obstacle_01_c": "obstacle_01_c_layout_x_range",
    "obstacle_02_a": "obstacle_02_a_layout_x_range",
    "obstacle_02_b": "obstacle_02_b_layout_x_range",
    "obstacle_02_c": "obstacle_02_c_layout_x_range",
}
OBSTACLE_LAYOUT_Y_RANGE = (-0.8, 1.5)
OBSTACLE_LAYOUT_YAW_RANGE = (-0.785398, 0.785398)


def _quat_wxyz_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def _quat_wxyz_inverse(quat: np.ndarray) -> np.ndarray:
    quat = _quat_wxyz_normalize(quat)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32)


def _quat_wxyz_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    out = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )
    return _quat_wxyz_normalize(out)


def _quat_wxyz_rotate_vector(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat = _quat_wxyz_normalize(quat)
    vec = np.asarray(vec, dtype=np.float32)
    quat_xyz = quat[1:]
    t = 2.0 * np.cross(quat_xyz, vec)
    return (vec + quat[0] * t + np.cross(quat_xyz, t)).astype(np.float32)

@configclass
class SmallWarehouseVisionNavigationTaskSceneCfg(SmallWarehouseVisionNavigationSceneCfg):
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
    success = DoneTerm(func=mdp.success_reached)


@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward, weight=1.0)


@configclass
class EventCfg:
    pass


@configclass
class MoveSmallWarehouseVisionNavigationG129Dex3WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    scene: SmallWarehouseVisionNavigationTaskSceneCfg = SmallWarehouseVisionNavigationTaskSceneCfg(
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
        self.object_reset_seed_source = "time"
        self.target_sign_startup_randomization_enabled = False
        self.target_sign_reset_randomization_enabled = True
        self.target_sign_static_collision_patch_enabled = True
        self.target_sign_debug_pose_logging = True
        self.obstacle_01_a_layout_x_range = list(OBSTACLE_01_A_LAYOUT_X_RANGE)
        self.obstacle_01_b_layout_x_range = list(OBSTACLE_01_B_LAYOUT_X_RANGE)
        self.obstacle_01_c_layout_x_range = list(OBSTACLE_01_C_LAYOUT_X_RANGE)
        self.obstacle_02_a_layout_x_range = list(OBSTACLE_02_A_LAYOUT_X_RANGE)
        self.obstacle_02_b_layout_x_range = list(OBSTACLE_02_B_LAYOUT_X_RANGE)
        self.obstacle_02_c_layout_x_range = list(OBSTACLE_02_C_LAYOUT_X_RANGE)
        self.obstacle_layout_y_range = list(OBSTACLE_LAYOUT_Y_RANGE)
        self.obstacle_layout_yaw_range = list(OBSTACLE_LAYOUT_YAW_RANGE)
        self.obstacle_01_a_layout_z = float(OBSTACLE_01_A_LAYOUT_Z)
        self.obstacle_01_b_layout_z = float(OBSTACLE_01_B_LAYOUT_Z)
        self.obstacle_01_c_layout_z = float(OBSTACLE_01_C_LAYOUT_Z)
        self.obstacle_02_a_layout_z = float(OBSTACLE_02_A_LAYOUT_Z)
        self.obstacle_02_b_layout_z = float(OBSTACLE_02_B_LAYOUT_Z)
        self.obstacle_02_c_layout_z = float(OBSTACLE_02_C_LAYOUT_Z)
        self.deterministic_object_resets = [
            {
                "record_name": "target_sign",
                "prim_paths": ["/World/envs/env_{env_idx}/TargetSign"],
                "allow_prroot_fallback": False,
                "prim_pose_write_mode": "components",
                "pose_range": TARGET_SIGN_POSE_RANGE,
                "zero_velocity_on_reset": True,
            },
            {
                "record_name": "obstacle_01_a",
                "prim_paths": ["/World/envs/env_{env_idx}/Obstacle01_A"],
                "allow_prroot_fallback": False,
                "prim_pose_write_mode": "local_matrix",
                "pose_range": {},
                "zero_velocity_on_reset": True,
            },
            {
                "record_name": "obstacle_01_b",
                "prim_paths": ["/World/envs/env_{env_idx}/Obstacle01_B"],
                "allow_prroot_fallback": False,
                "prim_pose_write_mode": "local_matrix",
                "pose_range": {},
                "zero_velocity_on_reset": True,
            },
            {
                "record_name": "obstacle_01_c",
                "prim_paths": ["/World/envs/env_{env_idx}/Obstacle01_C"],
                "allow_prroot_fallback": False,
                "prim_pose_write_mode": "local_matrix",
                "pose_range": {},
                "zero_velocity_on_reset": True,
            },
            {
                "record_name": "obstacle_02_a",
                "prim_paths": ["/World/envs/env_{env_idx}/Obstacle02_A"],
                "allow_prroot_fallback": False,
                "prim_pose_write_mode": "local_matrix",
                "pose_range": {},
                "zero_velocity_on_reset": True,
            },
            {
                "record_name": "obstacle_02_b",
                "prim_paths": ["/World/envs/env_{env_idx}/Obstacle02_B"],
                "allow_prroot_fallback": False,
                "prim_pose_write_mode": "local_matrix",
                "pose_range": {},
                "zero_velocity_on_reset": True,
            },
            {
                "record_name": "obstacle_02_c",
                "prim_paths": ["/World/envs/env_{env_idx}/Obstacle02_C"],
                "allow_prroot_fallback": False,
                "prim_pose_write_mode": "local_matrix",
                "pose_range": {},
                "zero_velocity_on_reset": True,
            },
        ]
        self._sync_navigation_obstacle_reset_specs()
        self._replay_initial_env_state_active = False
        self._navigation_startup_reset_pending = True

        self.sim.dt = 0.005
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_enhanced_determinism = True
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
        self._sync_navigation_obstacle_reset_specs()
        self._replay_initial_env_state_active = (
            bool(getattr(args_cli, "replay_file", "")) if args_cli else False
        )
        self._navigation_startup_reset_pending = True
        self._make_navigation_assets_static_collision(env)
        self._disable_navigation_scene_cameras(env)
        if self.target_sign_startup_randomization_enabled:
            self._log_target_sign_pose("startup_randomization_pending")
        else:
            self._log_target_sign_pose("startup_init_state")
        self._cache_obstacle_default_states(env)

    def _reset_object_self(self, env):
        applied = self._apply_navigation_layout(
            env,
        )
        if applied:
            print("[object_reset] " + ", ".join(applied))

    def _reset_all_self(self, env):
        base_mdp.reset_scene_to_default(
            env,
            torch.arange(env.num_envs, device=env.device),
        )
        applied = self._apply_navigation_layout(
            env,
        )
        if applied:
            print("[object_reset] " + ", ".join(applied))

    def _apply_navigation_layout_for_seed(
        self,
        env,
        *,
        episode_seed: int,
        seed_source: str,
        randomize_target_sign: bool,
    ) -> dict[str, bool]:
        self._sync_navigation_obstacle_reset_specs()
        set_current_episode_object_seed(self, episode_seed, seed_source)

        applied_summary = {
            "target_sign": False,
            "obstacle_layout": False,
            "obstacle_spawn_height": False,
        }

        if randomize_target_sign:
            target_applied = apply_deterministic_object_resets_with_seed(
                self,
                env,
                episode_seed=episode_seed,
                seed_source=seed_source,
                selected_record_names={"target_sign"},
            )
            applied_summary["target_sign"] = bool(target_applied)

        obstacle_default_states = getattr(self, "_cached_obstacle_default_states", None)
        if (not isinstance(obstacle_default_states, dict)) or any(
            not isinstance(obstacle_default_states.get(record_name), dict)
            for record_name in OBSTACLE_RECORD_NAMES
        ):
            self._cache_obstacle_default_states(env)
            obstacle_default_states = getattr(self, "_cached_obstacle_default_states", {})

        obstacle_states = build_obstacle_layout_states(
            default_states=obstacle_default_states,
            episode_seed=int(episode_seed),
            x_ranges=self._get_obstacle_layout_x_ranges(),
            y_range=tuple(self.obstacle_layout_y_range),
            z_ranges=self._get_obstacle_layout_z_ranges(),
            yaw_range=tuple(self.obstacle_layout_yaw_range),
        )
        obstacle_prim_states = self._build_obstacle_prim_layout_states(
            obstacle_states=obstacle_states,
            prim_default_states=obstacle_default_states,
        )
        applied_summary["obstacle_layout"] = bool(
            apply_explicit_env_object_states(
                env,
                self,
                obstacle_prim_states,
                log_prefix="obstacle_layout_prim",
            )
        )
        applied_summary["obstacle_spawn_height"] = bool(
            self._apply_navigation_obstacle_spawn_heights(
                env,
                layout_states=obstacle_states,
                prim_default_states=obstacle_default_states,
            )
        )
        return applied_summary

    def _apply_navigation_layout(self, env):
        self._sync_navigation_obstacle_reset_specs()
        if self._replay_initial_env_state_active:
            return []

        applied = []
        is_startup_reset = bool(getattr(self, "_navigation_startup_reset_pending", False))
        episode_seed, seed_source = begin_new_episode_object_seed(self)
        randomize_target_sign = should_randomize_target_sign(
            is_startup_reset=is_startup_reset,
            startup_randomization_enabled=self.target_sign_startup_randomization_enabled,
            reset_randomization_enabled=self.target_sign_reset_randomization_enabled,
        )
        applied_summary = self._apply_navigation_layout_for_seed(
            env,
            episode_seed=episode_seed,
            seed_source=seed_source,
            randomize_target_sign=randomize_target_sign,
        )
        if applied_summary["target_sign"]:
            applied.append("target_sign")
            self._log_target_sign_pose(
                "startup_randomized" if is_startup_reset else "reset_randomized"
            )
        elif self.target_sign_debug_pose_logging:
            self._log_target_sign_pose(
                "startup_init_state" if is_startup_reset else "reset_init_state"
            )
        if applied_summary["obstacle_layout"]:
            applied.append("obstacle_layout")
        if applied_summary["obstacle_spawn_height"]:
            applied.append("obstacle_spawn_height")
        self._navigation_startup_reset_pending = False
        return applied

    def restore_replay_initial_env_state(
        self,
        env,
        *,
        episode_seed: int | None = None,
        episode_seed_source: str = "",
        object_states: dict | None = None,
    ) -> set[str]:
        handled_names: set[str] = set()
        explicit_states = dict(object_states or {})
        known_names = {"target_sign", *OBSTACLE_RECORD_NAMES}

        if episode_seed is not None:
            is_startup_reset = bool(getattr(self, "_navigation_startup_reset_pending", False))
            randomize_target_sign = should_randomize_target_sign(
                is_startup_reset=is_startup_reset,
                startup_randomization_enabled=self.target_sign_startup_randomization_enabled,
                reset_randomization_enabled=self.target_sign_reset_randomization_enabled,
            )
            applied_summary = self._apply_navigation_layout_for_seed(
                env,
                episode_seed=int(episode_seed),
                seed_source=str(episode_seed_source or "recorded"),
                randomize_target_sign=randomize_target_sign,
            )
            if applied_summary["target_sign"]:
                handled_names.add("target_sign")
            if applied_summary["obstacle_layout"] or applied_summary["obstacle_spawn_height"]:
                handled_names.update(OBSTACLE_RECORD_NAMES)
            self._navigation_startup_reset_pending = False

        fallback_states = {
            name: state
            for name, state in explicit_states.items()
            if name in known_names and name not in handled_names
        }
        if fallback_states:
            if apply_explicit_env_object_states(
                env,
                self,
                fallback_states,
                log_prefix="replay_env_init_navigation_fallback",
            ):
                handled_names.update(fallback_states)
        return handled_names

    def _get_obstacle_layout_x_ranges(self):
        x_ranges = {}
        for record_name in OBSTACLE_RECORD_NAMES:
            attr_name = OBSTACLE_LAYOUT_X_RANGE_ATTRS[record_name]
            raw_range = getattr(self, attr_name)
            x_ranges[record_name] = (float(raw_range[0]), float(raw_range[1]))
        return x_ranges

    def _get_obstacle_layout_z_ranges(self):
        z_ranges = {}
        for record_name in OBSTACLE_RECORD_NAMES:
            attr_name = OBSTACLE_LAYOUT_Z_ATTRS[record_name]
            z_value = float(getattr(self, attr_name))
            z_ranges[record_name] = (z_value, z_value)
        return z_ranges

    def _sync_navigation_obstacle_reset_specs(self):
        specs = getattr(self, "deterministic_object_resets", None)
        if not isinstance(specs, list):
            return

        specs_by_name = {
            str(spec.get("record_name")): spec
            for spec in specs
            if isinstance(spec, dict) and spec.get("record_name")
        }
        z_ranges = self._get_obstacle_layout_z_ranges()

        for record_name in OBSTACLE_RECORD_NAMES:
            spec = specs_by_name.get(record_name)
            if not isinstance(spec, dict):
                continue

            pose_range = spec.get("pose_range")
            if not isinstance(pose_range, dict):
                pose_range = {}
                spec["pose_range"] = pose_range

            pose_range["x"] = list(self._get_obstacle_layout_x_ranges()[record_name])
            pose_range["y"] = list(self.obstacle_layout_y_range)
            pose_range["z"] = list(z_ranges[record_name])
            pose_range["yaw"] = list(self.obstacle_layout_yaw_range)

    def _cache_obstacle_default_states(self, env):
        current_states = collect_recordable_env_object_states(env, self)
        self._cached_obstacle_default_states = {
            record_name: current_states.get(record_name)
            for record_name in OBSTACLE_RECORD_NAMES
        }
        self._cached_obstacle_scene_root_states = self._collect_obstacle_scene_root_states(env)

    def _collect_obstacle_scene_root_states(self, env):
        scene_states = {}
        for record_name in OBSTACLE_RECORD_NAMES:
            scene_key = resolve_env_object_scene_key(env, self, record_name)
            if scene_key is None:
                scene_states[record_name] = None
                continue
            try:
                asset = env.scene[scene_key]
                root_state = asset.data.root_state_w
                scene_states[record_name] = {
                    "position": root_state[0, 0:3].detach().cpu().numpy().astype(np.float32).copy(),
                    "orientation": root_state[0, 3:7].detach().cpu().numpy().astype(np.float32).copy(),
                    "linear_velocity": root_state[0, 7:10].detach().cpu().numpy().astype(np.float32).copy(),
                    "angular_velocity": root_state[0, 10:13].detach().cpu().numpy().astype(np.float32).copy(),
                }
            except Exception:
                scene_states[record_name] = None
        return scene_states

    def _build_obstacle_prim_layout_states(self, *, obstacle_states, prim_default_states):
        prim_states = {}
        for record_name, layout_state in (obstacle_states or {}).items():
            prim_default_state = (prim_default_states or {}).get(record_name)
            if not isinstance(layout_state, dict) or not isinstance(prim_default_state, dict):
                continue
            target_position = np.asarray(layout_state["position"], dtype=np.float32).copy()
            target_position[2] = float(np.asarray(prim_default_state["position"], dtype=np.float32)[2])
            prim_states[record_name] = {
                "position": target_position,
                "orientation": np.asarray(layout_state["orientation"], dtype=np.float32).copy(),
                "linear_velocity": np.zeros(3, dtype=np.float32),
                "angular_velocity": np.zeros(3, dtype=np.float32),
            }
        return prim_states

    def _apply_navigation_obstacle_spawn_heights(self, env, *, layout_states, prim_default_states):
        scene_default_states = getattr(self, "_cached_obstacle_scene_root_states", None)
        if not isinstance(scene_default_states, dict):
            scene_default_states = self._collect_obstacle_scene_root_states(env)
            self._cached_obstacle_scene_root_states = scene_default_states

        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        applied = []
        scene_object_changed = False

        for record_name in OBSTACLE_RECORD_NAMES:
            layout_state = (layout_states or {}).get(record_name)
            prim_default_state = (prim_default_states or {}).get(record_name)
            scene_default_state = (scene_default_states or {}).get(record_name)
            if not (
                isinstance(layout_state, dict)
                and isinstance(prim_default_state, dict)
                and isinstance(scene_default_state, dict)
            ):
                continue

            scene_key = resolve_env_object_scene_key(env, self, record_name)
            if scene_key is None:
                continue

            try:
                asset = env.scene[scene_key]
                root_state = asset.data.default_root_state.clone()
            except Exception:
                try:
                    root_state = asset.data.root_state_w.clone()
                except Exception:
                    root_state = None
            if root_state is None:
                continue

            prim_default_position = np.asarray(prim_default_state["position"], dtype=np.float32)
            prim_default_orientation = _quat_wxyz_normalize(
                np.asarray(prim_default_state["orientation"], dtype=np.float32)
            )
            scene_default_position = np.asarray(scene_default_state["position"], dtype=np.float32)
            scene_default_orientation = _quat_wxyz_normalize(
                np.asarray(scene_default_state["orientation"], dtype=np.float32)
            )

            target_prim_position = np.asarray(layout_state["position"], dtype=np.float32).copy()
            desired_spawn_z = float(target_prim_position[2])
            target_prim_position[2] = float(prim_default_position[2])
            target_prim_orientation = _quat_wxyz_normalize(
                np.asarray(layout_state["orientation"], dtype=np.float32)
            )

            local_root_offset = _quat_wxyz_rotate_vector(
                _quat_wxyz_inverse(prim_default_orientation),
                scene_default_position - prim_default_position,
            )
            rotated_root_offset = _quat_wxyz_rotate_vector(
                target_prim_orientation,
                local_root_offset,
            )
            target_root_position = target_prim_position + rotated_root_offset
            target_root_position[2] = desired_spawn_z

            local_root_orientation = _quat_wxyz_mul(
                scene_default_orientation,
                _quat_wxyz_inverse(prim_default_orientation),
            )
            target_root_orientation = _quat_wxyz_mul(
                local_root_orientation,
                target_prim_orientation,
            )

            target_root_state = root_state.clone()
            target_root_state[:, 0:3] = torch.as_tensor(
                target_root_position,
                device=target_root_state.device,
                dtype=target_root_state.dtype,
            ).reshape(1, 3).repeat(env.num_envs, 1)
            target_root_state[:, 3:7] = torch.as_tensor(
                target_root_orientation,
                device=target_root_state.device,
                dtype=target_root_state.dtype,
            ).reshape(1, 4).repeat(env.num_envs, 1)
            target_root_state[:, 7:13] = 0.0

            asset.write_root_state_to_sim(target_root_state, env_ids=env_ids)
            scene_object_changed = True
            applied.append(
                f"{record_name}->{scene_key}:root_pos="
                f"{target_root_state[0, 0:3].detach().cpu().numpy().tolist()}"
            )

        if scene_object_changed:
            env.scene.write_data_to_sim()
        if not applied:
            return False

        print("[obstacle_spawn_height] " + ", ".join(applied))
        return True

    def _log_target_sign_pose(self, tag: str):
        if not self.target_sign_debug_pose_logging:
            return
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(f"[target_sign_pose:{tag}] stage unavailable")
                return

            for path in (
                "/World/envs/env_0/TargetSign",
                "/World/envs/env_0/TargetSign/PRootNode",
            ):
                prim = stage.GetPrimAtPath(path)
                if prim is None or not prim.IsValid() or not prim.IsActive():
                    continue
                cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                matrix = cache.GetLocalToWorldTransform(prim)
                translation = matrix.ExtractTranslation()
                print(
                    f"[target_sign_pose:{tag}] {path} "
                    f"world_pos={[float(translation[0]), float(translation[1]), float(translation[2])]}"
                )
                return
            print(f"[target_sign_pose:{tag}] target prim not found")
        except Exception as exc:
            print(f"[target_sign_pose:{tag}] failed: {exc}")

    def _make_navigation_assets_static_collision(self, env):
        try:
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            removed = []
            for env_idx in range(env.num_envs):
                prim_roots = []
                if self.target_sign_static_collision_patch_enabled:
                    prim_roots.insert(0, f"/World/envs/env_{env_idx}/TargetSign")
                for prim_root in prim_roots:
                    root = stage.GetPrimAtPath(prim_root)
                    if not root or not root.IsValid():
                        continue
                    for prim in Usd.PrimRange(root):
                        for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI):
                            if prim.HasAPI(api):
                                prim.RemoveAPI(api)
                                api_name = getattr(api, "__name__", str(api))
                                removed.append(f"{prim.GetPath().pathString}:{api_name}")
            if removed:
                print(
                    "[navigation_static_collision] removed dynamic physics APIs: "
                    + ", ".join(removed)
                )
            else:
                print("[navigation_static_collision] no dynamic physics APIs found")
        except Exception as exc:
            print(f"[navigation_static_collision] failed: {exc}")

    def _disable_navigation_scene_cameras(self, env):
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print("[navigation_scene_cameras] stage unavailable")
                return

            disabled = []
            for env_idx in range(env.num_envs):
                env_root = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}")
                if env_root is None or not env_root.IsValid():
                    continue
                for prim in Usd.PrimRange(env_root):
                    if not prim.IsA(UsdGeom.Camera):
                        continue
                    prim_path = prim.GetPath().pathString
                    if "/Robot/" in prim_path:
                        continue
                    prim.SetActive(False)
                    disabled.append(prim_path)

            if disabled:
                print("[navigation_scene_cameras] disabled: " + ", ".join(disabled))
            else:
                print("[navigation_scene_cameras] no extra scene cameras found")
        except Exception as exc:
            print(f"[navigation_scene_cameras] failed: {exc}")
