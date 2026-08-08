import numpy as np
import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp
from common_env_objects import (
    apply_deterministic_object_resets_with_seed,
    apply_explicit_env_object_states,
    begin_new_episode_object_seed,
    make_local_spawn_rng,
    set_current_episode_object_seed,
)
from tasks.common_config import CameraPresets, G1RobotPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_boxing_bag import BoxingBagSceneCfg

ROBOT_INIT_POS = (-1.6, -2.0, 0.8)
ROBOT_INIT_ROT = (0.70711, 0.0, 0.0, -0.70711)
BOXING_TARGET_RECORD_NAME = "boxing_target"
BOXING_TARGET_PRIM_TEMPLATE = "/World/envs/env_{env_idx}/BoxingTarget"


@configclass
class BoxingBagTaskSceneCfg(BoxingBagSceneCfg):
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
    # world_camera = CameraPresets.g1_world_camera()


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
    reward = RewTerm(func=mdp.compute_reward, weight=1.0)


@configclass
class EventCfg:
    pass


@configclass
class MoveBoxingBagG129Dex3WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    scene: BoxingBagTaskSceneCfg = BoxingBagTaskSceneCfg(
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
        self.object_reset_seed_source = "env_seed"
        self.recordable_env_objects = [
            {
                "record_name": BOXING_TARGET_RECORD_NAME,
                "prim_paths": [BOXING_TARGET_PRIM_TEMPLATE],
                "prim_pose_write_mode": "local_matrix",
                "allow_prroot_fallback": True,
                "zero_velocity_on_reset": True,
            }
        ]
        self.deterministic_object_resets = []
        self.boxing_target_randomization = {
            "enabled": True,
            "record_name": BOXING_TARGET_RECORD_NAME,
            "prim_path": BOXING_TARGET_PRIM_TEMPLATE,
            "cylinder_selection": "uniform",
            "cylinders": [
                {
                    "name": "bag_1",
                    "center": [-1.6, 2.0, 0.0],
                    "radius": 0.8,
                    "height_range": [1.0, 1.6],
                    "azimuth_range": [-3.141592653589793, 3.141592653589793],
                    "yaw_range": [0.0, 0.0],
                },
                {
                    "name": "bag_2",
                    "center": [-1.6, 2.0, 0.0],
                    "radius": 0.8,
                    "height_range": [1.0, 1.6],
                    "azimuth_range": [-3.141592653589793, 3.141592653589793],
                    "yaw_range": [0.0, 0.0],
                },
            ],
            # Legacy single-cylinder fallback remains supported for older YAMLs.
            "center": [-1.6, 2.0, 0.0],
            "radius": 0.8,
            "height_range": [1.0, 1.6],
            "azimuth_range": [-3.141592653589793, 3.141592653589793],
            "yaw_range": [0.0, 0.0],
        }
        self._replay_initial_env_state_active = False

        self.sim.dt = 0.005
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.physx.enable_enhanced_determinism = True
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
            SimpleEvent(
                func=lambda env: self._reset_object_self(env),
            ),
        )
        self.event_manager.register(
            "reset_all_self",
            SimpleEvent(
                func=lambda env: self._reset_all_self(env),
            ),
        )

    def initialize_task_scene(self, env, args_cli=None):
        self._replay_initial_env_state_active = bool(getattr(args_cli, "replay_file", "")) if args_cli else False
        self._deactivate_room_embedded_cameras(env)

    def _reset_object_self(self, env):
        applied = self._apply_boxing_task_object_resets(env)
        if applied:
            print("[object_reset] " + ", ".join(applied))

    def _reset_all_self(self, env):
        base_mdp.reset_scene_to_default(
            env,
            torch.arange(env.num_envs, device=env.device),
        )
        applied = self._apply_boxing_task_object_resets(env)
        if applied:
            print("[object_reset] " + ", ".join(applied))

    def _apply_boxing_task_object_resets(self, env):
        if self._replay_initial_env_state_active:
            return []

        episode_seed, seed_source = begin_new_episode_object_seed(self)
        applied = []
        standard_names = {
            str(spec.get("record_name"))
            for spec in (self.deterministic_object_resets or [])
            if isinstance(spec, dict) and spec.get("record_name")
        }
        if standard_names:
            generic_applied = apply_deterministic_object_resets_with_seed(
                self,
                env,
                episode_seed=episode_seed,
                seed_source=seed_source,
                selected_record_names=standard_names,
            )
            applied.extend(generic_applied)

        if self._apply_boxing_target_randomization_for_seed(
            env,
            episode_seed=episode_seed,
            seed_source=seed_source,
            log_prefix="boxing_target_reset",
        ):
            applied.append(
                f"{self._get_boxing_target_record_name()}:episode_seed={episode_seed}:seed_source={seed_source}"
            )
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
        record_name = self._get_boxing_target_record_name()
        explicit_states = dict(object_states or {})

        if record_name in explicit_states:
            if apply_explicit_env_object_states(
                env,
                self,
                {record_name: explicit_states[record_name]},
                log_prefix="replay_boxing_target_explicit",
            ):
                handled_names.add(record_name)
                return handled_names

        if episode_seed is not None and self._boxing_target_randomization_enabled():
            if self._apply_boxing_target_randomization_for_seed(
                env,
                episode_seed=int(episode_seed),
                seed_source=str(episode_seed_source or "recorded"),
                log_prefix="replay_boxing_target_seed",
            ):
                handled_names.add(record_name)
        return handled_names

    def _boxing_target_randomization_enabled(self) -> bool:
        cfg = self._get_boxing_target_randomization_cfg()
        return bool(cfg.get("enabled", True))

    def _get_boxing_target_record_name(self) -> str:
        cfg = self._get_boxing_target_randomization_cfg()
        return str(cfg.get("record_name") or BOXING_TARGET_RECORD_NAME)

    def _get_boxing_target_randomization_cfg(self) -> dict:
        cfg = getattr(self, "boxing_target_randomization", {}) or {}
        return cfg if isinstance(cfg, dict) else {}

    @staticmethod
    def _normalize_range(range_value, default_low: float, default_high: float) -> tuple[float, float]:
        if isinstance(range_value, (list, tuple)) and len(range_value) >= 2:
            low = float(range_value[0])
            high = float(range_value[1])
        else:
            low, high = float(default_low), float(default_high)
        return (min(low, high), max(low, high))

    @staticmethod
    def _normalize_center(center_value) -> np.ndarray:
        arr = np.asarray(center_value if center_value is not None else [0.0, 0.0, 0.0], dtype=np.float32).reshape(-1)
        if arr.size < 3:
            arr = np.pad(arr, (0, 3 - arr.size), mode="constant")
        return arr[:3].astype(np.float32)

    def _normalize_cylinder_cfg(self, cylinder_cfg: dict, *, fallback_cfg: dict | None = None) -> dict:
        fallback_cfg = fallback_cfg or {}
        center = self._normalize_center(cylinder_cfg.get("center", fallback_cfg.get("center")))
        radius = max(
            0.0,
            float(cylinder_cfg.get("radius", fallback_cfg.get("radius", 0.0)) or 0.0),
        )
        z_low, z_high = self._normalize_range(
            cylinder_cfg.get("height_range", fallback_cfg.get("height_range")),
            center[2],
            center[2],
        )
        az_low, az_high = self._normalize_range(
            cylinder_cfg.get("azimuth_range", fallback_cfg.get("azimuth_range")),
            -float(np.pi),
            float(np.pi),
        )
        yaw_low, yaw_high = self._normalize_range(
            cylinder_cfg.get("yaw_range", fallback_cfg.get("yaw_range")),
            0.0,
            0.0,
        )
        return {
            "name": str(cylinder_cfg.get("name") or fallback_cfg.get("name") or ""),
            "center": center,
            "radius": radius,
            "height_range": (z_low, z_high),
            "azimuth_range": (az_low, az_high),
            "yaw_range": (yaw_low, yaw_high),
        }

    def _get_boxing_target_cylinders(self) -> list[dict]:
        cfg = self._get_boxing_target_randomization_cfg()
        cylinders_cfg = cfg.get("cylinders")
        cylinders = []
        if isinstance(cylinders_cfg, (list, tuple)):
            for entry in cylinders_cfg:
                if isinstance(entry, dict):
                    cylinders.append(self._normalize_cylinder_cfg(entry, fallback_cfg=cfg))
        if cylinders:
            return cylinders
        return [self._normalize_cylinder_cfg(cfg)]

    def _select_boxing_target_cylinder(self, rng, cylinders: list[dict]) -> dict:
        if not cylinders:
            raise RuntimeError("boxing_target_randomization requires at least one cylinder")
        if len(cylinders) == 1:
            return cylinders[0]
        selection_mode = str(
            self._get_boxing_target_randomization_cfg().get("cylinder_selection", "uniform")
        ).strip().lower()
        if selection_mode not in {"", "uniform"}:
            raise ValueError(f"Unsupported boxing target cylinder_selection: {selection_mode}")
        cylinder_idx = int(rng.integers(0, len(cylinders)))
        return cylinders[cylinder_idx]

    def _build_boxing_target_explicit_states_for_seed(self, env, *, episode_seed: int) -> dict[str, dict]:
        if not self._boxing_target_randomization_enabled():
            return {}

        record_name = self._get_boxing_target_record_name()
        cylinders = self._get_boxing_target_cylinders()
        positions = np.zeros((env.num_envs, 3), dtype=np.float32)
        orientations = np.zeros((env.num_envs, 4), dtype=np.float32)
        linear_velocities = np.zeros((env.num_envs, 3), dtype=np.float32)
        angular_velocities = np.zeros((env.num_envs, 3), dtype=np.float32)

        for env_idx in range(env.num_envs):
            rng = make_local_spawn_rng(episode_seed, record_name, env_idx)
            cylinder = self._select_boxing_target_cylinder(rng, cylinders)
            az_low, az_high = cylinder["azimuth_range"]
            z_low, z_high = cylinder["height_range"]
            yaw_low, yaw_high = cylinder["yaw_range"]

            theta = float(rng.uniform(az_low, az_high)) if az_high > az_low else az_low
            z_value = float(rng.uniform(z_low, z_high)) if z_high > z_low else z_low
            yaw_value = float(rng.uniform(yaw_low, yaw_high)) if yaw_high > yaw_low else yaw_low

            center = cylinder["center"]
            radius = float(cylinder["radius"])
            positions[env_idx, 0] = center[0] + radius * float(np.cos(theta))
            positions[env_idx, 1] = center[1] + radius * float(np.sin(theta))
            positions[env_idx, 2] = z_value

            half_yaw = 0.5 * yaw_value
            orientations[env_idx] = np.array(
                [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
                dtype=np.float32,
            )

        return {
            record_name: {
                "position": positions,
                "orientation": orientations,
                "linear_velocity": linear_velocities,
                "angular_velocity": angular_velocities,
            }
        }

    def _apply_boxing_target_randomization_for_seed(
        self,
        env,
        *,
        episode_seed: int,
        seed_source: str,
        log_prefix: str,
    ) -> bool:
        explicit_states = self._build_boxing_target_explicit_states_for_seed(
            env,
            episode_seed=int(episode_seed),
        )
        if not explicit_states:
            return False

        set_current_episode_object_seed(
            self,
            int(episode_seed),
            str(seed_source or "recorded"),
        )
        return bool(
            apply_explicit_env_object_states(
                env,
                self,
                explicit_states,
                log_prefix=log_prefix,
            )
        )

    def _deactivate_room_embedded_cameras(self, env):
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            deactivated = []
            for env_idx in range(env.num_envs):
                cameras_prim = stage.GetPrimAtPath(
                    f"/World/envs/env_{env_idx}/Room/Lab/Cameras"
                )
                if cameras_prim and cameras_prim.IsValid() and cameras_prim.IsActive():
                    cameras_prim.SetActive(False)
                    deactivated.append(cameras_prim.GetPath().pathString)
            if deactivated:
                print(
                    "[boxing_bag_scene] deactivated embedded room camera roots: "
                    f"{deactivated}"
                )
        except Exception as exc:
            print(f"[boxing_bag_scene] room camera cleanup skipped: {exc}")
