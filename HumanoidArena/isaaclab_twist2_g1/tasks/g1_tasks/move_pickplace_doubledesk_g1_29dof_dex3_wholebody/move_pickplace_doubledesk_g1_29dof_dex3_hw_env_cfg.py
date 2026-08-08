import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg

from . import mdp
from tasks.common_config import G1RobotPresets, CameraPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_pickplace_doubledesk import BASKET_RIGID_SUBPRIM, DoubleTableSceneCfg
from common_env_objects import apply_deterministic_object_resets


@configclass
class PickPlaceDoubleDeskSceneCfg(DoubleTableSceneCfg):
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_wholebody(
        init_pos=(-3.7, -2.2, 0.8),
        # init_pos=(-1.8, -6.0, 0.8), # far_distance
        init_rot=(1, 0.0, 0.0, 0.0),
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
    reward = RewTerm(
        func=mdp.compute_reward,
        weight=1.0,
        params={"object_cfg": SceneEntityCfg("object_l")},
    )


@configclass
class EventCfg:
    pass


@configclass
class MovePickPlaceDoubleDeskG129Dex3WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickPlaceDoubleDeskSceneCfg = PickPlaceDoubleDeskSceneCfg(
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
        self.deterministic_object_resets = [
            {
                "record_name": "hammer",
                "scene_keys": ["object_l", "ObjectL"],
                "pose_range": {
                    "x": [-3.24, -3.02],
                    "y": [-3.10, -2.90],
                    "z": [1.0106319, 1.0106319],
                    "yaw": [-0.35, 0.35],
                },
                "zero_velocity_on_reset": True,
            },
            {
                "record_name": "basket",
                "prim_paths": ["/World/envs/env_{env_idx}/Basket"],
                "pose_range": {
                    "x": [-3.82, -3.58],
                    "y": [-3.34, -3.06],
                    "z": [0.80, 0.80],
                    "yaw": [-0.785398, 0.785398],
                },
                "zero_velocity_on_reset": True,
            },
        ]
        self._replay_initial_env_state_active = False

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
            SimpleEvent(
                func=lambda env: self._reset_object_self(env)
            ),
        )

        self.event_manager.register(
            "reset_all_self",
            SimpleEvent(
                func=lambda env: self._reset_all_self(env)
            ),
        )

    def initialize_task_scene(self, env, args_cli=None):
        self._replay_initial_env_state_active = bool(getattr(args_cli, "replay_file", "")) if args_cli else False
        self._deactivate_embedded_room_crate(env)
        self._set_basket_collision_approximation(env)
        self._invalidate_reward_cache(env)
        self._deactivate_room_embedded_cameras(env)
        self._apply_grey_studio_light_rig()

    def _reset_object_self(self, env):
        applied = apply_deterministic_object_resets(
            self,
            env,
            selected_record_names={"hammer", "basket"},
        )
        self._invalidate_reward_cache(env)
        if applied:
            print("[object_reset] " + ", ".join(applied))

    def _reset_all_self(self, env):
        base_mdp.reset_scene_to_default(
            env,
            torch.arange(env.num_envs, device=env.device),
        )
        applied = apply_deterministic_object_resets(self, env)
        self._invalidate_reward_cache(env)
        if applied:
            print("[object_reset] " + ", ".join(applied))

    def _deactivate_embedded_room_crate(self, env):
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            deactivated = []
            for env_idx in range(env.num_envs):
                prim = stage.GetPrimAtPath(
                    f"/World/envs/env_{env_idx}/Room/Root/Lab/SM_Crate_A14_Gray_01_physics"
                )
                if prim and prim.IsValid() and prim.IsActive():
                    prim.SetActive(False)
                    deactivated.append(prim.GetPath().pathString)
            if deactivated:
                print(f"[scene_crate_cleanup] deactivated embedded crates: {deactivated}")
        except Exception as exc:
            print(f"[scene_crate_cleanup] failed: {exc}")

    def _invalidate_reward_cache(self, env):
        try:
            reset_fn = getattr(mdp, "reset_ppdd_reward_cache", None)
            if callable(reset_fn):
                reset_fn(env)
                return
        except Exception as exc:
            print(f"[reward_cache] reset failed: {exc}")
        for key in ("_ppdd_goal_boxes_local", "_ppdd_goal_boxes_source"):
            if hasattr(env, key):
                delattr(env, key)

    def _set_basket_collision_approximation(self, env):
        try:
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            updates = []
            for env_idx in range(env.num_envs):
                basket_root = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/Basket")
                if not basket_root or not basket_root.IsValid():
                    basket_root = stage.GetPrimAtPath(
                        f"/World/envs/env_{env_idx}/Basket/{BASKET_RIGID_SUBPRIM}"
                    )
                if not basket_root or not basket_root.IsValid():
                    continue
                for sub_prim in Usd.PrimRange(basket_root):
                    collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, str(sub_prim.GetPath()))
                    if not collision_api:
                        continue
                    approximation = collision_api.GetApproximationAttr().Get()
                    updates.append(
                        f"{sub_prim.GetPath().pathString}:"
                        f"{approximation}"
                    )
            if updates:
                print("[basket_collision] approximation(preserved)=" + ", ".join(updates))
            else:
                print("[basket_collision] no MeshCollisionAPI found under /Basket")
        except Exception as exc:
            print(f"[basket_collision] failed: {exc}")

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
            print(
                f"[scene_camera_cleanup] deactivated embedded room camera roots: {deactivated}"
            )
        except Exception as exc:
            print(f"[scene_camera_cleanup] failed: {exc}")

    def _apply_grey_studio_light_rig(self):
        try:
            import omni.kit.actions.core
            import omni.usd

            usd_context = omni.usd.get_context()
            action_registry = omni.kit.actions.core.get_action_registry()
            set_lighting_mode_rig = action_registry.get_action(
                "omni.kit.viewport.menubar.lighting",
                "set_lighting_mode_rig",
            )
            if set_lighting_mode_rig is None:
                print("[scene_light_rig] viewport lighting action not found; using stage lights only")
                return

            result = set_lighting_mode_rig.execute("Grey_Studio", usd_context=usd_context)
            print(f"[scene_light_rig] applied Grey_Studio rig: {result}")
        except Exception as exc:
            print(f"[scene_light_rig] failed: {exc}")
