# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import os
from datetime import datetime

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg

from . import mdp

from tasks.common_config import G1RobotPresets, CameraPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_runtime import apply_optional_runtime_augments
from tasks.common_scene.base_scene_football_cfg_wholebody import (
    TableFootballSceneCfgWH,
    ROBOT_INIT_X,
    ROBOT_INIT_Y,
    ROBOT_INIT_Z,
)

GOAL_REFERENCE_LINE_RELATIVE_OFFSETS = (
    (0.0, 0.0),
    (0.0, 0.0),
)
GOAL_REFERENCE_LINE_ABSOLUTE_CENTERS = (
    (0.0, 50.0),
    (0.0, -50.0),
)
GOAL_REFERENCE_LINE_LENGTH = 5.0
GOAL_REFERENCE_LINE_WIDTH_RATIO = 0.5
GOAL_REFERENCE_LINE_COLOR = (1.0, 1.0, 1.0)
FOOT_COLLISION_TARGET_APPROXIMATION = "convexDecomposition"
FOOT_COLLISION_LOG_FILENAME = "foot_collision_validation.log"
FOOT_COLLISION_CONVEX_HULL_APPROXIMATION = "convexHull"
ANKLE_COLLISION_LINKS = [
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
]
FOOT_COLLISION_EXPECTED_APPROXIMATIONS = {
    "left_ankle_pitch_link": FOOT_COLLISION_CONVEX_HULL_APPROXIMATION,
    "left_ankle_roll_link": FOOT_COLLISION_TARGET_APPROXIMATION,
    "right_ankle_pitch_link": FOOT_COLLISION_CONVEX_HULL_APPROXIMATION,
    "right_ankle_roll_link": FOOT_COLLISION_TARGET_APPROXIMATION,
}


##
# Scene definition
##


@configclass
class FootballTableSceneCfg(TableFootballSceneCfgWH):
    """Football table scene with G1 29DOF Dex3 wholebody robot."""

    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_wholebody(
        init_pos=(ROBOT_INIT_X, ROBOT_INIT_Y, ROBOT_INIT_Z),
        init_rot=(0.7071, 0.0, 0.0, 0.7071),  # 向左旋轉 90° (繞 Z 軸)
    )
    # robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_wholebody(
    #     init_pos=(ROBOT_INIT_X, ROBOT_INIT_Y, ROBOT_INIT_Z),
    #     init_rot=(1, 0.0, 0.0, 0.0),
    # )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        # history_length=20,
        history_length=10,
        track_air_time=True,
        debug_vis=False,
    )

    front_camera = CameraPresets.g1_front_camera()


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Joint position action configuration."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation configuration."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group."""

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
class MoveFootballG129Dex3WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    """
    Environment configuration for G1 29DOF Dex3 wholebody robot with football task.
    """

    scene: FootballTableSceneCfg = FootballTableSceneCfg(
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

        self.sim.dt = 0.005
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_enhanced_determinism = True
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        # self.sim.physx.friction_correlation_distance = 0.025

        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"

        self.event_manager = SimpleEventManager()
        self._task_adjust_args_cli = None

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

        # Setup foot collision rebuild callback
        self._foot_collision_setup_done = False

    def initialize_task_scene(self, env, args_cli=None):
        self._task_adjust_args_cli = args_cli
        self.adjust_task_scene(env, phase="init", args_cli=args_cli)

    def adjust_task_scene(self, env, phase="init", args_cli=None):
        args_cli = args_cli if args_cli is not None else self._task_adjust_args_cli
        if phase == "init":
            self._apply_init_adjustments(args_cli=args_cli)
        elif phase == "reset":
            self._apply_reset_adjustments()

    def _reset_object_self(self, env):
        base_mdp.reset_root_state_uniform(
            env,
            torch.arange(env.num_envs, device=env.device),
            pose_range={"x": [-0.05, 0.05], "y": [0.0, 0.05]},
            velocity_range={},
            asset_cfg=SceneEntityCfg("object"),
        )

    def _reset_all_self(self, env):
        base_mdp.reset_scene_to_default(
            env,
            torch.arange(env.num_envs, device=env.device),
        )
        self.adjust_task_scene(env, phase="reset")

    def _apply_init_adjustments(self, args_cli=None):
        """Apply football-specific stage patches before the first reset."""
        try:
            from tools.football_physics_material import apply_football_physics_material
            from tools.grass_ground_material import apply_grass_pbr_to_ground
            from tools.semantic_basketball import ensure_semantic_basketballs
            import omni.usd
            from tools.pitch_lines import DEFAULT_LINE_WIDTH, create_simple_debug_lines

            if args_cli is not None:
                apply_optional_runtime_augments(args_cli)
            grass_ok_pre = apply_grass_pbr_to_ground(
                prim_path="/World/GroundPlane",
                uv_scale=(100.0, 100.0),
            )
            print(f"[grass_ground_material] before reset apply result: {grass_ok_pre}")
            try:
                apply_football_physics_material(restitution=0.75)
            except Exception as exc:
                print(f"[football_physics] skipped: {exc}")

            try:
                ensure_semantic_basketballs()
            except Exception as exc:
                print(f"[semantic_basketball] setup skipped: {exc}")

            stage = omni.usd.get_context().get_stage()
            create_simple_debug_lines(
                stage,
                line_color=(1.0, 1.0, 1.0),
                draw_goal_reference_lines=True,
                goal_centers=GOAL_REFERENCE_LINE_ABSOLUTE_CENTERS,
                goal_relative_offsets=GOAL_REFERENCE_LINE_RELATIVE_OFFSETS,
                goal_line_length=GOAL_REFERENCE_LINE_LENGTH,
                goal_line_width=DEFAULT_LINE_WIDTH * GOAL_REFERENCE_LINE_WIDTH_RATIO,
                goal_line_color=GOAL_REFERENCE_LINE_COLOR,
            )
            print(
                f"[pitch_lines] goal reference lines color={GOAL_REFERENCE_LINE_COLOR}, "
                f"centers={GOAL_REFERENCE_LINE_ABSOLUTE_CENTERS}, "
                f"offsets={GOAL_REFERENCE_LINE_RELATIVE_OFFSETS}"
            )
            print("[football_runtime] applying foot collision rebuild before first reset")
            self.setup_foot_collisions()
        except Exception as exc:
            print(f"[football_runtime] setup skipped: {exc}")

    def _apply_reset_adjustments(self):
        """Apply football-specific runtime validation after reset."""
        try:
            from tools.grass_ground_material import apply_grass_pbr_to_ground

            grass_ok_post = apply_grass_pbr_to_ground(
                prim_path="/World/GroundPlane",
                uv_scale=(15.0, 15.0),
            )
            print(f"[grass_ground_material] after reset apply result: {grass_ok_post}")
            try:
                from tools.semantic_basketball import ensure_semantic_basketballs

                ensure_semantic_basketballs()
            except Exception as exc:
                print(f"[semantic_basketball] reset skipped: {exc}")
        except Exception as exc:
            print(f"[football_runtime] post-reset grass skipped: {exc}")

        try:
            print("[football_runtime] post-reset foot collision validation only (no live rebuild)")
            self.log_foot_collision_status()
        except Exception as exc:
            print(f"[football_runtime] foot collision setup skipped: {exc}")

    def _emit_foot_collision_log(self, message: str) -> str:
        """Print foot collision validation output and append it to a log file."""
        project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, FOOT_COLLISION_LOG_FILENAME)

        timestamped_message = f"{datetime.now().isoformat(timespec='seconds')} {message}"
        print(timestamped_message)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(timestamped_message + "\n")
        return log_path

    def _get_foot_collision_api(self, stage, foot_prim_path: str):
        """Resolve the collision prim that actually carries the approximation setting."""
        from pxr import Usd, UsdPhysics

        collision_prim_path = f"{foot_prim_path}/collisions"
        collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, collision_prim_path)
        if collision_api:
            return collision_prim_path, collision_api

        foot_prim = stage.GetPrimAtPath(foot_prim_path)
        if not foot_prim.IsValid():
            return collision_prim_path, None

        for sub_prim in Usd.PrimRange(foot_prim):
            sub_prim_path = str(sub_prim.GetPath())
            sub_collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, sub_prim_path)
            if sub_collision_api:
                return sub_prim_path, sub_collision_api

        return collision_prim_path, None

    def setup_foot_collisions(self):
        """Rebuild ankle-roll collisions using convex decomposition from mesh.

        This must run before the first simulation reset. Reapplying it after PhysX
        tensor views are constructed can invalidate the live simulation view.
        """
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        log_path = ""

        if self._foot_collision_setup_done:
            return self._emit_foot_collision_log(
                "[foot_collision] apply skipped already_initialized"
            )

        for env_idx in range(self.scene.num_envs):
            robot_prim_path = f"/World/envs/env_{env_idx}/Robot"

            for foot_link in ANKLE_COLLISION_LINKS:
                foot_prim_path = f"{robot_prim_path}/{foot_link}"
                expected_approximation = FOOT_COLLISION_EXPECTED_APPROXIMATIONS[foot_link]
                if expected_approximation != FOOT_COLLISION_TARGET_APPROXIMATION:
                    continue

                prim = stage.GetPrimAtPath(foot_prim_path)
                collision_prim_path, _ = self._get_foot_collision_api(stage, foot_prim_path)
                collision_prim = stage.GetPrimAtPath(collision_prim_path)

                if prim.IsValid() and collision_prim.IsValid():
                    collision_api = UsdPhysics.MeshCollisionAPI.Apply(collision_prim)
                    collision_api.GetApproximationAttr().Set(expected_approximation)
                    current_approximation = collision_api.GetApproximationAttr().Get()
                    log_path = self._emit_foot_collision_log(
                        "[foot_collision] apply "
                        f"env={env_idx} link={foot_link} prim={foot_prim_path} "
                        f"actual_prim={collision_prim_path} approximation={current_approximation}"
                    )
                else:
                    log_path = self._emit_foot_collision_log(
                        "[foot_collision] apply "
                        f"env={env_idx} link={foot_link} prim={foot_prim_path} "
                        f"actual_prim={collision_prim_path} missing_collision_api"
                    )

        self._foot_collision_setup_done = True
        return log_path

    def log_foot_collision_status(self):
        """Read back the current foot collision approximation and log the result."""
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        log_path = ""

        for env_idx in range(self.scene.num_envs):
            robot_prim_path = f"/World/envs/env_{env_idx}/Robot"

            for foot_link in ANKLE_COLLISION_LINKS:
                foot_prim_path = f"{robot_prim_path}/{foot_link}"
                expected_approximation = FOOT_COLLISION_EXPECTED_APPROXIMATIONS[foot_link]
                prim = stage.GetPrimAtPath(foot_prim_path)

                if not prim.IsValid():
                    log_path = self._emit_foot_collision_log(
                        "[foot_collision] verify "
                        f"env={env_idx} link={foot_link} prim={foot_prim_path} missing_prim"
                    )
                    continue

                collision_prim_path, collision_api = self._get_foot_collision_api(stage, foot_prim_path)
                approximation = None
                if collision_api:
                    approximation = collision_api.GetApproximationAttr().Get()

                if approximation == expected_approximation:
                    status = "OK"
                elif approximation == FOOT_COLLISION_CONVEX_HULL_APPROXIMATION:
                    status = "CONVEX_HULL"
                else:
                    status = "MISMATCH"
                log_path = self._emit_foot_collision_log(
                    "[foot_collision] verify "
                    f"env={env_idx} link={foot_link} prim={foot_prim_path} "
                    f"actual_prim={collision_prim_path} approximation={approximation} "
                    f"expected={expected_approximation} "
                    f"status={status}"
                )

        return log_path
