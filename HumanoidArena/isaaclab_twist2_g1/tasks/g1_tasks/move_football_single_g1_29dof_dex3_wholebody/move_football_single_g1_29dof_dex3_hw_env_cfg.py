# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import os
from datetime import datetime

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from common_env_objects import apply_deterministic_object_resets
from tasks.common_scene.base_scene_football_single_cfg_wholebody import (
    TableFootballSceneCfgWH,
    ROBOT_INIT_X,
    ROBOT_INIT_Y,
    ROBOT_INIT_Z,
)

GOAL_REFERENCE_LINE_RELATIVE_OFFSETS = (
    (0.0, 0.0),
)
GOAL_REFERENCE_LINE_ABSOLUTE_CENTERS = (
    (0.0, 50.0),
)
GOAL_REFERENCE_LINE_LENGTH = 5.0
GOAL_REFERENCE_LINE_WIDTH_RATIO = 0.5
GOAL_REFERENCE_LINE_COLOR = (1.0, 1.0, 1.0)


##
# Scene definition
##


@configclass
class FootballTableSceneCfg(TableFootballSceneCfgWH):
    """Football table scene with G1 29DOF Dex3 wholebody robot."""

    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_wholebody(
        init_pos=(ROBOT_INIT_X, ROBOT_INIT_Y-2, ROBOT_INIT_Z),
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
        self.decimation = 10
        self.episode_length_s = 20.0
        self.object_reset_seed_source = "time"
        self.deterministic_object_resets = [
            {
                "record_name": "football",
                "scene_keys": ["object", "football"],
                "pose_range": {
                    "x": [-0.05, 0.05],
                    "y": [0.0, 0.05],
                },
                "zero_velocity_on_reset": True,
            }
        ]
        self._replay_initial_env_state_active = False

        self.sim.dt = 0.001
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
        self._replay_initial_env_state_active = bool(getattr(args_cli, "replay_file", "")) if args_cli else False
        self.adjust_task_scene(env, phase="init", args_cli=args_cli)

    def adjust_task_scene(self, env, phase="init", args_cli=None):
        args_cli = args_cli if args_cli is not None else self._task_adjust_args_cli
        if phase == "init":
            self._apply_init_adjustments(args_cli=args_cli)
        elif phase == "reset":
            self._apply_reset_adjustments()

    def _reset_object_self(self, env):
        applied = apply_deterministic_object_resets(
            self,
            env,
            selected_record_names={"football"},
        )
        if applied:
            print("[object_reset] " + ", ".join(applied))

    def _reset_all_self(self, env):
        base_mdp.reset_scene_to_default(
            env,
            torch.arange(env.num_envs, device=env.device),
        )
        applied = apply_deterministic_object_resets(self, env)
        if applied:
            print("[object_reset] " + ", ".join(applied))
        self.adjust_task_scene(env, phase="reset")

    def _apply_init_adjustments(self, args_cli=None):
        """Apply football-single-specific stage patches before the first reset."""
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
        except Exception as exc:
            print(f"[football_runtime] setup skipped: {exc}")

    def _apply_reset_adjustments(self):
        """Apply football-single-specific runtime validation after reset."""
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
