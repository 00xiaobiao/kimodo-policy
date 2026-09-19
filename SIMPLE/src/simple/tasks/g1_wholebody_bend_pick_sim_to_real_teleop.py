"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces

from simple.assets import AssetManager
from simple.core.actor import Actor, ObjectActor
from simple.core.layout import Layout
from simple.core.object import Object
from simple.core.scene import Scene
from simple.core.task import Task
from simple.dr import (
    CameraDRCfg,
    DistractorDRCfg,
    LanguageDRCfg,
    LightingDRCfg,
    MaterialDRCfg,
    SpatialDRCfg,
    TabletopSceneDRCfg,
    TargetDRCfg,
)
from simple.dr.manager import DRManager
from simple.dr.types import Box
from simple.robots.protocols import Controllable
from simple.robots.registry import RobotRegistry
from simple.sensors import SensorCfg, StereoCameraCfg
from simple.tasks.registry import TaskRegistry

_LIFT_HEIGHT = 0.1

_G1_COLOR_INTRINSICS: dict[str, Any] = {
    "resolution": [640, 480],
    "model": "distortion.inverse_brown_conrady",
    "fx": 606.2996826171875,
    "fy": 606.292236328125,
    "cx": 330.7660217285156,
    "cy": 252.64605712890625,
    "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
}

_G1_DEPTH_INTRINSICS: dict[str, Any] = {
    "resolution": [640, 480],
    "model": "distortion.brown_conrady",
    "fx": 386.00103759765625,
    "fy": 386.00103759765625,
    "cx": 320.4173278808594,
    "cy": 240.45770263671875,
    "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
}

_G1_DEPTH_TO_COLOR_EXTRINSICS: dict[str, Any] = {
    "rotation": [
        [0.999901533126831, -0.012537548318505287, 0.006302413530647755],
        [0.012552149593830109, 0.9999186396598816, -0.002282573375850916],
        [-0.006273282691836357, 0.002361457562074065, 0.9999775290489197],
    ],
    "translation": [
        0.01490413211286068,
        -0.000005230475835560355,
        0.00011887117580045015,
    ],
}

_G1_COLOR_FOV_Y = 2 * np.arctan(
    _G1_COLOR_INTRINSICS["resolution"][1] / (2 * _G1_COLOR_INTRINSICS["fy"])
)


class SimToRealTeleopDRManager(DRManager):
    """Use one consistent DR policy for every teleop reset."""

    def __init__(self, level: int, **kwargs) -> None:
        self.randomizers = {}
        super().__init__(level=level, **kwargs)

    def set_level(self, dr_level: int) -> None:
        self._dr_level = dr_level

    def load_state_dict(self, state_dict: dict[str, Any], dr_level: int | None = None) -> None:
        super().load_state_dict(state_dict, dr_level=None)


@TaskRegistry.register("g1_wholebody_bend_pick_sim_to_real_teleop")
class G1WholebodyBendPickSimToRealTeleop(Task):
    uid: str = "g1_wholebody_bend_pick_sim_to_real_teleop"
    label: str = "G1 Wholebody Bend Pick Sim-to-Real Teleop"
    keep_ground_at_world_origin: bool = True
    description: str = (
        "Teleoperation task for bending to pick a tabletop object in a measured "
        "sim-to-real table setup."
    )

    metadata: dict[str, Any] = {
        "physics_dt": 0.005,
        "control_hz": 200,
        "render_hz": 50,
        "dr_level": 0,
        "version": 1.0,
        "reward_dt": 0.02,
        "image_dt": 0.033333,
        "need_gravity": True,
        "max_episode_steps": 800,
        "success_criteria": 0.3,
    }

    robot_cfg: dict[str, Any] = dict(
        uid="g1_sonic",
    )

    sensor_cfgs: dict[str, SensorCfg] = dict(
        head_stereo=StereoCameraCfg(
            uid="Realsense_D435i",
            mount="eye_in_head",
            width=640,
            height=480,
            focal_length=1.93,
            fov=_G1_COLOR_FOV_Y,
            near=0.2,
            far=5,
            baseline=0.05,
            pose=dict(position=[0.0, 0.0, 0.0]),
            intrinsics=_G1_COLOR_INTRINSICS,
            depth_intrinsics=_G1_DEPTH_INTRINSICS,
            depth_to_color_extrinsics=_G1_DEPTH_TO_COLOR_EXTRINSICS,
        ),
    )

    dr_cfgs = dict(
        language=LanguageDRCfg(
            instructions=[
                "pick up the {}",
            ]
        ),
        target=TargetDRCfg(asset_id="graspnet1b:0"),
        distractors=DistractorDRCfg(
            res_id="graspnet1b",
            number_of_distractors=8,
            allow_duplicates=False,
            exclude=["0"],
        ),
        spatial=SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(low=[-0.60, 0.0, 0.0], high=[-0.66, 0.0, 0.0]),
            target_region=Box(low=[-0.28, -0.05], high=[-0.24, 0.03]),
            distractors_region=[
                Box(low=[-0.32, -0.38], high=[0.03, -0.17]),
                Box(low=[-0.32, 0.13], high=[0.03, 0.38]),
                Box(low=[-0.04, -0.12], high=[0.03, 0.12]),
                Box(low=[-0.34, 0.16], high=[-0.23, 0.38]),
            ],
            target_stable_indices=[0],
            target_rotate_z=Box(low=-0.15, high=0.15),
            placement_attempts=500,
        ),
        camera=CameraDRCfg(
            cam_id="head_stereo",
        ),
        scene=TabletopSceneDRCfg(
            scene_mode="random",
            table_size=Box(low=[0.40, 0.80, 0.015], high=[0.40, 0.80, 0.015]),
            table_position=Box(low=[-0.17, 0.0], high=[-0.17, 0.0]),
            table_height=Box(low=0.7575, high=0.7575),
            rotation_z=Box(low=0.0, high=0.0),
            room_choices=[f"hssd:scene{i}" for i in range(50)],
            scene_manager="hssd",
            randomize_scene_pose=False,
        ),
        lighting=LightingDRCfg(
            light_mode="random",
            light_num=(2, 3),
            light_color_temperature=Box(low=6001, high=8001),
            light_intensity=Box(low=1e4*0.8, high=1e4*1.2),
            light_radius=Box(0.08, 0.12),
            light_length=Box(0.51, 2.1),
            light_spacing=Box((1.0, 1.0), (2.5, 2.5)),
            light_position=Box((-1.1, -1.1, 1.3), (1.1, 1.1, 1.5)),
            light_eulers=Box((0, 0, -0.5 * np.pi), (0, 0, 0.5 * np.pi)),
        ),
        material=MaterialDRCfg(
            material_mode="rand_objects",
            table_material={
                "path": "data/vMaterials_2/Paint/Paint_Eggshell.mdl",
                "name": "Paint_Eggshell_White",
            },
        ),
    )

    def __init__(
        self,
        robot_uid: str = "g1_sonic",
        scene_uid: str | Scene = "hssd:scene0",
        target: str | None = None,
        target_object: str | Object | None = None,
        controller_uid: str = "pd_joint_pos",
        split: str = "train",
        render_hz: int | None = None,
        dr_level: int = 0,
        # success_criteria: float = 0.9,
        *args,
        **kwargs,
    ):
        self._instruction = None
        self._target = None
        self._layout = None
        self._init_target_height = None

        self.robot_cfg.update(dict(uid=robot_uid))
        self.reward = 0
        # self.success_criteria = success_criteria
        self._robot = RobotRegistry.make(**self.robot_cfg, **kwargs)

        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        requested_target = target if target is not None else target_object
        if requested_target is not None:
            assert isinstance(self.dr_cfgs["target"], TargetDRCfg)
            self.dr_cfgs["target"].asset_id = requested_target

            target_id = str(requested_target).split(":")[-1]
            distractor_cfg = self.dr_cfgs.get("distractors")
            if isinstance(distractor_cfg, DistractorDRCfg):
                if distractor_cfg.exclude is None:
                    distractor_cfg.exclude = []
                if target_id not in distractor_cfg.exclude:
                    distractor_cfg.exclude.append(target_id)

        drmgr = SimToRealTeleopDRManager(level=dr_level, **self.dr_cfgs)
        super().__init__(
            dr=drmgr,
            split=split,
            render_hz=render_hz,
            dr_level=dr_level,
            *args,
            **kwargs,
        )

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def instruction(self) -> str:
        assert self._instruction is not None, "call reset() first"
        return self._instruction

    @property
    def target(self) -> Actor:
        assert self._target is not None, "call reset() first"
        return self._target

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    @property
    def observation_space(self) -> spaces.Space:
        default_obs = super().observation_space
        obs: dict[str, Any] = {
            "joint_qpos": spaces.Box(
                -np.pi, np.pi, shape=(self.robot.wholebody_dof,), dtype=np.float32
            ),
        }
        if isinstance(default_obs, spaces.Dict):
            obs.update(dict(default_obs))
        return spaces.Dict(obs)

    def reset(
        self, seed: int | None = None, options: Optional[dict[str, Any]] = None
    ) -> None:
        super().reset(seed, options)
        split = self.metadata.get("split", "train")
        self._target = self.layout.actors.get("target")
        lang_dr = self.dr.get_randomizer("language")
        assert lang_dr is not None
        language_template = lang_dr(split)
        self._instruction = language_template.format(self._target.asset.name)
        self._init_target_height = None
        self.reward = 0
        self.robot.reset(spawn_pose=self.layout.robot.pose)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({})
        return state_dict

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        reward = self.compute_reward(info, *args, **kwargs)
        return reward >= self.metadata["success_criteria"]

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        target_obj_height = info["target"][2]
        if self._init_target_height is None:
            self._init_target_height = target_obj_height
        reward = np.clip(
            (target_obj_height - self._init_target_height) / _LIFT_HEIGHT, 0, 1
        )
        return reward

    def preload_objects(self) -> list[Actor]:
        asset_manager = AssetManager.get("graspnet1b")
        return [ObjectActor(asset=asset) for asset in asset_manager]
