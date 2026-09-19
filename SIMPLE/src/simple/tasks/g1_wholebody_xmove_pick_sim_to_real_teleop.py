"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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
from simple.dr.types import Box
from simple.tasks.g1_wholebody_bend_pick_sim_to_real_teleop import (
    G1WholebodyBendPickSimToRealTeleop,
)
from simple.tasks.registry import TaskRegistry


@TaskRegistry.register("g1_wholebody_xmove_pick_sim_to_real_teleop")
class G1WholebodyXMovePickSimToRealTeleop(G1WholebodyBendPickSimToRealTeleop):
    uid: str = "g1_wholebody_xmove_pick_sim_to_real_teleop"
    label: str = "G1 Wholebody XMove Pick Sim-to-Real Teleop"
    description: str = (
        "Teleoperation task for moving forward to pick a tabletop object in a "
        "measured sim-to-real table setup."
    )

    metadata: dict[str, Any] = {
        **G1WholebodyBendPickSimToRealTeleop.metadata,
        "success_criteria": 0.3,
    }

    dr_cfgs = dict(
        language=LanguageDRCfg(
            instructions=[
                "move forward to the table and pick up the {}",
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
            robot_region=Box(low=[-1.00, 0.0, 0.0], high=[-0.90, 0.0, 0.0]),
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
