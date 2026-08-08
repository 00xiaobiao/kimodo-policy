# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Football (soccer ball) scene configuration for G1 wholebody tasks.
Uses textured soccer ball USD and goal net USD. 完全移除 warehouse / room / box，
僅保留 ground + 球 + 球門 + 燈光 + 相機。
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from tasks.common_config import CameraBaseCfg

project_root = os.environ.get("PROJECT_ROOT")

# Layout: 統一在此調整機器人、球、球門的初始位置
# 機器人已向左旋轉 90° (init_rot=(0.7071,0,0,0.7071))，朝 +Y 方向
# 球與球門的相對位置已同步旋轉，保持與機器人的相對位姿
ROBOT_INIT_X = 0.0
ROBOT_INIT_Y = 0.0
ROBOT_INIT_Z = 0.8  # 機器人站立高度
GOAL_DISTANCE = 3.0  # 球門與機器人距離（沿機器人朝向）
BALL_DISTANCE = 1.0  # 足球在機器人前方距離
GOAL_Z = 0.65  # 球門高度（Z 座標）
GOAL_CENTER_Y_OFFSET = 2.5  # 球門中心相對於機器人朝向的橫向偏移
GOAL_COLLISION_ENABLED = True
GOAL_CONTACT_OFFSET = 0.001
GOAL_REST_OFFSET = 0.0

# 90° 左旋後：原 (dx,dy) → (-dy, dx)，機器人朝 +Y
# Ball: 原 (1,0) → (0, 1)
# Goal: 原 (6, 2.5) → (-2.5, 6)
BALL_OFFSET_X = 0.0  # -dy
BALL_OFFSET_Y = BALL_DISTANCE  # dx
GOAL_OFFSET_X = -GOAL_CENTER_Y_OFFSET  # -dy
GOAL_OFFSET_Y = GOAL_DISTANCE  # dx
GOAL_NET_1_ORIGIN = (ROBOT_INIT_X + GOAL_OFFSET_X, ROBOT_INIT_Y + GOAL_OFFSET_Y)
GOAL_NET_2_ORIGIN = (ROBOT_INIT_X - GOAL_OFFSET_X, ROBOT_INIT_Y - GOAL_OFFSET_Y)
GOAL_NET_ORIGIN_TO_CENTER_LOCAL = (0.0, 0.0)
GOAL_NET_1_CENTER = (
    GOAL_NET_1_ORIGIN[0] + GOAL_NET_ORIGIN_TO_CENTER_LOCAL[0],
    GOAL_NET_1_ORIGIN[1] + GOAL_NET_ORIGIN_TO_CENTER_LOCAL[1],
)
GOAL_NET_2_CENTER = (
    GOAL_NET_2_ORIGIN[0] - GOAL_NET_ORIGIN_TO_CENTER_LOCAL[0],
    GOAL_NET_2_ORIGIN[1] - GOAL_NET_ORIGIN_TO_CENTER_LOCAL[1],
)
GOAL_BACKDROP_THICKNESS = 0.08
GOAL_BACKDROP_HEIGHT = 2.6
GOAL_BACKDROP_Y_GAP = 3.0
GOAL_BACKDROP_SIDE_MARGIN_X = 4.3
GOAL_BACKDROP_SIDE_X = abs(GOAL_OFFSET_X) + GOAL_BACKDROP_SIDE_MARGIN_X
GOAL_BACKDROP_HALF_Y = GOAL_OFFSET_Y + GOAL_BACKDROP_Y_GAP
GOAL_BACKDROP_WIDTH = GOAL_BACKDROP_SIDE_X * 2.0 + GOAL_BACKDROP_THICKNESS
GOAL_BACKDROP_SIDE_LENGTH = GOAL_BACKDROP_HALF_Y * 2.0 + GOAL_BACKDROP_THICKNESS
GOAL_BACKDROP_COLOR = (0.35, 0.35, 0.35)

# FIFA standard football specifications:
# - Circumference: 68-70 cm -> diameter ~22 cm, radius = 0.11 m
# - Mass: 410-450 g -> 0.43 kg
# - Restitution (bounciness): FIFA 合格足球約 0.7–0.8，設 0.75 接近真實觸感
# ImageToStl STL/USD typically uses mm units -> scale 0.001 for 220mm ball


@configclass
class TableFootballSceneCfgWH(InteractiveSceneCfg):
    """Football scene configuration.
    足球場景：完全移除 warehouse / room_walls / box，僅保留 ground + 球 + 球門 + 燈光 + 相機。
    """

    # Football - 在機器人前方 BALL_DISTANCE 處（隨機器人 90° 左旋同步旋轉）
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[ROBOT_INIT_X + BALL_OFFSET_X, ROBOT_INIT_Y + BALL_OFFSET_Y, 0.11],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/football_scene/interaction_obj/soccer_ball/soccer_ball_physics.usd",
            scale=(1.0, 1.0, 1.0),  # OBJ in meters -> FIFA ball ~0.22m
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            # UsdFileCfg 不支援 physics_material，需於 reset 後由 apply_football_physics_material() 動態套用
        ),
    )

    # Ground plane - 10×10m 草地渲染區域（非 UV），UV 維持 150×150
    # 使用 Cuboid 取代 GroundPlane 以限制渲染範圍為 10×10m，視覺 PBR 由 apply_grass_pbr_to_ground() 動態套用
    ground = RigidObjectCfg(
        prim_path="/World/GroundPlane",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, -0.005],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(14.0, 14.0, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.5,
                restitution=0.1,
            ),
        ),
    )

    # Goal net 1 - 機器人正前方，球門開口朝向機器人
    goal_net = AssetBaseCfg(
        prim_path="/World/envs/env_.*/GoalNet",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[GOAL_NET_1_ORIGIN[0], GOAL_NET_1_ORIGIN[1], GOAL_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/football_scene/interaction_obj/football_goal/football_goal_physics.usd",
            scale=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=GOAL_COLLISION_ENABLED,
                contact_offset=GOAL_CONTACT_OFFSET,
                rest_offset=GOAL_REST_OFFSET,
            ),
        ),
    )

    # Goal net 2 - 對稱於另一側，球門開口朝向場中央
    goal_net_2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/GoalNet2",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[GOAL_NET_2_ORIGIN[0], GOAL_NET_2_ORIGIN[1], GOAL_Z],
            rot=[0.0, 0.0, 0.0, 1.0],  # 180° 繞 Z，開口朝中心
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/football_scene/interaction_obj/football_goal/football_goal_physics.usd",
            scale=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=GOAL_COLLISION_ENABLED,
                contact_offset=GOAL_CONTACT_OFFSET,
                rest_offset=GOAL_REST_OFFSET,
            ),
        ),
    )

    goal_backdrop_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GoalBackdrop1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[
                ROBOT_INIT_X,
                ROBOT_INIT_Y + GOAL_BACKDROP_HALF_Y,
                GOAL_BACKDROP_HEIGHT * 0.5,
            ],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(GOAL_BACKDROP_WIDTH, GOAL_BACKDROP_THICKNESS, GOAL_BACKDROP_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GOAL_BACKDROP_COLOR),
        ),
    )

    goal_backdrop_2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GoalBackdrop2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[
                ROBOT_INIT_X,
                ROBOT_INIT_Y - GOAL_BACKDROP_HALF_Y,
                GOAL_BACKDROP_HEIGHT * 0.5,
            ],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(GOAL_BACKDROP_WIDTH, GOAL_BACKDROP_THICKNESS, GOAL_BACKDROP_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GOAL_BACKDROP_COLOR),
        ),
    )

    goal_backdrop_3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GoalBackdrop3",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[
                ROBOT_INIT_X - GOAL_BACKDROP_SIDE_X,
                ROBOT_INIT_Y,
                GOAL_BACKDROP_HEIGHT * 0.5,
            ],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(GOAL_BACKDROP_THICKNESS, GOAL_BACKDROP_SIDE_LENGTH, GOAL_BACKDROP_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GOAL_BACKDROP_COLOR),
        ),
    )

    goal_backdrop_4 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GoalBackdrop4",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[
                ROBOT_INIT_X + GOAL_BACKDROP_SIDE_X,
                ROBOT_INIT_Y,
                GOAL_BACKDROP_HEIGHT * 0.5,
            ],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(GOAL_BACKDROP_THICKNESS, GOAL_BACKDROP_SIDE_LENGTH, GOAL_BACKDROP_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GOAL_BACKDROP_COLOR),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )

    world_camera = CameraBaseCfg.get_world_camera_config(
        pos_offset=(-2.01826, 3.33365, 1.26749),
        rot_offset=(0.85990, 0.24942, -0.43352, -0.10206),
        focal_length=12,
        horizontal_aperture=27,
        convention="opengl",
    )
