# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Boxing bag scene configuration for G1 wholebody tasks.
Uses OBJ-converted USD with physics for punching bag.
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

project_root = os.environ.get("PROJECT_ROOT")

# Layout: 統一在此調整機器人、拳擊沙袋的初始位置
# 機器人已向左旋轉 90° (init_rot=(0.7071,0,0,0.7071))，朝 +Y 方向
ROBOT_INIT_X = -1.9
ROBOT_INIT_Y = -5.2
ROBOT_INIT_Z = 0.8  # 機器人站立高度
BAG_DISTANCE = 1.2  # 沙袋在機器人前方距離（揮拳舒適距離）

# 90° 左旋後：原 (dx,dy) → (-dy, dx)
# Bag: 原 (BAG_DISTANCE,0) → (0, BAG_DISTANCE)
BAG_OFFSET_X = 0.0
BAG_OFFSET_Y = BAG_DISTANCE

# 拳擊沙袋尺度與朝向
# 朝向已 baking 在 USD 中（convert_boxing_bag_assets.py 用 MeshConverter rotation 旋轉）
# 若模型平躺：長軸為 X 時用 rotation=(0.7071,0,0.7071,0) 繞 Y 軸 90°
BAG_SCALE = (0.15, 0.15, 0.15)
BAG_ROT_UPRIGHT = (0.7071, 0.7071, 0, 0)  # 已在 USD 中 upright，此處 identity 即可
BAG_HEIGHT_APPROX = 1.2  # 沙袋約 1.2m 高，底部貼地時中心 z = half_height


@configclass
class TableBoxingBagSceneCfgWH(InteractiveSceneCfg):
    """Boxing bag scene configuration.
    Self-contained boxing bag scene without the removed cylinder base scene.
    """

    # Override room
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/small_warehouse_digital_twin_boxtarget.usd",
        ),
    )

    # Hide box (no table needed)
    box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-50.0, -50.0, -10.0),
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
    )

    # Boxing bag - dynamic rigid body，受擊時會晃動，更貼近真實遙操作數據
    # 尺度與朝向：參照 football 球門，OBJ 多為 cm，scale 0.01；旋轉使直立
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[ROBOT_INIT_X + BAG_OFFSET_X, ROBOT_INIT_Y + BAG_OFFSET_Y, BAG_HEIGHT_APPROX / 2.0],
            rot=BAG_ROT_UPRIGHT,  # 繞 Y 軸 90° 使平躺模型直立
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/boxing_bag/boxing_bag_physics.usd",
            scale=BAG_SCALE,  # OBJ in meters (Blender 預設)，參照 football ball
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )
