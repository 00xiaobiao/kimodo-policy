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

# Layout: Configure the initial robot and punching-bag positions here
# The robot is rotated 90 degrees left (init_rot=(0.7071,0,0,0.7071)) and faces +Y
ROBOT_INIT_X = -1.9
ROBOT_INIT_Y = -5.2
ROBOT_INIT_Z = 0.8  # Robot standing height
BAG_DISTANCE = 1.2  # Distance from the robot to the bag for comfortable punches

# 90° After a 90-degree left rotation: original (dx,dy) → (-dy, dx)
# Bag: original (BAG_DISTANCE, 0) -> (0, BAG_DISTANCE)
BAG_OFFSET_X = 0.0
BAG_OFFSET_Y = BAG_DISTANCE

# Punching-bag scale and orientation
# orientation is baked into the USD by MeshConverter in convert_boxing_bag_assets.py
# If the model lies flat with its long axis along X, use rotation=(0.7071,0,0.7071,0) rotate 90 degrees about Y
BAG_SCALE = (0.15, 0.15, 0.15)
BAG_ROT_UPRIGHT = (0.7071, 0.7071, 0, 0)  # The model is upright in the USD, so identity is sufficient here
BAG_HEIGHT_APPROX = 1.2  # The bag is about 1.2 m tall; when its base touches the ground, its center is at half_height


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

    # Boxing bag - dynamic rigid body. It swings when struck, better matching real teleoperation data.
    # Scale and orientation follow the football goal; OBJ units are usually cm, so scale by 0.01 and rotate upright
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[ROBOT_INIT_X + BAG_OFFSET_X, ROBOT_INIT_Y + BAG_OFFSET_Y, BAG_HEIGHT_APPROX / 2.0],
            rot=BAG_ROT_UPRIGHT,  # Rotate 90 degrees about Y to stand the flat model upright
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/boxing_bag/boxing_bag_physics.usd",
            scale=BAG_SCALE,  # OBJ in meters (Blender default), following the football
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
