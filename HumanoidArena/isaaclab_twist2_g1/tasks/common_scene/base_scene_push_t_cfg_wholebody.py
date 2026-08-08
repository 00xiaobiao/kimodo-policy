import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg

project_root = os.environ.get("PROJECT_ROOT")

ROBOT_INIT_X = -2.25
ROBOT_INIT_Y = -2.65
ROBOT_INIT_Z = 0.8

T_TOP_INIT_POS = (-2.75, -3.05, 0.06)
T_STEM_INIT_POS = (-2.75, -3.20, 0.06)

T_ZONE_TOP_POS = (-1.75, -3.05, 0.005)
T_ZONE_STEM_POS = (-1.75, -3.20, 0.005)


@configclass
class PushTSceneCfgWH(InteractiveSceneCfg):
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/small_warehouse_digital_twin.usd",
        ),
    )

    object_t_top = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object_t_top",
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(T_TOP_INIT_POS), rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.48, 0.14, 0.12),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, retain_accelerations=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.2, 0.2), metallic=0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=10.0,
                dynamic_friction=1.5,
                restitution=0.01,
            ),
        ),
    )

    object_t_stem = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object_t_stem",
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(T_STEM_INIT_POS), rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.14, 0.30, 0.12),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, retain_accelerations=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.2, 0.2), metallic=0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=10.0,
                dynamic_friction=1.5,
                restitution=0.01,
            ),
        ),
    )

    t_zone_top_h = AssetBaseCfg(
        prim_path="/World/envs/env_.*/t_zone_top_h",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[T_ZONE_TOP_POS[0], T_ZONE_TOP_POS[1] + 0.09, T_ZONE_TOP_POS[2]], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.52, 0.02, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.95, 0.1), metallic=0),
        ),
    )
    t_zone_bottom_h = AssetBaseCfg(
        prim_path="/World/envs/env_.*/t_zone_bottom_h",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[T_ZONE_TOP_POS[0], T_ZONE_TOP_POS[1] - 0.09, T_ZONE_TOP_POS[2]], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.52, 0.02, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.95, 0.1), metallic=0),
        ),
    )
    t_zone_left_v = AssetBaseCfg(
        prim_path="/World/envs/env_.*/t_zone_left_v",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[T_ZONE_TOP_POS[0] - 0.25, T_ZONE_TOP_POS[1], T_ZONE_TOP_POS[2]], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.20, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.95, 0.1), metallic=0),
        ),
    )
    t_zone_right_v = AssetBaseCfg(
        prim_path="/World/envs/env_.*/t_zone_right_v",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[T_ZONE_TOP_POS[0] + 0.25, T_ZONE_TOP_POS[1], T_ZONE_TOP_POS[2]], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.20, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.95, 0.1), metallic=0),
        ),
    )

    t_zone_stem_top_h = AssetBaseCfg(
        prim_path="/World/envs/env_.*/t_zone_stem_top_h",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[T_ZONE_STEM_POS[0], T_ZONE_STEM_POS[1] + 0.16, T_ZONE_STEM_POS[2]], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.18, 0.02, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2), metallic=0),
        ),
    )
    t_zone_stem_bottom_h = AssetBaseCfg(
        prim_path="/World/envs/env_.*/t_zone_stem_bottom_h",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[T_ZONE_STEM_POS[0], T_ZONE_STEM_POS[1] - 0.16, T_ZONE_STEM_POS[2]], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.18, 0.02, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2), metallic=0),
        ),
    )
    t_zone_stem_left_v = AssetBaseCfg(
        prim_path="/World/envs/env_.*/t_zone_stem_left_v",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[T_ZONE_STEM_POS[0] - 0.08, T_ZONE_STEM_POS[1], T_ZONE_STEM_POS[2]], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.34, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2), metallic=0),
        ),
    )
    t_zone_stem_right_v = AssetBaseCfg(
        prim_path="/World/envs/env_.*/t_zone_stem_right_v",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[T_ZONE_STEM_POS[0] + 0.08, T_ZONE_STEM_POS[1], T_ZONE_STEM_POS[2]], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.34, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2), metallic=0),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    world_camera = CameraBaseCfg.get_world_camera_config(
        pos_offset=(-1.9, -5.0, 1.8),
        rot_offset=(-0.40614, 0.78544, 0.4277, -0.16986),
    )
