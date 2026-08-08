import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from tasks.common_config import CameraBaseCfg

project_root = os.environ.get("PROJECT_ROOT")

STAIR_ANCHOR_POS = [0.0, -1.0, 0.0]
STAIR_STEP_GAP_Y = 0.05
STAIR_STEP_DEPTH_Y = 0.40
STAIR_STEP_HEIGHT_Z = 0.12
STAIR_TOTAL_WIDTH_X = 2.00

PLATFORM_GAP_Y = 0.10
PLATFORM_DEPTH_Y = 1.60
PLATFORM_THICKNESS_Z = 0.12

STEP1_HEIGHT = STAIR_STEP_HEIGHT_Z
STEP2_HEIGHT = STAIR_STEP_HEIGHT_Z * 2.0
STEP3_HEIGHT = STAIR_STEP_HEIGHT_Z * 3.0

STEP1_CENTER_Y = STAIR_ANCHOR_POS[1]
STEP2_CENTER_Y = STEP1_CENTER_Y + STAIR_STEP_DEPTH_Y + STAIR_STEP_GAP_Y
STEP3_CENTER_Y = STEP2_CENTER_Y + STAIR_STEP_DEPTH_Y + STAIR_STEP_GAP_Y

STEP1_CENTER_Z = STEP1_HEIGHT * 0.5
STEP2_CENTER_Z = STEP2_HEIGHT * 0.5
STEP3_CENTER_Z = STEP3_HEIGHT * 0.5

PLATFORM_CENTER_Y = STEP3_CENTER_Y + STAIR_STEP_DEPTH_Y * 0.5 + PLATFORM_GAP_Y + PLATFORM_DEPTH_Y * 0.5
PLATFORM_CENTER_Z = STEP3_HEIGHT - PLATFORM_THICKNESS_Z * 0.5


@configclass
class ThreeStepPlatformSceneCfg(InteractiveSceneCfg):
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/small_warehouse_digital_twin.usd",
        ),
    )

    stair_step_1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/StairStep1",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[STAIR_ANCHOR_POS[0], STEP1_CENTER_Y, STEP1_CENTER_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(STAIR_TOTAL_WIDTH_X, STAIR_STEP_DEPTH_Y, STEP1_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.45, 0.45),
                metallic=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=10.0,
                dynamic_friction=1.5,
                restitution=0.01,
            ),
        ),
    )

    stair_step_2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/StairStep2",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[STAIR_ANCHOR_POS[0], STEP2_CENTER_Y, STEP2_CENTER_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(STAIR_TOTAL_WIDTH_X, STAIR_STEP_DEPTH_Y, STEP2_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.48, 0.48, 0.48),
                metallic=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=10.0,
                dynamic_friction=1.5,
                restitution=0.01,
            ),
        ),
    )

    stair_step_3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/StairStep3",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[STAIR_ANCHOR_POS[0], STEP3_CENTER_Y, STEP3_CENTER_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(STAIR_TOTAL_WIDTH_X, STAIR_STEP_DEPTH_Y, STEP3_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.52, 0.52, 0.52),
                metallic=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=10.0,
                dynamic_friction=1.5,
                restitution=0.01,
            ),
        ),
    )

    platform = AssetBaseCfg(
        prim_path="/World/envs/env_.*/StairPlatform",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[STAIR_ANCHOR_POS[0], PLATFORM_CENTER_Y, PLATFORM_CENTER_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(STAIR_TOTAL_WIDTH_X, PLATFORM_DEPTH_Y, PLATFORM_THICKNESS_Z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.42, 0.42, 0.42),
                metallic=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=10.0,
                dynamic_friction=1.5,
                restitution=0.01,
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
    world_camera = CameraBaseCfg.get_world_camera_config(
        pos_offset=(-3.41952, 3.24861, 3.08657),
        rot_offset=(-0.18435, -0.08619, 0.41467, 0.88692),
        focal_length=12,
        horizontal_aperture=27,
        convention="opengl"
    )
