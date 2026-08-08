import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from task_runtime_profiles import resolve_vision_navi_room_usd_path
from tasks.common_config import CameraBaseCfg


project_root = os.environ.get("PROJECT_ROOT")

ROOM_ASSET_SCALE = (100.0, 100.0, 100.0)
TARGET_ASSET_SCALE = (0.01, 0.01, 0.01)
# TARGET_ASSET_SCALE = (0.1, 0.1, 0.1)
# TARGET_ASSET_SCALE = (1, 1, 1)
# OBSTACLE_01_ASSET_SCALE = (0.1,0.02,0.025)
# OBSTACLE_02_ASSET_SCALE = (0.02,0.02,0.05)
OBSTACLE_01_ASSET_SCALE = (1.0,1.0,1.0)
OBSTACLE_02_ASSET_SCALE = (1.0,1.0,1.0)
# Runtime profile selects the room asset variant:
# - live inference/validation uses the optimized validation USD to reduce scene load.
# - replay/rerecord uses the original digital-twin USD so released recordings stay geometrically compatible.
ROOM_USD_PATH = resolve_vision_navi_room_usd_path(project_root)
TARGET_USD_PATH = (
    f"{project_root}/assets/objects/small_warehouse/small_warehouse_vision_navigation/"
    "interaction_obj/wetfloorsign.usd"
)
# TARGET_USD_PATH = (
#     f"{project_root}/assets/objects/small_warehouse/small_warehouse_vision_navigation/"
#     "interaction_obj/wetfloorsign_semantic.usd"
# )
OBSTACLE_01_USD_PATH = (
    f"{project_root}/assets/objects/small_warehouse/small_warehouse_vision_navigation/"
    "interaction_obj/obstacle_01.usd"
)
OBSTACLE_02_USD_PATH = (
    f"{project_root}/assets/objects/small_warehouse/small_warehouse_vision_navigation/"
    "interaction_obj/obstacle_02.usd"
)

# Keep init pose inside target_sign pose range so the asset stays in-bounds
# when startup randomization is disabled.
TARGET_INIT_POS = (-2.00, -0.25, 0.0)
TARGET_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

OBSTACLE_01_A_INIT_POS = (-1.25, -4.05, 0.0)
OBSTACLE_01_A_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
OBSTACLE_01_B_INIT_POS = (-2.45, -2.75, 0.0)
OBSTACLE_01_B_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
OBSTACLE_01_C_INIT_POS = (-1.65, -2.15, 0.0)
OBSTACLE_01_C_INIT_ROT = (1.0, 0.0, 0.0, 0.0)

OBSTACLE_02_A_INIT_POS = (-0.95, -3.15, 0.0)
OBSTACLE_02_A_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
OBSTACLE_02_B_INIT_POS = (-2.85, -1.85, 0.0)
OBSTACLE_02_B_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
OBSTACLE_02_C_INIT_POS = (-3.25, -3.45, 0.0)
OBSTACLE_02_C_INIT_ROT = (1.0, 0.0, 0.0, 0.0)


@configclass
class SmallWarehouseVisionNavigationSceneCfg(InteractiveSceneCfg):
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=ROOM_USD_PATH,
            scale=ROOM_ASSET_SCALE,
        ),
    )

    target_sign = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TargetSign",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=TARGET_INIT_POS,
            rot=TARGET_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=TARGET_USD_PATH,
            scale=TARGET_ASSET_SCALE,
        ),
    )

    obstacle_01_a = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle01_A",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=OBSTACLE_01_A_INIT_POS,
            rot=OBSTACLE_01_A_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=OBSTACLE_01_USD_PATH,
            scale=OBSTACLE_01_ASSET_SCALE,
            activate_contact_sensors=False,
        ),
    )

    obstacle_01_b = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle01_B",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=OBSTACLE_01_B_INIT_POS,
            rot=OBSTACLE_01_B_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=OBSTACLE_01_USD_PATH,
            scale=OBSTACLE_01_ASSET_SCALE,
            activate_contact_sensors=False,
        ),
    )

    obstacle_01_c = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle01_C",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=OBSTACLE_01_C_INIT_POS,
            rot=OBSTACLE_01_C_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=OBSTACLE_01_USD_PATH,
            scale=OBSTACLE_01_ASSET_SCALE,
            activate_contact_sensors=False,
        ),
    )

    obstacle_02_a = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle02_A",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=OBSTACLE_02_A_INIT_POS,
            rot=OBSTACLE_02_A_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=OBSTACLE_02_USD_PATH,
            scale=OBSTACLE_02_ASSET_SCALE,
            activate_contact_sensors=False,
        ),
    )

    obstacle_02_b = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle02_B",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=OBSTACLE_02_B_INIT_POS,
            rot=OBSTACLE_02_B_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=OBSTACLE_02_USD_PATH,
            scale=OBSTACLE_02_ASSET_SCALE,
            activate_contact_sensors=False,
        ),
    )

    obstacle_02_c = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle02_C",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=OBSTACLE_02_C_INIT_POS,
            rot=OBSTACLE_02_C_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=OBSTACLE_02_USD_PATH,
            scale=OBSTACLE_02_ASSET_SCALE,
            activate_contact_sensors=False,
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
        pos_offset=(0.45796, -1.01443, 2.42268),
        rot_offset=(0.50211, 0.27898, 0.39756, 0.71555),
        focal_length=12,
        horizontal_aperture=27,
        convention="opengl",
    )
