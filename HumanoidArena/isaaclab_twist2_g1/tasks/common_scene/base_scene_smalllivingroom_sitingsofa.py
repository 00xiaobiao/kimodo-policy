import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg

project_root = os.environ.get("PROJECT_ROOT")

SIT_SOFA_USD_PATH = (
    f"{project_root}/assets/objects/small_warehouse/"
    "small_warehouse_sitsofa/small_warehouse_digital_twin_sitsofa.usd"
)
CARTON_USD_PATH = (
    f"{project_root}/assets/objects/small_warehouse/"
    "small_warehouse_sitsofa/intreaction_obj/CubeBox_A03_21cm_PR_NVD_01/CubeBox_A03_21cm_PR_NVD_01_physics.usd"
)
CARTON_ASSET_SCALE = (0.02, 0.02, 0.02)
CARTON_OBSTACLE_01_INIT_POS = (-1.45, -0.68, 0.10)
CARTON_OBSTACLE_01_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
CARTON_OBSTACLE_02_INIT_POS = (-1.35, 0.06, 0.10)
CARTON_OBSTACLE_02_INIT_ROT = (1.0, 0.0, 0.0, 0.0)


@configclass
class SmallLivingroomSitingSofaSceneCfg(InteractiveSceneCfg):
    """Small livingroom scene with sofa collision authored in the USD."""

    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=SIT_SOFA_USD_PATH,
        ),
    )

    carton_obstacle_01 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CartonObstacle01",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CARTON_OBSTACLE_01_INIT_POS,
            rot=CARTON_OBSTACLE_01_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=CARTON_USD_PATH,
            scale=CARTON_ASSET_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    carton_obstacle_02 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CartonObstacle02",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CARTON_OBSTACLE_02_INIT_POS,
            rot=CARTON_OBSTACLE_02_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=CARTON_USD_PATH,
            scale=CARTON_ASSET_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
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
        pos_offset=(-5.00612, 1.22158, 2.27319),
        rot_offset=(-0.43051, -0.25339, 0.43941, 0.74657),
        focal_length=12,
        horizontal_aperture=27,
        convention="opengl",
    )
