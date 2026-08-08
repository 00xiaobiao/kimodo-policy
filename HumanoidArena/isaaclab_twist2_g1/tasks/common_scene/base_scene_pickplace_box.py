import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
import os
from tasks.common_config import CameraBaseCfg


project_root = os.environ.get("PROJECT_ROOT")
ROOM_SCALE = (100.0, 100.0, 100.0)
BOX_ASSET_SCALE = (0.01, 0.01, 0.01)
# SHELF_ASSET_SCALE = (0.01, 0.01, 0.01)
SHELF_ASSET_SCALE = (1.0, 1.0, 1.0)
BOX_INIT_POS = [-3.0, -3.05, 0.82]
BOX_INIT_ROT = [1.0, 0.0, 0.0, 0.0]
SHELF_INIT_POS = [-2.7, -0.15, 0.0]
SHELF_INIT_ROT = [1.0, 0.0, 0.0, 0.0]


@configclass
class PickPlaceBoxSceneCfg(InteractiveSceneCfg):
    """Box pick-place scene configuration."""

    # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],  # room center point
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=(
                f"{project_root}/assets/objects/small_warehouse/"
                "small_warehouse_box/small_warehouse_box.usd"
            ),
            scale=ROOM_SCALE,
        ),
    )

    box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BOX_INIT_POS,
            rot=BOX_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=(
                f"{project_root}/assets/objects/small_warehouse/small_warehouse_box/"
                "intreaction_obj/CubeBox_A03_21cm_PR_NVD_01/"
                "CubeBox_A03_21cm_PR_NVD_01_physics_rigid.usd"
            ),
            scale=BOX_ASSET_SCALE,
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

    shelf = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Shelf",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=SHELF_INIT_POS,
            rot=SHELF_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=(
                f"{project_root}/assets/objects/small_warehouse/small_warehouse_box/"
                "intreaction_obj/"
                "HeavyDutySteelShelving_A06_PR_NVD_01_physics.usd"
            ),
            scale=SHELF_ASSET_SCALE,
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-4, -1, 18],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.DistantLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=5000.0,
            angle=15.0,
        ),
    )

    world_camera = CameraBaseCfg.get_world_camera_config(
        pos_offset=(0.08792, -4.59362, 2.05127),
        rot_offset=(0.63441, 0.29145, 0.29887, 0.65058),
        focal_length=12,
        horizontal_aperture=27,
        convention="opengl",
    )


# Backward-compatible alias for any older references.
DoubleTableSceneCfg = PickPlaceBoxSceneCfg
