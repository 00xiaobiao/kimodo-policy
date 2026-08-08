import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
import os
from tasks.common_config import CameraBaseCfg
project_root = os.environ.get("PROJECT_ROOT")
PACKING_TABLE_L_POS = [-0.1, -3.2, -0.2]
PACKING_TABLE_R_POS = [-4.0, -3.2, -0.2]
OBJECT_L_POS_OFFSET = [-0.3, 0.0, 1.04]
CONTAINER_R_POS_OFFSET = [0.3, 0.0, 1.00]
ROOM_SCALE = (100.0, 100.0, 100.0)
HAMMER_INIT_POS = [-3.137042543141045, -3.0097883381183254, 1.0106319452873727]
HAMMER_INIT_ROT = [0.7139050960540771, 0.0, 0.0, -0.7002424597740173]
# HAMMER_INIT_POS = [0.0, 0.0, 0.0]
# HAMMER_INIT_ROT = [1.0, 0.0, 0.0, 0.0]
CRATE_INIT_POS = [-2.8, -0.1, 0.745]
CRATE_INIT_ROT = [1.0, 0.0, 0.0, 0.0]
BASKET_RIGID_SUBPRIM = "PRootNode"


@configclass
class DoubleTableSceneCfg(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
    # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],  # room center point
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/small_warehouse_doubledesk/small_doubledesk.usd",
            # usd_path=f"{project_root}/assets/objects/small_warehouse/small_cart.usd",
            # small_doubledesk.usd is authored 100x smaller than the warehouse scenes
            # already used elsewhere in this repo (it lacks the top-level /Lab scale=100).
            scale=ROOM_SCALE,
        ),
    )

    object_l = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ObjectL",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=HAMMER_INIT_POS,
            rot=HAMMER_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/small_warehouse_doubledesk/interaction_obj/hammer/hammer.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.35),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    basket = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Basket",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=CRATE_INIT_POS,
            rot=CRATE_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/SM_Crate_A14_Gray_01/SM_Crate_A14_Gray_01_physics.usd",
            scale=(0.01, 0.01, 0.01),
            activate_contact_sensors=False,
        ),
    )

    # Lights
    # 4. light configuration
    # light = AssetBaseCfg(
    #     prim_path="/World/light",   # light in the scene
    #     spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75),
    #                                  intensity=5000.0),
    # )

    light = AssetBaseCfg(
      prim_path="/World/light",
      init_state=AssetBaseCfg.InitialStateCfg(
          pos=[-4, -1, 18],
          rot=[1.0, 0.0, 0.0, 0.0],  # 示例四元数，表示一个倾斜方向
      ),
      spawn=sim_utils.DistantLightCfg(
          color=(0.75, 0.75, 0.75),
          intensity=5000.0,
          angle=15.0,
      ),
    )

    world_camera = CameraBaseCfg.get_world_camera_config(
        pos_offset=(-0.55955, -2.63763, 2.16241),
        rot_offset=(0.70287, 0.42371, 0.29498, 0.48932),
        focal_length=12,
        horizontal_aperture=27,
        convention="opengl",
    )
