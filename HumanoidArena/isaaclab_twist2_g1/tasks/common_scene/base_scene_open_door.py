import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from task_runtime_profiles import resolve_open_door_door_usd_path
from tasks.common_config import CameraBaseCfg

project_root = os.environ.get("PROJECT_ROOT")

DOOR_POS = [-1.614, 2.314, 0.002]
DOOR_ROT = [1.0, 0.0, 0.0, 0.0]
USE_DOOR_ARTICULATION_SCENE = os.environ.get("OPEN_DOOR_SCENE_AS_ARTICULATION", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

DOOR_SPAWN_CFG = UsdFileCfg(
    usd_path=resolve_open_door_door_usd_path(project_root),
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        kinematic_enabled=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=5.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        fix_root_link=True,
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.01,
        rest_offset=0.0,
    ),
)


@configclass
class OpenDoorSceneCfg(InteractiveSceneCfg):
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/small_warehouse_opendoor/small_warehouse_digital_twin_opendoor.usd",
        ),
    )

    door = ArticulationCfg(
        prim_path="/World/envs/env_.*/Door",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=DOOR_POS,
            rot=DOOR_ROT,
        ),
        spawn=DOOR_SPAWN_CFG,
        actuators={
            "door_leaf": ImplicitActuatorCfg(
                joint_names_expr=["RevoluteJoint_door001"],
                stiffness=0.0,
                damping=0.0,
                effort_limit_sim=1000000.0,
                velocity_limit_sim=None,
            ),
            "door_handle": ImplicitActuatorCfg(
                joint_names_expr=["RevoluteJoint"],
                stiffness=5.0,
                damping=0.1,
                effort_limit_sim=0.001,
                velocity_limit_sim=None,
            ),
        },
    ) if USE_DOOR_ARTICULATION_SCENE else AssetBaseCfg(
        prim_path="/World/envs/env_.*/Door",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=DOOR_POS,
            rot=DOOR_ROT,
        ),
        spawn=DOOR_SPAWN_CFG,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )

    world_camera = CameraBaseCfg.get_world_camera_config(
        pos_offset=(0.26556, 1.33827, 3.77471),
        rot_offset=(0.6811, 0.21343, 0.20944, 0.66834),
        focal_length=12,
        horizontal_aperture=27,
        convention="opengl",
    )
