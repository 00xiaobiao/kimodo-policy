import gymnasium as gym

from . import move_small_warehouse_vision_navigation_g1_29dof_dex3_hw_env_cfg


gym.register(
    id="Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            move_small_warehouse_vision_navigation_g1_29dof_dex3_hw_env_cfg
            .MoveSmallWarehouseVisionNavigationG129Dex3WholebodyEnvCfg
        ),
    },
    disable_env_checker=True,
)
