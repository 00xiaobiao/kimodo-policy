import gymnasium as gym

from . import move_pickplace_doubledesk_g1_29dof_dex3_hw_env_cfg


gym.register(
    id="Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": move_pickplace_doubledesk_g1_29dof_dex3_hw_env_cfg.MovePickPlaceDoubleDeskG129Dex3WholebodyEnvCfg,
    },
    disable_env_checker=True,
)
