import gymnasium as gym

from . import push_t_g1_29dof_dex3_hw_env_cfg


gym.register(
    id="Isaac-Push-T-G129-Dex3-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": push_t_g1_29dof_dex3_hw_env_cfg.PushTG129Dex3WholebodyEnvCfg,
    },
    disable_env_checker=True,
)
