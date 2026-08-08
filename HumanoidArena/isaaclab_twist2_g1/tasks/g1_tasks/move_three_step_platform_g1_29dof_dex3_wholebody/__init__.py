import gymnasium as gym

from . import move_three_step_platform_g1_29dof_dex3_hw_env_cfg


gym.register(
    id="Isaac-Move-Three-Step-Platform-G129-Dex3-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": move_three_step_platform_g1_29dof_dex3_hw_env_cfg.MoveThreeStepPlatformG129Dex3WholebodyEnvCfg,
    },
    disable_env_checker=True,
)
