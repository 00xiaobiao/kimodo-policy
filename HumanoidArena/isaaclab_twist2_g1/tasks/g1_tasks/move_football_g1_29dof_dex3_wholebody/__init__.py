# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import move_football_g1_29dof_dex3_hw_env_cfg


gym.register(
    id="Isaac-Move-Football-G129-Dex3-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": move_football_g1_29dof_dex3_hw_env_cfg.MoveFootballG129Dex3WholebodyEnvCfg,
    },
    disable_env_checker=True,
)
