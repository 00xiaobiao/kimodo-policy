# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def compute_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_x: float = -0.42,                # minimum x position threshold
    max_x: float = 1.0,                # maximum x position threshold
    min_y: float = 0.2,                # minimum y position threshold
    max_y: float = 0.7,                # maximum y position threshold
    min_height: float = 0.5,
    post_min_x: float = 0.28,
    post_max_x: float = 0.96,
    post_min_y: float = 0.24,
    post_max_y: float = 0.57,
    post_min_height: float = 0.81,
    post_max_height: float = 0.9,
) -> torch.Tensor:
   # when the object is not in the set return, reset
    # Get object entity from the scene
    # 1. get object entity from the scene
    object: RigidObject = env.scene[object_cfg.name]
    
    # Extract wheel position relative to environment origin
    # 2. get object position
    wheel_x = object.data.root_pos_w[:, 0]         # x position
    wheel_y = object.data.root_pos_w[:, 1]        # y position
    wheel_height = object.data.root_pos_w[:, 2]   # z position (height)
    # print(f"wheel_height: {wheel_height}")
    # Check conditions for each environment (element-wise operations)
    done_x = (wheel_x < max_x) & (wheel_x > min_x)
    done_y = (wheel_y < max_y) & (wheel_y > min_y)
    done_height = (wheel_height > min_height)
    done = done_x & done_y & done_height

    # 3. get post position conditions
    done_post_x = (wheel_x < post_max_x) & (wheel_x > post_min_x)
    done_post_y = (wheel_y < post_max_y) & (wheel_y > post_min_y)
    done_post_height = (wheel_height > post_min_height) & (wheel_height < post_max_height)
    done_post = done_post_x & done_post_y & done_post_height

    # Create reward tensor for all environments
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
    
    # Set rewards based on conditions
    reward[~done] = -1.0  # Not in valid area
    reward[done_post] = 1.0  # In target post area
    reward[done & ~done_post] = 0.0  # In valid area but not target
    
    return reward
