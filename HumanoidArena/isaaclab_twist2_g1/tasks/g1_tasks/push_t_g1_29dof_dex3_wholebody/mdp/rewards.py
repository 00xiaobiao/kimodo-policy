from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def compute_reward_push_t(
    env: "ManagerBasedRLEnv",
    t_top_cfg: SceneEntityCfg = SceneEntityCfg("object_t_top"),
    t_stem_cfg: SceneEntityCfg = SceneEntityCfg("object_t_stem"),
    target_top_x: float = -1.75,
    target_top_y: float = -3.05,
    target_stem_x: float = -1.75,
    target_stem_y: float = -3.20,
    top_tol_x: float = 0.18,
    top_tol_y: float = 0.08,
    stem_tol_x: float = 0.08,
    stem_tol_y: float = 0.16,
    min_height: float = 0.02,
) -> torch.Tensor:
    t_top: RigidObject = env.scene[t_top_cfg.name]
    t_stem: RigidObject = env.scene[t_stem_cfg.name]

    top_x = t_top.data.root_pos_w[:, 0]
    top_y = t_top.data.root_pos_w[:, 1]
    top_z = t_top.data.root_pos_w[:, 2]

    stem_x = t_stem.data.root_pos_w[:, 0]
    stem_y = t_stem.data.root_pos_w[:, 1]
    stem_z = t_stem.data.root_pos_w[:, 2]

    top_in_zone = (
        (top_x > target_top_x - top_tol_x)
        & (top_x < target_top_x + top_tol_x)
        & (top_y > target_top_y - top_tol_y)
        & (top_y < target_top_y + top_tol_y)
        & (top_z > min_height)
    )
    stem_in_zone = (
        (stem_x > target_stem_x - stem_tol_x)
        & (stem_x < target_stem_x + stem_tol_x)
        & (stem_y > target_stem_y - stem_tol_y)
        & (stem_y < target_stem_y + stem_tol_y)
        & (stem_z > min_height)
    )

    done = top_in_zone & stem_in_zone

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
    reward[~done] = -1.0
    reward[done] = 1.0
    return reward


__all__ = [
    "compute_reward_push_t",
]
