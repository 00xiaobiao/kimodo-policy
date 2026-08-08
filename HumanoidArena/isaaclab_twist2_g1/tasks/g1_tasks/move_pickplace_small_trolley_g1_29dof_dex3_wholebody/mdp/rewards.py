from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def compute_reward_pickplace_small_trolley(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float)


__all__ = [
    "compute_reward_pickplace_small_trolley",
]
