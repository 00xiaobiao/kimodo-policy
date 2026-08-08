from __future__ import annotations

from typing import TYPE_CHECKING

from .rewards import compute_success_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def success_reached(env: "ManagerBasedRLEnv"):
    """Terminate the episode immediately when the binary success condition is met."""

    return compute_success_mask(env)


__all__ = ["success_reached"]
