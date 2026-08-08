# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Football goal reward — all geometry for this task lives in this file only.

Scoring rule: ball *center* must (1) cross the **inner edge** of the painted goal reference line
(same width as ``tools.pitch_lines``: ``0.12 * 0.5`` m), (2) stay within the goal mouth in X,
(3) stay within goal height in Z.

Line center in ``(x, y)`` is taken as ``GOAL_NET_*_ORIGIN`` (goal spawn origin), matching a
``goal_reference`` line drawn with relative offset ``(0, 0)`` at that center.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from tasks.common_scene.base_scene_football_single_cfg_wholebody import GOAL_NET_1_ORIGIN, GOAL_NET_2_ORIGIN

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# --- constants (kept here only; must match ``create_goal_reference_lines`` width formula) ---
# ``tools.pitch_lines.DEFAULT_LINE_WIDTH`` * env ``GOAL_REFERENCE_LINE_WIDTH_RATIO`` (0.5)
_PITCH_DEFAULT_LINE_WIDTH_M = 0.12
_GOAL_REFERENCE_LINE_WIDTH_RATIO = 0.5
_GOAL_LINE_WIDTH_M = _PITCH_DEFAULT_LINE_WIDTH_M * _GOAL_REFERENCE_LINE_WIDTH_RATIO

# Mouth inner span along world X: offsets from ``GOAL_NET_*_ORIGIN[0]`` (meters), no padding.
_GOAL_MOUTH_X_MIN_LOCAL = 0.0
_GOAL_MOUTH_X_MAX_LOCAL = 5.1

# Height under crossbar (world Z), meters.
_GOAL_HEIGHT_Z_MIN = 0.0
_GOAL_HEIGHT_Z_MAX = 1.85


def compute_reward(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    front_goal_origin_x: float = GOAL_NET_1_ORIGIN[0],
    front_goal_origin_y: float = GOAL_NET_1_ORIGIN[1],
    back_goal_origin_x: float = GOAL_NET_2_ORIGIN[0],
    back_goal_origin_y: float = GOAL_NET_2_ORIGIN[1],
    goal_mouth_x_min_local: float = _GOAL_MOUTH_X_MIN_LOCAL,
    goal_mouth_x_max_local: float = _GOAL_MOUTH_X_MAX_LOCAL,
    goal_min_height: float = _GOAL_HEIGHT_Z_MIN,
    goal_max_height: float = _GOAL_HEIGHT_Z_MAX,
    goal_line_width: float = _GOAL_LINE_WIDTH_M,
) -> torch.Tensor:
    """Binary reward: +1 if scored, else -1."""

    football: RigidObject = env.scene[object_cfg.name]

    ball_x = football.data.root_pos_w[:, 0]
    ball_y = football.data.root_pos_w[:, 1]
    ball_z = football.data.root_pos_w[:, 2]

    front_x_min = front_goal_origin_x + goal_mouth_x_min_local
    front_x_max = front_goal_origin_x + goal_mouth_x_max_local
    front_goal_line_center_y = front_goal_origin_y
    front_goal_line_inner_y = front_goal_line_center_y + goal_line_width * 0.5

    in_goal_z = (ball_z > goal_min_height) & (ball_z < goal_max_height)

    in_front_goal_x = (ball_x > front_x_min) & (ball_x < front_x_max)
    in_front_goal_y = ball_y > front_goal_line_inner_y
    scored_front = in_front_goal_x & in_front_goal_y & in_goal_z

    has_back_goal = "goal_net_2" in env.scene.keys()
    if has_back_goal:
        back_x_min = back_goal_origin_x - goal_mouth_x_max_local
        back_x_max = back_goal_origin_x - goal_mouth_x_min_local
        back_goal_line_center_y = back_goal_origin_y
        back_goal_line_inner_y = back_goal_line_center_y - goal_line_width * 0.5
        in_back_goal_x = (ball_x > back_x_min) & (ball_x < back_x_max)
        in_back_goal_y = ball_y < back_goal_line_inner_y
        scored_back = in_back_goal_x & in_back_goal_y & in_goal_z
    else:
        scored_back = torch.zeros_like(scored_front)

    scored = scored_front | scored_back

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
    reward[~scored] = -1.0
    reward[scored] = 1.0
    return reward


__all__ = ["compute_reward"]
