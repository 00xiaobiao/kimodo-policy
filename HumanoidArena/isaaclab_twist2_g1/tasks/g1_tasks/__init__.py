# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Unitree G1 task registrations used by HumanoidArena."""

# Import maintained task packages so their gym registrations are available.
from . import move_football_g1_29dof_dex3_wholebody
from . import move_football_single_g1_29dof_dex3_wholebody
from . import move_boxing_bag_g1_29dof_dex3_wholebody
from . import move_pickplace_doubledesk_g1_29dof_dex3_wholebody
from . import move_pickplace_box_g1_29dof_dex3_wholedoby
from . import move_open_door_g1_29dof_dex3_wholebody
from . import move_sit_sofa_g1_29dof_dex3_wholebody
from . import move_small_warehouse_vision_navigation_g1_29dof_dex3_wholebody

# Additional existing task packages kept importable for compatibility.
from . import push_t_g1_29dof_dex3_wholebody
from . import move_three_step_platform_g1_29dof_dex3_wholebody
from . import move_pickplace_small_trolley_g1_29dof_dex3_wholebody

__all__ = [
    "move_football_g1_29dof_dex3_wholebody",
    "move_football_single_g1_29dof_dex3_wholebody",
    "move_boxing_bag_g1_29dof_dex3_wholebody",
    "move_pickplace_doubledesk_g1_29dof_dex3_wholebody",
    "move_pickplace_box_g1_29dof_dex3_wholedoby",
    "move_open_door_g1_29dof_dex3_wholebody",
    "move_sit_sofa_g1_29dof_dex3_wholebody",
    "move_small_warehouse_vision_navigation_g1_29dof_dex3_wholebody",
    "push_t_g1_29dof_dex3_wholebody",
    "move_three_step_platform_g1_29dof_dex3_wholebody",
    "move_pickplace_small_trolley_g1_29dof_dex3_wholebody",
]
