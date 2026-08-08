# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
dex3 state
"""     
from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import sys
import os
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


import torch

def get_robot_girl_joint_names() -> list[str]:
    return [
        # hand joints (14)
        # left hand (7)
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        # right hand (7)
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
    ]

# global variable to cache the DDS instance
_dex3_dds = None
_dds_initialized = False

def _get_dex3_dds_instance():
    """get the DDS instance, delay initialization"""
    global _dex3_dds, _dds_initialized
    
    if not _dds_initialized or _dex3_dds is None:
        try:
            # dynamically import the DDS module
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.dds_master import dds_manager
            
            _dex3_dds = dds_manager.get_object("dex3")
            # print("[Observations Dex3] DDS communication instance obtained")
            
            # register the cleanup function
            import atexit
            def cleanup_dds():
                try:
                    if _dex3_dds:
                        dds_manager.unregister_object("dex3")
                        print("[dex3_state] DDS communication closed correctly")
                except Exception as e:
                    print(f"[dex3_state] Error closing DDS: {e}")
            atexit.register(cleanup_dds)
            
        except Exception as e:
            print(f"[Observations Dex3] Failed to get DDS instances: {e}")
            _dex3_dds = None
        
        _dds_initialized = True
    
    return _dex3_dds

def get_robot_dex3_joint_states(
    env: ManagerBasedRLEnv,
    enable_dds: bool = True,
) -> torch.Tensor:
    """get the robot gripper joint states and publish them to DDS
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
        enable_dds: bool - whether to enable the DDS publish function
    
    Returns:
        torch.Tensor
    """
    # get the gripper joint positions, velocities, torques
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel  
    joint_torque = env.scene["robot"].data.applied_torque
    
    # get the gripper joint indices (last 14 joints)
    gripper_joint_indices=[31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]
    if len(gripper_joint_indices) == 14:
        # extract the gripper joint states in the specified order
        gripper_positions = joint_pos[:, gripper_joint_indices]
        gripper_velocities = joint_vel[:, gripper_joint_indices]  
        gripper_torques = joint_torque[:, gripper_joint_indices]
        
        # publish to DDS (only publish the data of the first environment)
        if enable_dds and len(gripper_positions) > 0:
            try:

                dex3_dds = _get_dex3_dds_instance()
                if dex3_dds:
                    pos = gripper_positions[0].cpu().numpy()
                    vel = gripper_velocities[0].cpu().numpy()
                    torque = gripper_torques[0].cpu().numpy()
                    left_pos = pos[:7]
                    right_pos = pos[7:]
                    left_vel = vel[:7]
                    right_vel = vel[7:]
                    left_torque = torque[:7]
                    right_torque = torque[7:]
                    dex3_dds.write_hand_states(left_pos, left_vel, left_torque, right_pos, right_vel, right_torque)
            except Exception as e:
                print(f"dex3_state [dex3_state] Failed to write to DDS: {e}")
        
        return gripper_positions
    else:
        # if the gripper joints are not found, return a zero tensor
        return torch.zeros((joint_pos.shape[0], 14))

