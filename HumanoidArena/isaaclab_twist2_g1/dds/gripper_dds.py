# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
Gripper DDS communication class
Handle the state publishing and command receiving of the gripper
"""

import threading
from typing import Any, Dict, Optional
from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_, unitree_go_msg_dds__MotorState_


class GripperDDS(DDSObject):
    """Gripper DDS communication class - singleton pattern
    
    Features:
    - Publish the state of the gripper to DDS (rt/unitree_actuator/state)
    - Receive the control command of the gripper (rt/unitree_actuator/cmd)
    """
    
    def __init__(self,node_name:str="gripper"):
        """Initialize the gripper DDS node"""
        # avoid duplicate initialization
        if hasattr(self, '_initialized'):
            return
            
        super().__init__()
        self.node_name = node_name
        
        # initialize the gripper state message (2 grippers)
        self.left_gripper_state = MotorStates_()
        self.right_gripper_state = MotorStates_()
        for i in range(1):
            motor_state = unitree_go_msg_dds__MotorState_()
            self.left_gripper_state.states.append(motor_state)
            self.right_gripper_state.states.append(motor_state)
        self._initialized = True
        
        # setup the shared memory
        self.setup_shared_memory(
            input_shm_name="isaac_gripper_state",  # read the state of the gripper from Isaac Lab
            input_size=512,
            output_shm_name="isaac_gripper_cmd",  # output the command to Isaac Lab
            output_size=512,  # output the command to Isaac Lab
        )
        
        print(f"[{self.node_name}] Gripper DDS node initialized")
    
    def setup_publisher(self) -> bool:
        """Setup the publisher of the gripper"""
        try:
            self.left_gripper_state_publisher = ChannelPublisher("rt/dex1/left/state", MotorStates_)
            self.left_gripper_state_publisher.Init()
            self.right_gripper_state_publisher = ChannelPublisher("rt/dex1/right/state", MotorStates_)
            self.right_gripper_state_publisher.Init()
            self.publisher=True
            print(f"[{self.node_name}] Gripper state publisher initialized")
            return True
        except Exception as e:
            print(f"gripper_dds [{self.node_name}] Gripper state publisher initialization failed: {e}")
            return False
    
    def setup_subscriber(self) -> bool:
        """Setup the subscriber of the gripper"""
        try:
            self.left_gripper_cmd_subscriber = ChannelSubscriber("rt/dex1/left/cmd", MotorCmds_)
            self.left_gripper_cmd_subscriber.Init(lambda msg: self.dds_subscriber(msg, "left"), 1)
            self.right_gripper_cmd_subscriber = ChannelSubscriber("rt/dex1/right/cmd", MotorCmds_)
            self.right_gripper_cmd_subscriber.Init(lambda msg: self.dds_subscriber(msg, "right"), 1)
            self.subscriber = True
            print(f"[{self.node_name}] Gripper command subscriber initialized")
            return True
        except Exception as e:
            print(f"gripper_dds [{self.node_name}] Gripper command subscriber initialization failed: {e}")
            return False
        
    def dds_subscriber(self, msg: Any, hand_side: str):
        """Subscribe message handler"""
        try:
            # process received message
            data = self._process_subscribe_data(msg, hand_side)
            if data and self.output_shm:
                # write to shared memory
                existing_data = self.output_shm.read_data() or {}
                existing_data[f"{hand_side}_gripper_cmd"] = data
                self.output_shm.write_data(existing_data)
        except Exception as e:
            print(f"gripper_dds [{self.node_name}] Error processing subscribe message: {e}")
    def dds_publisher(self) -> Any:
        """Process the publish data: convert the Isaac Lab state to the DDS message
        
        Expected data format:
        {
            "positions": [2 gripper joint positions] (Isaac Lab joint angle range [-0.02, 0.03])
            "velocities": [2 gripper joint velocities],
            "torques": [2 gripper joint torques]
        }
        """
        try:
            data = self.input_shm.read_data() or {}
            # process the left hand data
            if "left_hand" in data:
                left_data = data["left_hand"]
                self._update_gripper_state(self.left_gripper_state, left_data)
                if self.left_gripper_state_publisher:
                    self.left_gripper_state_publisher.Write(self.left_gripper_state)
            
            # process the right hand data
            if "right_hand" in data:
                right_data = data["right_hand"]
                self._update_gripper_state(self.right_gripper_state, right_data)
                if self.right_gripper_state_publisher:
                    self.right_gripper_state_publisher.Write(self.right_gripper_state)

            return None
        except Exception as e:
            print(f"gripper_dds [{self.node_name}] Error processing publish data: {e}")    
            return None
    def _update_gripper_state(self, gripper_state, gripper_data: Dict[str, Any]):
        """Update the gripper state"""
        try:
            if all(key in gripper_data for key in ["positions", "velocities", "torques"]):
                positions = gripper_data["positions"]
                velocities = gripper_data["velocities"]
                torques = gripper_data["torques"]
                for i in range(min(1, len(positions))):  # at most 2 grippers
                    if i < len(positions):
                        gripper_state.states[i].q = self.convert_to_gripper_range(float(positions[i]))
                    if i < len(velocities):
                        gripper_state.states[i].dq = float(velocities[i])
                    if i < len(torques):
                        gripper_state.states[i].tau_est = float(torques[i])
        except Exception as e:
            print(f"gripper_dds [{self.node_name}] Error updating gripper state: {e}")
    def _process_subscribe_data(self, msg: Any, hand_side: str) -> Dict[str, Any]:
        """Process the subscribe data: convert the DDS command to the Isaac Lab format
        
        Returns:
            Dict: the gripper command, format as follows:
            {
                "positions": [2 gripper joint position target values] (Isaac Lab joint angle)
                "velocities": [2 gripper joint velocity target values],
                "torques": [2 gripper joint torque target values],
                "kp": [2 gripper position gains],
                "kd": [2 gripper speed gains]
            }
        """
        try:
            cmd_data = {
                "positions": [],
                "velocities": [],
                "torques": [],
                "kp": [],
                "kd": []
            }
            # process the gripper command (at most 2 grippers)
            for i in range(min(1, len(msg.cmds))):
                # convert the gripper control value to the Isaac Lab joint angle
                joint_angle = self.convert_to_joint_range(float(msg.cmds[i].q))
                
                cmd_data["positions"].append(joint_angle)
                cmd_data["velocities"].append(float(msg.cmds[i].dq))
                cmd_data["torques"].append(float(msg.cmds[i].tau))
                cmd_data["kp"].append(float(msg.cmds[i].kp))
                cmd_data["kd"].append(float(msg.cmds[i].kd))
            
            return cmd_data
            
        except Exception as e:
            print(f"gripper_dds [{self.node_name}] Error processing {hand_side} subscribe data: {e}")
            return {}
    
    def get_gripper_command(self) -> Optional[Dict[str, Any]]:
        """Get the gripper control command
        
        Returns:
            Dict: the gripper command, return None if there is no new command
        """
        if self.output_shm:
            return self.output_shm.read_data()
        return None
    
    def write_gripper_state(self, left_positions, left_velocities, left_torques, 
                        right_positions, right_velocities, right_torques):
        """Write the hand states to the shared memory directly
        
        Args:
            left_positions: the list or torch.Tensor of the left hand joint positions
            left_velocities: the list or torch.Tensor of the left hand joint velocities
            left_torques: the list or torch.Tensor of the left hand joint torques
            right_positions: the list or torch.Tensor of the right hand joint positions
            right_velocities: the list or torch.Tensor of the right hand joint velocities
            right_torques: the list or torch.Tensor of the right hand joint torques
        """
        try:
            # prepare the left hand data
            left_hand_data = {
                "positions": left_positions.tolist() if hasattr(left_positions, 'tolist') else left_positions,
                "velocities": left_velocities.tolist() if hasattr(left_velocities, 'tolist') else left_velocities,
                "torques": left_torques.tolist() if hasattr(left_torques, 'tolist') else left_torques
            }
            
            # prepare the right hand data
            right_hand_data = {
                "positions": right_positions.tolist() if hasattr(right_positions, 'tolist') else right_positions,
                "velocities": right_velocities.tolist() if hasattr(right_velocities, 'tolist') else right_velocities,
                "torques": right_torques.tolist() if hasattr(right_torques, 'tolist') else right_torques
            }
            
            # publish the states
            self.publish_hand_states(left_hand_data, right_hand_data)
            
        except Exception as e:
            print(f"dex3_dds [{self.node_name}] Error writing hand states: {e}")
    def publish_hand_states(self, left_hand_data: Dict[str, Any], right_hand_data: Dict[str, Any]):
        """Publish the left and right hand states
        
        Args:
            left_hand_data: the data of the left hand
            right_hand_data: the data of the right hand
        """
        try:
            combined_data = {
                "left_hand": left_hand_data,
                "right_hand": right_hand_data
            }
            
            # write to the input shared memory for publishing
            if self.input_shm:
                self.input_shm.write_data(combined_data)
                
        except Exception as e:
            print(f"dex3_dds [{self.node_name}] Error publishing hand states: {e}")
    def convert_to_joint_range(self, value):
        """Convert the command value to the Isaac Lab joint angle [5.6, 0] -> [-0.02, 0.03]
        
        Args:
            value: the input value, range in [5.6, 0]
                  5.6: fully open
                  0.0: fully closed
            
        Returns:
            float: the converted value, range in [-0.02, 0.03]
                  -0.02: fully open
                  0.03: fully closed
        """
        # input range (gripper control value)
        input_min = 0.0    # fully closed
        input_max = 5.6    # fully open
        
        # output range (joint angle)
        output_min = 0.03  # fully closed
        output_max = -0.02 # fully open
        
        # ensure the input value is in the valid range
        value = max(input_min, min(input_max, value))
        
        # linear mapping conversion
        converted_value = output_min + (output_max - output_min) * (value - input_min) / (input_max - input_min)
        
        return converted_value

    def convert_to_gripper_range(self, value):
        """Convert the Isaac Lab joint angle to the gripper control value [-0.02, 0.03] -> [5.6, 0]
        
        Args:
            value: the input value, range in [-0.02, 0.03]
                  -0.02: fully open
                  0.03: fully closed
            
        Returns:
            float: the converted value, range in [5.6, 0]
                  5.6: fully open
                  0.0: fully closed
        """
        # input range (joint angle)
        input_min = 0.03   # fully closed
        input_max = -0.02  # fully open
        
        # output range (gripper control value)
        output_min = 0.0   # fully closed
        output_max = 5.6   # fully open
        
        # ensure the input value is in the valid range
        value = max(input_max, min(input_min, value))
        
        # linear mapping conversion
        converted_value = output_min + (output_max - output_min) * (input_min - value) / (input_min - input_max)
        
        return converted_value
