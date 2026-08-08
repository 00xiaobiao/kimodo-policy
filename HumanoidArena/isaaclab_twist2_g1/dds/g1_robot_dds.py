# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
G1 robot DDS communication class
Handle the state publishing and command receiving of the G1 robot
"""

import numpy as np
from typing import Any, Dict, Optional
# from dds.dds_base import BaseDDSNode, node_manager
from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.utils.crc import CRC


class G1RobotDDS(DDSObject):
    """G1 robot DDS communication class - singleton pattern
    
    Features:
    - Publish the state of the G1 robot to DDS (rt/lowstate)
    - Receive the control command of the G1 robot (rt/lowcmd)
    """
    
    def __init__(self,node_name:str="g1_robot"):
        """Initialize the G1 robot DDS node"""
        # avoid duplicate initialization
        if hasattr(self, '_initialized'):
            return
            
        super().__init__()
        self.node_name = node_name
        self.crc = CRC()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self._initialized = True
        
        # setup the shared memory
        self.setup_shared_memory(
            input_shm_name="isaac_robot_state",  # read the state of the G1 robot from Isaac Lab
            output_shm_name="dds_robot_cmd",  # output the command to Isaac Lab
            input_size=3072,
            output_size=3072  # output the command to Isaac Lab
        )
        
        print(f"[{self.node_name}] G1 robot DDS node initialized")
    
    def setup_publisher(self) -> bool:
        """Setup the publisher of the G1 robot"""
        try:
            self.publisher = ChannelPublisher("rt/lowstate", LowState_)
            self.publisher.Init()
            print(f"[{self.node_name}] State publisher initialized (rt/lowstate)")
            return True
        except Exception as e:
            print(f"g1_robot_dds [{self.node_name}] State publisher initialization failed: {e}")    
            return False
    
    def setup_subscriber(self) -> bool:
        """Setup the subscriber of the G1 robot"""
        try:
            print(f"[{self.node_name}] Create ChannelSubscriber...")
            self.subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
            self.subscriber.Init(lambda msg: self.dds_subscriber(msg, ""), 1)
            return True
        except Exception as e:
            print(f"g1_robot_dds [{self.node_name}] Command subscriber initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def dds_publisher(self) -> Any:
        """Convert Isaac Lab state to DDS message and publish."""
        try:
            data = self.input_shm.read_data()
            if data is None:
                return

            motor_state = self.low_state.motor_state
            imu_state = self.low_state.imu_state
            num_motors = min(29, len(motor_state))  # 假设最多29个电机

            # 使用 NumPy 批量转换 joint 数据
            positions = data.get("joint_positions")
            velocities = data.get("joint_velocities")
            torques = data.get("joint_torques")

            if positions and velocities and torques:
                q_array = np.asarray(positions, dtype=np.float32)
                dq_array = np.asarray(velocities, dtype=np.float32)
                tau_array = np.asarray(torques, dtype=np.float32)

                for i in range(num_motors):
                    motor = motor_state[i]
                    motor.q = q_array[i]
                    motor.dq = dq_array[i]
                    motor.tau_est = tau_array[i]

            # 使用 NumPy 批量转换 IMU 数据
            imu = data.get("imu_data")
            if imu and len(imu) >= 13:
                imu_array = np.asarray(imu, dtype=np.float32)

                # 四元数 (x, y, z, w)
                imu_state.quaternion[:] = imu_array[[4, 5, 6, 3]]

                # 线加速度 (加速度计)
                imu_state.accelerometer[:] = imu_array[7:10]

                # 角速度 (陀螺仪)
                imu_state.gyroscope[:] = imu_array[10:13]

            # 更新时间戳和 CRC
            self.low_state.tick += 1
            self.low_state.crc = self.crc.Crc(self.low_state)

            # 发布 DDS 消息
            self.publisher.Write(self.low_state)

        except Exception as e:
            print(f"g1_robot_dds [{self.node_name}] Error processing publish data: {e}")

    
    def dds_subscriber(self, msg: LowCmd_,datatype:str=None) -> Dict[str, Any]:
        """Process the subscribe data: convert the DDS command to the Isaac Lab format
        
        Return data format:
        {
            "mode_pr": int,
            "mode_machine": int,
            "motor_cmd": {
                "positions": [29 joint position commands],
                "velocities": [29 joint velocity commands],
                "torques": [29 joint torque commands],
                "kp": [29 position gains],
                "kd": [29 speed gains]
            }
        }
        """
        try:
            # verify the CRC
            if self.crc.Crc(msg) != msg.crc:
                print(f"g1_robot_dds [{self.node_name}] Warning: CRC verification failed!")
                return {}
            
            # extract the command data
            num_cmd_motors = min(29, len(msg.motor_cmd))  # G1机器人29个自由度
            cmd_data = {
                "mode_pr": int(msg.mode_pr),
                "mode_machine": int(msg.mode_machine),
                "motor_cmd": {
                    "positions": [float(msg.motor_cmd[i].q) for i in range(num_cmd_motors)],
                    "velocities": [float(msg.motor_cmd[i].dq) for i in range(num_cmd_motors)],
                    "torques": [float(msg.motor_cmd[i].tau) for i in range(num_cmd_motors)],
                    "kp": [float(msg.motor_cmd[i].kp) for i in range(num_cmd_motors)],
                    "kd": [float(msg.motor_cmd[i].kd) for i in range(num_cmd_motors)]
                }
            }
            # print(f"cmd_data: {cmd_data}")
            self.output_shm.write_data(cmd_data)
            
        except Exception as e:
            print(f"g1_robot_dds [{self.node_name}] Error processing subscribe data: {e}")
            return {}
    
    def get_robot_command(self) -> Optional[Dict[str, Any]]:
        """Get the robot control command
        
        Returns:
            Dict: the robot control command, return None if there is no new command
        """
        if self.output_shm:
            return self.output_shm.read_data()
        return None
    
    def write_robot_state(self, joint_positions, joint_velocities, joint_torques, imu_data):
        """Write the robot state to the shared memory
        
        Args:
            joint_positions: the joint position list or torch.Tensor
            joint_velocities: the joint velocity list or torch.Tensor
            joint_torques: the joint torque list or torch.Tensor
            imu_data: the IMU data list or torch.Tensor
        """
        if self.input_shm is None:
            return
        try:
            state_data = {
                "joint_positions": joint_positions.tolist() if hasattr(joint_positions, 'tolist') else joint_positions,
                "joint_velocities": joint_velocities.tolist() if hasattr(joint_velocities, 'tolist') else joint_velocities,
                "joint_torques": joint_torques.tolist() if hasattr(joint_torques, 'tolist') else joint_torques,
                "imu_data": imu_data.tolist() if hasattr(imu_data, 'tolist') else imu_data
            }
            self.input_shm.write_data(state_data)
        except Exception as e:
            print(f"g1_robot_dds [{self.node_name}] Error writing robot state: {e}")