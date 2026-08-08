#!/usr/bin/env python3
"""
关节位置可视化集成模块
可以轻松集成到任何仿真脚本中
"""

import torch
import numpy as np
import sys
import os

# Fix Qt plugin conflict with OpenCV
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

print("[JointVizIntegration] Importing visualizer...")
try:
    from tools.joint_position_visualizer import JointPositionVisualizer
    print("[JointVizIntegration] Visualizer imported successfully")
except Exception as e:
    print(f"[JointVizIntegration] ERROR: Failed to import visualizer: {e}")
    print(f"[JointVizIntegration] sys.path: {sys.path[:3]}")
    raise


class JointVisualizerIntegration:
    """集成到仿真循环的可视化器包装类"""

    def __init__(self, robot, env_id=0, window_size=200, update_frequency=2):
        """
        Args:
            robot: Isaac Lab robot对象
            env_id: 要可视化的环境ID
            window_size: 滑动窗口大小
            update_frequency: 更新频率（每N步更新一次）
        """
        print(f"[JointVizIntegration] Initializing with robot: {type(robot)}")
        print(f"[JointVizIntegration] Robot num_joints: {robot.num_joints}")

        self.robot = robot
        self.env_id = env_id
        self.update_frequency = update_frequency
        self.step_count = 0

        # 创建可视化器（在主线程中）
        print(f"[JointVizIntegration] Creating visualizer in main thread...")
        try:
            self.visualizer = JointPositionVisualizer(
                num_joints=robot.num_joints,
                window_size=window_size,
                update_interval=50
            )
            print(f"[JointVizIntegration] Visualizer created successfully")
        except Exception as e:
            print(f"[JointVizIntegration] ERROR creating visualizer: {e}")
            import traceback
            traceback.print_exc()
            raise

        # 更新关节名称
        if hasattr(robot.data, 'joint_names'):
            self.visualizer.joint_names = robot.data.joint_names
            print(f"[JointVizIntegration] Updated joint names: {len(robot.data.joint_names)} joints")

        # 启动非阻塞模式（不使用线程，在主线程中更新）
        print(f"[JointVizIntegration] Starting visualizer in non-blocking mode...")
        try:
            self.visualizer.start_non_blocking()
            print(f"[JointVizIntegration] Visualizer started successfully")
        except Exception as e:
            print(f"[JointVizIntegration] ERROR starting visualizer: {e}")
            import traceback
            traceback.print_exc()
            raise

        print(f"\n{'='*80}")
        print("Joint Position Visualizer Started")
        print(f"{'='*80}")
        print(f"Monitoring {robot.num_joints} joints from environment {env_id}")
        print("Controls:")
        print("  - Left/Right Arrow: Navigate between joint groups")
        print("  - Close window or Ctrl+C to stop")
        print(f"{'='*80}\n")

    def update(self):
        """在仿真循环中调用此方法"""
        self.step_count += 1

        if self.step_count % self.update_frequency == 0:
            # 获取目标位置和当前位置
            target_pos = self.robot.data.joint_pos_target[self.env_id].cpu().numpy()
            current_pos = self.robot.data.joint_pos[self.env_id].cpu().numpy()

            self.visualizer.update_data(target_pos, current_pos, timestamp=self.step_count)

    def print_statistics(self):
        """打印统计信息"""
        self.visualizer.print_statistics()

    def get_statistics(self):
        """获取统计信息"""
        return self.visualizer.get_statistics()


# 使用示例：
# 在你的 sim_main.py 中添加以下代码：
#
# from tools.joint_visualizer_integration import JointVisualizerIntegration
#
# # 在创建环境后：
# if args.visualize_joints:
#     joint_viz = JointVisualizerIntegration(env.scene["robot"], env_id=0)
#
# # 在主循环中：
# while simulation_app.is_running():
#     obs, reward, terminated, truncated, info = env.step(actions)
#
#     if args.visualize_joints:
#         joint_viz.update()
#
#     # 每1000步打印统计
#     if args.visualize_joints and step_count % 1000 == 0:
#         joint_viz.print_statistics()
