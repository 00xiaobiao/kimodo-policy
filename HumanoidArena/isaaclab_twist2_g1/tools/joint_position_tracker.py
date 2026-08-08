#!/usr/bin/env python3
"""
关节位置跟踪统计工具（纯文本版本，无GUI）
实时统计29个关节的跟踪误差，用于PD参数调试
"""

import numpy as np
from collections import deque
import time


class JointPositionTracker:
    """跟踪关节目标位置和当前位置的统计信息"""

    def __init__(self, num_joints=29, window_size=200):
        """
        Args:
            num_joints: 关节数量
            window_size: 滑动窗口大小（用于计算移动平均）
        """
        self.num_joints = num_joints
        self.window_size = window_size

        # 数据缓冲区
        self.errors = [deque(maxlen=window_size) for _ in range(num_joints)]
        self.timestamps = deque(maxlen=window_size)

        # 关节名称（G1 29DOF）
        self.joint_names = [
            # Legs (12)
            "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
            "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
            # Waist (3)
            "waist_yaw", "waist_roll", "waist_pitch",
            # Arms (14)
            "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow",
            "L_wrist_roll", "L_wrist_pitch", "L_wrist_yaw",
            "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow",
            "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw",
        ]

        # 统计信息
        self.max_error = np.zeros(num_joints)
        self.mean_error = np.zeros(num_joints)
        self.current_error = np.zeros(num_joints)

        # 最近的目标和当前位置
        self.last_target = None
        self.last_current = None

        # 更新计数
        self.update_count = 0

    def update_data(self, target_pos, current_pos, timestamp=None):
        """
        更新数据

        Args:
            target_pos: 目标位置 (num_joints,)
            current_pos: 当前位置 (num_joints,)
            timestamp: 时间戳（可选）
        """
        if timestamp is None:
            timestamp = self.update_count

        self.timestamps.append(timestamp)
        self.last_target = target_pos.copy()
        self.last_current = current_pos.copy()

        for i in range(self.num_joints):
            error = abs(target_pos[i] - current_pos[i])
            self.errors[i].append(error)
            self.current_error[i] = error

            # 更新统计
            self.max_error[i] = max(self.max_error[i], error)
            if len(self.errors[i]) > 0:
                self.mean_error[i] = np.mean(list(self.errors[i]))

        self.update_count += 1

    def get_statistics(self):
        """获取统计信息"""
        stats = {}
        for i in range(self.num_joints):
            joint_name = self.joint_names[i] if i < len(self.joint_names) else f"Joint_{i}"
            stats[joint_name] = {
                'current_error': self.current_error[i],
                'max_error': self.max_error[i],
                'mean_error': self.mean_error[i],
            }
        return stats

    def print_statistics(self, top_n=10):
        """打印统计信息，只显示误差最大的前N个关节"""
        print("\n" + "="*100)
        print(f"Joint Tracking Statistics (Updates: {self.update_count}, Window: {len(self.timestamps)})")
        print("="*100)

        # 按当前误差排序
        stats = self.get_statistics()
        sorted_joints = sorted(stats.items(), key=lambda x: x[1]['current_error'], reverse=True)

        print(f"{'Joint Name':<25} | {'Current Error':>12} | {'Mean Error':>12} | {'Max Error':>12}")
        print("-"*100)

        # 显示前N个误差最大的关节
        for joint_name, stat in sorted_joints[:top_n]:
            print(f"{joint_name:<25} | {stat['current_error']:12.6f} | {stat['mean_error']:12.6f} | {stat['max_error']:12.6f}")

        if len(sorted_joints) > top_n:
            print(f"... and {len(sorted_joints) - top_n} more joints")

        # 总体统计
        all_current_errors = [s['current_error'] for s in stats.values()]
        all_mean_errors = [s['mean_error'] for s in stats.values()]
        all_max_errors = [s['max_error'] for s in stats.values()]

        print("-"*100)
        print(f"{'OVERALL':<25} | {np.mean(all_current_errors):12.6f} | {np.mean(all_mean_errors):12.6f} | {np.max(all_max_errors):12.6f}")
        print("="*100 + "\n")

    def print_compact_summary(self):
        """打印紧凑的摘要信息"""
        if self.update_count == 0:
            return

        stats = self.get_statistics()
        all_current_errors = [s['current_error'] for s in stats.values()]
        all_mean_errors = [s['mean_error'] for s in stats.values()]

        # 找出误差最大的3个关节
        sorted_joints = sorted(stats.items(), key=lambda x: x[1]['current_error'], reverse=True)
        top3 = sorted_joints[:3]

        print(f"[JointTracker] Updates: {self.update_count:6d} | "
              f"Avg Error: {np.mean(all_current_errors):.4f} | "
              f"Top3: {top3[0][0]}({top3[0][1]['current_error']:.4f}), "
              f"{top3[1][0]}({top3[1][1]['current_error']:.4f}), "
              f"{top3[2][0]}({top3[2][1]['current_error']:.4f})")


# 示例用法
if __name__ == "__main__":
    import time

    # 创建跟踪器
    tracker = JointPositionTracker(num_joints=29, window_size=200)

    print("Simulating joint tracking...")
    print("Press Ctrl+C to stop")

    try:
        t = 0
        while True:
            # 生成模拟数据
            target = np.sin(t * 0.1 + np.arange(29) * 0.2)
            current = target + np.random.randn(29) * 0.05  # 添加噪声

            tracker.update_data(target, current, timestamp=t)

            t += 1
            time.sleep(0.02)  # 50Hz

            # 每50步打印紧凑摘要
            if t % 50 == 0:
                tracker.print_compact_summary()

            # 每200步打印详细统计
            if t % 200 == 0:
                tracker.print_statistics(top_n=10)

    except KeyboardInterrupt:
        print("\nStopping...")
        tracker.print_statistics(top_n=29)
