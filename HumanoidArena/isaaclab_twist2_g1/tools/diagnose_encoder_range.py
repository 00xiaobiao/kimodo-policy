#!/usr/bin/env python3
"""
诊断encoder输入中哪个字段超出正常范围
"""

import numpy as np

# 从日志中提取的encoder输入范围
encoder_input_min = -31.945160
encoder_input_max = 29.832436

print("=" * 80)
print("Encoder输入范围诊断")
print("=" * 80)
print(f"\n当前范围: [{encoder_input_min:.6f}, {encoder_input_max:.6f}]")
print(f"正常范围: [-10.0, 10.0]")
print(f"\n❌ 输入范围异常！超出正常范围约 {max(abs(encoder_input_min), abs(encoder_input_max)) / 10:.1f}x")

print("\n" + "=" * 80)
print("各字段的正常范围")
print("=" * 80)

fields = [
    ("encoder_mode", 4, "[-1, 1]", "模式标志"),
    ("motion_joint_pos_step5", 290, "[-3.14, 3.14]", "关节位置（弧度）"),
    ("motion_joint_vel_step5", 290, "[-10, 10]", "关节速度（rad/s）"),
    ("motion_root_z_step5", 10, "[0, 2]", "根位置高度（m）"),
    ("motion_root_z", 1, "[0, 2]", "根位置高度（m）"),
    ("motion_anchor_orient", 6, "[-1, 1]", "6D旋转"),
    ("motion_anchor_orient_step5", 60, "[-1, 1]", "6D旋转历史"),
    ("motion_joint_pos_lowerbody", 120, "[-3.14, 3.14]", "下半身关节位置"),
    ("motion_joint_vel_lowerbody", 120, "[-10, 10]", "下半身关节速度"),
    ("vr_3pt_pos", 9, "[-2, 2]", "VR目标位置"),
    ("vr_3pt_orn", 12, "[-1, 1]", "VR目标旋转"),
    ("smpl_joints", 720, "[-1, 1]", "SMPL关节位置（局部坐标）"),
    ("smpl_anchor_orient", 60, "[-1, 1]", "SMPL锚点旋转"),
    ("motion_wrist_pos", 60, "[-3.14, 3.14]", "手腕关节位置"),
]

print("\n字段名                          维度    正常范围           说明")
print("-" * 80)
for name, dims, range_str, desc in fields:
    print(f"{name:30s} {dims:4d}    {range_str:15s}    {desc}")

print("\n" + "=" * 80)
print("最可能的问题")
print("=" * 80)

print("\n1. 关节速度过大（motion_joint_vel_step5）")
print("   原因：仿真不稳定，机器人在抖动")
print("   解决：检查PD参数，或者clip速度到[-10, 10]")

print("\n2. SMPL数据单位错误（smpl_joints）")
print("   原因：SMPL数据可能是mm而不是m")
print("   解决：检查pico_server中的SMPL数据单位")

print("\n3. 历史缓冲区未初始化")
print("   原因：历史缓冲区包含NaN或极大值")
print("   解决：检查on_env_reset是否正确初始化")

print("\n" + "=" * 80)
print("下一步")
print("=" * 80)
print("\n请在IsaacLab日志中查找：")
print("  [SONIC][ENCODER_INPUT] motion_pos_step5=[...] motion_vel_step5=[...]")
print("\n然后告诉我哪个字段的范围异常。")
