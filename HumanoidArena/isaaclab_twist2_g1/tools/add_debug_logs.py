#!/usr/bin/env python3
"""
为action_provider_sonic.py添加详细的调试日志

这个脚本会在关键位置插入print语句，追踪数据在每个环节的变化。

使用方法：
1. 备份原文件：cp action_provider_sonic.py action_provider_sonic.py.bak
2. 运行此脚本：python add_debug_logs.py
3. 启动IsaacLab，在VR中规律甩手
4. 观察日志中哪个环节的数据没有规律变化
5. 恢复原文件：mv action_provider_sonic.py.bak action_provider_sonic.py
"""

import os
import sys

# 要插入的调试代码片段
DEBUG_SNIPPETS = {
    # 在_fetch_zmq_pose()函数中，接收到smpl_joints后
    "after_smpl_joints_received": '''
        # [DEBUG] 检查SMPL joints的变化
        if hasattr(self, '_prev_smpl_joints_frame'):
            smpl_diff = np.abs(frame - self._prev_smpl_joints_frame).max()
            print(f"[DEBUG_ZMQ] SMPL joints最大变化: {smpl_diff:.6f}")
        self._prev_smpl_joints_frame = frame.copy()
''',

    # 在_run_gear_sonic()函数中，构建encoder输入后
    "after_encoder_input_built": '''
        # [DEBUG] 检查encoder输入的变化
        if hasattr(self, '_prev_encoder_input'):
            enc_diff = np.abs(encoder_input - self._prev_encoder_input).max()
            print(f"[DEBUG_ENC] Encoder输入最大变化: {enc_diff:.6f}")
            smpl_part = encoder_input[0, -780:-720]  # smpl_joints的一部分
            smpl_part_diff = np.abs(smpl_part - self._prev_encoder_input[0, -780:-720]).max()
            print(f"[DEBUG_ENC] SMPL部分最大变化: {smpl_part_diff:.6f}")
        self._prev_encoder_input = encoder_input.copy()
''',

    # 在encoder推理后
    "after_encoder_inference": '''
        # [DEBUG] 检查latent的变化
        if hasattr(self, '_prev_latent'):
            latent_diff = np.abs(latent - self._prev_latent).max()
            print(f"[DEBUG_LATENT] Latent最大变化: {latent_diff:.6f}")
        self._prev_latent = latent.copy()
''',

    # 在decoder推理后
    "after_decoder_inference": '''
        # [DEBUG] 检查action的变化
        if hasattr(self, '_prev_action_raw'):
            action_diff = np.abs(action_raw - self._prev_action_raw).max()
            print(f"[DEBUG_ACTION] Action最大变化: {action_diff:.6f}")
            print(f"[DEBUG_ACTION] Action前5维: {action_raw[0, :5]}")
        self._prev_action_raw = action_raw.copy()

        # [DEBUG] 检查decoder输入的robot状态
        print(f"[DEBUG_ROBOT] joint_pos前5维: {self._robot_joint_pos_hist[-1, :5]}")
        print(f"[DEBUG_ROBOT] joint_vel前5维: {self._robot_joint_vel_hist[-1, :5]}")
''',

    # 在最终输出前
    "before_final_output": '''
        # [DEBUG] 检查最终targets的变化
        if hasattr(self, '_prev_sonic_targets'):
            targets_diff = np.abs(sonic_targets - self._prev_sonic_targets).max()
            print(f"[DEBUG_TARGETS] Targets最大变化: {targets_diff:.6f}")
            print(f"[DEBUG_TARGETS] Targets前5维: {sonic_targets[:5]}")
        self._prev_sonic_targets = sonic_targets.copy()
''',
}

def add_debug_logs():
    """在action_provider_sonic.py中添加调试日志"""

    file_path = "../action_provider/action_provider_sonic.py"

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        print(f"   请确保在tools/目录下运行此脚本")
        return

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 查找插入位置并插入调试代码
    insertions = []

    # 1. 在smpl_joints接收后（约638行）
    for i, line in enumerate(lines):
        if 'self._smpl_joints_buf[-1] = frame' in line:
            insertions.append((i+1, DEBUG_SNIPPETS["after_smpl_joints_received"]))
            print(f"✓ 找到插入点1: 第{i+1}行 (SMPL joints接收后)")
            break

    # 2. 在encoder输入构建后（约874行）
    for i, line in enumerate(lines):
        if 'print(f"[SONIC] Encoder input shape: {encoder_input.shape}' in line:
            insertions.append((i+1, DEBUG_SNIPPETS["after_encoder_input_built"]))
            print(f"✓ 找到插入点2: 第{i+1}行 (Encoder输入构建后)")
            break

    # 3. 在encoder推理后（约930行）
    for i, line in enumerate(lines):
        if 'print(f"[SONIC] ✓ Encoder output latent shape: {latent.shape}")' in line:
            insertions.append((i+1, DEBUG_SNIPPETS["after_encoder_inference"]))
            print(f"✓ 找到插入点3: 第{i+1}行 (Encoder推理后)")
            break

    # 4. 在decoder推理后（约960行）
    for i, line in enumerate(lines):
        if 'print(f"[SONIC] ✓ Decoder output action shape: {action_raw.shape}")' in line:
            insertions.append((i+1, DEBUG_SNIPPETS["after_decoder_inference"]))
            print(f"✓ 找到插入点4: 第{i+1}行 (Decoder推理后)")
            break

    # 5. 在最终输出前（约1000行）
    for i, line in enumerate(lines):
        if 'sonic_targets = SONIC_DEFAULT_POS + action_scaled' in line:
            insertions.append((i+1, DEBUG_SNIPPETS["before_final_output"]))
            print(f"✓ 找到插入点5: 第{i+1}行 (最终输出前)")
            break

    if len(insertions) < 5:
        print(f"\n❌ 只找到 {len(insertions)}/5 个插入点")
        print(f"   文件可能已被修改，请检查")
        return

    # 按行号倒序插入（避免行号偏移）
    insertions.sort(reverse=True)

    for line_num, code in insertions:
        lines.insert(line_num, code + '\n')

    # 写回文件
    output_path = file_path + ".debug"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"\n✅ 调试版本已生成: {output_path}")
    print(f"\n使用方法:")
    print(f"1. 备份原文件: cp {file_path} {file_path}.bak")
    print(f"2. 替换文件: mv {output_path} {file_path}")
    print(f"3. 启动IsaacLab，在VR中规律甩手（左右甩，约1Hz，持续5秒）")
    print(f"4. 观察日志中的 [DEBUG_*] 输出")
    print(f"5. 恢复原文件: mv {file_path}.bak {file_path}")
    print(f"\n关键指标:")
    print(f"- [DEBUG_ZMQ] SMPL joints最大变化 - 应该>0.01")
    print(f"- [DEBUG_ENC] Encoder输入最大变化 - 应该>0.01")
    print(f"- [DEBUG_LATENT] Latent最大变化 - 应该>0.01")
    print(f"- [DEBUG_ACTION] Action最大变化 - 应该>0.01")
    print(f"- [DEBUG_TARGETS] Targets最大变化 - 应该>0.01")
    print(f"\n如果某个环节的变化<0.001，说明问题就在这个环节！")

if __name__ == "__main__":
    add_debug_logs()
