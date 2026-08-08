#!/usr/bin/env python3
"""
检测SONIC数据流中的规律变化

用法：
1. 启动pico_server
2. 运行此脚本：python detect_periodic_motion.py
3. 在VR中规律地甩手（比如左右甩，频率约1Hz）
4. 脚本会分析数据的周期性，找出哪个环节丢失了规律

输出：
- Pico端数据的周期性分析
- 每个关键数据字段的变化幅度
- FFT频谱分析（检测主频率）
"""

import json
import sys
import time
import numpy as np
import zmq
from collections import deque
import matplotlib
matplotlib.use('Agg')  # 不显示窗口
import matplotlib.pyplot as plt

_HEADER_SIZE = 1280

def parse_zmq_pose(raw: bytes):
    """解析ZMQ pose消息"""
    try:
        topic_end = raw.index(b"{")
        header_bytes = raw[topic_end: topic_end + _HEADER_SIZE]
        payload = raw[topic_end + _HEADER_SIZE:]
        header = json.loads(header_bytes.rstrip(b"\x00").decode("utf-8"))

        _DTYPE_MAP = {
            "f32": (np.float32, 4), "f64": (np.float64, 8),
            "i32": (np.int32, 4), "i64": (np.int64, 8),
            "u8": (np.uint8, 1), "bool": (np.bool_, 1),
        }
        result, offset = {}, 0
        for f in header.get("fields", []):
            np_dtype, itemsize = _DTYPE_MAP.get(f["dtype"], (np.float32, 4))
            n = int(np.prod(f["shape"]))
            arr = np.frombuffer(payload[offset: offset + n * itemsize],
                                dtype=np_dtype).reshape(f["shape"])
            result[f["name"]] = arr
            offset += n * itemsize
        return result
    except Exception as e:
        return None

def analyze_periodicity(signal, fps=50):
    """分析信号的周期性"""
    if len(signal) < 50:
        return None, None

    # 去除均值
    signal = signal - np.mean(signal)

    # FFT
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1.0/fps)

    # 只看正频率
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft = np.abs(fft[:len(fft)//2])

    # 找主频率（排除DC分量）
    if len(positive_freqs) > 1:
        main_freq_idx = np.argmax(positive_fft[1:]) + 1
        main_freq = positive_freqs[main_freq_idx]
        main_power = positive_fft[main_freq_idx]
        return main_freq, main_power
    return None, None

def detect_periodic_motion():
    """检测周期性运动"""
    ZMQ_HOST = "localhost"
    ZMQ_PORT = 5556
    BUFFER_SIZE = 200  # 4秒数据（50Hz）

    print("=" * 80)
    print("SONIC周期性运动检测")
    print("=" * 80)
    print(f"连接到: {ZMQ_HOST}:{ZMQ_PORT}")
    print("\n请在VR中做规律的甩手动作（左右甩，频率约1Hz，持续5秒）")
    print("按Ctrl+C停止\n")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{ZMQ_HOST}:{ZMQ_PORT}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVTIMEO, 1000)

    # 数据缓冲
    smpl_joints_history = deque(maxlen=BUFFER_SIZE)
    body_quat_history = deque(maxlen=BUFFER_SIZE)
    joint_pos_history = deque(maxlen=BUFFER_SIZE)

    # 提取特定关节的历史（右手相关）
    right_shoulder_pitch_history = deque(maxlen=BUFFER_SIZE)  # joint_pos[12]
    right_elbow_history = deque(maxlen=BUFFER_SIZE)  # joint_pos[22]

    # SMPL右手腕位置历史
    smpl_right_wrist_x_history = deque(maxlen=BUFFER_SIZE)  # smpl_joints[20, 0]

    frame_count = 0
    start_time = time.time()

    print("开始采集数据...\n")

    try:
        while True:
            try:
                raw = socket.recv()
            except zmq.error.Again:
                continue

            data = parse_zmq_pose(raw)
            if data is None:
                continue

            frame_count += 1

            # 提取数据
            smpl_joints = data.get("smpl_joints", np.zeros((1, 24, 3)))[-1]  # (24, 3)
            body_quat = data.get("body_quat_w", np.zeros((1, 4)))[-1]  # (4,)
            joint_pos = data.get("joint_pos", np.zeros((1, 29)))[-1]  # (29,)

            # 存入历史
            smpl_joints_history.append(smpl_joints)
            body_quat_history.append(body_quat)
            joint_pos_history.append(joint_pos)

            # 提取关键关节
            right_shoulder_pitch_history.append(joint_pos[12])
            right_elbow_history.append(joint_pos[22])
            smpl_right_wrist_x_history.append(smpl_joints[20, 0])  # 右手腕X坐标

            # 每秒报告一次
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"已采集 {frame_count} 帧 ({elapsed:.1f}秒)")

                # 如果数据足够，进行周期性分析
                if len(smpl_right_wrist_x_history) >= 100:
                    print("\n" + "=" * 80)
                    print("周期性分析（最近2秒数据）")
                    print("=" * 80)

                    # 分析SMPL右手腕X坐标
                    signal = np.array(list(smpl_right_wrist_x_history))
                    freq, power = analyze_periodicity(signal)
                    std = np.std(signal)

                    print(f"\n📊 SMPL右手腕X坐标:")
                    print(f"   标准差: {std:.4f}")
                    if freq is not None:
                        print(f"   主频率: {freq:.2f} Hz")
                        print(f"   功率: {power:.2f}")
                        if freq > 0.5 and freq < 2.0:
                            print(f"   ✅ 检测到周期性运动！频率约 {freq:.2f} Hz")
                        else:
                            print(f"   ⚠️  频率异常（预期0.5-2Hz）")
                    else:
                        print(f"   ❌ 未检测到明显周期性")

                    # 分析joint_pos右肩pitch
                    signal = np.array(list(right_shoulder_pitch_history))
                    freq, power = analyze_periodicity(signal)
                    std = np.std(signal)

                    print(f"\n📊 G1右肩pitch (joint_pos[12]):")
                    print(f"   标准差: {std:.4f}")
                    if freq is not None:
                        print(f"   主频率: {freq:.2f} Hz")
                        print(f"   功率: {power:.2f}")
                        if freq > 0.5 and freq < 2.0:
                            print(f"   ✅ 检测到周期性运动！频率约 {freq:.2f} Hz")
                        else:
                            print(f"   ⚠️  频率异常（预期0.5-2Hz）")
                    else:
                        print(f"   ❌ 未检测到明显周期性")

                    # 分析joint_pos右肘
                    signal = np.array(list(right_elbow_history))
                    freq, power = analyze_periodicity(signal)
                    std = np.std(signal)

                    print(f"\n📊 G1右肘 (joint_pos[22]):")
                    print(f"   标准差: {std:.4f}")
                    if freq is not None:
                        print(f"   主频率: {freq:.2f} Hz")
                        print(f"   功率: {power:.2f}")
                        if freq > 0.5 and freq < 2.0:
                            print(f"   ✅ 检测到周期性运动！频率约 {freq:.2f} Hz")
                        else:
                            print(f"   ⚠️  频率异常（预期0.5-2Hz）")
                    else:
                        print(f"   ❌ 未检测到明显周期性")

                    print("\n" + "=" * 80)
                    print("继续采集数据...\n")

    except KeyboardInterrupt:
        print("\n\n停止采集")

        # 最终分析
        if len(smpl_right_wrist_x_history) >= 50:
            print("\n" + "=" * 80)
            print("最终分析报告")
            print("=" * 80)

            # 绘制时间序列图
            fig, axes = plt.subplots(3, 1, figsize=(12, 8))

            # SMPL右手腕X
            signal1 = np.array(list(smpl_right_wrist_x_history))
            axes[0].plot(signal1)
            axes[0].set_title('SMPL Right Wrist X Coordinate')
            axes[0].set_ylabel('Position (m)')
            axes[0].grid(True)

            # G1右肩pitch
            signal2 = np.array(list(right_shoulder_pitch_history))
            axes[1].plot(signal2)
            axes[1].set_title('G1 Right Shoulder Pitch (joint_pos[12])')
            axes[1].set_ylabel('Angle (rad)')
            axes[1].grid(True)

            # G1右肘
            signal3 = np.array(list(right_elbow_history))
            axes[2].plot(signal3)
            axes[2].set_title('G1 Right Elbow (joint_pos[22])')
            axes[2].set_ylabel('Angle (rad)')
            axes[2].set_xlabel('Frame')
            axes[2].grid(True)

            plt.tight_layout()
            plt.savefig('periodic_motion_analysis.png', dpi=150)
            print("\n📈 时间序列图已保存到: periodic_motion_analysis.png")

            # 数值分析
            print(f"\n📊 数值统计:")
            print(f"   SMPL右手腕X: 范围[{signal1.min():.3f}, {signal1.max():.3f}], 标准差={np.std(signal1):.4f}")
            print(f"   G1右肩pitch: 范围[{signal2.min():.3f}, {signal2.max():.3f}], 标准差={np.std(signal2):.4f}")
            print(f"   G1右肘:      范围[{signal3.min():.3f}, {signal3.max():.3f}], 标准差={np.std(signal3):.4f}")

            # 判断
            print(f"\n🔍 诊断结果:")
            smpl_has_variation = np.std(signal1) > 0.01
            g1_shoulder_has_variation = np.std(signal2) > 0.01
            g1_elbow_has_variation = np.std(signal3) > 0.01

            if smpl_has_variation:
                print(f"   ✅ SMPL数据有明显变化")
            else:
                print(f"   ❌ SMPL数据几乎不变 - 问题在Pico追踪")

            if g1_shoulder_has_variation:
                print(f"   ✅ G1关节位置有明显变化")
            else:
                print(f"   ❌ G1关节位置几乎不变 - 问题在SMPL→G1转换")

            if smpl_has_variation and not g1_shoulder_has_variation:
                print(f"\n⚠️  关键发现：SMPL数据正常，但G1关节位置不变！")
                print(f"   这说明问题出在pico_server中的关节位置计算（1286-1344行）")
                print(f"   或者joint_pos数据没有正确发送到IsaacLab")

if __name__ == "__main__":
    detect_periodic_motion()
