#!/usr/bin/env python3
"""
SONIC实时数据监控工具

用法：
1. 启动pico_server
2. 运行此脚本：python monitor_sonic_live.py
3. 在VR中做动作，观察数据变化

显示：
- SMPL joints的变化幅度
- Body orientation的变化
- Joint positions的变化
- 数据接收频率
"""

import json
import sys
import time
import numpy as np
import zmq
from collections import deque

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

def monitor():
    """实时监控数据流"""
    ZMQ_HOST = "localhost"
    ZMQ_PORT = 5556

    print("=" * 80)
    print("SONIC实时数据监控")
    print("=" * 80)
    print(f"连接到: {ZMQ_HOST}:{ZMQ_PORT}")
    print("按Ctrl+C停止\n")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{ZMQ_HOST}:{ZMQ_PORT}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVTIMEO, 1000)

    # 历史数据用于计算变化
    smpl_joints_history = deque(maxlen=10)
    body_quat_history = deque(maxlen=10)
    joint_pos_history = deque(maxlen=10)

    frame_count = 0
    last_report_time = time.time()
    fps_counter = 0

    print("等待数据...\n")

    try:
        while True:
            try:
                raw = socket.recv()
                fps_counter += 1
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

            # 每秒报告一次
            current_time = time.time()
            if current_time - last_report_time >= 1.0:
                elapsed = current_time - last_report_time
                fps = fps_counter / elapsed

                # 计算变化幅度
                if len(smpl_joints_history) >= 2:
                    smpl_diff = np.abs(smpl_joints_history[-1] - smpl_joints_history[0]).max()
                    quat_diff = np.abs(body_quat_history[-1] - body_quat_history[0]).max()
                    joint_diff = np.abs(joint_pos_history[-1] - joint_pos_history[0]).max()
                else:
                    smpl_diff = quat_diff = joint_diff = 0.0

                # 计算数据范围
                smpl_range = f"[{smpl_joints.min():.3f}, {smpl_joints.max():.3f}]"
                joint_range = f"[{joint_pos.min():.3f}, {joint_pos.max():.3f}]"

                # 清屏并打印
                print("\033[2J\033[H")  # ANSI清屏
                print("=" * 80)
                print(f"SONIC实时监控 | 帧: {frame_count} | FPS: {fps:.1f}")
                print("=" * 80)
                print(f"\n📊 数据统计（最近1秒）:")
                print(f"  SMPL Joints 变化幅度: {smpl_diff:.4f}  范围: {smpl_range}")
                print(f"  Body Quat 变化幅度:   {quat_diff:.4f}  当前: {body_quat}")
                print(f"  Joint Pos 变化幅度:   {joint_diff:.4f}  范围: {joint_range}")

                # 状态指示
                print(f"\n🎯 状态:")
                if smpl_diff < 0.001:
                    print(f"  ⚠️  SMPL数据几乎不变 - 请在VR中做动作！")
                elif smpl_diff < 0.01:
                    print(f"  ⚡ SMPL数据有小幅变化")
                else:
                    print(f"  ✅ SMPL数据正常变化")

                if joint_diff < 0.001:
                    print(f"  ⚠️  关节位置几乎不变")
                elif joint_diff < 0.1:
                    print(f"  ⚡ 关节位置有小幅变化")
                else:
                    print(f"  ✅ 关节位置正常变化")

                # 关键关节监控（右手肩膀、肘部、手腕）
                print(f"\n🤖 关键关节位置（右手）:")
                print(f"  右肩pitch [12]: {joint_pos[12]:+.3f}")
                print(f"  右肩roll  [16]: {joint_pos[16]:+.3f}")
                print(f"  右肩yaw   [20]: {joint_pos[20]:+.3f}")
                print(f"  右肘      [22]: {joint_pos[22]:+.3f}")
                print(f"  右腕roll  [24]: {joint_pos[24]:+.3f}")

                print(f"\n💡 提示:")
                print(f"  - 在VR中举起右手，观察右肩/右肘数值变化")
                print(f"  - 如果数值不变，说明数据流有问题")
                print(f"  - 按Ctrl+C停止监控")

                last_report_time = current_time
                fps_counter = 0

    except KeyboardInterrupt:
        print("\n\n监控已停止")

if __name__ == "__main__":
    monitor()
