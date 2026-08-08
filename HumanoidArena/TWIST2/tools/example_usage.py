#!/usr/bin/env python3

"""
EpisodeReader使用示例

展示如何使用EpisodeReader类来：
1. 读取episode数据
2. 生成视频
3. 可视化关节点
4. 访问各种数据
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from episode_reader import EpisodeReader


def example_basic_usage():
    """基本使用示例"""
    print("\n" + "="*60)
    print("示例 1: 基本使用")
    print("="*60)

    # 加载episode数据
    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    # 打印详细信息
    reader.print_info()

    # 获取总帧数
    print(f"总帧数: {len(reader)}")

    # 获取第一帧数据
    frame_0 = reader.get_frame(0)
    print(f"\n第一帧的数据键: {list(frame_0.keys())}")


def example_create_videos():
    """生成视频示例"""
    print("\n" + "="*60)
    print("示例 2: 生成视频")
    print("="*60)

    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)

    # 生成前置相机视频
    if reader.has_front_cam:
        print("\n1. 生成前置相机视频...")
        reader.create_video(
            output_path=f"{output_dir}/front_camera.mp4",
            camera="front",
            fps=30
        )

    # 生成世界相机视频
    if reader.has_world_cam:
        print("\n2. 生成世界相机视频...")
        reader.create_video(
            output_path=f"{output_dir}/world_camera.mp4",
            camera="world",
            fps=30
        )


def example_visualize_keypoints():
    """可视化关节点示例"""
    print("\n" + "="*60)
    print("示例 3: 可视化关节点")
    print("="*60)

    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)

    # 在世界相机上可视化关节点
    if reader.has_world_cam:
        print("\n生成带关节点的世界相机视频...")
        reader.visualize_keypoints_on_world_cam(
            output_path=f"{output_dir}/world_with_keypoints.mp4",
            keypoint_radius=3,
            keypoint_color=(255, 0, 0),  # 红色 (RGB)
            show_frame_number=True,
            fps=30
        )


def example_access_data():
    """访问各种数据示例"""
    print("\n" + "="*60)
    print("示例 4: 访问数据")
    print("="*60)

    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    frame_idx = 100

    # 获取图像
    print(f"\n1. 获取第{frame_idx}帧的图像")
    if reader.has_front_cam:
        front_img = reader.get_image(frame_idx, "front")
        print(f"   前置相机图像形状: {front_img.shape}")

    if reader.has_world_cam:
        world_img = reader.get_image(frame_idx, "world")
        print(f"   世界相机图像形状: {world_img.shape}")

    # 获取关节点
    print(f"\n2. 获取关节点")
    keypoints = reader.get_keypoints(frame_idx)
    if keypoints:
        visible_count = sum(1 for kp in keypoints if kp is not None)
        print(f"   总关节点数: {len(keypoints)}")
        print(f"   可见关节点数: {visible_count}")
        if keypoints[0] is not None:
            print(f"   第一个关节点位置: {keypoints[0]}")

    # 获取SMPLX数据
    print(f"\n3. 获取SMPLX数据")
    smplx_data = reader.get_smplx_data(frame_idx)
    if smplx_data:
        print(f"   SMPLX数据键: {list(smplx_data.keys())}")

    # 获取状态和动作
    print(f"\n4. 获取状态和动作数据")
    state_body = reader.get_state_body(frame_idx)
    action_body = reader.get_action_body(frame_idx)
    if state_body:
        print(f"   身体状态维度: {len(state_body)}")
    if action_body:
        print(f"   身体动作维度: {len(action_body)}")

    state_hand_left = reader.get_state_hand(frame_idx, "left")
    state_hand_right = reader.get_state_hand(frame_idx, "right")
    if state_hand_left:
        print(f"   左手状态维度: {len(state_hand_left)}")
    if state_hand_right:
        print(f"   右手状态维度: {len(state_hand_right)}")


def example_partial_video():
    """生成部分视频示例"""
    print("\n" + "="*60)
    print("示例 5: 生成部分视频（指定帧范围）")
    print("="*60)

    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)

    # 只生成前300帧的视频
    start_frame = 0
    end_frame = 300

    print(f"\n生成第{start_frame}-{end_frame}帧的视频...")

    if reader.has_world_cam:
        reader.visualize_keypoints_on_world_cam(
            output_path=f"{output_dir}/world_partial_{start_frame}_{end_frame}.mp4",
            start_frame=start_frame,
            end_frame=end_frame,
            fps=30
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EpisodeReader使用示例")
    parser.add_argument(
        "--example",
        type=str,
        default="all",
        choices=["all", "basic", "videos", "keypoints", "data", "partial"],
        help="选择要运行的示例"
    )

    args = parser.parse_args()

    try:
        if args.example == "all":
            example_basic_usage()
            example_create_videos()
            example_visualize_keypoints()
            example_access_data()
            example_partial_video()
        elif args.example == "basic":
            example_basic_usage()
        elif args.example == "videos":
            example_create_videos()
        elif args.example == "keypoints":
            example_visualize_keypoints()
        elif args.example == "data":
            example_access_data()
        elif args.example == "partial":
            example_partial_video()

        print("\n" + "="*60)
        print("✓ 所有示例运行完成！")
        print("="*60)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
