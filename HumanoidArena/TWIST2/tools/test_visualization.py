#!/usr/bin/env python3
"""
测试SMPLX和qpos可视化功能

用于检查数据的正确性
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from episode_reader import EpisodeReader


def test_smplx_visualization():
    """测试SMPLX可视化"""
    print("\n" + "="*60)
    print("测试 1: SMPLX骨架可视化")
    print("="*60)

    # 加载episode数据
    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    # 可视化第100帧的SMPLX骨架
    print("\n可视化第100帧的SMPLX骨架...")
    reader.visualize_smplx(
        idx=100,
        save_path="test_output/smplx_frame_100.png",
        show=False  # 不显示窗口，只保存
    )

    # 可视化第500帧
    print("可视化第500帧的SMPLX骨架...")
    reader.visualize_smplx(
        idx=500,
        save_path="test_output/smplx_frame_500.png",
        show=False
    )

    print("\n✅ SMPLX可视化测试完成")


def test_qpos_visualization():
    """测试qpos可视化"""
    print("\n" + "="*60)
    print("测试 2: 关节角度(qpos)可视化")
    print("="*60)

    # 加载episode数据
    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    # 可视化前300帧的qpos
    print("\n可视化前300帧的关节角度...")
    reader.visualize_qpos(
        start_frame=0,
        end_frame=300,
        save_path="test_output/qpos_0_300.png",
        show=False
    )

    # 可视化中间300帧
    print("可视化中间300帧的关节角度...")
    reader.visualize_qpos(
        start_frame=500,
        end_frame=800,
        save_path="test_output/qpos_500_800.png",
        show=False
    )

    print("\n✅ Qpos可视化测试完成")


def test_data_inspection():
    """检查数据的基本信息"""
    print("\n" + "="*60)
    print("测试 3: 数据检查")
    print("="*60)

    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    # 打印详细信息
    reader.print_info()

    # 检查SMPLX数据
    print("\n检查SMPLX数据...")
    smplx = reader.get_smplx_data(100)
    if smplx:
        print(f"  ✓ SMPLX关节数: {len(smplx)}")
        print(f"  ✓ 关节名称: {list(smplx.keys())[:5]}...")
    else:
        print("  ✗ 没有SMPLX数据")

    # 检查qpos数据
    print("\n检查qpos数据...")
    state_body = reader.get_state_body(100)
    if state_body:
        print(f"  ✓ Body关节数: {len(state_body)}")
        print(f"  ✓ 数值范围: [{min(state_body):.3f}, {max(state_body):.3f}]")
    else:
        print("  ✗ 没有body state数据")

    state_hand_left = reader.get_state_hand(100, 'left')
    if state_hand_left:
        print(f"  ✓ 左手关节数: {len(state_hand_left)}")
    else:
        print("  ✗ 没有左手数据")

    state_hand_right = reader.get_state_hand(100, 'right')
    if state_hand_right:
        print(f"  ✓ 右手关节数: {len(state_hand_right)}")
    else:
        print("  ✗ 没有右手数据")

    print("\n✅ 数据检查完成")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试SMPLX和qpos可视化功能")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "smplx", "qpos", "inspect"],
        help="选择要运行的测试"
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs("test_output", exist_ok=True)

    try:
        if args.test == "all":
            test_smplx_visualization()
            test_qpos_visualization()
            test_data_inspection()
        elif args.test == "smplx":
            test_smplx_visualization()
        elif args.test == "qpos":
            test_qpos_visualization()
        elif args.test == "inspect":
            test_data_inspection()

        print("\n" + "="*60)
        print("✅ 所有测试完成！")
        print("="*60)
        print("\n查看生成的图片：")
        print("  - test_output/smplx_frame_*.png")
        print("  - test_output/qpos_*.png")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
