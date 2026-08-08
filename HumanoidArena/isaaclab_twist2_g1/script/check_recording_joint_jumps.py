import argparse
import math
import sys
from pathlib import Path

import numpy as np


DEFAULT_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

JOINT_KEY_CANDIDATES = [
    "robot_qpos_before_decimation",
    "robot_twist2_inference_qpos",
]


def normalize_input_path(path_str: str) -> Path:
    return Path(path_str.replace("\\", "/")).expanduser().resolve()


def resolve_npz_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix == ".npz" else []
    if input_path.is_dir():
        return sorted(input_path.rglob("*.npz"))
    return []


def pick_joint_key(npz_data: np.lib.npyio.NpzFile, specified_key: str | None) -> str:
    if specified_key:
        if specified_key not in npz_data:
            raise KeyError(f"指定关节键不存在: {specified_key}")
        return specified_key
    for key in JOINT_KEY_CANDIDATES:
        if key in npz_data:
            return key
    raise KeyError(f"未找到关节键，可用键: {sorted(npz_data.files)}")


def get_joint_names(joint_dim: int) -> list[str]:
    if joint_dim == len(DEFAULT_JOINT_NAMES):
        return DEFAULT_JOINT_NAMES
    return [f"joint_{i}" for i in range(joint_dim)]


def detect_joint_anomalies(
    qpos: np.ndarray,
    half_turn_threshold_rad: float,
    zero_wrap_threshold_rad: float,
) -> tuple[list[dict], list[dict]]:
    events = []
    summary = []
    joint_names = get_joint_names(qpos.shape[1])

    for joint_idx in range(qpos.shape[1]):
        series = qpos[:, joint_idx].astype(np.float64)
        raw_delta = np.diff(series)
        wrapped_delta = np.arctan2(np.sin(raw_delta), np.cos(raw_delta))

        over_half_turn = np.abs(raw_delta) > half_turn_threshold_rad
        zero_wrap_jump = over_half_turn & (np.abs(wrapped_delta) <= zero_wrap_threshold_rad)
        half_turn_jump = over_half_turn & (~zero_wrap_jump)

        half_turn_indices = np.where(half_turn_jump)[0]
        zero_wrap_indices = np.where(zero_wrap_jump)[0]

        for idx in half_turn_indices:
            events.append(
                {
                    "joint_idx": joint_idx,
                    "joint_name": joint_names[joint_idx],
                    "frame_from": int(idx),
                    "frame_to": int(idx + 1),
                    "kind": "over_half_turn",
                    "prev": float(series[idx]),
                    "curr": float(series[idx + 1]),
                    "raw_delta": float(raw_delta[idx]),
                    "wrapped_delta": float(wrapped_delta[idx]),
                }
            )
        for idx in zero_wrap_indices:
            events.append(
                {
                    "joint_idx": joint_idx,
                    "joint_name": joint_names[joint_idx],
                    "frame_from": int(idx),
                    "frame_to": int(idx + 1),
                    "kind": "zero_wrap_jump",
                    "prev": float(series[idx]),
                    "curr": float(series[idx + 1]),
                    "raw_delta": float(raw_delta[idx]),
                    "wrapped_delta": float(wrapped_delta[idx]),
                }
            )

        summary.append(
            {
                "joint_idx": joint_idx,
                "joint_name": joint_names[joint_idx],
                "over_half_turn_count": int(len(half_turn_indices)),
                "zero_wrap_jump_count": int(len(zero_wrap_indices)),
            }
        )

    events.sort(key=lambda item: (item["frame_from"], item["joint_idx"]))
    return events, summary


def print_file_report(
    npz_file: Path,
    joint_key: str,
    qpos_shape: tuple[int, ...],
    events: list[dict],
    summary: list[dict],
    max_events: int,
) -> int:
    total_half = sum(item["over_half_turn_count"] for item in summary)
    total_zero = sum(item["zero_wrap_jump_count"] for item in summary)
    total = total_half + total_zero

    print(f"\n=== 文件: {npz_file} ===")
    print(f"关节键: {joint_key}")
    print(f"数组形状: {qpos_shape}")
    print(f"异常统计: 总计={total}, 超过半圈={total_half}, 零点突变={total_zero}")

    if total == 0:
        print("结论: 未发现异常")
        return 0

    print("结论: 检测到异常")
    print("按关节统计(仅显示有异常的关节):")
    for item in summary:
        c1 = item["over_half_turn_count"]
        c2 = item["zero_wrap_jump_count"]
        if c1 == 0 and c2 == 0:
            continue
        print(
            f"  [{item['joint_idx']:02d}] {item['joint_name']}: "
            f"超过半圈={c1}, 零点突变={c2}"
        )

    print(f"异常明细(最多显示 {max_events} 条):")
    for event in events[:max_events]:
        print(
            f"  帧 {event['frame_from']}->{event['frame_to']}, "
            f"关节[{event['joint_idx']:02d}] {event['joint_name']}, "
            f"类型={event['kind']}, "
            f"prev={math.degrees(event['prev']):.2f}°, "
            f"curr={math.degrees(event['curr']):.2f}°, "
            f"raw={math.degrees(event['raw_delta']):.2f}°, "
            f"wrapped={math.degrees(event['wrapped_delta']):.2f}°"
        )

    if len(events) > max_events:
        print(f"  ... 还有 {len(events) - max_events} 条未显示")
    return total


def analyze_single_file(
    npz_file: Path,
    key: str | None,
    half_turn_threshold_rad: float,
    zero_wrap_threshold_rad: float,
    max_events: int,
) -> int:
    with np.load(npz_file, allow_pickle=True) as npz_data:
        joint_key = pick_joint_key(npz_data, key)
        qpos = np.asarray(npz_data[joint_key])
        if qpos.ndim != 2:
            raise ValueError(f"{joint_key} 不是二维数组，当前形状: {qpos.shape}")
        if qpos.shape[0] < 2:
            print(f"\n=== 文件: {npz_file} ===")
            print("帧数小于2，跳过")
            return 0
        events, summary = detect_joint_anomalies(
            qpos=qpos,
            half_turn_threshold_rad=half_turn_threshold_rad,
            zero_wrap_threshold_rad=zero_wrap_threshold_rad,
        )
        return print_file_report(
            npz_file=npz_file,
            joint_key=joint_key,
            qpos_shape=qpos.shape,
            events=events,
            summary=summary,
            max_events=max_events,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检测录制数据中机器人关节是否存在超过半圈或零点突变"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/0319",
        help="输入文件或目录，默认检查 recording_data/0319",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="指定关节键名，不填则自动在候选键中选择",
    )
    parser.add_argument(
        "--half-turn-deg",
        type=float,
        default=180.0,
        help="超过半圈阈值，单位度",
    )
    parser.add_argument(
        "--zero-wrap-deg",
        type=float,
        default=30.0,
        help="零点突变判定时的包裹后角度阈值，单位度",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=20,
        help="每个文件最多显示多少条异常明细",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="若检测到任何异常则返回非零退出码",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = normalize_input_path(args.input_path)
    npz_files = resolve_npz_files(input_path)
    if not npz_files:
        print(f"未找到 .npz 文件: {input_path}")
        return 1

    half_turn_threshold_rad = math.radians(args.half_turn_deg)
    zero_wrap_threshold_rad = math.radians(args.zero_wrap_deg)

    print(f"开始检测，共 {len(npz_files)} 个文件")
    print(f"输入路径: {input_path}")
    print(f"超过半圈阈值: {args.half_turn_deg:.2f}°")
    print(f"零点突变包裹阈值: {args.zero_wrap_deg:.2f}°")

    total_anomalies = 0
    failed_files = 0

    for npz_file in npz_files:
        try:
            total_anomalies += analyze_single_file(
                npz_file=npz_file,
                key=args.key,
                half_turn_threshold_rad=half_turn_threshold_rad,
                zero_wrap_threshold_rad=zero_wrap_threshold_rad,
                max_events=args.max_events,
            )
        except Exception as exc:
            failed_files += 1
            print(f"\n=== 文件: {npz_file} ===")
            print(f"处理失败: {exc}")

    print("\n=== 总结 ===")
    print(f"文件总数: {len(npz_files)}")
    print(f"失败文件: {failed_files}")
    print(f"异常总数: {total_anomalies}")

    if failed_files > 0:
        return 1
    if args.strict_exit and total_anomalies > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
