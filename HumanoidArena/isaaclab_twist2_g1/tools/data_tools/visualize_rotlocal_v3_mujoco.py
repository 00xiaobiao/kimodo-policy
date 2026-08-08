#!/usr/bin/env python

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

_RESAMPLE_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")

_ISAACLAB_ROOT = Path(__file__).resolve().parents[2]
if str(_ISAACLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_ISAACLAB_ROOT))

from action_provider.vla_robot_current_local_runtime_v3 import UnifiedRobotCurrentLocalActionRuntimeV3
from action_provider.vla_smpl_runtime import (
    CANONICAL_G1_JOINT_NAMES_29,
    rot6d_to_quat_wxyz_with_layout,
)
from verify_lerobot_rotlocal_v3 import load_lerobot_fps


_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XML_CANDIDATES = [
    _WORKSPACE_ROOT / "TWIST2" / "assets" / "g1" / "g1_sim2sim_29dof.xml",
    _WORKSPACE_ROOT / "TWIST2" / "assets" / "g1" / "g1_mocap_29dof.xml",
    _WORKSPACE_ROOT / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
]
FRONT_IMAGE_KEY = "observation.images.front"
CANVAS_HEIGHT = 480
SOURCE_WIDTH = 640
MUJOCO_WIDTH = 640
ACTION_PANEL_WIDTH = 512
ACTION_SHORT_NAMES = [
    "heading_dx",
    "heading_dy",
    "root_z",
    *[f"rel6d_{idx}" for idx in range(6)],
    *[f"j{idx:02d}" for idx in range(29)],
    "left_hand",
    "right_hand",
]


def resolve_xml(xml_path: Path | None) -> Path:
    if xml_path is not None:
        path = xml_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo XML not found: {path}")
        return path
    for candidate in DEFAULT_XML_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No default MuJoCo XML found. Pass --xml explicitly. Tried: "
        + ", ".join(str(path) for path in DEFAULT_XML_CANDIDATES)
    )


def build_joint_qpos_map(mujoco, model) -> dict[str, int]:
    joint_qpos: dict[str, int] = {}
    for name in CANONICAL_G1_JOINT_NAMES_29:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        joint_qpos[name] = int(model.jnt_qposadr[joint_id])
    return joint_qpos


def apply_frame_to_mujoco(mujoco, model, data, frame, joint_qpos: dict[str, int]) -> None:
    data.qpos[:] = 0.0
    if model.nq >= 7:
        data.qpos[0:3] = frame.body_pos_world
        data.qpos[3:7] = frame.target_root_quat_wxyz
    for joint_idx, name in enumerate(CANONICAL_G1_JOINT_NAMES_29):
        qpos_idx = joint_qpos.get(name)
        if qpos_idx is None or qpos_idx >= model.nq:
            continue
        data.qpos[qpos_idx] = frame.joint_pos_canonical_29[joint_idx]
    mujoco.mj_forward(model, data)


def _read_parquet_tables(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No parquet files found")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def load_episode_table(dataset_root: Path, episode_index: int) -> pd.DataFrame:
    data_paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    df = _read_parquet_tables(data_paths)
    required = {"episode_index", "frame_index", "timestamp", "observation.state", "action"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"LeRobot parquet missing columns: {missing}")
    ep_df = df[df["episode_index"] == int(episode_index)].sort_values("frame_index")
    if ep_df.empty:
        available = sorted(int(value) for value in df["episode_index"].unique())
        raise KeyError(f"Episode {episode_index} not found. Available episodes: {available}")
    return ep_df.reset_index(drop=True)


def _scalar(value):
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0]
        return value
    return value


def load_episode_video_metadata(dataset_root: Path, episode_index: int) -> dict | None:
    episode_paths = sorted((dataset_root / "meta" / "episodes").glob("*/*.parquet"))
    if not episode_paths:
        return None
    episodes = _read_parquet_tables(episode_paths)
    if "episode_index" not in episodes:
        return None
    rows = episodes[episodes["episode_index"] == int(episode_index)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    prefix = f"videos/{FRONT_IMAGE_KEY}"
    required = [
        f"{prefix}/chunk_index",
        f"{prefix}/file_index",
        f"{prefix}/from_timestamp",
    ]
    if any(key not in row.index for key in required):
        return None
    chunk_idx = int(_scalar(row[f"{prefix}/chunk_index"]))
    file_idx = int(_scalar(row[f"{prefix}/file_index"]))
    video_path = dataset_root / "videos" / FRONT_IMAGE_KEY / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Episode video metadata points to missing file: {video_path}")
    return {
        "path": video_path,
        "from_timestamp": float(_scalar(row[f"{prefix}/from_timestamp"])),
    }


def resize_with_padding(image: Image.Image, size: tuple[int, int], background=(0, 0, 0)) -> Image.Image:
    image = image.convert("RGB")
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    new_w = max(int(round(image.width * scale)), 1)
    new_h = max(int(round(image.height * scale)), 1)
    resized = image.resize((new_w, new_h), _RESAMPLE_BILINEAR)
    canvas = Image.new("RGB", size, background)
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def _decode_image_cell(cell, dataset_root: Path) -> Image.Image | None:
    if isinstance(cell, dict):
        raw_bytes = cell.get("bytes")
        if raw_bytes:
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        image_path = cell.get("path")
        if image_path:
            path = Path(image_path)
            if not path.is_absolute():
                candidates = [
                    dataset_root / path,
                    dataset_root / "images" / FRONT_IMAGE_KEY / path,
                ]
                path = next((candidate for candidate in candidates if candidate.is_file()), path)
            if path.is_file():
                return Image.open(path).convert("RGB")
    if isinstance(cell, (str, Path)):
        path = Path(cell)
        if not path.is_absolute():
            path = dataset_root / path
        if path.is_file():
            return Image.open(path).convert("RGB")
    return None


class FrontFrameSource:
    def __init__(self, *, dataset_root: Path, episode_df: pd.DataFrame, episode_index: int, fps: int):
        self._dataset_root = dataset_root
        self._episode_df = episode_df
        self._fps = fps
        self._reader = None
        self._video_fps = float(fps)
        self._video_from_timestamp = 0.0

        if FRONT_IMAGE_KEY in episode_df.columns:
            self._mode = "embedded"
            return

        metadata = load_episode_video_metadata(dataset_root, episode_index)
        if metadata is None:
            self._mode = "placeholder"
            return

        import imageio.v2 as imageio

        self._mode = "video"
        self._reader = imageio.get_reader(str(metadata["path"]))
        meta = self._reader.get_meta_data()
        self._video_fps = float(meta.get("fps", fps) or fps)
        self._video_from_timestamp = float(metadata["from_timestamp"])

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()

    def get(self, row_index: int) -> Image.Image:
        if self._mode == "embedded":
            image = _decode_image_cell(self._episode_df.iloc[row_index][FRONT_IMAGE_KEY], self._dataset_root)
            if image is not None:
                return resize_with_padding(image, (SOURCE_WIDTH, CANVAS_HEIGHT))
        elif self._mode == "video" and self._reader is not None:
            timestamp = float(self._episode_df.iloc[row_index]["timestamp"])
            video_frame = int(round((self._video_from_timestamp + timestamp) * self._video_fps))
            try:
                image = Image.fromarray(self._reader.get_data(video_frame)).convert("RGB")
                return resize_with_padding(image, (SOURCE_WIDTH, CANVAS_HEIGHT))
            except Exception as exc:
                logging.warning("Failed to read source video frame %d: %s", video_frame, exc)

        canvas = Image.new("RGB", (SOURCE_WIDTH, CANVAS_HEIGHT), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 24), "No IsaacLab front image", fill=(230, 230, 230))
        return canvas


def render_action_panel(
    action: np.ndarray,
    *,
    frame_index: int,
    timestamp: float,
    scale: float,
) -> Image.Image:
    panel = Image.new("RGB", (ACTION_PANEL_WIDTH, CANVAS_HEIGHT), (248, 248, 246))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, ACTION_PANEL_WIDTH - 1, CANVAS_HEIGHT - 1), outline=(190, 190, 190))
    draw.text((12, 10), f"40D v3 action  frame={frame_index}  t={timestamp:.3f}s", fill=(15, 15, 15), font=font)
    draw.text((12, 26), f"bar scale +/-{scale:.3g}", fill=(90, 90, 90), font=font)

    top = 48
    row_h = 10
    center_x = 300
    max_bar_w = 150
    draw.line((center_x, top - 2, center_x, top + row_h * 40 + 2), fill=(150, 150, 150))
    for dim, value in enumerate(np.asarray(action, dtype=np.float32).reshape(40)):
        y = top + dim * row_h
        value_f = float(value)
        clipped = float(np.clip(value_f / max(scale, 1e-6), -1.0, 1.0))
        bar_w = int(round(abs(clipped) * max_bar_w))
        color = (46, 125, 50) if value_f >= 0.0 else (198, 40, 40)
        rect = (center_x, y + 1, center_x + bar_w, y + row_h - 1)
        if value_f < 0.0:
            rect = (center_x - bar_w, y + 1, center_x, y + row_h - 1)
        draw.text((12, y), f"{dim:02d}", fill=(35, 35, 35), font=font)
        draw.text((44, y), ACTION_SHORT_NAMES[dim], fill=(35, 35, 35), font=font)
        draw.rectangle(rect, fill=color)
        draw.text((center_x + max_bar_w + 12, y), f"{value_f:+.3f}", fill=(35, 35, 35), font=font)
    return panel


def compose_visual_frame(
    *,
    source_image: Image.Image,
    mujoco_image: np.ndarray,
    action_panel: Image.Image,
) -> np.ndarray:
    middle = resize_with_padding(Image.fromarray(mujoco_image).convert("RGB"), (MUJOCO_WIDTH, CANVAS_HEIGHT))
    canvas = Image.new("RGB", (SOURCE_WIDTH + MUJOCO_WIDTH + ACTION_PANEL_WIDTH, CANVAS_HEIGHT), (0, 0, 0))
    canvas.paste(source_image, (0, 0))
    canvas.paste(middle, (SOURCE_WIDTH, 0))
    canvas.paste(action_panel, (SOURCE_WIDTH + MUJOCO_WIDTH, 0))
    return np.asarray(canvas, dtype=np.uint8)


def render_frames(
    *,
    actions: np.ndarray,
    states: np.ndarray,
    episode_df: pd.DataFrame,
    front_source: FrontFrameSource,
    xml_path: Path,
    output_path: Path,
    fps: int,
    max_frames: int,
    stride: int,
    mujoco_gl: str | None,
    frames_dir: bool,
) -> None:
    import imageio.v2 as imageio

    if mujoco_gl:
        os.environ["MUJOCO_GL"] = mujoco_gl
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    joint_qpos = build_joint_qpos_map(mujoco, model)
    if len(joint_qpos) < 29:
        missing = sorted(set(CANONICAL_G1_JOINT_NAMES_29) - set(joint_qpos))
        logging.warning("MuJoCo XML is missing %d canonical joints: %s", len(missing), missing[:8])

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.distance = 3.0
    camera.azimuth = 135.0
    camera.elevation = -18.0

    runtime = UnifiedRobotCurrentLocalActionRuntimeV3(root_rot6d_layout="row")
    selected_indices = np.arange(actions.shape[0], dtype=np.int64)[:: max(int(stride), 1)][:max_frames]
    selected_set = set(int(idx) for idx in selected_indices)
    selected_actions = actions[selected_indices]
    abs_max = float(np.nanpercentile(np.abs(selected_actions), 98.0)) if selected_actions.size else 1.0
    action_scale = max(abs_max, 1.0)
    frames: list[np.ndarray] = []
    try:
        for row_index, action in enumerate(actions):
            current_robot_quat = rot6d_to_quat_wxyz_with_layout(states[row_index, :6], layout="row").reshape(4)
            frame = runtime.step(
                action,
                current_robot_quat_wxyz=current_robot_quat,
                current_robot_xy_world=np.zeros((2,), dtype=np.float32),
            )
            if row_index not in selected_set:
                continue
            camera.lookat[:] = frame.body_pos_world
            apply_frame_to_mujoco(mujoco, model, data, frame, joint_qpos)
            renderer.update_scene(data, camera=camera)
            source_image = front_source.get(int(row_index))
            action_panel = render_action_panel(
                action,
                frame_index=int(episode_df.iloc[int(row_index)]["frame_index"]),
                timestamp=float(episode_df.iloc[int(row_index)]["timestamp"]),
                scale=action_scale,
            )
            frames.append(
                compose_visual_frame(
                    source_image=source_image,
                    mujoco_image=renderer.render(),
                    action_panel=action_panel,
                )
            )
    finally:
        front_source.close()
        renderer.close()

    if not frames:
        raise ValueError("No frames were selected for rendering.")

    output_path = resolve_render_output_path(output_path, frames_dir=frames_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames_dir:
        imageio.mimsave(output_path, frames, fps=max(int(fps / max(int(stride), 1)), 1))
        return

    output_path.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        imageio.imwrite(output_path / f"frame_{idx:06d}.png", frame)


def resolve_render_output_path(output_path: Path, *, frames_dir: bool) -> Path:
    path = output_path.expanduser().resolve()
    if frames_dir:
        return path
    if path.suffix.lower() == ".mp4":
        return path
    if path.suffix == "":
        return path.with_suffix(".mp4")
    raise ValueError(
        f"Output must be an .mp4 file or a path without suffix, got: {path}. "
        "Use --frames-dir if you intentionally want a PNG frame directory."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline MuJoCo visualization for rotlocal v3 LeRobot data. "
            "It replays the converted 40D actions without starting Isaac."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--xml", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("rotlocal_v3_preview.mp4"))
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--frames-dir",
        action="store_true",
        help="Write individual PNG frames to --output instead of an mp4 video.",
    )
    parser.add_argument(
        "--mujoco-gl",
        type=str,
        default=None,
        help="MuJoCo GL backend override, e.g. egl or osmesa. Defaults to existing MUJOCO_GL or egl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset_root = args.dataset_root.expanduser().resolve()
    episode_df = load_episode_table(dataset_root, args.episode)
    fps = args.fps or load_lerobot_fps(dataset_root) or 50
    xml_path = resolve_xml(args.xml)
    output_path = resolve_render_output_path(args.output, frames_dir=args.frames_dir)
    actions = np.stack(episode_df["action"].to_numpy()).astype(np.float32)
    states = np.stack(episode_df["observation.state"].to_numpy()).astype(np.float32)
    front_source = FrontFrameSource(
        dataset_root=dataset_root,
        episode_df=episode_df,
        episode_index=args.episode,
        fps=fps,
    )

    logging.info(
        "Rendering v3 episode=%d frames=%d xml=%s output=%s",
        args.episode,
        min(args.max_frames, actions.shape[0]),
        xml_path,
        output_path,
    )
    render_frames(
        actions=actions,
        states=states,
        episode_df=episode_df,
        front_source=front_source,
        xml_path=xml_path,
        output_path=args.output,
        fps=fps,
        max_frames=args.max_frames,
        stride=args.stride,
        mujoco_gl=args.mujoco_gl,
        frames_dir=args.frames_dir,
    )
    logging.info("Wrote MuJoCo v3 visualization to %s", output_path)


if __name__ == "__main__":
    main()
