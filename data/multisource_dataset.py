"""Unified G1 training dataset for Arena, UnifoLM, Everyday and HIW."""

from __future__ import annotations

import fnmatch
import json
import logging
import random
import time
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils import data

from motion.g1_reference import (
    CANONICAL_G1_JOINT_NAMES_29,
    HumanoidArenaActionDecoder,
    UNITREE_G1_JOINT_NAMES_29,
    resample_hand_binary,
    resample_motion,
    rot6d_row_to_matrix,
)
from motion.representation.kimodo_motionrep import KimodoMotionRep
from skeleton.definitions import G1Skeleton34
from utils.geometry import quaternion_to_matrix


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_ROOT = PROJECT_ROOT.parent / "checkpoints"
XML_PATH = PROJECT_ROOT / "skeleton/assets/g1skel34/xml/g1.xml"
STATS_PATH = CHECKPOINTS_ROOT / "Kimodo-G1-RP-v1/stats/motion"

SOURCE_HUMANOID_ARENA = "HumanoidArena"
SOURCE_HUMANOID_EVERYDAY = "HumanoidEveryday"
SOURCE_HIW500 = "HIW500"
SOURCE_UNIFOLM = "UnifoLM_WBT_Dataset"
KNOWN_SOURCES = (
    SOURCE_HUMANOID_ARENA,
    SOURCE_HUMANOID_EVERYDAY,
    SOURCE_HIW500,
    SOURCE_UNIFOLM,
)

ARENA_EXPECTED_SCHEMA = "unitree_g1_gmt_refpose_v3_1"
ARENA_TASK_KEY_BY_TASK_ID = {
    "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody": "HOI_double_desk",
    "Isaac-Move-Football-Single-G129-Dex3-Wholebody": "HOI_football",
    "Isaac-Move-ArtVIP-Livingroom-GrapCup-G129-Dex3-Wholebody": "HOI_grap_cup",
    "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby": "HOI_pp_box",
    "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody": "HSI_boxing",
    "Isaac-Move-Open-Door-G129-Dex3-Wholebody": "HSI_open_door",
    "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody": "HSI_sit_sofa",
    "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody": "HSI_vision_navi",
}
ARENA_TASK_INDEX_BY_TASK_ID = {
    task_id: index for index, task_id in enumerate(ARENA_TASK_KEY_BY_TASK_ID)
}
ARENA_TASK_NAMES = frozenset(ARENA_TASK_KEY_BY_TASK_ID.values())
ARENA_MERGED_DATASETS = frozenset(
    {
        "all_16_refpose_v3_1",
        "sonic_8_refpose_v3_1",
        "twist2_8_refpose_v3_1",
    }
)
# Keep the published merged dataset names for compatibility, but exclude tasks
# that are not part of the official evaluation protocol from merged training.
# Direct single-task selection remains available for inspection and ablations.
ARENA_EXCLUDED_MERGED_TRAIN_TASKS = frozenset({"HOI_grap_cup"})

HIW_G1_JOINT_FEATURE_NAMES_29 = (
    "kLeftHipPitch.q",
    "kLeftHipRoll.q",
    "kLeftHipYaw.q",
    "kLeftKnee.q",
    "kLeftAnklePitch.q",
    "kLeftAnkleRoll.q",
    "kRightHipPitch.q",
    "kRightHipRoll.q",
    "kRightHipYaw.q",
    "kRightKnee.q",
    "kRightAnklePitch.q",
    "kRightAnkleRoll.q",
    "kWaistYaw.q",
    "kWaistRoll.q",
    "kWaistPitch.q",
    "kLeftShoulderPitch.q",
    "kLeftShoulderRoll.q",
    "kLeftShoulderYaw.q",
    "kLeftElbow.q",
    "kLeftWristRoll.q",
    "kLeftWristPitch.q",
    "kLeftWristYaw.q",
    "kRightShoulderPitch.q",
    "kRightShoulderRoll.q",
    "kRightShoulderYaw.q",
    "kRightElbow.q",
    "kRightWristRoll.q",
    "kRightWristPitch.q",
    "kRightWristYaw.q",
)
HIW_WBC_FEATURE_NAMES_23 = (
    "pivot_vx",
    "pivot_vy",
    "pivot_vyaw",
    "pivot_roll",
    "pivot_pitch",
    "pivot_yaw",
    "pivot_height",
    "left_ee_x",
    "left_ee_y",
    "left_ee_z",
    "left_ee_roll",
    "left_ee_pitch",
    "left_ee_yaw",
    "right_ee_x",
    "right_ee_y",
    "right_ee_z",
    "right_ee_roll",
    "right_ee_pitch",
    "right_ee_yaw",
    "left_trigger",
    "left_squeeze",
    "right_trigger",
    "right_squeeze",
)
HIW_TRIGGER_CLOSE_THRESHOLD = 5.0
HIW_SQUEEZE_OPEN_THRESHOLD = 0.5

VIDEO_READ_MAX_ATTEMPTS = 4
VIDEO_READ_RETRY_DELAY_SECONDS = 0.05
DEFAULT_VIDEO_CACHE_SIZE = 32
PARQUET_FILE_CACHE_SIZE = 16
UNIFOLM_FIRST_ROOT_JUMP_THRESHOLD_METERS = 0.5
UNIFOLM_INTERNAL_ROOT_JUMP_THRESHOLD_METERS = 0.5
UNIFOLM_MAX_INITIAL_TRIM_FRAMES = 3
UNIFOLM_JOINT_JUMP_THRESHOLD_RADIANS = 0.5
UNIFOLM_ROOT_ROTATION_JUMP_THRESHOLD_DEGREES = 30.0
MAX_SAMPLE_ATTEMPTS = 32
_UINT64_MASK = (1 << 64) - 1
KIMODO_MOTION_DIM = 417
KIMODO_PLANAR_ROOT_FEATURE_INDICES = (0, 2)


class VideoFrameDecodeError(RuntimeError):
    """A video frame could not be decoded after the supported retries."""


def _canonicalize_kimodo_window_translation(
    gt_motion: torch.Tensor,
    condition_motion: torch.Tensor,
    condition_motion_mask: torch.Tensor,
    gt_mask: torch.Tensor,
) -> torch.Tensor:
    """Put the first valid target root x/z at the origin for one window.

    Official Kimodo clips preserve the world-axis heading but translate each
    clip so its first root lies above the planar origin.  The clean target and
    any valid state root constraints must use that same translation.  Invalid
    condition features and left-padding remain untouched.

    Returns:
        The detached two-dimensional ``(x, z)`` origin removed from the window.
    """
    if gt_mask.ndim != 1 or gt_mask.dtype != torch.bool:
        raise ValueError("gt_mask must be a 1D bool tensor")
    if condition_motion_mask.dtype != torch.bool:
        raise ValueError("condition_motion_mask must have bool dtype")
    expected_motion_shape = (gt_mask.shape[0], KIMODO_MOTION_DIM)
    if tuple(gt_motion.shape) != expected_motion_shape:
        raise ValueError(
            f"Expected gt_motion shape {expected_motion_shape}, got "
            f"{tuple(gt_motion.shape)}"
        )
    if tuple(condition_motion.shape) != expected_motion_shape:
        raise ValueError(
            f"Expected condition_motion shape {expected_motion_shape}, got "
            f"{tuple(condition_motion.shape)}"
        )
    if tuple(condition_motion_mask.shape) != expected_motion_shape:
        raise ValueError(
            f"Expected condition_motion_mask shape {expected_motion_shape}, got "
            f"{tuple(condition_motion_mask.shape)}"
        )
    valid_indices = torch.nonzero(gt_mask, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("Cannot canonicalize a window without a valid motion frame")
    first_valid_index = int(valid_indices[0].item())
    origin_xz = gt_motion[
        first_valid_index, list(KIMODO_PLANAR_ROOT_FEATURE_INDICES)
    ].detach().clone()
    if not torch.isfinite(origin_xz).all():
        raise ValueError("The first valid target root x/z contains NaN or Inf")

    # Translation changes only smooth_root_pos.x/z. All other Kimodo features
    # (including y, heading, velocities, rotations and contacts) stay intact.
    for planar_offset, feature_index in enumerate(
        KIMODO_PLANAR_ROOT_FEATURE_INDICES
    ):
        gt_motion[gt_mask, feature_index] = (
            gt_motion[gt_mask, feature_index] - origin_xz[planar_offset]
        )
        valid_condition = gt_mask & condition_motion_mask[:, feature_index]
        condition_motion[valid_condition, feature_index] = (
            condition_motion[valid_condition, feature_index]
            - origin_xz[planar_offset]
        )
    return origin_xz


@dataclass
class EpisodeRecord:
    source: str
    task_id: str
    task_name: str
    instruction: str
    episode_id: str
    data_path: Path
    source_length: int
    source_fps: float
    target_fps: float
    video_path: Path
    video_from_timestamp: float
    first_cut: int
    sample_count: int
    metadata: dict = field(default_factory=dict)

    @property
    def target_length(self) -> int:
        return int(
            round((self.source_length - 1) * self.target_fps / self.source_fps)
        ) + 1

    @property
    def cache_key(self) -> tuple[str, str, str, str, int, int]:
        """Uniquely identify an episode slice, including shared parquet files."""
        row_start = int(
            self.metadata.get(
                "row_start", self.metadata.get("dataset_from_index", 0)
            )
        )
        row_end = int(
            self.metadata.get(
                "row_end", self.metadata.get("dataset_to_index", self.source_length)
            )
        )
        return (
            self.source,
            self.task_id,
            self.episode_id,
            str(self.data_path),
            row_start,
            row_end,
        )


class ParquetEpisodeReader:
    """Read one episode slice without materializing a full multi-episode file."""

    def __init__(self) -> None:
        self._file_start_indices: dict[Path, int] = {}
        self._parquet_files: OrderedDict[Path, pq.ParquetFile] = OrderedDict()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_parquet_files"] = OrderedDict()
        return state

    @staticmethod
    def _close_parquet_file(parquet_file: pq.ParquetFile) -> None:
        close = getattr(parquet_file, "close", None)
        if close is not None:
            close()

    def close(self) -> None:
        while self._parquet_files:
            _, parquet_file = self._parquet_files.popitem(last=False)
            self._close_parquet_file(parquet_file)

    def _parquet_file(self, path: Path) -> pq.ParquetFile:
        parquet_file = self._parquet_files.pop(path, None)
        if parquet_file is None:
            parquet_file = pq.ParquetFile(path)
        self._parquet_files[path] = parquet_file
        while len(self._parquet_files) > PARQUET_FILE_CACHE_SIZE:
            _, evicted = self._parquet_files.popitem(last=False)
            self._close_parquet_file(evicted)
        return parquet_file

    def _file_start_index(self, path: Path) -> int:
        cached = self._file_start_indices.get(path)
        if cached is not None:
            return cached
        parquet_file = self._parquet_file(path)
        first_group = parquet_file.read_row_group(0, columns=["index"])
        if first_group.num_rows == 0:
            raise ValueError(f"Empty parquet file: {path}")
        start_index = int(first_group.column("index")[0].as_py())
        self._file_start_indices[path] = start_index
        return start_index

    def _read_row_range(
        self,
        path: Path,
        columns: list[str],
        row_start: int,
        row_end: int,
    ) -> dict:
        if row_start < 0 or row_end <= row_start:
            raise ValueError(
                f"Invalid parquet row range [{row_start}, {row_end}) for {path}"
            )
        parquet_file = self._parquet_file(path)
        total_rows = parquet_file.metadata.num_rows
        if row_end > total_rows:
            raise ValueError(
                f"Parquet row range [{row_start}, {row_end}) exceeds {total_rows} rows in {path}"
            )

        selected_groups = []
        selected_start = None
        cursor = 0
        for group_index in range(parquet_file.num_row_groups):
            group_rows = parquet_file.metadata.row_group(group_index).num_rows
            group_end = cursor + group_rows
            if group_end > row_start and cursor < row_end:
                if selected_start is None:
                    selected_start = cursor
                selected_groups.append(group_index)
            cursor = group_end
            if cursor >= row_end:
                break
        if not selected_groups or selected_start is None:
            raise RuntimeError(f"Could not resolve parquet row groups for {path}")
        table = parquet_file.read_row_groups(
            selected_groups,
            columns=columns,
            use_threads=False,
        )
        table = table.slice(row_start - selected_start, row_end - row_start)
        if table.num_rows != row_end - row_start:
            raise RuntimeError(
                f"Expected {row_end - row_start} rows from {path}, got {table.num_rows}"
            )
        return table.to_pydict()

    def read(self, episode: EpisodeRecord, columns: list[str]) -> dict:
        if "row_start" in episode.metadata:
            row_start = int(episode.metadata["row_start"])
            row_end = int(episode.metadata["row_end"])
        else:
            file_start = self._file_start_index(episode.data_path)
            row_start = int(episode.metadata["dataset_from_index"]) - file_start
            row_end = int(episode.metadata["dataset_to_index"]) - file_start
        result = self._read_row_range(
            episode.data_path, columns, row_start, row_end
        )
        if row_end - row_start != episode.source_length:
            raise ValueError(
                f"Episode {episode.episode_id} metadata length={episode.source_length}, "
                f"but parquet slice contains {row_end - row_start} rows"
            )
        return result


def _as_matrix(values, width: int, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        matrix = np.stack([np.asarray(value, dtype=np.float32) for value in values])
    if matrix.ndim != 2 or matrix.shape[1] != width:
        raise ValueError(f"Expected {name} shape (T, {width}), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return matrix


def _instruction(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def _humanoid_everyday_instruction(
    task_metadata: Mapping,
    episode_metadata: Mapping,
) -> str:
    """Prefer the task catalog because released episode instructions are stale."""
    for value in (
        task_metadata.get("description"),
        episode_metadata.get("instruction"),
        task_metadata.get("task"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return "unknown task"


def _patterns_from_selection(selection: Mapping | None) -> list[str]:
    if not selection:
        return ["*"]
    tasks = selection.get("tasks") if isinstance(selection, Mapping) else None
    if tasks is None and isinstance(selection, Mapping):
        option_keys = {
            "camera",
            "weight",
            "stereo_view",
            "hand_close_threshold",
            "hand_open_threshold",
            "hand_trigger_threshold",
            "hand_squeeze_threshold",
            "first_root_jump_threshold",
            "internal_root_jump_threshold",
            "max_initial_trim_frames",
            "joint_jump_threshold_radians",
            "root_rotation_jump_threshold_degrees",
        }
        tasks = [
            str(key)
            for key, enabled in selection.items()
            if key not in option_keys and bool(enabled)
        ]
    if tasks is None:
        return ["*"]
    if isinstance(tasks, str):
        return [tasks]
    return [str(task) for task in tasks]


def _matches_selection(selection: Mapping | None, *candidates: str) -> bool:
    patterns = _patterns_from_selection(selection)
    normalized_candidates = [str(candidate).strip() for candidate in candidates if candidate]
    return any(
        fnmatch.fnmatch(candidate, pattern)
        or fnmatch.fnmatch(candidate.lower(), pattern.lower())
        for pattern in patterns
        for candidate in normalized_candidates
    )


def _video_key(features: Mapping, preferred: tuple[str, ...]) -> str:
    keys = [
        key
        for key, spec in features.items()
        if key.startswith("observation.image")
        and isinstance(spec, Mapping)
        and spec.get("dtype") in {"video", "image"}
    ]
    for key in preferred:
        if key in keys:
            return key
    if not keys:
        raise ValueError("Dataset has no observation image/video feature")
    return sorted(keys)[0]


def _validate_named_feature(
    features: Mapping,
    feature_key: str,
    expected_names: tuple[str, ...],
) -> None:
    """Reject schemas whose dimensions would be silently decoded incorrectly."""
    feature = features.get(feature_key)
    if not isinstance(feature, Mapping):
        raise ValueError(f"Dataset has no {feature_key} feature specification")
    shape = tuple(feature.get("shape", ()))
    if shape != (len(expected_names),):
        raise ValueError(
            f"Expected {feature_key} shape {(len(expected_names),)}, got {shape}"
        )
    names = feature.get("names")
    if not isinstance(names, (list, tuple)) or len(names) != len(expected_names):
        raise ValueError(
            f"Expected {feature_key} to declare {len(expected_names)} ordered names"
        )

    # The released HIW metadata contains one capitalization typo
    # (kLeftWristyaw.q). Ignore punctuation/case, but never ignore reordering.
    def normalize(value: str) -> str:
        return "".join(
            character for character in str(value).lower() if character.isalnum()
        )
    normalized_names = tuple(normalize(name) for name in names)
    normalized_expected = tuple(normalize(name) for name in expected_names)
    if normalized_names != normalized_expected:
        mismatch = next(
            index
            for index, (actual, expected) in enumerate(
                zip(normalized_names, normalized_expected)
            )
            if actual != expected
        )
        raise ValueError(
            f"Unexpected {feature_key} field at index {mismatch}: "
            f"got {names[mismatch]!r}, expected {expected_names[mismatch]!r}"
        )


def _stereo_crop(
    feature: Mapping, selection: Mapping | None
) -> tuple[int, int, int, int] | None:
    """Return a single-eye crop for horizontally packed stereo video."""
    shape = tuple(feature.get("shape", ()))
    if len(shape) != 3:
        return None
    height, width, channels = map(int, shape)
    if channels != 3 or width < 2 * height:
        return None
    eye_width = width // 2
    stereo_view = str((selection or {}).get("stereo_view", "left")).lower()
    if stereo_view not in {"left", "right"}:
        raise ValueError("stereo_view must be 'left' or 'right'")
    x0 = 0 if stereo_view == "left" else width - eye_width
    return (x0, 0, x0 + eye_width, height)


def _resample_binary_mask(
    mask: torch.Tensor, source_fps: float, target_fps: float
) -> torch.Tensor:
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if abs(float(source_fps) - float(target_fps)) < 1e-6:
        return mask
    source_frames = mask.shape[0]
    target_frames = int(
        round((source_frames - 1) * float(target_fps) / float(source_fps))
    ) + 1
    target_times = torch.arange(target_frames, dtype=torch.float32) / float(
        target_fps
    )
    source_indices = (target_times * float(source_fps)).round().long()
    return mask[source_indices.clamp(max=source_frames - 1)]


def _binary_hysteresis(
    score: np.ndarray,
    close_threshold: float,
    open_threshold: float,
) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    if score.ndim != 2 or score.shape[1] != 2:
        raise ValueError(f"Expected hand closure score shape (T, 2), got {score.shape}")
    if open_threshold > close_threshold:
        raise ValueError("Hand open threshold must not exceed close threshold")
    output = np.zeros_like(score, dtype=np.float32)
    state = score[0] >= close_threshold
    output[0] = state
    for frame_index in range(1, score.shape[0]):
        state = np.where(
            state,
            score[frame_index] > open_threshold,
            score[frame_index] >= close_threshold,
        )
        output[frame_index] = state
    return output


def _known_hand_to_binary(
    values: np.ndarray,
    hand_type: str,
    selection: Mapping | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    hand_type = hand_type.lower()
    if hand_type == "dex1":
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(f"Expected Dex1 hand shape (T, 2), got {values.shape}")
        score = 1.0 - np.clip(values / 5.5, 0.0, 1.0)
        default_close, default_open = 0.55, 0.45
    elif hand_type in {"inspire", "brainco"}:
        if values.ndim != 2 or values.shape[1] != 12:
            raise ValueError(
                f"Expected {hand_type} hand shape (T, 12), got {values.shape}"
            )
        if hand_type == "inspire":
            closure_indices = (0, 1, 2, 3, 4)
            default_close, default_open = 0.60, 0.50
        else:
            closure_indices = (0, 2, 3, 4, 5)
            default_close, default_open = 0.45, 0.35
        left = values[:, closure_indices].mean(axis=1)
        right = values[:, [index + 6 for index in closure_indices]].mean(axis=1)
        score = np.stack((left, right), axis=1)
    else:
        raise ValueError(f"Unsupported known hand type: {hand_type}")
    close_threshold = float(
        (selection or {}).get("hand_close_threshold", default_close)
    )
    open_threshold = float(
        (selection or {}).get("hand_open_threshold", default_open)
    )
    binary = _binary_hysteresis(score, close_threshold, open_threshold)
    valid = np.isfinite(score)
    return binary, valid


def _dex3_pair_to_binary(
    observed: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = _as_matrix(observed, 14, "Dex3 observed hand")
    target = _as_matrix(target, 14, "Dex3 target hand")
    combined = np.concatenate((observed, target), axis=0)
    observed_binary = np.zeros((observed.shape[0], 2), dtype=np.float32)
    target_binary = np.zeros((target.shape[0], 2), dtype=np.float32)
    observed_valid = np.zeros_like(observed_binary, dtype=bool)
    target_valid = np.zeros_like(target_binary, dtype=bool)

    # Thumb rotation is excluded. Larger absolute flexion on the remaining six
    # motors corresponds to a more closed Dex3 hand, independent of side signs.
    for side in range(2):
        side_slice = slice(side * 7 + 1, side * 7 + 7)
        calibration = np.abs(combined[:, side_slice])
        low = np.quantile(calibration, 0.10, axis=0)
        high = np.quantile(calibration, 0.90, axis=0)
        span = high - low
        active = span >= 0.03
        if active.sum() < 2:
            continue

        def score(sequence: np.ndarray) -> np.ndarray:
            flexion = np.abs(sequence[:, side_slice])[:, active]
            normalized = (flexion - low[active]) / span[active]
            return np.clip(normalized, 0.0, 1.0).mean(axis=1)

        observed_score = score(observed)
        target_score = score(target)
        observed_binary[:, side] = _binary_hysteresis(
            np.stack((observed_score, observed_score), axis=1), 0.60, 0.40
        )[:, 0]
        target_binary[:, side] = _binary_hysteresis(
            np.stack((target_score, target_score), axis=1), 0.60, 0.40
        )[:, 0]
        observed_valid[:, side] = True
        target_valid[:, side] = True
    return observed_binary, target_binary, observed_valid, target_valid


def _root_rotation_from_rpy(rpy: np.ndarray) -> torch.Tensor:
    rpy = torch.as_tensor(rpy, dtype=torch.float32)
    roll, pitch, yaw = rpy.unbind(dim=-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    zero = torch.zeros_like(roll)
    one = torch.ones_like(roll)
    rotation_x = torch.stack(
        (one, zero, zero, zero, cr, -sr, zero, sr, cr), dim=-1
    ).reshape(-1, 3, 3)
    rotation_y = torch.stack(
        (cp, zero, sp, zero, one, zero, -sp, zero, cp), dim=-1
    ).reshape(-1, 3, 3)
    rotation_z = torch.stack(
        (cy, -sy, zero, sy, cy, zero, zero, zero, one), dim=-1
    ).reshape(-1, 3, 3)
    return rotation_z @ rotation_y @ rotation_x


def _episode_local_root_from_odometry(
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
) -> tuple[np.ndarray, torch.Tensor]:
    """Express z-up odometry in the episode's initial-heading frame.

    HumanoidEveryday stores map/odometry-frame positions and ``(w, x, y, z)``
    quaternions.  Kimodo clips should not inherit that arbitrary map heading.
    Remove the first frame's planar translation and yaw while preserving the
    measured root height and roll/pitch.  This is an SE(2), rather than a full
    SE(3), rebase so gravity remains aligned with the vertical axis.
    """
    root_position = _as_matrix(root_position, 3, "odometry root position")
    root_quaternion = _as_matrix(root_quaternion, 4, "odometry root quaternion")
    if root_position.shape[0] != root_quaternion.shape[0]:
        raise ValueError(
            "Odometry position/quaternion lengths differ: "
            f"{root_position.shape[0]} and {root_quaternion.shape[0]}"
        )

    quaternions = torch.from_numpy(root_quaternion)
    quaternion_norm = quaternions.norm(dim=-1, keepdim=True)
    if (quaternion_norm < 1e-6).any():
        raise ValueError("Odometry root quaternion has near-zero norm")
    root_rotations = quaternion_to_matrix(quaternions / quaternion_norm)
    first_yaw = torch.atan2(root_rotations[0, 1, 0], root_rotations[0, 0, 0])
    inverse_heading = _root_rotation_from_rpy(
        np.asarray([[0.0, 0.0, -float(first_yaw)]], dtype=np.float32)
    )[0]

    local_position = torch.from_numpy(root_position.copy())
    planar_delta = local_position[:, :2] - local_position[0, :2]
    local_position[:, :2] = torch.einsum(
        "ij,tj->ti", inverse_heading[:2, :2], planar_delta
    )
    local_rotations = torch.einsum(
        "ij,tjk->tik", inverse_heading, root_rotations
    )
    return local_position.numpy(), local_rotations


def _hiw_root_from_wbc(wbc: np.ndarray, fps: float) -> tuple[np.ndarray, torch.Tensor]:
    """Construct the best episode-local root proxy available in HIW LeRobot.

    HIW's first seven WBC values are commands, not an odometry pose:
    ``(base_vx, base_vy, base_vyaw, torso_roll, torso_pitch, torso_yaw,
    base_height)``.  In particular, the torso RPY command must not be decoded
    as the pelvis orientation.  The LeRobot export does not contain the raw
    MCAP odometry/IMU fields, so integrate the commanded body-frame SE(2)
    twist from an identity episode heading and retain the commanded height.

    This is a physically consistent command-trajectory proxy, not measured
    root-pose ground truth.  Accumulation is done in float64 to limit drift on
    long episodes and returned in the Unitree/MuJoCo xyz convention.
    """
    wbc = _as_matrix(wbc, 23, "HIW WBC state")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"HIW fps must be positive and finite, got {fps}")

    positions = np.zeros((wbc.shape[0], 3), dtype=np.float64)
    headings = np.zeros(wbc.shape[0], dtype=np.float64)
    positions[:, 2] = wbc[:, 6]
    dt = 1.0 / fps
    for frame_index in range(1, wbc.shape[0]):
        vx, vy, yaw_rate = map(float, wbc[frame_index - 1, :3])
        previous_heading = headings[frame_index - 1]
        delta_heading = yaw_rate * dt

        # Exact SE(2) integration for a piecewise-constant body twist.  The
        # small-rate branch avoids cancellation in (1 - cos(delta)) / omega.
        if abs(yaw_rate) < 1e-8:
            local_dx = vx * dt
            local_dy = vy * dt
        else:
            sin_scale = np.sin(delta_heading) / yaw_rate
            cos_scale = (1.0 - np.cos(delta_heading)) / yaw_rate
            local_dx = sin_scale * vx - cos_scale * vy
            local_dy = cos_scale * vx + sin_scale * vy

        cosine = np.cos(previous_heading)
        sine = np.sin(previous_heading)
        positions[frame_index, 0] = (
            positions[frame_index - 1, 0]
            + cosine * local_dx
            - sine * local_dy
        )
        positions[frame_index, 1] = (
            positions[frame_index - 1, 1]
            + sine * local_dx
            + cosine * local_dy
        )
        headings[frame_index] = previous_heading + delta_heading

    root_rpy = np.zeros((wbc.shape[0], 3), dtype=np.float32)
    wrapped_headings = (headings + np.pi) % (2.0 * np.pi) - np.pi
    root_rpy[:, 2] = wrapped_headings.astype(np.float32)
    return positions.astype(np.float32), _root_rotation_from_rpy(root_rpy)


def _hiw_hand_from_events(
    wbc: np.ndarray,
    *,
    trigger_threshold: float = HIW_TRIGGER_CLOSE_THRESHOLD,
    squeeze_threshold: float = HIW_SQUEEZE_OPEN_THRESHOLD,
) -> np.ndarray:
    """Decode HIW's stateful trigger/squeeze events into closed=1 states.

    Each hand starts open.  A trigger falling pulse closes it, a squeeze rising
    pulse opens it, and the state is held between events.  HIW's WBC state
    echoes these commands rather than reporting an independent finger pose.
    """
    wbc = _as_matrix(wbc, 23, "HIW WBC state")
    trigger_threshold = float(trigger_threshold)
    squeeze_threshold = float(squeeze_threshold)
    if not np.isfinite(trigger_threshold) or not np.isfinite(squeeze_threshold):
        raise ValueError("HIW hand thresholds must be finite")

    output = np.zeros((wbc.shape[0], 2), dtype=np.float32)
    for side, (trigger_index, squeeze_index) in enumerate(((19, 20), (21, 22))):
        trigger_active = wbc[:, trigger_index] < trigger_threshold
        squeeze_active = wbc[:, squeeze_index] > squeeze_threshold
        closed = False
        for frame_index in range(1, wbc.shape[0]):
            close_event = (
                trigger_active[frame_index]
                and not trigger_active[frame_index - 1]
            )
            open_event = (
                squeeze_active[frame_index]
                and not squeeze_active[frame_index - 1]
            )
            if close_event:
                closed = True
            if open_event:
                closed = False
            output[frame_index, side] = float(closed)
    return output


def _hiw_joint_discontinuity(
    joint_q: np.ndarray,
    *,
    joint_jump_threshold: float = UNIFOLM_JOINT_JUMP_THRESHOLD_RADIANS,
    max_initial_trim_frames: int = UNIFOLM_MAX_INITIAL_TRIM_FRAMES,
) -> tuple[int, dict[str, float | int] | None]:
    """Trim recorder startup jumps and reject internal HIW joint resets."""
    joint_q = _as_matrix(joint_q, 29, "HIW joint state")
    joint_jump_threshold = float(joint_jump_threshold)
    max_initial_trim_frames = int(max_initial_trim_frames)
    if joint_jump_threshold <= 0 or not np.isfinite(joint_jump_threshold):
        raise ValueError("HIW joint jump threshold must be positive and finite")
    if max_initial_trim_frames < 0:
        raise ValueError("max_initial_trim_frames must be non-negative")
    if joint_q.shape[0] <= 1:
        return 0, None

    joint_jumps = np.abs(np.diff(joint_q, axis=0)).max(axis=1)
    bad_transition = joint_jumps > joint_jump_threshold
    frame_offset = 0
    while (
        frame_offset < min(max_initial_trim_frames, bad_transition.shape[0])
        and bad_transition[frame_offset]
    ):
        frame_offset += 1
    remaining = np.flatnonzero(bad_transition[frame_offset:])
    if remaining.size == 0:
        return frame_offset, None
    transition_frame = frame_offset + int(remaining[0])
    return frame_offset, {
        "transition_frame": transition_frame,
        "joint_jump": float(joint_jumps[transition_frame]),
    }


def _unifolm_root_discontinuity(
    current: np.ndarray,
    desired: np.ndarray,
    *,
    first_jump_threshold: float = UNIFOLM_FIRST_ROOT_JUMP_THRESHOLD_METERS,
    internal_jump_threshold: float = UNIFOLM_INTERNAL_ROOT_JUMP_THRESHOLD_METERS,
    max_initial_trim_frames: int = UNIFOLM_MAX_INITIAL_TRIM_FRAMES,
) -> tuple[int, float, int | None]:
    """Find reset-like root jumps without mutating the source trajectory."""
    current = _as_matrix(current, 36, "UnifoLM current q")
    desired = _as_matrix(desired, 36, "UnifoLM desired q")
    if current.shape[0] != desired.shape[0]:
        raise ValueError(
            f"UnifoLM current/desired lengths differ: {current.shape[0]} and {desired.shape[0]}"
        )
    first_jump_threshold = float(first_jump_threshold)
    internal_jump_threshold = float(internal_jump_threshold)
    max_initial_trim_frames = int(max_initial_trim_frames)
    if first_jump_threshold <= 0 or internal_jump_threshold <= 0:
        raise ValueError("UnifoLM root jump thresholds must be positive")
    if max_initial_trim_frames < 0:
        raise ValueError("max_initial_trim_frames must be non-negative")
    if current.shape[0] <= 1:
        return 0, 0.0, None

    current_jumps = np.linalg.norm(np.diff(current[:, :3], axis=0), axis=1)
    desired_jumps = np.linalg.norm(np.diff(desired[:, :3], axis=0), axis=1)
    jumps = np.maximum(current_jumps, desired_jumps)
    frame_offset = 0
    while (
        frame_offset < min(max_initial_trim_frames, jumps.shape[0])
        and jumps[frame_offset] > first_jump_threshold
    ):
        frame_offset += 1

    remaining_jumps = jumps[frame_offset:]
    if remaining_jumps.size == 0:
        return frame_offset, 0.0, None
    internal_index = int(np.argmax(remaining_jumps))
    internal_max = float(remaining_jumps[internal_index])
    if internal_max <= internal_jump_threshold:
        return frame_offset, internal_max, None
    # Return the original source-frame index at the start of the bad transition.
    return frame_offset, internal_max, frame_offset + internal_index


def _unifolm_motion_discontinuity(
    current: np.ndarray,
    desired: np.ndarray,
    *,
    first_root_jump_threshold: float = UNIFOLM_FIRST_ROOT_JUMP_THRESHOLD_METERS,
    root_jump_threshold: float = UNIFOLM_INTERNAL_ROOT_JUMP_THRESHOLD_METERS,
    joint_jump_threshold: float = UNIFOLM_JOINT_JUMP_THRESHOLD_RADIANS,
    root_rotation_jump_threshold_degrees: float = (
        UNIFOLM_ROOT_ROTATION_JUMP_THRESHOLD_DEGREES
    ),
    max_initial_trim_frames: int = UNIFOLM_MAX_INITIAL_TRIM_FRAMES,
) -> tuple[int, dict[str, float | int] | None]:
    """Detect reset-like discontinuities across every UnifoLM pose component.

    Consecutive bad transitions at the start are treated as recorder reset
    frames and logically trimmed.  A bad transition after that point makes the
    episode unsafe for fixed-window supervision and is returned as an issue.
    """
    current = _as_matrix(current, 36, "UnifoLM current q")
    desired = _as_matrix(desired, 36, "UnifoLM desired q")
    if current.shape[0] != desired.shape[0]:
        raise ValueError(
            f"UnifoLM current/desired lengths differ: {current.shape[0]} and "
            f"{desired.shape[0]}"
        )
    first_root_jump_threshold = float(first_root_jump_threshold)
    root_jump_threshold = float(root_jump_threshold)
    joint_jump_threshold = float(joint_jump_threshold)
    root_rotation_jump_threshold_degrees = float(
        root_rotation_jump_threshold_degrees
    )
    max_initial_trim_frames = int(max_initial_trim_frames)
    if min(
        root_jump_threshold,
        first_root_jump_threshold,
        joint_jump_threshold,
        root_rotation_jump_threshold_degrees,
    ) <= 0:
        raise ValueError("UnifoLM motion jump thresholds must be positive")
    if max_initial_trim_frames < 0:
        raise ValueError("max_initial_trim_frames must be non-negative")
    if current.shape[0] <= 1:
        return 0, None

    root_jumps = np.maximum(
        np.linalg.norm(np.diff(current[:, :3], axis=0), axis=1),
        np.linalg.norm(np.diff(desired[:, :3], axis=0), axis=1),
    )
    joint_jumps = np.maximum(
        np.abs(np.diff(current[:, 7:], axis=0)).max(axis=1),
        np.abs(np.diff(desired[:, 7:], axis=0)).max(axis=1),
    )

    quaternion_norms = np.stack(
        (
            np.linalg.norm(current[:, 3:7], axis=1),
            np.linalg.norm(desired[:, 3:7], axis=1),
        ),
        axis=0,
    )
    if (quaternion_norms < 1e-6).any():
        bad_frame = int(np.argwhere(quaternion_norms < 1e-6)[0, 1])
        return 0, {
            "transition_frame": max(0, bad_frame - 1),
            "root_jump": 0.0,
            "joint_jump": 0.0,
            "root_rotation_jump_degrees": float("inf"),
        }

    rotation_jumps = []
    for sequence in (current[:, 3:7], desired[:, 3:7]):
        normalized = sequence / np.linalg.norm(sequence, axis=1, keepdims=True)
        dot = np.abs(np.sum(normalized[1:] * normalized[:-1], axis=1))
        rotation_jumps.append(
            np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))
        )
    root_rotation_jumps = np.maximum(*rotation_jumps)
    non_root_bad_transition = (
        (joint_jumps > joint_jump_threshold)
        | (root_rotation_jumps > root_rotation_jump_threshold_degrees)
    )
    bad_transition = (
        (root_jumps > root_jump_threshold)
        | non_root_bad_transition
    )
    initial_bad_transition = (
        (root_jumps > first_root_jump_threshold)
        | non_root_bad_transition
    )

    frame_offset = 0
    while (
        frame_offset < min(max_initial_trim_frames, bad_transition.shape[0])
        and initial_bad_transition[frame_offset]
    ):
        frame_offset += 1

    remaining = np.flatnonzero(bad_transition[frame_offset:])
    if remaining.size == 0:
        return frame_offset, None
    transition_frame = frame_offset + int(remaining[0])
    return frame_offset, {
        "transition_frame": transition_frame,
        "root_jump": float(root_jumps[transition_frame]),
        "joint_jump": float(joint_jumps[transition_frame]),
        "root_rotation_jump_degrees": float(
            root_rotation_jumps[transition_frame]
        ),
    }


class BaseSourceAdapter:
    source_name: str

    def __init__(
        self,
        root: Path,
        selection: Mapping | None,
        target_fps: float,
        action_chunk: int,
    ) -> None:
        self.root = root
        self.selection = dict(selection or {})
        self.target_fps = float(target_fps)
        self.action_chunk = int(action_chunk)
        self.reader = ParquetEpisodeReader()
        self.episodes: list[EpisodeRecord] = []
        self._decoders: dict[float, HumanoidArenaActionDecoder] = {}
        self._representation: KimodoMotionRep | None = None
        self.discover()

    def _decoder(self, source_fps: float) -> HumanoidArenaActionDecoder:
        key = float(source_fps)
        decoder = self._decoders.get(key)
        if decoder is None:
            decoder = HumanoidArenaActionDecoder(G1Skeleton34(), XML_PATH, key)
            self._decoders[key] = decoder
        return decoder

    def _motion_representation(self) -> KimodoMotionRep:
        if self._representation is None:
            self._representation = KimodoMotionRep(
                skeleton=G1Skeleton34(),
                fps=self.target_fps,
                stats_path=str(STATS_PATH),
            )
        return self._representation

    def _motion_feature_mask(self, *feature_names: str) -> torch.Tensor:
        representation = self._motion_representation()
        mask = torch.zeros(representation.motion_rep_dim, dtype=torch.bool)
        for feature_name in feature_names:
            if feature_name not in representation.slice_dict:
                raise KeyError(f"Unknown Kimodo motion feature {feature_name!r}")
            mask[representation.slice_dict[feature_name]] = True
        return mask

    def _record(
        self,
        *,
        task_id: str,
        task_name: str,
        instruction: str,
        episode_id: str,
        data_path: Path,
        source_length: int,
        source_fps: float,
        video_path: Path,
        video_from_timestamp: float,
        metadata: dict,
    ) -> None:
        target_length = int(
            round((source_length - 1) * self.target_fps / float(source_fps))
        ) + 1
        last_cut = target_length - self.action_chunk
        if last_cut < 0:
            return
        if not data_path.is_file() or not video_path.is_file():
            logger.warning(
                "Skipping %s episode %s with missing data/video: %s %s",
                self.source_name,
                episode_id,
                data_path,
                video_path,
            )
            return
        sample_stride = int(metadata.pop("sample_stride", 1))
        sample_count = last_cut // sample_stride + 1
        metadata["sample_stride"] = sample_stride
        self.episodes.append(
            EpisodeRecord(
                source=self.source_name,
                task_id=task_id,
                task_name=task_name,
                instruction=instruction,
                episode_id=episode_id,
                data_path=data_path,
                source_length=int(source_length),
                source_fps=float(source_fps),
                target_fps=self.target_fps,
                video_path=video_path,
                video_from_timestamp=float(video_from_timestamp),
                first_cut=0,
                sample_count=sample_count,
                metadata=metadata,
            )
        )

    def _finalize_motion(
        self,
        episode: EpisodeRecord,
        observed_local_rot: torch.Tensor,
        observed_root: torch.Tensor,
        target_local_rot: torch.Tensor,
        target_root: torch.Tensor,
        observed_hand: np.ndarray | torch.Tensor,
        target_hand: np.ndarray | torch.Tensor,
        observed_hand_valid: np.ndarray | torch.Tensor,
        target_hand_valid: np.ndarray | torch.Tensor,
        observed_motion_valid: np.ndarray | torch.Tensor | None = None,
        target_motion_source: str = "action",
    ) -> dict[str, torch.Tensor | str]:
        observed_local_rot, observed_root = resample_motion(
            observed_local_rot,
            observed_root,
            episode.source_fps,
            episode.target_fps,
        )
        target_local_rot, target_root = resample_motion(
            target_local_rot,
            target_root,
            episode.source_fps,
            episode.target_fps,
        )
        observed_hand = resample_hand_binary(
            observed_hand, episode.source_fps, episode.target_fps
        )
        target_hand = resample_hand_binary(
            target_hand, episode.source_fps, episode.target_fps
        )
        observed_hand_valid = _resample_binary_mask(
            observed_hand_valid, episode.source_fps, episode.target_fps
        )
        target_hand_valid = _resample_binary_mask(
            target_hand_valid, episode.source_fps, episode.target_fps
        )

        representation = self._motion_representation()
        observed_motion = representation(
            observed_local_rot, observed_root, to_normalize=False
        ).cpu()
        target_motion = representation(
            target_local_rot, target_root, to_normalize=False
        ).cpu()
        expected_length = episode.target_length
        if observed_motion_valid is None:
            observed_motion_valid = torch.ones_like(
                observed_motion, dtype=torch.bool
            )
        else:
            observed_motion_valid = torch.as_tensor(
                observed_motion_valid, dtype=torch.bool
            )
            if observed_motion_valid.ndim == 1:
                if observed_motion_valid.shape != (observed_motion.shape[1],):
                    raise ValueError(
                        "Observed motion feature mask must have shape "
                        f"{(observed_motion.shape[1],)}, got "
                        f"{tuple(observed_motion_valid.shape)}"
                    )
                observed_motion_valid = observed_motion_valid.unsqueeze(0).expand(
                    observed_motion.shape[0], -1
                )
            elif observed_motion_valid.ndim == 2:
                observed_motion_valid = _resample_binary_mask(
                    observed_motion_valid,
                    episode.source_fps,
                    episode.target_fps,
                )
            else:
                raise ValueError(
                    "Observed motion feature mask must be one- or two-dimensional"
                )
        tensors = {
            "observed_motion": observed_motion,
            "observed_motion_valid": observed_motion_valid.cpu(),
            "target_motion": target_motion,
            "observed_hand": observed_hand.cpu(),
            "target_hand": target_hand.cpu(),
            "observed_hand_valid": observed_hand_valid.cpu(),
            "target_hand_valid": target_hand_valid.cpu(),
        }
        for name, tensor in tensors.items():
            if tensor.shape[0] != expected_length:
                raise RuntimeError(
                    f"{episode.source} episode {episode.episode_id}: {name} has "
                    f"{tensor.shape[0]} frames, expected {expected_length}"
                )
        if observed_motion.shape[1] != 417 or target_motion.shape[1] != 417:
            raise RuntimeError(
                f"Expected Kimodo motion dimension 417, got "
                f"{observed_motion.shape[1]} and {target_motion.shape[1]}"
            )
        if not torch.isfinite(observed_motion).all() or not torch.isfinite(target_motion).all():
            raise ValueError(
                f"{episode.source} episode {episode.episode_id} produced NaN/Inf motion"
            )
        tensors["target_motion_source"] = str(target_motion_source)
        return tensors

    def discover(self) -> None:
        raise NotImplementedError

    def load_episode(self, episode: EpisodeRecord) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class HumanoidArenaAdapter(BaseSourceAdapter):
    source_name = SOURCE_HUMANOID_ARENA

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        backend = str(backend).strip().lower()
        backend = {"twice2": "twist2", "twist": "twist2"}.get(backend, backend)
        if backend not in {"sonic", "twist2"}:
            raise ValueError(f"Unsupported HumanoidArena backend: {backend}")
        return backend

    @staticmethod
    def _skip_merged_training_task(task_name: str) -> bool:
        return task_name in ARENA_EXCLUDED_MERGED_TRAIN_TASKS

    @classmethod
    def _parse_selection(cls, selection: Mapping) -> dict[str, str]:
        selection = dict(selection or {})
        if not selection:
            raise ValueError(
                "dataset_selection.HumanoidArena must select exactly one task/backend "
                "or one merged dataset"
            )

        if "merged" in selection:
            if set(selection) != {"merged"}:
                raise ValueError(
                    "HumanoidArena 'merged' mode cannot be combined with task/backend options"
                )
            merged = str(selection["merged"]).strip()
            if merged not in ARENA_MERGED_DATASETS:
                raise ValueError(
                    f"Unsupported HumanoidArena merged dataset {merged!r}; expected one of "
                    f"{sorted(ARENA_MERGED_DATASETS)}"
                )
            return {"mode": "merged", "merged": merged}

        if "task" in selection or "backend" in selection:
            if set(selection) != {"task", "backend"}:
                raise ValueError(
                    "HumanoidArena task mode requires exactly 'task' and 'backend'"
                )
            task_name = str(selection["task"]).strip()
            backend = cls._normalize_backend(selection["backend"])
        else:
            # Keep old single-task configs such as {HOI_football: sonic} working.
            if len(selection) != 1:
                raise ValueError(
                    "HumanoidArena legacy task selection must contain exactly one task/backend pair"
                )
            task_name, backend = next(iter(selection.items()))
            task_name = str(task_name).strip()
            backend = cls._normalize_backend(backend)

        if task_name not in ARENA_TASK_NAMES:
            raise ValueError(
                f"Unsupported HumanoidArena task {task_name!r}; expected one of "
                f"{sorted(ARENA_TASK_NAMES)}"
            )
        return {"mode": "task", "task": task_name, "backend": backend}

    @classmethod
    def _merged_episode_sources(cls, manifest: Mapping) -> list[tuple[str, str]]:
        episode_sources: list[tuple[str, str]] = []
        sources = manifest.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("HumanoidArena merge_manifest.json has no sources")
        for source in sources:
            task_name = str(source.get("task_name", "")).strip()
            if task_name not in ARENA_TASK_NAMES:
                raise ValueError(
                    f"HumanoidArena merge manifest contains unknown task {task_name!r}"
                )
            dataset_name = str(source.get("dataset_name", "")).strip().lower()
            if dataset_name.startswith("sonic"):
                backend = "sonic"
            elif dataset_name.startswith("twist2"):
                backend = "twist2"
            else:
                raise ValueError(
                    f"Cannot determine HumanoidArena backend from merged source {dataset_name!r}"
                )
            count = int(source.get("total_episodes", 0))
            if count <= 0:
                raise ValueError(
                    f"HumanoidArena merged source {task_name}/{dataset_name} has invalid "
                    f"total_episodes={count}"
                )
            episode_sources.extend([(task_name, backend)] * count)
        expected = int(manifest.get("total_episodes", len(episode_sources)))
        if len(episode_sources) != expected:
            raise ValueError(
                "HumanoidArena merge manifest source episode counts do not match "
                f"total_episodes: {len(episode_sources)} != {expected}"
            )
        return episode_sources

    def discover(self) -> None:
        selected = self._parse_selection(self.selection)
        is_merged = selected["mode"] == "merged"
        if is_merged:
            task_root = (
                self.root
                / "HumanoidArena_merged_datasets_v3_1"
                / selected["merged"]
            )
            manifest_path = task_root / "merge_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"HumanoidArena merged dataset manifest does not exist: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            merged_episode_sources = self._merged_episode_sources(manifest)
            selected_task_name = None
            selected_backend = None
        else:
            selected_task_name = selected["task"]
            selected_backend = selected["backend"]
            task_root = (
                self.root
                / selected_task_name
                / f"{selected_backend}_refpose_v3_1"
            )
            merged_episode_sources = None

        info_path = task_root / "meta/info.json"
        if not info_path.is_file():
            raise FileNotFoundError(
                f"Selected HumanoidArena dataset does not exist: {task_root}"
            )
        info = json.loads(info_path.read_text(encoding="utf-8"))
        protocol = info.get("vla_protocol", {})
        schema = protocol.get("schema")
        if schema and schema != ARENA_EXPECTED_SCHEMA:
            raise ValueError(
                f"Unsupported HumanoidArena schema {schema!r} in {task_root}"
            )
        action_shape = tuple(info.get("features", {}).get("action", {}).get("shape", ()))
        if action_shape and action_shape != (40,):
            raise ValueError(
                f"Expected HumanoidArena action shape (40,), got {action_shape} in {task_root}"
            )
        video_key = _video_key(
            info.get("features", {}),
            ("observation.images.front", "observation.image"),
        )
        tasks_table = pq.read_table(task_root / "meta/tasks.parquet").to_pydict()
        task_text_by_index = {
            int(index): str(text).strip()
            for index, text in zip(tasks_table["task_index"], tasks_table["task"])
        }
        seen_episode_indices: set[int] = set()
        excluded_merged_episodes: dict[str, int] = defaultdict(int)
        for meta_path in sorted((task_root / "meta/episodes").rglob("*.parquet")):
            metadata = pq.read_table(meta_path).to_pydict()
            for row in range(len(metadata.get("episode_index", []))):
                episode_index = int(metadata["episode_index"][row])
                if episode_index in seen_episode_indices:
                    raise ValueError(
                        f"Duplicate HumanoidArena episode_index={episode_index} in {task_root}"
                    )
                seen_episode_indices.add(episode_index)
                raw_task_id = _instruction(metadata["tasks"][row])
                mapped_task_name = ARENA_TASK_KEY_BY_TASK_ID.get(raw_task_id)
                if mapped_task_name is None:
                    raise KeyError(
                        f"Cannot map HumanoidArena task {raw_task_id!r} in {task_root}"
                    )
                if is_merged:
                    if episode_index < 0 or episode_index >= len(merged_episode_sources):
                        raise IndexError(
                            f"HumanoidArena merged episode_index={episode_index} is outside "
                            f"manifest range [0, {len(merged_episode_sources)})"
                        )
                    episode_task_name, backend = merged_episode_sources[episode_index]
                    if episode_task_name != mapped_task_name:
                        raise ValueError(
                            f"HumanoidArena merged manifest maps episode {episode_index} to "
                            f"{episode_task_name}, but metadata contains {mapped_task_name}"
                        )
                else:
                    episode_task_name = selected_task_name
                    backend = selected_backend
                    if mapped_task_name != episode_task_name:
                        raise ValueError(
                            f"Selected HumanoidArena directory {task_root} contains task "
                            f"{mapped_task_name}, expected {episode_task_name}"
                        )

                if len(task_text_by_index) == 1:
                    instruction = next(iter(task_text_by_index.values()))
                else:
                    task_index = ARENA_TASK_INDEX_BY_TASK_ID.get(raw_task_id)
                    if task_index not in task_text_by_index:
                        raise KeyError(
                            f"Cannot map HumanoidArena task {raw_task_id!r} in {task_root}"
                        )
                    instruction = task_text_by_index[task_index]

                if is_merged and self._skip_merged_training_task(episode_task_name):
                    excluded_merged_episodes[episode_task_name] += 1
                    continue

                data_chunk = int(metadata["data/chunk_index"][row])
                data_file = int(metadata["data/file_index"][row])
                video_chunk = int(metadata[f"videos/{video_key}/chunk_index"][row])
                video_file = int(metadata[f"videos/{video_key}/file_index"][row])
                self._record(
                    task_id=f"{self.source_name}::{raw_task_id}",
                    task_name=episode_task_name,
                    instruction=instruction,
                    episode_id=f"{task_root.name}:{episode_index}",
                    data_path=task_root / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet",
                    source_length=int(metadata["length"][row]),
                    source_fps=float(info["fps"]),
                    video_path=task_root / "videos" / video_key / f"chunk-{video_chunk:03d}" / f"file-{video_file:03d}.mp4",
                    video_from_timestamp=float(metadata[f"videos/{video_key}/from_timestamp"][row]),
                    metadata={
                        "dataset_from_index": int(metadata["dataset_from_index"][row]),
                        "dataset_to_index": int(metadata["dataset_to_index"][row]),
                        "backend": backend,
                        "dataset_variant": task_root.name,
                    },
                )

        expected_episodes = int(info.get("total_episodes", len(seen_episode_indices)))
        if len(seen_episode_indices) != expected_episodes:
            raise ValueError(
                f"HumanoidArena metadata below {task_root} contains "
                f"{len(seen_episode_indices)} episodes, expected {expected_episodes}"
            )
        if excluded_merged_episodes:
            logger.info(
                "Excluded HumanoidArena merged training tasks from %s: %s",
                task_root.name,
                dict(sorted(excluded_merged_episodes.items())),
            )

    def load_episode(self, episode: EpisodeRecord) -> dict[str, torch.Tensor]:
        table = self.reader.read(episode, ["observation.state", "action"])
        state = _as_matrix(
            table["observation.state"], 64, "HumanoidArena observation state"
        )
        actions = _as_matrix(table["action"], 40, "HumanoidArena action")
        decoder = self._decoder(episode.source_fps)
        observed_root_rotations = rot6d_row_to_matrix(
            torch.from_numpy(state[:, :6])
        )
        observed = decoder.decode_joint_configuration_pose(
            state[:, 6:35],
            np.zeros((state.shape[0], 3), dtype=np.float32),
            root_rotation_matrices=observed_root_rotations,
            joint_names=CANONICAL_G1_JOINT_NAMES_29,
        )
        target = decoder.decode_action_pose(actions)
        target_hand = actions[:, 38:40]
        target_hand_valid = np.ones_like(target_hand, dtype=bool)
        # Arena's 64D observation does not expose the hand state.  Use the
        # commanded action at the same frame as the recurrent hand state, just
        # as target_motion supplies the action-aligned clean motion sequence.
        # MultiSourceG1Dataset then slices [history_start:cut), so these states
        # are frame-aligned with the 100-frame 417D history window.
        observed_hand = target_hand.copy()
        observed_hand_valid = target_hand_valid.copy()
        observed_motion_valid = self._motion_feature_mask(
            "global_root_heading",
            "global_rot_data",
        )
        return self._finalize_motion(
            episode,
            observed["local_rot_mats"],
            observed["root_positions"],
            target["local_rot_mats"],
            target["root_positions"],
            observed_hand,
            target_hand,
            observed_hand_valid,
            target_hand_valid,
            observed_motion_valid=observed_motion_valid,
            target_motion_source="action",
        )


class UnifoLMAdapter(BaseSourceAdapter):
    source_name = SOURCE_UNIFOLM

    @staticmethod
    def _hand_type(task_name: str) -> str:
        lowered = task_name.lower()
        if "dex1" in lowered:
            return "dex1"
        if "brainco" in lowered:
            return "brainco"
        if "inspire" in lowered or "dex5" in lowered:
            return "inspire"
        raise ValueError(f"Cannot infer UnifoLM hand type from {task_name}")

    def discover(self) -> None:
        for info_path in sorted(self.root.rglob("meta/info.json")):
            task_root = info_path.parent.parent
            task_name = str(task_root.relative_to(self.root))
            top_level_name = task_name.split("/", 1)[0]
            info = json.loads(info_path.read_text(encoding="utf-8"))
            features = info.get("features", {})
            if tuple(features.get("action.robot_q_desired", {}).get("shape", ())) != (36,):
                logger.warning("Skipping incompatible UnifoLM dataset %s", task_root)
                continue
            hand_shape = tuple(features.get("action.hand_cmd", {}).get("shape", ()))
            if hand_shape not in {(2,), (12,)}:
                logger.warning("Skipping UnifoLM dataset with hand shape %s: %s", hand_shape, task_root)
                continue
            configured_camera = self.selection.get("camera")
            preferred = tuple(
                key
                for key in (
                    configured_camera,
                    "observation.images.head_stereo_left",
                    "observation.images.cam_0",
                )
                if key
            )
            video_key = _video_key(features, preferred)
            for meta_path in sorted((task_root / "meta/episodes").rglob("*.parquet")):
                columns = [
                    "episode_index",
                    "tasks",
                    "length",
                    "data/chunk_index",
                    "data/file_index",
                    "dataset_from_index",
                    "dataset_to_index",
                    f"videos/{video_key}/chunk_index",
                    f"videos/{video_key}/file_index",
                    f"videos/{video_key}/from_timestamp",
                ]
                metadata = pq.read_table(meta_path, columns=columns).to_pydict()
                for row in range(len(metadata["episode_index"])):
                    instruction = _instruction(metadata["tasks"][row])
                    if not _matches_selection(
                        self.selection, task_name, top_level_name, instruction
                    ):
                        continue
                    data_chunk = int(metadata["data/chunk_index"][row])
                    data_file = int(metadata["data/file_index"][row])
                    video_chunk = int(metadata[f"videos/{video_key}/chunk_index"][row])
                    video_file = int(metadata[f"videos/{video_key}/file_index"][row])
                    episode_index = int(metadata["episode_index"][row])
                    self._record(
                        task_id=f"{self.source_name}::{task_name}::{instruction}",
                        task_name=task_name,
                        instruction=instruction,
                        episode_id=f"{task_name}:{episode_index}",
                        data_path=task_root / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet",
                        source_length=int(metadata["length"][row]),
                        source_fps=float(info["fps"]),
                        video_path=task_root / "videos" / video_key / f"chunk-{video_chunk:03d}" / f"file-{video_file:03d}.mp4",
                        video_from_timestamp=float(metadata[f"videos/{video_key}/from_timestamp"][row]),
                        metadata={
                            "dataset_from_index": int(metadata["dataset_from_index"][row]),
                            "dataset_to_index": int(metadata["dataset_to_index"][row]),
                            "hand_type": self._hand_type(task_name),
                        },
                    )

    def load_episode(self, episode: EpisodeRecord) -> dict[str, torch.Tensor]:
        columns = [
            "observation.state.robot_q_current",
            "action.robot_q_desired",
            "observation.state.hand_state",
            "action.hand_cmd",
        ]
        table = self.reader.read(episode, columns)
        current = _as_matrix(
            table["observation.state.robot_q_current"], 36, "UnifoLM current q"
        )
        desired = _as_matrix(
            table["action.robot_q_desired"], 36, "UnifoLM desired q"
        )
        hand_width = 2 if episode.metadata["hand_type"] == "dex1" else 12
        observed_hand_raw = _as_matrix(
            table["observation.state.hand_state"], hand_width, "UnifoLM hand state"
        )
        target_hand_raw = _as_matrix(
            table["action.hand_cmd"], hand_width, "UnifoLM hand command"
        )
        frame_offset, motion_issue = _unifolm_motion_discontinuity(
            current,
            desired,
            first_root_jump_threshold=self.selection.get(
                "first_root_jump_threshold",
                UNIFOLM_FIRST_ROOT_JUMP_THRESHOLD_METERS,
            ),
            root_jump_threshold=self.selection.get(
                "internal_root_jump_threshold",
                UNIFOLM_INTERNAL_ROOT_JUMP_THRESHOLD_METERS,
            ),
            joint_jump_threshold=self.selection.get(
                "joint_jump_threshold_radians",
                UNIFOLM_JOINT_JUMP_THRESHOLD_RADIANS,
            ),
            root_rotation_jump_threshold_degrees=self.selection.get(
                "root_rotation_jump_threshold_degrees",
                UNIFOLM_ROOT_ROTATION_JUMP_THRESHOLD_DEGREES,
            ),
            max_initial_trim_frames=self.selection.get(
                "max_initial_trim_frames", UNIFOLM_MAX_INITIAL_TRIM_FRAMES
            ),
        )
        if motion_issue is not None:
            transition_frame = int(motion_issue["transition_frame"])
            quality_issue = (
                "motion discontinuity at frames "
                f"{transition_frame}->{transition_frame + 1}: "
                f"root={motion_issue['root_jump']:.3f} m, "
                f"joint={motion_issue['joint_jump']:.3f} rad, "
                "root_rotation="
                f"{motion_issue['root_rotation_jump_degrees']:.1f} deg"
            )
            logger.warning(
                "Excluding UnifoLM episode %s: %s",
                episode.episode_id,
                quality_issue,
            )
            return {
                "skip_episode": True,
                "quality_issue": quality_issue,
            }
        if frame_offset:
            logger.debug(
                "Logically trimming %d reset frame(s) from UnifoLM episode %s",
                frame_offset,
                episode.episode_id,
            )
            current = current[frame_offset:]
            desired = desired[frame_offset:]
            observed_hand_raw = observed_hand_raw[frame_offset:]
            target_hand_raw = target_hand_raw[frame_offset:]
            episode_for_motion = replace(
                episode, source_length=episode.source_length - frame_offset
            )
        else:
            episode_for_motion = episode
        if episode_for_motion.target_length < self.action_chunk:
            return {
                "skip_episode": True,
                "quality_issue": "episode is too short after initial reset trimming",
            }
        observed_hand, observed_valid = _known_hand_to_binary(
            observed_hand_raw, episode.metadata["hand_type"], self.selection
        )
        target_hand, target_valid = _known_hand_to_binary(
            target_hand_raw, episode.metadata["hand_type"], self.selection
        )

        planar_origin = current[0, :2].copy()
        decoder = self._decoder(episode.source_fps)
        observed = decoder.decode_joint_configuration_pose(
            current[:, 7:],
            current[:, :3],
            root_quaternions=current[:, 3:7],
            joint_names=UNITREE_G1_JOINT_NAMES_29,
            planar_origin=planar_origin,
        )
        target = decoder.decode_joint_configuration_pose(
            desired[:, 7:],
            desired[:, :3],
            root_quaternions=desired[:, 3:7],
            joint_names=UNITREE_G1_JOINT_NAMES_29,
            planar_origin=planar_origin,
        )
        motion = self._finalize_motion(
            episode_for_motion,
            observed["local_rot_mats"],
            observed["root_positions"],
            target["local_rot_mats"],
            target["root_positions"],
            observed_hand,
            target_hand,
            observed_valid,
            target_valid,
            target_motion_source="action",
        )
        motion["frame_offset"] = int(
            round(frame_offset * episode.target_fps / episode.source_fps)
        )
        motion["source_frame_offset"] = frame_offset
        return motion


class HumanoidEverydayAdapter(BaseSourceAdapter):
    source_name = SOURCE_HUMANOID_EVERYDAY

    def discover(self) -> None:
        info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))
        source_fps = float(info["fps"])
        available_data_paths = set((self.root / "data").rglob("*.parquet"))
        available_video_paths = set((self.root / "videos").rglob("*.mp4"))
        missing_data = 0
        missing_video = 0
        task_catalog = {}
        with (self.root / "meta/tasks.jsonl").open("r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                task_catalog[int(entry["task_index"])] = entry
        with (self.root / "meta/episodes.jsonl").open("r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                if str(entry.get("robot_type", "")).lower() != "g1":
                    continue
                episode_index = int(entry["episode_index"])
                task_indices = entry.get("tasks", [])
                task_metadata = (
                    task_catalog.get(int(task_indices[0]), {})
                    if task_indices
                    else {}
                )
                task_name = str(task_metadata.get("task") or "unknown").strip()
                instruction = _humanoid_everyday_instruction(
                    task_metadata, entry
                )
                if not _matches_selection(self.selection, task_name, instruction):
                    continue
                episode_chunk = episode_index // int(info.get("chunks_size", 1000))
                data_path = self.root / info["data_path"].format(
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                )
                video_path = self.root / info["video_path"].format(
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                )
                if data_path not in available_data_paths:
                    missing_data += 1
                    continue
                if video_path not in available_video_paths:
                    missing_video += 1
                    continue
                self._record(
                    task_id=f"{self.source_name}::{task_name}",
                    task_name=task_name,
                    instruction=instruction,
                    episode_id=str(episode_index),
                    data_path=data_path,
                    source_length=int(entry["length"]),
                    source_fps=source_fps,
                    video_path=video_path,
                    video_from_timestamp=0.0,
                    metadata={"row_start": 0, "row_end": int(entry["length"])},
                )
        if missing_data or missing_video:
            logger.info(
                "HumanoidEveryday subset discovery skipped unavailable files: "
                "missing_data=%d missing_video=%d available_episodes=%d",
                missing_data,
                missing_video,
                len(self.episodes),
            )

    def load_episode(self, episode: EpisodeRecord) -> dict[str, torch.Tensor]:
        columns = [
            "observation.arm_joints",
            "observation.leg_joints",
            "observation.hand_joints",
            "observation.odometry.position",
            "observation.odometry.quat",
            "action",
        ]
        table = self.reader.read(episode, columns)
        arm = _as_matrix(table["observation.arm_joints"], 14, "HumanoidEveryday arm")
        leg = _as_matrix(table["observation.leg_joints"], 15, "HumanoidEveryday leg")
        observed_hand_raw = _as_matrix(
            table["observation.hand_joints"], 14, "HumanoidEveryday hand"
        )
        action = _as_matrix(table["action"], 28, "HumanoidEveryday action")
        root_position = _as_matrix(
            table["observation.odometry.position"], 3, "HumanoidEveryday odometry position"
        )
        root_quaternion = _as_matrix(
            table["observation.odometry.quat"], 4, "HumanoidEveryday odometry quaternion"
        )

        # The released G1 LeRobot converter stores action as Dex3 hands first
        # (left7 + right7), followed by the 14 arm IK targets.  It contains no
        # leg or root target, so complete those features with measured state.
        observed_q = np.concatenate((leg, arm), axis=1)
        target_q = np.concatenate((leg, action[:, 14:28]), axis=1)
        current_configuration = np.concatenate(
            (root_position, root_quaternion, observed_q), axis=1
        )
        target_configuration = np.concatenate(
            (root_position, root_quaternion, target_q), axis=1
        )
        frame_offset, motion_issue = _unifolm_motion_discontinuity(
            current_configuration,
            target_configuration,
            first_root_jump_threshold=self.selection.get(
                "first_root_jump_threshold",
                UNIFOLM_FIRST_ROOT_JUMP_THRESHOLD_METERS,
            ),
            root_jump_threshold=self.selection.get(
                "internal_root_jump_threshold",
                UNIFOLM_INTERNAL_ROOT_JUMP_THRESHOLD_METERS,
            ),
            joint_jump_threshold=self.selection.get(
                "joint_jump_threshold_radians",
                UNIFOLM_JOINT_JUMP_THRESHOLD_RADIANS,
            ),
            root_rotation_jump_threshold_degrees=self.selection.get(
                "root_rotation_jump_threshold_degrees",
                UNIFOLM_ROOT_ROTATION_JUMP_THRESHOLD_DEGREES,
            ),
            max_initial_trim_frames=self.selection.get(
                "max_initial_trim_frames", UNIFOLM_MAX_INITIAL_TRIM_FRAMES
            ),
        )
        if motion_issue is not None:
            transition_frame = int(motion_issue["transition_frame"])
            quality_issue = (
                "motion discontinuity at frames "
                f"{transition_frame}->{transition_frame + 1}: "
                f"root={motion_issue['root_jump']:.3f} m, "
                f"joint={motion_issue['joint_jump']:.3f} rad, "
                "root_rotation="
                f"{motion_issue['root_rotation_jump_degrees']:.1f} deg"
            )
            logger.warning(
                "Excluding HumanoidEveryday episode %s: %s",
                episode.episode_id,
                quality_issue,
            )
            return {
                "skip_episode": True,
                "quality_issue": quality_issue,
            }
        if frame_offset:
            logger.debug(
                "Logically trimming %d reset frame(s) from HumanoidEveryday episode %s",
                frame_offset,
                episode.episode_id,
            )
            observed_hand_raw = observed_hand_raw[frame_offset:]
            action = action[frame_offset:]
            root_position = root_position[frame_offset:]
            root_quaternion = root_quaternion[frame_offset:]
            observed_q = observed_q[frame_offset:]
            target_q = target_q[frame_offset:]
            episode_for_motion = replace(
                episode, source_length=episode.source_length - frame_offset
            )
        else:
            episode_for_motion = episode
        if episode_for_motion.target_length < self.action_chunk:
            return {
                "skip_episode": True,
                "quality_issue": "episode is too short after initial reset trimming",
            }
        observed_hand, target_hand, observed_valid, target_valid = _dex3_pair_to_binary(
            observed_hand_raw, action[:, :14]
        )
        decoder = self._decoder(episode.source_fps)
        local_root_position, local_root_rotations = (
            _episode_local_root_from_odometry(root_position, root_quaternion)
        )
        observed = decoder.decode_joint_configuration_pose(
            observed_q,
            local_root_position,
            root_rotation_matrices=local_root_rotations,
            joint_names=UNITREE_G1_JOINT_NAMES_29,
        )
        target = decoder.decode_joint_configuration_pose(
            target_q,
            local_root_position,
            root_rotation_matrices=local_root_rotations,
            joint_names=UNITREE_G1_JOINT_NAMES_29,
        )
        motion = self._finalize_motion(
            episode_for_motion,
            observed["local_rot_mats"],
            observed["root_positions"],
            target["local_rot_mats"],
            target["root_positions"],
            observed_hand,
            target_hand,
            observed_valid,
            target_valid,
            target_motion_source="action_with_state_root_and_legs",
        )
        motion["frame_offset"] = int(
            round(frame_offset * episode.target_fps / episode.source_fps)
        )
        motion["source_frame_offset"] = frame_offset
        return motion


class HIW500Adapter(BaseSourceAdapter):
    source_name = SOURCE_HIW500

    def _task_catalog(self) -> dict[int, str]:
        tasks = pq.read_table(self.root / "meta/tasks.parquet").to_pydict()
        return {
            int(index): str(task).strip()
            for index, task in zip(tasks["task_index"], tasks["task"])
        }

    def discover(self) -> None:
        info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))
        features = info.get("features", {})
        _validate_named_feature(
            features, "observation.state", HIW_G1_JOINT_FEATURE_NAMES_29
        )
        _validate_named_feature(
            features, "observation.state.wbc", HIW_WBC_FEATURE_NAMES_23
        )
        _validate_named_feature(features, "action", HIW_WBC_FEATURE_NAMES_23)
        configured_camera = self.selection.get("camera")
        video_key = _video_key(
            features,
            tuple(
                key
                for key in (configured_camera, "observation.images.head")
                if key
            ),
        )
        video_crop = _stereo_crop(features[video_key], self.selection)
        task_catalog = self._task_catalog()
        episode_meta_paths = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        if episode_meta_paths:
            for meta_path in episode_meta_paths:
                metadata = pq.read_table(meta_path).to_pydict()
                for row in range(len(metadata["episode_index"])):
                    instruction = _instruction(metadata["tasks"][row])
                    if not _matches_selection(self.selection, instruction):
                        continue
                    data_chunk = int(metadata["data/chunk_index"][row])
                    data_file = int(metadata["data/file_index"][row])
                    video_chunk = int(metadata[f"videos/{video_key}/chunk_index"][row])
                    video_file = int(metadata[f"videos/{video_key}/file_index"][row])
                    episode_index = int(metadata["episode_index"][row])
                    self._record(
                        task_id=f"{self.source_name}::{instruction}",
                        task_name=instruction,
                        instruction=instruction,
                        episode_id=str(episode_index),
                        data_path=self.root / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet",
                        source_length=int(metadata["length"][row]),
                        source_fps=float(info["fps"]),
                        video_path=self.root / "videos" / video_key / f"chunk-{video_chunk:03d}" / f"file-{video_file:03d}.mp4",
                        video_from_timestamp=float(metadata[f"videos/{video_key}/from_timestamp"][row]),
                        metadata={
                            "dataset_from_index": int(metadata["dataset_from_index"][row]),
                            "dataset_to_index": int(metadata["dataset_to_index"][row]),
                            "video_crop": video_crop,
                        },
                    )
            return

        # Debug subsets may intentionally omit meta/episodes. Group contiguous
        # rows by episode_index and keep the standard chunk/file video mapping.
        for data_path in sorted((self.root / "data").rglob("*.parquet")):
            identifiers = pq.read_table(
                data_path,
                columns=["episode_index", "frame_index", "task_index", "timestamp"],
            ).to_pydict()
            episode_ids = np.asarray(identifiers["episode_index"], dtype=np.int64)
            if episode_ids.size == 0:
                continue
            boundaries = np.flatnonzero(np.diff(episode_ids) != 0) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [episode_ids.size]))
            chunk_name = data_path.parent.name
            file_name = data_path.with_suffix(".mp4").name
            video_path = self.root / "videos" / video_key / chunk_name / file_name
            for row_start, row_end in zip(starts.tolist(), ends.tolist()):
                task_index = int(identifiers["task_index"][row_start])
                instruction = task_catalog.get(task_index, f"task_{task_index}")
                if not _matches_selection(self.selection, instruction):
                    continue
                episode_index = int(episode_ids[row_start])
                self._record(
                    task_id=f"{self.source_name}::{instruction}",
                    task_name=instruction,
                    instruction=instruction,
                    episode_id=str(episode_index),
                    data_path=data_path,
                    source_length=row_end - row_start,
                    source_fps=float(info["fps"]),
                    video_path=video_path,
                    video_from_timestamp=float(identifiers["timestamp"][row_start]),
                    metadata={
                        "row_start": row_start,
                        "row_end": row_end,
                        "video_crop": video_crop,
                    },
                )

    def load_episode(self, episode: EpisodeRecord) -> dict[str, torch.Tensor]:
        table = self.reader.read(
            episode, ["observation.state", "observation.state.wbc", "action"]
        )
        joint_q = _as_matrix(table["observation.state"], 29, "HIW joint state")
        wbc_state = _as_matrix(table["observation.state.wbc"], 23, "HIW WBC state")
        action = _as_matrix(table["action"], 23, "HIW action")
        frame_offset, joint_issue = _hiw_joint_discontinuity(
            joint_q,
            joint_jump_threshold=self.selection.get(
                "joint_jump_threshold_radians",
                UNIFOLM_JOINT_JUMP_THRESHOLD_RADIANS,
            ),
            max_initial_trim_frames=self.selection.get(
                "max_initial_trim_frames", UNIFOLM_MAX_INITIAL_TRIM_FRAMES
            ),
        )
        if joint_issue is not None:
            transition_frame = int(joint_issue["transition_frame"])
            quality_issue = (
                "joint discontinuity at frames "
                f"{transition_frame}->{transition_frame + 1}: "
                f"joint={joint_issue['joint_jump']:.3f} rad"
            )
            logger.warning(
                "Excluding HIW episode %s: %s",
                episode.episode_id,
                quality_issue,
            )
            return {
                "skip_episode": True,
                "quality_issue": quality_issue,
            }
        if frame_offset:
            joint_q = joint_q[frame_offset:]
            wbc_state = wbc_state[frame_offset:]
            action = action[frame_offset:]
            episode_for_motion = replace(
                episode, source_length=episode.source_length - frame_offset
            )
        else:
            episode_for_motion = episode
        if episode_for_motion.target_length < self.action_chunk:
            return {
                "skip_episode": True,
                "quality_issue": "episode is too short after initial reset trimming",
            }
        # HIW LeRobot stores a commanded base twist and height, but no measured
        # odometry pose.  Build an episode-local command trajectory; torso RPY
        # is deliberately excluded because it is not the pelvis orientation.
        observed_root_position, observed_root_rotation = _hiw_root_from_wbc(
            wbc_state, episode.source_fps
        )
        target_root_position, target_root_rotation = _hiw_root_from_wbc(
            action, episode.source_fps
        )
        trigger_threshold = float(
            self.selection.get(
                "hand_trigger_threshold", HIW_TRIGGER_CLOSE_THRESHOLD
            )
        )
        squeeze_threshold = float(
            self.selection.get(
                "hand_squeeze_threshold", HIW_SQUEEZE_OPEN_THRESHOLD
            )
        )
        observed_hand = _hiw_hand_from_events(
            wbc_state,
            trigger_threshold=trigger_threshold,
            squeeze_threshold=squeeze_threshold,
        )
        target_hand = _hiw_hand_from_events(
            action,
            trigger_threshold=trigger_threshold,
            squeeze_threshold=squeeze_threshold,
        )
        observed_valid = np.ones_like(observed_hand, dtype=bool)
        target_valid = np.ones_like(target_hand, dtype=bool)
        decoder = self._decoder(episode.source_fps)
        observed = decoder.decode_joint_configuration_pose(
            joint_q,
            observed_root_position,
            root_rotation_matrices=observed_root_rotation,
            joint_names=UNITREE_G1_JOINT_NAMES_29,
        )
        target = decoder.decode_joint_configuration_pose(
            joint_q,
            target_root_position,
            root_rotation_matrices=target_root_rotation,
            joint_names=UNITREE_G1_JOINT_NAMES_29,
        )
        motion = self._finalize_motion(
            episode_for_motion,
            observed["local_rot_mats"],
            observed["root_positions"],
            target["local_rot_mats"],
            target["root_positions"],
            observed_hand,
            target_hand,
            observed_valid,
            target_valid,
            target_motion_source=(
                "integrated_commanded_base_twist_and_hand_events_"
                "with_executed_joint_completion"
            ),
        )
        motion["frame_offset"] = int(
            round(frame_offset * episode.target_fps / episode.source_fps)
        )
        motion["source_frame_offset"] = frame_offset
        return motion


ADAPTER_BY_SOURCE = {
    SOURCE_HUMANOID_ARENA: HumanoidArenaAdapter,
    SOURCE_HUMANOID_EVERYDAY: HumanoidEverydayAdapter,
    SOURCE_HIW500: HIW500Adapter,
    SOURCE_UNIFOLM: UnifoLMAdapter,
}


class MultiSourceG1Dataset(data.Dataset):
    """Source-balanced mixed dataset aligned to Kimodo's 417D G1 contract."""

    def __init__(
        self,
        dataset_root: str | None = None,
        dataset_roots: Mapping[str, str] | None = None,
        action_history: int = 100,
        action_chunk: int = 50,
        sample_stride: int = 1,
        episode_cache_size: int = 8,
        video_cache_size: int = DEFAULT_VIDEO_CACHE_SIZE,
        dataset_selection: Mapping | None = None,
        sampling: Mapping | None = None,
        target_fps: float = 30.0,
        sampling_seed: int = 0,
    ) -> None:
        self.action_history = int(action_history)
        self.action_chunk = int(action_chunk)
        self.sample_stride = int(sample_stride)
        self.episode_cache_size = int(episode_cache_size)
        self.video_cache_size = int(video_cache_size)
        self.target_fps = float(target_fps)
        self.sampling_seed = int(sampling_seed)
        if min(self.action_history, self.action_chunk, self.sample_stride) <= 0:
            raise ValueError("action_history, action_chunk and sample_stride must be positive")
        if self.episode_cache_size < 0 or self.video_cache_size < 0:
            raise ValueError("episode_cache_size and video_cache_size must be non-negative")

        roots = self._normalize_roots(dataset_root, dataset_roots)
        selections = self._normalize_selection(dataset_selection, roots)
        self.adapters: dict[str, BaseSourceAdapter] = {}
        self._episodes_by_source: dict[str, list[tuple[BaseSourceAdapter, EpisodeRecord]]] = {}
        self._episodes_by_source_task: dict[
            str, dict[str, list[tuple[BaseSourceAdapter, EpisodeRecord]]]
        ] = {}
        self._task_instructions: dict[str, str] = {}
        self._task_cache_names: dict[str, str] = {}
        total_windows = 0
        for source_name, selection in selections.items():
            root = roots.get(source_name)
            if root is None:
                raise KeyError(f"No dataset root configured for selected source {source_name}")
            if not root.is_dir():
                raise FileNotFoundError(f"{source_name} dataset root does not exist: {root}")
            adapter = ADAPTER_BY_SOURCE[source_name](
                root=root,
                selection=selection,
                target_fps=self.target_fps,
                action_chunk=self.action_chunk,
            )
            if not adapter.episodes:
                raise RuntimeError(
                    f"No compatible {source_name} episodes found below {root} for selection {selection}"
                )
            for episode in adapter.episodes:
                episode.metadata["sample_stride"] = self.sample_stride
                last_cut = episode.target_length - self.action_chunk
                episode.sample_count = last_cut // self.sample_stride + 1
                existing = self._task_instructions.get(episode.task_id)
                if existing is not None and existing != episode.instruction:
                    raise ValueError(
                        f"Task ID {episode.task_id!r} maps to conflicting instructions"
                    )
                existing_name = self._task_cache_names.get(episode.task_id)
                if existing_name is not None and existing_name != episode.task_name:
                    raise ValueError(
                        f"Task ID {episode.task_id!r} maps to conflicting task names"
                    )
                self._task_instructions[episode.task_id] = episode.instruction
                self._task_cache_names[episode.task_id] = episode.task_name
                total_windows += episode.sample_count
            self.adapters[source_name] = adapter
            self._episodes_by_source[source_name] = [
                (adapter, episode) for episode in adapter.episodes
            ]
            records_by_task: dict[
                str, list[tuple[BaseSourceAdapter, EpisodeRecord]]
            ] = defaultdict(list)
            for record in self._episodes_by_source[source_name]:
                records_by_task[record[1].task_id].append(record)
            self._episodes_by_source_task[source_name] = dict(records_by_task)

        self._length = total_windows
        self._episode_cache: OrderedDict[
            tuple[str, str, str, str, int, int], dict[str, torch.Tensor]
        ] = OrderedDict()
        self._video_cache: OrderedDict[str, tuple[object, object]] = OrderedDict()
        self._runtime_invalid_episodes: set[
            tuple[str, str, str, str, int, int]
        ] = set()
        self._text_embeddings: dict[str, torch.Tensor] = {}
        sampling = dict(sampling or {})
        windows_per_episode = sampling.get("windows_per_episode", 1)
        if (
            isinstance(windows_per_episode, bool)
            or not isinstance(windows_per_episode, int)
            or windows_per_episode <= 0
        ):
            raise ValueError("sampling.windows_per_episode must be a positive integer")
        self.windows_per_episode = windows_per_episode
        mode = str(sampling.get("mode", "episode_uniform"))
        if mode not in {
            "episode_uniform",
            "source_balanced",
            "source_task_balanced",
            "window_proportional",
        }:
            raise ValueError(
                "sampling.mode must be 'episode_uniform', 'source_balanced', "
                "'source_task_balanced', or 'window_proportional'"
            )
        self.sampling_mode = mode
        self._source_names = list(self._episodes_by_source)
        self._all_episode_records = [
            record
            for source in self._source_names
            for record in self._episodes_by_source[source]
        ]
        self._source_weights: list[float] | None = None
        self._episode_weights: list[int] | None = None
        sampling_detail = "all episodes have equal probability"
        if mode in {"source_balanced", "source_task_balanced"}:
            configured_weights = dict(sampling.get("source_weights", {}))
            self._source_weights = [
                float(configured_weights.get(source, 1.0))
                for source in self._source_names
            ]
            if any(weight <= 0 for weight in self._source_weights):
                raise ValueError("All selected source sampling weights must be positive")
            sampling_detail = (
                f"source_weights={dict(zip(self._source_names, self._source_weights))}"
            )
            if mode == "source_task_balanced":
                sampling_detail += "; tasks are uniform within each source"
        elif mode == "window_proportional":
            self._episode_weights = [
                episode.sample_count for _, episode in self._all_episode_records
            ]
            sampling_detail = "episode weights are proportional to valid start points"
        if self.windows_per_episode > 1:
            sampling_detail += (
                f"; {self.windows_per_episode} windows are sampled per episode group"
            )
        logger.info(
            "Loaded mixed G1 dataset: sources=%s episodes=%s windows=%d sampling=%s (%s)",
            self._source_names,
            {source: len(self._episodes_by_source[source]) for source in self._source_names},
            self._length,
            self.sampling_mode,
            sampling_detail,
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_video_cache"] = OrderedDict()
        return state

    def close(self) -> None:
        video_cache = getattr(self, "_video_cache", None)
        if video_cache is not None:
            while video_cache:
                _, (container, _) = video_cache.popitem(last=False)
                container.close()
        for adapter in getattr(self, "adapters", {}).values():
            adapter.reader.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _canonical_source_name(name: str) -> str:
        aliases = {
            "humanoidarena": SOURCE_HUMANOID_ARENA,
            "humanoideveryday": SOURCE_HUMANOID_EVERYDAY,
            "hiw500": SOURCE_HIW500,
            "hiw-500": SOURCE_HIW500,
            "unifolm_wbt_dataset": SOURCE_UNIFOLM,
            "unifolm": SOURCE_UNIFOLM,
        }
        canonical = aliases.get(str(name).strip().lower())
        if canonical is None:
            raise ValueError(f"Unsupported dataset source {name!r}; expected one of {KNOWN_SOURCES}")
        return canonical

    @classmethod
    def _normalize_roots(
        cls,
        dataset_root: str | None,
        dataset_roots: Mapping[str, str] | None,
    ) -> dict[str, Path]:
        roots = {}
        for name, value in dict(dataset_roots or {}).items():
            canonical = cls._canonical_source_name(name)
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            roots[canonical] = path.resolve()
        if dataset_root is not None and SOURCE_HUMANOID_ARENA not in roots:
            path = Path(dataset_root).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            roots[SOURCE_HUMANOID_ARENA] = path.resolve()
        if not roots:
            raise ValueError("Configure main.data_root or main.dataset_roots")
        return roots

    @classmethod
    def _normalize_selection(
        cls,
        selection: Mapping | None,
        roots: Mapping[str, Path],
    ) -> dict[str, Mapping]:
        selection = dict(selection or {})
        nested = any(
            str(key).strip().lower()
            in {
                "humanoidarena",
                "humanoideveryday",
                "hiw500",
                "hiw-500",
                "unifolm_wbt_dataset",
                "unifolm",
            }
            for key in selection
        )
        if not nested:
            return {SOURCE_HUMANOID_ARENA: selection}
        normalized = {}
        for name, source_selection in selection.items():
            canonical = cls._canonical_source_name(name)
            if source_selection is False or source_selection is None:
                continue
            if source_selection is True:
                source_selection = {}
            if not isinstance(source_selection, Mapping):
                raise TypeError(
                    f"dataset_selection.{canonical} must be a mapping"
                )
            normalized[canonical] = dict(source_selection)
        if not normalized:
            raise ValueError(
                "dataset_selection disables every configured data source"
            )
        return normalized

    @property
    def instructions(self) -> list[str]:
        return sorted(set(self._task_instructions.values()))

    @property
    def task_instructions(self) -> dict[str, str]:
        return dict(sorted(self._task_instructions.items()))

    @property
    def task_cache_names(self) -> dict[str, str]:
        return dict(sorted(self._task_cache_names.items()))

    @property
    def source_summary(self) -> dict[str, dict[str, int]]:
        return {
            source: {
                "episodes": len(records),
                "windows": sum(episode.sample_count for _, episode in records),
            }
            for source, records in self._episodes_by_source.items()
        }

    def set_text_embeddings(self, embeddings: Mapping[str, torch.Tensor]) -> None:
        missing = set(self._task_instructions) - set(embeddings)
        if missing:
            raise KeyError(f"Missing text embeddings for task IDs {sorted(missing)}")
        self._text_embeddings = {
            task_id: torch.as_tensor(embeddings[task_id]).cpu().contiguous()
            for task_id in self._task_instructions
        }

    def __len__(self) -> int:
        return self._length

    def _rng_for_index(self, index: int) -> random.Random:
        # SplitMix64 gives a stable, process-independent mapping from ordinal to seed.
        value = (int(index) + 0x9E3779B97F4A7C15) & _UINT64_MASK
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & _UINT64_MASK
        value = (value ^ (value >> 27)) * 0x94D049BB133111EB & _UINT64_MASK
        value ^= value >> 31
        seed = value ^ (int(getattr(self, "sampling_seed", 0)) & _UINT64_MASK)
        return random.Random(seed)

    def _sample_episode(
        self, rng=None
    ) -> tuple[BaseSourceAdapter, EpisodeRecord]:
        rng = random if rng is None else rng
        if self.sampling_mode in {"source_balanced", "source_task_balanced"}:
            source = rng.choices(
                self._source_names, weights=self._source_weights, k=1
            )[0]
            if self.sampling_mode == "source_task_balanced":
                records_by_task = self._episodes_by_source_task[source]
                task_id = rng.choice(list(records_by_task))
                adapter, episode = rng.choice(records_by_task[task_id])
            else:
                adapter, episode = rng.choice(self._episodes_by_source[source])
        elif self.sampling_mode == "window_proportional":
            adapter, episode = rng.choices(
                self._all_episode_records,
                weights=self._episode_weights,
                k=1,
            )[0]
        else:
            adapter, episode = rng.choice(self._all_episode_records)
        return adapter, episode

    def _sample_record(
        self, rng=None
    ) -> tuple[BaseSourceAdapter, EpisodeRecord, int]:
        rng = random if rng is None else rng
        adapter, episode = self._sample_episode(rng)
        local_index = rng.randrange(episode.sample_count)
        cut = episode.first_cut + local_index * self.sample_stride
        return adapter, episode, cut

    def _sampling_state_for_index(
        self, index: int
    ) -> tuple[random.Random, int]:
        windows_per_episode = int(getattr(self, "windows_per_episode", 1))
        if windows_per_episode == 1:
            return self._rng_for_index(index), 0
        group_index, group_slot = divmod(int(index), windows_per_episode)
        return self._rng_for_index(group_index), group_slot

    def _sample_grouped_record(
        self, rng: random.Random, group_slot: int
    ) -> tuple[BaseSourceAdapter, EpisodeRecord, int]:
        """Choose one episode and a deterministic window for one group slot."""
        windows_per_episode = int(getattr(self, "windows_per_episode", 1))
        if windows_per_episode == 1:
            return self._sample_record(rng)
        if not 0 <= int(group_slot) < windows_per_episode:
            raise ValueError(
                f"group_slot must be in [0, {windows_per_episode}), got {group_slot}"
            )

        adapter, episode = self._sample_episode(rng)
        cut_rng = random.Random(rng.getrandbits(64))
        local_indices = self._sample_group_local_indices(
            cut_rng,
            first_local_index=0,
            valid_sample_count=episode.sample_count,
        )
        local_index = local_indices[int(group_slot)]
        cut = episode.first_cut + local_index * self.sample_stride
        return adapter, episode, cut

    def _sample_group_local_indices(
        self,
        rng: random.Random,
        *,
        first_local_index: int,
        valid_sample_count: int,
    ) -> list[int]:
        windows_per_episode = int(getattr(self, "windows_per_episode", 1))
        first_local_index = int(first_local_index)
        valid_sample_count = int(valid_sample_count)
        if first_local_index < 0 or valid_sample_count <= 0:
            raise ValueError(
                "Grouped sampling requires a non-negative first index and at "
                "least one valid sample"
            )

        first_selected = first_local_index + rng.randrange(valid_sample_count)
        local_indices = [first_selected]
        remaining_count = windows_per_episode - 1
        if valid_sample_count >= windows_per_episode:
            # Sample the remaining indices without materializing the potentially
            # large range and remap around the already selected first index.
            remaining_indices = rng.sample(
                range(valid_sample_count - 1), remaining_count
            )
            first_relative_index = first_selected - first_local_index
            local_indices.extend(
                first_local_index
                + (index if index < first_relative_index else index + 1)
                for index in remaining_indices
            )
        else:
            local_indices.extend(
                first_local_index + rng.randrange(valid_sample_count)
                for _ in range(remaining_count)
            )
        return local_indices

    def _episode_motion(
        self, adapter: BaseSourceAdapter, episode: EpisodeRecord
    ) -> dict[str, torch.Tensor]:
        cache_key = episode.cache_key
        cached = self._episode_cache.pop(cache_key, None)
        if cached is not None:
            self._episode_cache[cache_key] = cached
            return cached
        motion = adapter.load_episode(episode)
        self._episode_cache[cache_key] = motion
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return motion

    def __getitem__(self, index: int) -> dict:
        rng, group_slot = self._sampling_state_for_index(index)
        windows_per_episode = int(getattr(self, "windows_per_episode", 1))
        skipped_reasons = []
        invalid_episodes = getattr(self, "_runtime_invalid_episodes", None)
        if invalid_episodes is None:
            invalid_episodes = set()
            self._runtime_invalid_episodes = invalid_episodes
        for _ in range(MAX_SAMPLE_ATTEMPTS):
            if windows_per_episode == 1:
                adapter, episode, original_cut = self._sample_record(rng)
            else:
                adapter, episode = self._sample_episode(rng)
                cut_rng = random.Random(rng.getrandbits(64))
            if episode.cache_key in invalid_episodes:
                skipped_reasons.append(
                    f"{episode.source}/{episode.episode_id}: "
                    "previously marked invalid"
                )
                continue
            episode_motion = self._episode_motion(adapter, episode)
            if episode_motion.get("skip_episode", False):
                invalid_episodes.add(episode.cache_key)
                self._episode_cache.pop(episode.cache_key, None)
                skipped_reasons.append(
                    f"{episode.source}/{episode.episode_id}: "
                    f"{episode_motion.get('quality_issue', 'invalid episode')}"
                )
                continue
            frame_offset = int(episode_motion.get("frame_offset", 0))
            available_length = int(episode_motion["target_motion"].shape[0])
            if windows_per_episode > 1:
                first_valid_cut = max(episode.first_cut, frame_offset)
                first_local_index = max(
                    0,
                    (
                        first_valid_cut
                        - episode.first_cut
                        + self.sample_stride
                        - 1
                    )
                    // self.sample_stride,
                )
                last_valid_cut = (
                    frame_offset + available_length - self.action_chunk
                )
                last_local_index = min(
                    episode.sample_count - 1,
                    (last_valid_cut - episode.first_cut) // self.sample_stride,
                )
                valid_sample_count = last_local_index - first_local_index + 1
                if valid_sample_count <= 0:
                    skipped_reasons.append(
                        f"{episode.source}/{episode.episode_id}: no valid window "
                        "after logical frame trimming"
                    )
                    continue
                if (
                    valid_sample_count < episode.sample_count
                    and cut_rng.randrange(episode.sample_count)
                    >= valid_sample_count
                ):
                    # Match the legacy rejection sampler: selecting an episode
                    # remains proportional to its configured sample_count, while
                    # logically trimmed windows are rejected before a group is built.
                    continue
                local_indices = self._sample_group_local_indices(
                    cut_rng,
                    first_local_index=first_local_index,
                    valid_sample_count=valid_sample_count,
                )
                original_cut = (
                    episode.first_cut
                    + local_indices[group_slot] * self.sample_stride
                )
            cut = original_cut - frame_offset
            if cut < 0:
                continue
            history_start = max(0, cut - self.action_history)
            history_length = cut - history_start
            future_end = min(available_length, cut + self.action_chunk)
            future_length = future_end - cut
            if future_length != self.action_chunk:
                continue
            video_timestamp = (
                episode.video_from_timestamp
                + original_cut / episode.target_fps
            )
            try:
                egoview = self._read_video_frame(
                    episode.video_path,
                    video_timestamp,
                    crop=episode.metadata.get("video_crop"),
                )
            except VideoFrameDecodeError as error:
                invalid_episodes.add(episode.cache_key)
                self._episode_cache.pop(episode.cache_key, None)
                reason = (
                    f"{episode.source}/{episode.episode_id}: video decoding "
                    f"failed for {episode.video_path} at "
                    f"{video_timestamp:.3f}s"
                )
                skipped_reasons.append(reason)
                logger.warning("Excluding %s: %s", reason, error)
                continue
            break
        else:
            details = "; ".join(skipped_reasons[-5:]) or "no valid sampled window"
            raise RuntimeError(
                f"Could not sample a valid episode window after {MAX_SAMPLE_ATTEMPTS} attempts: "
                f"{details}"
            )

        total_length = self.action_history + self.action_chunk
        gt_motion = torch.zeros(
            total_length, KIMODO_MOTION_DIM, dtype=torch.float32
        )
        condition_motion = torch.zeros(
            total_length, KIMODO_MOTION_DIM, dtype=torch.float32
        )
        condition_motion_mask = torch.zeros(
            total_length, KIMODO_MOTION_DIM, dtype=torch.bool
        )
        gt_hand = torch.zeros(total_length, 2, dtype=torch.float32)
        gt_hand_mask = torch.zeros(total_length, 2, dtype=torch.bool)
        gt_mask = torch.zeros(total_length, dtype=torch.bool)
        history_destination = self.action_history - history_length
        if history_length:
            history_slice = slice(history_start, cut)
            destination = slice(history_destination, self.action_history)
            gt_motion[destination] = episode_motion["target_motion"][history_slice]
            condition_motion[destination] = episode_motion["observed_motion"][
                history_slice
            ]
            observed_motion_valid = episode_motion.get("observed_motion_valid")
            if observed_motion_valid is None:
                condition_motion_mask[destination] = True
            else:
                condition_motion_mask[destination] = observed_motion_valid[
                    history_slice
                ]
            gt_hand[destination] = episode_motion["observed_hand"][history_slice]
            gt_hand_mask[destination] = episode_motion["observed_hand_valid"][history_slice]
            gt_mask[destination] = True
        future_slice = slice(cut, future_end)
        destination = slice(self.action_history, self.action_history + future_length)
        gt_motion[destination] = episode_motion["target_motion"][future_slice]
        gt_hand[destination] = episode_motion["target_hand"][future_slice]
        gt_hand_mask[destination] = episode_motion["target_hand_valid"][future_slice]
        gt_mask[destination] = True

        _canonicalize_kimodo_window_translation(
            gt_motion,
            condition_motion,
            condition_motion_mask,
            gt_mask,
        )
        sample = {
            "instruction": episode.instruction,
            "egoview": egoview,
            "gt_motion": gt_motion,
            "condition_motion": condition_motion,
            "condition_motion_mask": condition_motion_mask,
            "gt_hand": gt_hand,
            "gt_hand_mask": gt_hand_mask,
            "gt_mask": gt_mask,
            "source": episode.source,
            "task_id": episode.task_id,
            "episode_id": episode.episode_id,
            "cut_index": original_cut,
            "motion_cut_index": cut,
            "source_frame_offset": int(
                episode_motion.get("source_frame_offset", 0)
            ),
            "target_motion_source": episode_motion.get(
                "target_motion_source", "action"
            ),
        }
        if self._text_embeddings:
            sample["text_embedding"] = self._text_embeddings[episode.task_id]
            sample["text_length"] = torch.tensor(
                sample["text_embedding"].shape[0], dtype=torch.long
            )
        return sample

    def _read_video_frame(
        self,
        video_path: Path,
        timestamp: float,
        crop: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        for attempt in range(VIDEO_READ_MAX_ATTEMPTS):
            container = None
            cached = False
            try:
                cache = getattr(self, "_video_cache", None)
                if cache is None:
                    cache = OrderedDict()
                    self._video_cache = cache
                cache_key = str(video_path)
                entry = cache.pop(cache_key, None)
                if entry is None:
                    container = av.open(cache_key)
                    if not container.streams.video:
                        raise VideoFrameDecodeError(
                            f"No video stream in {video_path}"
                        )
                    stream = container.streams.video[0]
                    stream.codec_context.thread_count = 1
                    if int(getattr(self, "video_cache_size", 0)) > 0:
                        cache[cache_key] = (container, stream)
                        cached = True
                        while len(cache) > int(self.video_cache_size):
                            _, (evicted, _) = cache.popitem(last=False)
                            evicted.close()
                else:
                    container, stream = entry
                    cache[cache_key] = entry
                    cached = True

                time_base = getattr(stream, "time_base", None)
                if time_base is not None and float(time_base) > 0:
                    seek_offset = max(0, int(timestamp / float(time_base)))
                    container.seek(
                        seek_offset,
                        backward=True,
                        any_frame=False,
                        stream=stream,
                    )
                else:
                    container.seek(max(0, int(timestamp * av.time_base)))
                selected = None
                for frame in container.decode(stream):
                    selected = frame
                    frame_time = (
                        float(frame.pts * stream.time_base)
                        if frame.pts is not None
                        else timestamp
                    )
                    if frame_time + 1e-6 >= timestamp:
                        break
                if selected is None:
                    raise VideoFrameDecodeError(
                        f"Could not decode frame at {timestamp:.3f}s from {video_path}"
                    )
                image = selected.to_ndarray(format="rgb24")
                if crop is not None:
                    x0, y0, x1, y1 = map(int, crop)
                    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
                        raise ValueError(
                            f"Invalid video crop {crop} for frame shape {image.shape}"
                        )
                    image = image[y0:y1, x0:x1]
                return (
                    torch.from_numpy(image.copy())
                    .permute(2, 0, 1)
                    .contiguous()
                )
            except av.error.BlockingIOError as error:
                self._discard_video_cache_entry(video_path)
                if attempt == VIDEO_READ_MAX_ATTEMPTS - 1:
                    raise VideoFrameDecodeError(
                        f"PyAV repeatedly failed at {timestamp:.3f}s in {video_path}"
                    ) from error
                delay = VIDEO_READ_RETRY_DELAY_SECONDS * (2**attempt)
                time.sleep(delay)
            except VideoFrameDecodeError:
                self._discard_video_cache_entry(video_path)
                raise
            except av.error.FFmpegError as error:
                self._discard_video_cache_entry(video_path)
                raise VideoFrameDecodeError(
                    f"PyAV failed at {timestamp:.3f}s in {video_path}: {error}"
                ) from error
            except OSError as error:
                self._discard_video_cache_entry(video_path)
                raise VideoFrameDecodeError(
                    f"Video I/O failed at {timestamp:.3f}s in {video_path}: {error}"
                ) from error
            finally:
                if container is not None and not cached:
                    container.close()
        raise VideoFrameDecodeError(f"Could not read {video_path}")

    def _discard_video_cache_entry(self, video_path: Path) -> None:
        cache = getattr(self, "_video_cache", None)
        if cache is None:
            return
        entry = cache.pop(str(video_path), None)
        if entry is not None:
            entry[0].close()
