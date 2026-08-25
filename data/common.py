"""Unified G1 training dataset for Arena, real-world and other sources."""

from __future__ import annotations

import fnmatch
import json
import logging
import math
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

from data.motion_cache import (
    PRETRAIN_MOTION_CACHE_SOURCES,
    episode_cache_token,
    load_motion_cache_manifest,
)
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
SOURCE_REAL_WORLD = "RealWorld"
KNOWN_SOURCES = (
    SOURCE_HUMANOID_ARENA,
    SOURCE_HUMANOID_EVERYDAY,
    SOURCE_HIW500,
    SOURCE_UNIFOLM,
    SOURCE_REAL_WORLD,
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

# Keep pre-training subset selection reproducible without exposing another
# seed/configuration knob.  The episode inventory is sorted before shuffling,
# so discovery order, worker count, and GPU count do not affect the subset.
_PRETRAIN_DATA_FRACTION_SEED = 3407

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


def _as_vector(values, name: str) -> np.ndarray:
    """Convert a scalar parquet column into a finite one-dimensional vector."""
    try:
        vector = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        vector = np.asarray(
            [np.asarray(value, dtype=np.float32).reshape(-1)[0] for value in values],
            dtype=np.float32,
        )
    vector = vector.reshape(-1)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return vector


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




# Export the migrated compatibility surface, including legacy private helpers.
# Never export module dunder attributes through ``from .common import *``.
# Exporting ``__name__`` changes the importing module's identity and makes
# classes defined there appear to belong to ``data.common``; those classes
# then cannot be pickled by spawned DataLoader workers.
__all__ = [
    name
    for name in globals()
    if name != "__builtins__" and not name.startswith("__")
]
