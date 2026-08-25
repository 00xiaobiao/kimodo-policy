"""Legacy standalone HumanoidArena loader.

New mixed-source training uses :mod:`data.datasetloader` directly.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import OrderedDict
from pathlib import Path
from collections.abc import Mapping

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils import data

from motion.g1_reference import (
    HumanoidArenaActionDecoder,
    resample_hand_binary,
    resample_motion,
)
from motion.representation.kimodo_motionrep import KimodoMotionRep
from skeleton.definitions import G1Skeleton34


logger = logging.getLogger(__name__)

EXPECTED_SCHEMA = "unitree_g1_gmt_refpose_v3_1"
PREFERRED_VIDEO_KEYS = ("observation.images.front", "observation.image")
TASK_KEY_BY_TASK_ID = {
    "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody": "HOI_double_desk",
    "Isaac-Move-Football-Single-G129-Dex3-Wholebody": "HOI_football",
    "Isaac-Move-ArtVIP-Livingroom-GrapCup-G129-Dex3-Wholebody": "HOI_grap_cup",
    "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby": "HOI_pp_box",
    "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody": "HSI_boxing",
    "Isaac-Move-Open-Door-G129-Dex3-Wholebody": "HSI_open_door",
    "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody": "HSI_sit_sofa",
    "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody": "HSI_vision_navi",
}
TASK_INDEX_BY_TASK_ID = {
    task_id: task_index
    for task_index, task_id in enumerate(TASK_KEY_BY_TASK_ID)
}
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINTS_ROOT = _PROJECT_ROOT.parent / "checkpoints"
_XML_PATH = _PROJECT_ROOT / "skeleton/assets/g1skel34/xml/g1.xml"
_STATS_PATH = _CHECKPOINTS_ROOT / "Kimodo-G1-RP-v1/stats/motion"
_VIDEO_READ_MAX_ATTEMPTS = 4
_VIDEO_READ_RETRY_DELAY_SECONDS = 0.05


class HumanoidArenaDataset(data.Dataset):
    """Windowed HumanoidArena V3.1 dataset encoded in Kimodo's 417D motion space."""

    def __init__(
        self,
        dataset_root: str,
        action_history: int = 100,
        action_chunk: int = 50,
        sample_stride: int = 1,
        episode_cache_size: int = 2,
        dataset_selection: Mapping[str, str | list[str]] | None = None,
        target_fps: float = 30.0,
    ):
        dataset_path = Path(dataset_root).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = _PROJECT_ROOT / dataset_path
        self.dataset_root = dataset_path.resolve()
        self.action_history = int(action_history)
        self.action_chunk = int(action_chunk)
        self.sample_stride = int(sample_stride)
        self.episode_cache_size = int(episode_cache_size)
        self.dataset_selection = self._normalize_selection(dataset_selection)
        self.target_fps = float(target_fps)
        if self.action_history <= 0 or self.action_chunk <= 0 or self.sample_stride <= 0:
            raise ValueError("action_history, action_chunk, and sample_stride must be positive")
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"HumanoidArena dataset root does not exist: {self.dataset_root}. "
                "Expected a downloaded HumanoidArena_dataset_v3_1 directory."
            )

        self.episodes: list[dict] = []
        self._cumulative_samples: list[int] = []
        self._episode_cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._data_file_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
        self._decoder = None
        self._representation = None
        self._text_embeddings: dict[str, torch.Tensor] = {}
        self._task_instructions: dict[str, str] = {}
        self._load_metadata()

    @property
    def instructions(self) -> list[str]:
        return sorted(set(self._task_instructions.values()))

    @property
    def task_instructions(self) -> dict[str, str]:
        """Map simulator task IDs to the natural-language text used by LLM2Vec."""
        return dict(sorted(self._task_instructions.items()))

    def set_text_embeddings(self, embeddings: Mapping[str, torch.Tensor]) -> None:
        missing = set(self._task_instructions) - set(embeddings)
        if missing:
            raise KeyError(f"Missing text embeddings for task IDs {sorted(missing)}")
        self._text_embeddings = {
            task_id: torch.as_tensor(embeddings[task_id]).cpu().contiguous()
            for task_id in self._task_instructions
        }

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        backend = str(backend).strip().lower()
        aliases = {"twice2": "twist2", "twist": "twist2"}
        backend = aliases.get(backend, backend)
        if backend not in {"sonic", "twist2"}:
            raise ValueError(f"Unsupported HumanoidArena backend '{backend}'; use 'sonic' or 'twist2'")
        return backend

    @classmethod
    def _normalize_selection(
        cls, selection: Mapping[str, str | list[str]] | None
    ) -> dict[str, set[str]]:
        if not selection:
            return {}
        normalized = {}
        for task_name, backends in selection.items():
            if isinstance(backends, str):
                backends = [backends]
            normalized[str(task_name)] = {cls._normalize_backend(backend) for backend in backends}
        return normalized

    def _is_selected(self, task_name: str, backend: str) -> bool:
        if not self.dataset_selection:
            return True
        allowed_backends = self.dataset_selection.get(
            task_name, self.dataset_selection.get("*")
        )
        return allowed_backends is not None and backend in allowed_backends

    def _backend_may_be_selected(self, backend: str) -> bool:
        if not self.dataset_selection:
            return True
        return any(backend in backends for backends in self.dataset_selection.values())

    def _get_motion_tools(self, source_fps: float):
        if self._decoder is None:
            skeleton = G1Skeleton34()
            self._decoder = HumanoidArenaActionDecoder(skeleton, _XML_PATH, source_fps)
            self._representation = KimodoMotionRep(
                skeleton=skeleton,
                fps=self.target_fps,
                stats_path=str(_STATS_PATH),
            )
        elif abs(self._decoder.fps - source_fps) > 1e-6:
            raise ValueError(
                f"Mixed source FPS is unsupported in one loader worker: "
                f"{self._decoder.fps} vs {source_fps}"
            )
        return self._decoder, self._representation

    @staticmethod
    def _choose_video_key(features: dict) -> str:
        video_keys = [
            key
            for key, spec in features.items()
            if key.startswith("observation.image")
            and isinstance(spec, dict)
            and spec.get("dtype") in {"video", "image"}
        ]
        for preferred in PREFERRED_VIDEO_KEYS:
            if preferred in video_keys:
                return preferred
        if not video_keys:
            raise ValueError("Dataset has no observation image/video feature")
        return sorted(video_keys)[0]

    @staticmethod
    def _instruction(tasks) -> str:
        if isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()
        if isinstance(tasks, (list, tuple)):
            return str(tasks[0]) if tasks else ""
        return str(tasks)

    @staticmethod
    def _load_task_text_by_index(task_root: Path) -> dict[int, str]:
        tasks_path = task_root / "meta/tasks.parquet"
        if not tasks_path.is_file():
            raise FileNotFoundError(f"Dataset task language file is missing: {tasks_path}")
        task_table = pq.read_table(tasks_path, columns=["task_index", "task"]).to_pydict()
        task_text_by_index = {
            int(task_index): str(task_text).strip()
            for task_index, task_text in zip(
                task_table.get("task_index", []), task_table.get("task", [])
            )
        }
        if not task_text_by_index or any(not text for text in task_text_by_index.values()):
            raise ValueError(f"Dataset has invalid natural-language tasks in {tasks_path}")
        return task_text_by_index

    @staticmethod
    def _resolve_task_instruction(
        task_id: str,
        task_text_by_index: Mapping[int, str],
        is_aggregate: bool,
    ) -> str:
        if len(task_text_by_index) == 1:
            return next(iter(task_text_by_index.values()))
        if not is_aggregate:
            raise ValueError(
                f"Individual dataset exposes multiple language tasks for task ID {task_id!r}"
            )
        task_index = TASK_INDEX_BY_TASK_ID.get(task_id)
        if task_index is None or task_index not in task_text_by_index:
            raise KeyError(
                f"Cannot map aggregate task ID {task_id!r} to meta/tasks.parquet"
            )
        return task_text_by_index[task_index]

    @staticmethod
    def _is_aggregate_task_root(dataset_root: Path, task_root: Path) -> bool:
        relative_parts = task_root.relative_to(dataset_root).parts
        if relative_parts and relative_parts[0].lower().startswith(
            "humanoidarena_merged_datasets"
        ):
            return True
        return (
            task_root.parent.name.lower().startswith("humanoidarena_merged_datasets")
            and task_root.name.lower().startswith(
                ("all_", "sonic_", "twist2_", "twice2_")
            )
        )

    @staticmethod
    def _prefer_individual_task_roots(
        dataset_root: Path,
        task_roots: list[Path],
    ) -> list[Path]:
        individual_roots = []
        aggregate_roots = []
        for task_root in task_roots:
            if HumanoidArenaDataset._is_aggregate_task_root(dataset_root, task_root):
                aggregate_roots.append(task_root)
            else:
                individual_roots.append(task_root)
        if individual_roots and aggregate_roots:
            logger.info(
                "Ignoring %d aggregate dataset root(s) because individual task roots are present",
                len(aggregate_roots),
            )
            return individual_roots
        if aggregate_roots:
            backend_roots = [
                task_root
                for task_root in aggregate_roots
                if task_root.name.lower().startswith(("sonic_", "twist2_", "twice2_"))
            ]
            combined_roots = [
                task_root
                for task_root in aggregate_roots
                if task_root.name.lower().startswith("all_")
            ]
            if backend_roots and combined_roots:
                logger.info(
                    "Ignoring %d combined aggregate root(s) because backend-specific roots are present",
                    len(combined_roots),
                )
                return backend_roots
        return task_roots

    def _load_metadata(self) -> None:
        task_roots = sorted(path.parent.parent for path in self.dataset_root.rglob("meta/info.json"))
        if not task_roots:
            raise FileNotFoundError(
                f"No LeRobot meta/info.json found below {self.dataset_root}. "
                "HumanoidArena_dataset_v3_1 is a collection of LeRobot task directories."
            )
        task_roots = self._prefer_individual_task_roots(self.dataset_root, task_roots)

        total_samples = 0
        selected_roots = []
        for task_root in task_roots:
            info = json.loads((task_root / "meta/info.json").read_text(encoding="utf-8"))
            protocol = info.get("vla_protocol", {})
            schema = protocol.get("schema")
            if schema and schema != EXPECTED_SCHEMA:
                logger.warning("Skipping incompatible dataset %s with schema=%s", task_root, schema)
                continue
            relative_parts = task_root.relative_to(self.dataset_root).parts
            task_name = relative_parts[0] if relative_parts else task_root.name
            is_aggregate = self._is_aggregate_task_root(self.dataset_root, task_root)
            task_text_by_index = self._load_task_text_by_index(task_root)
            backend = protocol.get("backend_source")
            if backend is None:
                backend = "sonic" if "sonic" in task_root.name.lower() else "twist2"
            backend = self._normalize_backend(backend)
            if is_aggregate:
                if not self._backend_may_be_selected(backend):
                    continue
            elif not self._is_selected(task_name, backend):
                continue
            selected_roots.append(f"{task_name}/{backend}")
            features = info.get("features", {})
            action_spec = features.get("action", {})
            action_shape = tuple(action_spec.get("shape", ()))
            if action_shape and action_shape != (40,):
                logger.warning("Skipping %s because action shape is %s, expected (40,)", task_root, action_shape)
                continue
            video_key = self._choose_video_key(features)
            source_fps = float(info["fps"])
            episode_meta_root = task_root / "meta/episodes"
            if not episode_meta_root.is_dir():
                logger.warning("Skipping %s because meta/episodes is missing", task_root)
                continue

            for meta_file in sorted(episode_meta_root.rglob("*.parquet")):
                metadata = pq.read_table(meta_file).to_pydict()
                for row_index in range(len(metadata.get("episode_index", []))):
                    task_id = self._instruction(metadata["tasks"][row_index])
                    instruction = self._resolve_task_instruction(
                        task_id,
                        task_text_by_index,
                        is_aggregate,
                    )
                    episode_task_name = (
                        TASK_KEY_BY_TASK_ID.get(task_id, task_name)
                        if is_aggregate
                        else task_name
                    )
                    if is_aggregate and not self._is_selected(episode_task_name, backend):
                        continue
                    source_length = int(metadata["length"][row_index])
                    episode_length = int(
                        round((source_length - 1) * self.target_fps / source_fps)
                    ) + 1
                    first_cut = 0
                    last_cut = episode_length - self.action_chunk
                    if last_cut < first_cut:
                        continue
                    sample_count = (last_cut - first_cut) // self.sample_stride + 1

                    video_chunk = int(metadata[f"videos/{video_key}/chunk_index"][row_index])
                    video_file = int(metadata[f"videos/{video_key}/file_index"][row_index])
                    data_chunk = int(metadata["data/chunk_index"][row_index])
                    data_file = int(metadata["data/file_index"][row_index])
                    episode = {
                        "task_id": task_id,
                        "instruction": instruction,
                        "video_path": task_root / "videos" / video_key
                        / f"chunk-{video_chunk:03d}" / f"file-{video_file:03d}.mp4",
                        "video_from_timestamp": float(
                            metadata[f"videos/{video_key}/from_timestamp"][row_index]
                        ),
                        "data_path": task_root / "data" / f"chunk-{data_chunk:03d}"
                        / f"file-{data_file:03d}.parquet",
                        "data_from_index": int(metadata["dataset_from_index"][row_index]),
                        "data_to_index": int(metadata["dataset_to_index"][row_index]),
                        "length": episode_length,
                        "source_fps": source_fps,
                        "target_fps": self.target_fps,
                        "source_length": source_length,
                        "first_cut": first_cut,
                        "sample_count": sample_count,
                        "task_name": episode_task_name,
                        "backend": backend,
                    }
                    if not episode["data_path"].is_file() or not episode["video_path"].is_file():
                        logger.warning("Skipping episode with missing data/video files: %s", episode)
                        continue
                    existing_instruction = self._task_instructions.get(task_id)
                    if (
                        existing_instruction is not None
                        and existing_instruction != instruction
                    ):
                        raise ValueError(
                            f"Task ID {task_id!r} maps to conflicting instructions: "
                            f"{existing_instruction!r} and {instruction!r}"
                        )
                    self._task_instructions[task_id] = instruction
                    self.episodes.append(episode)
                    total_samples += sample_count
                    self._cumulative_samples.append(total_samples)

        if not self.episodes:
            raise RuntimeError(
                f"No compatible HumanoidArena V3.1 episodes found under {self.dataset_root}"
            )
        logger.info(
            "Loaded %d HumanoidArena episodes and %d windows from %s; selected=%s",
            len(self.episodes), total_samples, self.dataset_root, sorted(set(selected_roots)),
        )

    def __len__(self) -> int:
        return self._cumulative_samples[-1]

    def _sample_random_window(self) -> tuple[int, int]:
        episode_index = random.randrange(len(self.episodes))
        episode = self.episodes[episode_index]
        local_index = random.randrange(episode["sample_count"])
        cut = episode["first_cut"] + local_index * self.sample_stride
        return episode_index, cut

    def _load_episode_motion(self, episode_index: int) -> dict[str, torch.Tensor]:
        cached = self._episode_cache.pop(episode_index, None)
        if cached is not None:
            self._episode_cache[episode_index] = cached
            return cached

        episode = self.episodes[episode_index]
        data_path = episode["data_path"]
        cached_file = self._data_file_cache.get(data_path)
        if cached_file is None:
            table = pq.read_table(data_path, columns=["action", "index"]).to_pydict()
            indices = np.asarray(table["index"], dtype=np.int64).reshape(-1)
            actions = np.stack(table["action"]).astype(np.float32, copy=False)
            cached_file = (indices, actions)
            self._data_file_cache[data_path] = cached_file
        indices, actions = cached_file
        file_start = int(indices[0])
        relative_start = episode["data_from_index"] - file_start
        relative_end = episode["data_to_index"] - file_start
        actions = actions[relative_start:relative_end]
        if actions.shape != (episode["source_length"], 40):
            raise ValueError(
                f"Episode action shape mismatch in {episode['data_path']}: "
                f"expected {(episode['source_length'], 40)}, got {actions.shape}"
            )
        decoder, representation = self._get_motion_tools(episode["source_fps"])
        motion_dict = decoder.decode(actions)
        local_rot_mats, root_positions = resample_motion(
            motion_dict["local_rot_mats"],
            motion_dict["root_positions"],
            episode["source_fps"],
            episode["target_fps"],
        )
        motion_features = representation(
            local_rot_mats, root_positions, to_normalize=False
        ).cpu()
        hand_binary = resample_hand_binary(
            actions[:, 38:40],
            episode["source_fps"],
            episode["target_fps"],
        ).cpu()
        if hand_binary.shape[0] != motion_features.shape[0]:
            raise RuntimeError(
                "Motion and hand resampling produced different frame counts: "
                f"{motion_features.shape[0]} and {hand_binary.shape[0]}"
            )
        cached_episode = {
            "motion": motion_features,
            "hand_binary": hand_binary,
        }
        self._episode_cache[episode_index] = cached_episode
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return cached_episode

    def __getitem__(self, index: int) -> dict:
        del index
        episode_index, cut = self._sample_random_window()
        episode = self.episodes[episode_index]
        episode_data = self._load_episode_motion(episode_index)
        motion = episode_data["motion"]
        hand_binary = episode_data["hand_binary"]
        history_start = cut - self.action_history
        future_end = cut + self.action_chunk
        total_length = self.action_history + self.action_chunk
        gt_motion = torch.zeros(total_length, motion.shape[-1], dtype=motion.dtype)
        gt_hand = torch.zeros(total_length, 2, dtype=hand_binary.dtype)
        gt_mask = torch.zeros(total_length, dtype=torch.bool)
        source_start = max(0, history_start)
        destination_start = source_start - history_start
        valid_motion = motion[source_start:future_end]
        valid_hand = hand_binary[source_start:future_end]
        gt_motion[destination_start:destination_start + valid_motion.shape[0]] = valid_motion
        gt_hand[destination_start:destination_start + valid_hand.shape[0]] = valid_hand
        gt_mask[destination_start:destination_start + valid_motion.shape[0]] = True
        egoview = self._read_video_frame(
            episode["video_path"],
            episode["video_from_timestamp"] + cut / episode["target_fps"],
        )
        sample = {
            "instruction": episode["instruction"],
            "egoview": egoview,
            "gt_motion": gt_motion,
            "gt_hand": gt_hand,
            "gt_mask": gt_mask,
            "episode_index": episode_index,
            "cut_index": cut,
        }
        if self._text_embeddings:
            sample["text_embedding"] = self._text_embeddings[episode["task_id"]]
            sample["text_length"] = torch.tensor(
                sample["text_embedding"].shape[0], dtype=torch.long
            )
        return sample

    def _read_video_frame(self, video_path: Path, timestamp: float) -> torch.Tensor:
        for attempt in range(_VIDEO_READ_MAX_ATTEMPTS):
            try:
                with av.open(str(video_path)) as container:
                    stream = container.streams.video[0]
                    # FFmpeg's automatic thread count follows the host CPU count.
                    # On large servers, opening one decoder per sample can otherwise
                    # create hundreds of threads in every DataLoader worker.
                    stream.codec_context.thread_count = 1
                    container.seek(max(0, int(timestamp * av.time_base)))
                    selected = None
                    for frame in container.decode(stream):
                        selected = frame
                        frame_time = float(frame.pts * stream.time_base) if frame.pts is not None else timestamp
                        if frame_time + 1e-6 >= timestamp:
                            break
                    if selected is None:
                        raise RuntimeError(f"Could not decode frame at {timestamp:.3f}s from {video_path}")
                    # Keep the decoded RGB frame untouched. DINOv3Encoder owns all
                    # resize/rescale/normalize operations through the checkpoint's
                    # official AutoImageProcessor.
                    return (
                        torch.from_numpy(selected.to_ndarray(format="rgb24"))
                        .permute(2, 0, 1)
                        .contiguous()
                    )
            except av.error.BlockingIOError as error:
                if attempt == _VIDEO_READ_MAX_ATTEMPTS - 1:
                    raise RuntimeError(
                        f"PyAV repeatedly could not decode frame at {timestamp:.3f}s from {video_path}"
                    ) from error
                delay = _VIDEO_READ_RETRY_DELAY_SECONDS * (2 ** attempt)
                logger.warning(
                    "Transient PyAV video-read failure for %s at %.3fs; retrying (%d/%d) in %.2fs",
                    video_path,
                    timestamp,
                    attempt + 1,
                    _VIDEO_READ_MAX_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)
