"""SIMPLE MuJoCo replay data-source adapter.

The replay exporter writes one self-contained LeRobot-style dataset per
episode under ``<root>/<task>/episode_XXXXXX``.  This adapter intentionally
does not use HumanoidArena task registration or its global index metadata:
SIMPLE task names are its own namespace and each parquet contains exactly one
episode.  It reads the verified 64D/40D ref-pose contract and produces the
same 417D Kimodo tensors as the Arena adapter.
"""

from .common import *  # noqa: F401,F403
from .simple_hand import project_hand_closure, source_action_hand_targets


class SimpleReplayAdapter(BaseSourceAdapter):
    """Load completed SIMPLE replay episodes from ``Simple/<task>/episode_*``."""

    source_name = SOURCE_SIMPLE

    @property
    def hand_control_mode(self) -> str:
        mode = str(getattr(self, "selection", {}).get("hand_control_mode", "binary")).lower()
        if mode not in {"binary", "continuous"}:
            raise ValueError(
                "Simple hand_control_mode must be 'binary' or 'continuous', "
                f"got {mode!r}"
            )
        return mode

    @staticmethod
    def _patterns(value) -> list[str]:
        if value is None:
            return ["*"]
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        raise TypeError("Simple task selection must be a string or a sequence of strings")

    @classmethod
    def _matches_task(cls, task_name: str, patterns) -> bool:
        return any(
            fnmatch.fnmatch(task_name, pattern)
            or fnmatch.fnmatch(task_name.lower(), pattern.lower())
            for pattern in cls._patterns(patterns)
        )

    def _selected_task_roots(self) -> list[Path]:
        if "task" in self.selection and "tasks" in self.selection:
            raise ValueError("Simple selection accepts either 'task' or 'tasks', not both")
        patterns = self.selection.get("task", self.selection.get("tasks"))
        task_roots = [
            path
            for path in sorted(self.root.iterdir())
            if path.is_dir() and not path.name.startswith(".")
        ]
        if not task_roots:
            raise FileNotFoundError(
                f"Simple root {self.root} contains no task directories"
            )
        selected = [
            path for path in task_roots if self._matches_task(path.name, patterns)
        ]
        if not selected:
            raise RuntimeError(
                f"No Simple task matched selection {patterns!r} below {self.root}"
            )
        return selected

    @staticmethod
    def _task_catalog(episode_root: Path) -> dict[int, str]:
        tasks_path = episode_root / "meta/tasks.parquet"
        if not tasks_path.is_file():
            raise FileNotFoundError(f"Simple episode is missing task metadata: {tasks_path}")
        table = pq.read_table(tasks_path).to_pydict()
        try:
            catalog = {
                int(index): str(task).strip()
                for index, task in zip(table["task_index"], table["task"])
            }
        except KeyError as error:
            raise ValueError(
                f"Simple task metadata {tasks_path} must contain task_index and task"
            ) from error
        if not catalog or any(not text for text in catalog.values()):
            raise ValueError(f"Simple task metadata is empty or malformed: {tasks_path}")
        return catalog

    @staticmethod
    def _episode_metadata(episode_root: Path) -> dict:
        meta_paths = sorted((episode_root / "meta/episodes").glob("*.parquet"))
        if len(meta_paths) != 1:
            raise ValueError(
                f"Simple episode {episode_root} must have exactly one metadata parquet, "
                f"found {len(meta_paths)}"
            )
        metadata = pq.read_table(meta_paths[0]).to_pydict()
        if len(metadata.get("episode_index", ())) != 1:
            raise ValueError(
                f"Simple episode metadata must contain exactly one episode: {meta_paths[0]}"
            )
        return {name: values[0] for name, values in metadata.items()}

    @staticmethod
    def _task_index(metadata: Mapping, episode_root: Path) -> int:
        values = metadata.get("tasks")
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if not isinstance(values, (list, tuple)) or len(values) != 1:
            raise ValueError(
                f"Simple episode metadata must contain one task index: {episode_root}"
            )
        try:
            return int(values[0])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Simple episode task index is not an integer: {episode_root}"
            ) from error

    @staticmethod
    def _completed_length(episode_root: Path) -> int:
        report_path = episode_root / "validation.json"
        if not report_path.is_file():
            raise FileNotFoundError(
                f"Simple episode has not completed replay validation: {report_path}"
            )
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            frames = int(report["frames"])
            source_frames = int(report["source_frames"])
            recorded_frames = int(report["recorded_frames"])
            state_dim = int(report["state_dim"])
            action_dim = int(report["action_dim"])
            kimodo_dim = int(report["kimodo_dim"])
            target_kimodo_dim = int(report["target_kimodo_dim"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Simple episode validation report is malformed: {report_path}"
            ) from error
        if (
            frames <= 0
            # Replays stop when the task terminates successfully.  In that
            # valid case the recorded dataset is a prefix of the source
            # command episode, so source_frames may be larger than frames.
            or source_frames < frames
            or recorded_frames != frames
            or (state_dim, action_dim, kimodo_dim, target_kimodo_dim)
            != (64, 40, 417, 417)
        ):
            raise ValueError(
                f"Simple episode validation report is incompatible: {report_path}"
            )
        return frames

    @staticmethod
    def _episode_directory_index(episode_root: Path) -> int:
        prefix = "episode_"
        if not episode_root.name.startswith(prefix):
            raise ValueError(f"Invalid Simple episode directory name: {episode_root}")
        try:
            return int(episode_root.name[len(prefix) :])
        except ValueError as error:
            raise ValueError(f"Invalid Simple episode directory name: {episode_root}") from error

    def discover(self) -> None:
        for task_root in self._selected_task_roots():
            task_name = task_root.name
            episode_roots = sorted(
                path
                for path in task_root.glob("episode_*")
                if path.is_dir()
            )
            if not episode_roots:
                logger.warning("Skipping Simple task with no episode directories: %s", task_root)
                continue

            for episode_root in episode_roots:
                info_path = episode_root / "meta/info.json"
                if not info_path.is_file():
                    logger.warning("Skipping incomplete Simple episode without info: %s", episode_root)
                    continue
                if not (episode_root / "validation.json").is_file():
                    logger.warning(
                        "Skipping Simple episode that has not completed validation: %s",
                        episode_root,
                    )
                    continue
                info = json.loads(info_path.read_text(encoding="utf-8"))
                schema = info.get("vla_protocol", {}).get("schema")
                if schema != ARENA_EXPECTED_SCHEMA:
                    raise ValueError(
                        f"Unsupported Simple replay schema {schema!r} in {episode_root}"
                    )
                source_fps = float(info.get("fps", 0.0))
                if not np.isfinite(source_fps) or source_fps <= 0:
                    raise ValueError(f"Simple episode has invalid fps in {info_path}")
                features = info.get("features", {})
                state_shape = tuple(features.get("observation.state", {}).get("shape", ()))
                action_shape = tuple(features.get("action", {}).get("shape", ()))
                if state_shape != (64,) or action_shape != (40,):
                    raise ValueError(
                        f"Simple episode {episode_root} must expose state=(64,) and action=(40,), "
                        f"got {state_shape} and {action_shape}"
                    )
                video_key = _video_key(
                    features, (str(self.selection.get("camera", "observation.images.front")),)
                )
                metadata = self._episode_metadata(episode_root)
                episode_index = int(metadata["episode_index"])
                if episode_index != self._episode_directory_index(episode_root):
                    raise ValueError(
                        f"Simple episode index differs from directory name: {episode_root}"
                    )
                source_length = int(metadata["length"])
                validated_length = self._completed_length(episode_root)
                if source_length != validated_length:
                    raise ValueError(
                        f"Simple metadata length={source_length} differs from validation "
                        f"frames={validated_length}: {episode_root}"
                    )
                task_catalog = self._task_catalog(episode_root)
                task_index = self._task_index(metadata, episode_root)
                if task_index not in task_catalog:
                    raise KeyError(
                        f"Simple episode references unknown task index {task_index}: {episode_root}"
                    )
                instruction = task_catalog[task_index]

                video_chunk_key = f"videos/{video_key}/chunk_index"
                video_file_key = f"videos/{video_key}/file_index"
                video_timestamp_key = f"videos/{video_key}/from_timestamp"
                data_path = episode_root / "data" / (
                    f"chunk-{int(metadata['data/chunk_index']):03d}"
                ) / f"file-{int(metadata['data/file_index']):03d}.parquet"
                video_path = episode_root / "videos" / video_key / (
                    f"chunk-{int(metadata[video_chunk_key]):03d}"
                ) / f"file-{int(metadata[video_file_key]):03d}.mp4"
                if not data_path.is_file() or not video_path.is_file():
                    logger.warning(
                        "Skipping Simple episode with missing data/video: %s %s",
                        data_path,
                        video_path,
                    )
                    continue
                parquet_file = pq.ParquetFile(data_path)
                required_columns = {"observation.state", "action"}
                parquet_columns = set(parquet_file.schema_arrow.names)
                missing_columns = required_columns - parquet_columns
                if missing_columns:
                    raise ValueError(
                        f"Simple episode parquet is missing columns {sorted(missing_columns)}: {data_path}"
                    )
                if parquet_file.metadata.num_rows != source_length:
                    raise ValueError(
                        f"Simple episode parquet rows={parquet_file.metadata.num_rows} differs "
                        f"from metadata length={source_length}: {data_path}"
                    )
                if self.hand_control_mode == "continuous":
                    required_continuous = {"observation.hand_q"}
                    target_sources = {
                        "action.hand_closure",
                        "action.target_hand_q",
                        "source.action",
                    }
                    missing_continuous = required_continuous - parquet_columns
                    if missing_continuous or not (target_sources & parquet_columns):
                        raise ValueError(
                            "Simple continuous hand mode requires observation.hand_q and "
                            "one of action.hand_closure or source.action in "
                            f"{data_path}; missing={sorted(missing_continuous)}"
                        )
                episode_metadata = {"row_start": 0, "row_end": source_length}
                hand_columns = sorted(
                    parquet_columns
                    & {
                        "observation.hand_q",
                        "observation.hand_closure",
                        "action.target_hand_q",
                        "action.hand_closure",
                        "source.action",
                    }
                )
                if hand_columns:
                    episode_metadata["hand_columns"] = hand_columns
                self._record(
                    task_id=f"{self.source_name}::{task_name}",
                    task_name=task_name,
                    instruction=instruction,
                    episode_id=f"{task_name}:{episode_index:06d}",
                    data_path=data_path,
                    source_length=source_length,
                    source_fps=source_fps,
                    video_path=video_path,
                    video_from_timestamp=float(metadata[video_timestamp_key]),
                    # Each replay parquet is one complete episode.  Supplying an
                    # explicit row range deliberately avoids the Arena global-index
                    # convention and its writer-specific inclusive end field.
                    metadata=episode_metadata,
                )

    def load_episode(self, episode: EpisodeRecord) -> dict[str, torch.Tensor]:
        mode = self.hand_control_mode
        columns = ["observation.state", "action"]
        if mode == "continuous":
            available = set(episode.metadata.get("hand_columns", ()))
            # Manually constructed EpisodeRecords in focused tests have no
            # discovery metadata; their continuous caller must still expose the
            # legacy source/measurement fields.
            if not available:
                available = {"observation.hand_q", "source.action"}
            columns.extend(sorted(available))
        table = self.reader.read(episode, columns)
        state = _as_matrix(table["observation.state"], 64, "Simple observation state")
        actions = _as_matrix(table["action"], 40, "Simple action")
        if mode == "binary":
            observed_hand = actions[:, 38:40]
            target_hand = observed_hand
            if not np.logical_or(target_hand == 0, target_hand == 1).all():
                raise ValueError(
                    f"Simple episode {episode.episode_id} has non-binary hand actions"
                )
            hand_resampling = "binary"
        else:
            if "observation.hand_closure" in table:
                observed_hand = _as_matrix(
                    table["observation.hand_closure"], 2,
                    "Simple observed hand closure",
                )
            else:
                observed_hand = project_hand_closure(
                    _as_matrix(table["observation.hand_q"], 14, "Simple observation hand q"),
                    name="Simple observed hand q",
                )
            if "action.hand_closure" in table:
                target_hand = _as_matrix(
                    table["action.hand_closure"], 2,
                    "Simple target hand closure",
                )
            elif "action.target_hand_q" in table:
                target_hand = project_hand_closure(
                    _as_matrix(table["action.target_hand_q"], 14, "Simple target hand q"),
                    name="Simple target hand q",
                )
            elif "source.action" in table:
                target_hand = project_hand_closure(
                    source_action_hand_targets(
                        _as_matrix(table["source.action"], 36, "Simple source action")
                    ),
                    name="Simple source hand target",
                )
            else:
                raise ValueError(
                    f"Simple episode {episode.episode_id} is missing a continuous hand target"
                )
            for name, values in (
                ("observed", observed_hand),
                ("target", target_hand),
            ):
                if not np.isfinite(values).all() or (
                    (values < -1e-6) | (values > 1.0 + 1e-6)
                ).any():
                    raise ValueError(
                        f"Simple episode {episode.episode_id} has invalid continuous {name} hand closure"
                    )
            observed_hand = np.clip(observed_hand, 0.0, 1.0).astype(np.float32)
            target_hand = np.clip(target_hand, 0.0, 1.0).astype(np.float32)
            hand_resampling = "linear"
        decoder = self._decoder(episode.source_fps)
        observed_root_rotations = rot6d_row_to_matrix(torch.from_numpy(state[:, :6]))
        observed = decoder.decode_joint_configuration_pose(
            state[:, 6:35],
            np.zeros((state.shape[0], 3), dtype=np.float32),
            root_rotation_matrices=observed_root_rotations,
            joint_names=CANONICAL_G1_JOINT_NAMES_29,
        )
        target = decoder.decode_action_pose(actions)
        observed_hand_valid = np.ones_like(observed_hand, dtype=bool)
        target_hand_valid = np.ones_like(target_hand, dtype=bool)
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
            observed_motion_valid=self._motion_feature_mask(
                "global_root_heading", "global_rot_data"
            ),
            target_motion_source="action",
            hand_resampling=hand_resampling,
        )
