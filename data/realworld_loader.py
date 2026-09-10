"""Real-world data-source adapter."""

from .common import *  # noqa: F401,F403


# Native SONIC/WBC hand order is thumb(3), index(2), middle(2) after the
# named fields below are gathered from the packed 43-D vectors. These are the
# Dex3 middle-grip endpoints used by the real-world controller, so closure has
# the same semantics as SIMPLE: zero is open and one is fully closed.
REALWORLD_HAND_JOINT_NAMES_14 = tuple(
    f"{side}_hand_{finger}_{joint_index}_joint"
    for side in ("left", "right")
    for finger, joint_count in (("thumb", 3), ("index", 2), ("middle", 2))
    for joint_index in range(joint_count)
)
REALWORLD_LEFT_HAND_CLOSE = np.asarray(
    [0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5], dtype=np.float32
)
REALWORLD_RIGHT_HAND_CLOSE = -REALWORLD_LEFT_HAND_CLOSE
REALWORLD_HAND_CLOSE_POSES = np.stack(
    (REALWORLD_LEFT_HAND_CLOSE, REALWORLD_RIGHT_HAND_CLOSE), axis=0
)


def _realworld_hand_indices(feature_names, name: str) -> tuple[int, ...]:
    names = tuple(str(value) for value in feature_names)
    if len(names) != 43:
        raise ValueError(f"{name} metadata must contain 43 names, got {len(names)}")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} metadata contains duplicate feature names")
    missing = sorted(set(REALWORLD_HAND_JOINT_NAMES_14) - set(names))
    if missing:
        raise ValueError(f"{name} metadata is missing hand joints: {missing}")
    name_to_index = {value: index for index, value in enumerate(names)}
    return tuple(name_to_index[value] for value in REALWORLD_HAND_JOINT_NAMES_14)


def project_realworld_hand_closure(
    values: np.ndarray,
    feature_names,
    *,
    name: str = "RealWorld hand joints",
) -> np.ndarray:
    """Project named joints from a packed 43-D state/target to two closures."""
    matrix = _as_matrix(values, 43, name)
    indices = _realworld_hand_indices(feature_names, name)
    hand_q = matrix[:, indices].reshape(-1, 2, 7)
    denominators = np.sum(REALWORLD_HAND_CLOSE_POSES**2, axis=1)
    closure = (
        np.sum(hand_q * REALWORLD_HAND_CLOSE_POSES[None], axis=2)
        / denominators[None]
    )
    return np.clip(closure, 0.0, 1.0).astype(np.float32)


class RealWorldAdapter(BaseSourceAdapter):
    """Load the native LeRobot-style real-world export with Arena semantics.

    The real-world export does not use Arena's packed ``observation.state`` and
    ``action`` columns.  This adapter maps the native columns to the same pose
    decoder and Kimodo motion finalization used by ``HumanoidArenaAdapter``.
    In particular, observed root translation is intentionally zero, matching
    Arena's current condition convention; only the measured root orientation is
    used from the observation stream.  ``action.root_p``/``action.root_z`` and
    ``action.root_q`` are used only for the clean target motion.  The exported
    ``action.root_p`` is episode-relative in all three source axes, while
    ``action.root_z`` preserves the absolute pelvis height required by the
    Kimodo root-position representation.
    """

    source_name = SOURCE_REAL_WORLD

    DEFAULT_FIELDS = {
        "observed_joint_field": "observation.joint_q",
        # The action root quaternion is episode-heading-relative.  Use the
        # matching measured state quaternion by default; the raw absolute
        # orientation remains available in the parquet audit fields but must
        # not be mixed into the Kimodo conditioning frame.
        "observed_root_orientation_field": "observation.root_q_relative",
        "observed_hand_field": "observation.hand_binary",
        "observed_hand_state_field": "observation.state",
        "target_joint_field": "action.joint_q",
        "target_root_position_field": "action.root_p",
        "target_root_height_field": "action.root_z",
        "target_root_orientation_field": "action.root_q",
        "target_hand_field": "action.hand_binary",
        "target_hand_wbc_field": "action.wbc",
        "root_target_valid_field": "action.root_target_valid",
        "root_target_discontinuous_field": "action.root_target_discontinuous",
    }

    @property
    def hand_control_mode(self) -> str:
        mode = str(self.selection.get("hand_control_mode", "binary")).lower()
        if mode not in {"binary", "continuous"}:
            raise ValueError(
                "RealWorld hand_control_mode must be 'binary' or 'continuous', "
                f"got {mode!r}"
            )
        return mode

    @property
    def hand_observation_mode(self) -> str:
        mode = str(
            self.selection.get("hand_observation_mode", self.hand_control_mode)
        ).lower()
        if mode not in {"binary", "continuous"}:
            raise ValueError(
                "RealWorld hand_observation_mode must be 'binary' or 'continuous', "
                f"got {mode!r}"
            )
        return mode

    @staticmethod
    def _selection_patterns(value) -> list[str]:
        if value is None:
            return ["*"]
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        return [str(value)]

    @classmethod
    def _matches_patterns(cls, value: str, patterns) -> bool:
        return any(
            fnmatch.fnmatch(str(value), pattern)
            or fnmatch.fnmatch(str(value).lower(), pattern.lower())
            for pattern in cls._selection_patterns(patterns)
        )

    def _field(self, name: str) -> str:
        value = self.selection.get(name, self.DEFAULT_FIELDS[name])
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"RealWorld selection field {name!r} must be a non-empty string")
        return value.strip()

    def _selected_dataset_roots(self) -> list[Path]:
        info_paths = sorted(self.root.rglob("meta/info.json"))
        dataset_roots = [path.parent.parent for path in info_paths]
        if not dataset_roots:
            raise FileNotFoundError(
                f"RealWorld root {self.root} contains no dataset metadata at meta/info.json"
            )

        task_selection = self.selection.get("task")
        dataset_selection = self.selection.get(
            "dataset", self.selection.get("dataset_name", self.selection.get("variant"))
        )
        selected = []
        for dataset_root in dataset_roots:
            task_name = dataset_root.parent.name
            dataset_name = dataset_root.name
            if task_selection is not None and not self._matches_patterns(
                task_name, task_selection
            ):
                continue
            if dataset_selection is not None and not self._matches_patterns(
                dataset_name, dataset_selection
            ):
                continue
            selected.append(dataset_root)
        if not selected:
            raise RuntimeError(
                "No RealWorld dataset matched selection "
                f"task={task_selection!r}, dataset={dataset_selection!r} below {self.root}"
            )
        return selected

    @staticmethod
    def _episode_instruction(entry: Mapping, task_catalog: Mapping[int, str]) -> str:
        values = entry.get("tasks", entry.get("task", []))
        if isinstance(values, (list, tuple)) and values:
            first = values[0]
        else:
            first = values
        if isinstance(first, (int, np.integer)):
            return str(task_catalog.get(int(first), f"task_{int(first)}")).strip()
        instruction = _instruction(values)
        if instruction:
            return instruction
        if task_catalog:
            return next(iter(task_catalog.values()))
        return "unknown task"

    def _matches_episode_selection(
        self, task_name: str, instruction: str
    ) -> bool:
        tasks = self.selection.get("tasks")
        if tasks is None:
            return True
        return _matches_selection({"tasks": tasks}, task_name, instruction)

    def discover(self) -> None:
        hand_mode_fields = {
            "observed_hand_field",
            "target_hand_field",
            "observed_hand_state_field",
            "target_hand_wbc_field",
        }
        field_names = {
            name: self._field(name)
            for name in self.DEFAULT_FIELDS
            if name
            not in {
                "root_target_valid_field",
                "root_target_discontinuous_field",
                *hand_mode_fields,
            }
        }
        observed_hand_field = (
            "observed_hand_state_field"
            if self.hand_observation_mode == "continuous"
            else "observed_hand_field"
        )
        target_hand_field = (
            "target_hand_wbc_field"
            if self.hand_control_mode == "continuous"
            else "target_hand_field"
        )
        selected_hand_fields = (observed_hand_field, target_hand_field)
        field_names.update(
            {name: self._field(name) for name in selected_hand_fields}
        )
        optional_field_names = {
            name: self._field(name)
            for name in ("root_target_valid_field", "root_target_discontinuous_field")
        }

        for dataset_root in self._selected_dataset_roots():
            info_path = dataset_root / "meta/info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            source_fps = float(info.get("fps", 0.0))
            if not np.isfinite(source_fps) or source_fps <= 0:
                raise ValueError(
                    f"RealWorld dataset {dataset_root} has invalid fps={source_fps!r}"
                )
            features = info.get("features", {})
            hand_feature_names = {}
            continuous_hand_fields = []
            if self.hand_observation_mode == "continuous":
                continuous_hand_fields.append(("observed", observed_hand_field))
            if self.hand_control_mode == "continuous":
                continuous_hand_fields.append(("target", target_hand_field))
            for role, field_key in continuous_hand_fields:
                field_name = field_names[field_key]
                feature = features.get(field_name)
                if not isinstance(feature, Mapping):
                    raise ValueError(
                        f"RealWorld dataset {dataset_root} is missing metadata for "
                        f"continuous hand field {field_name!r}"
                    )
                shape = tuple(feature.get("shape", ()))
                if shape != (43,):
                    raise ValueError(
                        f"RealWorld continuous hand field {field_name!r} must have "
                        f"metadata shape [43], got {list(shape)}"
                    )
                names = feature.get("names")
                if not isinstance(names, (list, tuple)):
                    raise ValueError(
                        f"RealWorld continuous hand field {field_name!r} must define names"
                    )
                _realworld_hand_indices(names, field_name)
                hand_feature_names[role] = tuple(str(value) for value in names)
            configured_camera = self.selection.get(
                "camera", "observation.images.ego_view"
            )
            video_key = _video_key(features, (str(configured_camera),))
            video_crop = _stereo_crop(features[video_key], self.selection)
            chunks_size = int(info.get("chunks_size", 1000))
            if chunks_size <= 0:
                raise ValueError(
                    f"RealWorld dataset {dataset_root} has invalid chunks_size={chunks_size}"
                )

            task_catalog: dict[int, str] = {}
            tasks_path = dataset_root / "meta/tasks.jsonl"
            if tasks_path.is_file():
                with tasks_path.open("r", encoding="utf-8") as file:
                    for line in file:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        if "task_index" in entry:
                            task_catalog[int(entry["task_index"])] = str(
                                entry.get("task", entry.get("description", ""))
                            ).strip()

            episodes_path = dataset_root / "meta/episodes.jsonl"
            if not episodes_path.is_file():
                raise FileNotFoundError(
                    f"RealWorld dataset is missing episode metadata: {episodes_path}"
                )
            task_name = dataset_root.parent.name
            dataset_name = dataset_root.name
            data_template = info.get("data_path")
            video_template = info.get("video_path")
            if not data_template or not video_template:
                raise ValueError(
                    f"RealWorld dataset {dataset_root} must define data_path and video_path"
                )

            with episodes_path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    instruction = self._episode_instruction(entry, task_catalog)
                    if not self._matches_episode_selection(task_name, instruction):
                        continue
                    episode_index = int(entry["episode_index"])
                    source_length = int(entry["length"])
                    if source_length <= 0:
                        continue
                    episode_chunk = episode_index // chunks_size
                    data_path = dataset_root / str(data_template).format(
                        episode_chunk=episode_chunk,
                        episode_index=episode_index,
                    )
                    video_path = dataset_root / str(video_template).format(
                        episode_chunk=episode_chunk,
                        episode_index=episode_index,
                        video_key=video_key,
                    )
                    if not data_path.is_file() or not video_path.is_file():
                        logger.warning(
                            "Skipping RealWorld episode %s/%s with missing data/video: %s %s",
                            task_name,
                            episode_index,
                            data_path,
                            video_path,
                        )
                        continue

                    available_columns = set(
                        pq.ParquetFile(data_path).schema_arrow.names
                    )
                    required_columns = set(field_names.values())
                    missing = sorted(required_columns - available_columns)
                    if missing:
                        raise ValueError(
                            f"RealWorld dataset {dataset_root} is missing required columns: {missing}"
                        )
                    optional_columns = tuple(
                        field
                        for field in optional_field_names.values()
                        if field in available_columns
                    )
                    self._record(
                        task_id=f"{self.source_name}::{task_name}",
                        task_name=task_name,
                        instruction=instruction,
                        episode_id=str(episode_index),
                        data_path=data_path,
                        source_length=source_length,
                        source_fps=source_fps,
                        video_path=video_path,
                        video_from_timestamp=0.0,
                        metadata={
                            "row_start": 0,
                            "row_end": source_length,
                            "video_crop": video_crop,
                            "dataset_name": dataset_name,
                            "available_columns": tuple(sorted(available_columns)),
                            "optional_columns": optional_columns,
                            "field_names": field_names,
                            "optional_field_names": optional_field_names,
                            "hand_feature_names": hand_feature_names,
                        },
                    )

    def load_episode(self, episode: EpisodeRecord) -> dict[str, torch.Tensor]:
        field_names = dict(episode.metadata["field_names"])
        optional_field_names = dict(episode.metadata["optional_field_names"])
        available_columns = set(episode.metadata["available_columns"])
        columns = list(field_names.values())
        for field in episode.metadata.get("optional_columns", ()):
            if field not in columns:
                columns.append(field)
        table = self.reader.read(episode, columns)

        observed_q = _as_matrix(
            table[field_names["observed_joint_field"]],
            29,
            "RealWorld observed joint_q",
        )
        target_q = _as_matrix(
            table[field_names["target_joint_field"]],
            29,
            "RealWorld target joint_q",
        )
        observed_root_q = _as_matrix(
            table[field_names["observed_root_orientation_field"]],
            4,
            "RealWorld observed root quaternion",
        )
        target_root_p = _as_matrix(
            table[field_names["target_root_position_field"]],
            3,
            "RealWorld target root position",
        )
        target_root_z = _as_vector(
            table[field_names["target_root_height_field"]],
            "RealWorld target root height",
        )
        target_root_q = _as_matrix(
            table[field_names["target_root_orientation_field"]],
            4,
            "RealWorld target root quaternion",
        )
        hand_feature_names = episode.metadata.get("hand_feature_names", {})
        if self.hand_observation_mode == "continuous":
            observed_hand = project_realworld_hand_closure(
                table[field_names["observed_hand_state_field"]],
                hand_feature_names.get("observed", ()),
                name="RealWorld observed state",
            )
            observed_hand_resampling = "linear"
            observed_hand_description = "hand closure"
        else:
            observed_hand = _as_matrix(
                table[field_names["observed_hand_field"]],
                2,
                "RealWorld observed hand_binary",
            )
            observed_hand_resampling = "binary"
            observed_hand_description = "hand_binary"

        if self.hand_control_mode == "continuous":
            target_hand = project_realworld_hand_closure(
                table[field_names["target_hand_wbc_field"]],
                hand_feature_names.get("target", ()),
                name="RealWorld WBC hand target",
            )
            target_hand_resampling = "linear"
            target_hand_description = "hand closure"
            target_motion_source = (
                "action_joint_q_root_p_root_z_root_q + action.wbc hand closure"
            )
        else:
            target_hand = _as_matrix(
                table[field_names["target_hand_field"]],
                2,
                "RealWorld target hand_binary",
            )
            target_hand_resampling = "binary"
            target_hand_description = "hand_binary"
            target_motion_source = "action_joint_q_root_p_root_z_root_q"

        frame_count = observed_q.shape[0]
        arrays = {
            "target joint_q": target_q,
            "observed root quaternion": observed_root_q,
            "target root position": target_root_p,
            "target root height": target_root_z,
            "target root quaternion": target_root_q,
            f"observed {observed_hand_description}": observed_hand,
            f"target {target_hand_description}": target_hand,
        }
        for name, array in arrays.items():
            if array.shape[0] != frame_count:
                raise ValueError(
                    f"RealWorld episode {episode.episode_id}: {name} has "
                    f"{array.shape[0]} frames, expected {frame_count}"
                )

        valid_field = optional_field_names["root_target_valid_field"]
        if valid_field in available_columns:
            root_valid = _as_vector(table[valid_field], "RealWorld root_target_valid")
            if root_valid.shape[0] != frame_count:
                raise ValueError(
                    f"RealWorld episode {episode.episode_id}: root_target_valid length "
                    f"{root_valid.shape[0]} != {frame_count}"
                )
            if not np.all(root_valid >= 0.5):
                return {
                    "skip_episode": True,
                    "quality_issue": "root target contains invalid frames",
                }

        discontinuous_field = optional_field_names[
            "root_target_discontinuous_field"
        ]
        if discontinuous_field in available_columns:
            discontinuous = _as_vector(
                table[discontinuous_field],
                "RealWorld root_target_discontinuous",
            )
            if discontinuous.shape[0] != frame_count:
                raise ValueError(
                    f"RealWorld episode {episode.episode_id}: root_target_discontinuous "
                    f"length {discontinuous.shape[0]} != {frame_count}"
                )
            if np.any(discontinuous >= 0.5):
                return {
                    "skip_episode": True,
                    "quality_issue": "root target contains discontinuous frames",
                }

        if frame_count < self.action_chunk:
            return {
                "skip_episode": True,
                "quality_issue": "episode is shorter than action_chunk",
            }

        # ``action.root_p`` is an episode-relative GMR qpos trajectory.  Its
        # horizontal components are already in the episode frame, but its z
        # component is only a relative displacement (and is therefore zero at
        # the first frame).  ``action.root_z`` carries the absolute pelvis
        # height; restore it before converting the source xyz pose to Kimodo
        # coordinates.  Dropping this field would train on a pelvis at y=0.
        target_root_p = target_root_p.copy()
        target_root_p[:, 2] = target_root_z

        # Arena observations do not provide root translation.  Keep the same
        # convention here: root position is zero, while the measured root
        # orientation and joint state remain available to the decoder.
        observed_root_p = np.zeros((frame_count, 3), dtype=np.float32)
        decoder = self._decoder(episode.source_fps)
        observed = decoder.decode_joint_configuration_pose(
            observed_q,
            observed_root_p,
            root_quaternions=observed_root_q,
            joint_names=UNITREE_G1_JOINT_NAMES_29,
        )
        target = decoder.decode_joint_configuration_pose(
            target_q,
            target_root_p,
            root_quaternions=target_root_q,
            joint_names=UNITREE_G1_JOINT_NAMES_29,
            planar_origin=target_root_p[0],
        )
        observed_hand_valid = np.ones_like(observed_hand, dtype=bool)
        target_hand_valid = np.ones_like(target_hand, dtype=bool)
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
            target_motion_source=target_motion_source,
            hand_resampling=target_hand_resampling,
            observed_hand_resampling=observed_hand_resampling,
            target_hand_resampling=target_hand_resampling,
        )
