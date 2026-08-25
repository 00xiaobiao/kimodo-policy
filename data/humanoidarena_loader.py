"""HumanoidArena data-source adapter."""

from .common import *  # noqa: F401,F403

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
