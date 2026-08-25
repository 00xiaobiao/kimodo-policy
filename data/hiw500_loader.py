"""HIW-500 data-source adapter."""

from .common import *  # noqa: F401,F403

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
