"""UnifoLM WBT data-source adapter."""

from .common import *  # noqa: F401,F403

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
