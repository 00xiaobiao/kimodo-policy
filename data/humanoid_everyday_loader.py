"""HumanoidEveryday data-source adapter."""

from .common import *  # noqa: F401,F403

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
