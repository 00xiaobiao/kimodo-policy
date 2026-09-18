"""Read official SIMPLE LeRobot v2.1 episodes directly, without replay/export."""

from scipy.spatial.transform import Rotation

from .common import *  # noqa: F401,F403
from .simple_hand import project_hand_closure, source_action_hand_targets


SIMPLE_COLUMNS = (
    "observation.leg_joints",
    "observation.arm_joints",
    "observation.hand_joints",
    "action",
    "task_index",
)


def reference_root_from_source_action(source_action: np.ndarray, fps: float) -> dict:
    """Reconstruct commanded root motion, not measured floating-base poses.

    Keep the existing training convention: first displacement zero, subsequent
    displacements from the current velocity command, yaw relative to frame zero.
    """
    source_action = _as_matrix(source_action, 36, "Simple source action")
    if not len(source_action):
        raise ValueError("Cannot construct a reference root for an empty episode")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid Simple source fps: {fps}")
    delta = np.zeros((len(source_action), 2), dtype=np.float32)
    delta[1:] = source_action[1:, 32:34] / float(fps)
    yaw = np.unwrap(source_action[:, 35].astype(np.float64))
    yaw -= yaw[0]
    rotations = Rotation.from_euler("z", yaw).as_matrix().astype(np.float32)
    return {
        "local_xy_delta": delta,
        "height": source_action[:, 31].copy(),
        "rotation_matrices": rotations,
    }


def simple_training_arrays(table: Mapping, fps: float, hand_control_mode: str) -> dict:
    """Convert recorded joints and commands in memory without a simulator.

    Continuous mode retains command targets for arms/waist/hands, recorded
    lower-body targets and two hand closure scalars. Binary mode retains
    recorded body targets and command-threshold hand labels.
    """
    leg = _as_matrix(table["observation.leg_joints"], 15, "Simple leg joints")
    arm = _as_matrix(table["observation.arm_joints"], 14, "Simple arm joints")
    hand = _as_matrix(table["observation.hand_joints"], 14, "Simple hand joints")
    command = _as_matrix(table["action"], 36, "Simple source action")
    if not len(command) or len({len(leg), len(arm), len(hand), len(command)}) != 1:
        raise ValueError("Simple observation/action lengths must be equal and nonzero")
    indices = [
        UNITREE_G1_JOINT_NAMES_29.index(name)
        for name in CANONICAL_G1_JOINT_NAMES_29
    ]
    observed_body = np.concatenate((leg, arm), axis=1)[:, indices]
    reference = reference_root_from_source_action(command, fps)
    rot6d = reference["rotation_matrices"][:, :, :2].reshape(-1, 6)
    binary_hand = (
        np.abs(command[:, :14]).reshape(-1, 2, 7).max(axis=2) > 0.10
    ).astype(np.float32)
    target_body = observed_body.copy()
    if hand_control_mode == "continuous":
        canonical = {name: i for i, name in enumerate(CANONICAL_G1_JOINT_NAMES_29)}
        for source_index, name in enumerate(
            UNITREE_G1_JOINT_NAMES_29[15:29], start=14
        ):
            target_body[:, canonical[name]] = command[:, source_index]
        for name, index in (
            ("waist_yaw_joint", 30),
            ("waist_roll_joint", 28),
            ("waist_pitch_joint", 29),
        ):
            target_body[:, canonical[name]] = command[:, index]
        observed_hand = project_hand_closure(hand, name="Simple observed hand q")
        target_hand = project_hand_closure(
            source_action_hand_targets(command), name="Simple source hand target"
        )
    elif hand_control_mode == "binary":
        observed_hand = binary_hand
        target_hand = binary_hand
    else:
        raise ValueError(f"Unknown Simple hand_control_mode: {hand_control_mode!r}")
    actions = np.concatenate(
        (reference["local_xy_delta"], reference["height"][:, None],
         rot6d, target_body, binary_hand),
        axis=1,
    ).astype(np.float32)
    return {
        "observed_body": observed_body,
        "root_rot6d": rot6d,
        "target_actions": actions,
        "observed_hand": observed_hand,
        "target_hand": target_hand,
    }


class SimpleAdapter(BaseSourceAdapter):
    """Load <root>/<task>/{meta,data,videos} from official SIMPLE archives."""

    source_name = SOURCE_SIMPLE

    @property
    def terminal_hold_frames(self) -> int:
        return max(0, self.action_chunk - 1)

    @property
    def hand_control_mode(self) -> str:
        mode = str(self.selection.get("hand_control_mode", "binary")).lower()
        if mode not in {"binary", "continuous"}:
            raise ValueError(
                f"Simple hand_control_mode must be binary or continuous, got {mode!r}"
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
        raise TypeError("Simple task selection must be a string or sequence")

    def _selected_task_roots(self) -> list[Path]:
        if "task" in self.selection and "tasks" in self.selection:
            raise ValueError("Simple selection accepts either task or tasks, not both")
        patterns = self._patterns(
            self.selection.get("task", self.selection.get("tasks"))
        )
        roots = [
            p for p in sorted(self.root.iterdir())
            if p.is_dir() and not p.name.startswith(".")
        ]
        selected = [
            p for p in roots
            if any(fnmatch.fnmatch(p.name.lower(), pattern.lower()) for pattern in patterns)
        ]
        if not selected:
            raise RuntimeError(f"No Simple task matched {patterns!r} below {self.root}")
        return selected

    @staticmethod
    def _jsonl(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    @staticmethod
    def _source_path(root: Path, template: str, **values) -> Path:
        relative = Path(template.format(**values))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Simple metadata path escapes task directory: {relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing official Simple data/video: {path}")
        return path

    def discover(self) -> None:
        _ = self.hand_control_mode
        for task_root in self._selected_task_roots():
            info_path = task_root / "meta/info.json"
            if not info_path.is_file():
                raise FileNotFoundError(
                    f"Expected official SIMPLE meta/info.json at {info_path}; "
                    "extract the official task archive directly"
                )
            info = json.loads(info_path.read_text())
            fps = float(info["fps"])
            chunk_size = int(info["chunks_size"])
            if not np.isfinite(fps) or fps <= 0 or chunk_size <= 0:
                raise ValueError(f"Invalid Simple fps/chunks_size: {info_path}")
            features = info["features"]
            camera = str(self.selection.get("camera", "observation.images.egocentric"))
            if camera == "observation.images.front":
                camera = "observation.images.egocentric"
            if features.get(camera, {}).get("dtype") != "video":
                raise ValueError(f"Simple camera {camera!r} is not a video in {info_path}")
            rows = self._jsonl(task_root / "meta/tasks.jsonl")
            catalog = {
                int(row["task_index"]): str(row["task"]).strip() for row in rows
            }
            if (not catalog or len(catalog) != len(rows)
                    or any(not text for text in catalog.values())):
                raise ValueError(f"Invalid or duplicate Simple task metadata: {task_root}")
            episodes = self._jsonl(task_root / "meta/episodes.jsonl")
            if len(episodes) != int(info["total_episodes"]):
                raise ValueError(f"Simple episode count differs from info.json: {task_root}")
            seen = set()
            for metadata in sorted(
                episodes, key=lambda row: int(row["episode_index"])
            ):
                index = int(metadata["episode_index"])
                length = int(metadata["length"])
                if index < 0 or index in seen or length <= 0:
                    raise ValueError(
                        f"Invalid or duplicate Simple episode {index}: {task_root}"
                    )
                seen.add(index)
                task_indices = metadata["tasks"]
                if len(task_indices) != 1 or int(task_indices[0]) not in catalog:
                    raise ValueError(
                        f"Simple episode {index} must identify one known instruction"
                    )
                task_index = int(task_indices[0])
                values = dict(
                    episode_index=index,
                    episode_chunk=index // chunk_size,
                    video_key=camera,
                )
                data_path = self._source_path(task_root, info["data_path"], **values)
                video_path = self._source_path(task_root, info["video_path"], **values)
                parquet = pq.ParquetFile(data_path)
                missing = set(SIMPLE_COLUMNS) - set(parquet.schema_arrow.names)
                if missing or parquet.metadata.num_rows != length:
                    raise ValueError(
                        f"Invalid Simple parquet {data_path}: missing={sorted(missing)}, "
                        f"expected_rows={length}, actual_rows={parquet.metadata.num_rows}"
                    )
                task_id = f"{self.source_name}::{task_root.name}"
                if len(catalog) > 1:
                    task_id += f"::{task_index}"
                self._record(
                    task_id=task_id,
                    task_name=task_root.name,
                    instruction=catalog[task_index],
                    episode_id=f"{task_root.name}:{index:06d}",
                    data_path=data_path,
                    source_length=length,
                    source_fps=fps,
                    video_path=video_path,
                    video_from_timestamp=0.0,
                    metadata={"row_start": 0, "row_end": length, "task_index": task_index},
                )

    def load_episode(self, episode: EpisodeRecord) -> dict[str, torch.Tensor]:
        table = self.reader.read(episode, list(SIMPLE_COLUMNS))
        task_indices = np.asarray(table["task_index"]).reshape(-1)
        if (len(task_indices) != episode.source_length
                or not np.all(task_indices == episode.metadata["task_index"])):
            raise ValueError(
                "Simple parquet task_index differs from episode metadata: "
                f"{episode.episode_id}"
            )
        mode = self.hand_control_mode
        arrays = simple_training_arrays(table, episode.source_fps, mode)
        decoder = self._decoder(episode.source_fps)
        observed = decoder.decode_joint_configuration_pose(
            arrays["observed_body"],
            np.zeros((episode.source_length, 3), dtype=np.float32),
            root_rotation_matrices=rot6d_row_to_matrix(
                torch.from_numpy(arrays["root_rot6d"])
            ),
            joint_names=CANONICAL_G1_JOINT_NAMES_29,
        )
        target = decoder.decode_action_pose(arrays["target_actions"])
        return self._finalize_motion(
            episode,
            observed["local_rot_mats"], observed["root_positions"],
            target["local_rot_mats"], target["root_positions"],
            arrays["observed_hand"], arrays["target_hand"],
            np.ones_like(arrays["observed_hand"], dtype=bool),
            np.ones_like(arrays["target_hand"], dtype=bool),
            observed_motion_valid=self._motion_feature_mask(
                "global_root_heading", "global_rot_data"
            ),
            target_motion_source=(
                "source.action upper-body command" if mode == "continuous" else "action"
            ),
            hand_resampling="linear" if mode == "continuous" else "binary",
        )
