"""Central mixed-source dataset loader.

The source-specific adapters live in sibling modules; this module retains the
original MultiSourceG1Dataset orchestration and sampling implementation.
"""

from .common import *  # noqa: F401,F403
from .domain_randomization import build_domain_randomization
from .humanoidarena_loader import HumanoidArenaAdapter
from .unifolm_loader import UnifoLMAdapter
from .humanoid_everyday_loader import HumanoidEverydayAdapter
from .hiw500_loader import HIW500Adapter
from .realworld_loader import RealWorldAdapter
from .simple_loader import SimpleAdapter
from .humanoidarena_legacy import *  # noqa: F401,F403

ADAPTER_BY_SOURCE = {
    SOURCE_HUMANOID_ARENA: HumanoidArenaAdapter,
    SOURCE_HUMANOID_EVERYDAY: HumanoidEverydayAdapter,
    SOURCE_HIW500: HIW500Adapter,
    SOURCE_UNIFOLM: UnifoLMAdapter,
    SOURCE_REAL_WORLD: RealWorldAdapter,
    SOURCE_SIMPLE: SimpleAdapter,
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
        training: bool = False,
        domain_randomization: Mapping | None = None,
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
        self.training = bool(training)
        self._domain_randomization = (
            build_domain_randomization(domain_randomization) if self.training else None
        )
        if self._domain_randomization is not None:
            logger.info("Training domain randomization enabled: %s", domain_randomization)
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
                last_cut = adapter._last_sample_cut(episode.target_length)
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
        self._motion_cache_dir: Path | None = None
        self._motion_cache_signature: str | None = None
        self._motion_cache_invalid_tokens: set[str] = set()
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
        self._configured_source_weights = dict(sampling.get("source_weights", {}))
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
            configured_weights = self._configured_source_weights
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

    def attach_motion_cache(self, cache_dir: str | Path, signature: str) -> None:
        """Attach a completed pre-training motion cache to this dataset.

        HumanoidArena records are intentionally left untouched.  For the three
        supported pre-training sources, invalid records are removed before the
        sampler is built and valid records are loaded lazily from the cache
        instead of parquet and the source adapters.
        """

        cache_dir = Path(cache_dir).expanduser().resolve()
        manifest = load_motion_cache_manifest(cache_dir, signature)
        entries = manifest["entries"]
        invalid_tokens: set[str] = set()
        filtered_sources: dict[str, list[tuple[BaseSourceAdapter, EpisodeRecord]]] = {}
        removed = 0

        for source_name, records in self._episodes_by_source.items():
            if source_name not in PRETRAIN_MOTION_CACHE_SOURCES:
                filtered_sources[source_name] = records
                continue
            filtered_records: list[tuple[BaseSourceAdapter, EpisodeRecord]] = []
            for adapter, episode in records:
                token = episode_cache_token(episode.cache_key)
                entry = entries.get(token)
                if entry is None:
                    raise RuntimeError(
                        f"Motion cache has no entry for {source_name}/{episode.episode_id}"
                    )
                if entry.get("status") != "valid":
                    invalid_tokens.add(token)
                    removed += 1
                    continue
                filtered_records.append((adapter, episode))
            filtered_sources[source_name] = filtered_records

        self._episodes_by_source = filtered_sources
        self._episodes_by_source_task = {}
        for source_name, records in filtered_sources.items():
            records_by_task: dict[
                str, list[tuple[BaseSourceAdapter, EpisodeRecord]]
            ] = defaultdict(list)
            for record in records:
                records_by_task[record[1].task_id].append(record)
            self._episodes_by_source_task[source_name] = dict(records_by_task)

        self._source_names = [
            source for source in self._source_names if self._episodes_by_source[source]
        ]
        self._all_episode_records = [
            record
            for source in self._source_names
            for record in self._episodes_by_source[source]
        ]
        self._length = sum(episode.sample_count for _, episode in self._all_episode_records)
        if self.sampling_mode in {"source_balanced", "source_task_balanced"}:
            self._source_weights = [
                float(self._configured_source_weights.get(source, 1.0))
                for source in self._source_names
            ]
        elif self.sampling_mode == "window_proportional":
            self._episode_weights = [
                episode.sample_count for _, episode in self._all_episode_records
            ]

        self._motion_cache_dir = cache_dir
        self._motion_cache_signature = str(signature)
        self._motion_cache_invalid_tokens = invalid_tokens
        self._runtime_invalid_episodes.clear()
        logger.info(
            "Attached pre-training motion cache %s: removed_invalid=%d remaining_episodes=%d windows=%d",
            cache_dir,
            removed,
            len(self._all_episode_records),
            self._length,
        )

    def apply_pretrain_data_fraction(self, fraction: float) -> dict[str, int | float]:
        """Keep a deterministic fraction of the discovered pre-training episodes.

        The subset is selected globally across the configured pre-training
        sources.  A fixed internal seed is used after sorting by stable episode
        identity, and the selected records are the prefix of one fixed
        permutation.  Consequently, a 25% subset is nested in a 50% subset and
        repeated runs with the same dataset inventory select the same episodes.

        This method intentionally leaves non-pre-training sources untouched and
        is a no-op when no pre-training source is configured.  The fraction is
        defined over complete episodes; the existing sampler continues to draw
        windows from the selected episodes according to its configured mode.
        """

        if isinstance(fraction, bool):
            raise ValueError("pretrain_data_fraction must be a number in (0, 1]")
        fraction = float(fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"pretrain_data_fraction must be in (0, 1], got {fraction!r}"
            )

        pretrain_sources = set(self.adapters) & set(PRETRAIN_MOTION_CACHE_SOURCES)
        if not pretrain_sources:
            return {
                "fraction": fraction,
                "before_episodes": 0,
                "after_episodes": 0,
                "before_windows": 0,
                "after_windows": 0,
                "applied": 0,
            }

        pretrain_records = [
            record
            for source_name in self._source_names
            if source_name in pretrain_sources
            for record in self._episodes_by_source.get(source_name, [])
        ]
        before_episodes = len(pretrain_records)
        before_windows = sum(episode.sample_count for _, episode in pretrain_records)
        if fraction >= 1.0 or before_episodes == 0:
            return {
                "fraction": fraction,
                "before_episodes": before_episodes,
                "after_episodes": before_episodes,
                "before_windows": before_windows,
                "after_windows": before_windows,
                "applied": 0,
            }

        def stable_record_key(record):
            _, episode = record
            return (
                str(episode.source),
                str(episode.task_id),
                str(episode.task_name),
                str(episode.episode_id),
                tuple(str(value) for value in episode.cache_key),
            )

        ordered_records = sorted(pretrain_records, key=stable_record_key)
        random.Random(_PRETRAIN_DATA_FRACTION_SEED).shuffle(ordered_records)
        selected_count = max(1, int(before_episodes * fraction))
        selected_keys = {
            episode.cache_key for _, episode in ordered_records[:selected_count]
        }

        filtered_by_source = {}
        for source_name, records in self._episodes_by_source.items():
            if source_name not in pretrain_sources:
                filtered_by_source[source_name] = records
                continue
            filtered_by_source[source_name] = [
                record
                for record in records
                if record[1].cache_key in selected_keys
            ]

        self._episodes_by_source = filtered_by_source
        self._episodes_by_source_task = {}
        for source_name, records in filtered_by_source.items():
            records_by_task: dict[
                str, list[tuple[BaseSourceAdapter, EpisodeRecord]]
            ] = defaultdict(list)
            for record in records:
                records_by_task[record[1].task_id].append(record)
            self._episodes_by_source_task[source_name] = dict(records_by_task)

        # Rebuild all sampling tables after replacing the episode inventory.
        self._source_names = [
            source_name
            for source_name in self._source_names
            if self._episodes_by_source.get(source_name)
        ]
        self._all_episode_records = [
            record
            for source_name in self._source_names
            for record in self._episodes_by_source[source_name]
        ]
        self._length = sum(
            episode.sample_count for _, episode in self._all_episode_records
        )
        if self.sampling_mode in {"source_balanced", "source_task_balanced"}:
            self._source_weights = [
                float(self._configured_source_weights.get(source_name, 1.0))
                for source_name in self._source_names
            ]
        else:
            self._source_weights = None
        if self.sampling_mode == "window_proportional":
            self._episode_weights = [
                episode.sample_count for _, episode in self._all_episode_records
            ]
        else:
            self._episode_weights = None
        self._runtime_invalid_episodes.clear()

        after_episodes = len(selected_keys)
        after_windows = sum(
            episode.sample_count
            for _, episode in self._all_episode_records
            if episode.source in pretrain_sources
        )
        logger.info(
            "Applied deterministic pre-training data fraction %.6f: "
            "episodes %d -> %d, windows %d -> %d, seed=%d",
            fraction,
            before_episodes,
            after_episodes,
            before_windows,
            after_windows,
            _PRETRAIN_DATA_FRACTION_SEED,
        )
        return {
            "fraction": fraction,
            "before_episodes": before_episodes,
            "after_episodes": after_episodes,
            "before_windows": before_windows,
            "after_windows": after_windows,
            "applied": 1,
        }

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
            "realworld": SOURCE_REAL_WORLD,
            "real_world": SOURCE_REAL_WORLD,
            "simple": SOURCE_SIMPLE,
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
                "realworld",
                "real_world",
                "simple",
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
        motion_cache_dir = getattr(self, "_motion_cache_dir", None)
        if motion_cache_dir is not None and episode.source in PRETRAIN_MOTION_CACHE_SOURCES:
            token = episode_cache_token(cache_key)
            if token in getattr(self, "_motion_cache_invalid_tokens", set()):
                raise RuntimeError(
                    f"Invalid cached episode reached sampler: {episode.source}/{episode.episode_id}"
                )
            cache_path = (
                motion_cache_dir
                / "episodes"
                / token[:2]
                / f"{token}.pt"
            )
            try:
                payload = torch.load(
                    cache_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:  # torch versions without weights_only
                payload = torch.load(cache_path, map_location="cpu")
            if (
                payload.get("version") != 1
                or payload.get("signature")
                != getattr(self, "_motion_cache_signature", None)
                or payload.get("token") != token
            ):
                raise RuntimeError(f"Incompatible motion cache payload: {cache_path}")
            motion = payload["motion"]
        else:
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

        domain_randomization = getattr(self, "_domain_randomization", None)
        if getattr(self, "training", False) and domain_randomization is not None:
            egoview = domain_randomization(egoview, seed=self.sampling_seed, index=index)

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


# Preserve the complete import surface of the former monolithic module.
__all__ = [
    name
    for name in globals()
    if name != "__builtins__" and not name.startswith("__")
]
