"""Offline motion-cache helpers for the mixed pre-training datasets.

The cache intentionally stores episode-level motion tensors, not sampled
windows.  Window sampling remains deterministic and configurable at training
time while parquet decoding, motion conversion, and quality filtering happen
only once per cache build.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import torch

logger = logging.getLogger(__name__)

MOTION_CACHE_VERSION = 1
PRETRAIN_MOTION_CACHE_SOURCES = frozenset(
    {
        "UnifoLM_WBT_Dataset",
        "HumanoidEveryday",
        "HIW500",
    }
)


def episode_cache_token(cache_key: tuple[Any, ...]) -> str:
    """Return a stable filename token for an :class:`EpisodeRecord` key."""

    serialized = json.dumps(
        [str(value) for value in cache_key],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def motion_cache_signature(payload: Mapping[str, Any]) -> str:
    """Hash all preprocessing settings that affect cached motion tensors."""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def _episode_path(cache_dir: Path, token: str) -> Path:
    return cache_dir / "episodes" / token[:2] / f"{token}.pt"


def _valid_manifest(cache_dir: Path, signature: str) -> dict[str, Any] | None:
    manifest_path = _manifest_path(cache_dir)
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        manifest.get("version") != MOTION_CACHE_VERSION
        or manifest.get("signature") != signature
        or not isinstance(manifest.get("entries"), dict)
    ):
        return None
    stats = manifest.get("stats", {})
    entries = manifest["entries"]
    if (
        int(stats.get("error", 0)) != 0
        or int(stats.get("total", -1)) != len(entries)
        or int(stats.get("valid", 0)) + int(stats.get("invalid", 0))
        != int(stats.get("total", -1))
    ):
        return None
    for token, entry in manifest["entries"].items():
        if entry.get("status") != "valid":
            if entry.get("status") != "invalid":
                return None
            continue
        payload_path = _episode_path(cache_dir, token)
        if not payload_path.is_file():
            return None
        expected_size = entry.get("size_bytes")
        if expected_size is None or payload_path.stat().st_size != int(expected_size):
            return None
    return manifest


def _write_tensor_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_manifest(cache_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path = _manifest_path(cache_dir)
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    try:
        temporary_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary_path, manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _remove_stale_payloads(cache_dir: Path, valid_tokens: set[str]) -> None:
    episodes_root = cache_dir / "episodes"
    if not episodes_root.is_dir():
        return
    for payload_path in episodes_root.rglob("*.pt"):
        if payload_path.stem not in valid_tokens:
            payload_path.unlink(missing_ok=True)


def _worker_adapter_copy(adapter):
    """Make a small, process-safe adapter copy for one cache worker.

    Episode discovery metadata is already held by the parent dataset and is
    not needed by ``load_episode``.  Dropping it keeps the spawn payload small;
    the reader and lazy motion-representation caches must be process-local.
    """

    worker_adapter = copy.copy(adapter)
    if hasattr(worker_adapter, "episodes"):
        worker_adapter.episodes = []
    if hasattr(worker_adapter, "reader"):
        worker_adapter.reader = type(adapter.reader)()
    if hasattr(worker_adapter, "_decoders"):
        worker_adapter._decoders = {}
    if hasattr(worker_adapter, "_representation"):
        worker_adapter._representation = None
    return worker_adapter


_CACHE_WORKER_ADAPTERS = None
_CACHE_WORKER_CACHE_DIR = None
_CACHE_WORKER_SIGNATURE = None


def _initialize_cache_worker(adapters, cache_dir: str, signature: str) -> None:
    global _CACHE_WORKER_ADAPTERS, _CACHE_WORKER_CACHE_DIR, _CACHE_WORKER_SIGNATURE
    _CACHE_WORKER_ADAPTERS = adapters
    _CACHE_WORKER_CACHE_DIR = Path(cache_dir)
    _CACHE_WORKER_SIGNATURE = str(signature)
    # Each episode job is independent.  Prevent NumPy/PyTorch kernels in 24
    # workers per rank from multiplying into another layer of CPU threads.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _cache_episode_result(
    adapter,
    source_name: str,
    episode,
    cache_dir: Path,
    signature: str,
) -> tuple[str, dict[str, Any], str | None]:
    token = episode_cache_token(episode.cache_key)
    entry: dict[str, Any] = {
        "source": source_name,
        "episode_id": episode.episode_id,
        "key": [str(value) for value in episode.cache_key],
    }
    try:
        motion = adapter.load_episode(episode)
        if motion.get("skip_episode", False):
            entry["status"] = "invalid"
            entry["reason"] = str(motion.get("quality_issue", "invalid episode"))
            return token, entry, None

        tensor_motion = {
            key: value.cpu() if torch.is_tensor(value) else value
            for key, value in motion.items()
        }
        payload_path = _episode_path(cache_dir, token)
        _write_tensor_payload(
            payload_path,
            {
                "version": MOTION_CACHE_VERSION,
                "signature": signature,
                "token": token,
                "motion": tensor_motion,
            },
        )
        entry["status"] = "valid"
        entry["size_bytes"] = payload_path.stat().st_size
        return token, entry, None
    except Exception as error:  # fail before training, not mid-epoch
        entry["status"] = "error"
        entry["reason"] = f"{type(error).__name__}: {error}"
        return token, entry, f"{source_name}/{episode.episode_id}: {error}"


def _cache_worker_job(job) -> tuple[str, dict[str, Any], str | None]:
    source_name, episode = job
    return _cache_episode_result(
        _CACHE_WORKER_ADAPTERS[source_name],
        source_name,
        episode,
        _CACHE_WORKER_CACHE_DIR,
        _CACHE_WORKER_SIGNATURE,
    )


def _empty_cache_stats() -> dict[str, int]:
    return {"valid": 0, "invalid": 0, "error": 0, "total": 0}


def _accumulate_cache_result(
    entries: dict[str, dict[str, Any]],
    stats: dict[str, int],
    errors: list[str],
    result: tuple[str, dict[str, Any], str | None],
) -> None:
    token, entry, error = result
    entries[token] = entry
    stats["total"] += 1
    status = entry.get("status")
    if status == "valid":
        stats["valid"] += 1
    elif status == "invalid":
        stats["invalid"] += 1
    else:
        stats["error"] += 1
        errors.append(error or f"{entry.get('source')}/{entry.get('episode_id')}")


def _write_cache_shard(cache_dir: Path, rank: int, payload: dict[str, Any]) -> None:
    shard_dir = cache_dir / ".shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"rank_{int(rank)}.json"
    temporary_path = shard_path.with_name(f".{shard_path.name}.tmp-{os.getpid()}")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary_path, shard_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def prepare_motion_cache(
    dataset,
    cache_root: str | Path,
    signature: str,
    *,
    force_rebuild: bool = False,
    rank: int = 0,
    world_size: int = 1,
    workers_per_rank: int = 1,
) -> Path:
    """Build or reuse the configured pre-training episode motion cache.

    ``dataset`` is a freshly constructed ``MultiSourceG1Dataset``.  Only the
    three non-Arena pre-training sources are processed.  Invalid episodes are
    recorded in the manifest and intentionally receive no tensor file, so the
    training loader can remove them before sampling starts.
    """

    # The caller names this directory from the pre-training YAML stem.  The
    # manifest signature decides whether its contents are reusable or must be
    # rebuilt in place after that YAML's preprocessing settings change.
    cache_dir = Path(cache_root).expanduser().resolve()
    rank = int(rank)
    world_size = max(1, int(world_size))
    workers_per_rank = max(1, int(workers_per_rank))
    distributed = bool(
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if distributed:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} must be in [0, {world_size})")
    expected_tokens = {
        episode_cache_token(episode.cache_key)
        for source_name in dataset._source_names
        if source_name in PRETRAIN_MOTION_CACHE_SOURCES
        for _, episode in dataset._episodes_by_source.get(source_name, [])
    }
    reuse = False
    if rank == 0 and not force_rebuild:
        existing = _valid_manifest(cache_dir, signature)
        if existing is not None and set(existing["entries"]) == expected_tokens:
            logger.info(
                "Reusing motion cache %s (valid=%d invalid=%d error=%d)",
                cache_dir,
                existing.get("stats", {}).get("valid", 0),
                existing.get("stats", {}).get("invalid", 0),
                existing.get("stats", {}).get("error", 0),
            )
            reuse = True
        elif existing is not None:
            logger.info(
                "Motion cache episode inventory changed (%d cached vs %d selected); rebuilding %s",
                len(existing["entries"]),
                len(expected_tokens),
                cache_dir,
            )

    if distributed:
        reuse_flag = [reuse]
        torch.distributed.broadcast_object_list(reuse_flag, src=0)
        reuse = bool(reuse_flag[0])
    if reuse:
        return cache_dir

    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Never leave a previous manifest around when rebuilding with changed
        # source selection or data.  Payloads are pruned after a successful
        # rebuild so an interrupted build remains safely recoverable.
        stale_manifest = _manifest_path(cache_dir)
        if stale_manifest.exists():
            stale_manifest.unlink()
        (cache_dir / ".shards").mkdir(parents=True, exist_ok=True)
    if distributed:
        torch.distributed.barrier()
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[tuple[str, Any, Any]] = []
    cached_sources: list[str] = []
    for source_name in dataset._source_names:
        if source_name not in PRETRAIN_MOTION_CACHE_SOURCES:
            continue
        records = dataset._episodes_by_source.get(source_name, [])
        adapter = dataset.adapters.get(source_name)
        if adapter is None:
            continue
        cached_sources.append(source_name)
        all_records.extend(
            (source_name, adapter, episode) for _, episode in records
        )
    shard_records = [
        record
        for index, record in enumerate(all_records)
        if index % world_size == rank
    ]
    entries: dict[str, dict[str, Any]] = {}
    stats = _empty_cache_stats()
    errors: list[str] = []
    logger.info(
        "Motion cache rank %d/%d: %d episodes with up to %d CPU workers",
        rank,
        world_size,
        len(shard_records),
        workers_per_rank,
    )
    source_counts: dict[str, int] = {}
    for source_name, _, _ in shard_records:
        source_counts[source_name] = source_counts.get(source_name, 0) + 1
    for source_name, source_count in source_counts.items():
        logger.info(
            "Motion cache source=%s rank=%d episodes=%d",
            source_name,
            rank,
            source_count,
        )
    jobs = [(source_name, episode) for source_name, _, episode in shard_records]
    if workers_per_rank <= 1 or len(jobs) <= 1:
        for processed, (source_name, adapter, episode) in enumerate(
            shard_records, start=1
        ):
            result = _cache_episode_result(
                adapter, source_name, episode, cache_dir, signature
            )
            _accumulate_cache_result(entries, stats, errors, result)
            if processed % 100 == 0:
                logger.info(
                    "Motion cache rank %d/%d progress: %d/%d",
                    rank,
                    world_size,
                    processed,
                    len(shard_records),
                )
    elif jobs:
        worker_adapters = {
            source_name: _worker_adapter_copy(dataset.adapters[source_name])
            for source_name in source_counts
        }
        try:
            with ProcessPoolExecutor(
                max_workers=min(workers_per_rank, len(jobs)),
                mp_context=mp.get_context("spawn"),
                initializer=_initialize_cache_worker,
                initargs=(worker_adapters, str(cache_dir), signature),
            ) as pool:
                futures = [pool.submit(_cache_worker_job, job) for job in jobs]
                for processed, (job, future) in enumerate(
                    zip(jobs, futures), start=1
                ):
                    source_name, failed_episode = job
                    try:
                        result = future.result()
                    except Exception as error:
                        failed_token = episode_cache_token(failed_episode.cache_key)
                        result = (
                            failed_token,
                            {
                                "source": source_name,
                                "episode_id": failed_episode.episode_id,
                                "key": [
                                    str(value) for value in failed_episode.cache_key
                                ],
                                "status": "error",
                                "reason": f"{type(error).__name__}: {error}",
                            },
                            f"{source_name}/{failed_episode.episode_id}: {error}",
                        )
                    _accumulate_cache_result(entries, stats, errors, result)
                    if processed % 100 == 0:
                        logger.info(
                            "Motion cache rank %d/%d progress: %d/%d",
                            rank,
                            world_size,
                            processed,
                            len(shard_records),
                        )
        except Exception as error:
            errors.append(f"worker-pool: {type(error).__name__}: {error}")

    local_payload = {"entries": entries, "stats": stats, "errors": errors}
    if distributed:
        _write_cache_shard(cache_dir, rank, local_payload)
        torch.distributed.barrier()

    result = [None, None]
    if rank == 0:
        try:
            gathered_payloads = [local_payload]
            if distributed:
                gathered_payloads = []
                for shard_rank in range(world_size):
                    shard_path = cache_dir / ".shards" / f"rank_{shard_rank}.json"
                    if not shard_path.is_file():
                        raise RuntimeError(f"Missing motion-cache shard: {shard_path}")
                    gathered_payloads.append(
                        json.loads(shard_path.read_text(encoding="utf-8"))
                    )
            combined_entries: dict[str, dict[str, Any]] = {}
            combined_stats = _empty_cache_stats()
            combined_errors: list[str] = []
            for payload in gathered_payloads:
                combined_entries.update(payload["entries"])
                for key in combined_stats:
                    combined_stats[key] += int(payload["stats"].get(key, 0))
                combined_errors.extend(payload.get("errors", []))
            if set(combined_entries) != expected_tokens:
                missing = expected_tokens - set(combined_entries)
                extra = set(combined_entries) - expected_tokens
                raise RuntimeError(
                    "Motion cache shard inventory mismatch: "
                    f"missing={len(missing)} extra={len(extra)}"
                )
            manifest = {
                "version": MOTION_CACHE_VERSION,
                "signature": signature,
                "sources": cached_sources,
                "stats": combined_stats,
                "entries": combined_entries,
            }
            _write_manifest(cache_dir, manifest)
            if not combined_errors:
                _remove_stale_payloads(
                    cache_dir,
                    {
                        token
                        for token, entry in combined_entries.items()
                        if entry.get("status") == "valid"
                    },
                )
            if combined_errors:
                preview = "\n  ".join(combined_errors[:10])
                suffix = (
                    ""
                    if len(combined_errors) <= 10
                    else f"\n  ... and {len(combined_errors) - 10} more"
                )
                raise RuntimeError(
                    "Motion cache encountered read/conversion errors; training was not started:\n  "
                    + preview
                    + suffix
                )
            logger.info(
                "Motion cache ready: %s valid=%d invalid=%d error=%d workers=%d",
                cache_dir,
                combined_stats["valid"],
                combined_stats["invalid"],
                combined_stats["error"],
                workers_per_rank * world_size,
            )
            result[0] = str(cache_dir)
        except Exception as error:
            result[1] = f"{type(error).__name__}: {error}"
        finally:
            if distributed:
                shutil.rmtree(cache_dir / ".shards", ignore_errors=True)
    if distributed:
        torch.distributed.broadcast_object_list(result, src=0)
    if result[1] is not None:
        raise RuntimeError(result[1])
    return Path(result[0] or cache_dir)


def load_motion_cache_manifest(cache_dir: str | Path, signature: str) -> dict[str, Any]:
    """Load and validate a completed motion-cache manifest."""

    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest = _valid_manifest(cache_dir, signature)
    if manifest is None:
        raise RuntimeError(
            f"Motion cache is missing, incomplete, or incompatible: {cache_dir}"
        )
    if manifest.get("stats", {}).get("error", 0):
        raise RuntimeError(f"Motion cache contains read errors: {cache_dir}")
    return manifest


def cache_payload_path(cache_dir: str | Path, token: str) -> Path:
    """Return the episode payload path used by the Dataset loader."""

    return _episode_path(Path(cache_dir).expanduser().resolve(), token)
