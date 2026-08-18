"""Offline motion-cache helpers for the mixed pre-training datasets.

The cache intentionally stores episode-level motion tensors, not sampled
windows.  Window sampling remains deterministic and configurable at training
time while parquet decoding, motion conversion, and quality filtering happen
only once per cache build.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
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


def prepare_motion_cache(
    dataset,
    cache_root: str | Path,
    signature: str,
    *,
    force_rebuild: bool = False,
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
    expected_tokens = {
        episode_cache_token(episode.cache_key)
        for source_name in dataset._source_names
        if source_name in PRETRAIN_MOTION_CACHE_SOURCES
        for _, episode in dataset._episodes_by_source.get(source_name, [])
    }
    if not force_rebuild:
        existing = _valid_manifest(cache_dir, signature)
        if existing is not None and set(existing["entries"]) == expected_tokens:
            logger.info(
                "Reusing motion cache %s (valid=%d invalid=%d error=%d)",
                cache_dir,
                existing.get("stats", {}).get("valid", 0),
                existing.get("stats", {}).get("invalid", 0),
                existing.get("stats", {}).get("error", 0),
            )
            return cache_dir
        if existing is not None:
            logger.info(
                "Motion cache episode inventory changed (%d cached vs %d selected); rebuilding %s",
                len(existing["entries"]),
                len(expected_tokens),
                cache_dir,
            )

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Never leave a previous manifest's entries around when rebuilding with
    # changed source selection or data.  The signature directory normally
    # isolates this, but force_rebuild must also be safe in-place.
    stale_manifest = _manifest_path(cache_dir)
    if stale_manifest.exists():
        stale_manifest.unlink()
    entries: dict[str, dict[str, Any]] = {}
    stats = {"valid": 0, "invalid": 0, "error": 0, "total": 0}
    errors: list[str] = []
    processed = 0
    cached_sources: list[str] = []
    expected_total = sum(
        len(dataset._episodes_by_source.get(source_name, []))
        for source_name in dataset._source_names
        if source_name in PRETRAIN_MOTION_CACHE_SOURCES
    )

    # Iterate the dataset's configured source order.  This both avoids
    # touching HumanoidArena and ensures only selected pre-training sources
    # are materialized in the cache manifest.
    for source_name in dataset._source_names:
        if source_name not in PRETRAIN_MOTION_CACHE_SOURCES:
            continue
        records = dataset._episodes_by_source.get(source_name, [])
        adapter = dataset.adapters.get(source_name)
        if adapter is None:
            continue
        cached_sources.append(source_name)
        logger.info("Building motion cache source=%s episodes=%d", source_name, len(records))
        for adapter_record, episode in records:
            del adapter_record  # the tuple carries the same adapter instance
            token = episode_cache_token(episode.cache_key)
            key_payload = [str(value) for value in episode.cache_key]
            entry: dict[str, Any] = {
                "source": source_name,
                "episode_id": episode.episode_id,
                "key": key_payload,
            }
            stats["total"] += 1
            try:
                motion = adapter.load_episode(episode)
                if motion.get("skip_episode", False):
                    entry["status"] = "invalid"
                    entry["reason"] = str(motion.get("quality_issue", "invalid episode"))
                    stats["invalid"] += 1
                else:
                    tensor_motion = {
                        key: value.cpu() if torch.is_tensor(value) else value
                        for key, value in motion.items()
                    }
                    _write_tensor_payload(
                        payload_path := _episode_path(cache_dir, token),
                        {
                            "version": MOTION_CACHE_VERSION,
                            "signature": signature,
                            "token": token,
                            "motion": tensor_motion,
                        },
                    )
                    entry["status"] = "valid"
                    entry["size_bytes"] = payload_path.stat().st_size
                    stats["valid"] += 1
            except Exception as error:  # fail before training, not mid-epoch
                entry["status"] = "error"
                entry["reason"] = f"{type(error).__name__}: {error}"
                stats["error"] += 1
                errors.append(f"{source_name}/{episode.episode_id}: {error}")
            entries[token] = entry
            processed += 1
            if processed % 100 == 0:
                logger.info(
                    "Motion cache progress: %d/%d valid=%d invalid=%d error=%d",
                    processed,
                    expected_total,
                    stats["valid"],
                    stats["invalid"],
                    stats["error"],
                )

    manifest = {
        "version": MOTION_CACHE_VERSION,
        "signature": signature,
        "sources": cached_sources,
        "stats": stats,
        "entries": entries,
    }
    _write_manifest(cache_dir, manifest)
    if not errors:
        _remove_stale_payloads(
            cache_dir,
            {token for token, entry in entries.items() if entry.get("status") == "valid"},
        )
    logger.info(
        "Motion cache ready: %s valid=%d invalid=%d error=%d",
        cache_dir,
        stats["valid"],
        stats["invalid"],
        stats["error"],
    )
    if errors:
        preview = "\n  ".join(errors[:10])
        suffix = "" if len(errors) <= 10 else f"\n  ... and {len(errors) - 10} more"
        raise RuntimeError(
            "Motion cache encountered read/conversion errors; training was not started:\n  "
            + preview
            + suffix
        )
    return cache_dir


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
