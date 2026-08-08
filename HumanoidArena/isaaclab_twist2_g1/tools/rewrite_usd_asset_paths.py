#!/usr/bin/env python3
"""Rewrite authored USD asset paths in-place using Pixar USD APIs.

This script is intentionally conservative:
- It never edits USD bytes directly.
- It only saves files whose old prefix can be removed from the USD layer export.
- It performs a full preflight before saving any file, so strict failures do not
  leave a partially modified tree.

Run with Isaac Sim/IsaacLab Python so that ``pxr`` is available, for example:
    /ai/Yichi/taowen/isaac-sim/python.sh tools/rewrite_usd_asset_paths.py \
        --root assets/objects/small_warehouse
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Iterable

try:
    from pxr import Sdf, UsdUtils
except Exception as exc:  # pragma: no cover - depends on Isaac/USD runtime.
    raise SystemExit(
        "Failed to import pxr. Run this script with Isaac Sim/IsaacLab Python, "
        "for example: /ai/Yichi/taowen/isaac-sim/python.sh tools/rewrite_usd_asset_paths.py\n"
        f"Import error: {exc}"
    ) from exc


DEFAULT_OLD_PREFIX = "/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1"
DEFAULT_NEW_PREFIX = "/ai/Yichi/taowen/HumanoidArena/isaaclab_twist2_g1"
USD_EXTENSIONS = {".usd", ".usda", ".usdc"}


@dataclass
class FilePlan:
    path: Path
    layer: Sdf.Layer
    changed_asset_paths: int
    changed_layer_metadata: int
    old_occurrences_before: int
    old_occurrences_after: int
    error: str = ""

    @property
    def needs_save(self) -> bool:
        return self.changed_asset_paths > 0 or self.changed_layer_metadata > 0


def _normalize_prefix(prefix: str) -> str:
    return prefix.rstrip("/")


def _rewrite_text(text: str, old_prefix: str, new_prefix: str) -> str:
    if not text:
        return text
    return text.replace(old_prefix, new_prefix)


def _count_old_prefix_in_layer(layer: Sdf.Layer, old_prefix: str) -> int:
    """Count old-prefix occurrences in the layer's USD text representation."""
    try:
        return layer.ExportToString().count(old_prefix)
    except Exception:
        # ExportToString can fail for malformed layers. Treat as unknown/unsafe.
        return -1


def _iter_usd_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.name.startswith("._"):
            return
        if root.suffix.lower() in USD_EXTENSIONS:
            yield root
        return
    for path in sorted(root.rglob("*")):
        if path.name.startswith("._"):
            continue
        if path.is_file() and path.suffix.lower() in USD_EXTENSIONS:
            yield path


def _build_asset_path_rewriter(
    old_prefix: str,
    new_prefix: str,
) -> tuple[Callable[[str], str], Callable[[], int]]:
    changed_count = 0

    def rewrite_asset_path(asset_path: str) -> str:
        nonlocal changed_count
        rewritten = _rewrite_text(str(asset_path), old_prefix, new_prefix)
        if rewritten != asset_path:
            changed_count += 1
        return rewritten

    def get_changed_count() -> int:
        return changed_count

    return rewrite_asset_path, get_changed_count


def _rewrite_layer_metadata(layer: Sdf.Layer, old_prefix: str, new_prefix: str) -> int:
    """Rewrite non-semantic layer metadata strings such as root-layer doc text.

    UsdUtils.ModifyAssetPaths deliberately handles authored asset paths only.
    Some exported/baked files also keep the source machine path in the root
    layer documentation:

        doc = "Generated from Composed Stage of root layer /old/path/..."

    That path is not used for dependency resolution, but it is still an old
    absolute path. Rewriting this metadata through Sdf keeps the file format
    valid and avoids binary string replacement.
    """
    changed = 0
    for attr_name in ("documentation", "comment"):
        try:
            value = getattr(layer, attr_name)
        except Exception:
            continue
        if not isinstance(value, str) or old_prefix not in value:
            continue
        try:
            setattr(layer, attr_name, _rewrite_text(value, old_prefix, new_prefix))
        except Exception as exc:
            raise RuntimeError(f"failed to rewrite layer.{attr_name}: {exc}") from exc
        changed += 1
    return changed


def _prepare_file_plan(path: Path, old_prefix: str, new_prefix: str) -> FilePlan:
    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        return FilePlan(
            path=path,
            layer=None,  # type: ignore[arg-type]
            changed_asset_paths=0,
            changed_layer_metadata=0,
            old_occurrences_before=-1,
            old_occurrences_after=-1,
            error="Sdf.Layer.FindOrOpen returned None",
        )

    before_count = _count_old_prefix_in_layer(layer, old_prefix)
    if before_count == 0:
        return FilePlan(
            path=path,
            layer=layer,
            changed_asset_paths=0,
            changed_layer_metadata=0,
            old_occurrences_before=0,
            old_occurrences_after=0,
        )
    if before_count < 0:
        return FilePlan(
            path=path,
            layer=layer,
            changed_asset_paths=0,
            changed_layer_metadata=0,
            old_occurrences_before=before_count,
            old_occurrences_after=-1,
            error="failed to export layer to string before rewrite",
        )

    rewriter, get_changed_count = _build_asset_path_rewriter(old_prefix, new_prefix)
    try:
        UsdUtils.ModifyAssetPaths(layer, rewriter)
    except Exception as exc:
        return FilePlan(
            path=path,
            layer=layer,
            changed_asset_paths=get_changed_count(),
            changed_layer_metadata=0,
            old_occurrences_before=before_count,
            old_occurrences_after=-1,
            error=f"UsdUtils.ModifyAssetPaths failed: {exc}",
        )

    try:
        metadata_changed_count = _rewrite_layer_metadata(layer, old_prefix, new_prefix)
    except Exception as exc:
        return FilePlan(
            path=path,
            layer=layer,
            changed_asset_paths=get_changed_count(),
            changed_layer_metadata=0,
            old_occurrences_before=before_count,
            old_occurrences_after=-1,
            error=str(exc),
        )

    after_count = _count_old_prefix_in_layer(layer, old_prefix)
    if after_count < 0:
        error = "failed to export layer to string after rewrite"
    else:
        error = ""
    return FilePlan(
        path=path,
        layer=layer,
        changed_asset_paths=get_changed_count(),
        changed_layer_metadata=metadata_changed_count,
        old_occurrences_before=before_count,
        old_occurrences_after=after_count,
        error=error,
    )


def _save_plan(plan: FilePlan) -> str:
    if not plan.needs_save:
        return "skip"
    try:
        ok = plan.layer.Save()
    except Exception as exc:
        raise RuntimeError(f"failed to save {plan.path}: {exc}") from exc
    if ok is False:
        raise RuntimeError(f"failed to save {plan.path}: Sdf.Layer.Save returned False")
    return "saved"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite old absolute asset paths inside USD files in-place using "
            "UsdUtils.ModifyAssetPaths."
        )
    )
    parser.add_argument(
        "--root",
        default="assets/objects/small_warehouse",
        help="USD file or directory to scan recursively.",
    )
    parser.add_argument(
        "--old-prefix",
        default=DEFAULT_OLD_PREFIX,
        help="Old absolute path prefix to remove from authored USD asset paths.",
    )
    parser.add_argument(
        "--new-prefix",
        default=DEFAULT_NEW_PREFIX,
        help="Replacement path prefix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and rewrite in memory, but do not save files.",
    )
    parser.add_argument(
        "--allow-residual",
        action="store_true",
        help=(
            "Save even if the old prefix remains in a layer after "
            "UsdUtils.ModifyAssetPaths. Without this flag, any residual old path "
            "aborts before saving any file."
        ),
    )
    parser.add_argument(
        "--must-find",
        action="store_true",
        help="Exit non-zero if no matching old-prefix occurrence is found.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.root).expanduser().resolve()
    old_prefix = _normalize_prefix(args.old_prefix)
    new_prefix = _normalize_prefix(args.new_prefix)

    if not root.exists():
        print(f"[usd_path_rewrite] ERROR: root does not exist: {root}", file=sys.stderr)
        return 2
    if not old_prefix:
        print("[usd_path_rewrite] ERROR: --old-prefix must not be empty", file=sys.stderr)
        return 2
    if old_prefix == new_prefix:
        print("[usd_path_rewrite] ERROR: old and new prefixes are identical", file=sys.stderr)
        return 2

    usd_files = list(_iter_usd_files(root))
    if not usd_files:
        print(f"[usd_path_rewrite] ERROR: no USD files found under {root}", file=sys.stderr)
        return 2

    print(f"[usd_path_rewrite] root={root}")
    print(f"[usd_path_rewrite] old_prefix={old_prefix}")
    print(f"[usd_path_rewrite] new_prefix={new_prefix}")
    print(f"[usd_path_rewrite] files_scanned={len(usd_files)} dry_run={int(args.dry_run)}")

    plans: list[FilePlan] = []
    for path in usd_files:
        plan = _prepare_file_plan(path, old_prefix, new_prefix)
        plans.append(plan)
        if plan.error:
            print(f"[usd_path_rewrite] ERROR {path}: {plan.error}", file=sys.stderr)
            continue
        if plan.old_occurrences_before > 0:
            print(
                "[usd_path_rewrite] PLAN "
                f"{path}: before={plan.old_occurrences_before} "
                f"asset_paths_changed={plan.changed_asset_paths} "
                f"metadata_changed={plan.changed_layer_metadata} "
                f"after={plan.old_occurrences_after}"
            )

    hard_errors = [plan for plan in plans if plan.error]
    residuals = [
        plan
        for plan in plans
        if not plan.error and plan.old_occurrences_after > 0
    ]
    changed_plans = [plan for plan in plans if not plan.error and plan.needs_save]
    found_occurrences = sum(
        max(plan.old_occurrences_before, 0)
        for plan in plans
        if not plan.error
    )

    if args.must_find and found_occurrences == 0:
        print("[usd_path_rewrite] ERROR: old prefix was not found", file=sys.stderr)
        return 3

    if hard_errors:
        print(
            f"[usd_path_rewrite] ABORT: {len(hard_errors)} file(s) failed preflight; no files saved.",
            file=sys.stderr,
        )
        return 4

    if residuals and not args.allow_residual:
        print(
            "[usd_path_rewrite] ABORT: old prefix remains after USD asset-path rewrite; "
            "no files saved. These occurrences may be non-asset string fields:",
            file=sys.stderr,
        )
        for plan in residuals:
            print(
                f"  {plan.path}: residual_occurrences={plan.old_occurrences_after}",
                file=sys.stderr,
            )
        return 5

    if args.dry_run:
        print(
            f"[usd_path_rewrite] DRY_RUN complete: would_save={len(changed_plans)} "
            f"old_occurrences_found={found_occurrences}"
        )
        return 0

    saved = 0
    for plan in changed_plans:
        status = _save_plan(plan)
        if status == "saved":
            saved += 1
            verify_layer = Sdf.Layer.FindOrOpen(str(plan.path))
            if verify_layer is None:
                raise RuntimeError(f"failed to reopen saved layer: {plan.path}")
            residual = _count_old_prefix_in_layer(verify_layer, old_prefix)
            if residual > 0 and not args.allow_residual:
                raise RuntimeError(
                    f"old prefix still present after saving {plan.path}: {residual}"
                )
            print(f"[usd_path_rewrite] SAVED {plan.path}")

    print(
        f"[usd_path_rewrite] DONE files_scanned={len(usd_files)} "
        f"files_saved={saved} old_occurrences_found={found_occurrences}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
