#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

OLD_ROOTS = (
    "/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/assets",
    "/ai/Yichi/taowen/HumanoidArena/isaaclab_twist2_g1/assets",
)
ASSET_EXTS = {".usd", ".usda", ".usdc"}
TEXT_EXTS = {".yaml", ".yml", ".json", ".mdl"}
PATH_RE = re.compile(
    r"(?:(?:/home/dreams/Users/taowen)|(?:/ai/Yichi/taowen))"
    r"/HumanoidArena/isaaclab_twist2_g1/assets/[^'\"\s\]\)>,@]+"
)


def is_excluded_asset_ref(rel_assets_path: str) -> bool:
    # User requested to leave the missing SafetyTape/Props dependency aside for now.
    if "objects/Props/general/SM_DeluxeSafetyTape2M5CM_A01_Yellow_01" in rel_assets_path:
        return True
    # User requested not to touch open-door door asset references in this pass.
    if "small_warehouse_opendoor" in rel_assets_path:
        return True
    if "/door001/" in rel_assets_path or rel_assets_path.startswith("objects/small_warehouse/door001/"):
        return True
    return False


def rel_from_old_abs(old_abs: str) -> str | None:
    for root in OLD_ROOTS:
        if old_abs.startswith(root + "/"):
            return old_abs[len(root) + 1 :]
    return None


def replacement_for(old_abs: str, file_path: Path, assets_root: Path) -> tuple[str | None, str]:
    rel_assets = rel_from_old_abs(old_abs)
    if rel_assets is None:
        return None, "unknown_root"
    if is_excluded_asset_ref(rel_assets):
        return None, "excluded"
    target = assets_root / rel_assets
    if not target.exists():
        return None, "target_missing"
    new_rel = os.path.relpath(target, file_path.parent)
    return Path(new_rel).as_posix(), "replace"


def backup_file(path: Path, backup_root: Path, project_root: Path) -> Path:
    rel = path.relative_to(project_root)
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return dst


def load_usd_text(path: Path) -> str:
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        raise RuntimeError(f"failed to open USD layer: {path}")
    return layer.ExportToString()


def save_usd_text(path: Path, text: str) -> None:
    from pxr import Sdf

    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if tmp.exists():
        tmp.unlink()
    layer = Sdf.Layer.CreateNew(str(tmp))
    if layer is None:
        raise RuntimeError(f"failed to create temporary USD layer: {tmp}")
    if not layer.ImportFromString(text):
        raise RuntimeError(f"failed to import updated USD text: {path}")
    if not layer.Save():
        raise RuntimeError(f"failed to save temporary USD layer: {tmp}")
    # Replace the original crate atomically; saving an already opened crate layer
    # may keep old asset paths due to layer-cache behavior in this Isaac Sim build.
    tmp.replace(path)


def process_file(path: Path, project_root: Path, assets_root: Path, backup_root: Path, apply: bool) -> dict:
    suffix = path.suffix.lower()
    if suffix in ASSET_EXTS:
        try:
            original = load_usd_text(path)
        except Exception as exc:  # noqa: BLE001 - report and continue
            return {
                "file": str(path.relative_to(project_root)),
                "error": f"usd_open_failed: {type(exc).__name__}: {exc}",
            }
    elif suffix in TEXT_EXTS:
        try:
            original = path.read_text(errors="ignore")
        except Exception as exc:  # noqa: BLE001 - report and continue
            return {
                "file": str(path.relative_to(project_root)),
                "error": f"text_read_failed: {type(exc).__name__}: {exc}",
            }
    else:
        return {"file": str(path.relative_to(project_root)), "skipped": "unsupported_ext"}

    old_paths = sorted(set(PATH_RE.findall(original)))
    changes = []
    skipped = []
    updated = original
    for old_abs in old_paths:
        new_rel, reason = replacement_for(old_abs, path, assets_root)
        rel_assets = rel_from_old_abs(old_abs)
        entry = {"old": old_abs, "assets_relative": rel_assets, "reason": reason}
        if new_rel is None:
            skipped.append(entry)
            continue
        updated = updated.replace(old_abs, new_rel)
        entry["new"] = new_rel
        changes.append(entry)

    result = {
        "file": str(path.relative_to(project_root)),
        "old_path_count": len(old_paths),
        "change_count": len(changes),
        "skip_count": len(skipped),
        "changes": changes,
        "skipped": skipped,
    }
    if changes and apply:
        backup = backup_file(path, backup_root, project_root)
        result["backup"] = str(backup.relative_to(project_root))
        if suffix in ASSET_EXTS:
            save_usd_text(path, updated)
        else:
            path.write_text(updated)
    return result


def iter_candidate_files(scan_roots: list[Path]) -> list[Path]:
    files = []
    for root in scan_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in ASSET_EXTS | TEXT_EXTS:
                continue
            if "small_warehouse_opendoor" in path.as_posix() or "/door001/" in path.as_posix():
                continue
            files.append(path)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report-dir", default="assets_migration_reports")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    assets_root = project_root / "assets"
    scan_roots = [assets_root / "objects" / "small_warehouse"]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = project_root / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    backup_root = assets_root / ".path_fix_backup" / run_id

    results = []
    for file_path in iter_candidate_files(scan_roots):
        if file_path.suffix.lower() in ASSET_EXTS:
            results.append(process_file(file_path, project_root, assets_root, backup_root, args.apply))
            continue
        try:
            raw = file_path.read_bytes()
        except Exception:  # noqa: BLE001 - pxr/text path will report detailed failures
            raw = b""
        if not any(root.encode() in raw for root in OLD_ROOTS):
            continue
        results.append(process_file(file_path, project_root, assets_root, backup_root, args.apply))

    summary = {
        "run_id": run_id,
        "applied": bool(args.apply),
        "project_root": str(project_root),
        "backup_root": str(backup_root.relative_to(project_root)) if args.apply else None,
        "files_seen": len(results),
        "files_changed": sum(1 for r in results if r.get("change_count", 0) > 0),
        "total_changes": sum(r.get("change_count", 0) for r in results),
        "total_skipped": sum(r.get("skip_count", 0) for r in results),
        "results": results,
    }
    report_path = report_dir / f"asset_relative_path_fix_{run_id}_{'apply' if args.apply else 'dryrun'}.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {k: summary[k] for k in ("run_id", "applied", "backup_root", "files_seen", "files_changed", "total_changes", "total_skipped")},
            indent=2,
        )
    )
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
