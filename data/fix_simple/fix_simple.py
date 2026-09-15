#!/usr/bin/env python3
"""Apply data-driven fixes to an extracted SIMPLE evaluation dataset.

The rules file is intentionally separate from this script so new corrections
can be reviewed and released without changing the fixer implementation. Rules
marked ``"enabled": false`` are retained as audit records and are skipped.

Example:
    python fix_simple.py /path/to/simple-eval --dry-run
    python fix_simple.py /path/to/simple-eval

By default a timestamped backup is written next to every file that changes.
The script only edits extracted dataset directories; it never edits a zip
archive in place.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any


DEFAULT_RULES = Path(__file__).with_name("fix_simple_rules.json")
_MISSING = object()


def _values_equal(actual: Any, expected: Any, *, tol: float = 1e-9) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _values_equal(a, e, tol=tol) for a, e in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _values_equal(actual[k], expected[k], tol=tol) for k in actual
        )
    return actual == expected


def _path_parts(path: str | list[str]) -> list[str]:
    if isinstance(path, list):
        if not all(isinstance(part, str) and part for part in path):
            raise ValueError(f"Invalid JSON path: {path!r}")
        return path
    if not isinstance(path, str) or not path:
        raise ValueError(f"Invalid JSON path: {path!r}")
    parts = path.split(".")
    if not all(parts):
        raise ValueError(f"Invalid JSON path: {path!r}")
    return parts


def _get_path(document: Any, parts: list[str]) -> Any:
    value = document
    for part in parts:
        if isinstance(value, dict):
            if part not in value:
                return _MISSING
            value = value[part]
        elif isinstance(value, list):
            try:
                index = int(part)
            except ValueError:
                return _MISSING
            if index < 0 or index >= len(value):
                return _MISSING
            value = value[index]
        else:
            return _MISSING
    return value


def _set_path(document: dict[str, Any], parts: list[str], replacement: Any) -> None:
    value: Any = document
    for part in parts[:-1]:
        if isinstance(value, dict):
            if part not in value:
                raise ValueError(f"JSON path does not exist: {'.'.join(parts)}")
            value = value[part]
        elif isinstance(value, list):
            try:
                value = value[int(part)]
            except (ValueError, IndexError):
                raise ValueError(f"JSON path does not exist: {'.'.join(parts)}") from None
        else:
            raise ValueError(f"JSON path does not exist: {'.'.join(parts)}")
    leaf = parts[-1]
    if isinstance(value, dict):
        if leaf not in value:
            raise ValueError(f"JSON path does not exist: {'.'.join(parts)}")
        value[leaf] = copy.deepcopy(replacement)
    elif isinstance(value, list):
        try:
            value[int(leaf)] = copy.deepcopy(replacement)
        except (ValueError, IndexError):
            raise ValueError(f"JSON path does not exist: {'.'.join(parts)}") from None
    else:
        raise ValueError(f"JSON path does not exist: {'.'.join(parts)}")


def _field_document(entry: dict[str, Any], field: str | None) -> tuple[Any, bool]:
    """Return a nested document, decoding a JSON string field when needed."""
    if field is None:
        return entry, False
    if field not in entry:
        raise ValueError(f"Document field {field!r} is missing")
    raw = entry[field]
    if isinstance(raw, str):
        try:
            return json.loads(raw), True
        except json.JSONDecodeError as exc:
            raise ValueError(f"Document field {field!r} is not valid JSON") from exc
    if isinstance(raw, dict):
        return raw, False
    raise ValueError(f"Document field {field!r} must contain an object or JSON string")


def _set_change(
    entry: dict[str, Any],
    parts: list[str],
    replacement: Any,
    document_field: str | None,
) -> None:
    document, was_string = _field_document(entry, document_field)
    if not isinstance(document, dict):
        raise ValueError(f"Document field {document_field!r} must decode to an object")
    _set_path(document, parts, replacement)
    if document_field is not None and was_string:
        entry[document_field] = json.dumps(document, ensure_ascii=False)


def _resolve_dataset_file(root: Path, task: str, level: str, relative_file: str) -> Path:
    """Resolve both common SIMPLE layouts and level naming conventions."""
    # Accept the normal simple-eval root, a root containing a ``simple``
    # namespace, or a single task directory for convenient one-task fixes.
    task_roots = [root / task, root / "simple" / task]
    if root.name == task:
        task_roots.insert(0, root)
    levels = [level]
    if level.startswith("dr-level-"):
        levels.append("level-" + level.removeprefix("dr-level-"))
    elif level.startswith("level-"):
        levels.append("dr-level-" + level.removeprefix("level-"))
    for task_root in task_roots:
        for level_name in levels:
            candidate = task_root / level_name / relative_file
            if candidate.is_file():
                return candidate
    tried = ", ".join(str(r / l / relative_file) for r in task_roots for l in levels)
    raise FileNotFoundError(f"Could not find SIMPLE metadata file; tried: {tried}")


def _read_jsonl(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    raw_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    entries: list[dict[str, Any]] = []
    for line_number, raw in enumerate(raw_lines, start=1):
        content = raw.rstrip("\r\n")
        if not content.strip():
            continue
        try:
            entry = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(entry, dict):
            raise ValueError(f"Expected an object in {path}:{line_number}")
        entries.append(entry)
    return raw_lines, entries


def _apply_file(path: Path, changes: list[dict[str, Any]], *, dry_run: bool) -> tuple[int, int]:
    raw_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    by_episode: dict[int, tuple[int, dict[str, Any], str]] = {}
    for line_number, raw in enumerate(raw_lines, start=1):
        content = raw.rstrip("\r\n")
        if not content.strip():
            continue
        try:
            entry = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(entry, dict) or "episode_index" not in entry:
            raise ValueError(f"Missing episode_index in {path}:{line_number}")
        index = int(entry["episode_index"])
        if index in by_episode:
            raise ValueError(f"Duplicate episode_index {index} in {path}")
        by_episode[index] = (line_number - 1, entry, raw)

    pending: dict[int, dict[str, Any]] = {}
    already_fixed = 0
    for change in changes:
        episode_index = int(change["episode_index"])
        if episode_index not in by_episode:
            raise ValueError(f"Episode {episode_index} is missing from {path}")
        line_index, entry, _ = by_episode[episode_index]
        # Multiple rules may update different paths in the same JSONL entry.
        # Build each subsequent change on the already-patched pending entry so
        # earlier changes are not lost when the entry is serialized.
        current_entry = pending.get(line_index, entry)
        parts = _path_parts(change["path"])
        document, _ = _field_document(current_entry, change.get("document_field"))
        actual = _get_path(document, parts)
        if actual is _MISSING:
            raise ValueError(f"Path {change['path']!r} is missing in episode {episode_index} of {path}")
        expected = change.get("expected", _MISSING)
        replacement = change["replacement"]
        if expected is not _MISSING and not _values_equal(actual, expected):
            if _values_equal(actual, replacement):
                already_fixed += 1
                continue
            raise ValueError(
                f"Unexpected value in {path}, episode {episode_index}, path {change['path']!r}: "
                f"expected {expected!r}, found {actual!r}"
            )
        if _values_equal(actual, replacement):
            already_fixed += 1
            continue
        updated = copy.deepcopy(current_entry)
        _set_change(updated, parts, replacement, change.get("document_field"))
        pending[line_index] = updated

    if not pending or dry_run:
        return len(pending), already_fixed

    output_lines = list(raw_lines)
    for line_index, updated in pending.items():
        ending = "\n" if output_lines[line_index].endswith("\n") else ""
        if output_lines[line_index].endswith("\r\n"):
            ending = "\r\n"
        output_lines[line_index] = json.dumps(
            updated, ensure_ascii=False, separators=(",", ":")
        ) + ending

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.bak.{timestamp}")
    with open(backup, "xb") as handle:
        handle.write(path.read_bytes())

    mode = path.stat().st_mode & 0o777
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.writelines(output_lines)
    os.chmod(temporary, mode)
    os.replace(temporary, path)
    print(f"  backup: {backup}")
    return len(pending), already_fixed


def apply_rules(simple_eval: Path, rules_path: Path, *, dry_run: bool = False) -> int:
    if not simple_eval.is_dir():
        raise NotADirectoryError(f"SIMPLE eval path is not a directory: {simple_eval}")
    rules_doc = json.loads(rules_path.read_text(encoding="utf-8"))
    if rules_doc.get("schema_version") != 1:
        raise ValueError("Unsupported or missing rules schema_version (expected 1)")
    rules = rules_doc.get("rules")
    if not isinstance(rules, list):
        raise ValueError("Rules file must contain a list under 'rules'")

    grouped: dict[Path, list[dict[str, Any]]] = {}
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError(f"Invalid rule: {rule!r}")
        if rule.get("enabled", True) is False:
            print(f"Skipping disabled rule: {rule.get('id', '<unnamed>')}")
            continue
        task = rule["task"]
        level = rule["level"]
        relative_file = rule.get("file", "meta/episodes.jsonl")
        rule_path = _path_parts(rule.get("path"))
        document_field = rule.get("document_field")
        for entry in rule.get("entries", []):
            if not isinstance(entry, dict) or "episode_index" not in entry or "replacement" not in entry:
                raise ValueError(f"Invalid entry in rule {rule.get('id', '<unnamed>')!r}")
            grouped.setdefault(
                _resolve_dataset_file(simple_eval, task, level, relative_file), []
            ).append({
                **entry,
                "path": rule_path,
                "document_field": document_field,
            })

    total_changed = 0
    total_already = 0
    for path, changes in grouped.items():
        print(f"{('[dry-run] ' if dry_run else '')}{path}")
        changed, already = _apply_file(path, changes, dry_run=dry_run)
        total_changed += changed
        total_already += already
        print(f"  changes: {changed}; already fixed: {already}")
    print(f"Finished: {total_changed} change(s), {total_already} already fixed.")
    return total_changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "simple_eval",
        type=Path,
        help="Extracted SIMPLE eval root, containing <task>/<dr-level-N>/ directories",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=DEFAULT_RULES,
        help=f"Rules JSON (default: {DEFAULT_RULES})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview changes without writing files or backups",
    )
    args = parser.parse_args()
    try:
        apply_rules(args.simple_eval.expanduser().resolve(), args.rules.expanduser().resolve(), dry_run=args.dry_run)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# python data/fix_simple/fix_simple.py /path/to/simple-eval
