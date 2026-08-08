#!/usr/bin/env python3
# Rewrite training-server checkpoint prefixes inside checkpoint JSON files.

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_ROOT = Path('/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000')
DEFAULT_OLD = '/mnt/workspace/users/xujunzhe/yunhengwang/lerobot/lerobot/checkpoints'
DEFAULT_NEW = '/ai/Yichi/taowen/ckpts/checkpoints'


def iter_json_files(root: Path):
    yield from sorted(path for path in root.rglob('*.json') if path.is_file())


def rewrite_file(path: Path, old: str, new: str, apply: bool) -> int:
    text = path.read_text(encoding='utf-8')
    count = text.count(old)
    if count == 0:
        return 0
    if apply:
        path.write_text(text.replace(old, new), encoding='utf-8')
    print(f'{count:4d}  {path}')
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Find .json files and replace an old checkpoint path prefix.'
    )
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--old', default=DEFAULT_OLD)
    parser.add_argument('--new', default=DEFAULT_NEW)
    parser.add_argument('--apply', action='store_true', help='write changes; default is dry-run')
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f'root does not exist or is not a directory: {root}')

    matched_files = 0
    total_replacements = 0
    mode = 'APPLY' if args.apply else 'DRY-RUN'
    print(f'[{mode}] root={root}')
    print(f'old={args.old}')
    print(f'new={args.new}')

    for json_path in iter_json_files(root):
        count = rewrite_file(json_path, args.old, args.new, args.apply)
        if count:
            matched_files += 1
            total_replacements += count

    print(f'[{mode}] matched_files={matched_files} replacements={total_replacements}')
    if not args.apply and total_replacements:
        print('Run again with --apply to write changes.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
