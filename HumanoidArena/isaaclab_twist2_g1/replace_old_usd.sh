#!/usr/bin/env bash
set -euo pipefail

ROOT="/ai/Yichi/taowen/HumanoidArena/isaaclab_twist2_g1/assets"
OLD="/home/dreams/Users/taowen/HumanoidArena"
NEW="/ai/Yichi/taowen/HumanoidArena"
DRY_RUN=0
DO_BACKUP=1
BACKUP_DIR=""
INCLUDE_ALL=0

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [options]

Options:
  --root PATH         Root directory to scan. Default: $ROOT
  --old PATH          Old absolute path prefix.
  --new PATH          New absolute path prefix.
  --backup-dir PATH   Backup directory. Default: /ai/Yichi/taowen/temp/asset_path_backup_<timestamp>
  --no-backup         Skip backups.
  --all-files         Replace in every matched file, not only usd/usda/usdc/mdl.
  --dry-run           Only print matched files, do not modify them.
  -h, --help          Show this help.

Default file filters:
  *.usd *.usda *.usdc *.mdl
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --old)
      OLD="$2"
      shift 2
      ;;
    --new)
      NEW="$2"
      shift 2
      ;;
    --backup-dir)
      BACKUP_DIR="$2"
      shift 2
      ;;
    --no-backup)
      DO_BACKUP=0
      shift
      ;;
    --all-files)
      INCLUDE_ALL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$ROOT" ]]; then
  echo "root directory does not exist: $ROOT" >&2
  exit 2
fi

SEARCH_CMD=()
if command -v rg >/dev/null 2>&1; then
  SEARCH_CMD=(rg -a -l "$OLD" "$ROOT")
  if [[ $INCLUDE_ALL -eq 0 ]]; then
    SEARCH_CMD+=(-g '*.usd' -g '*.usda' -g '*.usdc' -g '*.mdl')
  fi
else
  SEARCH_CMD=(grep -RIa -l "$OLD" "$ROOT")
fi

mapfile -t MATCHED_FILES < <("${SEARCH_CMD[@]}")

if [[ ${#MATCHED_FILES[@]} -eq 0 ]]; then
  echo "No files contain: $OLD"
  exit 0
fi

echo "Matched ${#MATCHED_FILES[@]} file(s):"
printf '  %s\n' "${MATCHED_FILES[@]}"

if [[ $DRY_RUN -eq 1 ]]; then
  exit 0
fi

if [[ $DO_BACKUP -eq 1 ]]; then
  if [[ -z "$BACKUP_DIR" ]]; then
    BACKUP_DIR="/ai/Yichi/taowen/temp/asset_path_backup_$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "$BACKUP_DIR"
  for f in "${MATCHED_FILES[@]}"; do
    rel="${f#$ROOT/}"
    mkdir -p "$BACKUP_DIR/$(dirname "$rel")"
    cp -a "$f" "$BACKUP_DIR/$rel"
  done
  echo "Backup created at: $BACKUP_DIR"
fi

for f in "${MATCHED_FILES[@]}"; do
  python3 - "$OLD" "$NEW" "$f" <<'PY'
from pathlib import Path
import sys

old, new, file_path = sys.argv[1], sys.argv[2], sys.argv[3]
path = Path(file_path)
data = path.read_bytes()
updated = data.replace(old.encode(), new.encode())
if updated != data:
    path.write_bytes(updated)
PY
done

echo "Replacement complete. Verifying..."
if command -v rg >/dev/null 2>&1; then
  if [[ $INCLUDE_ALL -eq 0 ]]; then
    rg -a -n "$OLD" "$ROOT" -g '*.usd' -g '*.usda' -g '*.usdc' -g '*.mdl' || true
  else
    rg -a -n "$OLD" "$ROOT" || true
  fi
else
  grep -RIa -n "$OLD" "$ROOT" || true
fi
