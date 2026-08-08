#!/usr/bin/env bash

if [[ -n "${_VLA_MODEL_BATCH_UTILS_SH:-}" ]]; then
  return 0
fi
readonly _VLA_MODEL_BATCH_UTILS_SH=1

MODEL_BATCH_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$(cd "${MODEL_BATCH_UTILS_DIR}/../../.." && pwd)}"
source "${ISAACLAB_ROOT}/script/common/runtime_paths.sh"

DEFAULT_BATCH_MODEL_ROOT="${DEFAULT_BATCH_MODEL_ROOT:-${LEROBOT_ROOT}/outputs/train}"
readonly DEFAULT_BATCH_MODEL_ROOT

trim_model_batch_value() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

resolve_model_dir() {
  local path
  path="$(trim_model_batch_value "$1")"
  [[ -z "$path" ]] && return 1

  if [[ -d "$path/pretrained_model" ]]; then
    printf '%s' "$path/pretrained_model"
    return 0
  fi

  if [[ -d "$path" ]]; then
    printf '%s' "$path"
    return 0
  fi

  return 1
}

append_model_path_if_valid() {
  local raw_path resolved_path
  raw_path="$(trim_model_batch_value "$1")"
  [[ -z "$raw_path" ]] && return 0

  if ! resolved_path="$(resolve_model_dir "$raw_path")"; then
    echo "Warning: skip missing model path: $raw_path" >&2
    return 0
  fi

  MODEL_PATHS+=("$resolved_path")
}

load_model_paths_from_file() {
  local file_path="$1"
  local line=""

  if [[ ! -f "$file_path" ]]; then
    echo "Error: MODEL_PATHS_FILE does not exist: $file_path" >&2
    return 1
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="$(trim_model_batch_value "$line")"
    [[ -z "$line" ]] && continue
    [[ "$line" == \#* ]] && continue
    append_model_path_if_valid "$line"
  done <"$file_path"
}

load_model_paths_from_csv() {
  local csv="$1"
  local raw_path=""
  IFS=',' read -r -a raw_paths <<<"$csv"
  for raw_path in "${raw_paths[@]}"; do
    append_model_path_if_valid "$raw_path"
  done
}

discover_model_paths() {
  local default_glob="$1"
  local model_root="${MODEL_ROOT:-$DEFAULT_BATCH_MODEL_ROOT}"
  local model_glob="${MODEL_GLOB:-$default_glob}"
  local exclude_glob="${MODEL_EXCLUDE_GLOB:-}"
  local model_limit="${MODEL_LIMIT:-0}"
  local -a discovered_paths=()
  local path=""

  MODEL_PATHS=()

  if [[ -n "${MODEL_PATHS_FILE:-}" ]]; then
    load_model_paths_from_file "$MODEL_PATHS_FILE"
  elif [[ -n "${MODEL_PATHS_CSV:-}" ]]; then
    load_model_paths_from_csv "$MODEL_PATHS_CSV"
  else
    if [[ ! -d "$model_root" ]]; then
      echo "Error: model root does not exist: $model_root" >&2
      return 1
    fi

    mapfile -t discovered_paths < <(
      {
        find "$model_root" -path '*/checkpoints/*/pretrained_model' -type d
        find "$model_root" -mindepth 2 -maxdepth 2 -name pretrained_model -type d
      } | sort -u
    )
    for path in "${discovered_paths[@]}"; do
      if [[ -n "$model_glob" && ! "$path" == $model_glob && ! "${path%/pretrained_model}" == $model_glob ]]; then
        continue
      fi
      if [[ -n "$exclude_glob" && ( "$path" == $exclude_glob || "${path%/pretrained_model}" == $exclude_glob ) ]]; then
        continue
      fi
      append_model_path_if_valid "$path"
    done
  fi

  if [[ "$model_limit" =~ ^[0-9]+$ ]] && (( model_limit > 0 )) && (( ${#MODEL_PATHS[@]} > model_limit )); then
    MODEL_PATHS=("${MODEL_PATHS[@]:0:model_limit}")
  fi

  if (( ${#MODEL_PATHS[@]} == 0 )); then
    echo "Error: no model paths selected." >&2
    echo "  MODEL_ROOT=${model_root}" >&2
    echo "  MODEL_GLOB=${model_glob}" >&2
    echo "  MODEL_EXCLUDE_GLOB=${exclude_glob}" >&2
    echo "  MODEL_PATHS_FILE=${MODEL_PATHS_FILE:-}" >&2
    echo "  MODEL_PATHS_CSV=${MODEL_PATHS_CSV:-}" >&2
    return 1
  fi
}

print_model_paths_summary() {
  local index=0
  echo "Selected ${#MODEL_PATHS[@]} model(s):"
  for path in "${MODEL_PATHS[@]}"; do
    printf '  [%02d] %s\n' "$index" "$path"
    index=$((index + 1))
  done
}
