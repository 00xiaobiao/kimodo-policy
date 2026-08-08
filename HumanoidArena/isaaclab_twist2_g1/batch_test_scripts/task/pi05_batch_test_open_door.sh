#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BATCH_START_TS="$(date +%s)"
BATCH_START_HUMAN="$(date '+%F %T %Z')"

TEST_MODE="${TEST_MODE:-base_test}"
CONFIG_STEM="open_door"
TEST_CONFIG_DIR="tasks/common_test_config/$TEST_MODE"
SONIC_CONFIG="$TEST_CONFIG_DIR/$CONFIG_STEM""_sonic_test.yaml"
TWIST2_CONFIG="$TEST_CONFIG_DIR/$CONFIG_STEM""_twist2_test.yaml"
DEFAULT_RESULTS_TAG="$TEST_MODE""_$CONFIG_STEM"
export RESULTS_TAG="${RESULTS_TAG:-$DEFAULT_RESULTS_TAG}"

format_duration() {
  local total_seconds="$1"
  local hours=$(( total_seconds / 3600 ))
  local minutes=$(( (total_seconds % 3600) / 60 ))
  local seconds=$(( total_seconds % 60 ))
  printf '%02d:%02d:%02d' "${hours}" "${minutes}" "${seconds}"
}

print_batch_summary() {
  local exit_code="$1"
  local batch_end_ts="$(date +%s)"
  local batch_end_human="$(date '+%F %T %Z')"
  local batch_elapsed=$(( batch_end_ts - BATCH_START_TS ))
  echo "[batch_test] finished_at=${batch_end_human} total_elapsed=$(format_duration "${batch_elapsed}") exit_code=${exit_code}"
}

run_task() {
  local task_name="$1"
  local script_path="$2"
  local eval_backend="${EVAL_BACKEND:-all}"
  case "${eval_backend}" in
    all|"${task_name}") ;;
    sonic|twist2)
      echo "[batch_test] task=${task_name} skipped_by_EVAL_BACKEND=${eval_backend}"
      return 0
      ;;
    *)
      echo "[batch_test] invalid EVAL_BACKEND=${eval_backend}; expected all, sonic, or twist2" >&2
      return 2
      ;;
  esac
  local task_start_ts="$(date +%s)"
  local task_start_human="$(date '+%F %T %Z')"

  echo "[batch_test] task=${task_name} started_at=${task_start_human}"
  bash "${script_path}"

  local task_end_ts="$(date +%s)"
  local task_end_human="$(date '+%F %T %Z')"
  local task_elapsed=$(( task_end_ts - task_start_ts ))
  echo "[batch_test] task=${task_name} finished_at=${task_end_human} elapsed=$(format_duration "${task_elapsed}")"
}

trap 'print_batch_summary "$?"' EXIT

echo "[batch_test] started_at=${BATCH_START_HUMAN}"

export RESUME_LATEST="${RESUME_LATEST:-0}"
if [[ -z "${MODEL_ROOT:-}" ]]; then
  echo "[batch_test] MODEL_ROOT is required. Pass a task-specific checkpoint directory from the top-level batch wrapper." >&2
  exit 2
fi
export MODEL_ROOT

ENV_CONFIG_YAML="${SONIC_CONFIG}"   run_task sonic "${ISAACLAB_ROOT}/script/eval_scripts/sonic_pi05/HSI_open_door_run_vla_eval_parallel.sh"
ENV_CONFIG_YAML="${TWIST2_CONFIG}"  run_task twist2 "${ISAACLAB_ROOT}/script/eval_scripts/twist2_pi05/HSI_open_door_run_vla_eval_parallel.sh"
