#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT_BASE="${MODEL_ROOT_BASE:-}"
if [[ -z "${MODEL_ROOT_BASE}" ]]; then
  echo "[batch_1_test] MODEL_ROOT_BASE is required. Download the released checkpoints and set MODEL_ROOT_BASE=/path/to/humanoidarena_checkpoints" >&2
  exit 2
fi
SMALL_MODEL_ROOT="${SMALL_MODEL_ROOT:-${MODEL_ROOT_BASE}/small}"

export RESUME_LATEST=0
SELECTED_EVAL_BACKEND="${EVAL_BACKEND:-${BATCH_TEST_BACKEND:-all}}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export VLA_MAX_ROOT_DELTA_DEG="${VLA_MAX_ROOT_DELTA_DEG:-0}"
if [[ "${VLA_MAX_ROOT_DELTA_DEG}" != "0" && "${VLA_MAX_ROOT_DELTA_DEG}" != "0.0" ]]; then
  echo "[batch_1_test] VLA_MAX_ROOT_DELTA_DEG=${VLA_MAX_ROOT_DELTA_DEG}"
fi
TEST_MODE="${TEST_MODE:-base_test}"
export TEST_MODE
export RESULTS_TAG_PREFIX="${RESULTS_TAG_PREFIX:-${BATCH_RESULTS_PREFIX:-${1:-}}}"
if [[ -n "${RESULTS_TAG_PREFIX}" ]]; then
  export RESULTS_TAG_PREFIX
  echo "[batch_1_test] RESULTS_TAG_PREFIX=${RESULTS_TAG_PREFIX}"
fi
case "${SELECTED_EVAL_BACKEND}" in
  all|sonic|twist2) ;;
  *)
    echo "[batch_1_test] invalid EVAL_BACKEND=${SELECTED_EVAL_BACKEND}; expected all, sonic, or twist2" >&2
    exit 2
    ;;
esac
echo "[batch_1_test] EVAL_BACKEND=${SELECTED_EVAL_BACKEND}"
if [[ -n "${MODEL_GLOB:-}" ]]; then
  export MODEL_GLOB
  echo "[batch_1_test] MODEL_GLOB=${MODEL_GLOB}"
fi

run_batch_backend() {
  local backend="$1"
  local model_root="$2"
  local script_path="$3"

  case "${SELECTED_EVAL_BACKEND}" in
    all|"${backend}") ;;
    *)
      echo "[batch_1_test] backend=${backend} skipped_by_EVAL_BACKEND=${SELECTED_EVAL_BACKEND} script=${script_path}"
      return 0
      ;;
  esac

  echo "[batch_1_test] backend=${backend} MODEL_GLOB=${MODEL_GLOB:-<unset>} MODEL_ROOT=${model_root} script=${script_path}"
  EVAL_BACKEND="${backend}" MODEL_ROOT="${model_root}" bash "${script_path}"
}

run_batch() {
  local model_root="$1"
  local script_path="$2"
  run_batch_backend sonic "${model_root}" "${script_path}"
  run_batch_backend twist2 "${model_root}" "${script_path}"
}

run_batch "${SMALL_MODEL_ROOT}/HSI_open_door" "${SCRIPT_DIR}/task/batch_test_open_door.sh"
run_batch "${SMALL_MODEL_ROOT}/HOI_pp_box" "${SCRIPT_DIR}/task/batch_test_pp_box.sh"
run_batch "${SMALL_MODEL_ROOT}/HSI_boxing" "${SCRIPT_DIR}/task/batch_test_boxing.sh"
run_batch "${SMALL_MODEL_ROOT}/HOI_football" "${SCRIPT_DIR}/task/batch_test_football.sh"
run_batch "${SMALL_MODEL_ROOT}/HOI_double_desk" "${SCRIPT_DIR}/task/batch_test_doubledesk.sh"
run_batch "${SMALL_MODEL_ROOT}/HSI_sit_sofa" "${SCRIPT_DIR}/task/batch_test_sit_sofa.sh"
run_batch "${SMALL_MODEL_ROOT}/HSI_vision_navi" "${SCRIPT_DIR}/task/batch_test_vision_navi.sh"
