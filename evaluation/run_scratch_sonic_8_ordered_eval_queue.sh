#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2"
EVAL_SCRIPT="${PROJECT_ROOT}/evaluation/humanoidarena_eval_sonic.sh"
RESULTS_ROOT="${PROJECT_ROOT}/eval_results/experiments/scratch_sonic_8/ordered_5tasks_2checkpoints"
QUEUE_LOG_DIR="${RESULTS_ROOT}/queue_logs"
STATUS_FILE="${RESULTS_ROOT}/queue_status.tsv"

SIM_ENV="/ai/Yichi/0_Systems/miniconda3/envs/unitree_sim_env"
CONDA_ROOT="/ai/Yichi/0_Systems/miniconda3"
SERVER_PYTHON="/ai/Yichi/0_Systems/miniconda3/envs/MOGE3/bin/python"
GPU_LIST="5,6,7"
EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-10800}"

TASKS=(
  pp_box
  boxing
  sit_sofa
  vision_navi
  open_door
)

CHECKPOINTS=(
  "${PROJECT_ROOT}/log/experiments/scratch_sonic_8/multi_task_gbs128_100w_controlnet8_detach_true_kimodo/checkpoint_800000"
  "${PROJECT_ROOT}/log/experiments/scratch_sonic_8/multi_task_gbs128_100w_controlnet8_detach_true_mse/checkpoint_800000"
)

CHECKPOINT_LABELS=(
  kimodo_checkpoint_800000
  mse_checkpoint_800000
)

mkdir -p "${QUEUE_LOG_DIR}"

exec 9>"${RESULTS_ROOT}/.queue.lock"
if ! flock -n 9; then
  echo "Another scratch_sonic_8 evaluation queue is already running." >&2
  exit 3
fi

if [[ ! -x "${EVAL_SCRIPT}" ]]; then
  echo "Evaluation script is missing or not executable: ${EVAL_SCRIPT}" >&2
  exit 2
fi
if [[ ! -d "${SIM_ENV}" ]]; then
  echo "Simulator environment does not exist: ${SIM_ENV}" >&2
  exit 2
fi
if [[ ! -x "${SERVER_PYTHON}" ]]; then
  echo "Model-server Python does not exist: ${SERVER_PYTHON}" >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required for evaluation watchdogs." >&2
  exit 2
fi
if ((${#CHECKPOINTS[@]} != ${#CHECKPOINT_LABELS[@]})); then
  echo "Checkpoint and label counts do not match." >&2
  exit 2
fi

for checkpoint in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "${checkpoint}/training_state.pt" || ! -f "${checkpoint}/config.json" ]]; then
    echo "Checkpoint is incomplete: ${checkpoint}" >&2
    exit 2
  fi
done

write_status() {
  local state="$1"
  local index="$2"
  local task="$3"
  local label="$4"
  local detail="${5:-}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "${state}" "${index}" "${task}" "${label}" "${detail}" \
    | tee -a "${STATUS_FILE}"
}

verify_result() {
  local aggregate_summary="$1"
  "${SERVER_PYTHON}" - "${aggregate_summary}" <<'PY_VERIFY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
if not summary_path.is_file():
    raise SystemExit(f"missing aggregate summary: {summary_path}")

summary = json.loads(summary_path.read_text())
episodes = int(summary.get("total_episodes", 0))
reason_counts = summary.get("result_reason_counts", {}) or {}
error_episodes = sum(
    int(count) for reason, count in reason_counts.items() if "error" in str(reason).lower()
)

if episodes != 60:
    raise SystemExit(f"expected 60 episodes, got {episodes}")
if error_episodes:
    raise SystemExit(
        f"evaluation contains {error_episodes} error episodes: {reason_counts}"
    )

print(
    "verified "
    f"episodes={episodes} "
    f"successes={int(summary.get('total_successes', 0))} "
    f"success_rate={float(summary.get('overall_success_rate', 0.0)):.6f}"
)
PY_VERIFY
}

TOTAL_RUNS=$((${#TASKS[@]} * ${#CHECKPOINTS[@]}))
RUN_INDEX=0

write_status \
  "QUEUE_STARTED" \
  "0/${TOTAL_RUNS}" \
  "-" \
  "-" \
  "gpus=${GPU_LIST}; timeout=${EVAL_TIMEOUT_SECONDS}s"

for task in "${TASKS[@]}"; do
  for checkpoint_index in "${!CHECKPOINTS[@]}"; do
    RUN_INDEX=$((RUN_INDEX + 1))
    checkpoint="${CHECKPOINTS[checkpoint_index]}"
    label="${CHECKPOINT_LABELS[checkpoint_index]}"
    result_dir="${RESULTS_ROOT}/${task}/${label}"
    item_log="${QUEUE_LOG_DIR}/$(printf '%02d' "${RUN_INDEX}")_${task}_${label}.log"
    completed_marker="${result_dir}/.queue_completed"

    if [[ -f "${completed_marker}" ]]; then
      write_status "SKIPPED_COMPLETED" "${RUN_INDEX}/${TOTAL_RUNS}" "${task}" "${label}" "${result_dir}"
      continue
    fi

    mkdir -p "${result_dir}"
    write_status "STARTED" "${RUN_INDEX}/${TOTAL_RUNS}" "${task}" "${label}" "${result_dir}"

    set +e
    timeout \
      --signal=TERM \
      --kill-after=120s \
      "${EVAL_TIMEOUT_SECONDS}" \
      env \
      KIMODO_SIM_ENV="${SIM_ENV}" \
      KIMODO_CONDA_ROOT="${CONDA_ROOT}" \
      KIMODO_SERVER_PYTHON="${SERVER_PYTHON}" \
      bash "${EVAL_SCRIPT}" \
      --project "${PROJECT_ROOT}" \
      --task "${task}" \
      --checkpoint "${checkpoint}" \
      --gpus "${GPU_LIST}" \
      --dtype fp32 \
      --diffusion-steps 10 \
      --execution-frames 15 \
      --rtc 0 \
      --rtc-overlap-frames 12 \
      --rtc-frozen-frames 1 \
      --rtc-ramp-power 1.0 \
      --persistent-sim 0 \
      --deterministic-eval 1 \
      --record-video-every-n 1 \
      --port-base 18080 \
      --results-dir "${result_dir}" 2>&1 | tee "${item_log}"
    eval_status=${PIPESTATUS[0]}
    set -e

    if ((eval_status == 124)); then
      write_status \
        "TIMEOUT" \
        "${RUN_INDEX}/${TOTAL_RUNS}" \
        "${task}" \
        "${label}" \
        "limit=${EVAL_TIMEOUT_SECONDS}s; log=${item_log}"
      exit "${eval_status}"
    fi
    if ((eval_status != 0)); then
      write_status "FAILED" "${RUN_INDEX}/${TOTAL_RUNS}" "${task}" "${label}" "exit=${eval_status}; log=${item_log}"
      exit "${eval_status}"
    fi

    if ! verification_output=$(verify_result "${result_dir}/aggregate_summary.json" 2>&1); then
      write_status "INVALID_RESULT" "${RUN_INDEX}/${TOTAL_RUNS}" "${task}" "${label}" "${verification_output}; log=${item_log}"
      exit 4
    fi

    touch "${completed_marker}"
    write_status "COMPLETED" "${RUN_INDEX}/${TOTAL_RUNS}" "${task}" "${label}" "${verification_output}"
  done
done

write_status "QUEUE_COMPLETED" "${TOTAL_RUNS}/${TOTAL_RUNS}" "-" "-" "all evaluations completed"
