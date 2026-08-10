#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-100000}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/log/2026-08-08_12-56-22/checkpoint_${CHECKPOINT_STEP}}"
KIMODO_SERVER_PYTHON="/root/miniconda3/envs/kimodo/bin/python"
SIM_PYTHON="/ai/Yichi/taowen/isaac-sim/python.sh"
CONDA_BASE="/ai/Yichi/0_Systems/miniconda3"
TWIST2_MODEL_PATH="${PROJECT_ROOT}/HumanoidArena/TWIST2/assets/ckpts/twist2_1017_20k.onnx"
SIM_SITE_PACKAGES="/ai/Yichi/0_Systems/miniconda3/envs/unitree_sim_env/lib/python3.10/site-packages"
HUMANOIDARENA_ISAACLAB="${PROJECT_ROOT}/HumanoidArena/isaaclab_twist2_g1"
SIM_PYTHONPATH="${PROJECT_ROOT}/evaluation/sim_python_bootstrap"
NVIDIA_LIB_ROOT="/root/miniconda3/envs/kimodo/lib/python3.10/site-packages/nvidia"
SIM_NVIDIA_LIBRARY_PATH="$(find "${NVIDIA_LIB_ROOT}" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/eval_results/checkpoint_${CHECKPOINT_STEP}_sonic_then_twist2_${RUN_ID}}"
MASTER_LOG="${RUN_ROOT}/sequence.log"
STATUS_FILE="${RUN_ROOT}/status.tsv"

TASKS=(
  doubledesk
  football
  pp_box
  open_door
  sit_sofa
  vision_navi
  boxing
)

mkdir -p "${RUN_ROOT}"
exec >>"${MASTER_LOG}" 2>&1
printf 'backend\ttask\tstatus\texit_code\tstarted_at\tfinished_at\tresults_dir\n' >"${STATUS_FILE}"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*"
}

record_status() {
  local backend="$1" task="$2" status="$3" code="$4" started_at="$5" finished_at="$6" results_dir="$7"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${backend}" "${task}" "${status}" "${code}" "${started_at}" "${finished_at}" "${results_dir}" \
    >>"${STATUS_FILE}"
}

run_sonic() {
  local task="$1" task_index="$2"
  local results_dir="${RUN_ROOT}/sonic/${task}"
  local port_base=$((18080 + task_index * 20))

  mkdir -p "${results_dir}"
  KIMODO_SERVER_PYTHON="${KIMODO_SERVER_PYTHON}" \
    KIMODO_SIM_PYTHON="${SIM_PYTHON}" \
    KIMODO_CONDA_ROOT="${CONDA_BASE}" \
    KIMODO_SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES}" \
    KIMODO_HUMANOIDARENA_ISAACLAB="${HUMANOIDARENA_ISAACLAB}" \
    PYTHONPATH="${SIM_PYTHONPATH}" \
    LD_LIBRARY_PATH="${SIM_NVIDIA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}" \
    bash "${PROJECT_ROOT}/evaluation/humanoidarena_eval_sonic.sh" \
      --project "${PROJECT_ROOT}" \
      --task "${task}" \
      --checkpoint "${CHECKPOINT}" \
      --gpus 5,6,7 \
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
      --port-base "${port_base}" \
      --results-dir "${results_dir}"
}

run_twist2() {
  local task="$1" task_index="$2"
  local results_dir="${RUN_ROOT}/twist2/${task}"
  local port_base=$((19080 + task_index * 20))

  mkdir -p "${results_dir}"
  KIMODO_SERVER_PYTHON="${KIMODO_SERVER_PYTHON}" \
    KIMODO_SIM_PYTHON="${SIM_PYTHON}" \
    KIMODO_CONDA_ROOT="${CONDA_BASE}" \
    KIMODO_SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES}" \
    KIMODO_HUMANOIDARENA_ISAACLAB="${HUMANOIDARENA_ISAACLAB}" \
    PYTHONPATH="${SIM_PYTHONPATH}" \
    LD_LIBRARY_PATH="${SIM_NVIDIA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}" \
    bash "${PROJECT_ROOT}/evaluation/humanoidarena_eval_twist2.sh" \
      --project "${PROJECT_ROOT}" \
      --task "${task}" \
      --checkpoint "${CHECKPOINT}" \
      --gpus 5,6,7 \
      --seeds 0,1,2 \
      --dtype fp32 \
      --diffusion-steps 10 \
      --execution-frames 15 \
      --rtc 0 \
      --persistent-sim 0 \
      --deterministic-eval 1 \
      --record-video-every-n 1 \
      --workers-per-gpu 1 \
      --max-root-delta-deg 26 \
      --port-base "${port_base}" \
      --server-ready-timeout 600 \
      --request-timeout 120 \
      --twist2-model "${TWIST2_MODEL_PATH}" \
      --results-dir "${results_dir}"
}

run_one() {
  local backend="$1" task="$2" task_index="$3"
  local started_at finished_at code status results_dir
  results_dir="${RUN_ROOT}/${backend}/${task}"
  started_at="$(date -Iseconds)"
  log "START backend=${backend} task=${task} results=${results_dir}"

  code=0
  if [[ "${backend}" == sonic ]]; then
    run_sonic "${task}" "${task_index}" || code=$?
  else
    run_twist2 "${task}" "${task_index}" || code=$?
  fi

  finished_at="$(date -Iseconds)"
  if (( code == 0 )); then
    status=SUCCESS
  else
    status=FAILED
  fi
  record_status "${backend}" "${task}" "${status}" "${code}" "${started_at}" "${finished_at}" "${results_dir}"
  log "END backend=${backend} task=${task} status=${status} exit_code=${code}"
}

log "Batch started"
log "Checkpoint: ${CHECKPOINT}"
log "Order: all SONIC tasks, then all TWIST2 tasks"
log "GPUs: 5,6,7"

for backend in sonic twist2; do
  for task_index in "${!TASKS[@]}"; do
    run_one "${backend}" "${TASKS[$task_index]}" "${task_index}"
  done
done

log "Batch finished"
log "Status summary: ${STATUS_FILE}"
