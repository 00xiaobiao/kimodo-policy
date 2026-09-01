#!/usr/bin/env bash
set -Eeuo pipefail

# Official TWIST2 evaluator wrapper. Root anchoring is enabled by default;
# use --anchor-root 0 for the historical (unanchored) baseline.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ARG="$DEFAULT_PROJECT_ROOT"; TASK_KEY="doubledesk"; CHECKPOINT=""
GPU_LIST="0"; SEED_LIST="0"; REPEATS_PER_SEED="1"; RESULTS_DIR=""; RESULTS_TAG=""
DTYPE="fp32"; DIFFUSION_STEPS="10"; EXECUTION_FRAMES="15"; RTC="0"
RTC_OVERLAP_FRAMES="0"; RTC_FROZEN_FRAMES="0"; RTC_RAMP_POWER="1.0"
PERSISTENT_SIM="0"; DETERMINISTIC_EVAL="1"; MAX_STEPS=""
RECORD_VIDEO_EVERY_N="1"; VIDEO_FPS="30"; POST_TERMINATION_RECORD_STEPS="10"
PORT_BASE="19220"; SERVER_PORT_MAX=""; SERVER_READY_TIMEOUT="360"; NUM_WORKERS="1"
ENV_CONFIG_ARG=""; ANCHOR_ROOT="1"; SERVER_TRACE="1"; DRY_RUN="0"

usage() {
  cat <<'EOF'
Kimodo HumanoidArena TWIST2 evaluator.
Usage: humanoidarena_eval_signal_task_twist2.sh [options]
Core: --project PATH --task doubledesk --checkpoint PATH --gpus LIST --seeds LIST --repeats N --results-dir PATH
Inference: --dtype fp32|bf16 --diffusion-steps N --execution-frames N --rtc 0|1
           --rtc-overlap-frames N --rtc-frozen-frames N --rtc-ramp-power X
           --persistent-sim 0|1 --deterministic-eval 0|1 --max-steps N
           --record-video-every-n N --video-fps N --port-base N
           --server-ready-timeout N --num-workers N --env-config PATH
Root fix: --anchor-root 0|1 --trace 0|1
Other: --dry-run, -h|--help
EOF
}
need_value() { [[ $# -ge 2 && -n "${2:-}" ]] || { echo "Missing value for $1" >&2; exit 2; }; }

# Read the task's official episode horizon from the corresponding TWIST2
# runner.  Keep this in the wrapper (as the SONIC wrapper does) so callers do
# not need to duplicate task-specific MAX_STEPS values in every command.
official_max_steps() {
  local config_name="$1"
  local runner_path max_steps
  local -a preferred_runners=()

  case "$config_name" in
    doubledesk_twist2_test.yaml) preferred_runners=("HOI_double_desk_run_vla_eval_parallel.sh");;
    football_single_twist2_test.yaml) preferred_runners=("HOI_football_run_vla_eval_parallel.sh");;
    pp_box_twist2_test.yaml) preferred_runners=("HOI_pp_box_run_vla_eval_parallel.sh");;
    boxing_twist2_test.yaml) preferred_runners=("HSI_boxing_run_vla_eval_parallel.sh");;
    open_door_twist2_test.yaml) preferred_runners=("HSI_open_door_run_vla_eval_parallel.sh");;
    sit_sofa_twist2_test.yaml) preferred_runners=("HSI_sit_sofa_run_vla_eval_parallel.sh");;
    vision_navi_twist2_test.yaml) preferred_runners=("HSI_vision_navi_run_vla_eval_parallel.sh");;
  esac

  for runner_path in "${preferred_runners[@]}"; do
    runner_path="${ISAACLAB_ROOT}/script/eval_scripts/twist2/${runner_path}"
    [[ -f "$runner_path" ]] || continue
    max_steps="$(sed -n 's/.*MAX_STEPS="${MAX_STEPS:-\([0-9][0-9]*\)}".*/\1/p' "$runner_path" | head -n 1)"
    [[ -n "$max_steps" ]] && { echo "$max_steps"; return 0; }
  done

  # Fallback for a newly added task wrapper: match the YAML name, while
  # preferring task wrappers over the generic runner (whose default is 1300).
  while IFS= read -r -d '' runner_path; do
    [[ "$(basename "$runner_path")" == "run_vla_eval_parallel.sh" ]] && continue
    grep -Fq "$config_name" "$runner_path" || continue
    max_steps="$(sed -n 's/.*MAX_STEPS="${MAX_STEPS:-\([0-9][0-9]*\)}".*/\1/p' "$runner_path" | head -n 1)"
    [[ -n "$max_steps" ]] && { echo "$max_steps"; return 0; }
  done < <(find "${ISAACLAB_ROOT}/script/eval_scripts/twist2" -maxdepth 1 -type f -name '*_run_vla_eval_parallel.sh' -print0)

  # Last resort: use a matching generic runner if no task wrapper exists.
  while IFS= read -r -d '' runner_path; do
    grep -Fq "$config_name" "$runner_path" || continue
    max_steps="$(sed -n 's/.*MAX_STEPS="${MAX_STEPS:-\([0-9][0-9]*\)}".*/\1/p' "$runner_path" | head -n 1)"
    [[ -n "$max_steps" ]] && { echo "$max_steps"; return 0; }
  done < <(find "${ISAACLAB_ROOT}/script/eval_scripts/twist2" -maxdepth 1 -type f -name '*_run_vla_eval_parallel.sh' -print0)
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) need_value "$@"; PROJECT_ARG="$2"; shift 2;;
    --task) need_value "$@"; TASK_KEY="$2"; shift 2;;
    --checkpoint) need_value "$@"; CHECKPOINT="$2"; shift 2;;
    --gpus) need_value "$@"; GPU_LIST="$2"; shift 2;;
    --seeds) need_value "$@"; SEED_LIST="$2"; shift 2;;
    --repeats) need_value "$@"; REPEATS_PER_SEED="$2"; shift 2;;
    --results-dir) need_value "$@"; RESULTS_DIR="$2"; shift 2;;
    --dtype) need_value "$@"; DTYPE="$2"; shift 2;;
    --diffusion-steps) need_value "$@"; DIFFUSION_STEPS="$2"; shift 2;;
    --execution-frames) need_value "$@"; EXECUTION_FRAMES="$2"; shift 2;;
    --rtc) need_value "$@"; RTC="$2"; shift 2;;
    --rtc-overlap-frames) need_value "$@"; RTC_OVERLAP_FRAMES="$2"; shift 2;;
    --rtc-frozen-frames) need_value "$@"; RTC_FROZEN_FRAMES="$2"; shift 2;;
    --rtc-ramp-power) need_value "$@"; RTC_RAMP_POWER="$2"; shift 2;;
    --persistent-sim) need_value "$@"; PERSISTENT_SIM="$2"; shift 2;;
    --deterministic-eval) need_value "$@"; DETERMINISTIC_EVAL="$2"; shift 2;;
    --max-steps) need_value "$@"; MAX_STEPS="$2"; shift 2;;
    --record-video-every-n) need_value "$@"; RECORD_VIDEO_EVERY_N="$2"; shift 2;;
    --video-fps) need_value "$@"; VIDEO_FPS="$2"; shift 2;;
    --port-base) need_value "$@"; PORT_BASE="$2"; shift 2;;
    --server-port-max) need_value "$@"; SERVER_PORT_MAX="$2"; shift 2;;
    --server-ready-timeout) need_value "$@"; SERVER_READY_TIMEOUT="$2"; shift 2;;
    --num-workers) need_value "$@"; NUM_WORKERS="$2"; shift 2;;
    --env-config) need_value "$@"; ENV_CONFIG_ARG="$2"; shift 2;;
    --anchor-root) need_value "$@"; ANCHOR_ROOT="$2"; shift 2;;
    --trace) need_value "$@"; SERVER_TRACE="$2"; shift 2;;
    --dry-run) DRY_RUN="1"; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done

if [[ "$PROJECT_ARG" == "." ]]; then PROJECT_ROOT="$(pwd -P)";
elif [[ "$PROJECT_ARG" = /* ]]; then PROJECT_ROOT="$PROJECT_ARG";
elif [[ -d "$PROJECT_ARG" ]]; then PROJECT_ROOT="$(cd -- "$PROJECT_ARG" && pwd -P)";
else PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}/${PROJECT_ARG}"; fi
PROJECT_ROOT="$(readlink -m "$PROJECT_ROOT")"
HUMANOIDARENA_ROOT="${PROJECT_ROOT}/HumanoidArena"; ISAACLAB_ROOT="${HUMANOIDARENA_ROOT}/isaaclab_twist2_g1"

case "${TASK_KEY,,}" in
  doubledesk|double_desk|double-desk|doubledesk_twist2|double_desk_twist2) TASK_CONFIG_NAME="doubledesk_twist2_test.yaml";;
  football|football_single|football-single|football_twist2|football_single_twist2) TASK_CONFIG_NAME="football_single_twist2_test.yaml";;
  pp_box|pp-box|ppbox|pp_box_twist2) TASK_CONFIG_NAME="pp_box_twist2_test.yaml";;
  boxing|boxing_twist2) TASK_CONFIG_NAME="boxing_twist2_test.yaml";;
  open_door|open-door|open_door_twist2) TASK_CONFIG_NAME="open_door_twist2_test.yaml";;
  sit_sofa|sit-sofa|sit_sofa_twist2) TASK_CONFIG_NAME="sit_sofa_twist2_test.yaml";;
  vision_navi|vision-navi|navigation|vision_navi_twist2) TASK_CONFIG_NAME="vision_navi_twist2_test.yaml";;
  *) echo "Unsupported TWIST2 task '$TASK_KEY'; supported tasks: doubledesk, football, pp_box, boxing, open_door, sit_sofa, vision_navi." >&2; exit 2;;
esac
if [[ -n "$ENV_CONFIG_ARG" ]]; then
  if [[ "$ENV_CONFIG_ARG" = /* ]]; then ENV_CONFIG_YAML="$ENV_CONFIG_ARG"; else ENV_CONFIG_YAML="${ISAACLAB_ROOT}/${ENV_CONFIG_ARG}"; fi
else ENV_CONFIG_YAML="${ISAACLAB_ROOT}/tasks/common_test_config/base_test/${TASK_CONFIG_NAME}"; fi
MAX_STEPS_SOURCE="command"
if [[ -z "$MAX_STEPS" ]]; then
  MAX_STEPS="$(official_max_steps "$(basename "$ENV_CONFIG_YAML")" || true)"
  MAX_STEPS_SOURCE="official_twist2_runner"
fi
[[ -n "$CHECKPOINT" ]] || { echo "--checkpoint is required" >&2; exit 2; }
if [[ "$CHECKPOINT" != /* ]]; then CHECKPOINT="${PROJECT_ROOT}/${CHECKPOINT}"; fi
CHECKPOINT="$(readlink -m "$CHECKPOINT")"

CONDA_BASE="${CONDA_BASE:-/ai/Yichi/0_Systems/miniconda3}"; CONDA_ENV_NAME="${CONDA_ENV_NAME:-unitree_sim_env}"
EVAL_PYTHON="${EVAL_PYTHON:-${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python}"
KIMODO_SERVER_PYTHON="${KIMODO_SERVER_PYTHON:-${CONDA_BASE}/envs/kimodo/bin/python}"
if [[ ! -x "$KIMODO_SERVER_PYTHON" && -x "${CONDA_BASE}/envs/lerobot/bin/python" ]]; then
  # Current HumanoidArena installs the Kimodo HTTP server in the lerobot env.
  KIMODO_SERVER_PYTHON="${CONDA_BASE}/envs/lerobot/bin/python"
fi
TWIST2_MODEL_PATH="${TWIST2_MODEL_PATH:-${HUMANOIDARENA_ROOT}/TWIST2/assets/ckpts/twist2_1017_20k.onnx}"
ROBOT_USD_OVERRIDE="${ROBOT_USD_OVERRIDE:-${ISAACLAB_ROOT}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd}"
GPU_LIST="${GPU_LIST//,/ }"; SEED_LIST="${SEED_LIST//,/ }"
read -r -a GPUS <<< "$GPU_LIST"; read -r -a SEEDS <<< "$SEED_LIST"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "No GPUs specified" >&2; exit 2; }; [[ ${#SEEDS[@]} -gt 0 ]] || { echo "No seeds specified" >&2; exit 2; }

for value_name in REPEATS_PER_SEED DIFFUSION_STEPS EXECUTION_FRAMES RTC_OVERLAP_FRAMES RTC_FROZEN_FRAMES MAX_STEPS RECORD_VIDEO_EVERY_N VIDEO_FPS PORT_BASE SERVER_READY_TIMEOUT NUM_WORKERS; do
  value="${!value_name}"; [[ "$value" =~ ^[0-9]+$ ]] || { echo "$value_name must be a non-negative integer, got '$value'" >&2; exit 2; }
done
for gpu in "${GPUS[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU id: $gpu" >&2; exit 2; }; done
for seed in "${SEEDS[@]}"; do [[ "$seed" =~ ^-?[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }; done
[[ "$RTC" == 0 || "$RTC" == 1 ]] || { echo "--rtc must be 0 or 1" >&2; exit 2; }
[[ "$PERSISTENT_SIM" == 0 || "$PERSISTENT_SIM" == 1 ]] || { echo "--persistent-sim must be 0 or 1" >&2; exit 2; }
[[ "$DETERMINISTIC_EVAL" == 0 || "$DETERMINISTIC_EVAL" == 1 ]] || { echo "--deterministic-eval must be 0 or 1" >&2; exit 2; }
[[ "$DETERMINISTIC_EVAL" == 0 || "$PERSISTENT_SIM" == 0 ]] || { echo "--deterministic-eval 1 requires --persistent-sim 0" >&2; exit 2; }
[[ "$ANCHOR_ROOT" == 0 || "$ANCHOR_ROOT" == 1 ]] || { echo "--anchor-root must be 0 or 1" >&2; exit 2; }
[[ "$SERVER_TRACE" == 0 || "$SERVER_TRACE" == 1 ]] || { echo "--trace must be 0 or 1" >&2; exit 2; }
[[ "$DTYPE" == fp32 || "$DTYPE" == bf16 ]] || { echo "--dtype must be fp32 or bf16" >&2; exit 2; }

[[ -d "$PROJECT_ROOT" ]] || { echo "Project root missing: $PROJECT_ROOT" >&2; exit 3; }
[[ -f "$ENV_CONFIG_YAML" ]] || { echo "TWIST2 env config missing: $ENV_CONFIG_YAML" >&2; exit 3; }
[[ -f "$CHECKPOINT/config.json" ]] || { echo "Checkpoint config.json missing: $CHECKPOINT" >&2; exit 3; }
[[ -f "$CHECKPOINT/training_state.pt" ]] || { echo "Checkpoint training_state.pt missing: $CHECKPOINT" >&2; exit 3; }
[[ -x "$EVAL_PYTHON" ]] || { echo "Simulator Python is not executable: $EVAL_PYTHON" >&2; exit 3; }
[[ -x "$KIMODO_SERVER_PYTHON" ]] || { echo "Kimodo server Python is not executable: $KIMODO_SERVER_PYTHON" >&2; exit 3; }
[[ -f "$TWIST2_MODEL_PATH" ]] || { echo "TWIST2 ONNX model missing: $TWIST2_MODEL_PATH" >&2; exit 3; }
[[ -n "$MAX_STEPS" ]] || { echo "Official MAX_STEPS was not found for $(basename "$ENV_CONFIG_YAML"); pass --max-steps N explicitly" >&2; exit 3; }

CHECKPOINT_NAME="$(basename "$CHECKPOINT")"
if [[ -z "$RESULTS_DIR" ]]; then RESULTS_DIR="${PROJECT_ROOT}/eval_results/twist2_${TASK_KEY}_${CHECKPOINT_NAME}_anchor${ANCHOR_ROOT}_$(date +%Y%m%d_%H%M%S)"; fi
RESULTS_DIR="$(readlink -m "$RESULTS_DIR")"; RESULTS_TAG="${RESULTS_TAG:-twist2_${TASK_KEY}_anchor${ANCHOR_ROOT}}"
cat <<EOF
TWIST2 evaluation plan
  project: $PROJECT_ROOT
  checkpoint: $CHECKPOINT
  task config: $ENV_CONFIG_YAML
  GPUs: ${GPUS[*]}  seeds: ${SEEDS[*]}  repeats/seed: $REPEATS_PER_SEED
  max steps: $MAX_STEPS (source: $MAX_STEPS_SOURCE)  dtype: $DTYPE  diffusion: $DIFFUSION_STEPS  execution frames: $EXECUTION_FRAMES
  RTC: $RTC (overlap=$RTC_OVERLAP_FRAMES frozen=$RTC_FROZEN_FRAMES power=$RTC_RAMP_POWER)
  root anchor: $ANCHOR_ROOT  video: every $RECORD_VIDEO_EVERY_N episode(s) @ ${VIDEO_FPS}fps
  results: $RESULTS_DIR
EOF
[[ "$DRY_RUN" == 1 ]] && { echo "Dry run passed; no evaluator launched."; exit 0; }
mkdir -p "$RESULTS_DIR"

export CONDA_BASE CONDA_ENV_NAME AUTO_ACTIVATE_CONDA=1 EVAL_PYTHON KIMODO_SERVER_PYTHON
SERVER_GPU_IDS_CSV="$(IFS=,; echo "${GPUS[*]}")"
export MODEL_PATHS_CSV="$CHECKPOINT" ENV_CONFIG_YAML SERVER_GPU_IDS="$SERVER_GPU_IDS_CSV" NUM_WORKERS
export SEEDS_OVERRIDE="${SEEDS[*]}" REPEATS_PER_SEED PERSISTENT_SIM RESULTS_DIR RESULTS_TAG RESUME_LATEST=0
export MAX_STEPS VIDEO_FPS POST_TERMINATION_RECORD_STEPS RECORD_VIDEO_EVERY_N
export SERVER_PORT_BASE="$PORT_BASE" SERVER_READY_TIMEOUT
if [[ -n "$SERVER_PORT_MAX" ]]; then export SERVER_PORT_MAX; fi
export ROBOT_USD_OVERRIDE TWIST2_MODEL_PATH
export SERVER_SCRIPT="${PROJECT_ROOT}/evaluation/humanoidarena_server.py"
export KIMODO_DTYPE="$DTYPE" KIMODO_DIFFUSION_STEPS="$DIFFUSION_STEPS" KIMODO_EXECUTION_FRAMES="$EXECUTION_FRAMES"
export KIMODO_RTC="$RTC" KIMODO_RTC_OVERLAP_FRAMES="$RTC_OVERLAP_FRAMES" KIMODO_RTC_FROZEN_FRAMES="$RTC_FROZEN_FRAMES" KIMODO_RTC_RAMP_POWER="$RTC_RAMP_POWER"
export KIMODO_DETERMINISTIC_EVAL="$DETERMINISTIC_EVAL" KIMODO_SERVER_ANCHOR_ROOT="$ANCHOR_ROOT" KIMODO_SERVER_TRACE="$SERVER_TRACE"

cd "$ISAACLAB_ROOT/script/eval_scripts/twist2"
exec bash "$ISAACLAB_ROOT/script/eval_scripts/twist2/run_vla_eval_parallel.sh"
