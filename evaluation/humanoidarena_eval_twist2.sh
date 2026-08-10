#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_ROOT="${KIMODO_EVAL_BASE:-${DEFAULT_PROJECT_ROOT}}"
PROJECT_ROOT="${KIMODO_PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
HUMANOIDARENA_ROOT_OVERRIDE="${KIMODO_HUMANOIDARENA_ROOT:-}"

TASK_KEY="football"
CHECKPOINT=""
GPU_LIST="0"
SEED_LIST=""
REPEATS_PER_SEED=""
RESULTS_DIR=""
DTYPE="fp32"
DIFFUSION_STEPS=10
EXECUTION_FRAMES=15
RTC=0
RTC_OVERLAP_FRAMES=12
RTC_FROZEN_FRAMES=1
RTC_RAMP_POWER=1.0
PERSISTENT_SIM=""
DETERMINISTIC_EVAL=0
MAX_STEPS=""
RECORD_VIDEO_EVERY_N=0
WORKERS_PER_GPU=1
MAX_ROOT_DELTA_DEG=26
PORT_BASE=19080
SERVER_READY_TIMEOUT=600
REQUEST_TIMEOUT=120
TEXT_EMBEDDING_CACHE=""
TWIST2_MODEL_PATH=""
BENCHMARK_TASK=""
ENV_CONFIG_YAML=""
ROBOT_USD_PATH=""
SERVER_PYTHON="${KIMODO_SERVER_PYTHON:-${SERVER_PYTHON:-}}"
EVAL_PYTHON="${KIMODO_SIM_PYTHON:-${EVAL_PYTHON:-${ISAACLAB_PYTHON:-}}}"
LIST_TASKS=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Kimodo HumanoidArena TWIST2 evaluator.

Usage:
  humanoidarena_eval_twist2.sh [options]

Core options:
  --project NAME|PATH          Project containing model/ and evaluation/humanoidarena_server.py
  --task NAME                  HumanoidArena TWIST2 task alias (default: football)
  --checkpoint NAME|PATH       Kimodo checkpoint directory
  --gpus LIST                  GPU indices, for example 5,6,7 (default: 0)
  --seeds LIST                 Override official YAML seeds
  --repeats N                  Override official YAML episodes per seed
  --results-dir PATH           Output directory; generated automatically if omitted

Runtime options:
  --dtype fp32|bf16|fp16       Kimodo server dtype (default: fp32)
  --diffusion-steps N          DDIM steps (default: 10)
  --execution-frames N         Execute this many 30 Hz model frames (default: 15; 0 means all)
  --rtc 0|1                    DDIM real-time chunking (default: 0)
  --rtc-overlap-frames N       Previous model frames used as the RTC prior (default: 12)
  --rtc-frozen-frames N        Leading overlap frames kept exactly (default: 1)
  --rtc-ramp-power X           Positive cosine-ramp exponent (default: 1.0)
  --persistent-sim 0|1         Override task default
  --deterministic-eval 0|1     Deterministic Kimodo inference; requires persistent-sim=0
  --max-steps N                Override official task maximum steps
  --record-video-every-n N     0 disables video (default: 0)
  --workers-per-gpu N          TWIST2 workers on each GPU (default: 1)
  --max-root-delta-deg X       Per-control-step root rotation clamp (default: 26; 0 disables)
  --port-base N                First Kimodo server port (default: 19080)
  --server-ready-timeout N     Server startup timeout in seconds (default: 600)
  --request-timeout N          Per-request timeout in seconds (default: 120)
  --text-cache PATH            HumanoidArena text embedding cache
  --twist2-model PATH          TWIST2 low-level ONNX checkpoint
  --benchmark-task NAME        Override HumanoidArena task identifier
  --env-config PATH            Override TWIST2 task YAML
  --robot-usd PATH             Override the task's G1 robot USD
  --server-python PATH         Python used by the Kimodo model server
  --sim-python PATH            Isaac Sim Python executable or python.sh
  --list-tasks                 List supported TWIST2 tasks
  --dry-run                    Validate and print the launch plan only
  -h, --help                   Show this help

Examples:
  humanoidarena_eval_twist2.sh --task doubledesk --checkpoint /path/to/checkpoint --gpus 5,6,7
  humanoidarena_eval_twist2.sh --task football --checkpoint /path/to/checkpoint \
    --gpus 5,6,7 --seeds 0,1,2 --repeats 20 --record-video-every-n 1
EOF
}

need_value() {
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

yaml_top_level_value() {
  local field="$1" yaml_path="$2"
  sed -n "s/^${field}:[[:space:]]*//p" "$yaml_path" | head -n 1
}

yaml_test_default() {
  local field="$1" yaml_path="$2"
  awk -v field="$field" '
    /^test_defaults:[[:space:]]*$/ { in_defaults = 1; next }
    in_defaults && /^[^[:space:]]/ { exit }
    in_defaults && $1 == field ":" {
      line = $0
      sub(/^[[:space:]]*/, "", line)
      sub("^" field ":[[:space:]]*", "", line)
      print line
      exit
    }
  ' "$yaml_path"
}

normalize_list() {
  local value="$1"
  value="${value#[}"
  value="${value%]}"
  value="${value//,/ }"
  echo "$value"
}

boolean_to_int() {
  case "${1,,}" in
    true|1|yes|on) echo 1 ;;
    false|0|no|off) echo 0 ;;
    *) return 1 ;;
  esac
}

list_tasks() {
  cat <<'EOF'
Supported TWIST2 tasks:
  football      -> football_single_twist2_test.yaml (2000 steps)
  doubledesk    -> doubledesk_twist2_test.yaml (2000 steps)
  pp_box        -> pp_box_twist2_test.yaml (1450 steps)
  open_door     -> open_door_twist2_test.yaml (1800 steps)
  sit_sofa      -> sit_sofa_twist2_test.yaml (2000 steps)
  vision_navi   -> vision_navi_twist2_test.yaml (1800 steps)
  boxing        -> boxing_twist2_test.yaml (900 steps)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) need_value "$@"; PROJECT_ROOT="$2"; shift 2 ;;
    --task) need_value "$@"; TASK_KEY="$2"; shift 2 ;;
    --checkpoint) need_value "$@"; CHECKPOINT="$2"; shift 2 ;;
    --gpus) need_value "$@"; GPU_LIST="$2"; shift 2 ;;
    --seeds) need_value "$@"; SEED_LIST="$2"; shift 2 ;;
    --repeats) need_value "$@"; REPEATS_PER_SEED="$2"; shift 2 ;;
    --results-dir) need_value "$@"; RESULTS_DIR="$2"; shift 2 ;;
    --dtype) need_value "$@"; DTYPE="$2"; shift 2 ;;
    --diffusion-steps) need_value "$@"; DIFFUSION_STEPS="$2"; shift 2 ;;
    --execution-frames) need_value "$@"; EXECUTION_FRAMES="$2"; shift 2 ;;
    --rtc) need_value "$@"; RTC="$2"; shift 2 ;;
    --rtc-overlap-frames) need_value "$@"; RTC_OVERLAP_FRAMES="$2"; shift 2 ;;
    --rtc-frozen-frames) need_value "$@"; RTC_FROZEN_FRAMES="$2"; shift 2 ;;
    --rtc-ramp-power) need_value "$@"; RTC_RAMP_POWER="$2"; shift 2 ;;
    --persistent-sim) need_value "$@"; PERSISTENT_SIM="$2"; shift 2 ;;
    --deterministic-eval) need_value "$@"; DETERMINISTIC_EVAL="$2"; shift 2 ;;
    --max-steps) need_value "$@"; MAX_STEPS="$2"; shift 2 ;;
    --record-video-every-n) need_value "$@"; RECORD_VIDEO_EVERY_N="$2"; shift 2 ;;
    --workers-per-gpu) need_value "$@"; WORKERS_PER_GPU="$2"; shift 2 ;;
    --max-root-delta-deg) need_value "$@"; MAX_ROOT_DELTA_DEG="$2"; shift 2 ;;
    --port-base) need_value "$@"; PORT_BASE="$2"; shift 2 ;;
    --server-ready-timeout) need_value "$@"; SERVER_READY_TIMEOUT="$2"; shift 2 ;;
    --request-timeout) need_value "$@"; REQUEST_TIMEOUT="$2"; shift 2 ;;
    --text-cache) need_value "$@"; TEXT_EMBEDDING_CACHE="$2"; shift 2 ;;
    --twist2-model) need_value "$@"; TWIST2_MODEL_PATH="$2"; shift 2 ;;
    --benchmark-task) need_value "$@"; BENCHMARK_TASK="$2"; shift 2 ;;
    --env-config) need_value "$@"; ENV_CONFIG_YAML="$2"; shift 2 ;;
    --robot-usd) need_value "$@"; ROBOT_USD_PATH="$2"; shift 2 ;;
    --server-python) need_value "$@"; SERVER_PYTHON="$2"; shift 2 ;;
    --sim-python) need_value "$@"; EVAL_PYTHON="$2"; shift 2 ;;
    --list-tasks) LIST_TASKS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$LIST_TASKS" == 1 ]]; then
  list_tasks
  exit 0
fi

if [[ "$PROJECT_ROOT" != /* ]]; then
  if [[ "$PROJECT_ROOT" == "$(basename "$DEFAULT_PROJECT_ROOT")" ]]; then
    PROJECT_ROOT="$DEFAULT_PROJECT_ROOT"
  elif [[ -d "${DEFAULT_PROJECT_ROOT}/${PROJECT_ROOT}" ]]; then
    PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}/${PROJECT_ROOT}"
  elif [[ -d "${BASE_ROOT}/Kimodo-Policy/${PROJECT_ROOT}" ]]; then
    PROJECT_ROOT="${BASE_ROOT}/Kimodo-Policy/${PROJECT_ROOT}"
  else
    PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}/${PROJECT_ROOT}"
  fi
fi
PROJECT_ROOT="$(readlink -m "$PROJECT_ROOT")"
HUMANOIDARENA_ROOT="${HUMANOIDARENA_ROOT_OVERRIDE:-${PROJECT_ROOT}/HumanoidArena}"
ISAACLAB_ROOT="${HUMANOIDARENA_ROOT}/isaaclab_twist2_g1"
TWIST2_RUNNER="${ISAACLAB_ROOT}/script/eval_scripts/twist2/run_vla_eval_parallel.sh"

case "${TASK_KEY,,}" in
  football|football_twist2|football_single|football_single_twist2)
    TASK_SLUG="football_twist2"; CONFIG_NAME="football_single_twist2_test.yaml"; OFFICIAL_MAX_STEPS=2000; ROBOT_USD_VARIANT="m2" ;;
  doubledesk|double_desk|double-desk|doubledesk_twist2|double_desk_twist2)
    TASK_SLUG="doubledesk_twist2"; CONFIG_NAME="doubledesk_twist2_test.yaml"; OFFICIAL_MAX_STEPS=2000; ROBOT_USD_VARIANT="m2" ;;
  pp_box|pp-box|pp_box_twist2|pickplace_box|pick_place_box)
    TASK_SLUG="pp_box_twist2"; CONFIG_NAME="pp_box_twist2_test.yaml"; OFFICIAL_MAX_STEPS=1450; ROBOT_USD_VARIANT="m2_thumd" ;;
  open_door|open-door|open_door_twist2)
    TASK_SLUG="open_door_twist2"; CONFIG_NAME="open_door_twist2_test.yaml"; OFFICIAL_MAX_STEPS=1800; ROBOT_USD_VARIANT="m2_thumd" ;;
  sit_sofa|sit-sofa|sit_sofa_twist2)
    TASK_SLUG="sit_sofa_twist2"; CONFIG_NAME="sit_sofa_twist2_test.yaml"; OFFICIAL_MAX_STEPS=2000; ROBOT_USD_VARIANT="m2_thumd" ;;
  vision_navi|vision-navi|navigation|vision_navi_twist2)
    TASK_SLUG="vision_navi_twist2"; CONFIG_NAME="vision_navi_twist2_test.yaml"; OFFICIAL_MAX_STEPS=1800; ROBOT_USD_VARIANT="m2_thumd" ;;
  boxing|boxing_twist2)
    TASK_SLUG="boxing_twist2"; CONFIG_NAME="boxing_twist2_test.yaml"; OFFICIAL_MAX_STEPS=900; ROBOT_USD_VARIANT="m2_thumd" ;;
  *)
    echo "Unsupported TWIST2 task: $TASK_KEY" >&2
    echo "Run '$0 --list-tasks' to see supported tasks" >&2
    exit 2
    ;;
esac

ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_test_config/base_test/${CONFIG_NAME}}"
if [[ "$ENV_CONFIG_YAML" == /* ]]; then
  ENV_CONFIG_PATH="$ENV_CONFIG_YAML"
else
  ENV_CONFIG_PATH="${ISAACLAB_ROOT}/${ENV_CONFIG_YAML}"
fi
[[ -f "$ENV_CONFIG_PATH" ]] || { echo "TWIST2 config is missing: $ENV_CONFIG_PATH" >&2; exit 3; }

BENCHMARK_TASK="${BENCHMARK_TASK:-$(yaml_top_level_value task_name "$ENV_CONFIG_PATH")}"
if [[ -z "$SEED_LIST" ]]; then
  SEED_LIST="$(normalize_list "$(yaml_test_default seeds "$ENV_CONFIG_PATH")")"
fi
if [[ -z "$REPEATS_PER_SEED" ]]; then
  REPEATS_PER_SEED="$(yaml_test_default repeats_per_seed "$ENV_CONFIG_PATH")"
fi
if [[ -z "$PERSISTENT_SIM" ]]; then
  PERSISTENT_SIM="$(boolean_to_int "$(yaml_test_default persistent_sim "$ENV_CONFIG_PATH")" || true)"
fi
MAX_STEPS="${MAX_STEPS:-${OFFICIAL_MAX_STEPS}}"

[[ -n "$BENCHMARK_TASK" ]] || { echo "Official task_name is missing in $ENV_CONFIG_PATH" >&2; exit 3; }
[[ -n "$SEED_LIST" ]] || { echo "Official test_defaults.seeds is missing in $ENV_CONFIG_PATH" >&2; exit 3; }
[[ -n "$REPEATS_PER_SEED" ]] || { echo "Official test_defaults.repeats_per_seed is missing in $ENV_CONFIG_PATH" >&2; exit 3; }
[[ -n "$PERSISTENT_SIM" ]] || { echo "Official test_defaults.persistent_sim is missing in $ENV_CONFIG_PATH" >&2; exit 3; }

TEXT_EMBEDDING_CACHE="${TEXT_EMBEDDING_CACHE:-${PROJECT_ROOT}/data/cache/HumanoidArena}"
TWIST2_MODEL_PATH="${TWIST2_MODEL_PATH:-${HUMANOIDARENA_ROOT}/TWIST2/assets/ckpts/twist2_1017_20k.onnx}"
if [[ -z "$ROBOT_USD_PATH" ]]; then
  ROBOT_USD_PATH="${ISAACLAB_ROOT}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_${ROBOT_USD_VARIANT}.usd"
elif [[ "$ROBOT_USD_PATH" != /* ]]; then
  ROBOT_USD_PATH="${ISAACLAB_ROOT}/${ROBOT_USD_PATH}"
fi

if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="${PROJECT_ROOT}/eval_checkpoints/${TASK_SLUG}_checkpoint_100000"
elif [[ "$CHECKPOINT" != /* ]]; then
  if [[ -e "${PROJECT_ROOT}/${CHECKPOINT}" ]]; then
    CHECKPOINT="${PROJECT_ROOT}/${CHECKPOINT}"
  else
    CHECKPOINT="${PROJECT_ROOT}/eval_checkpoints/${CHECKPOINT}"
  fi
fi
CHECKPOINT="$(readlink -m "$CHECKPOINT")"

if [[ -z "$SERVER_PYTHON" ]]; then
  for candidate in \
    "${PROJECT_ROOT}/.venv/bin/python" \
    "${PROJECT_ROOT}/envs/kimodo_eval/bin/python" \
    "${BASE_ROOT}/envs/kimodo_eval/bin/python" \
    "/root/miniconda3/envs/kimodo/bin/python"; do
    if [[ -x "$candidate" ]]; then SERVER_PYTHON="$candidate"; break; fi
  done
fi
if [[ -z "$SERVER_PYTHON" ]]; then SERVER_PYTHON="$(command -v python3 || command -v python || true)"; fi
if [[ -n "$SERVER_PYTHON" && "$SERVER_PYTHON" != /* ]]; then SERVER_PYTHON="$(command -v "$SERVER_PYTHON" || true)"; fi
[[ -x "$SERVER_PYTHON" ]] || { echo "Kimodo server Python was not found; use --server-python." >&2; exit 3; }

if [[ -z "$EVAL_PYTHON" ]]; then
  for candidate in \
    "/ai/Yichi/taowen/isaac-sim/python.sh" \
    "${KIMODO_CONDA_ROOT:-}/envs/unitree_sim_env/bin/python" \
    "${CONDA_BASE:-}/envs/unitree_sim_env/bin/python"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then EVAL_PYTHON="$candidate"; break; fi
  done
fi
if [[ -n "$EVAL_PYTHON" && "$EVAL_PYTHON" != /* ]]; then EVAL_PYTHON="$(command -v "$EVAL_PYTHON" || true)"; fi
if [[ "$DRY_RUN" != 1 ]]; then
  [[ -x "$EVAL_PYTHON" ]] || { echo "Isaac Sim Python was not found; use --sim-python." >&2; exit 3; }
fi

for path in \
  "$PROJECT_ROOT/model" \
  "$PROJECT_ROOT/evaluation/humanoidarena_server.py" \
  "$CHECKPOINT/training_state.pt" \
  "$CHECKPOINT/config.json" \
  "$TEXT_EMBEDDING_CACHE" \
  "$TWIST2_MODEL_PATH" \
  "$ENV_CONFIG_PATH" \
  "$ROBOT_USD_PATH" \
  "$TWIST2_RUNNER"; do
  [[ -e "$path" ]] || { echo "Required path is missing: $path" >&2; exit 3; }
done

case "$DTYPE" in fp32|bf16|fp16) ;; *) echo "Invalid dtype: $DTYPE" >&2; exit 2 ;; esac
for value in "$REPEATS_PER_SEED" "$DIFFUSION_STEPS" "$EXECUTION_FRAMES" "$RTC_OVERLAP_FRAMES" "$RTC_FROZEN_FRAMES" "$RECORD_VIDEO_EVERY_N" "$PORT_BASE" "$SERVER_READY_TIMEOUT" "$MAX_STEPS"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "Expected non-negative integer, got: $value" >&2; exit 2; }
done
[[ "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || { echo "--workers-per-gpu must be positive" >&2; exit 2; }
[[ "$RTC" == 0 || "$RTC" == 1 ]] || { echo "--rtc must be 0 or 1" >&2; exit 2; }
[[ "$PERSISTENT_SIM" == 0 || "$PERSISTENT_SIM" == 1 ]] || { echo "--persistent-sim must be 0 or 1" >&2; exit 2; }
[[ "$DETERMINISTIC_EVAL" == 0 || "$DETERMINISTIC_EVAL" == 1 ]] || { echo "--deterministic-eval must be 0 or 1" >&2; exit 2; }
(( RTC_FROZEN_FRAMES <= RTC_OVERLAP_FRAMES )) || { echo "--rtc-frozen-frames cannot exceed --rtc-overlap-frames" >&2; exit 2; }
if [[ "$DETERMINISTIC_EVAL" == 1 && "$PERSISTENT_SIM" != 0 ]]; then
  echo "--deterministic-eval 1 requires --persistent-sim 0" >&2
  exit 2
fi

read -r PREDICTION_FRAMES MODEL_FPS RESOLVED_EXECUTION_FRAMES CONTROL_FRAMES RTC_ACTIVE RESOLVED_RTC_OVERLAP RESOLVED_RTC_FROZEN < <(
  "$SERVER_PYTHON" - "$CHECKPOINT/config.json" "$EXECUTION_FRAMES" "$RTC" "$RTC_OVERLAP_FRAMES" "$RTC_FROZEN_FRAMES" "$RTC_RAMP_POWER" "$MAX_ROOT_DELTA_DEG" "$REQUEST_TIMEOUT" <<'PY_VALIDATE'
import json
import math
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
requested = int(sys.argv[2])
rtc_enabled = bool(int(sys.argv[3]))
rtc_overlap = int(sys.argv[4])
rtc_frozen = int(sys.argv[5])
for name, raw, allow_zero in (
    ("--rtc-ramp-power", sys.argv[6], False),
    ("--max-root-delta-deg", sys.argv[7], True),
    ("--request-timeout", sys.argv[8], False),
):
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid {name}: {raw!r}") from exc
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise SystemExit(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
try:
    cfg = json.loads(config_path.read_text())
    prediction_frames = int(cfg["main"]["action_chunk"])
    model_fps = float(cfg["model"]["fps"])
except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Invalid checkpoint frame config in {config_path}: {exc}")
if prediction_frames <= 0 or model_fps <= 0:
    raise SystemExit("Checkpoint action_chunk and fps must be positive")
if requested > prediction_frames:
    raise SystemExit(f"--execution-frames {requested} exceeds action_chunk {prediction_frames}")
resolved = prediction_frames if requested == 0 else requested
control_frames = max(1, int(round(resolved * 50.0 / model_fps)))
tail = max(0, prediction_frames - resolved)
overlap = min(rtc_overlap, tail)
frozen = min(rtc_frozen, overlap)
print(prediction_frames, f"{model_fps:g}", resolved, control_frames, int(rtc_enabled and overlap > 0), overlap, frozen)
PY_VALIDATE
)

GPU_LIST="$(normalize_list "$GPU_LIST")"
SEED_LIST="$(normalize_list "$SEED_LIST")"
read -r -a GPUS <<< "$GPU_LIST"
read -r -a SEEDS <<< "$SEED_LIST"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "No GPUs specified" >&2; exit 2; }
[[ ${#SEEDS[@]} -gt 0 ]] || { echo "No seeds specified" >&2; exit 2; }
for gpu in "${GPUS[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU: $gpu" >&2; exit 2; }; done
for seed in "${SEEDS[@]}"; do [[ "$seed" =~ ^-?[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }; done
GPUS_CSV="$(IFS=,; echo "${GPUS[*]}")"
TOTAL_WORKERS=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))

PROJECT_NAME="$(basename "$PROJECT_ROOT")"
CHECKPOINT_NAME="$(basename "$CHECKPOINT")"
if [[ -z "$RESULTS_DIR" ]]; then
  RESULTS_DIR="${PROJECT_ROOT}/eval_results/${PROJECT_NAME}_${TASK_SLUG}_${CHECKPOINT_NAME}_${DTYPE}_$(date +%Y%m%d_%H%M%S)"
fi
RESULTS_DIR="$(readlink -m "$RESULTS_DIR")"

cat <<EOF
TWIST2 evaluation plan
  project:             $PROJECT_ROOT
  checkpoint:          $CHECKPOINT
  task:                $TASK_SLUG
  benchmark task:      $BENCHMARK_TASK
  env config:          $ENV_CONFIG_YAML
  robot USD:           $ROBOT_USD_PATH
  GPUs:                ${GPUS[*]}
  workers/GPU:         $WORKERS_PER_GPU
  total workers:       $TOTAL_WORKERS
  seeds:               ${SEEDS[*]}
  repeats/seed:        $REPEATS_PER_SEED
  max steps:           $MAX_STEPS
  dtype:               $DTYPE
  diffusion steps:     $DIFFUSION_STEPS
  model prediction:    $PREDICTION_FRAMES frames @ $MODEL_FPS FPS
  model execution:     $RESOLVED_EXECUTION_FRAMES frames
  sim control:         $CONTROL_FRAMES frames @ 50 FPS
  RTC active:          $RTC_ACTIVE
  RTC overlap/frozen:  $RESOLVED_RTC_OVERLAP/$RESOLVED_RTC_FROZEN model frames
  root delta clamp:    $MAX_ROOT_DELTA_DEG degrees
  persistent sim:      $PERSISTENT_SIM
  deterministic:       $DETERMINISTIC_EVAL
  request timeout:     $REQUEST_TIMEOUT seconds
  record every N:      $RECORD_VIDEO_EVERY_N
  results:             $RESULTS_DIR
EOF

if [[ "$DRY_RUN" == 1 ]]; then
  echo "Dry run passed; no evaluation was launched."
  exit 0
fi

mkdir -p "$RESULTS_DIR"
cat > "${RESULTS_DIR}/run_config.txt" <<EOF
started_at=$(date -Iseconds)
backend=twist2
project=$PROJECT_ROOT
checkpoint=$CHECKPOINT
task_slug=$TASK_SLUG
benchmark_task=$BENCHMARK_TASK
env_config=$ENV_CONFIG_YAML
robot_usd=$ROBOT_USD_PATH
gpus=${GPUS[*]}
workers_per_gpu=$WORKERS_PER_GPU
total_workers=$TOTAL_WORKERS
seeds=${SEEDS[*]}
repeats_per_seed=$REPEATS_PER_SEED
max_steps=$MAX_STEPS
dtype=$DTYPE
diffusion_steps=$DIFFUSION_STEPS
prediction_frames=$PREDICTION_FRAMES
model_fps=$MODEL_FPS
execution_frames=$RESOLVED_EXECUTION_FRAMES
simulator_control_frames=$CONTROL_FRAMES
rtc_requested=$RTC
rtc_active=$RTC_ACTIVE
rtc_overlap_frames=$RESOLVED_RTC_OVERLAP
rtc_frozen_frames=$RESOLVED_RTC_FROZEN
rtc_ramp_power=$RTC_RAMP_POWER
max_root_delta_deg=$MAX_ROOT_DELTA_DEG
persistent_sim=$PERSISTENT_SIM
deterministic_eval=$DETERMINISTIC_EVAL
server_ready_timeout=$SERVER_READY_TIMEOUT
request_timeout=$REQUEST_TIMEOUT
record_video_every_n=$RECORD_VIDEO_EVERY_N
EOF

SIM_BOOTSTRAP="${PROJECT_ROOT}/evaluation/sim_python_bootstrap"
SIM_SITE_PACKAGES="${KIMODO_SIM_SITE_PACKAGES:-}"
CONDA_ROOT="${KIMODO_CONDA_ROOT:-${CONDA_BASE:-}}"
if [[ -z "$SIM_SITE_PACKAGES" && -n "$CONDA_ROOT" ]]; then
  SIM_SITE_PACKAGES="$(find "${CONDA_ROOT}/envs/unitree_sim_env/lib" -maxdepth 2 -type d -name site-packages -print 2>/dev/null | head -n 1)"
fi
SERVER_PREFIX="$(cd "$(dirname "$SERVER_PYTHON")/.." && pwd)"
NVIDIA_LIBRARY_PATH="$(find "${SERVER_PREFIX}/lib" -path '*/site-packages/nvidia/*/lib' -type d -print 2>/dev/null | sort | paste -sd: -)"
RUNTIME_LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATH}${NVIDIA_LIBRARY_PATH:+:}${LD_LIBRARY_PATH:-}"
RUNTIME_PYTHONPATH="${SIM_BOOTSTRAP}${PYTHONPATH:+:${PYTHONPATH}}"

set +e
env \
  CONDA_BASE="$CONDA_ROOT" \
  CONDA_ENV_NAME=unitree_sim_env \
  AUTO_ACTIVATE_CONDA=0 \
  EVAL_PYTHON="$EVAL_PYTHON" \
  CONFIG_PYTHON="$SERVER_PYTHON" \
  KIMODO_SIM_SITE_PACKAGES="$SIM_SITE_PACKAGES" \
  KIMODO_HUMANOIDARENA_ISAACLAB="$ISAACLAB_ROOT" \
  PYTHONPATH="$RUNTIME_PYTHONPATH" \
  LD_LIBRARY_PATH="$RUNTIME_LD_LIBRARY_PATH" \
  MODEL_PATHS_CSV="$CHECKPOINT" \
  RESULTS_DIR="$RESULTS_DIR" \
  RESUME_LATEST=0 \
  SERVER_PYTHON="$SERVER_PYTHON" \
  SERVER_SCRIPT="${PROJECT_ROOT}/evaluation/humanoidarena_server.py" \
  SERVER_GPU_IDS="$GPUS_CSV" \
  SERVER_DEVICE="cuda:${GPUS[0]}" \
  SERVER_PORT_BASE="$PORT_BASE" \
  SERVER_PORT_MAX=65535 \
  SERVER_READY_TIMEOUT="$SERVER_READY_TIMEOUT" \
  LEROBOT_SERVER_TIMEOUT="$REQUEST_TIMEOUT" \
  NUM_WORKERS="$WORKERS_PER_GPU" \
  SEEDS_OVERRIDE="${SEEDS[*]}" \
  REPEATS_PER_SEED="$REPEATS_PER_SEED" \
  PERSISTENT_SIM="$PERSISTENT_SIM" \
  RECORD_VIDEO_EVERY_N="$RECORD_VIDEO_EVERY_N" \
  VIDEO_FPS=50 \
  POST_TERMINATION_RECORD_STEPS=10 \
  STEP_LOG_EVERY_N=100 \
  MAX_STEPS="$MAX_STEPS" \
  ENV_CONFIG_YAML="$ENV_CONFIG_PATH" \
  ROBOT_USD_OVERRIDE="$ROBOT_USD_PATH" \
  TWIST2_MODEL_PATH="$TWIST2_MODEL_PATH" \
  VLA_MAX_ROOT_DELTA_DEG="$MAX_ROOT_DELTA_DEG" \
  KIMODO_TEXT_EMBEDDING_CACHE="$TEXT_EMBEDDING_CACHE" \
  KIMODO_DTYPE="$DTYPE" \
  KIMODO_DIFFUSION_STEPS="$DIFFUSION_STEPS" \
  KIMODO_EXECUTION_FRAMES="$RESOLVED_EXECUTION_FRAMES" \
  KIMODO_RTC="$RTC_ACTIVE" \
  KIMODO_RTC_OVERLAP_FRAMES="$RESOLVED_RTC_OVERLAP" \
  KIMODO_RTC_FROZEN_FRAMES="$RESOLVED_RTC_FROZEN" \
  KIMODO_RTC_RAMP_POWER="$RTC_RAMP_POWER" \
  KIMODO_DETERMINISTIC_EVAL="$DETERMINISTIC_EVAL" \
  PYTHONHASHSEED=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMNI_KIT_ACCEPT_EULA=YES \
  bash "$TWIST2_RUNNER"
code=$?
set -e

if (( code != 0 )); then
  printf '%s\n' "$code" > "${RESULTS_DIR}/.failed"
  echo "TWIST2 evaluation failed with exit code $code: $RESULTS_DIR" >&2
  exit "$code"
fi

date -Iseconds > "${RESULTS_DIR}/.done"
echo "TWIST2 evaluation complete: $RESULTS_DIR"
