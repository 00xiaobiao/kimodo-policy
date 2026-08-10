#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

run_internal_task() {
  local kimodo_root="${KIMODO_ROOT:-${DEFAULT_PROJECT_ROOT}}"
  local humanoidarena_root="${HUMANOIDARENA_ROOT:?Set HUMANOIDARENA_ROOT}"
  local checkpoint="${CHECKPOINT:?Set CHECKPOINT}"
  local server_python="${SERVER_PYTHON:?Set SERVER_PYTHON}"
  local task="${TASK:?Set TASK}"
  local env_config_yaml="${ENV_CONFIG_YAML:?Set ENV_CONFIG_YAML}"
  local results_dir="${RESULTS_DIR:?Set RESULTS_DIR}"
  local max_steps="${MAX_STEPS:?Set MAX_STEPS}"
  local text_embedding_cache="${TEXT_EMBEDDING_CACHE:?Set TEXT_EMBEDDING_CACHE to the HumanoidArena cache directory or a legacy cache file}"
  local sonic_policy_root="${SONIC_POLICY_ROOT:-${humanoidarena_root}/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release}"
  local isaac_device="${ISAAC_DEVICE:-cuda:0}"
  local server_device="${SERVER_DEVICE:-cuda:0}"
  local eval_seeds="${EVAL_SEEDS:-0 1 2}"
  local repeats_per_seed="${REPEATS_PER_SEED:-20}"
  local persistent_sim="${PERSISTENT_SIM:-1}"
  local record_video_every_n="${RECORD_VIDEO_EVERY_N:-0}"
  local server_port="${SERVER_PORT:-18080}"
  local server_ready_timeout="${KIMODO_SERVER_READY_TIMEOUT:-600}"
  local execution_frames="${KIMODO_EXECUTION_FRAMES:-0}"
  local rtc="${KIMODO_RTC:-1}"
  local rtc_overlap_frames="${KIMODO_RTC_OVERLAP_FRAMES:-12}"
  local rtc_frozen_frames="${KIMODO_RTC_FROZEN_FRAMES:-1}"
  local rtc_ramp_power="${KIMODO_RTC_RAMP_POWER:-1.0}"
  local deterministic_eval="${KIMODO_DETERMINISTIC_EVAL:-0}"
  local sim_python_bin="${KIMODO_SIM_PYTHON:-${SIM_PYTHON_BIN:-}}"
  local sim_env="${KIMODO_SIM_ENV:-${SIM_ENV:-}}"
  local conda_root="${KIMODO_CONDA_ROOT:-${CONDA_ROOT:-}}"
  local candidate seed

  export KIMODO_DTYPE="${KIMODO_DTYPE:-fp32}"
  export KIMODO_DIFFUSION_STEPS="${KIMODO_DIFFUSION_STEPS:-10}"
  export KIMODO_TEXT_EMBEDDING_CACHE="${text_embedding_cache}"
  export KIMODO_DETERMINISTIC_EVAL="${deterministic_eval}"
  export KIMODO_EXECUTION_FRAMES="${execution_frames}"
  export KIMODO_RTC="${rtc}"
  export KIMODO_RTC_OVERLAP_FRAMES="${rtc_overlap_frames}"
  export KIMODO_RTC_FROZEN_FRAMES="${rtc_frozen_frames}"
  export KIMODO_RTC_RAMP_POWER="${rtc_ramp_power}"
  export OMNI_KIT_ACCEPT_EULA=YES

  local args=(
    --task "${task}"
    --env_config_yaml "${env_config_yaml}"
    --model-path "${checkpoint}"
    --repeats_per_seed "${repeats_per_seed}"
    --persistent_sim "${persistent_sim}"
    --max_steps "${max_steps}"
    # One rendered video frame is captured after every 50 Hz control action.
    --video_fps 50
    --post_termination_record_steps 10
    --record_video_every_n "${record_video_every_n}"
    --step_log_every_n 100
    --robot_type unitree_g1_refpose_v3_1
    --sonic_encoder_path "${sonic_policy_root}/model_encoder.onnx"
    --sonic_decoder_path "${sonic_policy_root}/model_decoder.onnx"
    --sonic_vla_root_rot6d_layout row
    --sonic_vla_root_max_delta_deg 26
    --results_dir "${results_dir}"
    --headless
    --isaac_device "${isaac_device}"
    --server_python "${server_python}"
    --server_script "${kimodo_root}/evaluation/humanoidarena_server.py"
    --server_device "${server_device}"
    --server_host 127.0.0.1
    --server_port "${server_port}"
    --server_scheme http
    --server_ready_timeout "${server_ready_timeout}"
    --lerobot_server_timeout 30
  )

  for seed in ${eval_seeds}; do
    args+=(--seed "${seed}")
  done

  if [[ -n "$sim_python_bin" ]]; then
    [[ -x "$sim_python_bin" ]] || {
      echo "KIMODO_SIM_PYTHON is not executable: $sim_python_bin" >&2
      exit 3
    }
  else
    if [[ -z "$sim_env" ]]; then
      for candidate in \
        "${kimodo_root}/envs/unitree_sim_env" \
        "${kimodo_root}/.venv-sim"; do
        if [[ -x "${candidate}/bin/python" ]]; then
          sim_env="$candidate"
          break
        fi
      done
    fi

    if [[ -n "$sim_env" && -x "${sim_env}/bin/python" ]]; then
      if [[ -n "$conda_root" && -f "${conda_root}/etc/profile.d/conda.sh" ]]; then
        set +u
        # shellcheck disable=SC1090
        source "${conda_root}/etc/profile.d/conda.sh"
        conda activate "$sim_env"
        set -u
      fi
      sim_python_bin="${sim_env}/bin/python"
    elif [[ -n "$sim_env" && -n "$conda_root" && -f "${conda_root}/etc/profile.d/conda.sh" ]]; then
      set +u
      # shellcheck disable=SC1090
      source "${conda_root}/etc/profile.d/conda.sh"
      conda activate "$sim_env"
      set -u
      sim_python_bin="$(command -v python)"
    fi
  fi

  if [[ -z "$sim_python_bin" ]]; then
    cat >&2 <<'EOF'
Isaac simulator Python was not found.
Set one of:
  KIMODO_SIM_PYTHON=/path/to/isaac-sim-python
  KIMODO_SIM_ENV=/path/to/unitree_sim_env
Optionally set KIMODO_CONDA_ROOT=/path/to/miniconda3 when activation is required.
EOF
    exit 3
  fi

  cd "${humanoidarena_root}/isaaclab_twist2_g1"
  exec "$sim_python_bin" script/eval_scripts/sonic/eval_vla_suite.py "${args[@]}"
}

if [[ "${1:-}" == "--internal-run-task" ]]; then
  shift
  [[ $# -eq 0 ]] || { echo "Internal worker does not accept arguments" >&2; exit 2; }
  run_internal_task
fi

BASE_ROOT="${KIMODO_EVAL_BASE:-${DEFAULT_PROJECT_ROOT}}"
PROJECT_ROOT="${KIMODO_PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
HUMANOIDARENA_ROOT="${KIMODO_HUMANOIDARENA_ROOT:-${DEFAULT_PROJECT_ROOT}/HumanoidArena}"
SERVER_PYTHON="${KIMODO_SERVER_PYTHON:-${SERVER_PYTHON:-}}"
ISAACLAB_ROOT="${HUMANOIDARENA_ROOT}/isaaclab_twist2_g1"
BASE_TEST_CONFIG_DIR="${ISAACLAB_ROOT}/tasks/common_test_config/base_test"
SONIC_EVAL_SCRIPT_DIR="${ISAACLAB_ROOT}/script/eval_scripts/sonic"
TASK_KEY="football"
CHECKPOINT=""
GPU_LIST="0"
SEED_LIST=""
REPEATS_PER_SEED=""
RESULTS_DIR=""
DTYPE="fp32"
DIFFUSION_STEPS=10
EXECUTION_FRAMES=0
RTC=1
RTC_OVERLAP_FRAMES=12
RTC_FROZEN_FRAMES=1
RTC_RAMP_POWER=1.0
PERSISTENT_SIM=""
DETERMINISTIC_EVAL=0
RECORD_VIDEO_EVERY_N=0
PORT_BASE=18080
SERVER_READY_TIMEOUT=600
MAX_STEPS=""
TEXT_EMBEDDING_CACHE=""
BENCHMARK_TASK=""
ENV_CONFIG_YAML=""
DRY_RUN=0
LIST_TASKS=0

usage() {
  cat <<'EOF'
Kimodo HumanoidArena SONIC evaluator.

Usage:
  humanoidarena_eval_sonic.sh [options]

Core options:
  --project NAME|PATH       Project containing model/ and evaluation/humanoidarena_server.py
                            (default: directory containing this script)
  --task NAME               HumanoidArena SONIC task alias (default: football)
  --checkpoint NAME|PATH    Checkpoint directory. Defaults to the task's 100k checkpoint.
  --gpus LIST               Comma/space-separated GPU indices (default: 0)
  --seeds LIST              Override official YAML group seeds
  --repeats N               Override official YAML episodes per seed
  --results-dir PATH        Output directory; generated automatically if omitted

Runtime options:
  --dtype fp32|bf16|fp16    Model server dtype (default: fp32)
  --diffusion-steps N       DDIM steps (default: 10)
  --execution-frames N      Execute first N predicted model frames; 0 means all
  --rtc 0|1                 DDIM real-time chunking for chunk continuity (default: 1)
  --rtc-overlap-frames N    Previous model frames used as the RTC prior (default: 12)
  --rtc-frozen-frames N     Leading overlap frames kept exactly (default: 1)
  --rtc-ramp-power X        Positive cosine-ramp exponent (default: 1.0)
  --persistent-sim 0|1      Override task default
  --deterministic-eval 0|1  Use explicit diffusion RNG and deterministic CUDA mode
                            Requires --persistent-sim 0 (default: 0)
  --max-steps N             Override task maximum steps
  --record-video-every-n N  0 disables video (default: 0)
  --port-base N             First server port (default: 18080)
  --server-ready-timeout N  Model server startup timeout in seconds (default: 600)
  --text-cache PATH         Override the HumanoidArena cache directory or legacy cache file
  --benchmark-task NAME     Override HumanoidArena task identifier
  --env-config PATH         Override HumanoidArena task YAML
  --list-tasks              List supported HumanoidArena SONIC tasks
  --dry-run                 Validate and print the launch plan only
  -h, --help                Show this help

Examples:
  humanoidarena_eval_sonic.sh --task football --gpus 0,1,2 --seeds 0,1,2
  humanoidarena_eval_sonic.sh --task sit_sofa --gpus 0,1,2 --seeds 0,1,2
  humanoidarena_eval_sonic.sh --task doubledesk --gpus 0,1,2 --seeds 0,1,2
  humanoidarena_eval_sonic.sh --task pp_box --checkpoint /path/to/checkpoint --text-cache /path/to/cache
  humanoidarena_eval_sonic.sh --project cross_attention_v1_eval_scaled \
    --checkpoint eval_checkpoints/checkpoint_100000 --task football --gpus 0,1,2
  humanoidarena_eval_sonic.sh --task football --gpus 7 --seeds 0 --repeats 1 --dry-run
EOF
}

need_value() {
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

yaml_top_level_value() {
  local field="$1"
  local yaml_path="$2"
  sed -n "s/^${field}:[[:space:]]*//p" "$yaml_path" | head -n 1
}

yaml_test_default() {
  local field="$1"
  local yaml_path="$2"
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

normalize_seed_list() {
  local seed_list="$1"
  seed_list="${seed_list#[}"
  seed_list="${seed_list%]}"
  seed_list="${seed_list//,/ }"
  echo "$seed_list"
}

boolean_to_int() {
  case "${1,,}" in
    true|1|yes|on) echo 1 ;;
    false|0|no|off) echo 0 ;;
    *) return 1 ;;
  esac
}

official_max_steps() {
  local config_name="$1"
  local runner_path
  local max_steps
  while IFS= read -r -d '' runner_path; do
    grep -Fq "$config_name" "$runner_path" || continue
    max_steps="$(sed -n 's/.*MAX_STEPS="${MAX_STEPS:-\([0-9][0-9]*\)}".*/\1/p' "$runner_path" | head -n 1)"
    if [[ -n "$max_steps" ]]; then
      echo "$max_steps"
      return 0
    fi
  done < <(find "$SONIC_EVAL_SCRIPT_DIR" -maxdepth 1 -type f -name '*_run_vla_eval_parallel.sh' -print0)
  return 1
}

list_tasks() {
  cat <<'EOF'
Supported SONIC tasks:
  football      -> football_single_sonic_test.yaml
  sit_sofa      -> sit_sofa_sonic_test.yaml
  vision_navi   -> vision_navi_sonic_test.yaml
  boxing        -> boxing_sonic_test.yaml
  open_door     -> open_door_sonic_test.yaml
  doubledesk    -> doubledesk_sonic_test.yaml
  pp_box        -> pp_box_sonic_test.yaml
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
    --port-base) need_value "$@"; PORT_BASE="$2"; shift 2 ;;
    --server-ready-timeout) need_value "$@"; SERVER_READY_TIMEOUT="$2"; shift 2 ;;
    --text-cache) need_value "$@"; TEXT_EMBEDDING_CACHE="$2"; shift 2 ;;
    --benchmark-task) need_value "$@"; BENCHMARK_TASK="$2"; shift 2 ;;
    --env-config) need_value "$@"; ENV_CONFIG_YAML="$2"; shift 2 ;;
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

case "${TASK_KEY,,}" in
  football|football_sonic|football_single|football_single_sonic)
    TASK_SLUG="football_sonic"
    CONFIG_NAME="football_single_sonic_test.yaml"
    ;;
  sit_sofa|sit-sofa|sit_sofa_sonic)
    TASK_SLUG="sit_sofa_sonic"
    CONFIG_NAME="sit_sofa_sonic_test.yaml"
    ;;
  vision_navi|vision-navi|navigation|vision_navi_sonic)
    TASK_SLUG="vision_navi_sonic"
    CONFIG_NAME="vision_navi_sonic_test.yaml"
    ;;
  boxing|boxing_sonic)
    TASK_SLUG="boxing_sonic"
    CONFIG_NAME="boxing_sonic_test.yaml"
    ;;
  open_door|open-door|open_door_sonic)
    TASK_SLUG="open_door_sonic"
    CONFIG_NAME="open_door_sonic_test.yaml"
    ;;
  doubledesk|double_desk|double-desk|doubledesk_sonic|double_desk_sonic)
    TASK_SLUG="doubledesk_sonic"
    CONFIG_NAME="doubledesk_sonic_test.yaml"
    ;;
  pp_box|pp-box|pp_box_sonic|pickplace_box|pick_place_box)
    TASK_SLUG="pp_box_sonic"
    CONFIG_NAME="pp_box_sonic_test.yaml"
    ;;
  *)
    echo "Unsupported task: $TASK_KEY" >&2
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
[[ -f "$ENV_CONFIG_PATH" ]] || { echo "HumanoidArena config is missing: $ENV_CONFIG_PATH" >&2; exit 3; }

BENCHMARK_TASK="${BENCHMARK_TASK:-$(yaml_top_level_value task_name "$ENV_CONFIG_PATH")}"
if [[ -z "$SEED_LIST" ]]; then
  SEED_LIST="$(normalize_seed_list "$(yaml_test_default seeds "$ENV_CONFIG_PATH")")"
fi
if [[ -z "$REPEATS_PER_SEED" ]]; then
  REPEATS_PER_SEED="$(yaml_test_default repeats_per_seed "$ENV_CONFIG_PATH")"
fi
if [[ -z "$PERSISTENT_SIM" ]]; then
  PERSISTENT_SIM="$(boolean_to_int "$(yaml_test_default persistent_sim "$ENV_CONFIG_PATH")" || true)"
fi
if [[ -z "$MAX_STEPS" ]]; then
  MAX_STEPS="$(official_max_steps "$(basename "$ENV_CONFIG_PATH")" || true)"
fi

[[ -n "$BENCHMARK_TASK" ]] || { echo "Official task_name is missing in $ENV_CONFIG_PATH" >&2; exit 3; }
[[ -n "$SEED_LIST" ]] || { echo "Official test_defaults.seeds is missing in $ENV_CONFIG_PATH" >&2; exit 3; }
[[ -n "$REPEATS_PER_SEED" ]] || { echo "Official test_defaults.repeats_per_seed is missing in $ENV_CONFIG_PATH" >&2; exit 3; }
[[ -n "$PERSISTENT_SIM" ]] || { echo "Official test_defaults.persistent_sim is missing in $ENV_CONFIG_PATH" >&2; exit 3; }
[[ -n "$MAX_STEPS" ]] || { echo "Official MAX_STEPS was not found for $(basename "$ENV_CONFIG_PATH")" >&2; exit 3; }

TEXT_EMBEDDING_CACHE="${TEXT_EMBEDDING_CACHE:-${PROJECT_ROOT}/data/cache/HumanoidArena}"

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

EVALUATOR_SCRIPT="${PROJECT_ROOT}/evaluation/humanoidarena_eval_sonic.sh"
SONIC_POLICY_ROOT="${HUMANOIDARENA_ROOT}/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"

if [[ -z "$SERVER_PYTHON" ]]; then
  for candidate in \
    "${PROJECT_ROOT}/.venv/bin/python" \
    "${PROJECT_ROOT}/envs/kimodo_eval/bin/python" \
    "${BASE_ROOT}/envs/kimodo_eval/bin/python"; do
    if [[ -x "$candidate" ]]; then
      SERVER_PYTHON="$candidate"
      break
    fi
  done
fi
if [[ -z "$SERVER_PYTHON" ]]; then
  SERVER_PYTHON="$(command -v python3 || command -v python || true)"
fi
[[ -n "$SERVER_PYTHON" ]] || {
  echo "Model server Python was not found. Set KIMODO_SERVER_PYTHON=/path/to/python." >&2
  exit 3
}

for path in \
  "$PROJECT_ROOT/model" \
  "$PROJECT_ROOT/evaluation/humanoidarena_server.py" \
  "$CHECKPOINT/training_state.pt" \
  "$CHECKPOINT/config.json" \
  "$TEXT_EMBEDDING_CACHE" \
  "$ENV_CONFIG_PATH" \
  "$EVALUATOR_SCRIPT" \
  "$SERVER_PYTHON" \
  "$SONIC_POLICY_ROOT/model_encoder.onnx" \
  "$SONIC_POLICY_ROOT/model_decoder.onnx"; do
  [[ -e "$path" ]] || { echo "Required path is missing: $path" >&2; exit 3; }
done

case "$DTYPE" in fp32|bf16|fp16) ;; *) echo "Invalid dtype: $DTYPE" >&2; exit 2;; esac
for value in "$REPEATS_PER_SEED" "$DIFFUSION_STEPS" "$EXECUTION_FRAMES" "$RTC_OVERLAP_FRAMES" "$RTC_FROZEN_FRAMES" "$RECORD_VIDEO_EVERY_N" "$PORT_BASE" "$SERVER_READY_TIMEOUT" "$MAX_STEPS"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "Expected non-negative integer, got: $value" >&2; exit 2; }
done
[[ "$RTC" == 0 || "$RTC" == 1 ]] || { echo "--rtc must be 0 or 1" >&2; exit 2; }
(( RTC_FROZEN_FRAMES <= RTC_OVERLAP_FRAMES )) || {
  echo "--rtc-frozen-frames cannot exceed --rtc-overlap-frames" >&2
  exit 2
}
"$SERVER_PYTHON" - "$RTC_RAMP_POWER" <<'PY_VALIDATE_RTC'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit(f"Invalid --rtc-ramp-power: {sys.argv[1]!r}") from exc
if not math.isfinite(value) or value <= 0:
    raise SystemExit("--rtc-ramp-power must be finite and positive")
PY_VALIDATE_RTC
[[ "$PERSISTENT_SIM" == 0 || "$PERSISTENT_SIM" == 1 ]] || { echo "--persistent-sim must be 0 or 1" >&2; exit 2; }
[[ "$DETERMINISTIC_EVAL" == 0 || "$DETERMINISTIC_EVAL" == 1 ]] || { echo "--deterministic-eval must be 0 or 1" >&2; exit 2; }
if [[ "$DETERMINISTIC_EVAL" == 1 && "$PERSISTENT_SIM" != 0 ]]; then
  echo "--deterministic-eval 1 requires --persistent-sim 0 for per-episode process isolation" >&2
  exit 2
fi

read -r PREDICTION_FRAMES MODEL_FPS RESOLVED_EXECUTION_FRAMES CONTROL_FRAMES RTC_ACTIVE RESOLVED_RTC_OVERLAP RESOLVED_RTC_FROZEN < <(
  "$SERVER_PYTHON" - "$CHECKPOINT/config.json" "$EXECUTION_FRAMES" "$RTC" "$RTC_OVERLAP_FRAMES" "$RTC_FROZEN_FRAMES" <<'PY_FRAME_CONFIG'
import json
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
requested = int(sys.argv[2])
rtc_enabled = bool(int(sys.argv[3]))
rtc_overlap = int(sys.argv[4])
rtc_frozen = int(sys.argv[5])
try:
    config = json.loads(config_path.read_text())
    prediction_frames = int(config["main"]["action_chunk"])
    model_fps = float(config["model"]["fps"])
except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Invalid checkpoint frame config in {config_path}: {exc}")
if prediction_frames <= 0 or model_fps <= 0:
    raise SystemExit(
        f"Invalid checkpoint frame config: action_chunk={prediction_frames}, fps={model_fps}"
    )
if requested > prediction_frames:
    raise SystemExit(
        f"--execution-frames {requested} exceeds checkpoint action_chunk {prediction_frames}"
    )
resolved = prediction_frames if requested == 0 else requested
control_frames = max(1, int(round(resolved * 50.0 / model_fps)))
available_tail = max(0, prediction_frames - resolved)
effective_overlap = min(rtc_overlap, available_tail)
effective_frozen = min(rtc_frozen, effective_overlap)
rtc_active = int(rtc_enabled and effective_overlap > 0)
print(
    prediction_frames,
    f"{model_fps:g}",
    resolved,
    control_frames,
    rtc_active,
    effective_overlap,
    effective_frozen,
)
PY_FRAME_CONFIG
)

GPU_LIST="${GPU_LIST//,/ }"
SEED_LIST="${SEED_LIST//,/ }"
read -r -a GPUS <<< "$GPU_LIST"
read -r -a SEEDS <<< "$SEED_LIST"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "No GPUs specified" >&2; exit 2; }
[[ ${#SEEDS[@]} -gt 0 ]] || { echo "No seeds specified" >&2; exit 2; }
for gpu in "${GPUS[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU: $gpu" >&2; exit 2; }; done
for seed in "${SEEDS[@]}"; do [[ "$seed" =~ ^-?[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }; done

PROJECT_NAME="$(basename "$PROJECT_ROOT")"
CHECKPOINT_NAME="$(basename "$CHECKPOINT")"
if [[ -z "$RESULTS_DIR" ]]; then
  RESULTS_DIR="${PROJECT_ROOT}/eval_results/${PROJECT_NAME}_${TASK_SLUG}_${CHECKPOINT_NAME}_${DTYPE}_$(date +%Y%m%d_%H%M%S)"
fi
RESULTS_DIR="$(readlink -m "$RESULTS_DIR")"

cat <<EOF
Evaluation plan
  project:          $PROJECT_ROOT
  checkpoint:       $CHECKPOINT
  task:             $TASK_SLUG
  benchmark task:   $BENCHMARK_TASK
  env config:       $ENV_CONFIG_YAML
  GPUs:             ${GPUS[*]}
  seeds:            ${SEEDS[*]}
  repeats/seed:     $REPEATS_PER_SEED
  max steps:        $MAX_STEPS
  dtype:            $DTYPE
  diffusion steps:  $DIFFUSION_STEPS
  model prediction: $PREDICTION_FRAMES frames @ $MODEL_FPS FPS
  model execution:  $RESOLVED_EXECUTION_FRAMES frames
  sim control:      $CONTROL_FRAMES frames @ 50 FPS
  RTC active:       $RTC_ACTIVE
  RTC overlap:      $RESOLVED_RTC_OVERLAP model frames
  RTC frozen:       $RESOLVED_RTC_FROZEN model frames
  RTC ramp power:   $RTC_RAMP_POWER
  persistent sim:   $PERSISTENT_SIM
  deterministic:    $DETERMINISTIC_EVAL
  server timeout:   $SERVER_READY_TIMEOUT seconds
  results:          $RESULTS_DIR
EOF

if [[ "$DRY_RUN" == 1 ]]; then
  echo "Dry run passed; no evaluation was launched."
  exit 0
fi

mkdir -p "$RESULTS_DIR"
cat > "${RESULTS_DIR}/run_config.txt" <<EOF
started_at=$(date -Iseconds)
project=$PROJECT_ROOT
checkpoint=$CHECKPOINT
task_slug=$TASK_SLUG
benchmark_task=$BENCHMARK_TASK
env_config=$ENV_CONFIG_YAML
gpus=${GPUS[*]}
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
persistent_sim=$PERSISTENT_SIM
deterministic_eval=$DETERMINISTIC_EVAL
server_ready_timeout=$SERVER_READY_TIMEOUT
record_video_every_n=$RECORD_VIDEO_EVERY_N
EOF

run_seed() {
  local seed="$1" gpu="$2" slot="$3"
  local seed_dir="${RESULTS_DIR}/seed_${seed}"
  local port=$((PORT_BASE + slot))
  mkdir -p "$seed_dir"
  printf 'seed=%s\ngpu=%s\nport=%s\nstarted_at=%s\n' "$seed" "$gpu" "$port" "$(date -Iseconds)" > "${seed_dir}/run_metadata.txt"

  if env \
    KIMODO_ROOT="$PROJECT_ROOT" \
    HUMANOIDARENA_ROOT="$HUMANOIDARENA_ROOT" \
    CHECKPOINT="$CHECKPOINT" \
    SERVER_PYTHON="$SERVER_PYTHON" \
    SONIC_POLICY_ROOT="$SONIC_POLICY_ROOT" \
    RESULTS_DIR="$seed_dir" \
    TASK="$BENCHMARK_TASK" \
    ENV_CONFIG_YAML="$ENV_CONFIG_YAML" \
    MAX_STEPS="$MAX_STEPS" \
    TEXT_EMBEDDING_CACHE="$TEXT_EMBEDDING_CACHE" \
    ISAAC_DEVICE="cuda:${gpu}" \
    SERVER_DEVICE="cuda:${gpu}" \
    EVAL_SEEDS="$seed" \
    REPEATS_PER_SEED="$REPEATS_PER_SEED" \
    PERSISTENT_SIM="$PERSISTENT_SIM" \
    RECORD_VIDEO_EVERY_N="$RECORD_VIDEO_EVERY_N" \
    SERVER_PORT="$port" \
    KIMODO_SERVER_READY_TIMEOUT="$SERVER_READY_TIMEOUT" \
    KIMODO_DIFFUSION_STEPS="$DIFFUSION_STEPS" \
    KIMODO_EXECUTION_FRAMES="$RESOLVED_EXECUTION_FRAMES" \
    KIMODO_RTC="$RTC_ACTIVE" \
    KIMODO_RTC_OVERLAP_FRAMES="$RESOLVED_RTC_OVERLAP" \
    KIMODO_RTC_FROZEN_FRAMES="$RESOLVED_RTC_FROZEN" \
    KIMODO_RTC_RAMP_POWER="$RTC_RAMP_POWER" \
    KIMODO_DTYPE="$DTYPE" \
    KIMODO_DETERMINISTIC_EVAL="$DETERMINISTIC_EVAL" \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    bash "$EVALUATOR_SCRIPT" --internal-run-task > "${seed_dir}/pipeline.log" 2>&1; then
    date -Iseconds > "${seed_dir}/.done"
  else
    local code=$?
    printf '%s\n' "$code" > "${seed_dir}/.failed"
    return "$code"
  fi
}

failures=0
for ((wave_start=0; wave_start<${#SEEDS[@]}; wave_start+=${#GPUS[@]})); do
  pids=()
  labels=()
  for ((slot=0; slot<${#GPUS[@]}; slot++)); do
    index=$((wave_start + slot))
    (( index < ${#SEEDS[@]} )) || break
    seed="${SEEDS[$index]}"
    gpu="${GPUS[$slot]}"
    echo "Launching seed=$seed on GPU=$gpu port=$((PORT_BASE + slot))"
    run_seed "$seed" "$gpu" "$slot" &
    pids+=("$!")
    labels+=("seed=$seed,gpu=$gpu")
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "Completed ${labels[$i]}"
    else
      echo "Failed ${labels[$i]}" >&2
      failures=$((failures + 1))
    fi
  done
done

"$SERVER_PYTHON" - "$RESULTS_DIR" <<'PY_AGGREGATE'
import collections
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
per_seed = []
reason_counts = collections.Counter()
for summary_path in sorted(root.glob("seed_*/summary.json")):
    data = json.loads(summary_path.read_text())
    seed_name = summary_path.parent.name.removeprefix("seed_")
    episodes = int(data.get("total_episodes", data.get("episodes", 0)))
    successes = int(data.get("total_successes", data.get("successes", 0)))
    failures = int(data.get("total_failures", data.get("failures", episodes - successes)))
    reason_counts.update(data.get("result_reason_counts", data.get("failure_reason_counts", {})))
    per_seed.append({"seed": int(seed_name), "episodes": episodes, "successes": successes,
                     "failures": failures, "success_rate": successes / episodes if episodes else 0.0})
total_episodes = sum(item["episodes"] for item in per_seed)
total_successes = sum(item["successes"] for item in per_seed)
payload = {"total_episodes": total_episodes, "total_successes": total_successes,
           "total_failures": total_episodes - total_successes,
           "overall_success_rate": total_successes / total_episodes if total_episodes else 0.0,
           "result_reason_counts": dict(reason_counts), "per_seed": per_seed}
(root / "aggregate_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY_AGGREGATE

if (( failures > 0 )); then
  printf '%s\n' "$failures" > "${RESULTS_DIR}/.failed"
  echo "Evaluation finished with $failures failed seed job(s): $RESULTS_DIR" >&2
  exit 1
fi

date -Iseconds > "${RESULTS_DIR}/.done"
echo "Evaluation complete: $RESULTS_DIR"
