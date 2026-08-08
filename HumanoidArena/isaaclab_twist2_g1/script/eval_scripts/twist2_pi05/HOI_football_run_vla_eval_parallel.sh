#!/usr/bin/env bash

set -euo pipefail

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${RUN_SCRIPT_DIR}/../../.." && pwd)"
source "${RUN_SCRIPT_DIR}/../common/model_batch_utils.sh"

CONDA_BASE="${CONDA_BASE:-/ai/Yichi/0_Systems/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-unitree_sim_env}"
AUTO_ACTIVATE_CONDA="${AUTO_ACTIVATE_CONDA:-1}"
if [[ "${AUTO_ACTIVATE_CONDA}" == "1" ]]; then
  TARGET_CONDA_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV_NAME}"
  if [[ "${CONDA_PREFIX:-}" != "${TARGET_CONDA_PREFIX}" ]]; then
    export ZSH_VERSION="${ZSH_VERSION:-}"
    export PYTHONPATH="${PYTHONPATH:-}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
  fi
fi

EVAL_PYTHON="${EVAL_PYTHON:-python}"

resolve_config_path() {
  local config_path="$1"
  if [[ "$config_path" = /* ]]; then
    printf "%s" "$config_path"
  else
    printf "%s" "${ISAACLAB_ROOT}/$config_path"
  fi
}

export ROBOT_USD_OVERRIDE="${ISAACLAB_ROOT}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_test_config/base_test/football_single_twist2_test.yaml}"
ENV_CONFIG_YAML="$(resolve_config_path "${ENV_CONFIG_YAML}")"
ISAAC_DEVICE="${ISAAC_DEVICE:-cuda}"
HEADLESS="${HEADLESS:-1}"
ENABLE_DEPTH="${ENABLE_DEPTH:-0}"
MAX_STEPS="${MAX_STEPS:-2000}"
VIDEO_FPS="${VIDEO_FPS:-30}"
POST_TERMINATION_RECORD_STEPS="${POST_TERMINATION_RECORD_STEPS:-10}"
RECORD_VIDEO_EVERY_N="${RECORD_VIDEO_EVERY_N:-10}"
STEP_LOG_EVERY_N="${STEP_LOG_EVERY_N:-100}"
SIM_VERBOSE_STARTUP="${SIM_VERBOSE_STARTUP:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-15001}"
SERVER_PORT_MAX="${SERVER_PORT_MAX:-20000}"
ROBOT_TYPE="${ROBOT_TYPE:-unitree_g1_refpose_v3_1}"

TWIST2_MODEL_PATH="${TWIST2_MODEL_PATH:-${ISAACLAB_ROOT}/../TWIST2/assets/ckpts/twist2_1017_20k.onnx}"

SERVER_PYTHON="${SERVER_PYTHON:-/ai/Yichi/0_Systems/miniconda3/envs/lerobot2/bin/python}"
SERVER_SCRIPT="${SERVER_SCRIPT:-${ISAACLAB_ROOT}/../lerobot/scripts/serve_lerobot_vla_http.py}"
SERVER_GPU_IDS="${SERVER_GPU_IDS:-0,1,2,3,4,5,6,7}"
SERVER_DEVICE="${SERVER_DEVICE:-cuda:0}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_SCHEME="${SERVER_SCHEME:-http}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-360}"
LEROBOT_SERVER_TIMEOUT="${LEROBOT_SERVER_TIMEOUT:-360.0}"
LEROBOT_VERIFY_SSL="${LEROBOT_VERIFY_SSL:-0}"
TLS_CERT_FILE="${TLS_CERT_FILE:-}"
TLS_KEY_FILE="${TLS_KEY_FILE:-}"

MODEL_ROOT="${MODEL_ROOT:-$DEFAULT_BATCH_MODEL_ROOT}"
MODEL_GLOB="${MODEL_GLOB:-}"
RESULTS_TAG_BASE="${RESULTS_TAG:-HOI_football_twist2_mix_1300_60}"
if [[ -n "${RESULTS_TAG_PREFIX:-}" ]]; then
  RESULTS_TAG="${RESULTS_TAG_PREFIX}_${RESULTS_TAG_BASE}"
else
  RESULTS_TAG="${RESULTS_TAG_BASE}"
fi
RESUME_LATEST="${RESUME_LATEST:-1}"
DRY_RUN="${DRY_RUN:-0}"

resolve_results_dir() {
  local results_root="${RUN_SCRIPT_DIR}/eval_results"
  mkdir -p "${results_root}"

  if [[ -n "${RESULTS_DIR:-}" ]]; then
    printf "%s" "${RESULTS_DIR}"
    return 0
  fi

  if [[ "${RESUME_LATEST}" == "1" ]]; then
    local latest_dir=""
    latest_dir="$(find "${results_root}" -maxdepth 1 -type d -name "${RESULTS_TAG}_*" | sort | tail -n 1)"
    if [[ -n "${latest_dir}" ]]; then
      printf "%s" "${latest_dir}"
      return 0
    fi
  fi

  printf "%s" "${results_root}/${RESULTS_TAG}_$(date +%Y%m%d_%H%M%S)"
}

load_task_name_from_yaml() {
  "${EVAL_PYTHON}" - "${ISAACLAB_ROOT}/tasks/common_env_config/loader.py" "${1}" <<'PY2'
import importlib.util
import pathlib
import sys

loader_path = pathlib.Path(sys.argv[1])
config_path = sys.argv[2]
spec = importlib.util.spec_from_file_location("common_env_config_loader", loader_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

task_name = module.get_env_config_task_name(config_path)
if not task_name:
    raise SystemExit(
        f"Error: env config YAML must define a top-level task_name: {config_path}"
    )
print(task_name)
PY2
}

load_vision_randomization_from_yaml() {
  "${EVAL_PYTHON}" - "${1}" <<'PY3'
import json
import pathlib
import sys
import yaml
config_path = pathlib.Path(sys.argv[1])
raw_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
test_defaults = raw_cfg.get("test_defaults", {})
vis_cfg = test_defaults.get("vision_randomization", {})
if isinstance(vis_cfg, dict) and vis_cfg.get("enabled"):
    print(json.dumps(vis_cfg))
else:
    print("")
PY3
}


load_test_default_from_yaml() {
  "${EVAL_PYTHON}" - "${1}" "${2}" "${3}" <<'PY2'
import pathlib
import sys
import yaml

config_path = pathlib.Path(sys.argv[1])
field_name = sys.argv[2]
default_value = sys.argv[3]
raw_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
test_defaults = raw_cfg.get("test_defaults", {})
value = test_defaults.get(field_name, default_value)
if isinstance(value, list):
    print(" ".join(str(item) for item in value))
elif isinstance(value, bool):
    print("1" if value else "0")
else:
    print(value)
PY2
}



DEFAULT_REPEATS_PER_SEED="$(load_test_default_from_yaml "${ENV_CONFIG_YAML}" repeats_per_seed 40)"
REPEATS_PER_SEED="${REPEATS_PER_SEED:-${DEFAULT_REPEATS_PER_SEED}}"
DEFAULT_SEEDS="$(load_test_default_from_yaml "${ENV_CONFIG_YAML}" seeds "0 1 2 3 4")"
SEEDS=(${SEEDS_OVERRIDE:-${DEFAULT_SEEDS}})
PERSISTENT_SIM="${PERSISTENT_SIM:-$(load_test_default_from_yaml "${ENV_CONFIG_YAML}" persistent_sim 1)}"
echo "Seeds: ${SEEDS[*]}"
DEFAULT_VISION_RANDOMIZATION="$(load_vision_randomization_from_yaml "${ENV_CONFIG_YAML}")"
export VISION_RANDOMIZATION="${VISION_RANDOMIZATION:-${DEFAULT_VISION_RANDOMIZATION}}"
if [[ -n "${VISION_RANDOMIZATION}" ]]; then
  echo "[vision_test] enabled; config=$VISION_RANDOMIZATION"
fi


TASK_NAME="${TASK_NAME:-$(load_task_name_from_yaml "${ENV_CONFIG_YAML}")}"
discover_model_paths "${MODEL_GLOB}"
print_model_paths_summary

RESULTS_DIR="$(resolve_results_dir)"
echo "Task: ${TASK_NAME}"
echo "Results dir: ${RESULTS_DIR}"
if [[ -d "${RESULTS_DIR}/episodes" ]]; then
  echo "Resume mode: reuse existing results in ${RESULTS_DIR}"
fi

ARGS=(
  --task "${TASK_NAME}"
  --env_config_yaml "${ENV_CONFIG_YAML}"
  --repeats_per_seed "${REPEATS_PER_SEED}"
  --max_steps "${MAX_STEPS}"
  --video_fps "${VIDEO_FPS}"
  --post_termination_record_steps "${POST_TERMINATION_RECORD_STEPS}"
  --record_video_every_n "${RECORD_VIDEO_EVERY_N}"
  --step_log_every_n "${STEP_LOG_EVERY_N}"
  --num_workers "${NUM_WORKERS}"
  --server_port_base "${SERVER_PORT_BASE}"
  --server_port_max "${SERVER_PORT_MAX}"
  --robot_type "${ROBOT_TYPE}"
  --twist2_model_path "${TWIST2_MODEL_PATH}"
  --results_dir "${RESULTS_DIR}"
  --isaac_device "${ISAAC_DEVICE}"
  --server_python "${SERVER_PYTHON}"
  --server_script "${SERVER_SCRIPT}"
  --server_device "${SERVER_DEVICE}"
  --server_gpu_ids "${SERVER_GPU_IDS}"
  --server_host "${SERVER_HOST}"
  --server_scheme "${SERVER_SCHEME}"
  --server_ready_timeout "${SERVER_READY_TIMEOUT}"
  --lerobot_server_timeout "${LEROBOT_SERVER_TIMEOUT}"
  --persistent_sim "${PERSISTENT_SIM}"
)

if [[ "${HEADLESS}" == "1" ]]; then
  ARGS+=(--headless)
fi

if [[ "${SIM_VERBOSE_STARTUP}" == "1" ]]; then
  ARGS+=(--verbose_startup)
fi

if [[ "${LEROBOT_VERIFY_SSL}" == "1" ]]; then
  ARGS+=(--lerobot_server_verify_ssl)
fi

if [[ -n "${TLS_CERT_FILE}" ]]; then
  ARGS+=(--tls_cert_file "${TLS_CERT_FILE}")
fi

if [[ -n "${TLS_KEY_FILE}" ]]; then
  ARGS+=(--tls_key_file "${TLS_KEY_FILE}")
fi

for seed in "${SEEDS[@]}"; do
  ARGS+=(--seed "${seed}")
done

for model_path in "${MODEL_PATHS[@]}"; do
  ARGS+=(--model-path "${model_path}")
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1, skip evaluator launch."
  exit 0
fi

cd "${RUN_SCRIPT_DIR}"
export ENABLE_DEPTH
"${EVAL_PYTHON}" eval_vla_suite_parallel.py "${ARGS[@]}"
