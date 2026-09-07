#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Train the 4-layer ControlNet continuous-hand experiment on one SIMPLE task.

Usage:
  bash scripts/experiments/simple_single_gbs64_8w_controlnet4_detach_true_mse_continuous_hand.sh [TASK]

TASK defaults to G1WholebodyXMovePickTeleop-v0. The expected directory is
${KIMODO_SIMPLE_ROOT:-/data/local-data/data/Humanoid/Simple}/TASK/episode_XXXXXX.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TASK="${1:-${KIMODO_SIMPLE_TASK:-G1WholebodyXMovePickTeleop-v0}}"
if [[ -z "${TASK}" || "${TASK}" == */* ]]; then
  echo "TASK must be one SIMPLE task directory name, got: ${TASK}" >&2
  exit 2
fi

if [[ -n "${KIMODO_ENV:-}" ]]; then
  PYTHON_BIN="${KIMODO_ENV}/bin/python"
else
  PYTHON_BIN="$(command -v python || command -v python3 || true)"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "python was not found; activate the training environment or set KIMODO_ENV=/path/to/env" >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c 'import accelerate' >/dev/null 2>&1; then
  echo "accelerate is not installed in ${PYTHON_BIN}" >&2
  exit 1
fi

CONFIG_PATH="${KIMODO_CONFIG:-${SCRIPT_DIR}/simple_single_gbs64_8w_controlnet4_detach_true_mse_continuous_hand.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Training config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi

SIMPLE_ROOT="${KIMODO_SIMPLE_ROOT:-/data/local-data/data/Humanoid/Simple}"
if [[ ! -d "${SIMPLE_ROOT}/${TASK}" ]]; then
  echo "SIMPLE task directory does not exist: ${SIMPLE_ROOT}/${TASK}" >&2
  exit 2
fi

GPU_LIST="${KIMODO_GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NUM_PROCESSES="${#GPU_IDS[@]}"
if (( NUM_PROCESSES < 1 )); then
  echo "KIMODO_GPUS must contain at least one GPU index" >&2
  exit 2
fi

RUN_NAME="simple_single_gbs64_8w_controlnet4_detach_true_mse_continuous_hand_${TASK}"
export KIMODO_SIMPLE_TASK="${TASK}"
export KIMODO_SIMPLE_ROOT="${SIMPLE_ROOT}"
export KIMODO_SIMPLE_RUN_NAME="${KIMODO_SIMPLE_RUN_NAME:-${RUN_NAME}}"
export KIMODO_SIMPLE_SAVE_ROOT="${KIMODO_SIMPLE_SAVE_ROOT:-log/experiments/${KIMODO_SIMPLE_RUN_NAME}/}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "SIMPLE task: ${TASK}"
echo "SIMPLE root: ${SIMPLE_ROOT}"
echo "GPUs: ${GPU_LIST}"
echo "Python: ${PYTHON_BIN}"
echo "Config: ${CONFIG_PATH}"
echo "Save root: ${KIMODO_SIMPLE_SAVE_ROOT}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  --main_process_port "${KIMODO_MASTER_PORT:-29659}" \
  train.py \
  --config "${CONFIG_PATH}"


# cd /data/local-data/data/code/yunhengwang/kimodo-polocy/controlnet_v1.2
# conda activate kimodo
# KIMODO_GPUS=0,1,2,3 \
# KIMODO_SIMPLE_ROOT=/data/local-data/data/Humanoid/Simple \
# KIMODO_SIMPLE_TASK=G1WholebodyXMovePickTeleop-v0 \
# bash scripts/experiments/simple_single_gbs64_8w_controlnet4_detach_true_mse_continuous_hand.sh
