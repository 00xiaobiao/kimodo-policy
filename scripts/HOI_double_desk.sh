#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
KIMODO_ENV="${KIMODO_ENV:-/home/CONNECT/yfang870/miniconda3/envs/kimodo}"
ACCELERATE="${KIMODO_ENV}/bin/accelerate"
PYTHON="${KIMODO_ENV}/bin/python"
CONFIG_PATH="${KIMODO_CONFIG:-${PROJECT_ROOT}/train.yaml}"
GPU_LIST="${KIMODO_GPUS:-4,5,6,7}"
MASTER_PORT="${KIMODO_MASTER_PORT:-29647}"

IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NUM_PROCESSES="${#GPU_IDS[@]}"
if (( NUM_PROCESSES < 1 )); then
  echo "KIMODO_GPUS must contain at least one GPU index" >&2
  exit 2
fi
if [[ ! -x "${ACCELERATE}" || ! -x "${PYTHON}" ]]; then
  echo "kimodo environment is incomplete: ${KIMODO_ENV}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Training config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_DIR="${SCRIPT_DIR}/logs"
LAUNCH_LOG="${LAUNCH_LOG_DIR}/HOI_double_desk_${RUN_TIMESTAMP}.log"
INDUCTOR_CACHE="${KIMODO_INDUCTOR_CACHE:-/tmp/kimodo_torchinductor_HOI_double_desk}"
mkdir -p "${LAUNCH_LOG_DIR}" "${INDUCTOR_CACHE}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCHINDUCTOR_CACHE_DIR="${INDUCTOR_CACHE}"

echo "project=${PROJECT_ROOT}"
echo "config=${CONFIG_PATH}"
echo "physical_gpus=${CUDA_VISIBLE_DEVICES} processes=${NUM_PROCESSES}"
echo "environment=${KIMODO_ENV}"
echo "launcher_log=${LAUNCH_LOG}"

cd "${PROJECT_ROOT}"
"${ACCELERATE}" launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  --main_process_port "${MASTER_PORT}" \
  "${PROJECT_ROOT}/train.py" \
  --config "${CONFIG_PATH}" \
  2>&1 | tee -a "${LAUNCH_LOG}"
