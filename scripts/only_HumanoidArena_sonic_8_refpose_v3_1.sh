#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
KIMODO_ENV="${KIMODO_ENV:-/home/CONNECT/yfang870/miniconda3/envs/kimodo}"
CONFIG_PATH="${KIMODO_CONFIG:-${SCRIPT_DIR}/only_HumanoidArena_sonic_8_refpose_v3_1.yaml}"
GPU_LIST="${KIMODO_GPUS:-0,1,2,3,4,5,6,7}"

IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NUM_PROCESSES="${#GPU_IDS[@]}"
if (( NUM_PROCESSES < 1 )); then
  echo "KIMODO_GPUS must contain at least one GPU index" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
"${KIMODO_ENV}/bin/accelerate" launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  --main_process_port "${KIMODO_MASTER_PORT:-29652}" \
  train.py \
  --config "${CONFIG_PATH}"
