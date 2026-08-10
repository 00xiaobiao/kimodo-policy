#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
KIMODO_ENV="${KIMODO_ENV:-/home/CONNECT/yfang870/miniconda3/envs/kimodo}"

export CUDA_VISIBLE_DEVICES="${KIMODO_GPUS:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
"${KIMODO_ENV}/bin/accelerate" launch \
  --multi_gpu \
  --num_processes 8 \
  --mixed_precision bf16 \
  --main_process_port "${KIMODO_MASTER_PORT:-29650}" \
  train.py \
  --config "${SCRIPT_DIR}/pt_UnifoLM_HumanoidEveryday_HIW_gbs1024_20w.yaml"


# UNIFOLM_ROOT=/实际路径/UnifoLM_WBT_Dataset \
# HUMANOID_EVERYDAY_ROOT=/实际路径/HumanoidEveryday \
# HIW500_ROOT=/实际路径/HIW500-LeRobot \
# bash scripts/pt_UnifoLM_HumanoidEveryday_HIW_gbs1024_20w.sh