#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -n "${KIMODO_ENV:-}" ]]; then
  ACCELERATE_BIN="${KIMODO_ENV}/bin/accelerate"
else
  ACCELERATE_BIN="$(command -v accelerate || true)"
fi
if [[ -z "${ACCELERATE_BIN}" || ! -x "${ACCELERATE_BIN}" ]]; then
  echo "accelerate was not found; activate kimodo-env or set KIMODO_ENV=/path/to/env" >&2
  exit 1
fi

CONFIG_PATH="${KIMODO_CONFIG:-${SCRIPT_DIR}/real_world_bottle_packing_gbs64_5w_controlnet4_detach_true_mse.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Training config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi

# Real-world training is initialized from a compatible Kimodo checkpoint.
# Accept either the first positional argument or KIMODO_INIT_CHECKPOINT, with
# the positional argument taking precedence just like the fine-tuning launchers.
INIT_CHECKPOINT="${1:-${KIMODO_INIT_CHECKPOINT:-}}"
if [[ -z "${INIT_CHECKPOINT}" ]]; then
  echo "Usage: bash $0 /path/to/pretrain/checkpoint_STEP" >&2
  echo "Or set KIMODO_INIT_CHECKPOINT=/path/to/pretrain/checkpoint_STEP" >&2
  exit 2
fi
if [[ ! -f "${INIT_CHECKPOINT}/training_state.pt" || ! -f "${INIT_CHECKPOINT}/config.json" ]]; then
  echo "Initialization checkpoint is incomplete: ${INIT_CHECKPOINT}" >&2
  echo "Expected training_state.pt and config.json in that directory." >&2
  exit 2
fi

GPU_LIST="${KIMODO_GPUS:-0,1,2,3}"
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
"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  --main_process_port "${KIMODO_MASTER_PORT:-29653}" \
  train.py \
  --config "${CONFIG_PATH}" \
  --init-checkpoint "${INIT_CHECKPOINT}"

# Example:
# conda activate kimodo
# export KIMODO_ENV="$CONDA_PREFIX"
# export REAL_WORLD_ROOT=/data/local-data/data/code/yunhengwang/kimodo-polocy/controlnet_v1.2/real-world
# KIMODO_GPUS=4,5,6,7 \
# KIMODO_INIT_CHECKPOINT=/data/local-data/data/code/yunhengwang/kimodo-polocy/controlnet_v1.2/log/ablation/multi_task_gbs128_50w_controlnet4_detach_true_mse/2026-08-23_18-26-31/checkpoint_500000 \
# bash scripts/experiments/real_world/real_world_bottle_packing_gbs64_5w_controlnet4_detach_true_mse.sh
