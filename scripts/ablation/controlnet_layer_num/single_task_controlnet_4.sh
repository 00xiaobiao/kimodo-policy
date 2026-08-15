#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_SLUG="${1:-${KIMODO_ARENA_TASK:-task}}"
BACKEND_SLUG="${2:-${KIMODO_ARENA_BACKEND:-sonic}}"
RUN_NAME="single_task_controlnet_4_${TASK_SLUG}_${BACKEND_SLUG}"
export KIMODO_CONFIG="${SCRIPT_DIR}/single_task_controlnet_4.yaml"
export KIMODO_SINGLE_TASK_SAVE_ROOT="${KIMODO_SINGLE_TASK_SAVE_ROOT:-log/ablation/${RUN_NAME}/}"
export KIMODO_SINGLE_TASK_RUN_NAME="${KIMODO_SINGLE_TASK_RUN_NAME:-${RUN_NAME}}"
export KIMODO_MASTER_PORT="${KIMODO_MASTER_PORT:-29664}"

exec "${SCRIPT_DIR}/../../only_HumanoidArena_single_task.sh" "$@"

# conda activate kimodo-env
# export KIMODO_ENV="$CONDA_PREFIX"
# KIMODO_GPUS=0,1,2,3 \
# HUMANOID_ARENA_ROOT=/path/to/HumanoidArena_dataset_v3_1 \
# bash scripts/ablation/controlnet_layer_num/single_task_controlnet_4.sh doubledesk sonic
