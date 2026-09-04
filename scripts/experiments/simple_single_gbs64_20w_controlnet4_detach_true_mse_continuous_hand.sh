#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

TASK="${1:-${KIMODO_SIMPLE_TASK:-G1WholebodyXMovePickTeleop-v0}}"
export KIMODO_CONFIG="${KIMODO_CONFIG:-${SCRIPT_DIR}/simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand.yaml}"
export KIMODO_SIMPLE_RUN_NAME="${KIMODO_SIMPLE_RUN_NAME:-simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand_${TASK}}"
export KIMODO_SIMPLE_SAVE_ROOT="${KIMODO_SIMPLE_SAVE_ROOT:-log/experiments/${KIMODO_SIMPLE_RUN_NAME}/}"

exec bash "${SCRIPT_DIR}/simple_single_gbs64_20w_controlnet4_detach_true_mse.sh" "${TASK}"
