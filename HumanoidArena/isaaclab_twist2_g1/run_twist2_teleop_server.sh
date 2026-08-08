#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REDIS_IP="${REDIS_IP:-localhost}"
ACTUAL_HUMAN_HEIGHT="${ACTUAL_HUMAN_HEIGHT:-1.76}"
GMR_ROOT="${GMR_ROOT:-${REPO_ROOT}/GMR}"

if [[ -d "${GMR_ROOT}/general_motion_retargeting" ]]; then
  export GMR_ROOT
  export PYTHONPATH="${GMR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

# --target_backend sonic_joint29 \ # twist2 / sonic_joint29
cd "${SCRIPT_DIR}"
python pico_server/twist2_teleop_server.py \
  --robot unitree_g1 \
  --actual_human_height "${ACTUAL_HUMAN_HEIGHT}" \
  --redis_ip "${REDIS_IP}" \
  --target_fps 100 \
  --measure_fps 1 \
  "$@"
