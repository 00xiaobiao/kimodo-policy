#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REDIS_IP="${REDIS_IP:-localhost}"
ACTUAL_HUMAN_HEIGHT="${ACTUAL_HUMAN_HEIGHT:-1.72}"
GMR_ROOT="${GMR_ROOT:-${REPO_ROOT}/GMR}"

if [[ -d "${GMR_ROOT}/general_motion_retargeting" ]]; then
  export GMR_ROOT
  export PYTHONPATH="${GMR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

cd "${SCRIPT_DIR}"
python twist2_teleop_server.py \
  --robot unitree_g1 \
  --actual_human_height "${ACTUAL_HUMAN_HEIGHT}" \
  --redis_ip "${REDIS_IP}" \
  --target_fps 100 \
  --measure_fps 1 \
  "$@"
