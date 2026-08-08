#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TWIST2="${RUN_TWIST2:-1}"
RUN_SONIC="${RUN_SONIC:-1}"

TWIST2_PARALLEL_JOBS="${TWIST2_PARALLEL_JOBS:-2}"
SONIC_PARALLEL_JOBS="${SONIC_PARALLEL_JOBS:-2}"
TWIST2_INPUT_ROOT="${TWIST2_INPUT_ROOT:-}"
SONIC_SOURCE_ROOT="${SONIC_SOURCE_ROOT:-}"

IMAGE_PORT_STRIDE="${IMAGE_PORT_STRIDE:-10}"
IMAGE_XROBOT_PORT_STRIDE="${IMAGE_XROBOT_PORT_STRIDE:-10}"

TWIST2_IMAGE_PORT_BASE="${TWIST2_IMAGE_PORT_BASE:-5600}"
SONIC_IMAGE_PORT_BASE="${SONIC_IMAGE_PORT_BASE:-5800}"

TWIST2_IMAGE_XROBOT_PORT_BASE="${TWIST2_IMAGE_XROBOT_PORT_BASE:-12400}"
SONIC_IMAGE_XROBOT_PORT_BASE="${SONIC_IMAGE_XROBOT_PORT_BASE:-12600}"

TWIST2_SHM_PREFIX="${TWIST2_SHM_PREFIX:-isaac_multi_image_shm_twist2}"
SONIC_SHM_PREFIX="${SONIC_SHM_PREFIX:-isaac_multi_image_shm_sonic}"

declare -a child_pids=()

cleanup() {
  for pid in "${child_pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

launch_twist2() {
  local cmd=(
    "${PYTHON_BIN}" tools/data_tools/rerecord_twist2_recordings_to_multicam.py
  )
  if [ -n "${TWIST2_INPUT_ROOT}" ]; then
    cmd+=("${TWIST2_INPUT_ROOT}")
  fi
  cmd+=(
    --parallel-jobs "${TWIST2_PARALLEL_JOBS}"
    --image-port-base "${TWIST2_IMAGE_PORT_BASE}"
    --image-port-stride "${IMAGE_PORT_STRIDE}"
    --image-xrobot-port-base "${TWIST2_IMAGE_XROBOT_PORT_BASE}"
    --image-xrobot-port-stride "${IMAGE_XROBOT_PORT_STRIDE}"
    --shm-prefix "${TWIST2_SHM_PREFIX}"
  )
  "${cmd[@]}"
}

launch_sonic() {
  local cmd=(
    "${PYTHON_BIN}" tools/data_tools/rerecord_sonic_recordings_to_multicam.py
  )
  if [ -n "${SONIC_SOURCE_ROOT}" ]; then
    cmd+=("${SONIC_SOURCE_ROOT}")
  fi
  cmd+=(
    --parallel-jobs "${SONIC_PARALLEL_JOBS}"
    --image-port-base "${SONIC_IMAGE_PORT_BASE}"
    --image-port-stride "${IMAGE_PORT_STRIDE}"
    --image-xrobot-port-base "${SONIC_IMAGE_XROBOT_PORT_BASE}"
    --image-xrobot-port-stride "${IMAGE_XROBOT_PORT_STRIDE}"
    --shm-prefix "${SONIC_SHM_PREFIX}"
  )
  "${cmd[@]}"
}

if [ "${RUN_TWIST2}" != "1" ] && [ "${RUN_SONIC}" != "1" ]; then
  echo "Nothing to run: RUN_TWIST2=${RUN_TWIST2} RUN_SONIC=${RUN_SONIC}" >&2
  exit 1
fi

echo "[run_rerecord] PYTHON_BIN=${PYTHON_BIN}"
echo "[run_rerecord] twist2 input_root=${TWIST2_INPUT_ROOT:-<script default>}"
echo "[run_rerecord] twist2 workers=${TWIST2_PARALLEL_JOBS} port_base=${TWIST2_IMAGE_PORT_BASE} shm_prefix=${TWIST2_SHM_PREFIX}"
echo "[run_rerecord] sonic  source_root=${SONIC_SOURCE_ROOT:-<script default>}"
echo "[run_rerecord] sonic  workers=${SONIC_PARALLEL_JOBS} port_base=${SONIC_IMAGE_PORT_BASE} shm_prefix=${SONIC_SHM_PREFIX}"

twist2_pid=""
sonic_pid=""

if [ "${RUN_TWIST2}" = "1" ]; then
  launch_twist2 &
  twist2_pid=$!
  child_pids+=("${twist2_pid}")
fi

if [ "${RUN_SONIC}" = "1" ]; then
  launch_sonic &
  sonic_pid=$!
  child_pids+=("${sonic_pid}")
fi

rc=0

if [ -n "${twist2_pid}" ]; then
  if ! wait "${twist2_pid}"; then
    rc=$?
  fi
fi

if [ -n "${sonic_pid}" ]; then
  if ! wait "${sonic_pid}"; then
    rc=$?
  fi
fi

exit "${rc}"
