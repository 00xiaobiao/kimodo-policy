#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

for arg in "$@"; do
  case "${arg}" in
    --transport|--port|--zmq_host|--zmq_port|--zmq*)
      echo "run_sonic_teleop_server.sh is Redis-only. Remove deprecated ZMQ option: ${arg}" >&2
      exit 1
      ;;
  esac
done

cd "${SCRIPT_DIR}"
#python pico_server_pose_only.py \
#  --redis_host "${REDIS_HOST}" \
#  --redis_port "${REDIS_PORT}" \
#  --vis_vr3pt \
#  --vis_smpl \
#  "$@"

python pico_server/pico_server_pose_only.py \
  --redis_host "${REDIS_HOST}" \
  --redis_port "${REDIS_PORT}" \
  "$@"
