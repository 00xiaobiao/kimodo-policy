#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${REPO_ROOT}/isaaclab_twist2_g1/pico_server/run_twist2_teleop_server.sh" "$@"
