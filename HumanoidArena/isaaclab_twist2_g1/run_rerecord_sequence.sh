#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/script/common/runtime_paths.sh"

"${ISAACLAB_PYTHON}" "${ISAACLAB_ROOT}/tools/data_tools/rerecord_twist2_recordings_to_multicam.py"
"${ISAACLAB_PYTHON}" "${ISAACLAB_ROOT}/tools/data_tools/rerecord_sonic_recordings_to_multicam.py"
