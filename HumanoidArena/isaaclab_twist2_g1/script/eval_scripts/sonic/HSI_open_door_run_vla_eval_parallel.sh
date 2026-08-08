#!/usr/bin/env bash
set -euo pipefail

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${RUN_SCRIPT_DIR}/../../.." && pwd)"

export ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_test_config/base_test/open_door_sonic_test.yaml}"
RESULTS_TAG_BASE="${RESULTS_TAG:-1test_HSI_open_door_sonic_batch_2000_mix}"
if [[ -n "${RESULTS_TAG_PREFIX:-}" ]]; then
  export RESULTS_TAG="${RESULTS_TAG_PREFIX}_${RESULTS_TAG_BASE}"
else
  export RESULTS_TAG="${RESULTS_TAG_BASE}"
fi
export MAX_STEPS="${MAX_STEPS:-1800}"
exec "${RUN_SCRIPT_DIR}/run_vla_eval_parallel.sh"
