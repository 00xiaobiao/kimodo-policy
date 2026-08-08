#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT_BASE="${MODEL_ROOT_BASE:-}"
if [[ -z "${MODEL_ROOT_BASE}" ]]; then
  echo "[batch_pi05] MODEL_ROOT_BASE is required. Download the released checkpoints and set MODEL_ROOT_BASE=/path/to/humanoidarena_checkpoints" >&2
  exit 2
fi
PI05_MODEL_ROOT="${PI05_MODEL_ROOT:-${MODEL_ROOT_BASE}/pi}"

TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_boxing/pi05_sonic_boxing_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_base bash "${SCRIPT_DIR}/task/pi05_batch_test_boxing.sh"
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_boxing/pi05_sonic_boxing_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_semantic bash "${SCRIPT_DIR}/task/pi05_batch_test_boxing.sh"
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_boxing/pi05_sonic_boxing_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_vision bash "${SCRIPT_DIR}/task/pi05_batch_test_boxing.sh"
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_boxing/pi05_sonic_boxing_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_execution bash "${SCRIPT_DIR}/task/pi05_batch_test_boxing.sh"


TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HOI_football/pi05_sonic_football_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_base bash "${SCRIPT_DIR}/task/pi05_batch_test_football.sh"
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HOI_football/pi05_sonic_football_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_semantic bash "${SCRIPT_DIR}/task/pi05_batch_test_football.sh"
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HOI_football/pi05_sonic_football_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_vision bash "${SCRIPT_DIR}/task/pi05_batch_test_football.sh"
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HOI_football/pi05_sonic_football_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_execution bash "${SCRIPT_DIR}/task/pi05_batch_test_football.sh"


TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HOI_pp_box/pi05_sonic_ppbox_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_base bash "${SCRIPT_DIR}/task/pi05_batch_test_pp_box.sh"
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HOI_pp_box/pi05_sonic_ppbox_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_semantic bash "${SCRIPT_DIR}/task/pi05_batch_test_pp_box.sh"
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HOI_pp_box/pi05_sonic_ppbox_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_vision bash "${SCRIPT_DIR}/task/pi05_batch_test_pp_box.sh"
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HOI_pp_box/pi05_sonic_ppbox_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_execution bash "${SCRIPT_DIR}/task/pi05_batch_test_pp_box.sh"


TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HOI_double_desk/pi05_sonic_doubledesk_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_base bash "${SCRIPT_DIR}/task/pi05_batch_test_doubledesk.sh"
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HOI_double_desk/pi05_sonic_doubledesk_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_semantic bash "${SCRIPT_DIR}/task/pi05_batch_test_doubledesk.sh"
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HOI_double_desk/pi05_sonic_doubledesk_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_vision bash "${SCRIPT_DIR}/task/pi05_batch_test_doubledesk.sh"
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HOI_double_desk/pi05_sonic_doubledesk_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_execution bash "${SCRIPT_DIR}/task/pi05_batch_test_doubledesk.sh"


TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_open_door/pi05_sonic_opendoor_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_base bash "${SCRIPT_DIR}/task/pi05_batch_test_open_door.sh"
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_open_door/pi05_sonic_opendoor_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_semantic bash "${SCRIPT_DIR}/task/pi05_batch_test_open_door.sh"
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_open_door/pi05_sonic_opendoor_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_vision bash "${SCRIPT_DIR}/task/pi05_batch_test_open_door.sh"
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_open_door/pi05_sonic_opendoor_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_execution bash "${SCRIPT_DIR}/task/pi05_batch_test_open_door.sh"


TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_sit_sofa/pi05_sonic_sitsofa_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_base bash "${SCRIPT_DIR}/task/pi05_batch_test_sit_sofa.sh"
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_sit_sofa/pi05_sonic_sitsofa_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_semantic bash "${SCRIPT_DIR}/task/pi05_batch_test_sit_sofa.sh"
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_sit_sofa/pi05_sonic_sitsofa_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_vision bash "${SCRIPT_DIR}/task/pi05_batch_test_sit_sofa.sh"
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_sit_sofa/pi05_sonic_sitsofa_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_execution bash "${SCRIPT_DIR}/task/pi05_batch_test_sit_sofa.sh"


TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_vision_navi/pi05_sonic_visionnavi_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_base bash "${SCRIPT_DIR}/task/pi05_batch_test_vision_navi.sh"
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_vision_navi/pi05_sonic_visionnavi_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_semantic bash "${SCRIPT_DIR}/task/pi05_batch_test_vision_navi.sh"
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_vision_navi/pi05_sonic_visionnavi_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_vision bash "${SCRIPT_DIR}/task/pi05_batch_test_vision_navi.sh"
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="${PI05_MODEL_ROOT}/HSI_vision_navi/pi05_sonic_visionnavi_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_execution bash "${SCRIPT_DIR}/task/pi05_batch_test_vision_navi.sh"

