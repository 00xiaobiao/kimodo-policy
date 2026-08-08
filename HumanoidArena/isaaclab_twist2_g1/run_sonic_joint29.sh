#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/script/common/runtime_paths.sh"
cd "${SCRIPT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-${ISAACLAB_PYTHON}}"

VISION_RANDOMIZATION="${VISION_RANDOMIZATION:-0}"
if [ "${VISION_RANDOMIZATION}" = "1" ]; then
  DEFAULT_ENV_CONFIG_YAML="tasks/common_test_config/vision/vision_navi_sonic_test.yaml"
else
  DEFAULT_ENV_CONFIG_YAML="${ISAACLAB_ROOT}/tasks/common_env_config/opendoor_sonic.yaml"
fi
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-${DEFAULT_ENV_CONFIG_YAML}}"
# ENV_CONFIG_YAML="${ISAACLAB_ROOT}/tasks/common_env_config/doubledesk_sonic.yaml"
# ENV_CONFIG_YAML="${ISAACLAB_ROOT}/tasks/common_env_config/livingroom_sitsofa_sonic.yaml"
# ENV_CONFIG_YAML="tasks/common_env_config/pickplace_box_sonic.yaml"
# ENV_CONFIG_YAML="${ISAACLAB_ROOT}/tasks/common_env_config/football_single_sonic.yaml"
# ENV_CONFIG_YAML="${ISAACLAB_ROOT}/tasks/common_env_config/boxing_bag_sonic.yaml"

RUN_DEVICE="cpu"
ROBOT_TYPE="g129"
ROBOT_COLLIDER_MODE="box"
SEED="42"
ENABLE_CAMERAS=1
ENABLE_DEX3_DDS=1
HEADLESS=0
SONIC_REDIS_HOST="localhost"
SONIC_REDIS_PORT="6379"
SONIC_ENCODER_PATH="${SONIC_ENCODER_PATH:-${SONIC_POLICY_ROOT}/model_encoder.onnx}"
SONIC_DECODER_PATH="${SONIC_DECODER_PATH:-${SONIC_POLICY_ROOT}/model_decoder.onnx}"
IMAGE_TRANSPORT="xrobot"
# IMAGE_XROBOT_HOST="10.42.0.35"
IMAGE_XROBOT_HOST="192.168.100.87"
IMAGE_XROBOT_PORT="12345"
IMAGE_XROBOT_BITRATE="2097152"
IMAGE_FPS="30"
IMAGE_XROBOT_FFMPEG="/usr/bin/ffmpeg"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HOI_double_desk/sonic_v2/tw"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HOI_football_v2/sonic_v3/zk"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_vision_target/sonic/yb"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_open_door/sonic/zk"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_open_door/sonic/zz"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_boxing/sonic/zz"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_sitingsofa/sonic/yb"
RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_open_door/sonic/zz"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_sitingsofa/sonic/zk"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HSI_pp_box/sonic/yb"
# RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HOI_double_desk/sonic/tw"
LOG_DIR="${SCRIPT_DIR}/logs"

export PROJECT_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

load_task_name_from_yaml() {
  "${PYTHON_BIN}" - "${SCRIPT_DIR}/tasks/common_env_config/loader.py" "${1}" <<'PY'
import importlib.util
import pathlib
import sys

loader_path = pathlib.Path(sys.argv[1])
config_path = sys.argv[2]
spec = importlib.util.spec_from_file_location("common_env_config_loader", loader_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

task_name = module.get_env_config_task_name(config_path)
if not task_name:
    raise SystemExit(
        f"Error: env config YAML must define a top-level 'task_name': {config_path}"
    )
print(task_name)
PY
}

TASK_NAME="${TASK_NAME:-$(load_task_name_from_yaml "${ENV_CONFIG_YAML}")}"

if [ ! -f "${SONIC_ENCODER_PATH}" ]; then
  echo "Error: SONIC encoder not found: ${SONIC_ENCODER_PATH}"
  exit 1
fi
if [ ! -f "${SONIC_DECODER_PATH}" ]; then
  echo "Error: SONIC decoder not found: ${SONIC_DECODER_PATH}"
  exit 1
fi
redis-cli DEL \
  action_body_unitree_g1_with_hands \
  action_hand_left_unitree_g1_with_hands \
  action_hand_right_unitree_g1_with_hands \
  action_neck_unitree_g1_with_hands \
  human_smplx_data_unitree_g1_with_hands \
  human_info_unitree_g1_with_hands \
  gmr_full_qpos_unitree_g1_with_hands \
  gmr_joint_pos_unitree_g1_with_hands \
  gmr_joint_vel_unitree_g1_with_hands \
  gmr_body_pos_unitree_g1_with_hands \
  gmr_body_quat_w_unitree_g1_with_hands \
  gmr_frame_index_unitree_g1_with_hands \
  recording_control_unitree_g1_with_hands \
  isaac_reset_trigger \
  isaac_reset_complete_unitree_g1_with_hands \
  isaac_input_ready_sonic_joint29_unitree_g1_with_hands \
  controller_data \
  t_action >/dev/null 2>&1 || true


# export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2_thumd.usd"
export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
# export ROBOT_USD_OVERRIDE="$echo "[robot_usd] mode=${ROBOT_COLLIDER_MODE}"
echo "[robot_usd] path=${ROBOT_USD_OVERRIDE}"
echo "[sonic_joint29] task=${TASK_NAME}"
echo "[sonic_joint29] env_config=${ENV_CONFIG_YAML}"
echo "[sonic_joint29] vision_randomization=${VISION_RANDOMIZATION}"
echo "[sonic_joint29] seed=${SEED}"
echo "[sonic_joint29] encoder=${SONIC_ENCODER_PATH}"
echo "[sonic_joint29] decoder=${SONIC_DECODER_PATH}"
echo "[recording] save_dir=${RECORDING_SAVE_DIR}"

mkdir -p "${RECORDING_SAVE_DIR}"
mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SIM_LOG="${LOG_DIR}/sim_main_joint29_${RUN_TS}.log"
echo "[log] file=${SIM_LOG}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sim_main.py"
  --device "${RUN_DEVICE}"
  --env_config_yaml "${ENV_CONFIG_YAML}"
  --task "${TASK_NAME}"
  --robot_type "${ROBOT_TYPE}"
  --input_source pico_twist2
  --gmt_backend sonic_joint29
  --sonic_pose_source redis
  --sonic_redis_host "${SONIC_REDIS_HOST}"
  --sonic_redis_port "${SONIC_REDIS_PORT}"
  --sonic_encoder_path "${SONIC_ENCODER_PATH}"
  --sonic_decoder_path "${SONIC_DECODER_PATH}"
  --image_transport "${IMAGE_TRANSPORT}"
  --image_xrobot_host "${IMAGE_XROBOT_HOST}"
  --image_xrobot_port "${IMAGE_XROBOT_PORT}"
  --image_xrobot_bitrate "${IMAGE_XROBOT_BITRATE}"
  --image_fps "${IMAGE_FPS}"
  --image_xrobot_ffmpeg "${IMAGE_XROBOT_FFMPEG}"
  --recording_save_dir "${RECORDING_SAVE_DIR}"
  --seed "${SEED}"
)

if [ "${ENABLE_CAMERAS}" = "1" ]; then
  cmd+=(--enable_cameras)
fi
if [ "${ENABLE_DEX3_DDS}" = "1" ]; then
  cmd+=(--enable_dex3_dds)
fi
if [ "${HEADLESS}" = "1" ]; then
  cmd+=(--headless)
fi

if [ "$#" -gt 0 ]; then
  cmd+=("$@")
fi

{
  echo "[$(date '+%F %T')] Starting SONIC joint29 run"
  echo "[$(date '+%F %T')] task=${TASK_NAME}"
  echo "[$(date '+%F %T')] env_config=${ENV_CONFIG_YAML}"
  echo "[$(date '+%F %T')] recording_save_dir=${RECORDING_SAVE_DIR}"
  echo "[$(date '+%F %T')] log_file=${SIM_LOG}"
} | tee -a "${SIM_LOG}"

exec "${cmd[@]}" 2>&1 | tee -a "${SIM_LOG}"

# 可切换：
#一般沙袋: tasks/common_env_config/boxing_bag_sonic.yaml
#吊挂沙袋: tasks/common_env_config/boxing_bag_hanging_sonic.yaml
#足球: tasks/common_env_config/football_sonic.yaml
#单足球: tasks/common_env_config/football_single_sonic.yaml
#双桌面拾放: tasks/common_env_config/doubledesk_sonic.yaml
#Push-T: tasks/common_env_config/push_t_sonic.yaml
#三级台阶平台：tasks/common_env_config/three_step_platform_sonic.yaml
#开门：tasks/common_env_config/opendoor_sonic.yaml
#小推车：tasks/common_env_config/pickplace_small_trolley_sonic.yaml
# 箱子：tasks/common_env_config/pickplace_box_sonic.yaml
