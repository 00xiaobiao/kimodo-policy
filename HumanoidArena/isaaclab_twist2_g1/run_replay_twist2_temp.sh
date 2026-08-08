#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

# ------------------------------------------------------------------
# User config: edit here
# ------------------------------------------------------------------
PYTHON_BIN="python"
REPLAY_FILE="/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/HOI_double_desk/twist2/zz/Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody_1776343243956570.npz"
REPLAY_MODE="direct"   # inference | direct
REPLAY_LOOP=0             # 1 | 0
TASK_NAME=""              # 留空则从 replay 文件读取
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_env_config/doubledesk_twist2.yaml}"
RUN_DEVICE="cpu"
ROBOT_TYPE="g129"
ROBOT_COLLIDER_MODE="box" # box | fourpoints
ENABLE_CAMERAS=1
ENABLE_WRIST_CAMERAS=0
ENABLE_DEX3_DDS=1
HEADLESS=0
MODEL_PATH="/home/dreams/Users/taowen/HumanoidArena/TWIST2/assets/ckpts/twist2_1017_20k.onnx"
IMAGE_TRANSPORT="zmq"
IMAGE_ZMQ_PORT="5555"
LEFT_WRIST_CAMERA_PORT="5557"
RIGHT_WRIST_CAMERA_PORT="5558"
IMAGE_FPS="30"
STATS_INTERVAL="10.0"
STEP_HZ="50"
VIDEO_FPS="30"
SEED="42"

export PROJECT_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

if [ -z "${REPLAY_FILE}" ]; then
  echo "Error: REPLAY_FILE is empty"
  exit 1
fi
if [ ! -f "${REPLAY_FILE}" ]; then
  echo "Error: replay file not found: ${REPLAY_FILE}"
  exit 1
fi
if [ "${REPLAY_MODE}" != "inference" ] && [ "${REPLAY_MODE}" != "direct" ]; then
  echo "Error: replay mode must be inference or direct, got: ${REPLAY_MODE}"
  exit 1
fi

REPLAY_FILE="$(realpath "${REPLAY_FILE}")"
if [ -z "${TASK_NAME}" ]; then
  TASK_NAME="$("${PYTHON_BIN}" - "${REPLAY_FILE}" <<'PY'
import sys
import numpy as np

replay_file = sys.argv[1]
with np.load(replay_file, allow_pickle=True) as data:
    task = data.get("task")
    if task is None:
        raise SystemExit("Replay file missing 'task' metadata")
    if hasattr(task, "item"):
        task = task.item()
    print(task)
PY
)"
fi

if [ "${ROBOT_COLLIDER_MODE}" = "fourpoints" ]; then
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/temp/g1_29dof_with_dex3_rev_1_0_fourpoints.usd"
else
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
fi

echo "[robot_usd] mode=${ROBOT_COLLIDER_MODE}"
echo "[robot_usd] path=${ROBOT_USD_OVERRIDE}"
echo "[twist2 replay] file=${REPLAY_FILE}"
echo "[twist2 replay] mode=${REPLAY_MODE}"
echo "[twist2 replay] task=${TASK_NAME}"
echo "[twist2 replay] env_config=${ENV_CONFIG_YAML}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sim_main.py"
  --device "${RUN_DEVICE}"
  --task "${TASK_NAME}"
  --env_config_yaml "${ENV_CONFIG_YAML}"
  --robot_type "${ROBOT_TYPE}"
  --input_source replay
  --gmt_backend twist2
  --replay_file "${REPLAY_FILE}"
  --replay_mode "${REPLAY_MODE}"
  --model_path "${MODEL_PATH}"
  --image_transport "${IMAGE_TRANSPORT}"
  --image_fps "${IMAGE_FPS}"
  --image_zmq_port "${IMAGE_ZMQ_PORT}"
  --stats_interval "${STATS_INTERVAL}"
  --step_hz "${STEP_HZ}"
  --video_fps "${VIDEO_FPS}"
  --seed "${SEED}"
)

if [ "${ENABLE_CAMERAS}" = "1" ]; then
  cmd+=(--enable_cameras)
fi
if [ "${ENABLE_WRIST_CAMERAS}" = "1" ]; then
  cmd+=(--enable_wrist_cameras)
  cmd+=(--left_wrist_camera_port "${LEFT_WRIST_CAMERA_PORT}")
  cmd+=(--right_wrist_camera_port "${RIGHT_WRIST_CAMERA_PORT}")
fi
if [ "${ENABLE_DEX3_DDS}" = "1" ]; then
  cmd+=(--enable_dex3_dds)
fi
if [ "${HEADLESS}" = "1" ]; then
  cmd+=(--headless)
fi
if [ "${REPLAY_LOOP}" = "1" ]; then
  cmd+=(--replay_loop)
fi

exec "${cmd[@]}"

# 可切换：
#一般沙袋: tasks/common_env_config/boxing_bag_twist2.yaml
#吊挂沙袋: tasks/common_env_config/boxing_bag_hanging_twist2.yaml
#足球: tasks/common_env_config/football_twist2.yaml
#单足球: tasks/common_env_config/football_single_twist2.yaml
#双桌面拾放: tasks/common_env_config/doubledesk_twist2.yaml
#Push-T: tasks/common_env_config/push_t_twist2.yaml
#三级台阶平台：tasks/common_env_config/three_step_platform_twist2.yaml
#开门：tasks/common_env_config/opendoor_twist2.yaml
#小推车：tasks/common_env_config/pickplace_small_trolley_twist2.yaml