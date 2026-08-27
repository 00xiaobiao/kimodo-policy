## Replay one episode

```bash
cd /ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2

PROJ=/ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2
SIMPLE=$PROJ/SIMPLE
SP=$SIMPLE/.venv/lib/python3.10/site-packages
ISO=/ai/Yichi/kimodo-policy/simpledata/pinocchio-iso/site-packages

env -u CUDA_VISIBLE_DEVICES \
  OMNI_KIT_ACCEPT_EULA=YES \
  SIMPLE_DATA_DIR=/ai/Yichi/kimodo-policy/simpledata \
  SIMPLE_ISAAC_GPU=0 \
  SIMPLE_ISAAC_NO_CUDA_MASK=1 \
  SIMPLE_ISAAC_ALLOW_ZERO_GPU_COUNT=1 \
  SIMPLE_ISAAC_ACTIVE_GPU=0 \
  SIMPLE_ISAAC_PHYSICS_GPU=0 \
  SIMPLE_ISAAC_MAX_GPU_COUNT=1 \
  SIMPLE_ISAAC_EXTRA_PYTHON=$ISO \
  SIMPLE_ISAAC_PORTABLE_ROOT=/ai/Yichi/kimodo-policy/simpledata/isaac-cache/portable \
  TORCH_CUDA_ARCH_LIST=8.6+PTX \
  MUJOCO_GL=egl \
  HF_ENDPOINT=https://hf-mirror.com \
  TORCH_EXTENSIONS_DIR=/ai/Yichi/kimodo-policy/simpledata/torch-extensions \
  WARP_CACHE_PATH=/ai/Yichi/kimodo-policy/simpledata/isaac-cache/warp \
  XDG_CACHE_HOME=/ai/Yichi/kimodo-policy/simpledata/isaac-cache/xdg \
  TMPDIR=/ai/Yichi/kimodo-policy/simpledata/isaac-cache/tmp \
  PATH=$SIMPLE/.venv/bin:$PATH \
  PYTHONPATH=$SP:$PROJ/src:$SIMPLE/src:$SIMPLE/third_party:$SIMPLE/third_party/curobo/src:$SIMPLE/third_party/unitree_sdk2_python:$ISO \
  /ai/Yichi/taowen/isaac-sim/python.sh \
  data/playback/replay_capture.py \
  --sim-mode mujoco_isaac \
  --env-id simple/G1WholebodyBendPickTeleop-v0 \
  --source /ai/Yichi/kimodo-policy/simpledata/simple/G1WholebodyBendPickTeleop-v0 \
  --episode 0 \
  --output $PROJ/data/playback/output/test_pick_ep0_isaac
```

## Replay every episode in one task

This launches one worker on each GPU listed in `GPUS`. Episodes are distributed
across those workers, written to separate directories, and skipped when they
already have a validation report. Re-run the same command to resume an
interrupted batch. Update `GPUS` when server memory availability changes.

```bash
cd /ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2

PROJ=/ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2
SIMPLE=$PROJ/SIMPLE
SP=$SIMPLE/.venv/lib/python3.10/site-packages
ISO=/ai/Yichi/kimodo-policy/simpledata/pinocchio-iso/site-packages

TASK=G1WholebodyBendPickTeleop-v0
ENV_ID=simple/$TASK
SOURCE=/ai/Yichi/kimodo-policy/simpledata/simple/$TASK
OUTPUT_ROOT=$PROJ/data/playback/output/$TASK
GPUS=(0 2 5 7)
NUM_WORKERS=${#GPUS[@]}
CACHE_BASE=/ai/Yichi/kimodo-policy/simpledata/isaac-cache/playback-workers
PHYSICS_DT=0.005
if [[ "$TASK" == "G1WholebodyBendPickMP-v0" ]]; then
  PHYSICS_DT=0.002
fi

mapfile -t EPISODES < <(
  python3 -c '
import json
import sys

with open(sys.argv[1]) as episode_file:
    for line in episode_file:
        if line.strip():
            print(json.loads(line)["episode_index"])
' "$SOURCE/meta/episodes.jsonl"
)

mkdir -p "$OUTPUT_ROOT" "$CACHE_BASE"
echo "Found ${#EPISODES[@]} episodes for $TASK; launching workers on GPUs: ${GPUS[*]}"

run_worker() {
  local GPU=$1
  local SLOT=$2
  local WORKER_CACHE=$CACHE_BASE/gpu_$GPU
  local EPISODE OUTPUT ATTEMPT RUN_RC VALID_RC WORKER_STATUS=0

  mkdir -p \
    "$WORKER_CACHE/portable" \
    "$WORKER_CACHE/warp" \
    "$WORKER_CACHE/xdg" \
    "$WORKER_CACHE/tmp"

  echo "[gpu $GPU] worker started"

  for EPISODE in "${EPISODES[@]}"; do
    if (( EPISODE % NUM_WORKERS != SLOT )); then
      continue
    fi

    OUTPUT=$(printf "%s/episode_%06d" "$OUTPUT_ROOT" "$EPISODE")

    if [[ -s "$OUTPUT/validation.json" ]]; then
      echo "[gpu $GPU] skip episode $EPISODE: already completed"
      continue
    fi

    echo "[gpu $GPU] replay episode $EPISODE -> $OUTPUT"

    VALID_RC=1
    for ATTEMPT in 1 2 3; do
      RUN_RC=0
      echo "[gpu $GPU] episode $EPISODE attempt $ATTEMPT/3"
      env \
        -u CUDA_VISIBLE_DEVICES \
        -u ALL_PROXY -u all_proxy \
        HTTP_PROXY=http://127.0.0.1:17890 \
        http_proxy=http://127.0.0.1:17890 \
        HTTPS_PROXY=http://127.0.0.1:17890 \
        https_proxy=http://127.0.0.1:17890 \
        OMNI_KIT_ACCEPT_EULA=YES \
        SIMPLE_DATA_DIR=/ai/Yichi/kimodo-policy/simpledata \
        SIMPLE_ISAAC_GPU=$GPU \
        SIMPLE_ISAAC_NO_CUDA_MASK=1 \
        SIMPLE_ISAAC_ALLOW_ZERO_GPU_COUNT=1 \
        SIMPLE_ISAAC_ACTIVE_GPU=$GPU \
        SIMPLE_ISAAC_PHYSICS_GPU=$GPU \
        SIMPLE_ISAAC_MAX_GPU_COUNT=1 \
        SIMPLE_ISAAC_EXTRA_PYTHON=$ISO \
        SIMPLE_ISAAC_PORTABLE_ROOT=$WORKER_CACHE/portable \
        TORCH_CUDA_ARCH_LIST=8.6+PTX \
        MUJOCO_GL=egl \
        HF_ENDPOINT=https://hf-mirror.com \
        TORCH_EXTENSIONS_DIR=/ai/Yichi/kimodo-policy/simpledata/torch-extensions \
        WARP_CACHE_PATH=$WORKER_CACHE/warp \
        XDG_CACHE_HOME=$WORKER_CACHE/xdg \
        TMPDIR=$WORKER_CACHE/tmp \
        PATH=$SIMPLE/.venv/bin:$PATH \
        PYTHONPATH=$SP:$PROJ/src:$SIMPLE/src:$SIMPLE/third_party:$SIMPLE/third_party/curobo/src:$SIMPLE/third_party/unitree_sdk2_python:$ISO \
        /ai/Yichi/taowen/isaac-sim/python.sh \
        data/playback/replay_capture.py \
        --sim-mode mujoco_isaac \
        --env-id "$ENV_ID" \
        --source "$SOURCE" \
        --episode "$EPISODE" \
        --physics-dt "$PHYSICS_DT" \
        --output "$OUTPUT" || RUN_RC=$?

      VALID_RC=0
      python3 -c 'import json, sys; json.load(open(sys.argv[1]))' \
        "$OUTPUT/validation.json" >/dev/null 2>&1 || VALID_RC=$?
      if (( VALID_RC == 0 )); then
        break
      fi

      echo "[gpu $GPU] episode $EPISODE attempt $ATTEMPT failed: rc=$RUN_RC or invalid validation.json" >&2
      if (( ATTEMPT < 3 )); then
        sleep 10
      fi
    done

    if (( VALID_RC != 0 )); then
      echo "[gpu $GPU] failed episode $EPISODE after 3 attempts" >&2
      WORKER_STATUS=1
      continue
    fi
  done

  echo "[gpu $GPU] worker completed with status $WORKER_STATUS"
  return "$WORKER_STATUS"
}

declare -a WORKER_PIDS=()

for SLOT in "${!GPUS[@]}"; do
  GPU=${GPUS[$SLOT]}
  echo "[launch] GPU $GPU -> worker_gpu${GPU}.log"
  run_worker "$GPU" "$SLOT" >> "$OUTPUT_ROOT/worker_gpu${GPU}.log" 2>&1 &
  WORKER_PIDS[$SLOT]=$!
done

STATUS=0
for SLOT in "${!GPUS[@]}"; do
  GPU=${GPUS[$SLOT]}
  if wait "${WORKER_PIDS[$SLOT]}"; then
    echo "[complete] GPU $GPU worker"
  else
    echo "[failed] GPU $GPU worker; inspect worker_gpu${GPU}.log" >&2
    STATUS=1
  fi
done

exit "$STATUS"
```
