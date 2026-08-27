#!/usr/bin/env bash
set -Eeuo pipefail

# Replay every episode of one SIMPLE task in parallel workers.
# Usage:
#   bash data/playback/replay_task.sh TASK "0 2 5 7"
#
# The script starts a detached tmux session by default.  Pass --foreground
# as the first argument when running it from an existing tmux/session manager.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SIMPLE="$PROJ/SIMPLE"
SP="$SIMPLE/.venv/lib/python3.10/site-packages"
ISO="/ai/Yichi/kimodo-policy/simpledata/pinocchio-iso/site-packages"
ISAAC_PY="/ai/Yichi/taowen/isaac-sim/python.sh"

foreground=0
if [[ "${1:-}" == "--foreground" ]]; then
  foreground=1
  shift
fi

TASK="${1:-}"
GPU_SPEC="${2:-${GPUS:-0 2 5 7}}"
if [[ -z "$TASK" ]]; then
  echo "Usage: $0 TASK [GPUS]" >&2
  echo "Example: $0 G1WholebodyBendPickMP-v0 \"0 2 5 7\"" >&2
  exit 2
fi

# Accept either space-separated or comma-separated GPU lists.
GPU_SPEC="${GPU_SPEC//,/ }"
read -r -a GPU_LIST <<< "$GPU_SPEC"
if (( ${#GPU_LIST[@]} == 0 )); then
  echo "No GPUs were specified" >&2
  exit 2
fi

SOURCE="/ai/Yichi/kimodo-policy/simpledata/simple/$TASK"
OUTPUT_ROOT="$PROJ/data/playback/output/$TASK"
CACHE_BASE="${CACHE_BASE:-/ai/Yichi/kimodo-policy/simpledata/isaac-cache/playback-workers/$TASK}"
SESSION="${SESSION_NAME:-replay_${TASK//[^A-Za-z0-9_]/_}}"
MASTER_LOG="${MASTER_LOG:-/tmp/${SESSION}.log}"

if [[ ! -f "$SOURCE/meta/episodes.jsonl" ]]; then
  echo "Missing dataset metadata: $SOURCE/meta/episodes.jsonl" >&2
  exit 1
fi

# Do not accidentally launch a duplicate batch for the same task.
if pgrep -af "[r]eplay_capture.py.*--env-id simple/$TASK" >/dev/null 2>&1; then
  echo "A replay process for $TASK is already running; refusing to duplicate it." >&2
  exit 3
fi

if (( foreground == 0 )); then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION" >&2
    exit 3
  fi
  # Task names and GPU specs are constrained to dataset names and integers.
  tmux new-session -d -s "$SESSION" \
    "$0 --foreground $(printf '%q' "$TASK") $(printf '%q' "$GPU_SPEC")"
  echo "Started tmux session: $SESSION"
  echo "Master log: $MASTER_LOG"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

if [[ "$TASK" == *MP* ]]; then
  PHYSICS_DT="0.002"
else
  PHYSICS_DT="0.005"
fi

mapfile -t EPISODES < <(
  python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    for line in f:
        if line.strip():
            print(json.loads(line)["episode_index"])
' "$SOURCE/meta/episodes.jsonl"
)

mkdir -p "$OUTPUT_ROOT" "$CACHE_BASE"
{
  echo "[master] task=$TASK"
  echo "[master] episodes=${#EPISODES[@]} gpus=${GPU_LIST[*]} physics_dt=$PHYSICS_DT"
  echo "[master] output=$OUTPUT_ROOT"
} | tee "$MASTER_LOG"

run_worker() {
  local gpu="$1"
  local slot="$2"
  local worker_cache="$CACHE_BASE/gpu_$gpu"
  local episode output attempt run_rc valid_rc
  local worker_status=0

  mkdir -p "$worker_cache/portable" "$worker_cache/warp" "$worker_cache/xdg" "$worker_cache/tmp"
  echo "[gpu $gpu] worker started"

  for episode in "${EPISODES[@]}"; do
    if (( episode % ${#GPU_LIST[@]} != slot )); then
      continue
    fi

    output=$(printf "%s/episode_%06d" "$OUTPUT_ROOT" "$episode")
    if [[ -s "$output/validation.json" ]]; then
      echo "[gpu $gpu] skip episode $episode: already completed"
      continue
    fi

    echo "[gpu $gpu] replay episode $episode -> $output"
    valid_rc=1
    for attempt in 1 2 3; do
      echo "[gpu $gpu] episode $episode attempt $attempt/3"
      run_rc=0
      env \
        -u CONDA_PREFIX \
        -u CUDA_VISIBLE_DEVICES \
        -u ALL_PROXY \
        -u all_proxy \
        HTTP_PROXY=http://127.0.0.1:17890 \
        http_proxy=http://127.0.0.1:17890 \
        HTTPS_PROXY=http://127.0.0.1:17890 \
        https_proxy=http://127.0.0.1:17890 \
        OMNI_KIT_ACCEPT_EULA=YES \
        SIMPLE_DATA_DIR=/ai/Yichi/kimodo-policy/simpledata \
        SIMPLE_ISAAC_GPU="$gpu" \
        SIMPLE_ISAAC_NO_CUDA_MASK=1 \
        SIMPLE_ISAAC_ALLOW_ZERO_GPU_COUNT=1 \
        SIMPLE_ISAAC_ACTIVE_GPU="$gpu" \
        SIMPLE_ISAAC_PHYSICS_GPU="$gpu" \
        SIMPLE_ISAAC_MAX_GPU_COUNT=1 \
        SIMPLE_ISAAC_EXTRA_PYTHON="$ISO" \
        SIMPLE_ISAAC_PORTABLE_ROOT="$worker_cache/portable" \
        TORCH_CUDA_ARCH_LIST=8.6+PTX \
        MUJOCO_GL=egl \
        HF_ENDPOINT=https://hf-mirror.com \
        TORCH_EXTENSIONS_DIR=/ai/Yichi/kimodo-policy/simpledata/torch-extensions \
        WARP_CACHE_PATH="$worker_cache/warp" \
        XDG_CACHE_HOME="$worker_cache/xdg" \
        TMPDIR="$worker_cache/tmp" \
        PATH="$SIMPLE/.venv/bin:$PATH" \
        PYTHONPATH="$SP:$PROJ/src:$SIMPLE/src:$SIMPLE/third_party:$SIMPLE/third_party/curobo/src:$SIMPLE/third_party/unitree_sdk2_python:$ISO" \
        "$ISAAC_PY" \
        "$SCRIPT_DIR/replay_capture.py" \
        --sim-mode mujoco_isaac \
        --env-id "simple/$TASK" \
        --source "$SOURCE" \
        --episode "$episode" \
        --physics-dt "$PHYSICS_DT" \
        --output "$output" || run_rc=$?

      if [[ -s "$output/validation.json" ]] && python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$output/validation.json" >/dev/null 2>&1; then
        valid_rc=0
        break
      fi

      echo "[gpu $gpu] episode $episode attempt $attempt failed: rc=$run_rc or invalid validation.json" >&2
      if (( attempt < 3 )); then
        sleep 10
      fi
    done

    if (( valid_rc != 0 )); then
      echo "[gpu $gpu] failed episode $episode after 3 attempts" >&2
      worker_status=1
    fi
  done

  echo "[gpu $gpu] worker completed with status $worker_status"
  return "$worker_status"
}

worker_pids=()
for slot in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$slot]}"
  echo "[launch] GPU $gpu -> worker_gpu${gpu}.log"
  run_worker "$gpu" "$slot" >> "$OUTPUT_ROOT/worker_gpu${gpu}.log" 2>&1 &
  worker_pids[$slot]=$!
done

status=0
for slot in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$slot]}"
  if wait "${worker_pids[$slot]}"; then
    echo "[complete] GPU $gpu worker" | tee -a "$MASTER_LOG"
  else
    echo "[failed] GPU $gpu worker; inspect worker_gpu${gpu}.log" | tee -a "$MASTER_LOG" >&2
    status=1
  fi
done

echo "[master] finished task=$TASK status=$status" | tee -a "$MASTER_LOG"
exit "$status"


# cd /ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2
# bash data/playback/replay_task.sh G1WholebodyBendPickMP-v0 "0 2 5 7"