# SIMPLE MuJoCo replay + Arena export

`replay_capture.py` replays one SIMPLE v2.1 parquet episode through the
decoupled WBC controller, records measured MuJoCo state and rendered head
images, and writes the HumanoidArena ref-pose schema (`state=64`, `action=40`)
plus the raw source fields.  By default it uses `mujoco_isaac`: MuJoCo runs the
fast physics/WBC loop while Isaac Sim renders the matching HSSD room.  Use
`--sim-mode mujoco` only for a plain-MuJoCo diagnostic.

The default episode is `G1WholebodyBendHandoverTeleop-v0/episode_000000` and
expects SIMPLE assets under `/data/local-data/data/Humanoid/psi-data`.

```bash
cd /data/local-data/data/code/yunhengwang/kimodo-polocy/controlnet_v1.2/SIMPLE
export SIMPLE_DATA_DIR=/data/local-data/data/Humanoid/psi-data
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
# Isaac Sim is installed in the shared IsaacLab 5.1 Python environment.  The
# extra data-disk directory contains the Python 3.11 wheels that are not part
# of that environment (pyarrow/transforms3d/yourdfpy).
export ISAAC_PYTHON=/data/local-data/data/Humanoid/psi-data/isaac-python
export SIMPLE_ISAAC_EXTRA_PYTHON="$ISAAC_PYTHON"
export PYTHONPATH="$PWD/src:$PWD/third_party:$PWD/third_party/curobo/src:$PWD/third_party/unitree_sdk2_python:$PWD/.."
export SIMPLE_ISAAC_GPU=6  # choose an idle physical H20; 4/5 are busy on this host
# If the driver is healthy but CUDA/Vulkan ordinals disagree, use the UUID
# instead of a numeric CUDA mask (the exporter still checks GPU Foundation).
# export CUDA_VISIBLE_DEVICES=GPU-23560e59-f305-ec1f-ee9c-bc764b333751
/home/CONNECT/yfang870/miniconda3/envs/env_isaaclab/bin/python ../data/playback/replay_capture.py --episode 0
```

Output is written to
`data/playback/output/G1WholebodyBendHandoverTeleop-v0`.  The script validates
canonical joint order, finite 64D/40D fields, measured-to-target tracking,
video/frame alignment, and both observed and target 417D Kimodo motion
representations.

If Isaac Sim reports `GPU Foundation is not initialized` or
`device_count=0`, no Isaac output is written. Kit can reach `app ready` while
its RTX camera returns all-black frames; the exporter refuses those frames.
Fix the host's Vulkan/CUDA mapping (or restart the node) before rerunning
`mujoco_isaac`. Use `--sim-mode mujoco` to validate the state/action export
independently.
