# Data Pipeline

This guide describes the recording data path used by HumanoidArena: live recordings, replay, rerecord, NPZ inspection, and conversion into LeRobot-compatible training data.

## 1. Data Flow

The maintained data flow is:

```text
teleoperation -> NPZ recording -> replay / rerecord -> multicam data -> LeRobot dataset
```

The repository keeps code and small examples in git. Large generated datasets should be published through the Hugging Face dataset release.

## 2. NPZ Recording Format

TWIST2 and SONIC recordings use `.npz` containers. The exact keys depend on the backend and recording profile.

Reference documents:

- [TWIST2 data format](../isaaclab_twist2_g1/docs/TWIST2_DATA_FORMAT.md)
- [SONIC data format](../isaaclab_twist2_g1/docs/SONIC_DATA_FORMAT.md)
- [LeRobot V3.1 format](../isaaclab_twist2_g1/docs/UNITREE_G1_GMT_REFPOSE_V3_1_DATA_PROTOCOL.md)

For a quick local inspection:

```bash
python - <<'PY'
import numpy as np
path = "/path/to/recording.npz"
data = np.load(path, allow_pickle=True)
for key in data.files:
    value = data[key]
    print(key, getattr(value, "shape", None), value.dtype if hasattr(value, "dtype") else type(value))
PY
```

## 3. Replay

Use replay to validate a recorded `.npz` without creating a new dataset.

Replay and video capture are best run on a workstation with a working Isaac Sim rendering interface. Headless execution is not recommended for release validation, especially when cameras or video recording are enabled.

TWIST2:

```bash
bash isaaclab_twist2_g1/run_replay_twist2.sh
```

SONIC:

```bash
bash isaaclab_twist2_g1/run_replay_sonic.sh
```

For direct SONIC replay, pass the target recording through `--replay_file`:

```bash
python isaaclab_twist2_g1/sim_main.py \
  --device cpu \
  --env_config_yaml isaaclab_twist2_g1/tasks/common_env_config/opendoor_sonic.yaml \
  --task Isaac-Move-Open-Door-G129-Dex3-Wholebody \
  --robot_type g129 \
  --input_source replay \
  --gmt_backend sonic \
  --replay_file /path/to/open_door_sonic_recording.npz \
  --replay_mode direct_replay \
  --enable_cameras \
  --enable_dex3_dds \
  --seed 42
```

Common replay modes:

```text
direct_replay
inference_replay
```

## 4. Rerecord

Use rerecord to regenerate observations, cameras, or normalized outputs from existing recordings.

Rerecord depends on simulator rendering and camera capture. Run it on a machine with a visual/rendering-capable Isaac Sim setup; headless containers may stall or skip camera outputs depending on the deployment.

Unified wrapper:

```bash
TWIST2_INPUT_ROOT=/path/to/twist2 \
SONIC_SOURCE_ROOT=/path/to/sonic \
TWIST2_PARALLEL_JOBS=2 \
SONIC_PARALLEL_JOBS=2 \
bash isaaclab_twist2_g1/run_rerecord.sh
```

SONIC rerecord:

```bash
python isaaclab_twist2_g1/tools/data_tools/rerecord_sonic_recordings_to_multicam.py \
  /path/to/sonic_source_root \
  --parallel-jobs 1
```

TWIST2 rerecord:

```bash
python isaaclab_twist2_g1/tools/data_tools/rerecord_twist2_recordings_to_multicam.py \
  /path/to/twist2_input_root \
  --parallel-jobs 1
```

Open-door SONIC rerecord with perspective camera:

```bash
python isaaclab_twist2_g1/tools/data_tools/rerecord_sonic_recordings_to_multicam.py \
  ${ISAACLAB_ROOT}/recording_data/perspective-use/ \
  --enable-perspective-camera \
  --disable-front-camera \
  --disable-wrist-cameras \
  --parallel-jobs 1 \
  --force
```

## 5. Convert NPZ to LeRobot

The conversion entrypoint depends on the exact dataset release layout. Use the maintained converter under `isaaclab_twist2_g1/tools/data_tools/` when preparing the public Hugging Face dataset.

Recommended checklist before conversion:

- Replay a sample of each task successfully.
- Verify camera views and timestamps after rerecord.
- Confirm train/validation split policy.
- Confirm task names and episode metadata match the Hugging Face dataset card.
- Keep raw and multicam release plans separate from the git commit.

After conversion, validate the dataset from the `lerobot` environment:

```bash
cd lerobot
PYTHONPATH=src python - <<'PY'
import lerobot
print("lerobot import ok")
PY
```

## 6. Release Notes for Data

The README release checklist tracks these data artifacts separately:

- LeRobot dataset: released.
- Model checkpoints: released.
- Raw data: planned.
- Multicam data: planned.

Large data artifacts should remain outside this git repository and be linked from the README.
