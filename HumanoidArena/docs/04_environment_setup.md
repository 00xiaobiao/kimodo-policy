# Environment Setup

This guide is the release-facing setup index for HumanoidArena.

## 1. System Requirements

Recommended platform:

- Ubuntu 22.04 or newer.
- NVIDIA GPU and driver compatible with Isaac Sim 5.0 and CUDA 12.x.
- Miniconda or Anaconda.
- Git LFS.
- Network access to GitHub and Hugging Face.
- HumanoidArena asset package restored under `isaaclab_twist2_g1/assets` with `objects/` and `robots/` present.

The current simulation stack targets Isaac Sim 5.0.0 and Isaac Lab `release/2.2.0`.

## 2. Conda Environments

HumanoidArena uses two environments because Isaac Lab and LeRobot have different Python dependency constraints.

| Environment | Python | Use |
| --- | --- | --- |
| `unitree_sim_env` | 3.11 | Isaac Sim, Isaac Lab, teleoperation, replay, rerecord, evaluation workers |
| `lerobot` | 3.12+ | LeRobot training and VLA policy serving |

## 3. Optional Dry Run Installer

The setup helper prints the installation commands without changing the machine when run in dry-run mode:

```bash
bash isaaclab_twist2_g1/tools/setup_humanoidarena_envs.sh --dry-run
```

Execute only after reviewing the generated commands:

```bash
CONDA_BASE=/path/to/miniconda3 \
bash isaaclab_twist2_g1/tools/setup_humanoidarena_envs.sh --execute
```

## 4. Install Simulation Environment

Create and activate the environment:

```bash
conda create -n unitree_sim_env python=3.11
conda activate unitree_sim_env
python -m pip install --upgrade pip
```

Install the PyTorch stack compatible with Isaac Lab release 2.2.0:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install Isaac Sim 5.0.0:

```bash
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
isaacsim
```

Install Isaac Lab:

```bash
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout release/2.2.0
sudo apt install cmake build-essential cyclonedds-dev
./isaaclab.sh --install
```

Install project dependencies:

```bash
cd /path/to/HumanoidArena
pip install -r isaaclab_twist2_g1/requirements.txt
```

## 5. Download Simulation Assets

Download the release asset package from Google Drive:

```text
https://drive.google.com/file/d/1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk/view?usp=sharing
```

The recommended release zip layout has `objects/` and `robots/` at the archive root. Restore it into the Isaac Lab asset directory:

```bash
cd /path/to/HumanoidArena/isaaclab_twist2_g1
mkdir -p assets
unzip /path/to/humanoidarena_assets_isaaclab_twist2_g1_<date>.zip -d assets
```

After extraction, verify the expected runtime paths:

```bash
test -d /path/to/HumanoidArena/isaaclab_twist2_g1/assets/objects
test -d /path/to/HumanoidArena/isaaclab_twist2_g1/assets/robots
```

If the downloaded archive already contains a top-level `assets/` directory, unzip it from `/path/to/HumanoidArena/isaaclab_twist2_g1` instead so the final paths still resolve to `isaaclab_twist2_g1/assets/objects` and `isaaclab_twist2_g1/assets/robots`.

## 6. SONIC Policy Artifacts

SONIC uses release artifacts from Hugging Face:

```bash
cd /path/to/HumanoidArena
mkdir -p GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
hf download nvidia/GEAR-SONIC \
  model_encoder.onnx model_decoder.onnx observation_config.yaml \
  --local-dir GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
```

Maintained scripts resolve this directory through `SONIC_POLICY_ROOT`, defaulting to:

```text
/path/to/HumanoidArena/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
```

## 7. Install LeRobot Environment

```bash
conda create -n lerobot python=3.12
conda activate lerobot
python -m pip install --upgrade pip
cd /path/to/HumanoidArena/lerobot
pip install -e ".[pi]"
```

For base ACT/Diffusion work, the base editable install may be enough:

```bash
pip install -e .
```

## 8. Smoke Tests

Simulation import:

```bash
OMNI_KIT_ACCEPT_EULA=YES python -c "import isaacsim; print('isaacsim import ok')"
```

LeRobot import:

```bash
cd /path/to/HumanoidArena/lerobot
PYTHONPATH=src python - <<'PY'
import torch
import torchvision
import lerobot
print("lerobot imports ok")
PY
```

TWIST2 checkpoint presence:

```bash
ls TWIST2/assets/ckpts/twist2_1017_20k.onnx \
   TWIST2/assets/ckpts/twist2_1017_25k.onnx
```

## 9. Next Steps

After setup, continue with:

- [Teleoperation and recording](01_teleoperation.md)
- [Data pipeline](02_data_pipeline.md)
- [Evaluation](03_evaluation.md)
