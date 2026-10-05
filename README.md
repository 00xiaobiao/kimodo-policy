<h1 align="center">
  Kimodo-Policy: From Text-to-Motion Generators to<br/>
  Humanoid Vision-Language-Action Policies
</h1>

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="arXiv"></a>
  <a href="https://kimodo-policy.github.io/"><img src="https://img.shields.io/badge/Project_Page-Kimodo--Policy-blue?logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href=""><img src="https://img.shields.io/badge/HuggingFace-Checkpoints-yellow?logo=huggingface&logoColor=black" alt="Hugging Face checkpoints"></a>
  <a href=""><img src="https://img.shields.io/badge/HuggingFace-Datasets-yellow?logo=huggingface&logoColor=black" alt="Hugging Face datasets"></a>
</p>

<p align="center">
  <img src="asset/picture/teaser.png" alt="Kimodo-Policy overview" width="96%">
</p>

Kimodo-Policy is a vision-conditioned policy for whole-body manipulation with the G1 humanoid. This repository contains the policy model, training launchers, dataset adapters, and simulation evaluation tools for HumanoidArena and SIMPLE. Large model checkpoints and datasets are distributed separately.

The policy is built on a frozen Kimodo motion backbone. DINOv3 extracts image features, and a ControlNet injects visual conditioning into the action denoising network. Hand control is handled by a separate head. The standard configurations run at 30 FPS with a 100-frame history and a 50-frame action chunk; refer to the selected YAML file for exact settings.

- Code: [Yunheng-Wang/kimodo-policy](https://github.com/Yunheng-Wang/kimodo-policy)
- Pretrained and task checkpoints: [Hugging Face model repository](https://huggingface.co/YunhengWang/kimodo-policy/tree/main)
- Training launchers: [scripts/](scripts/)
- Evaluation guides: [HumanoidArena](evaluation/humanoidarena_eval.md) and [SIMPLE](evaluation/simple_eval.md)

## Contents

- [SETUP](#setup)
- [HumanoidArena Training & Evaluation](#humanoidarena-training--evaluation)
- [Simple Training & Evaluation](#simple-training--evaluation)
- [Real-World Deployment](#real-world-deployment)
- [Troubleshooting](#troubleshooting)

## SETUP

This project uses three separate runtime environments. Keep the Kimodo training environment, the HumanoidArena simulator environment, the HumanoidArena model-server environment, and the SIMPLE evaluation environment isolated from one another. The commands below assume that the repository has been cloned with its submodules:

~~~bash
git clone --recurse-submodules https://github.com/Yunheng-Wang/kimodo-policy.git
cd kimodo-policy
git lfs install
git submodule update --init --recursive
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
~~~

### 1. Kimodo Training and Model-Inference Environment

Use this environment for pretraining, task fine-tuning, standalone Kimodo inference, and the model-side dependencies used by the evaluation server. The local reference environment for this release is named kimodo and currently reports Python 3.10.20 and PyTorch 2.11.0+cu126. The environment name can be changed; launchers only require KIMODO_ENV to point to the active environment.

The core versions in the local reference snapshot are:

| Package | Version |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu126 |
| TorchVision | 0.26.0+cu126 |
| TorchAudio | 2.11.0+cu126 |
| Accelerate | 1.13.0 |
| Transformers | 5.1.0 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| OmegaConf | 2.3.0 |
| PEFT | 0.18.1 |
| Safetensors | 0.7.0 |

Requirements:

- Linux and an NVIDIA GPU with a driver compatible with the selected CUDA PyTorch wheel.
- Python 3.10. The training configurations use BF16 by default, so BF16-capable GPUs are recommended.
- The Kimodo, DINOv3, Llama/LLM2Vec, and task checkpoint files hosted in the [Hugging Face model repository](https://huggingface.co/YunhengWang/kimodo-policy/tree/main).

Create or reuse the local Kimodo environment:

~~~bash
conda create -n kimodo python=3.10 -y
conda activate kimodo
python -m pip install --upgrade pip

# Select the PyTorch wheel that matches the host NVIDIA driver and CUDA runtime.
# The local reference environment uses torch 2.11.0+cu126.
pip install torch torchvision torchaudio

pip install accelerate omegaconf wandb numpy av pyarrow scipy einops \
  pydantic safetensors transformers peft tqdm packaging huggingface_hub
~~~

Verify the environment from the repository root:

~~~bash
export KIMODO_ENV="$CONDA_PREFIX"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import model.kimodo_policy, motion.g1_reference, skeleton.definitions; print('Kimodo imports OK')"
accelerate --version
~~~

Training launchers use Accelerate and resolve the project root from the launcher location. Set KIMODO_ENV before launching any script under scripts/. Pretraining also requires the text-embedding dependencies and the base model directories under checkpoints/. Missing paths are reported before training starts.

### 2. HumanoidArena Evaluation Environment

The HumanoidArena setup follows the [open-source-prep guide](https://github.com/William-wAng618/HumanoidArena/tree/release/open-source-prep) and the local [HumanoidArena environment guide](HumanoidArena/docs/04_environment_setup.md). Evaluation uses two Conda environments because Isaac Sim/Isaac Lab and the Kimodo/LeRobot model server have different Python requirements.

| Environment | Target configuration | Role |
| --- | --- | --- |
| unitree_sim_env | Python 3.11, Isaac Sim 5.0.0, Isaac Lab release/2.2.0, PyTorch 2.7.0 with CUDA 12.8 | Isaac Sim/Isaac Lab simulator and evaluation workers |
| lerobot | Python 3.12, LeRobot with the pi extra, PyTorch and Kimodo dependencies | Kimodo HTTP inference server |

The target host is Ubuntu 22.04 or newer with an NVIDIA GPU, a CUDA 12.x-compatible driver, Git LFS, and network access to the Isaac Sim, GitHub, and Hugging Face packages. Accept the Isaac Sim license in every simulator process:

~~~bash
export OMNI_KIT_ACCEPT_EULA=YES
~~~

#### 2.1 Install unitree_sim_env

The repository includes a dry-run-first installer. Review its output before executing it:

~~~bash
cd "$PROJECT_ROOT/HumanoidArena"
CONDA_BASE="$(conda info --base)"
bash isaaclab_twist2_g1/tools/setup_humanoidarena_envs.sh --dry-run
CONDA_BASE="$CONDA_BASE" \
  bash isaaclab_twist2_g1/tools/setup_humanoidarena_envs.sh --execute
~~~

The installer targets the following stack:

~~~bash
conda create -n unitree_sim_env python=3.11 -y
conda activate unitree_sim_env
python -m pip install --upgrade pip
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.0.0" \
  --extra-index-url https://pypi.nvidia.com
~~~

Isaac Lab release/2.2.0 and the project dependencies are installed by the helper. The simulator-specific dependencies are listed in HumanoidArena/isaaclab_twist2_g1/requirements.txt:

~~~text
rerun-sdk==0.20.1
pyzmq==27.0.0
logging_mp==0.1.5
onnxruntime==1.22.1
onnx
pynput==1.8.1
redis
coverage
~~~

If you install manually, follow the complete [HumanoidArena setup guide](HumanoidArena/docs/04_environment_setup.md), including the Isaac Lab checkout and any required system packages. Validate the simulator environment before evaluation:

~~~bash
conda run -n unitree_sim_env python -c \
  "import sys, torch, isaacsim; print(sys.version); print(torch.__version__); print('isaacsim import OK')"
~~~

The release target is Python 3.11. A Python 3.10 simulator environment may fail while importing torch or isaacsim; do not reuse the Python 3.10 Kimodo environment as unitree_sim_env. Recreate the simulator environment if this check fails.

#### 2.2 Restore HumanoidArena assets and policy artifacts

The simulator requires the HumanoidArena asset package at these exact paths:

~~~text
HumanoidArena/isaaclab_twist2_g1/assets/objects
HumanoidArena/isaaclab_twist2_g1/assets/robots
~~~

Download the release asset archive from the link in [HumanoidArena Environment Setup](HumanoidArena/docs/04_environment_setup.md), extract it under HumanoidArena/isaaclab_twist2_g1/assets, and verify both directories exist. SONIC evaluation additionally requires the GEAR-SONIC release artifacts:

~~~bash
cd "$PROJECT_ROOT/HumanoidArena"
export SONIC_POLICY_ROOT="$PROJECT_ROOT/HumanoidArena/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"
mkdir -p "$SONIC_POLICY_ROOT"
hf download nvidia/GEAR-SONIC \
  model_encoder.onnx model_decoder.onnx observation_config.yaml \
  --local-dir "$SONIC_POLICY_ROOT"
~~~

The language embedding cache is loaded by the Kimodo server from data/cache/HumanoidArena by default. It must contain the task-specific .pt files required by the selected evaluation task:

~~~bash
test -d "$PROJECT_ROOT/data/cache/HumanoidArena"
find "$PROJECT_ROOT/data/cache/HumanoidArena" -name '*.pt' -print
~~~

TWIST2 evaluation also requires:

~~~text
HumanoidArena/TWIST2/assets/ckpts/twist2_1017_20k.onnx
~~~

#### 2.3 Install the lerobot model-server environment

Install LeRobot separately from the simulator:

~~~bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
python -m pip install --upgrade pip
cd "$PROJECT_ROOT/HumanoidArena/lerobot"
pip install -e ".[pi]"
cd "$PROJECT_ROOT"
~~~

The server environment must be able to import the Kimodo modules and its model dependencies:

~~~bash
PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/HumanoidArena/lerobot/src" \
  python -c "import torch, transformers, scipy, omegaconf, einops, safetensors, peft; import model.kimodo_policy, motion.g1_reference, skeleton.definitions; print('Kimodo server imports OK')"
~~~

Before running a wrapper, export the two environment paths and the cache location:

~~~bash
export HUMANOIDARENA_ROOT="$PROJECT_ROOT/HumanoidArena"
export CONDA_BASE="$(conda info --base)"
export KIMODO_SIM_ENV="$CONDA_BASE/envs/unitree_sim_env"
export KIMODO_SERVER_PYTHON="$CONDA_BASE/envs/lerobot/bin/python"
export SONIC_POLICY_ROOT="$HUMANOIDARENA_ROOT/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"
export TEXT_EMBEDDING_CACHE="$PROJECT_ROOT/data/cache/HumanoidArena"
~~~

Every Kimodo evaluation checkpoint must be a training checkpoint directory, not only a base model directory. It must contain at least:

~~~text
checkpoint/
├── config.json
├── training_state.pt
└── other model and RNG state files
~~~

The SONIC and TWIST2 wrappers check config.json and training_state.pt before starting. A TWIST2 ONNX checkpoint is a simulator asset and cannot be passed as a LeRobot pretrained model.

#### 2.4 Validate the evaluation environment

Run a dry run before starting a simulator worker. This checks the task YAML, checkpoint files, model-server Python, simulator Python, SONIC artifacts, and cache paths without launching Isaac Sim:

~~~bash
bash evaluation/humanoidarena_eval_signal_task_sonic.sh \
  --project "$PROJECT_ROOT" \
  --task doubledesk \
  --checkpoint /path/to/checkpoint_200000 \
  --gpus 5,6,7 \
  --dry-run
~~~

For TWIST2, use the corresponding wrapper and ensure the TWIST2 ONNX file is present:

~~~bash
bash evaluation/humanoidarena_eval_signal_task_twist2.sh \
  --project "$PROJECT_ROOT" \
  --task football \
  --checkpoint /path/to/checkpoint_200000 \
  --gpus 5,6,7 \
  --seeds 0,1,2 \
  --repeats 20 \
  --dry-run
~~~

### 3. SIMPLE Evaluation Environment

The SIMPLE setup follows the [upstream SIMPLE repository](https://github.com/physical-superintelligence-lab/SIMPLE) and the checked-in SIMPLE directory. SIMPLE targets Ubuntu 22.04, Isaac Sim 4.5, MuJoCo 3.3, CUDA 12.x, NVIDIA driver 535+, and Python 3.10. An RTX 3080 Ti/4090 or better and at least 100 GB of free disk space are recommended. Keep this environment separate from kimodo, unitree_sim_env, and lerobot.

#### 3.1 Install the uv environment

Install the host packages and uv, then synchronize the dependencies from the local SIMPLE project:

~~~bash
cd "$PROJECT_ROOT/SIMPLE"
sudo apt-get update
sudo apt-get install -y curl cmake python3-dev ffmpeg gstreamer1.0-libav git-lfs
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh

UV_HTTP_TIMEOUT=3000 GIT_LFS_SKIP_SMUDGE=1 \
  uv sync --all-groups --index-strategy unsafe-best-match
bash scripts/install_curobo.sh
source .venv/bin/activate
python -c "import simple; print(simple.__version__)"
~~~

The CuRobo build needs a local CUDA toolkit and a compatible GPU architecture. The first compilation can take several minutes. Download the minimal scene resources when they are not already present:

~~~bash
bash scripts/pre-minimal-download.sh
~~~

The evaluation wrapper uses SIMPLE/.venv/bin/python by default. If that environment does not contain the Kimodo model dependencies, expose the Python 3.10 site-packages from the Kimodo environment:

~~~bash
export KIMODO_MODEL_SITE_PACKAGES="$CONDA_BASE/envs/kimodo/lib/python3.10/site-packages"
~~~

#### 3.2 Prepare SIMPLE evaluation data

SIMPLE simulator resources and official Level data are provided through SIMPLE_DATA_DIR. The evaluation wrapper expects task-specific Level directories below the SIMPLE data root, for example:

~~~text
SIMPLE_DATA_DIR/
└── simple-eval/
    └── G1WholebodyXMoveBendPickTeleop-v0/
        ├── dr-level-0/
        ├── dr-level-1/
        └── dr-level-2/
~~~

Each Level directory must contain meta/episodes.jsonl. The Kimodo language embedding cache is read from data/cache/Simple by default:

~~~bash
export SIMPLE_DATA_DIR=/path/to/simple-eval-data
test -d "$SIMPLE_DATA_DIR"
test -d "$PROJECT_ROOT/data/cache/Simple"
find "$PROJECT_ROOT/data/cache/Simple" -name '*.pt' -print
~~~

The current release tree does not include SIMPLE/third_party/IsaacLab or its submodule metadata. The optional SIMPLE/scripts/install_isaaclab.sh workflow therefore requires a compatible Isaac Lab checkout to be prepared separately. The local SIMPLE pyproject intentionally does not install Isaac Sim; install a compatible Isaac Sim 4.5 runtime according to the upstream SIMPLE documentation before running the full mujoco_isaac evaluation.

#### 3.3 Verify the evaluation wrapper

Use a released Kimodo task checkpoint and run a dry run before launching simulation:

~~~bash
export CONDA_BASE="$(conda info --base)"
export KIMODO_MODEL_SITE_PACKAGES="$CONDA_BASE/envs/kimodo/lib/python3.10/site-packages"
bash evaluation/simple_eval_signal_task_sonic.sh \
  --task G1WholebodyXMoveBendPickTeleop-v0 \
  --checkpoint /path/to/checkpoint_200000 \
  --simple-data-dir "$SIMPLE_DATA_DIR" \
  --gpus 0 \
  --levels 0 \
  --seeds 0 \
  --episodes 1 \
  --dry-run
~~~

For the official protocol, use levels 0, 1, and 2 with the task-matched checkpoint and evaluation data. Do not mix the Kimodo training environment with the SIMPLE simulator environment; only the model dependency path should be shared through KIMODO_MODEL_SITE_PACKAGES when needed.

## HumanoidArena Training & Evaluation

This section gives the complete Arena workflow after the environments in [SETUP](#setup) are installed. Training runs in the Kimodo Conda environment. Evaluation uses the HumanoidArena simulator in unitree_sim_env and the Kimodo model server in lerobot.

### Environment Setup

Complete [SETUP, Part 2](#2-humanoidarena-evaluation-environment) before training or evaluation. For training, activate the Kimodo environment and point the dataset variable to the local HumanoidArena dataset:

~~~bash
conda activate kimodo
export KIMODO_ENV="$CONDA_PREFIX"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export HUMANOID_ARENA_ROOT="$PROJECT_ROOT/datasets/HumanoidArena_dataset_v3_1"
test -d "$HUMANOID_ARENA_ROOT" || {
  echo "Missing HumanoidArena dataset: $HUMANOID_ARENA_ROOT" >&2
  exit 2
}
~~~

For evaluation, keep the simulator and model server in separate environments:

~~~bash
export CONDA_BASE="$(conda info --base)"
export HUMANOIDARENA_ROOT="$PROJECT_ROOT/HumanoidArena"
export KIMODO_SIM_ENV="$CONDA_BASE/envs/unitree_sim_env"
export KIMODO_SERVER_PYTHON="$CONDA_BASE/envs/lerobot/bin/python"
export SONIC_POLICY_ROOT="$HUMANOIDARENA_ROOT/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"
export TEXT_EMBEDDING_CACHE="$PROJECT_ROOT/data/cache/HumanoidArena"
export OMNI_KIT_ACCEPT_EULA=YES
~~~

Before a full run, verify that the simulator Python, model-server Python, SONIC artifacts, text cache, and Isaac Lab task YAML are available. The evaluation wrappers provide dry-run validation without launching Isaac Sim.

### Data Preparation

Download the official [HumanoidArena_dataset_v3_1 dataset](https://huggingface.co/datasets/WilliamWang16/HumanoidArena_dataset_v3_1/tree/main) before training. From the repository root, download it into the default directory used by the Arena YAML files:

~~~bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
python -m pip install --upgrade huggingface_hub
hf download WilliamWang16/HumanoidArena_dataset_v3_1 \
  --repo-type dataset \
  --local-dir "$PROJECT_ROOT/datasets/HumanoidArena_dataset_v3_1"
~~~

If the dataset is gated or requires authentication, run hf auth login first.

Audit the dataset before training:

~~~bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
python utils/check_datasets.py \
  --config scripts/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.yaml \
  --workers 1 \
  --limit 5 \
  --output "$PROJECT_ROOT/log/humanoidarena_dataset_audit.json"
~~~

The checker validates episode metadata, task selection, and action continuity. It does not replace a small GPU run that verifies video decoding and memory usage.

### Kimodo-Policy Checkpoints

This release provides the trained Kimodo-Policy weights used for the HumanoidArena experiments. The first table lists policies trained independently on each Arena task, either from the large-scale Kimodo-Policy pretraining stage or directly from scratch. The second table lists policies trained jointly on all seven Arena tasks, again with either scratch or 419h-pretrained initialization. The `Pretrain` column identifies the initialization, and the reported results are taken from Tables 1 and 2 of the paper.

| Task | Pretrain | Kimodo-Policy checkpoint | Results&nbsp;(SR) |
| --- | --- | --- | --- |
| doubledesk | ✗ | [humanoidarena_single_gbs64_20w_HOI_double_desk](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_double_desk_sonic) | 40.0 ± 0.0% |
| doubledesk | ✓ | [ft_419h_humanoidarena_single_gbs64_20w_HOI_double_desk](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_double_desk_sonic) | 28.3 ± 2.4% |
| football | ✗ | [humanoidarena_single_gbs64_20w_HOI_football](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_football_sonic) | 28.3 ± 6.2% |
| football | ✓ | [ft_419h_humanoidarena_single_gbs64_20w_HOI_football](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_football_sonic) | 26.7 ± 4.7% |
| pp_box | ✗ | [humanoidarena_single_gbs64_20w_HOI_pp_box](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_pp_box_sonic) | 80.0 ± 4.1% |
| pp_box | ✓ | [ft_419h_humanoidarena_single_gbs64_20w_HOI_pp_box](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_pp_box_sonic) | 78.3 ± 8.5% |
| boxing | ✗ | [humanoidarena_single_gbs64_20w_HSI_boxing](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_boxing_sonic) | 73.3 ± 13.1% |
| boxing | ✓ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_boxing](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_boxing_sonic) | 70.0 ± 8.2% |
| open_door | ✗ | [humanoidarena_single_gbs64_20w_HSI_open_door](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_open_door_sonic) | 90.0 ± 4.1% |
| open_door | ✓ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_open_door](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_open_door_sonic) | 93.3 ± 2.4% |
| sit_sofa | ✗ | [humanoidarena_single_gbs64_20w_HSI_sit_sofa](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_sit_sofa_sonic) | 100.0 ± 0.0% |
| sit_sofa | ✓ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_sit_sofa](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_sit_sofa_sonic) | 96.7 ± 2.4% |
| vision_navi | ✗ | [humanoidarena_single_gbs64_20w_HSI_vision_navi](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_vision_navi_sonic) | 75.0 ± 10.8% |
| vision_navi | ✓ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_vision_navi](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_vision_navi_sonic) | 76.7 ± 18.9% |

#### HumanoidArena Multi-Task Checkpoints

Each row below is one policy jointly trained on the seven HumanoidArena tasks. The Results column reports the Overall SR from Table 2.

| Task | Pretrain | Kimodo-Policy checkpoint | HOI Avg. | HSI Avg. | Overall Results |
| --- | --- | --- | --- | --- | --- |
| multi-task | ✗ | [humanoidarena_sonicx7_gbs128_50w](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) | 51.7 ± 20.7% | 79.6 ± 16.5% | 67.6% |
| multi-task | ✓ | [ft_419h_humanoidarena_sonicx7_gbs128_50w](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/ft_419h_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) | 56.1 ± 24.2% | 84.2 ± 14.1% | 72.1% |

### Training on Arena

All Arena launchers use BF16 and Accelerate. KIMODO_GPUS controls the visible GPUs, and the global batch size changes when the number of GPUs changes. Set a different KIMODO_MASTER_PORT for simultaneous jobs on one host.

#### Multi-task SONIC training from scratch

This configuration trains one policy on the seven-task Sonic RefPose mixture for 500,000 steps and saves to log/HumanoidArena_Multi_Task/ by default:

~~~bash
conda activate kimodo
export KIMODO_ENV="$CONDA_PREFIX"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export HUMANOID_ARENA_ROOT="$PROJECT_ROOT/datasets/HumanoidArena_dataset_v3_1"

KIMODO_GPUS=0,1,2,3 \
  bash scripts/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.sh
~~~

Use the SEED, large_hand4, or large_hand8 launcher in the same directory for the corresponding model variant.

#### Multi-task fine-tuning from a Kimodo checkpoint

Pass a complete pretraining checkpoint as the only positional argument:

~~~bash
INIT_CKPT=/path/to/checkpoint_1000000
KIMODO_GPUS=0,1,2,3 \
  bash scripts/HumanoidArena_Multi_Task/ft_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.sh \
  "$INIT_CKPT"
~~~

The 105h and 419h released fine-tuning families use the same launcher interface; select the matching YAML through KIMODO_CONFIG when reproducing a particular published run.

#### Single-task training

Train one task from scratch:

~~~bash
conda activate kimodo
export KIMODO_ENV="$CONDA_PREFIX"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export HUMANOID_ARENA_ROOT="$PROJECT_ROOT/datasets/HumanoidArena_dataset_v3_1"

KIMODO_GPUS=0,1,2,3 \
  bash scripts/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.sh \
  doubledesk sonic
~~~

Fine-tune the same task from a pretrained checkpoint:

~~~bash
KIMODO_GPUS=0,1,2,3 \
  bash scripts/HumanoidArena_Single_Task/ft_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.sh \
  doubledesk sonic /path/to/checkpoint_1000000
~~~

Supported task aliases include doubledesk, football, pp_box, boxing, open_door, sit_sofa, and vision_navi. The script maps these aliases to the task names used by the dataset adapter. Outputs are saved under log/HumanoidArena_Single_Task/ with the task and backend included in the run name.

### Evaluation

Evaluation starts two processes: an Isaac Sim/Isaac Lab worker from unitree_sim_env and a Kimodo inference server from lerobot. Set the environment variables from the Environment Setup subsection first. The checkpoint passed to the wrapper must be a Kimodo training checkpoint; SONIC and TWIST2 simulator assets are separate files.

#### SONIC evaluation

The following command evaluates one task with three GPUs, deterministic simulator startup, and no persistent simulator reuse:

~~~bash
bash evaluation/humanoidarena_eval_signal_task_sonic.sh \
  --project "$PROJECT_ROOT" \
  --task doubledesk \
  --checkpoint /path/to/checkpoint_200000 \
  --gpus 5,6,7 \
  --seeds 0,1,2 \
  --repeats 20 \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 15 \
  --rtc 0 \
  --persistent-sim 0 \
  --deterministic-eval 1 \
  --results-dir "$PROJECT_ROOT/eval_results/arena_doubledesk"
~~~

For continuous multi-task evaluation, use evaluation/humanoidarena_eval_multi_task_sonic.sh and pass the task list in the order required by the benchmark. See [HumanoidArena Evaluation](evaluation/humanoidarena_eval.md) for video recording, RTC, seeds, and result aggregation.

#### TWIST2 evaluation

TWIST2 uses the same Kimodo server environment but requires the TWIST2 ONNX asset and the task-specific Isaac Lab YAML:

~~~bash
bash evaluation/humanoidarena_eval_signal_task_twist2.sh \
  --project "$PROJECT_ROOT" \
  --task football \
  --checkpoint /path/to/checkpoint_200000 \
  --gpus 5,6,7 \
  --seeds 0,1,2 \
  --repeats 20 \
  --results-dir "$PROJECT_ROOT/eval_results/twist2_football"
~~~

Every evaluation run should have its own results directory. The wrappers write summary statistics, per-episode outputs, and optional videos there. A failed preflight usually indicates a missing simulator asset, cache file, environment variable, or incomplete checkpoint; the wrapper prints the missing path before launching the worker.

## Simple Training & Evaluation

Complete [SETUP, Part 3](#3-simple-evaluation-environment) before using SIMPLE. Policy fine-tuning runs in the Kimodo training environment, while evaluation runs in the SIMPLE simulator environment. A checkpoint passed to either workflow must contain both `config.json` and `training_state.pt`.

### Training

The continuous-hand launcher fine-tunes one policy on one SIMPLE task. `KIMODO_SIMPLE_ROOT` must contain the selected task directory:

~~~bash
conda activate kimodo
export KIMODO_ENV="$CONDA_PREFIX"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export KIMODO_SIMPLE_ROOT="$PROJECT_ROOT/datasets/Simple"
test -d "$KIMODO_SIMPLE_ROOT/G1WholebodyXMoveBendPickTeleop-v0" || {
  echo "Missing SIMPLE task directory" >&2
  exit 2
}

KIMODO_GPUS=0,1,2,3 \
  bash scripts/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand.sh \
  G1WholebodyXMoveBendPickTeleop-v0 /path/to/checkpoint_1000000
~~~

Use a task-specific checkpoint and matching task directory for other SIMPLE tasks. Outputs are written to `log/Simple_Single_Task/<run-name>_<TASK>/`.

### Evaluation

The evaluator uses `SIMPLE/.venv/bin/python` by default and runs the official Level 0, Level 1, and Level 2 splits. Set `SIMPLE_DATA_DIR` to the SIMPLE data root containing simulator resources and `simple-eval/<task>/level-{0,1,2}` or `dr-level-{0,1,2}` directories. Each Level directory must contain `meta/episodes.jsonl`.

~~~bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
CONDA_BASE="$(conda info --base)"
CHECKPOINT=/path/to/Simple_Single_Task/checkpoint_200000
export SIMPLE_DATA_DIR=/path/to/simple-eval-data
# Set this when SIMPLE/.venv lacks transformers or safetensors.
export KIMODO_MODEL_SITE_PACKAGES="$CONDA_BASE/envs/kimodo/lib/python3.10/site-packages"

bash evaluation/simple_eval_signal_task_sonic.sh \
  --task G1WholebodyXMoveBendPickTeleop-v0 \
  --checkpoint "$CHECKPOINT" \
  --gpus 0 \
  --levels 0,1,2 \
  --seeds 0 \
  --episodes 10 \
  --simple-data-dir "$SIMPLE_DATA_DIR" \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 50 \
  --rtc 0 \
  --max-navigation-speed 1.5 \
  --max-steps auto \
  --results-dir "$PROJECT_ROOT/eval_results/simple_xmove_bendpick"
~~~

The task name and evaluation data must match the checkpoint. Results, per-level summaries, and optional videos are written under `eval_results/`. See [SIMPLE Evaluation](evaluation/simple_eval.md) for the complete input layout and reporting details.

## Real-World Deployment

**Coming in a future release.** Hardware configuration, deployment dependencies, launch commands, and safety procedures are not included yet.

## Troubleshooting

**Kimodo or DINOv3 weights cannot be found**

Confirm that checkpoints/ is at the repository root and contains Kimodo-G1-RP-v1/config.yaml, Kimodo-G1-RP-v1/model.safetensors, and dinov3-vitl16-pretrain-lvd1689m/. The SEED configurations additionally require Kimodo-G1-SEED-v1/.

**The checkpoint is incomplete**

Inference and fine-tuning use a training output directory such as checkpoint_<step>, not a base-model directory containing only model.safetensors. Check that the checkpoint includes config.json and training_state.pt.

**A dataset path error is reported**

Set HUMANOID_ARENA_ROOT, UNIFOLM_ROOT, HUMANOID_EVERYDAY_ROOT, HIW500_ROOT, or KIMODO_SIMPLE_ROOT to real dataset paths. A path can exist and still fail if it contains no compatible episodes; inspect dataset_selection in the selected YAML and the matching data/*_loader.py.

**SIMPLE evaluation cannot find model dependencies**

Evaluation uses SIMPLE/.venv/bin/python by default. Confirm that the uv environment is installed. If transformers or safetensors are available only in the Kimodo Conda environment, set KIMODO_MODEL_SITE_PACKAGES.

**Resume interrupted training**

Use the same configuration and dataset roots, then run train.py --resume /path/to/checkpoint_STEP to restore the optimizer, scheduler, and global step. To initialize a new task from a pretrained model, use the checkpoint argument of the corresponding launcher; internally it is passed as --init-checkpoint, so the new run starts with a fresh optimizer state.
