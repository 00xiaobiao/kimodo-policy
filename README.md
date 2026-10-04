# Kimodo-Policy

<p align="center">
  <img src="asset/picture/teaser.png" alt="Kimodo-Policy overview" width="96%">
</p>

Kimodo-Policy 是面向 G1 humanoid whole-body manipulation 的视觉条件策略项目。本仓库包含模型、训练入口、数据适配器，以及 HumanoidArena 和 SIMPLE 的仿真评估工具；大体积模型权重与数据集单独发布。

策略以冻结的 Kimodo motion backbone 为基础，使用 DINOv3 提取图像特征，并通过 ControlNet 将视觉条件注入动作去噪网络；手部控制由独立 head 处理。常用训练配置以 30 FPS 运行，使用 100 帧历史并预测 50 帧动作 chunk，具体设置以所选 YAML 为准。

- 代码仓库：[Yunheng-Wang/kimodo-policy](https://github.com/Yunheng-Wang/kimodo-policy)
- 预训练及任务 checkpoint：[Hugging Face 模型仓库](https://huggingface.co/YunhengWang/kimodo-policy/tree/main)
- 训练启动脚本：[scripts/](scripts/)
- 评估说明：[HumanoidArena](evaluation/humanoidarena_eval.md) · [SIMPLE](evaluation/simple_eval.md)

## 目录

- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [模型权重](#模型权重)
- [数据准备](#数据准备)
- [训练](#训练)
- [推理与评估](#推理与评估)
- [真机部署](#真机部署)
- [常见问题](#常见问题)

## 项目结构

| 路径 | 内容 |
| --- | --- |
| train.py, train.yaml | 通用训练入口和默认配置；正式实验建议使用对应的脚本配置 |
| model/, motion/, skeleton/ | Kimodo policy、视觉 ControlNet、动作表示和 G1 骨架 |
| data/ | HumanoidArena、SIMPLE、HIW500、HumanoidEveryday、UnifoLM 和 RealWorld 数据适配 |
| scripts/Pre_Train/ | 多源数据预训练 |
| scripts/HumanoidArena_Multi_Task/ | HumanoidArena SONIC 多任务训练与微调 |
| scripts/HumanoidArena_Single_Task/ | HumanoidArena 单任务训练与微调 |
| scripts/Simple_Single_Task/ | SIMPLE 单任务连续手部控制微调 |
| scripts/Real_World/ | RealWorld 离线数据微调入口 |
| evaluation/ | Kimodo 推理服务与仿真评估启动脚本 |
| HumanoidArena/ | HumanoidArena 仿真、评估和数据工具 |
| SIMPLE/ | SIMPLE 仿真环境及任务代码 |
| checkpoints/ | 本地基础模型和视觉/文本预训练依赖；需从模型仓库下载 |
| log/, eval_results/ | 训练输出及评估结果，默认不纳入 Git |

## 环境配置

### 通用要求

- Linux、NVIDIA GPU 和可用的 CUDA PyTorch。
- 训练脚本默认使用 BF16；请确保 GPU 支持 BF16，并按服务器驱动安装匹配的 PyTorch CUDA wheel。
- Git LFS 用于大型模型文件；Git 子模块用于检出仓库登记的外部代码。
- 本仓库当前没有统一的根目录 requirements.txt 或训练环境 lockfile。下面列出训练代码直接使用的主要依赖；不同 CUDA/驱动环境请选配套的 PyTorch 和 Python wheel 版本。

首次检出代码：

~~~bash
git clone --recurse-submodules https://github.com/Yunheng-Wang/kimodo-policy.git
cd kimodo-policy
git lfs install
git submodule update --init --recursive
~~~

### Kimodo 训练环境

推荐将模型训练和 Kimodo 推理服务放在独立 Conda 环境中。代码使用 Python 3.10 及以上语法；SIMPLE 集成环境固定使用 Python 3.10。

~~~bash
conda create -n kimodo-env python=3.10 -y
conda activate kimodo-env
python -m pip install --upgrade pip

# 先按机器的 NVIDIA driver/CUDA 组合安装 PyTorch。
# PyTorch wheel 选择说明：https://pytorch.org/get-started/locally/
pip install accelerate omegaconf wandb numpy av pyarrow scipy einops \
  pydantic safetensors transformers peft tqdm packaging
~~~

训练脚本通过 Accelerate 启动多 GPU 进程。设置 KIMODO_ENV 后，脚本会从该环境定位 accelerate：

~~~bash
export KIMODO_ENV="$CONDA_PREFIX"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
accelerate --version
~~~

预训练配置会预先计算文本 embedding，因此需要可用的 LLM2Vec / Llama 权重；DINOv3 和 Kimodo motion backbone 也需要放在仓库的 checkpoints/ 目录。见[模型权重](#模型权重)。

### HumanoidArena 仿真环境

HumanoidArena 仿真评估使用独立的 Isaac Sim/Isaac Lab 环境。当前仓库随附的设置文档针对 Ubuntu 22.04+、Isaac Sim 5.0.0 和 Isaac Lab release/2.2.0，仿真环境为 Python 3.11、PyTorch 2.7.0/CUDA 12.8。环境安装助手默认只打印命令：

~~~bash
cd HumanoidArena
bash isaaclab_twist2_g1/tools/setup_humanoidarena_envs.sh --dry-run
~~~

检查输出后再运行安装。助手默认环境名为 unitree_sim_env：

~~~bash
CONDA_BASE=/path/to/conda bash isaaclab_twist2_g1/tools/setup_humanoidarena_envs.sh --execute
~~~

评估还需要 HumanoidArena 仿真资产以及 SONIC policy artifacts。资产包下载地址、放置目录和完整安装步骤见 [HumanoidArena 环境设置](HumanoidArena/docs/04_environment_setup.md)。评估代码将 HumanoidArena/ 作为默认仿真项目目录。HumanoidArena 原生 LeRobot 训练另需 lerobot 环境；Kimodo 的训练和本 README 中的 Kimodo policy 评估不通过该环境运行。

### SIMPLE 仿真环境

SIMPLE 自带独立的 Python 3.10 环境定义，包含 PyTorch 2.7.0、TorchVision 0.22.0 和 NumPy 1.26.4。上游基线为 Ubuntu 22.04、Isaac Sim 4.5/MuJoCo 3.3、CUDA 12 和 NVIDIA driver 535+；建议使用 RTX 3080 Ti/4090 或更高配置，并预留 100 GB 以上磁盘空间。不要把 SIMPLE 仿真环境与 kimodo-env 的训练进程混为一个环境。

~~~bash
cd SIMPLE
# 按 SIMPLE 项目的安装说明安装 uv
UV_HTTP_TIMEOUT=3000 GIT_LFS_SKIP_SMUDGE=1 \
  uv sync --all-groups --index-strategy unsafe-best-match
bash scripts/install_curobo.sh
~~~

CuRobo CUDA 扩展需要本机 CUDA toolkit 和匹配的 GPU 架构；首次编译较慢。安装细节、系统依赖及可选资源下载见 [SIMPLE 安装说明](SIMPLE/docs/source/tutorials/installation.md)。需要预下载 SIMPLE 场景资源时，可在 SIMPLE 根目录运行 bash scripts/pre-minimal-download.sh。当前 SIMPLE 配置针对 Isaac Sim 4.5/MuJoCo 3.3，pyproject.toml 有意不安装 Isaac Sim。本仓库评估脚本默认使用 mujoco_isaac，因此完整评估还需单独安装兼容的 Isaac Sim runtime。SIMPLE/scripts/install_isaaclab.sh 假定 SIMPLE/third_party/IsaacLab 已检出，但当前 release tree 未包含该目录或对应子模块元数据；需要 Isaac Lab 的工作流请按兼容版本先准备其源码，再运行该脚本。本仓库的 SIMPLE 评估脚本通过 SIMPLE/.venv/bin/python 启动模拟器；如果该环境缺少 transformers 或 safetensors，可额外设置 KIMODO_MODEL_SITE_PACKAGES 指向包含这些库的 Python 3.10 site-packages。

## 模型权重

全部 Kimodo 训练 checkpoint 与基础模型权重在 [YunhengWang/kimodo-policy](https://huggingface.co/YunhengWang/kimodo-policy/tree/main)。Git 仓库不包含这些大文件。模型仓库的目录结构与下面的相对路径一致。

### 下载基础模型

在代码仓库根目录运行：

~~~bash
python -m pip install --upgrade huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="YunhengWang/kimodo-policy",
    repo_type="model",
    local_dir=".",
    allow_patterns=["checkpoints/**"],
)
PY
~~~

基础文件布局：

~~~text
checkpoints/
├── Kimodo-G1-RP-v1/                         # 默认 Kimodo motion backbone
├── Kimodo-G1-SEED-v1/                       # SEED 实验使用的 backbone
├── dinov3-vitl16-pretrain-lvd1689m/          # DINOv3 image encoder
├── LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/   # LLM2Vec base model
├── LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/
└── Meta-Llama-3-8B-Instruct/                # Llama base weights/tokenizer
~~~

如果要微调，除了 checkpoints/ 基础模型，还需要下载一个预训练训练 checkpoint。下面以发布的 419h 预训练 1,000,000 步 checkpoint 为例：

~~~bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="YunhengWang/kimodo-policy",
    repo_type="model",
    local_dir=".",
    allow_patterns=[
        "Pre_Train/pt_419h_gbs1024_100w_controlnet4_detach_true_mse/2026-08-26_17-06-35/checkpoint_1000000/**"
    ],
)
PY
~~~

下载后训练 checkpoint 路径为：

~~~text
Pre_Train/pt_419h_gbs1024_100w_controlnet4_detach_true_mse/
└── 2026-08-26_17-06-35/
    └── checkpoint_1000000/
        ├── config.json
        ├── training_state.pt
        └── ...
~~~

评估和微调入口要求 checkpoint 目录中至少有 config.json 与 training_state.pt。--init-checkpoint 用于从已训练模型初始化新任务；--resume 用于从训练状态继续同一轮训练。精确恢复分布式 RNG 时还需要 checkpoint 中对应的 rng_state_rank_*.pt 文件，所以建议下载完整 checkpoint 目录，不要只拷贝上述两个文件。

### 已发布 checkpoint 清单

以下目录在模型仓库中可见（核对日期：2026-10-04）。打开链接后进入对应实验目录，再选择具体日期和 <code>checkpoint_&lt;step&gt;</code>。建议按评估任务下载匹配的 checkpoint。

| 用途 | Hugging Face 目录 | 已发布 checkpoint |
| --- | --- | --- |
| 共享基础模型依赖 | [checkpoints/](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/checkpoints) | Kimodo-G1-RP-v1、Kimodo-G1-SEED-v1、DINOv3、Llama/LLM2Vec |
| 基础预训练 | [Pre_Train/pt_419h_gbs1024_100w_controlnet4_detach_true_mse](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Pre_Train/pt_419h_gbs1024_100w_controlnet4_detach_true_mse) | 200k、400k、600k、800k、1,000k |
| HumanoidArena 多任务，105h 初始化微调 | [ft_105h...](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/ft_105h_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) | 100k、200k、300k、400k、500k |
| HumanoidArena 多任务，419h 初始化微调 | [ft_419h...](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/ft_419h_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) | 100k、200k、300k、400k、500k |
| HumanoidArena 多任务，基础变体 | [SONIC x7](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) | 500k |
| HumanoidArena 多任务，SEED backbone | [SONIC x7 SEED](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse_SEED) | 500k |
| HumanoidArena 多任务，hand head 变体 | [large_hand4](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse_large_hand4) · [large_hand8](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse_large_hand8) | 各 500k |
| HumanoidArena 单任务，从头训练 | [double desk](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_double_desk_sonic) · [football](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_football_sonic) · [pp box](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_pp_box_sonic) · [boxing](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_boxing_sonic) · [open door](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_open_door_sonic) · [sit sofa](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_sit_sofa_sonic) · [vision navi](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_vision_navi_sonic) | 每个 200k |
| HumanoidArena 单任务，419h 初始化微调 | [double desk](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_double_desk_sonic) · [football](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_football_sonic) · [pp box](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_pp_box_sonic) · [boxing](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_boxing_sonic) · [open door](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_open_door_sonic) · [sit sofa](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_sit_sofa_sonic) · [vision navi](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_vision_navi_sonic) | 每个 200k |
| SIMPLE 单任务，continuous hand | [CloseDoor](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand_G1WholebodyCloseDoorTeleop-v0) · [Handover](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand_G1WholebodyHandoverTeleop-v0) · [LocomotionPickBetweenTables](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand_G1WholebodyLocomotionPickBetweenTablesTeleop-v0) · [OpenFaucet](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand_G1WholebodyOpenFaucetTeleop-v0) · [OpenOven](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand_G1WholebodyOpenOvenTeleop-v0) · [OpenTrashCan](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand_G1WholebodyOpenTrashCanTeleop-v0) · [PushOfficeChair](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand_G1WholebodyPushOfficeChairTeleop-v0) · [XMoveBendPick](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand_G1WholebodyXMoveBendPickTeleop-v0) | 每个 200k |
| Real_World | [模型仓库](https://huggingface.co/YunhengWang/kimodo-policy/tree/main) | 目前未发现已发布的 Real_World 训练 checkpoint |

HumanoidArena 和 SIMPLE 的单任务目录包含按任务命名的独立子目录；每个实验目录下的 checkpoint_200000 才是可传给评估器的 checkpoint。预训练 105h/210h 启动配置在代码中提供，但当前 Hugging Face 仓库中列出的预训练 run 是 419h。

基础 Kimodo、DINOv3、Llama/LLM2Vec 权重具有各自的许可条款；使用或再分发前请查看 HF 模型卡及上游许可。训练数据、HumanoidArena 仿真资产和 SIMPLE Level 评估数据不包含在 Kimodo checkpoint 下载步骤中。

## 数据准备

数据根目录既可以通过环境变量覆盖，也可以修改单次实验 YAML 中的 main.dataset_roots。目录需符合 data/ 下对应 adapter 的数据格式；不能只创建一个空目录。

| 数据源 | 环境变量 | 默认目录/选择 |
| --- | --- | --- |
| HumanoidArena | HUMANOID_ARENA_ROOT | datasets/HumanoidArena_dataset_v3_1；多任务配置选择 SONIC RefPose 数据 |
| UnifoLM whole-body | UNIFOLM_ROOT | datasets/UnifoLM_WBT_Dataset；使用 head stereo left camera |
| HumanoidEveryday | HUMANOID_EVERYDAY_ROOT | datasets/HumanoidEveryday |
| HIW500 | HIW500_ROOT | datasets/HIW500 |
| SIMPLE 训练集 | KIMODO_SIMPLE_ROOT | datasets/Simple；任务目录名由脚本参数指定 |
| RealWorld 离线数据 | REAL_WORLD_ROOT | real-world |

预训练配置组合 UnifoLM、HumanoidEveryday 和 HIW500 数据，按配置中的 pretrain_data_fraction 使用 25%、50% 或 100% 的可用 episode，对应 105h、210h 和 419h 实验名。HumanoidArena 多任务数据由配置选择；单任务实验通过启动脚本参数选择任务和 backend。SIMPLE 训练数据根目录与 SIMPLE 官方 Level 0/1/2 评估数据是两套不同输入，后者由评估时的 SIMPLE_DATA_DIR 提供。

数据路径不存在、未选择到 episode 或格式不匹配时，数据 adapter 会报错。可在正式训练前使用检查器抽样审计数据；请显式传入当前配置 YAML，避免依赖检查器的旧默认路径。此检查器审计 episode/schema 与动作连续性，不会解码视频：

~~~bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
python utils/check_datasets.py \
  --config scripts/Pre_Train/pt_419h_gbs1024_100w_controlnet4_detach_true_mse.yaml \
  --workers 1 \
  --limit 5 \
  --output "$PROJECT_ROOT/log/pretrain_dataset_audit.json"
~~~

正式训练前仍建议使用小规模作业验证完整视频读取和 GPU 显存。

## 训练

训练启动器会自动定位项目根目录和配套 YAML，并按 GPU 参数启动 accelerate launch。训练输出默认保存到 <code>log/&lt;任务类别&gt;/&lt;运行名&gt;/&lt;时间戳&gt;/checkpoint_&lt;step&gt;</code>。GPU 列表由 KIMODO_GPUS 指定；预训练脚本默认 8 个 GPU，其余常用脚本默认 4 个。更改 GPU 数量会改变总 batch size；脚本文件名中的 gbs 只有在使用 YAML 对应的进程数时才成立。

### 启动脚本一览

每个 shell 脚本默认读取同目录、同名 YAML；可通过 KIMODO_CONFIG 指定自定义配置。

| 任务 | 启动脚本 | 初始化与默认输出目录 |
| --- | --- | --- |
| 预训练 | [105h](scripts/Pre_Train/pt_105h_gbs1024_100w_controlnet4_detach_true_mse.sh) · [210h](scripts/Pre_Train/pt_210h_gbs1024_100w_controlnet4_detach_true_mse.sh) · [419h](scripts/Pre_Train/pt_419h_gbs1024_100w_controlnet4_detach_true_mse.sh) | 从 Kimodo 基础权重开始；log/Pre_Train/&lt;run-name&gt;/ |
| HumanoidArena 多任务 | [从头训练](scripts/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.sh) · [微调](scripts/HumanoidArena_Multi_Task/ft_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.sh) | 微调脚本需传入 checkpoint；log/HumanoidArena_Multi_Task/&lt;run-name&gt;/ |
| HumanoidArena 多任务变体 | [SEED](scripts/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse_SEED.sh) · [large_hand4](scripts/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse_large_hand4.sh) · [large_hand8](scripts/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse_large_hand8.sh) | 变体设置见配套 YAML；输出在 log/HumanoidArena_Multi_Task/ |
| HumanoidArena 单任务 | [从头训练](scripts/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.sh) · [微调](scripts/HumanoidArena_Single_Task/ft_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.sh) | 从头训练传 TASK BACKEND；微调再传 CHECKPOINT；log/HumanoidArena_Single_Task/&lt;run-name&gt;/ |
| SIMPLE 单任务微调 | [continuous hand](scripts/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand.sh) | 传 TASK CHECKPOINT；log/Simple_Single_Task/&lt;run-name&gt;_&lt;TASK&gt;/ |
| RealWorld 离线微调 | [Real_World](scripts/Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse.sh) | 传 CHECKPOINT 和可选数据集名；log/Real_World/&lt;run-name&gt;_&lt;dataset&gt;/ |

### 多源预训练

先下载基础模型权重，并准备三种预训练数据。419h 配置为全量数据，最多训练 1,000,000 步，每 200,000 步保存一次：

~~~bash
conda activate kimodo-env
export KIMODO_ENV="$CONDA_PREFIX"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export UNIFOLM_ROOT="$PROJECT_ROOT/datasets/UnifoLM_WBT_Dataset"
export HUMANOID_EVERYDAY_ROOT="$PROJECT_ROOT/datasets/HumanoidEveryday"
export HIW500_ROOT="$PROJECT_ROOT/datasets/HIW500"

for data_root in "$UNIFOLM_ROOT" "$HUMANOID_EVERYDAY_ROOT" "$HIW500_ROOT"; do
  test -d "$data_root" || { echo "Missing dataset directory: $data_root" >&2; exit 2; }
done

KIMODO_GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/Pre_Train/pt_419h_gbs1024_100w_controlnet4_detach_true_mse.sh
~~~

其他数据比例可用 [105h](scripts/Pre_Train/pt_105h_gbs1024_100w_controlnet4_detach_true_mse.sh) 或 [210h](scripts/Pre_Train/pt_210h_gbs1024_100w_controlnet4_detach_true_mse.sh) 脚本。

### HumanoidArena 多任务训练

多任务 SONIC 配置使用项目中的 7-task Sonic RefPose 训练集合。以下示例从头训练；ft 版本需要额外传入初始化 checkpoint：

~~~bash
conda activate kimodo-env
export KIMODO_ENV="$CONDA_PREFIX"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export HUMANOID_ARENA_ROOT="$PROJECT_ROOT/datasets/HumanoidArena_dataset_v3_1"
test -d "$HUMANOID_ARENA_ROOT" || { echo "Missing dataset directory: $HUMANOID_ARENA_ROOT" >&2; exit 2; }

KIMODO_GPUS=0,1,2,3 \
  bash scripts/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.sh

INIT_CKPT=/path/to/checkpoint_1000000
KIMODO_GPUS=0,1,2,3 \
  bash scripts/HumanoidArena_Multi_Task/ft_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.sh \
  "$INIT_CKPT"
~~~

同一个 ft_humanoidarena_sonicx7 启动器可接收来自 105h 或 419h 的初始化 checkpoint；SEED、large_hand4 和 large_hand8 变体各有独立配置，均位于 [HumanoidArena_Multi_Task](scripts/HumanoidArena_Multi_Task/)。

### HumanoidArena 单任务训练

从头训练示例：

~~~bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export HUMANOID_ARENA_ROOT="$PROJECT_ROOT/datasets/HumanoidArena_dataset_v3_1"
test -d "$HUMANOID_ARENA_ROOT" || { echo "Missing dataset directory: $HUMANOID_ARENA_ROOT" >&2; exit 2; }
KIMODO_GPUS=0,1,2,3 \
  bash scripts/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.sh \
  doubledesk sonic
~~~

从多源预训练 checkpoint 微调：

~~~bash
KIMODO_GPUS=0,1,2,3 \
  bash scripts/HumanoidArena_Single_Task/ft_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.sh \
  doubledesk sonic /path/to/checkpoint_1000000
~~~

可选 task 和 backend 取决于数据目录中实际存在的标注；常见 SONIC task 为 doubledesk、football、pp_box、boxing、open_door、sit_sofa 和 vision_navi。单任务脚本对应配置默认 batch size 为每进程 16、训练 200,000 步。

### SIMPLE 单任务微调

此任务使用 Kimodo 训练环境；KIMODO_SIMPLE_ROOT 指向 SIMPLE 格式化后的训练数据根目录。SIMPLE 仿真依赖只在评估时使用：

~~~bash
conda activate kimodo-env
export KIMODO_ENV="$CONDA_PREFIX"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export KIMODO_SIMPLE_ROOT="$PROJECT_ROOT/datasets/Simple"
test -d "$KIMODO_SIMPLE_ROOT" || { echo "Missing dataset directory: $KIMODO_SIMPLE_ROOT" >&2; exit 2; }

KIMODO_GPUS=0,1,2,3 \
  bash scripts/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand.sh \
  G1WholebodyXMovePickTeleop-v0 /path/to/checkpoint_1000000
~~~

脚本把任务名写入 run 目录，默认输出在 log/Simple_Single_Task/ 下。其他 task 需在训练数据根目录下有对应任务目录。

### RealWorld 离线数据微调

离线演示数据微调脚本位于 [scripts/Real_World](scripts/Real_World/)，需要 REAL_WORLD_ROOT 下的兼容数据集和初始化 checkpoint。真机部署流程暂未发布，见[真机部署](#真机部署)。

训练配置可通过 KIMODO_CONFIG=/path/to/experiment.yaml 覆盖默认 YAML；可通过 KIMODO_GPUS 指定进程使用的 GPU。单机多任务使用不同训练脚本时，如果 master port 冲突，可设置各自不同的 KIMODO_MASTER_PORT。

## 推理与评估

评估 checkpoint 需同时具备 config.json 和 training_state.pt。HumanoidArena 与 SIMPLE 的官方 wrapper 会启动 Kimodo inference server、运行对应模拟器并写出结果。评估数据、任务环境和 checkpoint 必须匹配。

### HumanoidArena SONIC

准备好 HumanoidArena 的 unitree_sim_env、资产和 SONIC policy artifacts。Shell 会从 KIMODO_SIM_ENV 启动仿真端，从 KIMODO_SERVER_PYTHON 启动模型服务端：

~~~bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
CONDA_BASE="$(conda info --base)"
CHECKPOINT=/path/to/downloaded/checkpoint_200000

export HUMANOIDARENA_ROOT="$PROJECT_ROOT/HumanoidArena"
export KIMODO_SIM_ENV="$CONDA_BASE/envs/unitree_sim_env"
export KIMODO_SERVER_PYTHON="$CONDA_BASE/envs/kimodo-env/bin/python"

bash evaluation/humanoidarena_eval_signal_task_sonic.sh \
  --project "$PROJECT_ROOT" \
  --task doubledesk \
  --checkpoint "$CHECKPOINT" \
  --gpus 0,1,2 \
  --seeds 0,1,2 \
  --repeats 20 \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 15 \
  --rtc 0 \
  --results-dir "$PROJECT_ROOT/eval_results/doubledesk"
~~~

多任务连续评估可改用 [humanoidarena_eval_multi_task_sonic.sh](evaluation/humanoidarena_eval_multi_task_sonic.sh)，并以 --task doubledesk football pp_box 指定顺序。TWIST2 使用独立 wrapper [humanoidarena_eval_signal_task_twist2.sh](evaluation/humanoidarena_eval_signal_task_twist2.sh)。详细的 seeds、RTC、录视频和正式统计参数见 [HumanoidArena 评估说明](evaluation/humanoidarena_eval.md)。

### SIMPLE Level 0/1/2

SIMPLE 评估 wrapper 使用本仓库的 SIMPLE/.venv 启动仿真。SIMPLE_DATA_DIR 应指向 SIMPLE data 根目录，包含仿真需要的资源及官方 Level 数据；脚本会在其下查找 <code>simple-eval/&lt;任务名&gt;/dr-level-0/</code>、<code>dr-level-1/</code> 和 <code>dr-level-2/</code>（部分数据集使用 level-0/1/2 命名）。每个 Level 的评估数据应包含 meta/episodes.jsonl。

~~~bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
CONDA_BASE="$(conda info --base)"
CHECKPOINT=/path/to/Simple_Single_Task/checkpoint_200000
export SIMPLE_DATA_DIR=/path/to/simple-eval-data
# 可选：仅当 SIMPLE/.venv 缺少 transformers 或 safetensors 时设置
export KIMODO_MODEL_SITE_PACKAGES="$CONDA_BASE/envs/kimodo-env/lib/python3.10/site-packages"

bash evaluation/simple_eval_signal_task_sonic.sh \
  --task G1WholebodyXMoveBendPickTeleop-v0 \
  --checkpoint "$CHECKPOINT" \
  --gpus 0,1,2 \
  --levels 0,1,2 \
  --seeds 0 \
  --episodes 10 \
  --simple-data-dir "$SIMPLE_DATA_DIR" \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 15 \
  --rtc 0 \
  --max-navigation-speed 10.0 \
  --max-steps auto \
  --results-dir "$PROJECT_ROOT/eval_results/simple_xmove_bendpick"
~~~

官方 Level 数据需要匹配 --task，每个 Level 通常有固定环境 episode。SIMPLE task 会决定 Teleop/Sonic 或 motion-planning/AMO 的动作适配分支，不要只换 task 字符串来把一种 checkpoint 当作另一种控制器评估。完整评估输入、episode 结果、视频和成功率统计见 [SIMPLE 评估说明](evaluation/simple_eval.md)。

评估结果默认位于 eval_results/。HumanoidArena wrapper 输出 summary.json、final.csv、逐 episode 结果和可选视频；SIMPLE wrapper 按 Level 分别保存日志、统计及视频。可为每次正式实验设置独立的 --results-dir。

## 真机部署

**待 release。** 真机硬件配置、部署依赖、启动命令和安全操作说明将在后续版本发布。

## 常见问题

**找不到 Kimodo 或 DINOv3 权重**

确认 checkpoints/ 位于项目根目录，并包含 Kimodo-G1-RP-v1/config.yaml、Kimodo-G1-RP-v1/model.safetensors 以及 dinov3-vitl16-pretrain-lvd1689m/。使用 SEED 配置时还需下载 Kimodo-G1-SEED-v1/。

**checkpoint 不完整**

评估/微调使用的是训练输出目录中的 <code>checkpoint_&lt;step&gt;</code>，而不是只含 model.safetensors 的基础模型目录。检查 checkpoint 中的 config.json 和 training_state.pt。

**数据路径报错**

将 HUMANOID_ARENA_ROOT、UNIFOLM_ROOT、HUMANOID_EVERYDAY_ROOT、HIW500_ROOT 或 KIMODO_SIMPLE_ROOT 设置为真实数据路径。路径存在但没有兼容 episode 时也会报错；查看所选 YAML 的 dataset_selection 和对应 data/*_loader.py。

**SIMPLE 评估找不到模型依赖**

评估默认使用 SIMPLE/.venv/bin/python。确认 uv 环境已安装；若 transformers 或 safetensors 位于 Kimodo Conda 环境，可设置 KIMODO_MODEL_SITE_PACKAGES。

**训练中断后恢复**

使用同一配置和数据根目录，通过 train.py --resume /path/to/checkpoint_STEP 恢复优化器、scheduler 和 global step。若只是将一个预训练模型迁移到新任务，使用对应脚本的 checkpoint 参数（内部传给 --init-checkpoint），训练进度和优化器会重新开始。
