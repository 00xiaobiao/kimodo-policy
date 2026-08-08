# Kimodo Policy

本仓库目前主要包含 Kimodo Policy 的训练、模型推理服务，以及在 HumanoidArena 上使用 SONIC tracker 进行闭环评估的代码。

主要流程：

```text
训练数据 -> Kimodo Policy -> 40D reference action
                              |
                              v
                    HumanoidArena / SONIC tracker
                              |
                              v
                        29D 机器人关节目标
```

## 目录结构

```text
.
├── train.py                         # 训练入口
├── train.yaml                       # 默认训练配置
├── data/                            # 数据集读取和多数据源采样
├── model/                           # Kimodo、ControlNet、Diffusion 和 hand head
├── motion/                          # 运动表示、root 处理和特征工具
├── skeleton/                        # G1 骨架和 FK 资源
├── scripts/                         # 多卡训练启动脚本
├── evaluation/
│   ├── humanoidarena_server.py      # Kimodo HTTP 推理服务
│   └── humanoidarena_eval.sh        # HumanoidArena 统一评估入口
└── HumanoidArena/                   # Isaac Lab 环境和 SONIC/TWIST2 控制链路
```

## 外部依赖

训练和 Arena 评估使用两个独立的 Python 环境：

- **Kimodo 环境**：PyTorch、Accelerate、DINOv3、LLM2Vec、W&B 等，用于训练和模型服务。
- **HumanoidArena 环境**：Isaac Sim、Isaac Lab 及机器人仿真依赖，用于 Arena 闭环评估。

HumanoidArena 的安装说明见 [环境配置文档](HumanoidArena/docs/04_environment_setup.md)。

以下大文件没有提交到 Git：

- 训练数据集
- 训练 checkpoint 和 W&B 日志
- `HumanoidArena/isaaclab_twist2_g1/assets/` 场景资源
- SONIC encoder/decoder ONNX 模型
- 预计算的任务文本 embedding cache

Arena 评估默认要求 SONIC 模型位于：

```text
HumanoidArena/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/
├── model_encoder.onnx
└── model_decoder.onnx
```

HumanoidArena 基础 assets 可以通过下面的脚本下载：

```bash
cd HumanoidArena/isaaclab_twist2_g1
bash fetch_assets.sh
```

## 训练

### 1. 配置数据和训练参数

编辑 [`train.yaml`](train.yaml)，重点检查：

```yaml
main:
  batch_size: 32
  max_steps: 200000
  save_steps: 50000
  action_chunk: 50
  action_history: 100
  save_root: "log/"

  dataset_roots:
    HumanoidArena: /path/to/HumanoidArena_dataset

  dataset_selection:
    HumanoidArena:
      merged: all_16_refpose_v3_1
```

还需要根据机器配置调整：

- `main.cpu_workers_num`
- `main.batch_size`
- `main.gradient.grad_accumulation_steps`
- `model.dinov3_checkpoint`
- `model.compile`
- optimizer 和 scheduler 参数

如果 `precompute_text_embeddings: true`，训练会优先读取任务文本 embedding cache；cache 不存在时需要本地 LLM2Vec checkpoint。

### 2. 单卡训练

```bash
python train.py --config train.yaml
```

### 3. 多卡训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --mixed_precision bf16 \
  --main_process_port 29647 \
  train.py \
  --config train.yaml
```

也可以使用仓库中的启动脚本，并覆盖环境、GPU 和配置路径：

```bash
KIMODO_ENV=/path/to/kimodo_env \
KIMODO_GPUS=0,1,2,3 \
KIMODO_CONFIG=./train.yaml \
bash scripts/HOI_double_desk.sh
```

### 4. 恢复训练

恢复训练时，GPU 数量和影响模型结构的配置必须与 checkpoint 兼容：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --mixed_precision bf16 \
  train.py \
  --config train.yaml \
  --resume /path/to/run/checkpoint_100000
```

每个 checkpoint 目录至少包含：

```text
checkpoint_100000/
├── training_state.pt
├── config.json
└── rng_state_rank_*.pt
```

训练结果默认写入 `main.save_root`，当前默认值为 `log/`。

## 推理

### 启动 HTTP 模型服务

```bash
python evaluation/humanoidarena_server.py \
  --checkpoint /path/to/checkpoint_100000 \
  --text-embedding-cache /path/to/text_embedding_cache \
  --device cuda:0 \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 15 \
  --rtc 0 \
  --host 127.0.0.1 \
  --port 18080
```

检查服务状态：

```bash
curl http://127.0.0.1:18080/health
```

服务提供两个主要接口：

- `POST /reset`：使用 episode seed 重置推理状态。
- `POST /infer`：接收任务、前视图图像和机器人 state history，返回 action chunk。

当前 HumanoidArena V3.1 action 每帧为 40 维：

```text
[0:2]   root reference local XY delta
[2]     root Z
[3:9]   root rotation 6D
[9:38]  29D reference joint q
[38:40] left/right hand binary state
```

一般不需要手工构造 `/infer` 请求，Arena 评估脚本会负责启动模型服务、发送观测并消费 action chunk。

## HumanoidArena 评估

### 1. 必要文件

开始评估前确认：

```text
/path/to/checkpoint_100000/training_state.pt
/path/to/checkpoint_100000/config.json
/path/to/text_embedding_cache/
HumanoidArena/isaaclab_twist2_g1/assets/
HumanoidArena/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx
HumanoidArena/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx
```

### 2. 查看支持的任务

```bash
bash evaluation/humanoidarena_eval.sh --list-tasks
```

当前统一入口支持：

```text
football
sit_sofa
vision_navi
boxing
open_door
doubledesk
pp_box
```

### 3. 先检查启动配置

`--dry-run` 只校验路径和参数，不启动模型或仿真：

```bash
KIMODO_SERVER_PYTHON=/path/to/kimodo/bin/python \
KIMODO_SIM_PYTHON=/path/to/unitree_sim_env/bin/python \
bash evaluation/humanoidarena_eval.sh \
  --project . \
  --task doubledesk \
  --checkpoint /path/to/checkpoint_100000 \
  --gpus 0 \
  --rtc 0 \
  --dry-run
```

### 4. 运行确定性评估

下面的配置会为不同 seed 启动独立进程，并记录每个 episode 的视频：

```bash
KIMODO_SERVER_PYTHON=/path/to/kimodo/bin/python \
KIMODO_SIM_PYTHON=/path/to/unitree_sim_env/bin/python \
bash evaluation/humanoidarena_eval.sh \
  --project . \
  --task doubledesk \
  --checkpoint /path/to/checkpoint_100000 \
  --gpus 5,6,7 \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 15 \
  --rtc 0 \
  --persistent-sim 0 \
  --deterministic-eval 1 \
  --record-video-every-n 1 \
  --port-base 18080 \
  --results-dir ./eval_results/doubledesk_deterministic
```

说明：

- `--execution-frames 0`：执行模型预测的完整 action chunk。
- `--execution-frames N`：只执行前 N 个模型帧，然后重新规划。
- `--rtc 1`：使用上一 chunk 未执行尾部作为下一次 diffusion 的连续性先验。
- `--rtc 0`：每个 chunk 根据最新观测独立生成，目前建议作为成功率基线。
- `--deterministic-eval 1`：固定每次 diffusion 的随机数；要求 `--persistent-sim 0`。
- `--record-video-every-n 0`：关闭视频录制。

如需测试 RTC，可以额外设置：

```bash
--rtc 1 \
--rtc-overlap-frames 12 \
--rtc-frozen-frames 1 \
--rtc-ramp-power 1.0
```

### 5. 评估输出

结果目录主要包含：

```text
eval_results/<run>/
├── run_config.txt
├── aggregate_summary.json
├── seed_<seed>/
│   ├── summary.json
│   ├── summary.csv
│   ├── final.csv
│   ├── episodes/
│   ├── logs/
│   ├── recordings/
│   └── videos/
└── .done 或 .failed
```

`aggregate_summary.json` 汇总全部 seed 的 episode 数、成功数、失败数和总体成功率。

## 常见问题

### 找不到 checkpoint

评估 checkpoint 必须同时包含 `training_state.pt` 和 `config.json`。本仓库的 `.gitignore` 不会提交 checkpoint。

### 找不到 SONIC 模型

确认 `model_encoder.onnx` 和 `model_decoder.onnx` 位于默认 SONIC release 目录，或调整评估脚本使用的 `SONIC_POLICY_ROOT`。

### 找不到 Isaac Sim Python

显式设置：

```bash
export KIMODO_SIM_PYTHON=/path/to/unitree_sim_env/bin/python
```

### 模型服务端口冲突

修改 `--port-base`。并行评估会从该端口开始，为每个 GPU slot 分配一个端口。

### `execution_frames` 参数错误

`--execution-frames` 不能超过 checkpoint `config.json` 中的 `main.action_chunk`。

## 相关文档

- [HumanoidArena 环境配置](HumanoidArena/docs/04_environment_setup.md)
- [HumanoidArena Evaluation](HumanoidArena/docs/03_evaluation.md)
- [V3.1 action protocol](HumanoidArena/isaaclab_twist2_g1/docs/UNITREE_G1_GMT_REFPOSE_V3_1_DATA_PROTOCOL.md)
- [SONIC data format](HumanoidArena/isaaclab_twist2_g1/docs/SONIC_DATA_FORMAT.md)
