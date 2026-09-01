# Kimodo + SIMPLE 单权重评估

本文件只说明单个任务、单个 checkpoint 的官方 SIMPLE Level 0/1/2 评估。
唯一需要使用的启动脚本是：

```text
evaluation/simple_eval_signal_task_sonic.sh
```

实现边界：`evaluation/humanoidarena_server.py` 保持 Arena 原始协议和默认
推理行为，不包含 SIMPLE 的速度保护、chunk 诊断或 root 边界修正。SIMPLE
专用适配全部位于 `evaluation/simple_server.py`，只在本评估脚本启动的
SIMPLE 进程中使用。因此 Arena 的其它评估脚本不受本逻辑影响。

## 1. 远端环境

以下命令在 4090 主机 `ai` 上执行。反向 SSH 隧道已经建立时，可以从当前
机器验证连接：

```bash
ssh -p 43175 root@127.0.0.1
```

进入远端 Kimodo 目录并设置资源路径：

```bash
cd /ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2
export SIMPLE_DATA_DIR=/ai/Yichi/kimodo-policy/simpledata
export KIMODO_MODEL_SITE_PACKAGES=/ai/Yichi/0_Systems/miniconda3/envs/MOGE3/lib/python3.10/site-packages
```

评估器使用 `SIMPLE/.venv/bin/python` 运行仿真和 Torch。MOGE3 环境只提供
训练模型需要的 `transformers`、`safetensors` 等依赖，不替换 SIMPLE 的
Torch。启动脚本会自动探测 MOGE3，但建议显式导出上面的变量。

## 2. 官方三等级启动命令

以下命令会对一个 checkpoint 自动执行三个等级：

```bash
bash evaluation/simple_eval_signal_task_sonic.sh \
  --task G1WholebodyCloseDoorTeleop-v0 \
  --checkpoint log/experiments/simple_single_gbs64_20w_controlnet4_detach_true_mse_G1WholebodyCloseDoorTeleop-v0/2026-08-28_23-55-15/checkpoint_100000 \
  --gpus 4,5,6 \
  --seeds 0 \
  --episodes 10 \
  --simple-data-dir /ai/Yichi/kimodo-policy/simpledata \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 15 \
  --rtc 0 \
  --rtc-overlap-frames 0 \
  --rtc-frozen-frames 0 \
  --rtc-ramp-power 1.0
```

上面命令中的推理参数是本项目当前约定，后续评估应保持这一组值：
`fp32`、10 步扩散、每次执行 15 帧、关闭 RTC。不要因为脚本内部还有
其他默认值而省略这些参数。

默认行为是：

```text
Level 0: 10 个 episode
Level 1: 10 个 episode
Level 2: 10 个 episode
总计:    30 个 episode
```

三个等级的数据来自：

```text
/ai/Yichi/kimodo-policy/simpledata/simple-eval/<TASK>/dr-level-0/
/ai/Yichi/kimodo-policy/simpledata/simple-eval/<TASK>/dr-level-1/
/ai/Yichi/kimodo-policy/simpledata/simple-eval/<TASK>/dr-level-2/
```

部分数据集使用 `level-0`、`level-1`、`level-2` 命名；脚本两种命名都支持。
每个目录必须包含 `meta/episodes.jsonl`，其中有 10 个固定环境配置。

评估时会读取每个 episode 的 `environment_config`，并执行：

```python
env.reset(options={"state_dict": environment_config})
```

因此使用的是官方固定 Level 环境，而不是重新随机生成环境。

## 3. 参数说明

### 任务和权重

`--task NAME`：SIMPLE 任务名，可写 `G1WholebodyCloseDoorTeleop-v0`，也可写
带 `simple/` 前缀的完整环境名。

`--checkpoint PATH`：要评估的单个 checkpoint 目录，必须同时包含
`config.json` 和 `training_state.pt`。

### 等级、回合和 GPU

`--levels LIST`：要评估的等级，默认 `0,1,2`。例如 `--levels 0` 只运行
Level 0。

`--episodes N`：每个等级运行的 episode 数，默认 `10`。官方结果使用 10；
`--episodes 1` 仅用于 smoke test。

`--gpus LIST`：物理 GPU 编号，例如 `--gpus 4,5,6`。Level 0、1、2 会依次
分配到 GPU 4、5、6；如果 GPU 数量少于等级数量，剩余等级会等待上一波完成。
启动前应确认这些 GPU 没有其他任务，否则可能发生显存不足。

`--seeds LIST`：运行时 reset 使用的 seed。固定官方环境由
`episodes.jsonl` 决定，通常保持 `--seeds 0` 即可。

### 环境和数据

`--simple-data-dir PATH`：SIMPLE 资源根目录，必须包含机器人、场景、材质和
assets。当前主机使用 `/ai/Yichi/kimodo-policy/simpledata`。

`--eval-data-root PATH`：Level 评估数据根目录，默认是
`$SIMPLE_DATA_DIR/simple-eval`。只有评估数据不在该位置时才需要显式指定。

`--python PATH`：Python 可执行文件，默认 `SIMPLE/.venv/bin/python`，也可以
通过环境变量 `KIMODO_PYTHON` 指定。

### 模型推理和仿真

`--dtype bf16|fp32`：模型推理精度。脚本默认 `bf16`，但本项目启动命令固定
使用 `fp32`。

`--diffusion-steps N`：扩散采样步数，默认 `10`。

`--execution-frames N`：每次重新规划执行的模型帧数。`0` 表示执行完整
action chunk；本项目固定使用 `15`。

`--rtc 0|1`：是否启用实时 chunking。本项目固定使用 `0`（关闭）。

`--rtc-overlap-frames N`、`--rtc-frozen-frames N`、`--rtc-ramp-power X`：
RTC 的重叠帧数、冻结前缀帧数和 ramp 指数。本项目固定传入 `0`、`0`、`1.0`；
RTC 关闭时前两项不会生效。

`--max-navigation-speed X`：导航速度保护阈值，单位 m/s，默认 `1.5`。每个
模型 chunk 编码后会检查平面导航速度；超过阈值时该 episode 立即报错并保留
`pipeline.log`，避免继续运行明显失真的 root 边界。日志中的
`[chunk_diagnostics]` 还会记录 state 段帧数、首帧/最大导航速度、根高度范围
和关节边界变化。

`--sim-mode mujoco|mujoco_isaac`：仿真模式。官方视觉评估使用
`mujoco_isaac`，默认也是该模式；`mujoco` 只适合轻量诊断。

`--max-steps N|auto`：每个 episode 的 TimeLimit。默认 `auto`，使用 SIMPLE
任务自身 `metadata["max_episode_steps"]`（例如 XMovePick 是 800，CloseDoor
任务当前定义为 450）。只有确实需要覆盖官方任务设置时才传入整数。

任务结束有两种来源：任务的 `check_success()` 返回真时
`terminated=True`，环境将其记为成功；达到 TimeLimit 时
`truncated=True`，记录为 `timeout` 并判定失败。Sonic 的稳定化预热步数不计入
正式 episode 的 TimeLimit，评估器会在预热后重置 Gym 的计数器。每个 episode
JSON 会写出 `terminated`、`truncated`、`termination_reason` 和实际
`max_episode_steps`，可以直接判断是成功还是超时。

`--no-save-video`：关闭视频录制。官方评估不要使用此参数。

`--dry-run`：只检查 checkpoint、资源目录、Level 数据目录和启动计划，不加载
模型或启动 Isaac Sim。

## 4. 启动前检查

建议先执行：

```bash
bash evaluation/simple_eval_signal_task_sonic.sh \
  --task G1WholebodyCloseDoorTeleop-v0 \
  --checkpoint log/experiments/simple_single_gbs64_20w_controlnet4_detach_true_mse_G1WholebodyCloseDoorTeleop-v0/2026-08-28_23-55-15/checkpoint_100000 \
  --gpus 4,5,6 \
  --episodes 10 \
  --simple-data-dir /ai/Yichi/kimodo-policy/simpledata \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 15 \
  --rtc 0 \
  --rtc-overlap-frames 0 \
  --rtc-frozen-frames 0 \
  --rtc-ramp-power 1.0 \
  --dry-run
```

查看 GPU 是否空闲：

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
```

## 5. 输出结果

默认输出目录：

```text
eval_results/simple/<TASK>_<CHECKPOINT>_<TIMESTAMP>/
```

目录结构：

```text
level_0/
level_1/
level_2/
summary.json
.done
```

每个 `level_N/` 包含 `pipeline.log`、10 个 episode JSON 和 10 个 ego MP4：

```text
ego_view_success.mp4 或 ego_view_failed.mp4
```

`summary.json` 记录每个 Level 的 episode 数、成功数和成功率。只有三个等级
都完成 10 个 episode，并且顶层 `.done` 存在时，才算一次完整的官方评估。

## 6. 本项目的时序对齐和安全诊断

评估桥接器会严格按 50 Hz 收集 SIMPLE 的 `info["proprio"]`：首次请求发送
稳定化后的当前帧，后续请求发送上一次请求之后每个 `env.step()` 产生的全部帧。
模型 64-D state 不含平面 root x/z，因此 chunk 边界不使用扩散输出的
`history_last_root_position`，而是由 future chunk 的首个速度外推得到，避免
窗口坐标系变化造成几十倍的虚假导航速度。

每个成功编码的 chunk 都会输出一行 `[chunk_diagnostics]`，包括：
`state_segment_frames`、首帧/最大导航速度、根高度范围和关节边界变化。若最大
平面速度超过 `--max-navigation-speed`（默认 1.5 m/s），推理会 fail-fast，
并将完整错误保留在对应 Level 的 `pipeline.log` 中。这样可以在正式跑 30 个
episode 前发现训练/推理时序或坐标系再次失配。

手部还会记录以下诊断字段：

```text
model_hand_active=[left,right]/50
control_hand_active=[left,right]/25
hand_tail_promoted=[left,right]
hand_latched=[left,right]
```

checkpoint 的模型以 30 Hz 预测 50 帧，而评估命令每次执行前 15 个模型帧
（约 25 个 50 Hz 控制帧）。因此如果闭合状态落在第 15 帧之后，SIMPLE 适配层会
将连续两帧以上的尾部闭合事件提升到执行边界，并在重规划间做去抖保持；连续三次
完整“打开”预测后才释放。这样不会把合法的离散抓取状态永久截掉。适配层还显式
处理了 MuJoCo（thumb-middle-index）与 WBC（thumb-index-middle）的手指顺序，
这些逻辑只在 `evaluation/simple_server.py` 生效，不改变 Arena 原始运行时。
