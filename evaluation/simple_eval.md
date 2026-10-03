# Kimodo + SIMPLE 单权重评估

本文件说明单个任务、单个 checkpoint 的官方 SIMPLE Level 0/1/2 评估，
同时适用于 G1 遥操 Teleop（Sonic WBC）和 G1 MP（AMO）任务。
两类任务使用同一个启动脚本：

```text
evaluation/simple_eval_signal_task_sonic.sh
```

脚本名中的 `sonic` 不代表强制使用 Sonic 控制器。`--task` 对应的 SIMPLE
环境决定机器人类型，`evaluation/simple_server.py` 使用相应的动作适配分支；
不需要额外的 `--mp` 或 `--amo` 参数。任务名、checkpoint 和 Level 数据应
对应同一个任务，不能仅将 MP 任务名改成 Teleop 来切换控制器。

| 项目 | G1 Teleop | G1 MP |
|---|---|---|
| 任务名示例 | `G1WholebodyBendPickTeleop-v0` | `G1WholebodyBendPickMP-v0` |
| 机器人／控制器 | `g1_sonic`／Sonic WBC | `g1_wholebody`／AMO |
| 动作适配 | 原有 Sonic agent 和 decoupled WBC | SIMPLE 36 维动作映射到 `eval_move_actuators` |
| 物理步长 | 当前 Teleop 任务配置为 `0.005` 秒 | 当前 MP 任务配置为 `0.002` 秒 |
| 预热 | Sonic 自身的稳定化流程 | 官方 `StandStabilizationWrapper`，60 步站立 |
| 回合上限 | `--max-steps auto`，取任务 metadata | 同样取任务 metadata；BendPickMP 为 400 步 |

MP 的动作映射、chunk 队列、观测历史和 AMO 设备兼容适配位于本项目评估器中；
这次 MP 修复没有编辑 `SIMPLE/` 源码文件，Sonic agent/WBC 分支保持原有逻辑。

实现边界：`evaluation/humanoidarena_server.py` 保持 Arena 原始协议和默认
推理行为，不包含 SIMPLE 的速度保护、chunk 诊断或 root 边界修正。SIMPLE
专用适配全部位于 `evaluation/simple_server.py`，只在本评估脚本启动的
SIMPLE 进程中使用。因此 Arena 的其它评估脚本不受本逻辑影响。

## 1. 远端环境

以下命令在 4090 主机 `ai` 上执行。反向 SSH 隧道已经建立时，可以从当前
机器验证连接：

```bash
ssh -p 43176 root@127.0.0.1
```

登录后先确认 `hostname` 为 `ai`、`whoami` 为 `root`，且 `nvidia-smi` 正常。
`127.0.0.1:43176` 是笔记本本地转发端口；后续项目命令均在登录后的远端执行。
保留笔记本的 SSH 转发进程运行。

长时间评估请在远端 tmux 会话内运行，例如 `tmux new-session -s simple_eval_mp`。
进入会话后设置下面的环境，再执行评估命令；可用 `Ctrl-b d` 分离会话。
定期检查进程、GPU 占用和各 Level 的 `pipeline.log`。

进入远端 Kimodo 目录并设置资源路径：

```bash
cd "$(git rev-parse --show-toplevel)"
export SIMPLE_DATA_DIR=/path/to/simpledata
export KIMODO_MODEL_SITE_PACKAGES=/path/to/model-env/lib/python3.10/site-packages
```

评估器使用 `SIMPLE/.venv/bin/python` 运行仿真和 Torch。MOGE3 环境只提供
训练模型需要的 `transformers`、`safetensors` 等依赖，不替换 SIMPLE 的
Torch。若 SIMPLE 环境缺少模型依赖，请将上面的
`KIMODO_MODEL_SITE_PACKAGES` 设置为包含 `transformers` 和 `safetensors` 的环境目录。

如果 PATH 找不到 `ninja`，或 `CC/CXX` 指向不存在的
Conda 编译器，可在同一 tmux 会话中使用以下已验证配置：

```bash
export PATH="$CONDA_PREFIX/bin:$PATH"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
```

这些是进程级依赖配置，不是 MP／Sonic 切换开关。仍由启动脚本选择
`SIMPLE/.venv/bin/python`。脚本已设置 `MUJOCO_GL=egl`、Isaac 的物理 GPU
选择和各 worker 的缓存目录，无需直接拼接 `simple_server.py` 启动命令。

## 2. 官方三等级启动命令

### 2.1 Teleop／Sonic 示例

以下命令会对一个 Teleop checkpoint 自动执行三个等级：

```bash
bash evaluation/simple_eval_signal_task_sonic.sh \
  --task G1WholebodyCloseDoorTeleop-v0 \
  --checkpoint log/experiments/simple_single_gbs64_20w_controlnet4_detach_true_mse_G1WholebodyCloseDoorTeleop-v0/2026-08-28_23-55-15/checkpoint_100000 \
  --gpus 4,5,6 \
  --seeds 0 \
  --episodes 10 \
  --simple-data-dir "$SIMPLE_DATA_DIR" \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 15 \
  --rtc 0 \
  --rtc-overlap-frames 0 \
  --rtc-frozen-frames 0 \
  --rtc-ramp-power 1.0 \
  --max-navigation-speed 10.0
```

### 2.2 MP／AMO 示例

下面使用已完成验证的 BendPickMP `checkpoint_80000`，GPU 5、6、7 分别
运行 Level 0、1、2，每个等级 10 回合。每次运行前重新生成 `MP_RESULTS_DIR`，
使用新的输出目录，不要复用已经有结果的目录。

```bash
MP_RESULTS_DIR="${PWD}/eval_results/mp_bendpick_table1_ckpt80000_exec15_gpu567_$(date +%Y%m%d_%H%M%S)"

bash evaluation/simple_eval_signal_task_sonic.sh \
  --task G1WholebodyBendPickMP-v0 \
  --checkpoint log/simple_table_1/simple_single_gbs64_8w_controlnet4_detach_true_mse_continuous_hand_G1WholebodyBendPickMP-v0/2026-09-07_17-22-38/checkpoint_80000 \
  --gpus 5,6,7 \
  --levels 0,1,2 \
  --seeds 0 \
  --episodes 10 \
  --simple-data-dir "$SIMPLE_DATA_DIR" \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 15 \
  --rtc 0 \
  --rtc-overlap-frames 0 \
  --rtc-frozen-frames 0 \
  --rtc-ramp-power 1.0 \
  --max-navigation-speed 10.0 \
  --max-steps auto \
  --results-dir "$MP_RESULTS_DIR"
```

复用以上命令评估其他 MP 权重时，替换任务名、匹配的 checkpoint 和结果目录。
无需增加控制器参数。正式评估使用 `--max-steps auto`；BendPickMP 的当前
任务上限是 400 步，不能用 v9 的 200 步短测上限替代完整评估。

**GPU 分配限制：** `--gpus 5,6,7` 指定模型和 Isaac 渲染 worker 使用的物理卡。
当前 MP 兼容适配将 AMO 固定到 `cuda:0`，因为随附 AMO TorchScript 内含
该设备的常量；因此 MP worker 还会占用物理 GPU 0。Isaac／Warp 等仿真库也
可能在其他可见卡上建立 CUDA 上下文；这不是严格的 GPU 显存隔离。
启动前应同时检查 GPU 0 和目标 GPU。Sonic 不使用 AMO 设备适配。

在 `mujoco_isaac` 模式下，脚本会清除 `CUDA_VISIBLE_DEVICES` 并显式选择
物理 GPU。不要另行套用 CUDA 掩码来推断 `cuda:0` 是 GPU 5，也不要把
`--gpus` 当成限制所有 CUDA 上下文只存在于这些卡上的开关。

### 2.3 两类任务共用的评估协议

以上命令使用 15 帧执行长度配置：`fp32`、10 步扩散、每次预测
50 帧但只执行前 15 帧、关闭 RTC，然后根据最新观测重新预测。启动脚本的
执行长度默认值仍为 `50`，因此必须显式传入 `--execution-frames 15`。

比较 50 帧与 15 帧执行长度时，已有结果保持不变。请为每次评估使用独立的
`--results-dir`（建议目录名包含 `exec15`），
保持 checkpoint、Level 初始场景、episode 数、seed 和其他推理参数一致，
只改变执行长度。若另行修正垃圾桶初始位置，应对两种执行长度都使用同一份
修正版场景，避免将初始化变化混入执行长度的成功率对比。

默认行为是：

```text
Level 0: 10 个 episode
Level 1: 10 个 episode
Level 2: 10 个 episode
总计:    30 个 episode
```

三个等级的数据来自：

```text
/simple-eval/<TASK>/dr-level-0/
/simple-eval/<TASK>/dr-level-1/
/simple-eval/<TASK>/dr-level-2/
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

`--task NAME`：SIMPLE 任务名，例如 `G1WholebodyCloseDoorTeleop-v0` 或
`G1WholebodyBendPickMP-v0`，也可带 `simple/` 前缀。任务选择相应的机器人
和评估分支；没有单独的 MP／Sonic 控制器开关。

`--checkpoint PATH`：要评估的单个 checkpoint 目录，必须同时包含
`config.json` 和 `training_state.pt`。

### 等级、回合和 GPU

`--levels LIST`：要评估的等级，默认 `0,1,2`。例如 `--levels 0` 只运行
Level 0。

`--episodes N`：每个等级运行的 episode 数，默认 `10`。官方结果使用 10；
`--episodes 1` 仅用于 smoke test。

`--gpus LIST`：物理 GPU 编号，例如 `--gpus 4,5,6`。Level 0、1、2 会依次
分配到 GPU 4、5、6；如果 GPU 数量少于等级数量，剩余等级会等待上一波完成。
启动前检查这些 GPU 的已有任务与显存余量。MP 还需要 GPU 0 承载 AMO；
`mujoco_isaac` 的上下文占用限制见第 2.2 节。

`--seeds LIST`：运行时 reset 使用的 seed。固定官方环境由
`episodes.jsonl` 决定，通常保持 `--seeds 0` 即可。

### 环境和数据

`--simple-data-dir PATH`：SIMPLE 资源根目录，必须包含机器人、场景、材质和
assets。通过 `SIMPLE_DATA_DIR` 设置该目录。

`--eval-data-root PATH`：Level 评估数据根目录，默认是
`$SIMPLE_DATA_DIR/simple-eval`。只有评估数据不在该位置时才需要显式指定。

`--python PATH`：Python 可执行文件，默认 `SIMPLE/.venv/bin/python`，也可以
通过环境变量 `KIMODO_PYTHON` 指定。

### 模型推理和仿真

`--dtype bf16|fp32`：模型推理精度。当前启动脚本默认 `fp32`，本文命令也
显式传入 `fp32`，MP 和 Sonic 保持一致。

`--diffusion-steps N`：扩散采样步数，默认 `10`。

`--execution-frames N`：每次重新规划执行的模型帧数。`0` 表示执行完整
action chunk；当前 checkpoint 仍预测 50 帧，本文件的对照命令使用 `15`，
只执行前 15 帧后重新规划。该参数不改变模型的预测长度；使用 `50` 则恢复
完整执行 50 帧的基线配置。

`--rtc 0|1`：是否启用实时 chunking。本项目固定使用 `0`（关闭）。

`--rtc-overlap-frames N`、`--rtc-frozen-frames N`、`--rtc-ramp-power X`：
RTC 的重叠帧数、冻结前缀帧数和 ramp 指数。本项目固定传入 `0`、`0`、`1.0`；
RTC 关闭时前两项不会生效。

`--max-navigation-speed X`：导航速度保护阈值，单位 m/s。本项目评估命令统一显式传入
`10.0`；脚本未显式传参时的默认值仍为 `1.5`。每个模型 chunk 编码后会检查平面
导航速度；超过阈值时该 episode 立即报错并保留
`pipeline.log`，避免继续运行明显失真的 root 边界。日志中的
`[chunk_diagnostics]` 还会记录 state 段帧数、首帧/最大导航速度、根高度范围
和关节边界变化。

`--sim-mode mujoco|mujoco_isaac`：仿真模式。官方视觉评估使用
`mujoco_isaac`，默认也是该模式；`mujoco` 只适合轻量诊断。

`--max-steps N|auto`：每个 episode 的 TimeLimit。默认 `auto`，使用 SIMPLE
任务自身 `metadata["max_episode_steps"]`（例如 XMovePick 是 800，CloseDoor
任务当前定义为 450，BendPickMP 为 400）。只有确实需要覆盖官方任务设置时
才传入整数。

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

以下是 Sonic 的启动前检查示例；MP 可在第 2.2 节命令末尾追加 `--dry-run`。
检查通过只表示启动参数和路径有效，不代表完成了仿真或任务验证。

```bash
bash evaluation/simple_eval_signal_task_sonic.sh \
  --task G1WholebodyCloseDoorTeleop-v0 \
  --checkpoint log/experiments/simple_single_gbs64_20w_controlnet4_detach_true_mse_G1WholebodyCloseDoorTeleop-v0/2026-08-28_23-55-15/checkpoint_100000 \
  --gpus 4,5,6 \
  --episodes 10 \
  --simple-data-dir "$SIMPLE_DATA_DIR" \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 15 \
  --rtc 0 \
  --rtc-overlap-frames 0 \
  --rtc-frozen-frames 0 \
  --rtc-ramp-power 1.0 \
  --max-navigation-speed 10.0 \
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
平面速度超过 `--max-navigation-speed`（本项目评估使用 10.0 m/s），推理会 fail-fast，
并将完整错误保留在对应 Level 的 `pipeline.log` 中。这样可以在正式跑 30 个
episode 前发现训练/推理时序或坐标系再次失配。

手部还会记录以下诊断字段：

```text
model_hand_active=[left,right]/50
control_hand_active=[left,right]/25
continuous_hand_queue=0
measured_hand_closure=[left,right]
```

checkpoint 的模型以 30 Hz 预测 50 帧，设置 `--execution-frames 15` 后，
身体和连续手部均只执行当前预测的前 15 个模型帧，重采样得到 25 个 50 Hz
控制帧，约每 0.5 秒仿真时间重新规划一次。RTC 关闭时，剩余 35 帧不会直接
执行，而是由下一次基于最新观测的预测替代。`model_hand_active` 的分母仍为
完整预测的 `50`，`control_hand_active` 的分母则为实际执行的 `25`。
作为对照，完整执行 50 个模型帧通常得到约 83 个控制帧，约 1.66 秒后重新规划。
连续手输出按 `[0,1]` 的闭合度映射到手指关节目标；`measured_hand_closure`
记录仿真实测闭合度。适配层还显式处理了
MuJoCo（thumb-middle-index）与 WBC（thumb-index-middle）的手指顺序，这些逻辑
只在 `evaluation/simple_server.py` 生效，不改变 Arena 原始运行时。

### MP 执行路径的检查要点

MP 的每次推理返回当前待执行片段；评估器将片段内所有控制帧加入队列，
逐帧执行，队列为空后才重新预测。它不会每个仿真步都重新预测并仅取
`predicted[0]`。每回合首次请求的 `state_segment_frames=1` 正常；对于
本文 15 模型帧／25 控制帧配置，后续请求通常应为 `state_segment_frames=25`。
连续手 checkpoint 同时使用测量的身体状态和手部闭合量历史。

MP 先将 Kimodo 的 40 维输出转换成 SIMPLE 的 36 维动作，再按官方 MP
agent 的布局生成 `eval_move_actuators`：AMO 高度指令为目标绝对高度减
`0.75`，腰部 yaw/pitch/roll、导航和手臂／手部目标分别按对应字段传入。
这些 MP 适配不会替换 Sonic 的 WBC 执行分支。

MP 日志应包含 `MP AMO runtime device override: cuda -> cuda:0` 和
`Applying StandStabilizationWrapper: running 60 stand steps`。确认 chunk
与预热生效后，还需要检查外部视角视频和回合结果；无报错本身不等于站稳
或任务成功。

## 7. 已完成的 MP 评估记录

2026-09-14 使用第 2.2 节对应的 BendPickMP 权重和推理配置，完成官方
Level 0/1/2 共 30 回合：Level 0 为 5/10，Level 1 为 4/10，Level 2 为
4/10，总计 13/30（43.33%）；其余 17 回合均为 400 步 `timeout`。
这是该 checkpoint 的一次评估记录，不代表所有 MP 任务或 checkpoint。

结果目录（只用于查看，不作为新运行的输出目录）：

```text
./eval_results/simple_table1_15/G1WholebodyBendPickMP-v0
```

其中 `launch_manifest.json`、`launch.sh` 和 `run_config.txt` 保存了本次启动
参数，`simple_server_used.py` 是运行时评估器快照，`summary.json` 为等级汇总，
每个 `level_N/episode_XXXXXX/` 保存 ego 和外部视角视频。本次启动器退出码
为 0，顶层 `.done` 存在，30 份回合记录和 150 个非空视频文件已核对。
