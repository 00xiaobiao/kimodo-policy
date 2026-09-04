# 环境
/root/miniconda3/envs/kimodo/bin/python
/ai/Yichi/0_Systems/miniconda3/envs/unitree_sim_env/bin/python

# SONIC 单任务评估
```bash
bash ./evaluation/humanoidarena_eval_signal_task_sonic.sh \
      --project . \
      --task doubledesk \
      --checkpoint /path/to/checkpoint \
      --gpus 5,6,7 \
      --dtype fp32 \
      --diffusion-steps 10 \
      --execution-frames 15 \
      --rtc 0 \
      --rtc-overlap-frames 0 \
      --rtc-frozen-frames 0 \
      --rtc-ramp-power 1.0 \
      --persistent-sim 0 \
      --deterministic-eval 1 \
      --record-video-every-n 1 \
      --port-base 18080 \
      --results-dir ./eval_results/doubledesk_deterministic
```



# SONIC 多任务连续评估

新脚本支持通过 `--task` 指定一个或多个任务，并严格按照输入顺序连续评估。

任务组展开规则：

```text
hoi -> doubledesk -> football -> pp_box
hsi -> boxing -> open_door -> sit_sofa -> vision_navi
all -> doubledesk -> football -> pp_box -> boxing -> open_door -> sit_sofa -> vision_navi
```

例如 `--task doubledesk football pp_box` 会按照 doubledesk、football、pp_box 的顺序执行；也支持 `--task double desk football` 和 `--task ppbox` 这种写法。

当使用 `--rtc 0` 时，RTC 不会参与预测，`--rtc-overlap-frames`、
`--rtc-frozen-frames` 和 `--rtc-ramp-power` 不会影响轨迹融合；它们仍会进行参数校验并记录到运行配置中。

```bash
bash ./evaluation/humanoidarena_eval_multi_task_sonic.sh \
      --task doubledesk football pp_box \
      --project . \
      --checkpoint /path/to/checkpoint \
      --gpus 5,6,7 \
      --dtype fp32 \
      --diffusion-steps 10 \
      --execution-frames 15 \
      --rtc 0 \
      --rtc-overlap-frames 0 \
      --rtc-frozen-frames 0 \
      --rtc-ramp-power 1.0 \
      --persistent-sim 0 \
      --deterministic-eval 1 \
      --record-video-every-n 1 \
      --port-base 18080 \
      --results-dir ./eval_results/hoi_tasks_deterministic
```

# TWIST2 football 正式统计评估

下面的命令按照 football 测试配置的默认规模运行 3 个 seed、每个 seed 20
次，共 60 个 episode。命令不显式传入 `--max-steps`；wrapper 会根据所选
TWIST2 任务的官方 runner 自动读取对应的 `MAX_STEPS`（football 为 2000，
其它任务同理）。`--anchor-root 1` 是 TWIST2 root window 对齐修正；
`--persistent-sim 0 --deterministic-eval 1` 保证 episode 之间隔离并保持可复现。
评估前建议在 RTX 服务器上使用 `tmux`，避免 SSH 断开导致评估中止。

```bash
tmux new -s football_twist2_formal

cd /ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2
export CONDA_BASE=/ai/Yichi/0_Systems/miniconda3
export KIMODO_SERVER_PYTHON="$CONDA_BASE/envs/lerobot/bin/python"
export VLA_MAX_ROOT_DELTA_DEG=26.0

bash evaluation/humanoidarena_eval_signal_task_twist2.sh \
  --project /ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2 \
  --task football \
  --checkpoint /ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2/log/experiments/scratch_sonic_8/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_football_twist2/2026-08-28_16-23-14/checkpoint_200000 \
  --gpus 5,6,7 \
  --seeds 0,1,2 \
  --repeats 20 \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 15 \
  --rtc 0 \
  --persistent-sim 0 \
  --deterministic-eval 1 \
  --anchor-root 1 \
  --record-video-every-n 1 \
  --port-base 19420 \
  --server-port-max 20000 \
  --trace 0 \
  --results-dir /ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2/eval_results/football_twist2_checkpoint200000_formal
```

完成后可用以下命令查看总体结果：

```bash
jq '.overall_success_rate, .failure_reason_counts' \
  /ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2/eval_results/football_twist2_checkpoint200000_formal/summary.json
```

结果目录中的 `summary.json`、`final.csv` 和 `episodes/` 保存统计与逐回合结果，
视频位于 `videos/success/` 和 `videos/failure/`。`--record-video-every-n 1`
会保存全部 60 个 episode 的视频；磁盘空间有限时可改为 `5`。

说明：wrapper 默认 `video-fps=30`、`num-workers=1`，因此正式命令无需重复指定。
`--trace 0` 需要保留，因为 wrapper 的默认值是 `1`；它仅用于关闭 Kimodo server
调试日志，不会关闭模型推理或视频录制。

TWIST2 的 checkpoint 是 Kimodo 目录（必须包含 `config.json` 和
`training_state.pt`），不能传给 LeRobot 的 `pretrained_model` server。仿真使用
`unitree_sim_env`，模型 server 通过 `KIMODO_SERVER_PYTHON` 指定（若没有独立
`kimodo` 环境，当前部署可使用 `lerobot/bin/python` 运行同一个
`humanoidarena_server.py`）。`MODEL_PATHS_CSV` 用于传入一个或多个直接 checkpoint 目录。

```bash
conda activate unitree_sim_env
export CONDA_BASE=/ai/Yichi/0_Systems/miniconda3
# 当前远端部署使用 lerobot 环境；若存在独立 kimodo 环境可替换此路径。
export KIMODO_SERVER_PYTHON="$CONDA_BASE/envs/lerobot/bin/python"
export MODEL_PATHS_CSV=/path/to/kimodo/checkpoint_200000
export ENV_CONFIG_YAML=tasks/common_test_config/base_test/doubledesk_twist2_test.yaml
export SERVER_GPU_IDS=5,6,7
export NUM_WORKERS=1
export SEEDS_OVERRIDE="0 1 2"
export REPEATS_PER_SEED=20

bash ./HumanoidArena/isaaclab_twist2_g1/script/eval_scripts/twist2/run_vla_eval_parallel.sh
```

TWIST2 server 的默认推理时序为 50 Hz 控制、15 个模型执行帧、10 个扩散步，
RTC 默认关闭；这些参数可通过 `KIMODO_*` 环境变量覆盖。录像保存在脚本目录的
`eval_results/` 下，每个 episode 是连续控制帧。
