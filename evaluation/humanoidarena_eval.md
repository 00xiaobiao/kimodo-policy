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
