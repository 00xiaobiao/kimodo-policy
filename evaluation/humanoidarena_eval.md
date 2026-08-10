# 环境
/root/miniconda3/envs/kimodo/bin/python
/ai/Yichi/0_Systems/miniconda3/envs/unitree_sim_env/bin/python

# SONIC 评估

bash ./evaluation/humanoidarena_eval_sonic.sh \
      --project . \
      --task doubledesk \
      --checkpoint /path/to/checkpoint \
      --gpus 5,6,7 \
      --dtype fp32 \
      --diffusion-steps 10 \
      --execution-frames 15 \
      --rtc 0 \
      --rtc-overlap-frames 12 \
      --rtc-frozen-frames 1 \
      --rtc-ramp-power 1.0 \
      --persistent-sim 0 \
      --deterministic-eval 1 \
      --record-video-every-n 1 \
      --port-base 18080 \
      --results-dir ./eval_results/doubledesk_deterministic

# TWIST2 评估

bash ./evaluation/humanoidarena_eval_twist2.sh \
      --project . \
      --task doubledesk \
      --checkpoint /path/to/checkpoint \
      --gpus 5,6,7 \
      --dtype fp32 \
      --diffusion-steps 10 \
      --execution-frames 15 \
      --rtc 0 \
      --rtc-overlap-frames 12 \
      --rtc-frozen-frames 1 \
      --rtc-ramp-power 1.0 \
      --persistent-sim 0 \
      --deterministic-eval 1 \
      --record-video-every-n 1 \
      --workers-per-gpu 1 \
      --max-root-delta-deg 26 \
      --port-base 19080 \
      --results-dir ./eval_results/doubledesk_twist2_deterministic
