  ### 1. 启动相机 server

  如果相机已经由 systemd 启动，先确认：

  sudo systemctl status composed_camera_server.service

  如果没有启动，在机器人电脑上执行：

  cd /path/to/GR00T-WholeBodyControl
  source .venv_camera/bin/activate

  python -m gear_sonic.camera.composed_camera \
      --ego-view-camera oak \
      --ego-view-device-id <EGO_CAMERA_MXID> \
      --port 5555 \
      --fps 30

  如果你已经配置好了相机，也可以不填写 --ego-view-device-id。

  ### 2. 启动 SONIC C++ 低层控制

  在工作站打开第二个终端：

  cd /data/local-data/data/code/yunhengwang/kimodo-polocy/controlnet_v1.2/real-world/
  GR00T-WholeBodyControl/gear_sonic_deploy

  ./deploy.sh \
      --cp policy/release/model \
      --obs-config policy/release/observation_config.yaml \
      --planner planner/target_vel/V2/planner_sonic.onnx \
      --motion-data reference/example \
      --input-type zmq_manager \
      --output-type zmq \
      --zmq-host localhost \
      real

  其中：

  - policy/release/model_decoder.onnx
  - policy/release/model_encoder.onnx
  - planner/target_vel/V2/planner_sonic.onnx

  必须是真实存在的 SONIC 模型文件。如果你的模型路径不同，需要替换。

  确认 C++ 输出：

  - g1_debug：5557
  - 接收 bridge action：5556

  这一步会出现真实机器人启动确认，请确保机器人处于安全状态，并准备好急停。

  ### 3. 启动 Kimodo 真机 server

  在第三个终端执行：

  cd /data/local-data/data/code/yunhengwang/kimodo-polocy/controlnet_v1.2

  /home/CONNECT/yfang870/miniconda3/envs/kimodo/bin/python \
      real-world/deploy/real_world_server.py \
      --checkpoint /path/to/your/kimodo/checkpoint \
      --text-embedding-cache data/cache/HumanoidArena \
      --host 127.0.0.1 \
      --port 18080 \
      --device cuda:0 \
      --control-fps 50

  把：

  /path/to/your/kimodo/checkpoint

  替换为实际 Kimodo checkpoint 目录。

  然后检查：

  curl http://127.0.0.1:18080/health

  应该返回类似：

  {
    "status": "ok",
    "mode": "real_world",
    "control_fps": 50.0,
    "state_dim": 64,
    "action_dim": 40
  }

  ### 4. 先做 bridge dry-run

  第四个终端执行：

  cd /data/local-data/data/code/yunhengwang/kimodo-polocy/controlnet_v1.2

  /home/CONNECT/yfang870/miniconda3/envs/kimodo/bin/python \
      real-world/deploy/run_kimodo_sonic.py \
      --server-url http://127.0.0.1:18080 \
      --task "place the bottle in the box" \
      --dry-run \
      --max-steps 20 \
      --verbose

  这一步只使用假的 state 和 image，不连接机器人，也不会发 action。

  ### 5. 连接真实 state 和 camera，但禁止发布 action

  确认 dry-run 正常后执行：

  /home/CONNECT/yfang870/miniconda3/envs/kimodo/bin/python \
      real-world/deploy/run_kimodo_sonic.py \
      --server-url http://127.0.0.1:18080 \
      --task "place the bottle in the box" \
      --state-zmq-host 127.0.0.1 \
      --state-zmq-port 5557 \
      --camera-host 192.168.123.164 \
      --camera-port 5555 \
      --action-zmq-host 127.0.0.1 \
      --action-zmq-port 5556 \
      --no-publish \
      --max-steps 300 \
      --verbose

  重点观察：

  - 是否持续收到真实 state；
  - 是否持续收到 camera；
  - 是否有 sensor skew；
  - 是否有 state gap；
  - 是否有 stale Kimodo result；
  - 是否有 action.root_z 或 joint jump 错误。

  --no-publish 表示不会向 SONIC 发送 pose action。

  ### 6. 发布 action，但暂时不自动启动机器人

  如果上一步正常，去掉 --no-publish，但不要加 --auto-start：

  /home/CONNECT/yfang870/miniconda3/envs/kimodo/bin/python \
      real-world/deploy/run_kimodo_sonic.py \
      --server-url http://127.0.0.1:18080 \
      --task "place the bottle in the box" \
      --state-zmq-host 127.0.0.1 \
      --state-zmq-port 5557 \
      --camera-host 192.168.123.164 \
      --camera-port 5555 \
      --action-zmq-host 127.0.0.1 \
      --action-zmq-port 5556 \
      --max-steps 300 \
      --verbose

  此时 bridge 会发布 Protocol v1 pose packet，但不会主动发送 SONIC start command。

  ### 7. 最后才启动真实执行

  确认前面所有步骤都正常后，再执行同样的命令并加入：

  --auto-start

  完整形式：

  /home/CONNECT/yfang870/miniconda3/envs/kimodo/bin/python \
      real-world/deploy/run_kimodo_sonic.py \
      --server-url http://127.0.0.1:18080 \
      --task "place the bottle in the box" \
      --state-zmq-host 127.0.0.1 \
      --state-zmq-port 5557 \
      --camera-host 192.168.123.164 \
      --camera-port 5555 \
      --action-zmq-host 127.0.0.1 \
      --action-zmq-port 5556 \
      --auto-start \
      --verbose

  --auto-start 只会在收到第一帧有效 Kimodo action 并成功发布 pose 后发送 SONIC
  start。

  停止时直接按：

  Ctrl-C