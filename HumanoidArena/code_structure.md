<div align="center">
  <h1>代码结构分析</h1>
  <p>isaaclab_twist2_g1 项目各模块逻辑说明</p>
  <p><i>Taowen Wang</i></p>
</div>

---

## 目录

- [整体数据流](#整体数据流)
- [关键文件索引](#关键文件索引)
- [IsaacLab 仿真环境](#isaaclab-仿真环境)
- [PICO 数据读取与转换](#pico-数据读取与转换)
- [图像串流逻辑](#图像串流逻辑)
- [Action Provider](#action-provider)
- [数据采集](#数据采集)

---

## 整体数据流

```
PICO 头显
  │  姿态数据（SMPLX）
  ▼
xrobot_teleop_to_robot_w_hand.py
  │  运动重定向（GMR）→ 35维 mimic_obs
  │  发布到 Redis
  ▼
action_provider_wh_twist2.py
  │  从 Redis 读取 → 构建 1402维观测
  │  ONNX 推理（TWIST2 策略）→ 29维动作
  ▼
sim_main.py（IsaacLab）
  │  执行仿真步进，驱动 G1 机器人
  ▼
image_server.py
  │  H.264 串流（ffmpeg over TCP）
  ▼
PICO 头显（第一人称视角回传）
```

---

## 关键文件索引

| 模块           | 文件路径                                                                                      |
| -------------- | --------------------------------------------------------------------------------------------- |
| 仿真入口       | `sim_main.py`                                                                                 |
| 仿真启动脚本   | `run.sh`                                                                                      |
| 遥操作脚本     | `TWIST2/teleop.sh`                                                                            |
| 数据采集脚本   | `TWIST2/data_record.sh`                                                                       |
| PICO 数据转换  | `TWIST2/deploy_real/xrobot_teleop_to_robot_w_hand.py`                                        |
| Action Provider | `action_provider/action_provider_wh_twist2.py`                                               |
| Provider 工厂  | `action_provider/create_action_provider.py`                                                   |
| 图像串流       | `image_server/image_server.py`                                                                |
| 数据采集程序   | `TWIST2/deploy_real/server_data_record.py`                                                    |
| 任务环境配置   | `tasks/g1_tasks/move_cylinder_g1_29dof_dex3_wholebody/move_cylinder_g1_29dof_dex3_hw_env_cfg.py` |

---

## IsaacLab 仿真环境

**入口：** `sim_main.py`，通过 `--task` 参数指定任务，如：

```bash
python sim_main.py \
  --task Isaac-Move-Cylinder-G129-Dex3-Wholebody \
  --robot_type g129 \
  --device cuda
```

**物理参数：**

| 参数 | 值 |
| ---- | -- |
| 物理引擎 | PhysX |
| 仿真步长 dt | 0.001s（1ms） |
| 策略频率 | 100Hz（decimation=10） |

**如何修改仿真环境：**

编辑任务配置文件 `move_cylinder_g1_29dof_dex3_hw_env_cfg.py`，可修改：

- 机器人初始位置：`init_pos=(-3.9, -2.81811, 0.8)`
- 物理参数：`self.sim.dt`、`self.decimation`、摩擦系数
- 相机配置：`front_camera`、`world_camera`、`left_wrist_camera` 等

---

## PICO 数据读取与转换

**核心文件：** `TWIST2/deploy_real/xrobot_teleop_to_robot_w_hand.py`

### 数据读取

使用 `XRobotStreamer` 类，通过 `get_current_frame()` 获取实时帧：

```python
smplx_data, left_hand_data, right_hand_data, controller_data, headset_data = \
    self.teleop_data_streamer.get_current_frame()
```

### 运动重定向（GMR）

使用 `GeneralMotionRetargeting` 将人体 SMPLX 数据转换为机器人关节角度：

```python
GMR(src_human="xrobot", tgt_robot="unitree_g1", actual_human_height=1.79)
qpos = self.retarget.retarget(smplx_data, offset_to_ground=True)
```

### 构建 35 维 mimic_obs

```
[xy_vel(2), z_pos(1), roll_pitch(2), yaw_vel(1), joints(29)] = 35D
```

### 发布到 Redis

| Redis Key                              | 维度 | 内容           |
| -------------------------------------- | ---- | -------------- |
| `action_body_unitree_g1_with_hands`    | 35   | mimic_obs      |
| `action_hand_left_unitree_g1_with_hands`  | 7  | 左手关节角度   |
| `action_hand_right_unitree_g1_with_hands` | 7  | 右手关节角度   |
| `action_neck_unitree_g1_with_hands`    | 2    | 颈部 yaw/pitch |
| `t_action`                             | -    | 时间戳         |

### 手部控制

通过手柄按钮插值控制手部开合：`index_trig` 闭合，`grip` 张开。

---

## 图像串流逻辑

**核心文件：** `image_server/image_server.py`

支持四种传输协议：

| 协议 | 类名 | 说明 |
| ---- | ---- | ---- |
| ZMQ（默认） | `_ZmqImagePublisher` | 端口 5555 |
| Redis | `_RedisImagePublisher` | 存储 JPEG + 元数据 |
| DDS | `_DdsImagePublisher` | Unitree SDK2 |
| XRobot（H.264） | `_XRobotImagePublisher` | ffmpeg over TCP，回传 PICO |

**XRobot 串流参数（run.sh 中配置）：**

```bash
--image_transport xrobot
--image_xrobot_host 10.42.0.35   # PICO 头显 IP
--image_xrobot_port 12345
--image_xrobot_width 640
--image_xrobot_height 480
--image_xrobot_bitrate 4194304   # 4Mbps
--image_fps 30
```

图像来源：从共享内存（`MultiImageReader`）读取多相机图像，编码为 JPEG 后通过 ffmpeg 压缩为 H.264 发送。

---

## Action Provider

**核心文件：** `action_provider/action_provider_wh_twist2.py`

**使用的类：** `DDSRLActionProvider`

### 观测构建（1402 维）

```
action_mimic   (35D)  ← 从 Redis 读取
obs_proprio    (92D)  ← 从 Isaac 读取
  ├─ ang_vel   (3)  × 0.25
  ├─ roll_pitch (2)
  ├─ dof_pos_delta (29)
  ├─ dof_vel   (29) × 0.05
  └─ last_action (29)
obs_hist      (1270D) ← 历史 10 帧（127×10）
future_obs     (35D)  ← 当前 mimic
─────────────────────
总计：1402D
```

### 策略推理

```python
obs = self.compute_observations()  # [1, 1402]
action = self.policy(obs)          # ONNX 推理，输出 [1, 29]
```

### 动作转换

1. 策略输出 29 维（MuJoCo actuator 顺序）
2. 映射到 Isaac 关节索引：`self.twist2_action_indices`
3. 应用 `action_scale=0.5`：`target = raw_action × 0.5 + default_pos`
4. 填充完整动作向量（含手部、颈部）

### 仿真步进

```python
for _ in range(self._twist2_decimation):  # decimation=10
    self.env.scene["robot"].set_joint_position_target(full_action)
    self.env.sim.step(render=False)
```

---

## 数据采集

**核心文件：** `TWIST2/deploy_real/server_data_record.py`
**启动脚本：** `TWIST2/data_record.sh`

### 采集内容

**视觉数据：**
- 从机器人 ZMQ 端口 5555 接收图像
- 双目图像，分辨率 640×2 × 360 × 3

**状态数据（Redis）：**

| Redis Key                              | 维度 | 内容 |
| -------------------------------------- | ---- | ---- |
| `state_body_unitree_g1_with_hands`     | 34   | ang_vel(3) + roll_pitch(2) + dof_pos(29) |
| `state_hand_left_unitree_g1_with_hands`  | 7  | 左手关节角度 |
| `state_hand_right_unitree_g1_with_hands` | 7  | 右手关节角度 |
| `state_neck_unitree_g1_with_hands`     | 2    | 颈部角度 |

**动作数据（Redis）：** 与遥操作发布的 action 键一致（见上文）。

### 存储格式

使用 `EpisodeWriter` 类，每个 episode 独立存储，包含 RGB 图像、状态、动作、时间戳，采集频率 30Hz。

### 录制控制

| 手柄按钮 | 功能 |
| -------- | ---- |
| 左手 `key_two` | 开始 / 停止录制 |
| 左手 `axis_click` | 退出程序 |