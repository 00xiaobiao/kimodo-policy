# Common Env Config YAML

`sim_main.py` 支持通过 `--env_config_yaml` 在 `gym.make(...)` 之前覆盖环境配置。

最简单写法：

```yaml
task_name: Isaac-Move-Football-Single-G129-Dex3-Wholebody

overrides:
  sim:
    dt: 0.005
  decimation: 4
```

也支持按路由或任务附加覆盖：

```yaml
task_name: Isaac-Move-Football-Single-G129-Dex3-Wholebody
backend: sonic

overrides:
  sim:
    dt: 0.005
  decimation: 4

routes:
  sonic:
    decimation: 8

tasks:
  Isaac-Move-Football-Single-G129-Dex3-Wholebody:
    episode_length_s: 30.0
```

合并顺序：
`overrides` -> `routes.<gmt_backend|action_source>` -> `tasks.<task_name>` -> `route_tasks.<route>.<task>`

脚本约定：
- `run_sonic.sh` / `run_twist2.sh` 会从 yaml 顶层 `task_name` 自动读取 Isaac task id
- 因此切换任务时，只需要改 `ENV_CONFIG_YAML`

当前会自动同步的派生字段：
- `scene.contact_forces.update_period <- sim.dt`
- `sim.render_interval <- decimation`

推荐做法：
- 通用默认继续放在 `twist2_default.yaml` / `sonic_default.yaml`
- 具体任务单独建文件，例如 `football_single_twist2.yaml` / `football_single_sonic.yaml` / `football_single_vla.yaml`
- 每个任务 yaml 顶层都带上 `task_name`
- run 脚本直接指向对应任务 YAML，避免再单独维护 `TASK_NAME`

如果任务里有需要 deterministic reset / recording / replay restore 的环境对象，建议在任务 YAML 中直接配置：

```yaml
overrides:
  sim:
    dt: 0.001
  decimation: 10
  object_reset_seed_source: time
  deterministic_object_resets:
    - record_name: football
      scene_keys: [object, football]
      pose_range:
        x: [-0.05, 0.05]
        y: [0.0, 0.05]
      zero_velocity_on_reset: true
```

种子来源建议：
- `object_reset_seed_source: time`
  适合真人录制采数。每次 episode 会拿到新的局部伪随机对象初始状态，但不会污染全局 RNG。
- `object_reset_seed_source: env_seed`
  适合 VLA / policy 评测。同一个 `--seed` 下，每次冷启动和每一轮 reset 的对象出生点序列都一致，方便不同模型横向对比。

场景随机化原则：
- 一个 task 的场景随机化只能有一颗主 seed，也就是 `episode_object_seed`
- `target_sign`、`obstacle layout`、task-specific scene init 都必须从这颗主 seed 派生
- 不要再为某个局部子系统单独维护第二颗 layout seed
- replay 必须优先使用录制文件中的 `episode_object_seed` 重建场景

详细约束见：
- [docs/SCENE_RANDOMIZATION_SEED_RULES.md](../../docs/SCENE_RANDOMIZATION_SEED_RULES.md)
