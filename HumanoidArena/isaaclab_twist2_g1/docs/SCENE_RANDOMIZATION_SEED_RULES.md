# Scene Randomization Seed Rules

这份说明约束 `isaaclab_twist2_g1` 中所有带随机场景初始化的任务，避免再次出现“录制时和 replay 时场景不是同一套随机链路”的错误。

## 核心原则

场景级随机化只能有一颗主 seed。

- 每个 episode 只允许生成一颗 `episode_object_seed`
- 所有环境对象随机化都必须从这颗主 seed 派生
- 录制文件只记录这颗主 seed，不再额外依赖 task-local 的隐藏 seed
- replay 必须优先用录制文件中的这颗主 seed 重建场景

## 允许的 seed 来源

`episode_object_seed` 可以有两种来源：

- `object_reset_seed_source: time`
  适合真人录制采数。每个 episode 都拿到新的场景随机化。
- `object_reset_seed_source: env_seed`
  适合评测和可复现实验。同一个 `--seed` 下，冷启动和 reset 序列都可复现。

这两种只是同一类主 seed 的不同来源，不是两条独立随机链路。

## 禁止事项

禁止为某个局部子系统单独维护第二颗“layout seed”或“scene seed”，例如：

- obstacle layout 单独一颗 seed
- target object 单独一颗 seed
- replay restore 时重新发明一颗 task-local seed

如果某个 task 需要额外的布局步骤，也必须吃同一颗 `episode_object_seed`。

## 任务实现要求

带随机场景初始化的 task 必须满足：

1. live reset 时先生成或设置当前 `episode_object_seed`
2. 该 task 的所有随机对象都从这颗 seed 派生
3. 录制时把 `episode_object_seed` 和 `episode_init_env` 一起写入 NPZ
4. replay restore 时优先按这颗 seed 重建 task-specific 场景初始化
5. 如果 task 还有额外的物理同步步骤，也必须在 replay restore 中重复执行

## SmallWarehouse 教训

`SmallWarehouse VisionNavigation` 曾经出现的问题是：

- `target_sign` 使用通用 deterministic reset seed 链
- obstacle layout 使用另一条 task-local seed 链
- replay 只知道 `episode_object_seed`，不知道 obstacle layout 的那颗 seed

结果是：

- 录制时场景初始化和 replay 时场景初始化不一致
- NPZ 虽然有对象状态，但 replay 仍然可能恢复出错误的 obstacle 物理布局

因此，后续任何新任务都必须遵守“单主 seed”原则。
