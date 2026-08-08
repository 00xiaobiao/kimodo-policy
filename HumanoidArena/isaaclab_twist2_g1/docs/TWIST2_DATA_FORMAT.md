# TWIST2 NPZ Recording Format

This document describes the raw `.npz` files recorded by the TWIST2 teleoperation pipeline in `isaaclab_twist2_g1`.

Implementation reference:

- `isaaclab_twist2_g1/action_provider/action_provider_wh_twist2.py`

## File Name

Default file names follow this pattern:

```text
<task_name>_<timestamp_us>.npz
```

Example:

```text
Isaac-Move-Football-Single-G129-Dex3-Wholebody_1774962955107099.npz
```

## Schema

Current schema versions:

```text
twist2_episode_v2
twist2_episode_v3_multicam
```

`twist2_episode_v2` is the standard raw TWIST2 episode schema. `twist2_episode_v3_multicam` is used when additional wrist camera streams are stored with the episode.

## Top-Level Fields

### Metadata

- `schema_version`
- `task`
- `num_frames`
- `observation_semantics`
- `episode_object_seed`
- `episode_object_seed_source`

Optional rerecord summary fields may be present:

- `rerecord_final_reward`
- `rerecord_max_reward`
- `rerecord_any_success`

### Human Input

- `human_hand_left`: `(T, 7)`
- `human_hand_right`: `(T, 7)`
- `human_neck`: `(T, 2)`
- `human_smplx_data`: JSON string containing per-frame `smplx_data_before_gmr`
- `human_info_data`: JSON string containing per-frame `human_info`

Controller binary fields:

- `pico_left_grip_binary`
- `pico_right_grip_binary`
- `pico_left_close_trigger_binary`
- `pico_right_close_trigger_binary`
- `pico_left_open_trigger_binary`
- `pico_right_open_trigger_binary`

### Robot State and TWIST2 Policy Inputs

- `robot_qpos_before_decimation`: `(T, 29)`
- `robot_qvel_before_decimation`: `(T, 29)`
- `robot_root_position`: `(T, 3)`
- `robot_root_orientation`: `(T, 4)` in `wxyz` order
- `robot_root_lin_vel_local`: `(T, 3)`
- `robot_root_ang_vel_local`: `(T, 3)`
- `robot_root_lin_vel_world`: `(T, 3)`
- `robot_root_ang_vel_world`: `(T, 3)`
- `robot_twist2_inference_qpos`: `(T, 29)`
- `robot_action_mimic`: `(T, 35)`
- `robot_obs_buf`: `(T, 1432)`

`robot_action_mimic` is the native TWIST2 command consumed by the low-level backend. It is also the preferred source for reconstructing TWIST2 reference-pose LeRobot actions.

Optional dynamics fields:

- `robot_applied_torque_before_decimation`
- `robot_body_net_contact_forces`

### VLA Conversion Fields

The current recorder stores canonical VLA state/action fields for conversion and validation:

- `vla_state`: `(T, 64)`
- `vla_state_root_rot6d`: `(T, 6)`
- `vla_state_dof_pos_29`: `(T, 29)`
- `vla_state_dof_vel_29`: `(T, 29)`
- `vla_action`: `(T, 40)`
- `vla_action_root_xy_delta`: `(T, 2)`
- `vla_action_root_z`: `(T, 1)`
- `vla_action_root_rot6d`: `(T, 6)`
- `vla_action_joint_pos_29`: `(T, 29)`
- `vla_action_hand_binary`: `(T, 2)`
- `vla_action_hand_binary_2`: `(T, 2)`

### Environment State

Frame-wise object fields are written when objects exist:

```text
env_obj_<name>_position
env_obj_<name>_orientation
env_obj_<name>_linear_velocity
env_obj_<name>_angular_velocity
```

Initial episode object fields are written as:

```text
episode_init_env_obj_<name>_position
episode_init_env_obj_<name>_orientation
episode_init_env_obj_<name>_linear_velocity
episode_init_env_obj_<name>_angular_velocity
```

### Vision and Multicam Fields

The current writer stores camera frames through video-backed fields when available:

- `vision_storage_format = "video_v1"`
- `vision_frame_indices`
- `vision_rgb_video_path`
- `vision_rgb_video_fps`
- `vision_rgb_video_num_frames`
- `vision_depth`

Multicam episodes may also contain:

- `vision_left_wrist_frame_indices`
- `vision_left_wrist_rgb_video_path`
- `vision_left_wrist_depth`
- `vision_right_wrist_frame_indices`
- `vision_right_wrist_rgb_video_path`
- `vision_right_wrist_depth`

### System Fields

- `system_control_frequency`
- `system_decimation`
- `system_physics_dt`
- `system_timestamp`

## Replay Dependencies

Direct replay mainly depends on:

- `robot_twist2_inference_qpos`
- `robot_qpos_before_decimation`
- `human_hand_left`
- `human_hand_right`
- `human_neck`
- recorded environment seed / initial object fields when scene randomization is enabled

Inference replay depends on:

- `robot_obs_buf`
- `robot_twist2_inference_qpos`

TWIST2 inference replay reuses the recorded `robot_obs_buf` instead of reconstructing observations from scratch.

## Read Example

```python
import numpy as np

with np.load("episode.npz", allow_pickle=True) as data:
    print(data["schema_version"].item())
    print(data["task"].item())
    print(data["robot_twist2_inference_qpos"].shape)
    print(data["robot_obs_buf"].shape)
```
