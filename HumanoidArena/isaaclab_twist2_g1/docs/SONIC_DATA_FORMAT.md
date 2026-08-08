# SONIC NPZ Recording Format

This document describes the raw `.npz` files recorded by the SONIC teleoperation pipeline in `isaaclab_twist2_g1`.

Implementation reference:

- `isaaclab_twist2_g1/action_provider/action_provider_sonic.py`

## File Name

Default file names follow this pattern:

```text
<task_name>_sonic_<timestamp_us>.npz
```

## Schema

Current schema versions:

```text
sonic_episode_v3
sonic_episode_v4_multicam
```

`sonic_episode_v3` is the standard raw SONIC episode schema. `sonic_episode_v4_multicam` is used when additional camera streams are stored with the episode.

## Top-Level Fields

### Metadata

- `schema_version`
- `task`
- `episode_id`
- `save_timestamp_us`
- `num_frames`
- `meta_control_dt`
- `meta_physics_dt`
- `meta_decimation`
- `meta_pose_source`
- `meta_encoder_path`
- `meta_decoder_path`
- `episode_object_seed`
- `episode_object_seed_source`

### Frame Markers

- `frame_index`
- `raw_frame_index`
- `consumed_frame_index`
- `episode_step`
- `timestamp_wall`
- `timestamp_monotonic`
- `timestamp_realtime`
- `raw_timestamp_monotonic`
- `raw_timestamp_realtime`
- `consumed_timestamp_monotonic`
- `consumed_timestamp_realtime`
- `consumed_new_this_step`
- `consumed_control_step`
- `executed_source_frame_index`
- `executed_source_timestamp_realtime`
- `executed_source_timestamp_monotonic`
- `executed_source_control_step`
- `recording_command`
- `reset_requested`
- `reset_completed`
- `save_triggered`

### Human Input and Processed Pose

- `human_left_hand`: `(T, 7)`
- `human_right_hand`: `(T, 7)`
- `human_raw_body_quat_w`: `(T, 4)`
- `human_raw_body_pos`: `(T, 3)`
- `human_smpl_joints`: `(T, 24, 3)`
- `human_smpl_pose`: `(T, 21, 3)`
- `human_body_quat_w`: `(T, 4)`
- `human_body_quat_w_aligned`: `(T, 4)`
- `human_body_pos`: `(T, 3)`
- `human_joint_pos`: `(T, 29)`
- `consumed_anchor_rot6d`: `(T, 6)`
- `human_vr_position`
- `human_vr_orientation`
- `human_heading_increment`

Controller binary fields:

- `pico_left_grip_binary`
- `pico_right_grip_binary`
- `pico_left_close_trigger_binary`
- `pico_right_close_trigger_binary`
- `pico_left_open_trigger_binary`
- `pico_right_open_trigger_binary`

### Anchor / Heading Alignment

- `anchor_heading_initialized`
- `anchor_use_heading_align`
- `anchor_init_base_quat_wxyz`
- `anchor_init_ref_quat_wxyz`
- `anchor_heading_align_quat_wxyz`

These fields are important for replay because SONIC aligns human reference heading to the robot heading.

### SONIC Model I/O

Encoder-side fields:

- `encoder_input`: `(T, 1762)`
- `encoder_smpl_joint_window`: `(T, 10, 24, 3)`
- `encoder_anchor_window`: `(T, 10, 6)`
- `encoder_wrist_window`
- `encoder_motion_joint_pos_hist`
- `encoder_motion_joint_vel_hist`
- `encoder_motion_root_z_hist`
- `encoder_motion_anchor_rot6d_hist`
- `encoder_robot_joint_pos_hist`
- `encoder_robot_joint_vel_hist`
- `encoder_latent`: `(T, 64)`

`encoder_latent` is the SONIC encoder output for each recorded frame. It can be used to train latent-output VLA policies for the live `SONIC_VLA_ACTION_FORMAT=latent64` inference interface.

Decoder-side fields:

- `decoder_obs`: `(T, 994)`
- `decoder_ang_vel_hist`
- `decoder_gravity_dir_hist`
- `decoder_last_action_hist`
- `decoder_raw_action`: `(T, 29)`
- `decoder_target_action`: `(T, 29)`

### Robot State

- `robot_qpos_before_decimation`: `(T, 29)`
- `robot_qvel_before_decimation`: `(T, 29)`
- `robot_root_position`: `(T, 3)`
- `robot_root_orientation`: `(T, 4)` in `wxyz` order
- `robot_root_lin_vel_local`
- `robot_root_ang_vel_local`
- `robot_root_lin_vel_world`
- `robot_root_ang_vel_world`

### Executed Actions

- `final_body_action_29dof`: `(T, 29)`
- `final_body_action_29dof_pre_delay`: `(T, 29)`
- `final_full_action`
- `body_effort_target`
- `hand_action_left`: `(T, 7)`
- `hand_action_right`: `(T, 7)`

### VLA Conversion Fields

These fields are used by the NPZ-to-LeRobot conversion path:

- `vla_state`: `(T, 64)`
- `vla_state_root_rot6d`: `(T, 6)`
- `vla_state_dof_pos_29`: `(T, 29)`
- `vla_state_dof_vel_29`: `(T, 29)`
- `vla_action`: `(T, 40)`
- `vla_action_raw`: `(T, 40)`
- `vla_action_executed`: `(T, 40)`
- `vla_action_executed_raw`: `(T, 40)`
- `vla_action_root_xy_delta`: `(T, 2)`
- `vla_action_root_z`: `(T, 1)`
- `vla_action_root_rot6d`: `(T, 6)`
- `vla_action_joint_pos_29`: `(T, 29)`
- `vla_action_hand_binary`: `(T, 2)`
- `vla_action_hand_binary_2`: `(T, 2)`
- `vla_action_semantics`
- `vla_action_heading_aligned`

### Raw JSON Payloads

These fields are JSON strings packed into scalar NumPy arrays:

- `human_raw_smplx_json`
- `human_controller_json`
- `human_recording_control_json`

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

- `vision_world_frame_indices`
- `vision_world_rgb_video_path`
- `vision_world_depth`
- `vision_left_wrist_frame_indices`
- `vision_left_wrist_rgb_video_path`
- `vision_left_wrist_depth`
- `vision_right_wrist_frame_indices`
- `vision_right_wrist_rgb_video_path`
- `vision_right_wrist_depth`

## Replay Dependencies

Direct replay mainly depends on:

- `final_body_action_29dof`
- `hand_action_left`
- `hand_action_right`
- recorded environment seed / initial object fields when scene randomization is enabled

Inference replay additionally depends on recorded model inputs and histories:

- `encoder_input`
- `decoder_obs`
- encoder history fields
- decoder history fields
- human pose and anchor alignment fields

## Read Example

```python
import numpy as np

with np.load("episode.npz", allow_pickle=True) as data:
    print(data["schema_version"].item())
    print(data["task"].item())
    print(data["encoder_input"].shape)
    print(data["decoder_target_action"].shape)
```
