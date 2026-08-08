# IsaacLab TWIST2 G1

`isaaclab_twist2_g1/` contains the Isaac Lab environments, teleoperation/replay entrypoints, rerecording utilities, and evaluation scripts used by HumanoidArena.

## Start Here

- [Environment setup](../docs/04_environment_setup.md)
- [Teleoperation and recording](../docs/01_teleoperation.md)
- [Data processing](../docs/02_data_pipeline.md)
- [Evaluation](../docs/03_evaluation.md)

## Main Directories

- `action_provider/`: TWIST2 and SONIC control, recording, and replay logic.
- `tasks/`: Isaac Lab task definitions and environment configuration.
- `pico_server/`: Pico bridge and Redis/ZMQ data publishing.
- `image_server/`: camera streaming helpers.
- `tools/data_tools/`: rerecording and LeRobot conversion tools.
- `script/eval_scripts/`: single-run and batch evaluation launch scripts.

## Common Entrypoints

- `run_twist2.sh`: launch TWIST2 teleoperation/recording.
- `run_sonic.sh`: launch SONIC teleoperation/recording.
- `run_replay_twist2.sh`: replay TWIST2 recordings.
- `run_replay_sonic.sh`: replay SONIC recordings.
- `run_rerecord.sh`: run rerecording jobs.
- `run_sonic_teleop_server.sh` / `run_twist2_teleop_server.sh`: teleoperation server launchers.

Most scripts keep their task, checkpoint, and path parameters near the top of the file. Update those values for your local environment before launching.

## Format And Protocol References

- [Scene randomization seed rules](docs/SCENE_RANDOMIZATION_SEED_RULES.md)
- [TWIST2 raw NPZ format](docs/TWIST2_DATA_FORMAT.md)
- [SONIC raw NPZ format](docs/SONIC_DATA_FORMAT.md)
- [LeRobot V3.1 dataset protocol](docs/UNITREE_G1_GMT_REFPOSE_V3_1_DATA_PROTOCOL.md)
