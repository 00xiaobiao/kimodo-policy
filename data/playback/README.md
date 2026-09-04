# SIMPLE expert-aligned export

The default capture mode is `expert_aligned`. It exports the recorded expert
trajectory without rerunning WBC or contact physics:

- `observation.leg_joints` plus `observation.arm_joints` are reordered into the
  canonical 29-DoF body pose.
- The realized expert body/hand pose is used for both observation and training
  target, so controller tracking drift is not learned as ground truth.
- The original expert MP4 is copied byte-for-byte and its frame count, FPS, and
  SHA256 are validated.
- The original processed `states`, `action`, and measured joint columns are
  retained under `source.*` for audit.

The processed SIMPLE dataset does not contain measured floating-base or object
poses. Root XY/yaw is therefore reconstructed from the synchronized navigation
command and is explicitly marked `command_reconstructed_not_measured` in
metadata. Exact physical re-simulation of contacts is not possible from this
processed dataset alone.

## Export one episode

```bash
cd /ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2

TASK=G1WholebodyXMovePickTeleop-v0
SOURCE=/ai/Yichi/kimodo-policy/simpledata/simple/$TASK
OUTPUT=data/playback/output/$TASK/episode_000000

PYTHONPATH="$PWD:$PWD/SIMPLE/src" \
  SIMPLE/.venv/bin/python \
  data/playback/replay_capture.py \
  --capture-mode expert_aligned \
  --env-id "simple/$TASK" \
  --source "$SOURCE" \
  --episode 0 \
  --output "$OUTPUT"
```

`--env-id` is retained for CLI compatibility but no environment is created in
`expert_aligned` mode.

## Export every episode

The batch helper defaults to expert-aligned export, resumes completed outputs,
and starts a detached tmux session:

```bash
cd /ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2
CAPTURE_MODE=expert_aligned \
  bash data/playback/replay_task.sh G1WholebodyXMovePickTeleop-v0 "4"
```

The GPU list controls worker allocation for compatibility with physics replay.
Expert-aligned export itself does not use the GPU. Useful status commands are:

```bash
tmux ls
tmux attach -t replay_G1WholebodyXMovePickTeleop_v0
tail -f /tmp/replay_G1WholebodyXMovePickTeleop_v0.log
```

An existing episode is skipped only when `validation.json` has the requested
`capture_mode`, `quality_passed: true`, and, for expert-aligned output,
`expert_alignment_passed: true`. Historical physics replays are therefore not
mistaken for completed expert-aligned exports.

## Physics replay diagnostics

Use `physics_replay` only when the goal is to inspect controller or simulator
behavior. This mode reruns WBC against live MuJoCo proprioception and can render
through Isaac Sim, so contact-sensitive outcomes are not expected to match the
recorded expert episode exactly.

```bash
cd /ai/Yichi/yunhengwang/Kimodo-Policy/controlnet_v1.2
CAPTURE_MODE=physics_replay \
  bash data/playback/replay_task.sh G1WholebodyXMovePickTeleop-v0 "4"
```

For a single physics replay, use the Isaac Python environment and cache setup
from `replay_task.sh`, then pass `--capture-mode physics_replay` and either
`--sim-mode mujoco_isaac` or `--sim-mode mujoco`.

## Output contract

The exported training protocol remains 64-D state and 40-D action:

```text
observation.state = root_rot6d + expert_q29 + finite_difference_dq29
action            = reference_root_xy_delta + reference_root_height
                    + reference_root_rot6d + expert_q29 + hand_binary
```

Continuous hand training reads:

```text
observation.hand_q       recorded expert hand joints (14-D)
observation.hand_closure projection of recorded expert hand joints (2-D)
action.target_hand_q     recorded expert hand joints (14-D)
action.hand_closure      projection of recorded expert hand joints (2-D)
```

The raw command remains available as `source.action`; it is not substituted for
the realized expert joint target because it has no lower-body joint target and
contains controller tracking error for the upper body.

`validation.json` records zero-error body/hand alignment metrics and a source
video SHA256 match. `success` is `null` for expert-aligned output because the
processed source does not store a task-success flag. `next.done` is preserved
verbatim but is only an episode termination signal, not proof of success.

## Reference-root semantics

The 40-D action root is reconstructed from the synchronized 36-D source command:

```text
source.action[31]    -> reference root height
source.action[32:34] -> reference local XY velocity
source.action[35]    -> reference target yaw
```

At 50 FPS, local XY displacement is velocity multiplied by `0.02`. Target yaw
is episode-relative and stored as a yaw-only quaternion/rot6d because the
processed source exposes no measured pelvis roll/pitch. Explicit reconstructed
targets are also stored as `action.reference_root_p` and
`action.reference_root_q` (`wxyz`).
