from __future__ import annotations

import unittest
from types import SimpleNamespace

from task_runtime_profiles import (
    OPEN_DOOR_DOOR_ASSET_VARIANT_ENV,
    OPEN_DOOR_DOOR_ASSET_VARIANT_INFERENCE_VALI,
    OPEN_DOOR_DOOR_ASSET_VARIANT_RECORDED_COMPAT,
    OPEN_DOOR_TASK,
    RUNTIME_CONTEXT_DIRECT_REPLAY,
    RUNTIME_CONTEXT_INFERENCE_REPLAY,
    RUNTIME_CONTEXT_LIVE_INFERENCE,
    RUNTIME_CONTEXT_RERECORD,
    TASK_RUNTIME_PROFILE_AUTO,
    TASK_RUNTIME_PROFILE_REPLAY_COMPAT,
    apply_task_runtime_profile,
    detect_runtime_context,
    resolve_open_door_door_usd_path,
)


class TaskRuntimeProfilesTests(unittest.TestCase):
    def test_detect_runtime_context_prefers_rerecord(self) -> None:
        args = SimpleNamespace(
            record_during_replay=True,
            input_source="replay",
            replay_file="/tmp/episode.npz",
            replay_mode="inference_replay",
        )

        self.assertEqual(detect_runtime_context(args), RUNTIME_CONTEXT_RERECORD)

    def test_detect_runtime_context_resolves_replay_modes(self) -> None:
        direct_args = SimpleNamespace(
            record_during_replay=False,
            input_source="replay",
            replay_file="",
            replay_mode="direct_replay",
        )
        inference_args = SimpleNamespace(
            record_during_replay=False,
            input_source="",
            replay_file="/tmp/episode.npz",
            replay_mode="inference_replay",
        )
        live_args = SimpleNamespace(
            record_during_replay=False,
            input_source="",
            replay_file="",
            replay_mode="",
        )

        self.assertEqual(detect_runtime_context(direct_args), RUNTIME_CONTEXT_DIRECT_REPLAY)
        self.assertEqual(detect_runtime_context(inference_args), RUNTIME_CONTEXT_INFERENCE_REPLAY)
        self.assertEqual(detect_runtime_context(live_args), RUNTIME_CONTEXT_LIVE_INFERENCE)

    def test_apply_task_runtime_profile_open_door_live_inference(self) -> None:
        args = SimpleNamespace(
            task=OPEN_DOOR_TASK,
            task_runtime_profile=TASK_RUNTIME_PROFILE_AUTO,
            record_during_replay=False,
            input_source="",
            replay_file="",
            replay_mode="",
        )
        env: dict[str, str] = {}

        applied = apply_task_runtime_profile(args, env=env)

        self.assertEqual(applied.context, RUNTIME_CONTEXT_LIVE_INFERENCE)
        self.assertEqual(applied.profile, "inference")
        self.assertEqual(env["OPEN_DOOR_LATCH_DISABLE"], "0")
        self.assertEqual(env["OPEN_DOOR_SCENE_AS_ARTICULATION"], "1")
        self.assertEqual(env[OPEN_DOOR_DOOR_ASSET_VARIANT_ENV], OPEN_DOOR_DOOR_ASSET_VARIANT_INFERENCE_VALI)

    def test_apply_task_runtime_profile_open_door_inference_replay_uses_replay_compat(self) -> None:
        args = SimpleNamespace(
            task=OPEN_DOOR_TASK,
            task_runtime_profile=TASK_RUNTIME_PROFILE_AUTO,
            record_during_replay=False,
            input_source="replay",
            replay_file="/tmp/episode.npz",
            replay_mode="inference_replay",
        )
        env: dict[str, str] = {}

        applied = apply_task_runtime_profile(args, env=env)

        self.assertEqual(applied.context, RUNTIME_CONTEXT_INFERENCE_REPLAY)
        self.assertEqual(applied.profile, TASK_RUNTIME_PROFILE_REPLAY_COMPAT)
        self.assertEqual(env["OPEN_DOOR_LATCH_DISABLE"], "1")
        self.assertEqual(env["OPEN_DOOR_SCENE_AS_ARTICULATION"], "0")
        self.assertEqual(env[OPEN_DOOR_DOOR_ASSET_VARIANT_ENV], OPEN_DOOR_DOOR_ASSET_VARIANT_RECORDED_COMPAT)

    def test_resolve_open_door_door_usd_path_by_variant(self) -> None:
        project_root = "/tmp/HumanoidArena"

        inference_path = resolve_open_door_door_usd_path(
            project_root,
            OPEN_DOOR_DOOR_ASSET_VARIANT_INFERENCE_VALI,
        )
        replay_path = resolve_open_door_door_usd_path(
            project_root,
            OPEN_DOOR_DOOR_ASSET_VARIANT_RECORDED_COMPAT,
        )

        self.assertTrue(inference_path.endswith("model_door001_vali.usd"))
        self.assertTrue(replay_path.endswith("model_door001.usd"))

    def test_explicit_profile_on_unsupported_task_raises(self) -> None:
        args = SimpleNamespace(
            task="Isaac-Move-Football-Single-G129-Dex3-Wholebody",
            task_runtime_profile="replay_compat",
            record_during_replay=False,
            input_source="",
            replay_file="",
            replay_mode="",
        )

        with self.assertRaises(ValueError):
            apply_task_runtime_profile(args, env={})


if __name__ == "__main__":
    unittest.main()
