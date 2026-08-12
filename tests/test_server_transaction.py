import base64
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from evaluation.humanoidarena_server import (
    KimodoHumanoidArenaRuntime,
    _partial_arena_state_to_motion,
    _partial_arena_state_history_to_motion,
    _resolve_rtc_parameters,
    resample_hand_binary_chunk,
)
from motion.g1_reference import HumanoidArenaActionDecoder
from motion.representation.kimodo_motionrep import KimodoMotionRep
from skeleton.definitions import G1Skeleton34


class _FakePolicy:
    fps = 30
    config = SimpleNamespace(action_history=3)
    representation = SimpleNamespace(motion_rep_dim=4)

    def __init__(self):
        self.received_history = None
        self.received_hand_history = None
        self.received_history_feature_mask = None
        self.received_rtc_motion_reference = None
        self.received_rtc_hand_reference = None

    def predict_future(
        self, history_motion, hand_history, history_feature_mask=None, **kwargs
    ):
        self.received_history = history_motion.detach().clone()
        self.received_hand_history = hand_history.detach().clone()
        self.received_history_feature_mask = (
            None
            if history_feature_mask is None
            else history_feature_mask.detach().clone()
        )
        rtc_motion_reference = kwargs.get("rtc_motion_reference")
        rtc_hand_reference = kwargs.get("rtc_hand_reference")
        self.received_rtc_motion_reference = (
            None
            if rtc_motion_reference is None
            else rtc_motion_reference.detach().clone()
        )
        self.received_rtc_hand_reference = (
            None
            if rtc_hand_reference is None
            else rtc_hand_reference.detach().clone()
        )
        local_rot_mats = torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 1, 1, 1)
        local_rot_mats[1, 0, 0, 0] = 0.75
        return {
            "motion_features": torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
            ),
            "local_rot_mats": local_rot_mats,
            "root_positions": torch.tensor(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            ),
            "history_last_root_position": torch.tensor([9.0, 8.0, 7.0]),
            "hand_binary": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        }


class _RtcTailPolicy(_FakePolicy):
    def predict_future(self, **kwargs):
        output = super().predict_future(**kwargs)
        output["motion_features"] = torch.cat(
            (
                output["motion_features"],
                torch.tensor([[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]),
            ),
            dim=0,
        )
        extra_rotations = torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 1, 1, 1)
        output["local_rot_mats"] = torch.cat(
            (output["local_rot_mats"], extra_rotations), dim=0
        )
        output["root_positions"] = torch.cat(
            (
                output["root_positions"],
                torch.tensor([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]),
            ),
            dim=0,
        )
        output["hand_binary"] = torch.cat(
            (output["hand_binary"], torch.tensor([[1.0, 1.0], [0.0, 0.0]])),
            dim=0,
        )
        output["hand_clean"] = output["hand_binary"] * 2.0 - 1.0
        return output


class _MutatingFailPolicy(_FakePolicy):
    def predict_future(self, history_motion, hand_history, **kwargs):
        history_motion.add_(100.0)
        hand_history.add_(1.0)
        raise RuntimeError("injected prediction failure")


class _FakePolicyWithoutHand(_FakePolicy):
    def predict_future(self, **kwargs):
        output = super().predict_future(**kwargs)
        output.pop("hand_binary")
        return output


class _FakeActionCodec:
    def __init__(self, error=None, mutate_previous_root=False):
        self.error = error
        self.mutate_previous_root = mutate_previous_root
        self.received_previous_root = None

    def encode(self, local_rot_mats, root_positions, previous_root_position, **kwargs):
        self.received_previous_root = previous_root_position.clone()
        if self.mutate_previous_root and previous_root_position is not None:
            previous_root_position.add_(100.0)
        if self.error is not None:
            raise self.error
        return torch.full((local_rot_mats.shape[0], 40), 0.25)


def _payload():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    return {
        "task": "task-id",
        "observation": {
            "images": {
                "front": {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "data_b64": base64.b64encode(image.tobytes()).decode("ascii"),
                }
            }
        },
    }


def _runtime(codec):
    runtime = KimodoHumanoidArenaRuntime.__new__(KimodoHumanoidArenaRuntime)
    runtime.deterministic_eval = False
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime.diffusion_steps = 2
    runtime.control_fps = 50.0
    runtime.execution_frames = 2
    runtime.rtc_enabled = False
    runtime.rtc_overlap_frames = 0
    runtime.rtc_frozen_frames = 0
    runtime.rtc_ramp_power = 1.0
    runtime.model = _FakePolicy()
    runtime.action_codec = codec
    runtime.text_embeddings = {"task-id": torch.zeros(1, 6)}
    runtime.task_instructions = {"task-id": "Do the task."}
    runtime.observation_state_history = np.empty((0, 64), dtype=np.float32)
    runtime.state_motion_representation = object()
    runtime.history_motion = torch.tensor([[[-1.0, -2.0, -3.0, -4.0]]])
    runtime.history_hand = torch.tensor([[[0.0, 0.0]]])
    runtime.previous_local_rot_mat = torch.eye(3).reshape(1, 3, 3)
    runtime.previous_root_position = torch.tensor([-0.1, -0.2, -0.3])
    runtime.rtc_motion_tail = None
    runtime.rtc_hand_tail = None
    runtime.rtc_task_id = None
    runtime.episode_seed = None
    runtime.inference_index = 0
    runtime.lock = threading.Lock()
    return runtime


def _snapshot(runtime):
    return {
        "history_motion": runtime.history_motion.clone(),
        "history_hand": runtime.history_hand.clone(),
        "previous_local_rot_mat": runtime.previous_local_rot_mat.clone(),
        "previous_root_position": runtime.previous_root_position.clone(),
        "rtc_motion_tail": (
            None
            if runtime.rtc_motion_tail is None
            else runtime.rtc_motion_tail.clone()
        ),
        "rtc_hand_tail": (
            None if runtime.rtc_hand_tail is None else runtime.rtc_hand_tail.clone()
        ),
        "rtc_task_id": runtime.rtc_task_id,
        "inference_index": runtime.inference_index,
    }


def _assert_snapshot(test_case, runtime, expected):
    torch.testing.assert_close(runtime.history_motion, expected["history_motion"])
    torch.testing.assert_close(runtime.history_hand, expected["history_hand"])
    torch.testing.assert_close(
        runtime.previous_local_rot_mat, expected["previous_local_rot_mat"]
    )
    torch.testing.assert_close(
        runtime.previous_root_position, expected["previous_root_position"]
    )
    if expected["rtc_motion_tail"] is None:
        test_case.assertIsNone(runtime.rtc_motion_tail)
    else:
        torch.testing.assert_close(
            runtime.rtc_motion_tail, expected["rtc_motion_tail"]
        )
    if expected["rtc_hand_tail"] is None:
        test_case.assertIsNone(runtime.rtc_hand_tail)
    else:
        torch.testing.assert_close(runtime.rtc_hand_tail, expected["rtc_hand_tail"])
    test_case.assertEqual(runtime.rtc_task_id, expected["rtc_task_id"])
    test_case.assertEqual(runtime.inference_index, expected["inference_index"])


class ServerTransactionTest(unittest.TestCase):
    def test_rtc_parameters_clamp_to_unexecuted_tail(self):
        resolved = _resolve_rtc_parameters(
            True,
            prediction_frames=50,
            execution_frames=45,
            overlap_frames=12,
            frozen_frames=2,
            ramp_power=1.5,
        )
        self.assertEqual(resolved, (True, 5, 2, 1.5))
        self.assertEqual(
            _resolve_rtc_parameters(
                True,
                prediction_frames=50,
                execution_frames=50,
                overlap_frames=12,
                frozen_frames=1,
                ramp_power=1.0,
            ),
            (False, 0, 0, 1.0),
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            _resolve_rtc_parameters(
                True,
                prediction_frames=50,
                execution_frames=15,
                overlap_frames=2,
                frozen_frames=3,
                ramp_power=1.0,
            )

    def test_action_conversion_uses_current_diffusion_window_root_boundary(self):
        codec = _FakeActionCodec()
        runtime = _runtime(codec)

        runtime.infer(_payload())

        torch.testing.assert_close(
            codec.received_previous_root, torch.tensor([9.0, 8.0, 7.0])
        )

    def test_observation_state_history_resamples_50hz_to_30hz(self):
        project_root = __import__(
            "evaluation.humanoidarena_server", fromlist=["PROJECT_ROOT"]
        ).PROJECT_ROOT
        skeleton = G1Skeleton34()
        decoder = HumanoidArenaActionDecoder(
            skeleton,
            project_root / "skeleton/assets/g1skel34/xml/g1.xml",
            fps=50.0,
        )
        representation = KimodoMotionRep(
            G1Skeleton34(), fps=30.0, stats_path=None
        )
        states = np.zeros((6, 64), dtype=np.float32)
        states[:, :6] = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        states[:, 6] = np.linspace(0.0, 0.1, 6)

        motion, feature_mask, local_rot_mats, root_positions = (
            _partial_arena_state_history_to_motion(
                states,
                decoder=decoder,
                representation=representation,
                source_fps=50.0,
                target_fps=30.0,
            )
        )

        self.assertEqual(tuple(motion.shape), (4, 417))
        self.assertEqual(tuple(feature_mask.shape), (4, 417))
        self.assertEqual(tuple(local_rot_mats.shape), (4, 34, 3, 3))
        self.assertEqual(tuple(root_positions.shape), (4, 3))
        self.assertEqual(int(feature_mask[0].sum()), 206)
        self.assertTrue(torch.isfinite(motion).all())
        torch.testing.assert_close(root_positions, torch.zeros_like(root_positions))

    def test_arena_observation_state_builds_a_206_feature_partial_constraint(self):
        project_root = __import__(
            "evaluation.humanoidarena_server", fromlist=["PROJECT_ROOT"]
        ).PROJECT_ROOT
        decoder = HumanoidArenaActionDecoder(
            G1Skeleton34(),
            project_root / "skeleton/assets/g1skel34/xml/g1.xml",
            fps=50.0,
        )
        representation = KimodoMotionRep(
            G1Skeleton34(), fps=30.0, stats_path=None
        )
        state = np.zeros(64, dtype=np.float32)
        state[:6] = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

        motion, feature_mask, _, _ = _partial_arena_state_to_motion(
            state,
            decoder=decoder,
            representation=representation,
        )

        self.assertEqual(tuple(motion.shape), (1, 417))
        self.assertEqual(int(feature_mask.sum()), 206)
        self.assertTrue(feature_mask[:, 3:5].all())
        self.assertTrue(feature_mask[:, 107:311].all())

    def test_observation_state_history_is_used_instead_of_predicted_history(self):
        runtime = _runtime(_FakeActionCodec())
        payload = _payload()
        payload["observation"]["state_history"] = np.zeros((2, 64), dtype=np.float32).tolist()
        state_motion = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        state_feature_mask = torch.tensor(
            [[True, False, True, False]] * 5
        )
        state_local_rot = torch.eye(3).reshape(1, 1, 3, 3).repeat(5, 1, 1, 1)
        state_root = torch.zeros(5, 3)

        with patch(
            "evaluation.humanoidarena_server._partial_arena_state_history_to_motion",
            return_value=(state_motion, state_feature_mask, state_local_rot, state_root),
        ):
            runtime.infer(payload)

        expected_history = state_motion[1:4].unsqueeze(0)
        torch.testing.assert_close(runtime.model.received_history, expected_history)
        self.assertEqual(tuple(runtime.model.received_hand_history.shape), (1, 3, 2))
        torch.testing.assert_close(
            runtime.model.received_history_feature_mask,
            state_feature_mask[1:4].unsqueeze(0),
        )
        torch.testing.assert_close(runtime.history_motion, expected_history)
        self.assertEqual(runtime.observation_state_history.shape, (2, 64))

    def test_current_cut_state_is_committed_but_not_conditioned_on(self):
        runtime = _runtime(_FakeActionCodec())
        payload = _payload()
        payload["observation"]["state_history"] = np.zeros(
            (1, 64), dtype=np.float32
        ).tolist()

        with patch(
            "evaluation.humanoidarena_server._partial_arena_state_history_to_motion",
            return_value=(
                torch.full((1, 4), 9.0),
                torch.ones(1, 4, dtype=torch.bool),
                torch.eye(3).reshape(1, 1, 3, 3),
                torch.zeros(1, 3),
            ),
        ):
            runtime.infer(payload)

        self.assertEqual(tuple(runtime.model.received_history.shape), (1, 0, 4))
        self.assertEqual(
            tuple(runtime.model.received_history_feature_mask.shape), (1, 0, 4)
        )
        self.assertEqual(runtime.observation_state_history.shape, (1, 64))

    def test_failed_prediction_does_not_commit_received_observation_history(self):
        runtime = _runtime(_FakeActionCodec())
        runtime.model = _MutatingFailPolicy()
        payload = _payload()
        payload["observation"]["state_history"] = np.zeros((2, 64), dtype=np.float32).tolist()
        before = _snapshot(runtime)

        with patch(
            "evaluation.humanoidarena_server._partial_arena_state_history_to_motion",
            return_value=(
                torch.zeros(2, 4),
                torch.ones(2, 4, dtype=torch.bool),
                torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 1, 1, 1),
                torch.zeros(2, 3),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected prediction failure"):
                runtime.infer(payload)

        _assert_snapshot(self, runtime, before)
        self.assertEqual(runtime.observation_state_history.shape, (0, 64))

    def test_state_history_segments_are_appended_transactionally(self):
        runtime = _runtime(_FakeActionCodec())
        runtime.observation_state_history = np.full((1, 64), 1.0, dtype=np.float32)
        payload = _payload()
        payload["observation"]["state_history"] = np.full(
            (2, 64), 2.0, dtype=np.float32
        ).tolist()

        def convert(states, **kwargs):
            self.assertEqual(states.shape, (3, 64))
            np.testing.assert_allclose(states[0], 1.0)
            np.testing.assert_allclose(states[1:], 2.0)
            return (
                torch.zeros(3, 4),
                torch.ones(3, 4, dtype=torch.bool),
                torch.eye(3).reshape(1, 1, 3, 3).repeat(3, 1, 1, 1),
                torch.zeros(3, 3),
            )

        with patch(
            "evaluation.humanoidarena_server._partial_arena_state_history_to_motion",
            side_effect=convert,
        ):
            runtime.infer(payload)

        self.assertEqual(runtime.observation_state_history.shape, (3, 64))

    def test_reset_clears_observation_state_history(self):
        runtime = _runtime(_FakeActionCodec())
        runtime.observation_state_history = np.ones((20, 64), dtype=np.float32)
        runtime.rtc_motion_tail = torch.ones(2, 4)
        runtime.rtc_hand_tail = torch.ones(2, 2)
        runtime.rtc_task_id = "task-id"

        runtime.reset(seed=7)

        self.assertEqual(runtime.observation_state_history.shape, (0, 64))
        self.assertEqual(tuple(runtime.history_motion.shape), (1, 0, 4))
        self.assertEqual(tuple(runtime.history_hand.shape), (1, 0, 2))
        self.assertIsNone(runtime.previous_local_rot_mat)
        self.assertIsNone(runtime.previous_root_position)
        self.assertIsNone(runtime.rtc_motion_tail)
        self.assertIsNone(runtime.rtc_hand_tail)
        self.assertIsNone(runtime.rtc_task_id)

    def test_success_commits_unexecuted_tail_for_next_rtc_call(self):
        runtime = _runtime(_FakeActionCodec())
        runtime.model = _RtcTailPolicy()
        runtime.rtc_enabled = True
        runtime.rtc_overlap_frames = 2
        runtime.rtc_frozen_frames = 1

        runtime.infer(_payload())

        torch.testing.assert_close(
            runtime.rtc_motion_tail,
            torch.tensor([[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]),
        )
        torch.testing.assert_close(
            runtime.rtc_hand_tail, torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
        )
        self.assertEqual(runtime.rtc_task_id, "task-id")

        committed_motion = runtime.rtc_motion_tail.clone()
        committed_hand = runtime.rtc_hand_tail.clone()
        runtime.infer(_payload())

        torch.testing.assert_close(
            runtime.model.received_rtc_motion_reference, committed_motion
        )
        torch.testing.assert_close(
            runtime.model.received_rtc_hand_reference, committed_hand
        )

    def test_rtc_tail_is_transactional_on_downstream_failure(self):
        runtime = _runtime(
            _FakeActionCodec(RuntimeError("injected action encoding failure"))
        )
        runtime.model = _RtcTailPolicy()
        runtime.rtc_enabled = True
        runtime.rtc_overlap_frames = 2
        runtime.rtc_frozen_frames = 1
        runtime.rtc_motion_tail = torch.full((2, 4), 3.0)
        runtime.rtc_hand_tail = torch.full((2, 2), -1.0)
        runtime.rtc_task_id = "task-id"
        before = _snapshot(runtime)

        with self.assertRaisesRegex(RuntimeError, "action encoding failure"):
            runtime.infer(_payload())

        _assert_snapshot(self, runtime, before)

    def test_invalid_state_history_is_rejected_before_prediction(self):
        runtime = _runtime(_FakeActionCodec())
        payload = _payload()
        payload["observation"]["state_history"] = np.zeros(
            (2, 65), dtype=np.float32
        ).tolist()

        with self.assertRaisesRegex(ValueError, r"shape \[T, 64\]"):
            runtime.infer(payload)

        self.assertIsNone(runtime.model.received_history)
        self.assertEqual(runtime.observation_state_history.shape, (0, 64))

    def test_legacy_single_state_client_fails_instead_of_silently_mismatching(self):
        runtime = _runtime(_FakeActionCodec())
        payload = _payload()
        payload["observation"]["state"] = np.zeros(64, dtype=np.float32).tolist()

        with self.assertRaisesRegex(ValueError, "state_history"):
            runtime.infer(payload)

        self.assertIsNone(runtime.model.received_history)

    def test_prediction_failure_cannot_mutate_committed_histories(self):
        runtime = _runtime(_FakeActionCodec())
        runtime.model = _MutatingFailPolicy()
        before = _snapshot(runtime)

        with self.assertRaisesRegex(RuntimeError, "injected prediction failure"):
            runtime.infer(_payload())

        _assert_snapshot(self, runtime, before)

    def test_resampling_failure_does_not_advance_runtime_state(self):
        runtime = _runtime(_FakeActionCodec())
        before = _snapshot(runtime)

        def mutate_then_fail(
            *args, previous_local_rot_mat, previous_root_position, **kwargs
        ):
            previous_local_rot_mat.add_(100.0)
            previous_root_position.add_(100.0)
            raise RuntimeError("injected resampling failure")

        with patch(
            "evaluation.humanoidarena_server.resample_motion_chunk",
            side_effect=mutate_then_fail,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected resampling failure"):
                runtime.infer(_payload())

        _assert_snapshot(self, runtime, before)

    def test_success_without_hand_output_preserves_hand_history(self):
        runtime = _runtime(_FakeActionCodec())
        runtime.model = _FakePolicyWithoutHand()
        previous_hand_history = runtime.history_hand.clone()

        action_chunk = runtime.infer(_payload())

        self.assertEqual(action_chunk.shape, (3, 40))
        self.assertEqual(tuple(runtime.history_motion.shape), (1, 3, 4))
        torch.testing.assert_close(runtime.history_hand, previous_hand_history)

    def test_model_sees_open_hands_but_resampling_uses_executed_hand_state(self):
        runtime = _runtime(_FakeActionCodec())
        runtime.history_hand = torch.tensor([[[1.0, 1.0]]])

        with patch(
            "evaluation.humanoidarena_server.resample_hand_binary_chunk",
            wraps=resample_hand_binary_chunk,
        ) as resample_hand:
            runtime.infer(_payload())

        torch.testing.assert_close(
            runtime.model.received_hand_history,
            torch.zeros_like(runtime.model.received_hand_history),
        )
        torch.testing.assert_close(
            resample_hand.call_args.kwargs["previous_hand_binary"],
            torch.tensor([1.0, 1.0]),
        )

    def test_encoding_failure_does_not_advance_runtime_state(self):
        runtime = _runtime(
            _FakeActionCodec(
                RuntimeError("injected action encoding failure"),
                mutate_previous_root=True,
            )
        )
        runtime.deterministic_eval = True
        runtime.episode_seed = 1234
        runtime.inference_index = 7
        before = _snapshot(runtime)

        with self.assertRaisesRegex(RuntimeError, "injected action encoding failure"):
            runtime.infer(_payload())

        _assert_snapshot(self, runtime, before)

    def test_hand_resampling_mismatch_does_not_advance_runtime_state(self):
        runtime = _runtime(_FakeActionCodec())
        before = _snapshot(runtime)

        with patch(
            "evaluation.humanoidarena_server.resample_hand_binary_chunk",
            return_value=torch.zeros(2, 2),
        ):
            with self.assertRaisesRegex(RuntimeError, "different lengths"):
                runtime.infer(_payload())

        _assert_snapshot(self, runtime, before)

    def test_response_conversion_failure_does_not_advance_runtime_state(self):
        runtime = _runtime(_FakeActionCodec())
        before = _snapshot(runtime)

        with patch.object(
            torch.Tensor,
            "numpy",
            side_effect=RuntimeError("injected response conversion failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "response conversion failure"):
                runtime.infer(_payload())

        _assert_snapshot(self, runtime, before)

    def test_success_commits_all_runtime_state_together(self):
        runtime = _runtime(_FakeActionCodec())
        old_history_motion = runtime.history_motion.clone()
        old_history_hand = runtime.history_hand.clone()

        action_chunk = runtime.infer(_payload())

        self.assertEqual(action_chunk.shape, (3, 40))
        self.assertEqual(action_chunk.dtype, np.float32)
        np.testing.assert_allclose(action_chunk, 0.25)
        expected_motion = torch.cat(
            (
                old_history_motion,
                torch.tensor(
                    [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
                ),
            ),
            dim=1,
        )
        expected_hand = torch.cat(
            (old_history_hand, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])),
            dim=1,
        )
        torch.testing.assert_close(runtime.history_motion, expected_motion)
        torch.testing.assert_close(runtime.history_hand, expected_hand)
        torch.testing.assert_close(
            runtime.previous_root_position, torch.tensor([0.4, 0.5, 0.6])
        )
        self.assertEqual(runtime.inference_index, 0)

    def test_repeated_success_truncates_histories_to_configured_length(self):
        runtime = _runtime(_FakeActionCodec())

        runtime.infer(_payload())
        runtime.infer(_payload())

        self.assertEqual(tuple(runtime.history_motion.shape), (1, 3, 4))
        self.assertEqual(tuple(runtime.history_hand.shape), (1, 3, 2))
        torch.testing.assert_close(
            runtime.history_motion,
            torch.tensor(
                [[[5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
            ),
        )
        torch.testing.assert_close(
            runtime.history_hand,
            torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]),
        )

    def test_deterministic_success_advances_inference_index_once(self):
        runtime = _runtime(_FakeActionCodec())
        runtime.deterministic_eval = True
        runtime.episode_seed = 1234
        runtime.inference_index = 7

        with patch("builtins.print"):
            runtime.infer(_payload())

        self.assertEqual(runtime.inference_index, 8)


if __name__ == "__main__":
    unittest.main()
