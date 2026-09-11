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
from motion.g1_reference import resample_hand_continuous_chunk
from evaluation import humanoidarena_server as arena_server
from evaluation.simple_server import (
    _make_simple_runtime_class,
    _reset_episode_step_counters,
    _resolve_future_root_boundary,
    RIGHT_HAND_CLOSE,
    _wbc_hand_to_mjcf,
    arena_action_to_simple,
    SIMPLE_RIGHT_HAND_CLOSE,
)

KimodoSimpleHumanoidArenaRuntime = _make_simple_runtime_class(arena_server)
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


class _ContinuousHandPolicy(_FakePolicy):
    def predict_future(self, **kwargs):
        output = super().predict_future(**kwargs)
        output["hand_closure"] = torch.tensor(
            [[0.2, 0.4], [0.8, 1.0]], dtype=torch.float32
        )
        output["hand_clean"] = output["hand_closure"] * 2.0 - 1.0
        output["hand_binary"] = (output["hand_closure"] >= 0.5).float()
        return output


class _ContinuousTailHandPolicy(_FakePolicy):
    """Predict a close event outside a 15-frame executed body prefix."""

    def predict_future(self, **kwargs):
        output = super().predict_future(**kwargs)
        frames = 50
        output["motion_features"] = torch.zeros(frames, 4)
        output["local_rot_mats"] = (
            torch.eye(3).reshape(1, 1, 3, 3).repeat(frames, 1, 1, 1)
        )
        output["root_positions"] = torch.zeros(frames, 3)
        hand_closure = torch.zeros(frames, 2)
        hand_closure[30:, 1] = 1.0
        output["hand_closure"] = hand_closure
        output["hand_clean"] = hand_closure * 2.0 - 1.0
        output["hand_binary"] = (hand_closure >= 0.5).float()
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


def _runtime(codec, runtime_cls=KimodoHumanoidArenaRuntime):
    runtime = runtime_cls.__new__(runtime_cls)
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
    def test_simple_binary_close_uses_training_hand_target_and_mjcf_order(self):
        action = np.zeros((1, 40), dtype=np.float32)
        action[0, 39] = 1.0

        source = arena_action_to_simple(action)
        np.testing.assert_allclose(source[0, 7:14], RIGHT_HAND_CLOSE)
        np.testing.assert_allclose(
            _wbc_hand_to_mjcf(source[0, 7:14]),
            np.asarray([-0.5, -0.7, -0.7, 0.6, 1.5, 1.5, 1.5], dtype=np.float32),
        )

    def test_simple_continuous_closure_maps_without_binary_threshold(self):
        action = np.zeros((1, 40), dtype=np.float32)
        action[0, 39] = 0.25

        source = arena_action_to_simple(action, hand_control_mode="continuous")

        np.testing.assert_allclose(source[0, 7:14], 0.25 * SIMPLE_RIGHT_HAND_CLOSE)
        self.assertFalse(np.allclose(source[0, 7:14], np.zeros(7)))

    def test_simple_yaw_command_matches_official_rate_and_target_contract(self):
        action = np.zeros((3, 40), dtype=np.float32)
        relative_yaws = (0.0, 0.01, 0.01)
        for frame, relative_yaw in enumerate(relative_yaws):
            relative_rotation = np.asarray(
                [
                    [np.cos(relative_yaw), -np.sin(relative_yaw), 0.0],
                    [np.sin(relative_yaw), np.cos(relative_yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            action[frame, 3:9] = relative_rotation[:, :2].reshape(6)
        initial_yaw = 3.0 * np.pi / 4.0
        episode_heading = np.asarray(
            [
                [np.cos(initial_yaw), -np.sin(initial_yaw), 0.0],
                [np.sin(initial_yaw), np.cos(initial_yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        source = arena_action_to_simple(
            action,
            control_fps=50.0,
            episode_heading=episode_heading,
            previous_target_yaw=initial_yaw,
        )

        np.testing.assert_allclose(source[:, 34], [0.0, 0.5, 0.0], atol=1e-5)
        np.testing.assert_allclose(
            source[:, 35], initial_yaw + np.asarray(relative_yaws), atol=1e-6
        )

    def test_simple_yaw_command_wraps_and_clips_like_official_streamer(self):
        action = np.zeros((1, 40), dtype=np.float32)
        target_yaw = -np.pi + 0.02
        rotation = np.asarray(
            [
                [np.cos(target_yaw), -np.sin(target_yaw), 0.0],
                [np.sin(target_yaw), np.cos(target_yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        action[0, 3:9] = rotation[:, :2].reshape(6)

        source = arena_action_to_simple(
            action,
            control_fps=50.0,
            previous_target_yaw=np.pi - 0.02,
        )

        np.testing.assert_allclose(source[:, 34], [1.0], atol=1e-6)

    def test_simple_continuous_runtime_executes_closure_without_fsm(self):
        runtime = _runtime(_FakeActionCodec(), KimodoSimpleHumanoidArenaRuntime)
        runtime.max_navigation_speed = 100.0
        runtime.simple_hand_control_mode = "continuous"
        runtime.execution_frames = 1
        runtime._init_simple_hand_fsm()
        runtime.model = _ContinuousHandPolicy()

        action_chunk = runtime.infer(_payload())

        source = torch.tensor([[0.2, 0.4]])
        expected = resample_hand_continuous_chunk(
            source,
            source_fps=30,
            target_fps=50,
            previous_hand_closure=torch.zeros(2),
        )
        np.testing.assert_allclose(action_chunk[:, 38:40], expected.numpy())
        torch.testing.assert_close(runtime.history_hand[0, -1:], source)
        self.assertFalse(runtime.simple_hand_fsm_enabled)

    def test_simple_continuous_hand_executes_same_prefix_as_body(self):
        runtime = _runtime(_FakeActionCodec(), KimodoSimpleHumanoidArenaRuntime)
        runtime.max_navigation_speed = 100.0
        runtime.simple_hand_control_mode = "continuous"
        runtime.execution_frames = 15
        runtime._init_simple_hand_fsm()
        runtime.model = _ContinuousTailHandPolicy()

        chunks = []
        queue_lengths = []
        for _ in range(4):
            chunks.append(runtime.infer(_payload())[:, 38:40])
            queue_lengths.append(runtime.simple_continuous_hand_queue.shape[0])

        self.assertTrue(all(chunk.shape == (25, 2) for chunk in chunks))
        for chunk in chunks:
            np.testing.assert_allclose(chunk, 0.0)
        self.assertEqual(queue_lengths, [0, 0, 0, 0])
        self.assertFalse(runtime.simple_hand_fsm_enabled)

    def test_simple_continuous_full_prediction_executes_without_queue(self):
        runtime = _runtime(_FakeActionCodec(), KimodoSimpleHumanoidArenaRuntime)
        runtime.max_navigation_speed = 100.0
        runtime.simple_hand_control_mode = "continuous"
        runtime.execution_frames = 50
        runtime._init_simple_hand_fsm()
        runtime.model = _ContinuousTailHandPolicy()

        action_chunk = runtime.infer(_payload())

        self.assertEqual(action_chunk.shape, (83, 40))
        self.assertGreater(float(action_chunk[:, 39].max()), 0.5)
        self.assertEqual(runtime.simple_continuous_hand_queue.shape, (0, 2))
        self.assertFalse(runtime.simple_hand_fsm_enabled)

    def test_simple_continuous_conditions_on_measured_hand_history(self):
        runtime = _runtime(_FakeActionCodec(), KimodoSimpleHumanoidArenaRuntime)
        runtime.max_navigation_speed = 100.0
        runtime.simple_hand_control_mode = "continuous"
        runtime.execution_frames = 1
        runtime._init_simple_hand_fsm()
        runtime.model = _ContinuousHandPolicy()
        payload = _payload()
        payload["observation"]["state_history"] = np.zeros(
            (2, 64), dtype=np.float32
        ).tolist()
        payload["observation"]["hand_closure_history"] = [
            [0.1, 0.2],
            [0.3, 0.4],
        ]

        with patch(
            "evaluation.humanoidarena_server._partial_arena_state_history_to_motion",
            return_value=(
                torch.zeros(2, 4),
                torch.ones(2, 4, dtype=torch.bool),
                torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 1, 1, 1),
                torch.zeros(2, 3),
            ),
        ):
            runtime.infer(payload)

        # The newest 50 Hz observation is the current prediction cut.  The
        # model sees only the preceding measured value, never its prior output.
        expected_condition = torch.tensor([[[0.1, 0.2]]])
        self.assertEqual(tuple(runtime.model.received_history.shape), (1, 1, 4))
        torch.testing.assert_close(runtime.model.received_hand_history, expected_condition)
        torch.testing.assert_close(runtime.history_hand, expected_condition)
        np.testing.assert_allclose(
            runtime.simple_observed_hand_closure_history,
            np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        )
        self.assertIsNone(runtime.rtc_hand_tail)

    def test_simple_continuous_observed_hand_history_is_transactional(self):
        runtime = _runtime(
            _FakeActionCodec(RuntimeError("injected action encoding failure")),
            KimodoSimpleHumanoidArenaRuntime,
        )
        runtime.max_navigation_speed = 100.0
        runtime.simple_hand_control_mode = "continuous"
        runtime._init_simple_hand_fsm()
        runtime.model = _ContinuousHandPolicy()
        payload = _payload()
        payload["observation"]["hand_closure_history"] = [[0.1, 0.2]]
        before_hand = runtime.history_hand.clone()

        with self.assertRaisesRegex(RuntimeError, "action encoding failure"):
            runtime.infer(payload)

        torch.testing.assert_close(runtime.history_hand, before_hand)
        self.assertEqual(runtime.simple_observed_hand_closure_history.shape, (0, 2))

    def test_simple_hand_event_in_unexecuted_tail_uses_countdown(self):
        runtime = KimodoSimpleHumanoidArenaRuntime.__new__(
            KimodoSimpleHumanoidArenaRuntime
        )
        runtime.control_fps = 50.0
        runtime._init_simple_hand_fsm()

        # Only the first two model frames are executed.  The close event is
        # predicted at frame 3, so one source-rate frame remains afterward.
        executed = torch.zeros(2, 2)
        complete = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )
        effective, transitions = runtime._resolve_simple_hand_prefix(
            executed, detection_hand=complete
        )
        torch.testing.assert_close(effective, executed)
        self.assertTrue((transitions == -1).all())
        self.assertFalse(bool(runtime.simple_hand_state[0]))
        self.assertEqual(runtime.simple_hand_phase[0], "stable_open")
        self.assertEqual(int(runtime.simple_hand_pending_steps[0]), 1)

        # A new prediction cannot push the committed event back into its tail.
        effective_next, transitions_next = runtime._resolve_simple_hand_prefix(
            executed, detection_hand=torch.zeros(5, 2)
        )
        torch.testing.assert_close(effective_next[:, 0], torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(effective_next[:, 1], torch.zeros(2))
        self.assertEqual(int(transitions_next[0]), 1)
        self.assertTrue(bool(runtime.simple_hand_state[0]))
        self.assertEqual(int(runtime.simple_hand_pending_steps[0]), -1)

    def test_simple_hand_open_requires_two_replans(self):
        runtime = KimodoSimpleHumanoidArenaRuntime.__new__(
            KimodoSimpleHumanoidArenaRuntime
        )
        runtime.control_fps = 50.0
        runtime._init_simple_hand_fsm()
        runtime.simple_hand_state[0] = True
        runtime.simple_hand_phase[0] = "stable_closed"
        predicted_open = torch.zeros(8, 2)

        first, first_transition = runtime._resolve_simple_hand_prefix(predicted_open)
        torch.testing.assert_close(first[:, 0], torch.ones(8))
        self.assertEqual(int(first_transition[0]), -1)
        self.assertTrue(bool(runtime.simple_hand_state[0]))

        second, second_transition = runtime._resolve_simple_hand_prefix(predicted_open)
        torch.testing.assert_close(second[:, 0], torch.zeros(8))
        self.assertEqual(int(second_transition[0]), 0)
        self.assertFalse(bool(runtime.simple_hand_state[0]))

    def test_simple_hand_does_not_open_while_full_prediction_returns_to_close(self):
        runtime = KimodoSimpleHumanoidArenaRuntime.__new__(
            KimodoSimpleHumanoidArenaRuntime
        )
        runtime.control_fps = 50.0
        runtime._init_simple_hand_fsm()
        runtime.simple_hand_state[0] = True
        runtime.simple_hand_phase[0] = "stable_closed"
        executed_open = torch.zeros(15, 2)
        complete = torch.zeros(50, 2)
        complete[32:, 0] = 1.0

        for _ in range(3):
            effective, transitions = runtime._resolve_simple_hand_prefix(
                executed_open, detection_hand=complete
            )
            torch.testing.assert_close(effective[:, 0], torch.ones(15))
            self.assertEqual(int(transitions[0]), -1)
            self.assertTrue(bool(runtime.simple_hand_state[0]))
            self.assertEqual(int(runtime.simple_hand_open_votes[0]), 0)

    def test_simple_hand_open_vote_is_reset_when_future_close_reappears(self):
        runtime = KimodoSimpleHumanoidArenaRuntime.__new__(
            KimodoSimpleHumanoidArenaRuntime
        )
        runtime.control_fps = 50.0
        runtime._init_simple_hand_fsm()
        runtime.simple_hand_state[0] = True
        runtime.simple_hand_phase[0] = "stable_closed"
        executed_open = torch.zeros(15, 2)

        runtime._resolve_simple_hand_prefix(
            executed_open, detection_hand=torch.zeros(50, 2)
        )
        self.assertEqual(int(runtime.simple_hand_open_votes[0]), 1)

        complete_with_future_close = torch.zeros(50, 2)
        complete_with_future_close[40:, 0] = 1.0
        effective, transitions = runtime._resolve_simple_hand_prefix(
            executed_open, detection_hand=complete_with_future_close
        )
        torch.testing.assert_close(effective[:, 0], torch.ones(15))
        self.assertEqual(int(transitions[0]), -1)
        self.assertTrue(bool(runtime.simple_hand_state[0]))
        self.assertEqual(int(runtime.simple_hand_open_votes[0]), 0)

    def test_reset_episode_step_counters_clears_nested_timelimit(self):
        class Wrapper:
            def __init__(self, env=None, elapsed=None):
                self.env = env
                if elapsed is not None:
                    self._elapsed_steps = elapsed

        inner = Wrapper(elapsed=17)
        outer = Wrapper(inner, elapsed=271)
        _reset_episode_step_counters(outer)
        self.assertEqual(outer._elapsed_steps, 0)
        self.assertEqual(inner._elapsed_steps, 0)

    def test_arena_runtime_has_no_simple_only_hooks(self):
        self.assertFalse(
            hasattr(KimodoHumanoidArenaRuntime, "_resolve_prediction_root_boundary")
        )
        self.assertFalse(hasattr(KimodoHumanoidArenaRuntime, "_after_action_encoded"))

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

    def test_arena_action_conversion_preserves_history_root_boundary(self):
        codec = _FakeActionCodec()
        runtime = _runtime(codec)

        runtime.infer(_payload())

        torch.testing.assert_close(
            codec.received_previous_root, torch.tensor([9.0, 8.0, 7.0])
        )

    def test_simple_action_conversion_uses_future_root_extrapolation_boundary(self):
        codec = _FakeActionCodec()
        runtime = _runtime(codec, KimodoSimpleHumanoidArenaRuntime)
        runtime.max_navigation_speed = 100.0
        runtime.model = _FakePolicy()
        runtime.infer(_payload())
        torch.testing.assert_close(
            codec.received_previous_root, torch.tensor([-0.2, -0.1, 0.0])
        )

    def test_simple_history_root_output_cannot_change_action_boundary(self):
        codec = _FakeActionCodec()
        runtime = _runtime(codec, KimodoSimpleHumanoidArenaRuntime)
        runtime.max_navigation_speed = 100.0
        runtime.model = _FakePolicy()
        runtime.infer(_payload())
        first_boundary = codec.received_previous_root.clone()
        runtime.model = _FakePolicy()
        runtime.model.predict_future = lambda **kwargs: {
            "motion_features": torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
            "local_rot_mats": torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 1, 1, 1),
            "root_positions": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            "history_last_root_position": torch.tensor([1000.0, -1000.0, 500.0]),
            "hand_binary": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        }
        runtime.infer(_payload())
        torch.testing.assert_close(codec.received_previous_root, first_boundary)

    def test_simple_navigation_safety_failure_restores_runtime_state(self):
        codec = _FakeActionCodec()
        runtime = _runtime(codec, KimodoSimpleHumanoidArenaRuntime)
        runtime.max_navigation_speed = 1.0
        expected = _snapshot(runtime)

        with self.assertRaisesRegex(ValueError, "navigation speed"):
            runtime.infer(_payload())

        _assert_snapshot(self, runtime, expected)

    def test_future_root_boundary_extrapolates_first_velocity(self):
        torch.testing.assert_close(
            _resolve_future_root_boundary(torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])),
            torch.tensor([-0.2, -0.1, 0.0]),
        )
        torch.testing.assert_close(
            _resolve_future_root_boundary(torch.tensor([[0.1, 0.2, 0.3]])),
            torch.tensor([0.1, 0.2, 0.3]),
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

    def test_model_sees_executed_predicted_hand_history(self):
        runtime = _runtime(_FakeActionCodec())
        runtime.history_hand = torch.tensor([[[1.0, 1.0]]])

        with patch(
            "evaluation.humanoidarena_server.resample_hand_binary_chunk",
            wraps=resample_hand_binary_chunk,
        ) as resample_hand:
            runtime.infer(_payload())

        torch.testing.assert_close(
            runtime.model.received_hand_history,
            runtime.history_hand[:, :1],
        )
        torch.testing.assert_close(
            resample_hand.call_args.kwargs["previous_hand_binary"],
            torch.tensor([1.0, 1.0]),
        )

    def test_predicted_hand_history_is_right_aligned_with_state_motion_history(self):
        runtime = _runtime(_FakeActionCodec())
        runtime.history_hand = torch.tensor(
            [[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]]
        )
        payload = _payload()
        payload["observation"]["state_history"] = np.zeros(
            (3, 64), dtype=np.float32
        ).tolist()

        with patch(
            "evaluation.humanoidarena_server._partial_arena_state_history_to_motion",
            return_value=(
                torch.zeros(3, 4),
                torch.ones(3, 4, dtype=torch.bool),
                torch.eye(3).reshape(1, 1, 3, 3).repeat(3, 1, 1, 1),
                torch.zeros(3, 3),
            ),
        ):
            runtime.infer(payload)

        self.assertEqual(tuple(runtime.model.received_history.shape), (1, 2, 4))
        torch.testing.assert_close(
            runtime.model.received_hand_history,
            torch.tensor([[[1.0, 1.0], [1.0, 0.0]]]),
        )

    def test_measured_state_does_not_replace_emitted_rotation_boundary(self):
        runtime = _runtime(_FakeActionCodec())
        payload = _payload()
        payload["observation"]["state_history"] = np.zeros(
            (2, 64), dtype=np.float32
        ).tolist()
        angle = 1.0
        measured_rotation = torch.tensor(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ).reshape(1, 3, 3)

        def convert(states, **kwargs):
            frame_count = states.shape[0]
            return (
                torch.zeros(frame_count, 4),
                torch.ones(frame_count, 4, dtype=torch.bool),
                measured_rotation.repeat(frame_count, 1, 1),
                torch.zeros(frame_count, 3),
            )

        with patch(
            "evaluation.humanoidarena_server._partial_arena_state_history_to_motion",
            side_effect=convert,
        ), patch(
            "evaluation.humanoidarena_server.resample_motion_chunk",
            wraps=arena_server.resample_motion_chunk,
        ) as resample_motion:
            runtime.infer(payload)
            emitted_boundary = runtime.previous_local_rot_mat.clone()
            runtime.infer(payload)

        self.assertEqual(resample_motion.call_count, 2)
        first_boundary = resample_motion.call_args_list[0].kwargs["previous_local_rot_mat"]
        second_boundary = resample_motion.call_args_list[1].kwargs["previous_local_rot_mat"]
        torch.testing.assert_close(first_boundary, torch.eye(3).reshape(1, 3, 3))
        torch.testing.assert_close(second_boundary, emitted_boundary)
        self.assertFalse(torch.allclose(first_boundary, measured_rotation))
        self.assertFalse(torch.allclose(second_boundary, measured_rotation))

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
