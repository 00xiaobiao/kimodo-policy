import importlib.util
import unittest
from pathlib import Path

import numpy as np


CLIENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "HumanoidArena/isaaclab_twist2_g1/action_provider/lerobot_vla_http_client.py"
)
SPEC = importlib.util.spec_from_file_location("humanoidarena_http_client", CLIENT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
LeRobotVLAHttpClient = MODULE.LeRobotVLAHttpClient


class _CapturingClient(LeRobotVLAHttpClient):
    def __init__(self):
        super().__init__("http://127.0.0.1:1")
        self.payload = None

    def _post_json(self, path, payload):
        self.payload = payload
        return {"action_chunk": np.zeros((2, 40), dtype=np.float32).tolist()}


class HumanoidArenaHttpClientTest(unittest.TestCase):
    def test_infer_chunk_sends_incremental_state_history(self):
        client = _CapturingClient()
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        state_history = np.arange(3 * 64, dtype=np.float32).reshape(3, 64)

        chunk = client.infer_chunk(
            front_rgb=image,
            observation_state=state_history[-1],
            observation_state_history=state_history,
            robot_type="g1_29dof_with_hand",
            task="HOI_double_desk",
        )

        self.assertEqual(chunk.shape, (2, 40))
        transmitted = np.asarray(
            client.payload["observation"]["state_history"], dtype=np.float32
        )
        np.testing.assert_array_equal(transmitted, state_history)
        np.testing.assert_array_equal(
            client.payload["observation"]["state"], state_history[-1]
        )

    def test_invalid_history_dimension_is_rejected(self):
        client = _CapturingClient()

        with self.assertRaisesRegex(ValueError, r"shape \[T, state_dim\]"):
            client.infer_chunk(
                front_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                observation_state=np.zeros(64, dtype=np.float32),
                observation_state_history=np.zeros((2, 65), dtype=np.float32),
                robot_type="g1_29dof_with_hand",
            )

        self.assertIsNone(client.payload)


if __name__ == "__main__":
    unittest.main()
