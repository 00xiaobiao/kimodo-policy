import importlib.util
from pathlib import Path
import time
import unittest

import numpy as np


DEPLOY_PATH = (
    Path(__file__).resolve().parents[1] / "real-world" / "deploy" / "run_kimodo_sonic.py"
)
SPEC = importlib.util.spec_from_file_location("run_kimodo_sonic", DEPLOY_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _FastClient:
    def __init__(self):
        self.calls = 0

    def infer_chunk(self, image, state, state_history, task):
        self.calls += 1
        action = np.zeros((2, 40), dtype=np.float32)
        action[:, 2] = 0.85
        action[:, 3:9] = [1, 0, 0, 1, 0, 0]
        return action


class KimodoSonicRuntimeTest(unittest.TestCase):
    def test_worker_busy_flag_clears_and_allows_second_request(self):
        client = _FastClient()
        worker = MODULE.AsyncKimodoWorker(client)
        request = {
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "state": np.zeros(64, dtype=np.float32),
            "history": np.zeros((2, 64), dtype=np.float32),
            "task": "test",
            "start_seq": 0,
            "end_seq": 1,
            "end_state_ts": 1.0,
            "episode_id": 0,
            "submitted_monotonic": time.monotonic(),
        }
        try:
            self.assertTrue(worker.submit(request))
            deadline = time.monotonic() + 2.0
            first = None
            while time.monotonic() < deadline:
                try:
                    first = worker.results.get_nowait()
                    break
                except Exception:
                    time.sleep(0.005)
            self.assertIsNotNone(first)
            self.assertTrue(first["ok"])
            deadline = time.monotonic() + 2.0
            while worker.busy.is_set() and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertFalse(worker.busy.is_set())
            request["start_seq"] = 2
            request["end_seq"] = 3
            self.assertTrue(worker.submit(request))
        finally:
            worker.close()
        self.assertEqual(client.calls, 2)

    def test_camera_and_state_metadata_preserve_source_identity(self):
        camera = MODULE._camera_metadata(
            {
                "sequence": 12,
                "server_monotonic_timestamp": 42.5,
                "receive_monotonic_timestamp": 42.51,
                "is_new": False,
            },
            42.52,
        )
        state = MODULE._state_metadata(
            {"index": 99, "state_monotonic_timestamp": 42.48},
            42.52,
            0,
        )
        self.assertEqual(camera["sequence"], 12)
        self.assertFalse(camera["is_new"])
        self.assertAlmostEqual(camera["source_ts"], 42.5)
        self.assertEqual(state["sequence"], 99)
        self.assertAlmostEqual(state["source_ts"], 42.48)


if __name__ == "__main__":
    unittest.main()
