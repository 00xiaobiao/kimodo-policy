import json
from pathlib import Path
import sys
import unittest

import numpy as np


# ``real-world`` is intentionally not a Python package name because of the
# hyphen, so add the deployment bridge directory explicitly for this test.
DEPLOY_DIR = Path(__file__).resolve().parents[1] / "real-world" / "deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from kimodo_sonic_bridge import (  # noqa: E402
    BridgeSafetyError,
    KimodoActionRuntime,
    SONIC_HEADER_SIZE,
    pack_pose_v1,
    state_msg_to_kimodo_state,
)


def _decode_protocol_v1(packet: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """Independent Python implementation of SONIC's fixed-header wire parser."""
    if not packet.startswith(b"pose"):
        raise AssertionError("missing pose topic prefix")
    wire = packet[len(b"pose") :]
    if len(wire) < SONIC_HEADER_SIZE:
        raise AssertionError("packet is shorter than the fixed SONIC header")
    header_bytes = wire[:SONIC_HEADER_SIZE].split(b"\x00", 1)[0]
    header = json.loads(header_bytes.decode("utf-8"))
    payload = wire[SONIC_HEADER_SIZE:]
    dtype_map = {
        "f32": (np.dtype("<f4"), 4),
        "f64": (np.dtype("<f8"), 8),
        "i64": (np.dtype("<i8"), 8),
        "bool": (np.dtype("<?"), 1),
    }
    decoded = {}
    offset = 0
    for field in header["fields"]:
        dtype, itemsize = dtype_map[field["dtype"]]
        shape = tuple(field["shape"])
        count = int(np.prod(shape))
        end = offset + count * itemsize
        if end > len(payload):
            raise AssertionError(f"field {field['name']} exceeds payload")
        decoded[field["name"]] = np.frombuffer(payload[offset:end], dtype=dtype).reshape(shape).copy()
        offset = end
    if offset != len(payload):
        raise AssertionError(f"unexpected trailing payload bytes: {len(payload) - offset}")
    return header, decoded


class KimodoSonicBridgeTest(unittest.TestCase):
    def test_state_mapping_uses_source_timestamp_and_relative_heading(self):
        q_mj = np.arange(29, dtype=np.float32) * 0.01
        dq_mj = np.arange(29, dtype=np.float32) * 0.02
        state, q, dq, root, timestamp, init = state_msg_to_kimodo_state(
            {
                "body_q": q_mj,
                "body_dq": dq_mj,
                "base_quat": [1, 0, 0, 0],
                "state_monotonic_timestamp": 123.456,
            },
            initial_root_quat_wxyz=None,
            now_monotonic=999.0,
        )
        mapping = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
        self.assertEqual(state.shape, (64,))
        np.testing.assert_allclose(q, q_mj[mapping])
        np.testing.assert_allclose(dq, dq_mj[mapping])
        np.testing.assert_allclose(state[:6], [1, 0, 0, 1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(init, [1, 0, 0, 0])
        self.assertAlmostEqual(timestamp, 123.456)

    def test_protocol_v1_fixed_4096_header_and_isaaclab_order_round_trip(self):
        # This ramp is deliberately asymmetric: any accidental MuJoCo
        # permutation at the bridge boundary is immediately visible.
        q = np.arange(29, dtype=np.float32) * 0.01
        dq = -q
        action = np.zeros(40, dtype=np.float32)
        action[2] = 0.85
        action[3:9] = [1, 0, 0, 1, 0, 0]
        action[9:38] = q
        action[38:40] = [1, 0]
        runtime = KimodoActionRuntime()
        reference = runtime.decode(action, current_root_quat_wxyz=[1, 0, 0, 0])
        packet = pack_pose_v1(
            joint_pos_isaaclab=reference["joint_pos"],
            joint_vel_isaaclab=dq,
            body_quat_wxyz=reference["body_quat"],
            root_position=reference["root_position"],
            frame_index=7,
            timestamp_monotonic=10.0,
        )
        header, decoded = _decode_protocol_v1(packet)
        self.assertEqual(len(packet[4 : 4 + SONIC_HEADER_SIZE]), SONIC_HEADER_SIZE)
        self.assertEqual(header["v"], 1)
        np.testing.assert_allclose(decoded["joint_pos"][0], q)
        np.testing.assert_allclose(decoded["joint_vel"][0], dq)
        np.testing.assert_array_equal(decoded["frame_index"], [7])
        np.testing.assert_allclose(decoded["root_position"][0], [0, 0, 0.85])

    def test_protocol_v1_does_not_permute_canonical_action_q29(self):
        action = np.zeros(40, dtype=np.float32)
        action[2] = 0.85
        action[3:9] = [1, 0, 0, 1, 0, 0]
        action[9:38] = np.arange(29, dtype=np.float32) * 0.05 + 0.125
        reference = KimodoActionRuntime().decode(
            action, current_root_quat_wxyz=[1, 0, 0, 0]
        )
        packet = pack_pose_v1(
            joint_pos_isaaclab=reference["joint_pos"],
            joint_vel_isaaclab=np.zeros(29, dtype=np.float32),
            body_quat_wxyz=reference["body_quat"],
            root_position=reference["root_position"],
            frame_index=0,
        )
        _, decoded = _decode_protocol_v1(packet)
        np.testing.assert_allclose(decoded["joint_pos"][0], action[9:38])

    def test_action_runtime_rejects_velocity_and_root_jumps(self):
        runtime = KimodoActionRuntime()
        base = np.zeros(40, dtype=np.float32)
        base[2] = 0.85
        base[3:9] = [1, 0, 0, 1, 0, 0]
        runtime.decode(base, current_root_quat_wxyz=[1, 0, 0, 0])

        jump = base.copy()
        jump[0] = 0.2
        with self.assertRaises(BridgeSafetyError):
            runtime.decode(jump, current_root_quat_wxyz=[1, 0, 0, 0])

        runtime = KimodoActionRuntime()
        runtime.decode(base, current_root_quat_wxyz=[1, 0, 0, 0])
        velocity_jump = base.copy()
        velocity_jump[9] = 0.75
        with self.assertRaises(BridgeSafetyError):
            runtime.decode(velocity_jump, current_root_quat_wxyz=[1, 0, 0, 0])

    def test_action_runtime_rejects_invalid_root_height(self):
        action = np.zeros(40, dtype=np.float32)
        action[3:9] = [1, 0, 0, 1, 0, 0]
        action[9:38] = 0.0
        for root_z in (0.0, 1.3):
            action[2] = root_z
            with self.assertRaises(BridgeSafetyError):
                KimodoActionRuntime().decode(
                    action, current_root_quat_wxyz=[1, 0, 0, 0]
                )


if __name__ == "__main__":
    unittest.main()
