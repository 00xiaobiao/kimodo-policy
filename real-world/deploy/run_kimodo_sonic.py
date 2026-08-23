#!/usr/bin/env python3
from __future__ import annotations

"""Run Kimodo as the high-level policy and SONIC as the G1 low-level tracker.

The script is deliberately conservative:

* it reads measured state from SONIC's ``g1_debug`` ZMQ stream;
* it reads ``ego_view`` from SONIC's camera server;
* it calls the existing Kimodo HTTP server asynchronously;
* it publishes only Protocol-v1 G1 reference motion to SONIC;
* it does not auto-start the robot unless ``--auto-start`` is supplied.

Before using a real robot, run ``--dry-run`` or ``--no-publish`` and verify
the generated state/action packet with the C++ deploy process in a safe mode.
"""

import argparse
import base64
import json
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
SONIC_ROOT = THIS_DIR.parent / "GR00T-WholeBodyControl"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SONIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SONIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from kimodo_sonic_bridge import (  # noqa: E402
    BridgeSafetyError,
    KimodoActionRuntime,
    SafetyLimits,
    pack_pose_v1,
    state_msg_to_kimodo_state,
)


class KimodoHttpClient:
    def __init__(self, base_url: str, timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Kimodo HTTP {exc.code}: {detail}") from exc

    def health(self) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/health", method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Kimodo health HTTP {exc.code}: {detail}") from exc
        if result.get("status") != "ok":
            raise RuntimeError(f"Kimodo health check failed: {result}")
        return result

    def reset(self) -> None:
        result = self._post("/reset", {})
        if result.get("status") != "ok":
            raise RuntimeError(f"Kimodo reset failed: {result}")

    def infer_chunk(self, image: np.ndarray, state: np.ndarray, state_history: np.ndarray, task: str) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"expected uint8 HWC RGB image, got {image.shape} {image.dtype}")
        state = np.asarray(state, dtype=np.float32).reshape(64)
        history = np.asarray(state_history, dtype=np.float32)
        if history.ndim != 2 or history.shape[1] != 64 or history.shape[0] == 0:
            raise ValueError(f"expected state history [T,64], got {history.shape}")
        payload = {
            "task": str(task),
            "observation": {
                "images": {
                    "front": {
                        "shape": list(image.shape),
                        "dtype": str(image.dtype),
                        "data_b64": base64.b64encode(np.ascontiguousarray(image).tobytes()).decode("ascii"),
                    }
                },
                "state": state.tolist(),
                "state_history": history.tolist(),
            },
        }
        response = self._post("/infer", payload)
        action = response.get("action_chunk")
        if action is None:
            raise RuntimeError(f"Kimodo response has no action_chunk: {response.keys()}")
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if action.ndim != 2 or action.shape[1] != 40 or action.shape[0] == 0:
            raise RuntimeError(f"Kimodo returned invalid action chunk shape {action.shape}")
        if not np.isfinite(action).all():
            raise RuntimeError("Kimodo returned NaN/Inf action")
        return action


class AsyncKimodoWorker:
    """One outstanding inference request, keeping the 50 Hz loop non-blocking."""

    def __init__(self, client: KimodoHttpClient):
        self.client = client
        self.requests: queue.Queue = queue.Queue(maxsize=1)
        self.results: queue.Queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.busy = threading.Event()
        self._submit_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="kimodo-inference", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                request = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                return
            try:
                action = self.client.infer_chunk(
                    request["image"], request["state"], request["history"], request["task"]
                )
                result = {
                    "ok": True,
                    "action": action,
                    "end_seq": request["end_seq"],
                    "start_seq": request["start_seq"],
                    "end_state_ts": request.get("end_state_ts"),
                    "episode_id": request.get("episode_id", 0),
                    "submitted_monotonic": request.get("submitted_monotonic"),
                }
            except Exception as exc:  # propagate to the control loop, never silently swallow
                result = {
                    "ok": False,
                    "error": exc,
                    "end_seq": request["end_seq"],
                    "start_seq": request["start_seq"],
                    "end_state_ts": request.get("end_state_ts"),
                    "episode_id": request.get("episode_id", 0),
                    "submitted_monotonic": request.get("submitted_monotonic"),
                }
            try:
                self.results.put_nowait(result)
            except queue.Full:
                try:
                    self.results.get_nowait()
                except queue.Empty:
                    pass
                self.results.put_nowait(result)
            self.busy.clear()

    def submit(self, request: dict) -> bool:
        # Set busy before enqueueing under the same lock.  This prevents a very
        # fast request from completing and clearing busy before submit() has
        # returned, which used to leave the worker permanently marked busy.
        with self._submit_lock:
            if self.busy.is_set():
                return False
            self.busy.set()
            try:
                self.requests.put_nowait(request)
                return True
            except queue.Full:
                self.busy.clear()
                return False

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=2.0)


def _load_sonic_runtime():
    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
    from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

    return ComposedCameraClientSensor, ZMQStateSubscriber, build_command_message


def _hand_targets(hand_binary: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Convert Kimodo binary hand commands to SONIC's optional 7D hand fields."""
    try:
        from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import G1GripperInverseKinematicsSolver

        left_close = G1GripperInverseKinematicsSolver(side="left")._get_middle_close_q_desired().astype(np.float32)
        right_close = G1GripperInverseKinematicsSolver(side="right")._get_middle_close_q_desired().astype(np.float32)
        left = left_close if float(hand_binary[0]) >= 0.5 else np.zeros(7, dtype=np.float32)
        right = right_close if float(hand_binary[1]) >= 0.5 else np.zeros(7, dtype=np.float32)
        return left, right
    except Exception as exc:
        raise RuntimeError(
            "SONIC hand solver is unavailable; refusing to silently drop hand action fields"
        ) from exc


def _scalar(value, default=None) -> float | int | None:
    if value is None:
        return default
    arr = np.asarray(value).reshape(-1)
    if arr.size != 1:
        return default
    try:
        result = float(arr[0])
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _integer(value, default: int | None = None) -> int | None:
    result = _scalar(value, default)
    if result is None:
        return default
    return int(result)


def _camera_metadata(image_msg: dict, receive_now: float) -> dict:
    """Extract the camera identity/clock fields without changing image bytes."""
    return {
        "sequence": _integer(image_msg.get("sequence"), None),
        # server_monotonic_timestamp is generated immediately before the
        # camera server publishes the message and is comparable with the
        # g1_debug monotonic clock when both processes run on the robot.
        "source_ts": _scalar(
            image_msg.get("server_monotonic_timestamp"),
            _scalar(image_msg.get("receive_monotonic_timestamp"), receive_now),
        ),
        "receive_ts": _scalar(image_msg.get("receive_monotonic_timestamp"), receive_now),
        "is_new": bool(image_msg.get("is_new", True)),
    }


def _state_metadata(state_msg: dict, receive_now: float, fallback_seq: int) -> dict:
    source_seq = _integer(state_msg.get("index"), None)
    has_source_seq = source_seq is not None
    if source_seq is None:
        source_seq = fallback_seq
    source_ts = _scalar(state_msg.get("state_monotonic_timestamp"), None)
    has_source_ts = source_ts is not None and source_ts > 0.0
    if source_ts is None or source_ts <= 0.0:
        source_ts = receive_now
    return {
        "sequence": int(source_seq),
        "source_ts": float(source_ts),
        "receive_ts": receive_now,
        "has_source_seq": has_source_seq,
        "has_source_ts": has_source_ts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument("--task", required=True)
    parser.add_argument("--server-timeout", type=float, default=5.0)
    parser.add_argument("--control-rate", type=float, default=50.0)
    parser.add_argument("--state-zmq-host", default="127.0.0.1")
    parser.add_argument("--state-zmq-port", type=int, default=5557)
    parser.add_argument("--camera-host", default="127.0.0.1")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--action-zmq-host", default="127.0.0.1")
    parser.add_argument("--action-zmq-port", type=int, default=5556)
    parser.add_argument("--history-frames", type=int, default=100)
    parser.add_argument("--request-when-queue-below", type=int, default=12)
    parser.add_argument("--max-root-delta-deg", type=float, default=26.0)
    parser.add_argument("--max-joint-step", type=float, default=0.75)
    parser.add_argument("--max-sensor-skew-ms", type=float, default=80.0)
    parser.add_argument("--max-camera-stale-s", type=float, default=0.25)
    parser.add_argument("--max-state-gap-s", type=float, default=0.08)
    parser.add_argument("--max-result-state-lag-s", type=float, default=0.25)
    parser.add_argument("--max-inference-age-s", type=float, default=0.50)
    parser.add_argument(
        "--auto-stop-on-timeout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="send SONIC stop on inference timeout or bridge shutdown (default: true)",
    )
    parser.add_argument("--auto-start", action="store_true", help="send command planner=false,start=true")
    parser.add_argument("--no-publish", action="store_true", help="run sensors/inference but do not publish to SONIC")
    parser.add_argument("--dry-run", action="store_true", help="use synthetic sensors; requires a reachable Kimodo server")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until Ctrl-C")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _synthetic_state(step: int) -> tuple[dict, np.ndarray]:
    return {
        "body_q": np.zeros(29, dtype=np.float32),
        "body_dq": np.zeros(29, dtype=np.float32),
        "base_quat": np.array([1, 0, 0, 0], dtype=np.float32),
        "index": step,
    }, np.full((96, 128, 3), 127, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    if args.control_rate <= 0:
        raise ValueError("--control-rate must be positive")
    period = 1.0 / max(float(args.control_rate), 1.0)
    client = KimodoHttpClient(args.server_url, args.server_timeout)
    health = client.health()
    server_fps = _scalar(health.get("control_fps"), None)
    if server_fps is None:
        raise RuntimeError(
            "Kimodo /health did not report control_fps; refusing to run with an unverified rate"
        )
    if abs(float(server_fps) - float(args.control_rate)) > 1e-3:
        raise RuntimeError(
            f"control-rate mismatch: bridge={args.control_rate:g}Hz, "
            f"Kimodo server={float(server_fps):g}Hz"
        )
    client.reset()
    worker = AsyncKimodoWorker(client)
    runtime = KimodoActionRuntime(
        limits=SafetyLimits(
            max_joint_step=float(args.max_joint_step),
            max_reference_age_s=min(0.25, float(args.max_inference_age_s)),
        ),
        max_root_delta_deg=float(args.max_root_delta_deg),
        reference_rate_hz=float(args.control_rate),
    )
    state_history: deque[tuple[int, np.ndarray, float]] = deque(
        maxlen=max(int(args.history_frames), 2)
    )
    fallback_state_seq = 0
    last_submitted_seq = -1
    last_reference: dict | None = None
    last_inference_completed = None
    last_state_meta: dict | None = None
    last_camera_meta: dict | None = None
    initial_root_quat = None
    previous_raw_state = None
    action_queue: deque[np.ndarray] = deque()
    publish_frame = 0
    episode_id = 0
    stop_sent = False
    streamed_motion_started = False
    zmq_socket = None
    build_command_message = None
    camera = state_subscriber = None

    def send_stop(reason: str) -> None:
        nonlocal stop_sent
        if (
            stop_sent
            or not args.auto_stop_on_timeout
            or zmq_socket is None
            or args.no_publish
            or build_command_message is None
        ):
            return
        try:
            zmq_socket.send(build_command_message(start=False, stop=True, planner=False))
            stop_sent = True
            print(f"[deploy] sent SONIC stop: {reason}", flush=True)
        except Exception as exc:
            print(f"[deploy] failed to send SONIC stop ({reason}): {exc}", flush=True)

    def reset_episode(reason: str) -> None:
        nonlocal episode_id, last_submitted_seq, last_reference
        nonlocal last_inference_completed, last_state_meta, last_camera_meta
        nonlocal initial_root_quat, previous_raw_state, streamed_motion_started, stop_sent
        episode_id += 1
        if streamed_motion_started:
            send_stop(f"episode reset: {reason}")
            streamed_motion_started = False
            # A new episode may be started after a fresh valid pose. The
            # publish frame itself intentionally remains monotonic across
            # episodes because SONIC's sliding-window merger rejects rollback.
            stop_sent = False
        client.reset()
        runtime.reset()
        state_history.clear()
        action_queue.clear()
        last_submitted_seq = -1
        last_reference = None
        last_inference_completed = None
        last_state_meta = None
        last_camera_meta = None
        initial_root_quat = None
        previous_raw_state = None
        print(f"[deploy] local episode reset ({reason}), episode_id={episode_id}", flush=True)

    if not args.dry_run:
        import zmq

        CameraClient, StateSubscriber, build_command_message = _load_sonic_runtime()
        camera = CameraClient(server_ip=args.camera_host, port=args.camera_port)
        state_subscriber = StateSubscriber(host=args.state_zmq_host, port=args.state_zmq_port)
        zmq_socket = zmq.Context.instance().socket(zmq.PUB)
        zmq_socket.bind(f"tcp://{args.action_zmq_host}:{args.action_zmq_port}")
        time.sleep(0.2)
        print(f"[deploy] publishing SONIC pose on tcp://{args.action_zmq_host}:{args.action_zmq_port}", flush=True)
    else:
        print("[deploy] dry-run enabled; no SONIC sensor or publisher is opened", flush=True)

    print(
        "[deploy] Kimodo bridge ready; verified control rate "
        f"{float(server_fps):g}Hz",
        flush=True,
    )
    steps = 0
    try:
        while not args.max_steps or steps < args.max_steps:
            tick_start = time.monotonic()
            receive_now = time.monotonic()
            if args.dry_run:
                state_msg, image = _synthetic_state(steps)
                image_meta = {
                    "sequence": steps,
                    "source_ts": receive_now,
                    "receive_ts": receive_now,
                    "is_new": True,
                }
            else:
                image_msg = camera.read()
                state_msg = state_subscriber.get_msg(clear=False)
                image = None if image_msg is None else image_msg.get("images", {}).get("ego_view")
                if state_msg is None or image is None:
                    time.sleep(min(period, 0.01))
                    continue
                image_meta = _camera_metadata(image_msg, receive_now)
                image = np.asarray(image)
                if image.ndim == 4:
                    image = image[-1]
                if image.ndim != 3 or image.shape[-1] not in (3, 4):
                    print(f"[deploy] rejecting camera frame with shape {image.shape}", flush=True)
                    time.sleep(period)
                    continue
                if image.shape[-1] == 4:
                    image = image[..., :3]
                if image.dtype != np.uint8:
                    image = np.clip(image, 0, 255).astype(np.uint8)

            if last_camera_meta is not None:
                camera_rollback = (
                    image_meta["source_ts"] < last_camera_meta["source_ts"]
                    or (
                        image_meta["sequence"] is not None
                        and last_camera_meta["sequence"] is not None
                        and image_meta["sequence"] < last_camera_meta["sequence"]
                    )
                )
                if camera_rollback:
                    reset_episode("camera sequence/timestamp rollback")
            if (
                last_camera_meta is None
                or image_meta["source_ts"] > last_camera_meta["source_ts"]
                or image_meta["sequence"] != last_camera_meta["sequence"]
            ):
                last_camera_meta = image_meta

            state_meta = _state_metadata(state_msg, receive_now, fallback_state_seq)
            fallback_state_seq = state_meta["sequence"] + 1
            if last_state_meta is not None:
                if (
                    state_meta["sequence"] < last_state_meta["sequence"]
                    or state_meta["source_ts"] < last_state_meta["source_ts"]
                ):
                    reset_episode("state sequence/timestamp rollback")
                elif state_meta["source_ts"] - last_state_meta["source_ts"] > args.max_state_gap_s:
                    reset_episode("state source gap")

            try:
                state, q, dq, root_quat, state_ts, initial_root_quat = state_msg_to_kimodo_state(
                    state_msg,
                    initial_root_quat_wxyz=initial_root_quat,
                    previous_state=previous_raw_state,
                )
            except BridgeSafetyError as exc:
                print(f"[deploy] rejecting robot state: {exc}", flush=True)
                send_stop(f"invalid robot state: {exc}")
                raise
            previous_raw_state = (np.asarray(state_msg["body_q"], dtype=np.float32).reshape(29), state_ts)
            is_new_state = (
                last_state_meta is None
                or state_meta["sequence"] > last_state_meta["sequence"]
                or (
                    not state_meta["has_source_seq"]
                    and state_meta["source_ts"] > last_state_meta["source_ts"]
                )
            )
            if is_new_state:
                state_history.append((state_meta["sequence"], state.copy(), state_ts))
                last_state_meta = state_meta
            if len(state_history) == 1:
                # Startup padding matches the server's short-history handling and
                # avoids feeding an all-zero proprioceptive history.
                first = state_history[0][1].copy()
                for _ in range(state_history.maxlen - 1):
                    state_history.appendleft(
                        (state_history[0][0] - 1, first.copy(), state_history[0][2])
                    )

            while True:
                try:
                    result = worker.results.get_nowait()
                except queue.Empty:
                    break
                if result.get("episode_id", episode_id) != episode_id:
                    print("[deploy] discarded result from an old episode", flush=True)
                    continue
                result_state_lag = 0.0
                if result.get("end_state_ts") is not None and last_state_meta is not None:
                    result_state_lag = max(
                        0.0,
                        last_state_meta["source_ts"] - float(result["end_state_ts"]),
                    )
                result_age = (
                    time.monotonic() - float(result["submitted_monotonic"])
                    if result.get("submitted_monotonic") is not None
                    else 0.0
                )
                if result["ok"] and (
                    result_state_lag > args.max_result_state_lag_s
                    or result_age > args.max_inference_age_s
                ):
                    print(
                        "[deploy] discarded stale Kimodo result: "
                        f"state_lag={result_state_lag:.3f}s age={result_age:.3f}s",
                        flush=True,
                    )
                    last_submitted_seq = min(last_submitted_seq, int(result["start_seq"]) - 1)
                    continue
                if result["ok"]:
                    action_queue.extend(np.asarray(result["action"], dtype=np.float32))
                    last_inference_completed = time.monotonic()
                    if args.verbose:
                        print(
                            f"[deploy] Kimodo chunk received: {result['action'].shape}, "
                            f"seq={result['end_seq']} age={result_age:.3f}s",
                            flush=True,
                        )
                else:
                    print(f"[deploy] Kimodo inference failed: {result['error']}", flush=True)
                    last_submitted_seq = min(last_submitted_seq, int(result["start_seq"]) - 1)

            reference_updated = False
            if action_queue:
                try:
                    last_reference = runtime.decode(action_queue.popleft(), current_root_quat_wxyz=root_quat)
                    reference_updated = True
                except BridgeSafetyError as exc:
                    print(f"[deploy] rejecting Kimodo action and holding last safe reference: {exc}", flush=True)
                    action_queue.clear()
                    send_stop(f"invalid Kimodo action: {exc}")
                    raise
            if last_reference is not None:
                try:
                    left_hand, right_hand = _hand_targets(last_reference["hand_binary"])
                except Exception as exc:
                    send_stop(f"hand target failure: {exc}")
                    raise
                publish_reference = dict(last_reference)
                # A chunk boundary can leave the bridge waiting for the next
                # Kimodo result.  Holding q is safe, but publishing a stale
                # non-zero reference velocity is not: make a held frame
                # kinematically consistent.
                inference_stale = (
                    last_inference_completed is not None
                    and time.monotonic() - last_inference_completed > runtime.limits.max_reference_age_s
                )
                if not reference_updated or inference_stale:
                    publish_reference["joint_vel"] = np.zeros((29,), dtype=np.float32)
                if inference_stale and args.verbose:
                    print("[deploy] Kimodo result is stale; holding last safe q with zero reference velocity", flush=True)
                packet = pack_pose_v1(
                    joint_pos_isaaclab=publish_reference["joint_pos"],
                    joint_vel_isaaclab=publish_reference["joint_vel"],
                    body_quat_wxyz=publish_reference["body_quat"],
                    root_position=publish_reference["root_position"],
                    frame_index=publish_frame,
                    timestamp_monotonic=time.monotonic(),
                    left_hand_joints=left_hand,
                    right_hand_joints=right_hand,
                )
                if zmq_socket is not None and not args.no_publish:
                    zmq_socket.send(packet)
                    if args.auto_start and not streamed_motion_started:
                        zmq_socket.send(build_command_message(start=True, stop=False, planner=False))
                        streamed_motion_started = True
                        print("[deploy] sent streamed-motion start after first valid pose", flush=True)
                publish_frame += 1

                if (
                    last_inference_completed is not None
                    and time.monotonic() - last_inference_completed > args.max_inference_age_s
                ):
                    send_stop("Kimodo inference timeout")
                    raise RuntimeError("Kimodo inference exceeded the configured safety timeout")

            newest_seq = state_history[-1][0] if state_history else -1
            newest_state_ts = state_history[-1][2] if state_history else state_ts
            pending = [item for item in state_history if item[0] > last_submitted_seq]
            camera_age = max(0.0, time.monotonic() - float(image_meta["source_ts"]))
            sensor_skew = abs(float(state_meta["source_ts"]) - float(image_meta["source_ts"]))
            camera_usable = bool(image_meta["is_new"]) and camera_age <= args.max_camera_stale_s
            if sensor_skew * 1000.0 > args.max_sensor_skew_ms:
                camera_usable = False
                if args.verbose:
                    print(
                        f"[deploy] withholding inference for sensor skew {sensor_skew*1000:.1f}ms",
                        flush=True,
                    )
            if (
                not worker.busy.is_set()
                and len(action_queue) <= int(args.request_when_queue_below)
                and pending
                and camera_usable
            ):
                segment = np.stack([item[1] for item in pending], axis=0).astype(np.float32)
                request = {
                    "image": image.copy(),
                    "state": state.copy(),
                    "history": segment,
                    "task": args.task,
                    "start_seq": pending[0][0],
                    "end_seq": newest_seq,
                    "end_state_ts": newest_state_ts,
                    "episode_id": episode_id,
                    "submitted_monotonic": time.monotonic(),
                }
                if worker.submit(request):
                    last_submitted_seq = newest_seq

            steps += 1
            elapsed = time.monotonic() - tick_start
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        send_stop("Ctrl-C")
        print("[deploy] stopped by Ctrl-C", flush=True)
    finally:
        send_stop("bridge shutdown")
        worker.close()
        if zmq_socket is not None:
            zmq_socket.close(0)
        if state_subscriber is not None:
            state_subscriber.close()


if __name__ == "__main__":
    main()
