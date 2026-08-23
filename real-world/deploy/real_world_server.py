#!/usr/bin/env python3
"""Kimodo inference HTTP server for the real-robot SONIC bridge.

This is the deployment entry point.  It deliberately has its own HTTP
handler and command line interface; ``evaluation/humanoidarena_server.py``
continues to serve HumanoidArena and is not modified or started by this
process.  The model runtime is loaded lazily so the protocol can be tested
without a GPU/checkpoint.

The bridge sends one RGB camera frame and a 64-D state history at every
replanning request.  A successful request returns a finite ``[N, 40]``
Kimodo action chunk.  The SONIC bridge, rather than this server, performs the
last safety checks and translates that semantic chunk into Protocol v1.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_TEXT_EMBEDDING_CACHE = PROJECT_ROOT / "data/cache/HumanoidArena"
MAX_REQUEST_BYTES = 128 * 1024 * 1024


def _dtype_from_wire(value: Any) -> np.dtype:
    try:
        dtype = np.dtype(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid image dtype: {value!r}") from exc
    if dtype.kind not in {"u", "i", "f"} or dtype.itemsize > 4:
        raise ValueError(f"unsupported image dtype: {dtype}")
    return dtype


def _validate_image_payload(image: Any) -> None:
    if not isinstance(image, dict):
        raise ValueError("observation.images.front must be an object")
    shape = image.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError("image shape must be [height, width, channels]")
    try:
        shape_tuple = tuple(int(v) for v in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("image shape contains a non-integer") from exc
    h, w, channels = shape_tuple
    if h <= 0 or w <= 0 or channels not in (3, 4):
        raise ValueError(f"unsupported image shape: {shape_tuple}")
    # Keep malformed/hostile payloads out of torch before base64 decoding.
    dtype = _dtype_from_wire(image.get("dtype"))
    encoded = image.get("data_b64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("image data_b64 is missing")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("image data_b64 is not valid base64") from exc
    expected = int(np.prod(shape_tuple)) * dtype.itemsize
    if len(raw) != expected:
        raise ValueError(
            f"image byte length mismatch: expected {expected}, got {len(raw)}"
        )
    if dtype != np.dtype(np.uint8):
        raise ValueError(
            f"real-world camera input must be uint8 RGB, got wire dtype {dtype}"
        )


def _validate_infer_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("/infer payload must be a JSON object")
    task = str(payload.get("task", "")).strip()
    if not task:
        raise ValueError("/infer requires a non-empty task")
    observation = payload.get("observation")
    if not isinstance(observation, dict):
        raise ValueError("/infer requires observation")
    images = observation.get("images")
    if not isinstance(images, dict) or "front" not in images:
        raise ValueError("observation.images.front is required")
    _validate_image_payload(images["front"])

    state = np.asarray(observation.get("state"), dtype=np.float32)
    if state.shape != (64,) or not np.isfinite(state).all():
        raise ValueError("observation.state must be finite shape [64]")
    history = np.asarray(observation.get("state_history"), dtype=np.float32)
    if history.ndim != 2 or history.shape[1:] != (64,) or history.shape[0] == 0:
        raise ValueError("observation.state_history must have shape [T, 64]")
    if not np.isfinite(history).all():
        raise ValueError("observation.state_history contains NaN or Inf")


class KimodoRealWorldRuntime:
    """Real-world facade around the checkpoint inference implementation.

    The implementation is imported only when the server is constructed.  It
    shares the tested Kimodo preprocessing/action codec with the simulator,
    but this process has no Arena environment, Arena HTTP handler, or sim
    state.  Keeping this composition here makes the simulator endpoint
    completely independent and gives the robot bridge a stable contract.
    """

    def __init__(self, args: argparse.Namespace):
        # Lazy import is important for protocol/unit tests and for machines
        # where the deployment process is started before CUDA is initialized.
        from evaluation.humanoidarena_server import KimodoHumanoidArenaRuntime

        self.control_fps = float(args.control_fps)
        if self.control_fps <= 0 or not np.isfinite(self.control_fps):
            raise ValueError("control_fps must be finite and positive")
        self._backend = KimodoHumanoidArenaRuntime(args)
        backend_fps = float(self._backend.control_fps)
        if abs(backend_fps - self.control_fps) > 1e-6:
            raise ValueError(
                f"runtime control FPS mismatch: server={self.control_fps}, backend={backend_fps}"
            )

    def reset(self, seed: Any = None) -> None:
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError) as exc:
                raise ValueError("reset seed must be an integer") from exc
        self._backend.reset(seed)

    def infer(self, payload: dict[str, Any]) -> np.ndarray:
        _validate_infer_payload(payload)
        action = np.asarray(self._backend.infer(payload), dtype=np.float32)
        if action.ndim != 2 or action.shape[0] <= 0 or action.shape[1] != 40:
            raise RuntimeError(f"Kimodo returned invalid action shape {action.shape}")
        if not np.isfinite(action).all():
            raise RuntimeError("Kimodo returned NaN/Inf action")
        return action


class RequestHandler(BaseHTTPRequestHandler):
    runtime: KimodoRealWorldRuntime | Any = None
    max_request_bytes = MAX_REQUEST_BYTES

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        if length <= 0:
            raise ValueError("request body is empty")
        if length > self.max_request_bytes:
            raise ValueError(
                f"request body exceeds {self.max_request_bytes} bytes"
            )
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc.msg}") from exc

    def do_GET(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path in {"/", "/health"}:
            runtime = self.runtime
            fps = getattr(runtime, "control_fps", None)
            self._send_json(
                200,
                {
                    "status": "ok",
                    "mode": "real_world",
                    "control_fps": float(fps) if fps is not None else None,
                    "state_dim": 64,
                    "action_dim": 40,
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        try:
            payload = self._read_json()
            if path == "/reset":
                if not isinstance(payload, dict):
                    raise ValueError("/reset payload must be a JSON object")
                self.runtime.reset(payload.get("seed"))
                self._send_json(200, {"status": "ok"})
                return
            if path == "/infer":
                _validate_infer_payload(payload)
                action = self.runtime.infer(payload)
                self._send_json(200, {"action_chunk": action.tolist()})
                return
            self._send_json(404, {"error": "not found"})
        except (ValueError, KeyError, TypeError) as exc:
            self._send_json(400, {"error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:  # never leak a traceback into the wire protocol
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--policy-path", dest="checkpoint", required=True)
    parser.add_argument(
        "--text-embedding-cache",
        default=os.environ.get(
            "KIMODO_TEXT_EMBEDDING_CACHE", str(DEFAULT_TEXT_EMBEDDING_CACHE)
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp32"),
        default=os.environ.get("KIMODO_DTYPE", "fp32"),
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=int(os.environ.get("KIMODO_DIFFUSION_STEPS", "10")),
    )
    parser.add_argument(
        "--execution-frames",
        type=int,
        default=int(os.environ.get("KIMODO_EXECUTION_FRAMES", "0")),
    )
    parser.add_argument(
        "--rtc", type=int, choices=(0, 1), default=int(os.environ.get("KIMODO_RTC", "1"))
    )
    parser.add_argument(
        "--rtc-overlap-frames",
        type=int,
        default=int(os.environ.get("KIMODO_RTC_OVERLAP_FRAMES", "12")),
    )
    parser.add_argument(
        "--rtc-frozen-frames",
        type=int,
        default=int(os.environ.get("KIMODO_RTC_FROZEN_FRAMES", "1")),
    )
    parser.add_argument(
        "--rtc-ramp-power",
        type=float,
        default=float(os.environ.get("KIMODO_RTC_RAMP_POWER", "1.0")),
    )
    parser.add_argument("--control-fps", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= int(args.port) <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    RequestHandler.runtime = KimodoRealWorldRuntime(args)
    server = ThreadingHTTPServer((args.host, int(args.port)), RequestHandler)
    print(
        f"Serving Kimodo real-world server on http://{args.host}:{args.port} "
        f"(control_fps={args.control_fps:g})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[real-world-server] stopped", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
