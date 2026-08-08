import base64
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


class LeRobotVLAHttpClient:
    """Thin HTTP(S) client for remote LeRobot VLA inference."""

    def __init__(self, base_url: str, timeout_s: float = 5.0, verify_ssl: bool = False):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.verify_ssl = bool(verify_ssl)
        self._trace_path = os.environ.get("LEROBOT_VLA_TRACE_PATH", "").strip()
        self._trace_step_idx = 0

    def _append_trace(self, payload: dict[str, Any]) -> None:
        if not self._trace_path:
            return
        trace_path = Path(self._trace_path).expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _build_url(self, path: str) -> str:
        return urllib.parse.urljoin(self.base_url + "/", path.lstrip("/"))

    def _ssl_context(self):
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme != "https":
            return None
        if self.verify_ssl:
            return ssl.create_default_context()
        return ssl._create_unverified_context()

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._build_url(path),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        handlers: list[Any] = [urllib.request.ProxyHandler({})]
        ssl_context = self._ssl_context()
        if ssl_context is not None:
            handlers.append(urllib.request.HTTPSHandler(context=ssl_context))
        opener = urllib.request.build_opener(*handlers)
        try:
            with opener.open(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {self._build_url(path)}: {error_body}") from exc
        return json.loads(raw.decode("utf-8"))

    def infer_chunk(
        self,
        front_rgb: np.ndarray,
        observation_state: np.ndarray,
        robot_type: str,
        task: str | None = None,
        observation_state_history: np.ndarray | None = None,
    ) -> np.ndarray:
        rgb = np.asarray(front_rgb)
        state = np.asarray(observation_state, dtype=np.float32).reshape(-1)
        state_history = None
        if observation_state_history is not None:
            state_history = np.asarray(observation_state_history, dtype=np.float32)
            if state_history.ndim == 1:
                state_history = state_history.reshape(1, -1)
            if state_history.ndim != 2 or state_history.shape[1] != state.shape[0]:
                raise ValueError(
                    "observation_state_history must have shape [T, state_dim], "
                    f"got {state_history.shape} for state_dim={state.shape[0]}"
                )
            if state_history.shape[0] == 0:
                raise ValueError("observation_state_history must not be empty")
            if not np.isfinite(state_history).all():
                raise ValueError("observation_state_history contains NaN or Inf")
        task_name = None if task is None else str(task).strip()
        payload = {
            "observation": {
                "images": {
                    "front": {
                        "shape": list(rgb.shape),
                        "dtype": str(rgb.dtype),
                        "data_b64": base64.b64encode(rgb.tobytes()).decode("ascii"),
                    }
                },
                "state": state.tolist(),
            },
            "robot_type": robot_type,
            "return_chunk": True,
        }
        if state_history is not None:
            payload["observation"]["state_history"] = state_history.tolist()
        if task_name:
            payload["task"] = task_name
        response = self._post_json("/infer", payload)
        action_chunk = response.get("action_chunk")
        if action_chunk is None:
            action_chunk = response.get("latent64_chunk")
        if action_chunk is None:
            action_chunk = response.get("encoder_latent_chunk")
        if action_chunk is None:
            if "action" in response:
                action_chunk = [response["action"]]
            elif "latent64" in response:
                action_chunk = [response["latent64"]]
            elif "encoder_latent" in response:
                action_chunk = [response["encoder_latent"]]
            else:
                raise RuntimeError(
                    "Expected one of action_chunk/action/latent64_chunk/latent64/"
                    "encoder_latent_chunk/encoder_latent in server response"
                )
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk.reshape(1, -1)
        if action_chunk.ndim != 2:
            raise RuntimeError(f"Expected 2D action chunk from server, got shape {action_chunk.shape}")
        self._append_trace(
            {
                "event": "infer_chunk",
                "timestamp": time.time(),
                "step_idx": self._trace_step_idx,
                "robot_type": robot_type,
                "task": task_name,
                "front_rgb_shape": list(rgb.shape),
                "observation_state": state.tolist(),
                "observation_state_history_frames": (
                    0 if state_history is None else int(state_history.shape[0])
                ),
                "chunk_size": int(action_chunk.shape[0]),
                "first_action": action_chunk[0].tolist() if action_chunk.size > 0 else [],
            }
        )
        self._trace_step_idx += 1
        return action_chunk

    def infer(
        self,
        front_rgb: np.ndarray,
        observation_state: np.ndarray,
        robot_type: str,
        task: str | None = None,
        observation_state_history: np.ndarray | None = None,
    ) -> np.ndarray:
        action_chunk = self.infer_chunk(
            front_rgb=front_rgb,
            observation_state=observation_state,
            robot_type=robot_type,
            task=task,
            observation_state_history=observation_state_history,
        )
        return action_chunk[0]

    def reset(self, seed: int | None = None) -> None:
        payload: dict[str, Any] = {}
        if seed is not None:
            payload["seed"] = int(seed)
        self._post_json("/reset", payload)
        self._append_trace(
            {
                "event": "reset",
                "timestamp": time.time(),
                "step_idx": self._trace_step_idx,
                "seed": None if seed is None else int(seed),
            }
        )
        self._trace_step_idx = 0
