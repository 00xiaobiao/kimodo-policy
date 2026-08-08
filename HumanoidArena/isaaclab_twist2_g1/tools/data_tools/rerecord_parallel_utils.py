from __future__ import annotations

import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeVar


DEFAULT_IMAGE_REDIS_KEY_PREFIX = "isaac_image"
DEFAULT_IMAGE_DDS_TOPIC = "rt/isaac_image"
DEFAULT_IMAGE_XROBOT_PORT_BASE = 12345
DEFAULT_SHM_PREFIX = "isaac_multi_image_shm"
DEFAULT_RERECORD_SUMMARY_FILENAME = "rerecord_conversion.log"
SHM_NAME_ENV_VAR = "ISAAC_IMAGE_SHM_NAME"

T = TypeVar("T")


@dataclass(frozen=True)
class WorkerRuntimeConfig:
    worker_index: int
    runtime_tag: str
    shm_name: str
    image_zmq_port: int
    world_camera_port: int
    left_wrist_camera_port: int
    right_wrist_camera_port: int
    image_xrobot_port: int
    image_redis_key_prefix: str
    image_dds_topic: str


def build_worker_runtime_config(
    *,
    worker_index: int,
    image_port_base: int,
    image_port_stride: int,
    image_xrobot_port_base: int = DEFAULT_IMAGE_XROBOT_PORT_BASE,
    image_xrobot_port_stride: int = 10,
    shm_prefix: str = DEFAULT_SHM_PREFIX,
    image_redis_key_prefix: str = DEFAULT_IMAGE_REDIS_KEY_PREFIX,
    image_dds_topic: str = DEFAULT_IMAGE_DDS_TOPIC,
    process_id: int | None = None,
    session_token: str | None = None,
) -> WorkerRuntimeConfig:
    process_id = os.getpid() if process_id is None else int(process_id)
    token = session_token or secrets.token_hex(4)
    port_base = int(image_port_base) + int(worker_index) * int(image_port_stride)
    xrobot_port = int(image_xrobot_port_base) + int(worker_index) * int(image_xrobot_port_stride)
    runtime_tag = f"w{int(worker_index):02d}_p{process_id}_{token}"
    shm_base = shm_prefix or DEFAULT_SHM_PREFIX
    redis_base = image_redis_key_prefix or DEFAULT_IMAGE_REDIS_KEY_PREFIX
    dds_base = image_dds_topic or DEFAULT_IMAGE_DDS_TOPIC
    return WorkerRuntimeConfig(
        worker_index=int(worker_index),
        runtime_tag=runtime_tag,
        shm_name=f"{shm_base}_{runtime_tag}",
        image_zmq_port=port_base,
        world_camera_port=port_base + 1,
        left_wrist_camera_port=port_base + 2,
        right_wrist_camera_port=port_base + 3,
        image_xrobot_port=xrobot_port,
        image_redis_key_prefix=f"{redis_base}_{runtime_tag}",
        image_dds_topic=f"{dds_base}_{runtime_tag}",
    )


def build_worker_env(base_env: dict[str, str], runtime_config: WorkerRuntimeConfig) -> dict[str, str]:
    env = dict(base_env)
    env[SHM_NAME_ENV_VAR] = runtime_config.shm_name
    return env


def append_image_runtime_args(command: list[str], runtime_config: WorkerRuntimeConfig) -> list[str]:
    command.extend(
        [
            "--image_zmq_port",
            str(runtime_config.image_zmq_port),
            "--world_camera_port",
            str(runtime_config.world_camera_port),
            "--left_wrist_camera_port",
            str(runtime_config.left_wrist_camera_port),
            "--right_wrist_camera_port",
            str(runtime_config.right_wrist_camera_port),
            "--image_xrobot_port",
            str(runtime_config.image_xrobot_port),
            "--image_redis_key_prefix",
            runtime_config.image_redis_key_prefix,
            "--image_dds_topic",
            runtime_config.image_dds_topic,
        ]
    )
    return command


def chunk_round_robin(items: Sequence[T], chunk_count: int) -> list[list[T]]:
    chunk_count = max(1, int(chunk_count))
    chunks: list[list[T]] = [[] for _ in range(chunk_count)]
    for index, item in enumerate(items):
        chunks[index % chunk_count].append(item)
    return chunks


def allocate_job_output_dir(output_root: Path, source_stem: str, runtime_tag: str) -> Path:
    safe_stem = source_stem.replace(os.sep, "_")
    return (output_root / ".tmp_rerecord" / f"{safe_stem}_{runtime_tag}").resolve()


def move_tree_contents(src_dir: Path, dst_dir: Path) -> None:
    src_dir = src_dir.resolve()
    dst_dir = dst_dir.resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in src_dir.iterdir():
        destination = dst_dir / child.name
        if child.is_dir():
            move_tree_contents(child, destination)
            child.rmdir()
            continue
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        shutil.move(str(child), str(destination))


def remove_dir_if_empty(path: Path, stop_at: Path | None = None) -> None:
    current = path.resolve()
    stop_resolved = stop_at.resolve() if stop_at is not None else None
    while current.exists():
        if stop_resolved is not None and current == stop_resolved:
            break
        try:
            current.rmdir()
        except OSError:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent


def reset_text_log(path: Path, header_lines: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in header_lines or ():
            handle.write(str(line).rstrip("\n"))
            handle.write("\n")


def append_text_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if text and not text.endswith("\n"):
            handle.write("\n")


def format_rerecord_summary_entry(
    *,
    index: int,
    total_jobs: int,
    source_file: Path | str,
    output_dir: Path | str,
    log_path: Path | str,
    status: str,
    rerecorded_npz: Path | str | None = None,
    final_reward: float | None = None,
    max_reward: float | None = None,
    any_success: bool | None = None,
    return_code: int | None = None,
) -> str:
    lines = [
        f"[{int(index)}/{int(total_jobs)}] rerecording {source_file}",
        f"  output_dir={output_dir}",
        f"  log={log_path}",
    ]
    if status == "success":
        effective_max_reward = max_reward if max_reward is not None else final_reward
        effective_any_success = any_success
        if effective_any_success is None and effective_max_reward is not None:
            effective_any_success = bool(float(effective_max_reward) > 1e-6)

        final_reward_text = (
            f"{float(final_reward):.4f}"
            if final_reward is not None
            else "<missing>"
        )
        max_reward_text = (
            f"{float(effective_max_reward):.4f}"
            if effective_max_reward is not None
            else "<missing>"
        )
        any_success_text = (
            str(bool(effective_any_success)).lower()
            if effective_any_success is not None
            else "<missing>"
        )
        lines.append(
            f"  success -> {rerecorded_npz}"
            f" final_reward={final_reward_text}"
            f" max_reward={max_reward_text}"
            f" any_success={any_success_text}"
        )
    else:
        failure_line = f"  {status}"
        if return_code is not None:
            failure_line += f" return_code={int(return_code)}"
        if rerecorded_npz:
            failure_line += f" output={rerecorded_npz}"
        lines.append(failure_line)
    return "\n".join(lines) + "\n"
