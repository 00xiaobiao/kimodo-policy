from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from image_server.shared_memory_utils import DEFAULT_SHM_NAME, SHM_NAME_ENV_VAR, resolve_shm_name
from tools.data_tools.rerecord_parallel_utils import (
    append_text_log,
    build_worker_runtime_config,
    chunk_round_robin,
    format_rerecord_summary_entry,
    move_tree_contents,
    reset_text_log,
)


def test_resolve_shm_name_prefers_explicit_then_env(monkeypatch) -> None:
    monkeypatch.delenv(SHM_NAME_ENV_VAR, raising=False)
    assert resolve_shm_name() == DEFAULT_SHM_NAME

    monkeypatch.setenv(SHM_NAME_ENV_VAR, "env_shm_name")
    assert resolve_shm_name() == "env_shm_name"
    assert resolve_shm_name("explicit_shm_name") == "explicit_shm_name"


def test_build_worker_runtime_config_allocates_unique_ports_and_shm() -> None:
    cfg0 = build_worker_runtime_config(
        worker_index=0,
        image_port_base=5600,
        image_port_stride=10,
        shm_prefix="custom_shm",
        process_id=4321,
        session_token="deadbeef",
    )
    cfg1 = build_worker_runtime_config(
        worker_index=1,
        image_port_base=5600,
        image_port_stride=10,
        shm_prefix="custom_shm",
        process_id=4321,
        session_token="deadbeef",
    )

    assert cfg0.image_zmq_port == 5600
    assert cfg0.world_camera_port == 5601
    assert cfg0.left_wrist_camera_port == 5602
    assert cfg0.right_wrist_camera_port == 5603
    assert cfg1.image_zmq_port == 5610
    assert cfg1.left_wrist_camera_port == 5612
    assert cfg0.runtime_tag == "w00_p4321_deadbeef"
    assert cfg1.runtime_tag == "w01_p4321_deadbeef"
    assert cfg0.shm_name == "custom_shm_w00_p4321_deadbeef"
    assert cfg1.shm_name == "custom_shm_w01_p4321_deadbeef"
    assert cfg0.image_redis_key_prefix.endswith(cfg0.runtime_tag)
    assert cfg1.image_dds_topic.endswith(cfg1.runtime_tag)


def test_chunk_round_robin_balances_items() -> None:
    chunks = chunk_round_robin(list(range(7)), 3)
    assert chunks == [[0, 3, 6], [1, 4], [2, 5]]


def test_move_tree_contents_merges_nested_outputs(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    (src_dir / "videos").mkdir(parents=True)
    (dst_dir / "videos").mkdir(parents=True)
    (src_dir / "episode.npz").write_bytes(b"npz")
    (src_dir / "videos" / "front.mp4").write_bytes(b"front")
    (dst_dir / "videos" / "existing.mp4").write_bytes(b"existing")

    move_tree_contents(src_dir, dst_dir)

    assert (dst_dir / "episode.npz").read_bytes() == b"npz"
    assert (dst_dir / "videos" / "front.mp4").read_bytes() == b"front"
    assert (dst_dir / "videos" / "existing.mp4").read_bytes() == b"existing"
    assert list(src_dir.iterdir()) == []


def test_format_rerecord_summary_entry_includes_reward_and_paths() -> None:
    entry = format_rerecord_summary_entry(
        index=2,
        total_jobs=17,
        source_file="/tmp/source.npz",
        output_dir="/tmp/out",
        log_path="/tmp/out/logs/source.log",
        status="success",
        rerecorded_npz="/tmp/out/result.npz",
        final_reward=0.02,
        max_reward=0.04,
        any_success=True,
    )

    assert "[2/17] rerecording /tmp/source.npz" in entry
    assert "  output_dir=/tmp/out" in entry
    assert "  log=/tmp/out/logs/source.log" in entry
    assert (
        "  success -> /tmp/out/result.npz final_reward=0.0200 max_reward=0.0400 any_success=true"
        in entry
    )


def test_reset_and_append_text_log(tmp_path: Path) -> None:
    log_path = tmp_path / "rerecord_conversion.log"
    reset_text_log(log_path, header_lines=["header"])
    append_text_log(log_path, "line1")
    append_text_log(log_path, "line2\n")

    assert log_path.read_text(encoding="utf-8") == "header\nline1\nline2\n"
