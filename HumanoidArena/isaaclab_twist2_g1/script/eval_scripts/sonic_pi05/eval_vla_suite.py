#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ISAACLAB_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = str(ISAACLAB_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from action_provider.lerobot_vla_http_client import LeRobotVLAHttpClient


TASK_FOOTBALL_SINGLE = "Isaac-Move-Football-Single-G129-Dex3-Wholebody"
EPISODE_SEED_FORMULA = "sha256(task_name|group_seed|repeat_idx)_positive_int32"


def _normalize_cuda_launch(device: str) -> tuple[str | None, str]:
    token = str(device or "").strip()
    if not token:
        return None, token

    lowered = token.lower()
    if lowered == "cpu":
        return None, token
    if token.isdigit():
        return token, "cuda:0"
    if lowered == "cuda":
        return "0", "cuda:0"
    if lowered.startswith("cuda:"):
        gpu_idx = token.split(":", 1)[1]
        if gpu_idx.isdigit():
            return gpu_idx, "cuda:0"
    return None, token


def _sanitize_label(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def _derive_episode_seed(task_name: str, group_seed: int, repeat_idx: int) -> int:
    payload = f"{task_name}|{int(group_seed)}|{int(repeat_idx)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[-4:], "big", signed=False) & 0x7FFFFFFF


def _result_key(model_path: str, seed: int, repeat_idx: int) -> tuple[str, int, int]:
    return str(Path(model_path).expanduser().resolve()), int(seed), int(repeat_idx)


def _episode_stem(model_label: str, seed: int, repeat_idx: int, episode_index: int) -> str:
    return f"{model_label}__seed_{seed}__repeat_{repeat_idx}__episode_{episode_index}"


def _wait_for_server_ready(base_url: str, timeout_s: float, verify_ssl: bool) -> None:
    client = LeRobotVLAHttpClient(base_url=base_url, timeout_s=2.0, verify_ssl=verify_ssl)
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            client.reset()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"LeRobot server not ready at {base_url}: {last_error}")


def _build_server_env(args) -> tuple[dict[str, str], str]:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "CARB_APP_PATH", "EXP_PATH", "PYTHONHOME"):
        env.pop(key, None)

    server_python = Path(args.server_python).expanduser().resolve()
    current_path = env.get("PATH", "")
    env["PATH"] = str(server_python.parent) if not current_path else str(server_python.parent) + os.pathsep + current_path
    env["PYTHONNOUSERSITE"] = "1"

    visible_gpu, launch_device = _normalize_cuda_launch(args.server_device)
    if visible_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_gpu
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env, launch_device


def _build_sim_env(args) -> tuple[dict[str, str], str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    # Keep all GPUs visible for Vulkan/RTX interop; sim_eval_vla sets the renderer activeGpu.
    return env, str(args.isaac_device or "").strip()


def _start_server(args, model_path: str, log_path: Path):
    server_script = Path(args.server_script).expanduser().resolve()
    server_env, launch_device = _build_server_env(args)
    server_env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        args.server_python,
        "-u",
        str(server_script),
        "--policy-path",
        model_path,
        "--device",
        launch_device,
        "--host",
        args.server_host,
        "--port",
        str(args.server_port),
    ]
    if args.server_scheme == "https":
        if not args.tls_cert_file or not args.tls_key_file:
            raise ValueError("HTTPS server requires --tls_cert_file and --tls_key_file")
        cmd.extend(["--tls-cert-file", args.tls_cert_file, "--tls-key-file", args.tls_key_file])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    process = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        cwd=str(server_script.parent),
        env=server_env,
    )
    return process, log_fp


def _load_result_or_fallback(
    result_json: Path,
    *,
    args,
    server_url: str,
    model_path: str,
    model_label: str,
    seed: int,
    repeat_idx: int,
    episode_seed: int,
    episode_index: int,
    returncode: int,
) -> dict:
    if result_json.exists():
        result = json.loads(result_json.read_text())
    else:
        result = {
            "task": args.task,
            "model_path": model_path,
            "model_label": model_label,
            "seed": seed,
            "repeat_idx": repeat_idx,
            "episode_seed": episode_seed,
            "episode_index": episode_index,
            "success": False,
            "failure_reason": "process_error",
            "episode_steps": 0,
            "max_steps": args.max_steps,
            "final_reward": 0.0,
            "final_reward_scaled": 0.0,
            "max_reward": 0.0,
            "max_reward_scaled": 0.0,
            "video_path": "",
            "server_url": server_url,
            "returncode": returncode,
        }
        result_json.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n")
    result["repeat_idx"] = int(result.get("repeat_idx", repeat_idx))
    result["episode_seed"] = int(result.get("episode_seed", episode_seed))
    result["returncode"] = returncode
    return result


def _run_episode(
    args,
    server_url: str,
    model_path: str,
    model_label: str,
    seed: int,
    repeat_idx: int,
    episode_seed: int,
    episode_index: int,
    run_dir: Path,
):
    episodes_dir = run_dir / "episodes"
    logs_dir = run_dir / "logs"
    recordings_dir = run_dir / "recordings" / "vla_outputs"
    success_video_dir = run_dir / "videos" / "success"
    failure_video_dir = run_dir / "videos" / "failure"
    for directory in (episodes_dir, logs_dir, recordings_dir, success_video_dir, failure_video_dir):
        directory.mkdir(parents=True, exist_ok=True)

    episode_stem = _episode_stem(model_label, seed, repeat_idx, episode_index)
    result_json = episodes_dir / f"{episode_stem}.json"
    sim_log = logs_dir / f"{episode_stem}.log"
    vla_trace_path = recordings_dir / f"{episode_stem}.jsonl"
    trace_enabled = os.environ.get("LEROBOT_VLA_RECORD_OUTPUTS", "1").strip() not in {"0", "false", "False"}
    if trace_enabled and vla_trace_path.exists():
        vla_trace_path.unlink()

    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve().parent / "sim_eval_vla.py"),
        "--task",
        args.task,
        "--env_config_yaml",
        args.env_config_yaml,
        "--seed",
        str(seed),
        "--repeat_idx",
        str(repeat_idx),
        "--episode_seed",
        str(episode_seed),
        "--max_steps",
        str(args.max_steps),
        "--sonic_encoder_path",
        args.sonic_encoder_path,
        "--sonic_decoder_path",
        args.sonic_decoder_path,
        "--sonic_vla_root_rot6d_layout",
        args.sonic_vla_root_rot6d_layout,
        "--sonic_vla_root_max_delta_deg",
        str(args.sonic_vla_root_max_delta_deg),
        "--model_path",
        args.sonic_encoder_path,
        "--lerobot_server_url",
        server_url,
        "--lerobot_server_timeout",
        str(args.lerobot_server_timeout),
        "--robot_type",
        args.robot_type,
        "--result_json",
        str(result_json),
        "--success_video_dir",
        str(success_video_dir),
        "--failure_video_dir",
        str(failure_video_dir),
        "--video_fps",
        str(args.video_fps),
        "--post_termination_record_steps",
        str(args.post_termination_record_steps),
        "--record_video_every_n",
        str(args.record_video_every_n),
        "--step_log_every_n",
        str(args.step_log_every_n),
        "--episode_index",
        str(episode_index),
        "--model_label",
        model_label,
        "--eval_model_path",
        model_path,
        "--recording_save_dir",
        str(run_dir / "recordings"),
        "--device",
        args.isaac_device,
        "--enable_cameras",
    ]
    if args.lerobot_server_verify_ssl:
        cmd.append("--lerobot_server_verify_ssl")
    if args.headless:
        cmd.append("--headless")
    if args.verbose_startup:
        cmd.append("--verbose_startup")

    child_env, sim_launch_device = _build_sim_env(args)
    try:
        cmd[cmd.index("--device") + 1] = sim_launch_device
    except ValueError:
        pass
    if trace_enabled:
        child_env["LEROBOT_VLA_TRACE_PATH"] = str(vla_trace_path)
    else:
        child_env.pop("LEROBOT_VLA_TRACE_PATH", None)

    with open(sim_log, "w", encoding="utf-8", buffering=1) as log_fp:
        completed = subprocess.run(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            cwd=str(ISAACLAB_ROOT),
            env=child_env,
            check=False,
        )

    result = _load_result_or_fallback(
        result_json,
        args=args,
        server_url=server_url,
        model_path=model_path,
        model_label=model_label,
        seed=seed,
        repeat_idx=repeat_idx,
        episode_seed=episode_seed,
        episode_index=episode_index,
        returncode=completed.returncode,
    )
    result["log_path"] = str(sim_log)
    result["vla_trace_path"] = str(vla_trace_path) if trace_enabled else ""
    return result


def _run_episode_batch(
    args,
    server_url: str,
    model_path: str,
    model_label: str,
    seed: int,
    jobs: list[dict],
    run_dir: Path,
) -> list[dict]:
    episodes_dir = run_dir / "episodes"
    logs_dir = run_dir / "logs"
    success_video_dir = run_dir / "videos" / "success"
    failure_video_dir = run_dir / "videos" / "failure"
    recordings_dir = run_dir / "recordings"
    batch_dir = run_dir / "episode_batches"
    for directory in (episodes_dir, logs_dir, success_video_dir, failure_video_dir, recordings_dir, batch_dir):
        directory.mkdir(parents=True, exist_ok=True)

    batch_entries = []
    for job in jobs:
        episode_stem = _episode_stem(model_label, job["seed"], job["repeat_idx"], job["episode_index"])
        batch_entries.append(
            {
                "seed": int(job["seed"]),
                "repeat_idx": int(job["repeat_idx"]),
                "episode_seed": int(job["episode_seed"]),
                "episode_index": int(job["episode_index"]),
                "result_json": str(episodes_dir / f"{episode_stem}.json"),
                "success_video_dir": str(success_video_dir),
                "failure_video_dir": str(failure_video_dir),
                "recording_save_dir": str(recordings_dir),
                "model_label": model_label,
                "eval_model_path": model_path,
                "max_steps": int(args.max_steps),
                "video_fps": int(args.video_fps),
                "post_termination_record_steps": int(args.post_termination_record_steps),
            }
        )

    batch_spec_path = batch_dir / f"{model_label}__seed_{seed}__batch.json"
    batch_spec_path.write_text(json.dumps({"episodes": batch_entries}, ensure_ascii=True, indent=2) + "\n")
    batch_log = logs_dir / f"{model_label}__seed_{seed}__batch.log"

    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve().parent / "sim_eval_vla.py"),
        "--task",
        args.task,
        "--env_config_yaml",
        args.env_config_yaml,
        "--seed",
        str(seed),
        "--max_steps",
        str(args.max_steps),
        "--sonic_encoder_path",
        args.sonic_encoder_path,
        "--sonic_decoder_path",
        args.sonic_decoder_path,
        "--sonic_vla_root_rot6d_layout",
        args.sonic_vla_root_rot6d_layout,
        "--sonic_vla_root_max_delta_deg",
        str(args.sonic_vla_root_max_delta_deg),
        "--model_path",
        args.sonic_encoder_path,
        "--lerobot_server_url",
        server_url,
        "--lerobot_server_timeout",
        str(args.lerobot_server_timeout),
        "--robot_type",
        args.robot_type,
        "--record_video_every_n",
        str(args.record_video_every_n),
        "--step_log_every_n",
        str(args.step_log_every_n),
        "--recording_save_dir",
        str(recordings_dir),
        "--episode_batch_json",
        str(batch_spec_path),
        "--device",
        args.isaac_device,
        "--enable_cameras",
    ]
    if args.lerobot_server_verify_ssl:
        cmd.append("--lerobot_server_verify_ssl")
    if args.headless:
        cmd.append("--headless")
    if args.verbose_startup:
        cmd.append("--verbose_startup")

    child_env, sim_launch_device = _build_sim_env(args)
    try:
        cmd[cmd.index("--device") + 1] = sim_launch_device
    except ValueError:
        pass
    child_env.pop("LEROBOT_VLA_TRACE_PATH", None)

    with open(batch_log, "w", encoding="utf-8", buffering=1) as log_fp:
        completed = subprocess.run(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            cwd=str(ISAACLAB_ROOT),
            env=child_env,
            check=False,
        )

    results = []
    for entry in batch_entries:
        result_json = Path(entry["result_json"])
        result = _load_result_or_fallback(
            result_json,
            args=args,
            server_url=server_url,
            model_path=model_path,
            model_label=model_label,
            seed=entry["seed"],
            repeat_idx=entry["repeat_idx"],
            episode_seed=entry["episode_seed"],
            episode_index=entry["episode_index"],
            returncode=completed.returncode,
        )
        result["log_path"] = str(batch_log)
        result["vla_trace_path"] = ""
        results.append(result)
    return results


def _normalize_result_reason(row: dict) -> str:
    reason = str(row.get("failure_reason", "") or "").strip()
    if reason:
        return reason
    return "success" if bool(row.get("success")) else "unknown"


def _increment_reason_count(counts: dict, reason: str) -> None:
    counts[reason] = int(counts.get(reason, 0)) + 1


def _sorted_reason_counts(counts: dict) -> dict:
    return {key: counts[key] for key in sorted(counts)}


def _build_summary(results: list[dict]) -> dict:
    total_successes = sum(int(bool(row.get("success"))) for row in results)
    total_failures = max(0, len(results) - total_successes)
    result_reason_counts: dict[str, int] = {}
    failure_reason_counts: dict[str, int] = {}
    per_model: dict[str, dict] = {}
    for row in results:
        success = bool(row.get("success"))
        reason = _normalize_result_reason(row)
        _increment_reason_count(result_reason_counts, reason)
        if not success:
            _increment_reason_count(failure_reason_counts, reason)

        model_label = row["model_label"]
        stats = per_model.setdefault(
            model_label,
            {
                "model_path": row["model_path"],
                "episodes": 0,
                "successes": 0,
                "failures": 0,
                "result_reason_counts": {},
                "failure_reason_counts": {},
                "per_seed": {},
            },
        )
        stats["episodes"] += 1
        stats["successes"] += int(success)
        stats["failures"] += int(not success)
        _increment_reason_count(stats["result_reason_counts"], reason)
        if not success:
            _increment_reason_count(stats["failure_reason_counts"], reason)

        seed_key = str(int(row["seed"]))
        seed_stats = stats["per_seed"].setdefault(
            seed_key,
            {
                "episodes": 0,
                "successes": 0,
                "failures": 0,
                "result_reason_counts": {},
                "failure_reason_counts": {},
            },
        )
        seed_stats["episodes"] += 1
        seed_stats["successes"] += int(success)
        seed_stats["failures"] += int(not success)
        _increment_reason_count(seed_stats["result_reason_counts"], reason)
        if not success:
            _increment_reason_count(seed_stats["failure_reason_counts"], reason)

    for stats in per_model.values():
        stats["success_rate"] = stats["successes"] / stats["episodes"] if stats["episodes"] else 0.0
        stats["result_reason_counts"] = _sorted_reason_counts(stats["result_reason_counts"])
        stats["failure_reason_counts"] = _sorted_reason_counts(stats["failure_reason_counts"])
        seed_rates = []
        per_seed_success_rate = {}
        repeats_per_seed = {}
        sorted_per_seed = {}
        for seed_key, seed_stats in sorted(stats["per_seed"].items(), key=lambda item: int(item[0])):
            success_rate = seed_stats["successes"] / seed_stats["episodes"] if seed_stats["episodes"] else 0.0
            seed_stats["success_rate"] = success_rate
            seed_stats["result_reason_counts"] = _sorted_reason_counts(seed_stats["result_reason_counts"])
            seed_stats["failure_reason_counts"] = _sorted_reason_counts(seed_stats["failure_reason_counts"])
            seed_rates.append(success_rate)
            per_seed_success_rate[seed_key] = success_rate
            repeats_per_seed[seed_key] = int(seed_stats["episodes"])
            sorted_per_seed[seed_key] = seed_stats
        stats["per_seed"] = sorted_per_seed
        stats["per_seed_success_rate"] = per_seed_success_rate
        stats["repeats_per_seed"] = repeats_per_seed
        stats["seed_success_rate_mean"] = statistics.mean(seed_rates) if seed_rates else 0.0
        stats["seed_success_rate_std"] = statistics.pstdev(seed_rates) if len(seed_rates) > 1 else 0.0

    return {
        "total_episodes": len(results),
        "total_successes": total_successes,
        "total_failures": total_failures,
        "overall_success_rate": total_successes / len(results) if results else 0.0,
        "result_reason_counts": _sorted_reason_counts(result_reason_counts),
        "failure_reason_counts": _sorted_reason_counts(failure_reason_counts),
        "episode_seed_formula": EPISODE_SEED_FORMULA,
        "per_model": per_model,
    }


def _clean_final_model_name(model_label: str) -> str:
    cleaned = str(model_label or "")
    if cleaned.startswith("0424_new__"):
        cleaned = cleaned[len("0424_new__"):]
    cleaned = cleaned.replace("__last", "_last")
    return cleaned


def _write_final_csv(run_dir: Path, results: list[dict], summary: dict) -> None:
    final_csv_path = run_dir / "final.csv"
    success_steps_by_model: dict[str, list[int]] = {}
    for row in results:
        if not bool(row.get("success")):
            continue
        model_label = str(row.get("model_label", ""))
        try:
            episode_steps = int(row.get("episode_steps", 0))
        except (TypeError, ValueError):
            continue
        success_steps_by_model.setdefault(model_label, []).append(episode_steps)

    fieldnames = [
        "model",
        "success_rate",
        "seed_success_rate_std",
        "avg_success_steps",
        "fall_in_failures",
    ]
    with open(final_csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for model_label, stats in summary.get("per_model", {}).items():
            episodes = int(stats.get("episodes", 0) or 0)
            successes = int(stats.get("successes", 0) or 0)
            failures = int(stats.get("failures", 0) or 0)
            success_rate = float(stats.get("success_rate", 0.0) or 0.0)
            seed_std = float(stats.get("seed_success_rate_std", 0.0) or 0.0)
            fall_count = int(stats.get("failure_reason_counts", {}).get("fall", 0) or 0)
            fall_rate = fall_count / failures if failures else 0.0
            success_steps = success_steps_by_model.get(str(model_label), [])
            avg_success_steps = sum(success_steps) / len(success_steps) if success_steps else None
            writer.writerow(
                {
                    "model": _clean_final_model_name(str(model_label)),
                    "success_rate": f"{successes}/{episodes} ({success_rate * 100:.1f}%)",
                    "seed_success_rate_std": f"{seed_std * 100:.1f}%",
                    "avg_success_steps": "" if avg_success_steps is None else f"{avg_success_steps:.1f}",
                    "fall_in_failures": f"{fall_count}/{failures} ({fall_rate * 100:.1f}%)",
                }
            )


def _write_summary(run_dir: Path, results: list[dict]) -> None:
    jsonl_path = run_dir / "summary.jsonl"
    csv_path = run_dir / "summary.csv"
    summary_json_path = run_dir / "summary.json"

    with open(jsonl_path, "w", encoding="utf-8") as fp:
        for row in results:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")

    fieldnames = [
        "task",
        "model_path",
        "model_label",
        "seed",
        "repeat_idx",
        "episode_seed",
        "episode_index",
        "episode_object_seed",
        "episode_object_seed_source",
        "success",
        "failure_reason",
        "episode_steps",
        "max_steps",
        "final_reward",
        "final_reward_scaled",
        "max_reward",
        "max_reward_scaled",
        "video_path",
        "log_path",
        "vla_trace_path",
        "server_url",
        "returncode",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    summary = _build_summary(results)
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    _write_final_csv(run_dir, results, summary)


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-parallel football-single VLA evaluation suite")
    parser.add_argument("--task", type=str, default=TASK_FOOTBALL_SINGLE)
    parser.add_argument("--env_config_yaml", type=str, default="tasks/common_test_config/base_test/football_single_sonic_test.yaml")
    parser.add_argument("--model-path", dest="model_paths", action="append", required=True)
    parser.add_argument("--seed", dest="seeds", action="append", type=int, required=True)
    parser.add_argument("--repeats_per_seed", type=int, default=1)
    parser.add_argument("--persistent_sim", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--post_termination_record_steps", type=int, default=0)
    parser.add_argument("--record_video_every_n", type=int, default=1)
    parser.add_argument("--step_log_every_n", type=int, default=0)
    parser.add_argument("--verbose_startup", action="store_true", default=False)
    parser.add_argument("--robot_type", type=str, default="unitree_g1_refpose_v3_1")
    parser.add_argument("--sonic_encoder_path", type=str, required=True)
    parser.add_argument("--sonic_decoder_path", type=str, required=True)
    parser.add_argument(
        "--sonic_vla_root_rot6d_layout",
        type=str,
        default="auto",
        choices=["auto", "row", "col"],
    )
    parser.add_argument(
        "--sonic_vla_root_max_delta_deg",
        type=float,
        default=26.0,
        help="Clamp max root orientation delta per step (degrees). <=0 to disable.",
    )
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--isaac_device", type=str, default="cpu")
    parser.add_argument("--server_python", type=str, required=True)
    parser.add_argument(
        "--server_script",
        type=str,
        default=str(ISAACLAB_ROOT.parent / "lerobot" / "serve_lerobot_vla_http.py"),
    )
    parser.add_argument("--server_device", type=str, default="cuda:0")
    parser.add_argument("--server_host", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=8443)
    parser.add_argument("--server_scheme", type=str, default="http", choices=["http", "https"])
    parser.add_argument("--tls_cert_file", type=str, default="")
    parser.add_argument("--tls_key_file", type=str, default="")
    parser.add_argument("--lerobot_server_timeout", type=float, default=5.0)
    parser.add_argument("--lerobot_server_verify_ssl", action="store_true", default=False)
    parser.add_argument("--server_ready_timeout", type=float, default=60.0)
    args = parser.parse_args()

    run_dir = Path(args.results_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    server_url = f"{args.server_scheme}://{args.server_host}:{args.server_port}"

    try:
        for model_path in args.model_paths:
            resolved_model_path = str(Path(model_path).expanduser().resolve())
            model_label = _sanitize_label(Path(resolved_model_path).parent.parent.name + "__" + Path(resolved_model_path).parent.name)
            server_log_path = run_dir / "logs" / f"server__{model_label}.log"
            print(f"[eval_vla_suite] starting server for model={resolved_model_path}")
            server_proc, server_log_fp = _start_server(args, resolved_model_path, server_log_path)
            try:
                _wait_for_server_ready(
                    server_url,
                    timeout_s=args.server_ready_timeout,
                    verify_ssl=args.lerobot_server_verify_ssl,
                )
                episode_index = 0
                for seed in args.seeds:
                    seed_jobs = []
                    for repeat_idx in range(args.repeats_per_seed):
                        seed_jobs.append(
                            {
                                "seed": int(seed),
                                "repeat_idx": int(repeat_idx),
                                "episode_seed": _derive_episode_seed(args.task, seed, repeat_idx),
                                "episode_index": int(episode_index),
                            }
                        )
                        episode_index += 1
                    if args.persistent_sim:
                        print(
                            f"[eval_vla_suite] model={model_label} seed={seed} "
                            f"persistent_batch repeats={len(seed_jobs)}"
                        )
                        results.extend(
                            _run_episode_batch(
                                args=args,
                                server_url=server_url,
                                model_path=resolved_model_path,
                                model_label=model_label,
                                seed=int(seed),
                                jobs=seed_jobs,
                                run_dir=run_dir,
                            )
                        )
                        continue

                    for job in seed_jobs:
                        print(
                            f"[eval_vla_suite] model={model_label} seed={seed} "
                            f"repeat={job['repeat_idx'] + 1}/{args.repeats_per_seed}"
                        )
                        results.append(
                            _run_episode(
                                args=args,
                                server_url=server_url,
                                model_path=resolved_model_path,
                                model_label=model_label,
                                seed=job["seed"],
                                repeat_idx=job["repeat_idx"],
                                episode_seed=job["episode_seed"],
                                episode_index=job["episode_index"],
                                run_dir=run_dir,
                            )
                        )
            finally:
                try:
                    server_proc.terminate()
                    server_proc.wait(timeout=10.0)
                except Exception:
                    server_proc.kill()
                server_log_fp.close()
    finally:
        _write_summary(run_dir, results)

    total_successes = sum(int(bool(row.get("success"))) for row in results)
    total_episodes = len(results)
    overall_rate = total_successes / total_episodes if total_episodes else 0.0
    print(
        f"[eval_vla_suite] completed episodes={total_episodes} successes={total_successes} "
        f"success_rate={overall_rate:.4f} results_dir={run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
