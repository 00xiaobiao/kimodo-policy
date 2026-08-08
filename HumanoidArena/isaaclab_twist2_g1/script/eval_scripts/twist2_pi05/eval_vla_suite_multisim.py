#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ISAACLAB_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = str(ISAACLAB_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval_vla_suite import (
    TASK_FOOTBALL_SINGLE,
    _derive_episode_seed,
    _episode_stem,
    _load_result_or_fallback,
    _result_key,
    _sanitize_label,
    _start_server,
    _wait_for_server_ready,
    _write_summary,
)


def _resolve_server_devices(args) -> list[str]:
    raw_values: list[str] = []
    raw_gpu_ids = str(getattr(args, "server_gpu_ids", "") or "").strip()
    if raw_gpu_ids:
        raw_values.extend(raw_gpu_ids.split(","))
    else:
        raw_server_device = str(getattr(args, "server_device", "") or "").strip()
        if raw_server_device:
            raw_values.extend(raw_server_device.split(","))

    devices: list[str] = []
    for raw in raw_values:
        token = raw.strip()
        if not token:
            continue
        if "-" in token and token.replace("-", "").isdigit():
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if end >= start else -1
            for gpu_idx in range(start, end + step, step):
                devices.append(f"cuda:{gpu_idx}")
            continue
        if token.isdigit():
            devices.append(f"cuda:{token}")
            continue
        devices.append(token)

    deduped: list[str] = []
    seen: set[str] = set()
    for device in devices:
        if device in seen:
            continue
        deduped.append(device)
        seen.add(device)
    if not deduped:
        raise ValueError(
            "No valid server devices resolved. Use --server_gpu_ids 0,1,2,3 or --server_device cuda:0."
        )
    return deduped


def _expand_server_devices(server_devices: list[str], count: int) -> list[str]:
    if count <= 0:
        return []
    expanded: list[str] = []
    while len(expanded) < count:
        for device in server_devices:
            expanded.append(device)
            if len(expanded) >= count:
                break
    return expanded


def _build_jobs(args) -> tuple[list[dict], dict[str, list[dict]]]:
    jobs: list[dict] = []
    jobs_by_model: dict[str, list[dict]] = {}
    episode_index = 0
    for model_path in args.model_paths:
        resolved_model_path = str(Path(model_path).expanduser().resolve())
        model_label = _sanitize_label(
            Path(resolved_model_path).parent.parent.name + "__" + Path(resolved_model_path).parent.name
        )
        model_jobs: list[dict] = []
        for seed in args.seeds:
            for repeat_idx in range(args.repeats_per_seed):
                job = {
                    "model_path": resolved_model_path,
                    "model_label": model_label,
                    "seed": int(seed),
                    "repeat_idx": int(repeat_idx),
                    "episode_seed": _derive_episode_seed(args.task, seed, repeat_idx),
                    "episode_index": int(episode_index),
                }
                jobs.append(job)
                model_jobs.append(job)
                episode_index += 1
        jobs_by_model[resolved_model_path] = model_jobs
    return jobs, jobs_by_model


def _load_existing_results(run_dir: Path) -> dict[tuple[str, int, int], dict]:
    episodes_dir = run_dir / "episodes"
    results_by_key: dict[tuple[str, int, int], dict] = {}
    if not episodes_dir.is_dir():
        return results_by_key

    for result_path in sorted(episodes_dir.glob("*.json")):
        try:
            row = json.loads(result_path.read_text())
        except Exception as exc:
            print(f"[eval_vla_suite_multisim] skip unreadable result file: {result_path} ({exc})")
            continue
        model_path = row.get("model_path")
        seed = row.get("seed")
        repeat_idx = row.get("repeat_idx")
        if model_path is None or seed is None or repeat_idx is None:
            continue
        results_by_key[_result_key(model_path, seed, repeat_idx)] = row
    return results_by_key


def _build_batch_entries(args, jobs: list[dict], run_dir: Path, model_path: str, model_label: str) -> list[dict]:
    episodes_dir = run_dir / "episodes"
    success_video_dir = run_dir / "videos" / "success"
    failure_video_dir = run_dir / "videos" / "failure"
    recordings_dir = run_dir / "recordings"
    for directory in (episodes_dir, success_video_dir, failure_video_dir, recordings_dir):
        directory.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for job in jobs:
        episode_stem = _episode_stem(model_label, job["seed"], job["repeat_idx"], job["episode_index"])
        entries.append(
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
    return entries


def _build_failure_result(args, server_urls: list[str], batch_log: Path, job: dict, error: str) -> dict:
    result_json = batch_log.parent.parent / "episodes" / (
        f"{_episode_stem(job['model_label'], job['seed'], job['repeat_idx'], job['episode_index'])}.json"
    )
    payload = {
        "task": args.task,
        "model_path": job["model_path"],
        "model_label": job["model_label"],
        "seed": int(job["seed"]),
        "repeat_idx": int(job["repeat_idx"]),
        "episode_seed": int(job["episode_seed"]),
        "episode_index": int(job["episode_index"]),
        "success": False,
        "failure_reason": "batch_error",
        "episode_steps": 0,
        "max_steps": int(args.max_steps),
        "final_reward": 0.0,
        "final_reward_scaled": 0.0,
        "max_reward": 0.0,
        "max_reward_scaled": 0.0,
        "video_path": "",
        "server_url": server_urls[0] if server_urls else "",
        "returncode": -1,
        "log_path": str(batch_log),
        "vla_trace_path": "",
        "error": error,
    }
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
    return payload


def _run_model_batch(args, run_dir: Path, model_path: str, model_label: str, jobs: list[dict], server_devices: list[str]):
    if not jobs:
        return []

    batch_num_envs = min(int(args.num_envs), len(jobs))
    slot_devices = _expand_server_devices(server_devices, batch_num_envs)
    logs_dir = run_dir / "logs"
    batch_dir = run_dir / "episode_batches"
    logs_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_entries = _build_batch_entries(args, jobs, run_dir, model_path, model_label)
    batch_spec_path = batch_dir / f"{model_label}__multisim_batch.json"
    batch_spec_path.write_text(json.dumps({"episodes": batch_entries}, ensure_ascii=True, indent=2) + "\n")

    server_urls_path = batch_dir / f"{model_label}__server_urls.json"
    batch_log = logs_dir / f"{model_label}__multisim.log"

    server_procs = []
    server_log_fps = []
    server_urls: list[str] = []
    try:
        for slot_idx, server_device in enumerate(slot_devices):
            server_port = int(args.server_port_base) + slot_idx
            server_url = f"{args.server_scheme}://{args.server_host}:{server_port}"
            server_log_path = logs_dir / f"server__multisim__slot_{slot_idx}__{model_label}.log"
            args.server_port = server_port
            args.server_device = server_device
            print(
                f"[eval_vla_suite_multisim] start slot={slot_idx} model={model_label} "
                f"server_device={server_device} port={server_port}"
            )
            server_proc, server_log_fp = _start_server(args, model_path, server_log_path)
            server_procs.append(server_proc)
            server_log_fps.append(server_log_fp)
            server_urls.append(server_url)

        for server_url in server_urls:
            _wait_for_server_ready(
                server_url,
                timeout_s=args.server_ready_timeout,
                verify_ssl=args.lerobot_server_verify_ssl,
            )

        server_urls_path.write_text(json.dumps(server_urls, ensure_ascii=True, indent=2) + "\n")

        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve().parent / "sim_eval_vla_multisim.py"),
            "--task",
            args.task,
            "--env_config_yaml",
            args.env_config_yaml,
            "--seed",
            str(jobs[0]["seed"]),
            "--episode_batch_json",
            str(batch_spec_path),
            "--server_urls_json",
            str(server_urls_path),
            "--num_envs",
            str(batch_num_envs),
            "--max_steps",
            str(args.max_steps),
            "--model_path",
            args.twist2_model_path,
            "--lerobot_server_timeout",
            str(args.lerobot_server_timeout),
            "--robot_type",
            args.robot_type,
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

        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
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
            result = _load_result_or_fallback(
                Path(entry["result_json"]),
                args=args,
                server_url=",".join(server_urls),
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
    except Exception as exc:
        print(f"[eval_vla_suite_multisim] model={model_label} failed: {exc}")
        return [_build_failure_result(args, server_urls, batch_log, job, str(exc)) for job in jobs]
    finally:
        for server_proc in server_procs:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=10.0)
            except Exception:
                try:
                    server_proc.kill()
                except Exception:
                    pass
        for server_log_fp in server_log_fps:
            try:
                server_log_fp.close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-sim football-single VLA evaluation suite")
    parser.add_argument("--task", type=str, default=TASK_FOOTBALL_SINGLE)
    parser.add_argument("--env_config_yaml", type=str, default="tasks/common_test_config/base_test/football_single_twist2_test.yaml")
    parser.add_argument("--model-path", dest="model_paths", action="append", required=True)
    parser.add_argument("--seed", dest="seeds", action="append", type=int, required=True)
    parser.add_argument("--repeats_per_seed", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--post_termination_record_steps", type=int, default=0)
    parser.add_argument("--robot_type", type=str, default="unitree_g1_refpose_v3_1")
    parser.add_argument("--twist2_model_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--isaac_device", type=str, default="cpu")
    parser.add_argument("--server_python", type=str, required=True)
    parser.add_argument("--server_script", type=str, required=True)
    parser.add_argument("--server_device", type=str, default="cuda:0")
    parser.add_argument("--server_gpu_ids", type=str, default="")
    parser.add_argument("--server_host", type=str, default="127.0.0.1")
    parser.add_argument("--server_scheme", type=str, default="http", choices=["http", "https"])
    parser.add_argument("--tls_cert_file", type=str, default="")
    parser.add_argument("--tls_key_file", type=str, default="")
    parser.add_argument("--lerobot_server_timeout", type=float, default=5.0)
    parser.add_argument("--lerobot_server_verify_ssl", action="store_true", default=False)
    parser.add_argument("--server_ready_timeout", type=float, default=60.0)
    parser.add_argument("--server_port_base", type=int, default=18443)
    args = parser.parse_args()

    if args.num_envs <= 0:
        raise ValueError("--num_envs must be >= 1")

    run_dir = Path(args.results_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    all_jobs, jobs_by_model = _build_jobs(args)
    if not all_jobs:
        raise ValueError("No evaluation jobs were generated")

    existing_results = _load_existing_results(run_dir)
    if existing_results:
        print(
            f"[eval_vla_suite_multisim] resume detected existing_results={len(existing_results)} "
            f"results_dir={run_dir}"
        )

    server_devices = _resolve_server_devices(args)
    print(
        f"[eval_vla_suite_multisim] total_jobs={len(all_jobs)} models={len(jobs_by_model)} "
        f"num_envs={args.num_envs} server_devices={server_devices} server_port_base={args.server_port_base}"
    )

    for model_index, (model_path, model_jobs) in enumerate(jobs_by_model.items(), start=1):
        pending_jobs = [
            job for job in model_jobs
            if _result_key(job["model_path"], job["seed"], job["repeat_idx"]) not in existing_results
        ]
        model_label = model_jobs[0]["model_label"]
        print(
            f"[eval_vla_suite_multisim] model={model_index}/{len(jobs_by_model)} label={model_label} "
            f"total_jobs={len(model_jobs)} pending={len(pending_jobs)}"
        )
        if not pending_jobs:
            continue
        batch_results = _run_model_batch(args, run_dir, model_path, model_label, pending_jobs, server_devices)
        for row in batch_results:
            existing_results[_result_key(row["model_path"], row["seed"], row["repeat_idx"])] = row
        partial_results = sorted(existing_results.values(), key=lambda row: int(row["episode_index"]))
        _write_summary(run_dir, partial_results)

    results = sorted(existing_results.values(), key=lambda row: int(row["episode_index"]))
    _write_summary(run_dir, results)
    total_successes = sum(int(bool(row.get("success"))) for row in results)
    total_episodes = len(results)
    overall_rate = total_successes / total_episodes if total_episodes else 0.0
    summary = {
        "total_jobs": len(all_jobs),
        "completed_results": len(results),
        "pending_results": max(0, len(all_jobs) - len(results)),
        "models_total": len(jobs_by_model),
        "models_completed": sum(
            1
            for model_jobs in jobs_by_model.values()
            if all(_result_key(job["model_path"], job["seed"], job["repeat_idx"]) in existing_results for job in model_jobs)
        ),
        "num_envs": int(args.num_envs),
        "server_devices": server_devices,
        "server_port_base": int(args.server_port_base),
        "overall_success_rate": overall_rate,
    }
    (run_dir / "multisim_config.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print(
        f"[eval_vla_suite_multisim] completed episodes={total_episodes} successes={total_successes} "
        f"success_rate={overall_rate:.4f} results_dir={run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
