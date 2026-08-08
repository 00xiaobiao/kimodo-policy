#!/usr/bin/env python3

import argparse
import collections
import concurrent.futures
import fcntl
import json
import socket
import time
from pathlib import Path

from eval_vla_suite import (
    TASK_FOOTBALL_SINGLE,
    _derive_episode_seed,
    _result_key,
    _run_episode,
    _run_episode_batch,
    _sanitize_label,
    _start_server,
    _wait_for_server_ready,
    _write_summary,
)




def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, int(port)))
            return True
        except OSError:
            return False


def _select_available_server_port(host: str, start_port: int, max_port: int) -> int:
    start_port = int(start_port)
    max_port = int(max_port)
    if start_port > max_port:
        raise RuntimeError(f"Invalid server port range: start_port={start_port} > max_port={max_port}")
    for port in range(start_port, max_port + 1):
        if _is_port_available(host, port):
            return port
    raise RuntimeError(f"No free server port found from {start_port} to {max_port}")


def _server_start_lock_path(run_dir: Path) -> Path:
    return Path("/tmp/humanoidarena_vla_server_port_selection.lock")

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

    deduped_devices: list[str] = []
    seen_devices: set[str] = set()
    for device in devices:
        if device in seen_devices:
            continue
        deduped_devices.append(device)
        seen_devices.add(device)

    if not deduped_devices:
        raise ValueError(
            "No valid server devices resolved. Use --server_gpu_ids 0,1,2,3 or --server_device cuda:0."
        )
    return deduped_devices


def _build_worker_slots(server_devices: list[str], workers_per_device: int) -> list[dict]:
    slots: list[dict] = []
    slot_id = 0
    for device_worker_index in range(workers_per_device):
        for server_device in server_devices:
            slots.append(
                {
                    "task_id": slot_id,
                    "server_device": server_device,
                    "device_worker_index": device_worker_index,
                }
            )
            slot_id += 1
    return slots


def _resolve_worker_isaac_device(base_isaac_device: str, server_device: str) -> str:
    token = str(base_isaac_device or "").strip()
    if not token:
        return server_device

    lowered = token.lower()
    if lowered == "cpu":
        return token
    if lowered.startswith("cuda") or token.isdigit():
        return server_device
    return token


def _build_model_label(model_path: str) -> str:
    resolved_path = Path(model_path).expanduser().resolve()
    checkpoint_name = resolved_path.parent.name
    experiment_name = resolved_path.parent.parent.parent.name
    return _sanitize_label(f"{experiment_name}__{checkpoint_name}")


def _build_jobs(args) -> tuple[list[dict], dict[str, str]]:
    jobs = []
    model_labels = {}
    episode_index = 0
    for model_path in args.model_paths:
        resolved_model_path = str(Path(model_path).expanduser().resolve())
        model_label = _build_model_label(resolved_model_path)
        model_labels[resolved_model_path] = model_label
        for seed in args.seeds:
            for repeat_idx in range(args.repeats_per_seed):
                jobs.append(
                    {
                        "model_path": resolved_model_path,
                        "model_label": model_label,
                        "seed": int(seed),
                        "repeat_idx": int(repeat_idx),
                        "episode_seed": _derive_episode_seed(args.task, seed, repeat_idx),
                        "episode_index": int(episode_index),
                    }
                )
                episode_index += 1
    return jobs, model_labels


def _load_existing_results(run_dir: Path) -> dict[tuple[str, int, int], dict]:
    episodes_dir = run_dir / "episodes"
    results_by_key: dict[tuple[str, int, int], dict] = {}
    if not episodes_dir.is_dir():
        return results_by_key

    for result_path in sorted(episodes_dir.glob('*.json')):
        try:
            row = json.loads(result_path.read_text())
        except Exception as exc:
            print(f"[eval_vla_suite_parallel] skip unreadable result file: {result_path} ({exc})")
            continue
        model_path = row.get('model_path')
        seed = row.get('seed')
        repeat_idx = row.get('repeat_idx')
        if model_path is None or seed is None or repeat_idx is None:
            continue
        results_by_key[_result_key(model_path, seed, repeat_idx)] = row
    return results_by_key


def _merged_summary_results(run_dir: Path, completed_results: dict[tuple[str, int, int], dict]) -> list[dict]:
    # sim_eval_vla writes per-episode JSON immediately, while persistent workers return only
    # after a full batch. Merge disk state so summary.* reflects already-finished episodes.
    merged_results = _load_existing_results(run_dir)
    merged_results.update(completed_results)
    return sorted(merged_results.values(), key=lambda row: int(row.get('episode_index', -1)))


def _build_task_specs_for_model_jobs(model_jobs: list[dict], *, persistent_sim: bool, max_tasks: int) -> list[dict]:
    if not model_jobs:
        return []

    task_specs = []
    task_spec_id = 0
    if persistent_sim:
        jobs_by_seed: dict[int, list[dict]] = {}
        for job in model_jobs:
            jobs_by_seed.setdefault(int(job["seed"]), []).append(job)
        for seed, jobs in sorted(jobs_by_seed.items()):
            task_specs.append(
                {
                    "task_spec_id": task_spec_id,
                    "model_path": jobs[0]["model_path"],
                    "model_label": jobs[0]["model_label"],
                    "seed": int(seed),
                    "jobs": sorted(jobs, key=lambda row: int(row["repeat_idx"])),
                }
            )
            task_spec_id += 1
    else:
        chunk_count = min(len(model_jobs), max_tasks)
        chunks = [model_jobs[idx::chunk_count] for idx in range(chunk_count)]
        for chunk in chunks:
            if not chunk:
                continue
            task_specs.append(
                {
                    "task_spec_id": task_spec_id,
                    "model_path": chunk[0]["model_path"],
                    "model_label": chunk[0]["model_label"],
                    "jobs": chunk,
                }
            )
            task_spec_id += 1

    return task_specs


def _interleave_task_specs_by_model(model_task_spec_lists: list[list[dict]]) -> list[dict]:
    pending = [list(task_specs) for task_specs in model_task_spec_lists if task_specs]
    interleaved: list[dict] = []
    while pending:
        next_round: list[list[dict]] = []
        for task_specs in pending:
            interleaved.append(task_specs.pop(0))
            if task_specs:
                next_round.append(task_specs)
        pending = next_round
    return interleaved


def _attach_worker_slot(task_spec: dict, worker_slot: dict) -> dict:
    assigned = dict(task_spec)
    assigned["task_id"] = worker_slot["task_id"]
    assigned["server_device"] = worker_slot["server_device"]
    assigned["device_worker_index"] = worker_slot["device_worker_index"]
    return assigned


def _build_failure_result(args, server_url: str, run_dir: Path, task_spec: dict, job: dict, exc: Exception) -> dict:
    logs_dir = run_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    sim_log = logs_dir / f"{job['model_label']}__seed_{job['seed']}__repeat_{job['repeat_idx']}__episode_{job['episode_index']}.log"
    return {
        'task': args.task,
        'model_path': job['model_path'],
        'model_label': job['model_label'],
        'seed': job['seed'],
        'repeat_idx': job['repeat_idx'],
        'episode_seed': job['episode_seed'],
        'episode_index': job['episode_index'],
        'success': False,
        'failure_reason': 'worker_error',
        'episode_steps': 0,
        'max_steps': args.max_steps,
        'final_reward': 0.0,
        'final_reward_scaled': 0.0,
        'max_reward': 0.0,
        'max_reward_scaled': 0.0,
        'video_path': '',
        'server_url': server_url,
        'log_path': str(sim_log),
        'returncode': -1,
        'error': str(exc),
        'worker_id': task_spec['task_id'],
        'worker_device': task_spec.get('server_device', ''),
    }


def _worker_run(task_spec: dict, args_dict: dict, run_dir_str: str) -> list[dict]:
    args = argparse.Namespace(**args_dict)
    run_dir = Path(run_dir_str)
    requested_server_port = args.server_port_base + task_spec['task_id']
    server_port_max = int(getattr(args, "server_port_max", 0) or min(65535, int(args.server_port_base) + 511))
    server_log_path = run_dir / 'logs' / (
        f"server__worker_{task_spec['task_id']}__task_{task_spec.get('task_spec_id', task_spec['task_id'])}__{task_spec['model_label']}.log"
    )

    args.server_device = task_spec['server_device']
    args.isaac_device = _resolve_worker_isaac_device(args.isaac_device, task_spec['server_device'])
    results = []
    server_proc = None
    server_log_fp = None
    server_port = requested_server_port
    server_url = f"{args.server_scheme}://{args.server_host}:{server_port}"
    try:
        lock_path = _server_start_lock_path(run_dir)
        with open(lock_path, "w", encoding="utf-8") as lock_fp:
            fcntl.flock(lock_fp, fcntl.LOCK_EX)
            server_port = _select_available_server_port(args.server_host, requested_server_port, server_port_max)
            server_url = f"{args.server_scheme}://{args.server_host}:{server_port}"
            args.server_port = server_port
            if server_port != requested_server_port:
                print(
                    f"[eval_vla_suite_parallel] worker={task_spec['task_id']} "
                    f"requested_port={requested_server_port} occupied; using port={server_port} "
                    f"within_range={requested_server_port}-{server_port_max}"
                )
            print(
                f"[eval_vla_suite_parallel] worker={task_spec['task_id']} "
                f"starting server model={task_spec['model_path']} port={server_port} "
                f"port_range={requested_server_port}-{server_port_max} "
                f"server_device={args.server_device} isaac_device={args.isaac_device}"
            )
            server_proc, server_log_fp = _start_server(args, task_spec['model_path'], server_log_path)
            _wait_for_server_ready(
                server_url,
                timeout_s=args.server_ready_timeout,
                verify_ssl=args.lerobot_server_verify_ssl,
            )
        if args.persistent_sim:
            print(
                f"[eval_vla_suite_parallel] worker={task_spec['task_id']} "
                f"model={task_spec['model_label']} seed={task_spec['seed']} persistent_batch repeats={len(task_spec['jobs'])}"
            )
            batch_results = _run_episode_batch(
                args=args,
                server_url=server_url,
                model_path=task_spec['model_path'],
                model_label=task_spec['model_label'],
                seed=int(task_spec['seed']),
                jobs=task_spec['jobs'],
                run_dir=run_dir,
            )
            for row in batch_results:
                row['worker_id'] = task_spec['task_id']
                row['worker_device'] = task_spec['server_device']
            return batch_results

        for job in task_spec['jobs']:
            print(
                f"[eval_vla_suite_parallel] worker={task_spec['task_id']} "
                f"model={job['model_label']} seed={job['seed']} repeat={job['repeat_idx'] + 1}/{args.repeats_per_seed} "
                f"episode_index={job['episode_index']}"
                )
            result = _run_episode(
                args=args,
                server_url=server_url,
                model_path=job['model_path'],
                model_label=job['model_label'],
                seed=job['seed'],
                repeat_idx=job['repeat_idx'],
                episode_seed=job['episode_seed'],
                episode_index=job['episode_index'],
                run_dir=run_dir,
            )
            result['worker_id'] = task_spec['task_id']
            result['worker_device'] = task_spec['server_device']
            results.append(result)
        return results
    except Exception as exc:
        print(f"[eval_vla_suite_parallel] worker={task_spec['task_id']} failed: {exc}")
        handled = {
            _result_key(row['model_path'], row['seed'], row['repeat_idx'])
            for row in results
        }
        for job in task_spec['jobs']:
            job_key = _result_key(job['model_path'], job['seed'], job['repeat_idx'])
            if job_key in handled:
                continue
            results.append(_build_failure_result(args, server_url, run_dir, task_spec, job, exc))
        return results
    finally:
        try:
            if server_proc is not None:
                server_proc.terminate()
                server_proc.wait(timeout=10.0)
        except Exception:
            if server_proc is not None:
                server_proc.kill()
        try:
            if server_log_fp is not None:
                server_log_fp.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description='Parallel football-single VLA evaluation suite')
    parser.add_argument('--task', type=str, default=TASK_FOOTBALL_SINGLE)
    parser.add_argument('--env_config_yaml', type=str, default='tasks/common_test_config/base_test/football_single_sonic_test.yaml')
    parser.add_argument('--model-path', dest='model_paths', action='append', required=True)
    parser.add_argument('--seed', dest='seeds', action='append', type=int, required=True)
    parser.add_argument('--repeats_per_seed', type=int, default=1)
    parser.add_argument('--persistent_sim', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=300)
    parser.add_argument('--video_fps', type=int, default=30)
    parser.add_argument('--post_termination_record_steps', type=int, default=0)
    parser.add_argument('--record_video_every_n', type=int, default=1)
    parser.add_argument('--step_log_every_n', type=int, default=0)
    parser.add_argument('--verbose_startup', action='store_true', default=False)
    parser.add_argument('--robot_type', type=str, default='unitree_g1_refpose_v3_1')
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
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--isaac_device', type=str, default='cpu')
    parser.add_argument('--server_python', type=str, required=True)
    parser.add_argument('--server_script', type=str, required=True)
    parser.add_argument('--server_device', type=str, default='cuda:0')
    parser.add_argument('--server_gpu_ids', type=str, default='')
    parser.add_argument('--server_host', type=str, default='127.0.0.1')
    parser.add_argument('--server_scheme', type=str, default='http', choices=['http', 'https'])
    parser.add_argument('--tls_cert_file', type=str, default='')
    parser.add_argument('--tls_key_file', type=str, default='')
    parser.add_argument('--lerobot_server_timeout', type=float, default=5.0)
    parser.add_argument('--lerobot_server_verify_ssl', action='store_true', default=False)
    parser.add_argument('--server_ready_timeout', type=float, default=60.0)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--server_port_base', type=int, default=18443)
    parser.add_argument('--server_port_max', type=int, default=0)
    args = parser.parse_args()

    if args.num_workers <= 0:
        raise ValueError('--num_workers must be >= 1')

    run_dir = Path(args.results_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    all_jobs, _ = _build_jobs(args)
    if not all_jobs:
        raise ValueError('No evaluation jobs were generated')

    existing_results = _load_existing_results(run_dir)
    if existing_results:
        print(
            f"[eval_vla_suite_parallel] resume detected existing_results={len(existing_results)} results_dir={run_dir}"
        )

    jobs_by_model: dict[str, list[dict]] = {}
    for job in all_jobs:
        jobs_by_model.setdefault(job['model_path'], []).append(job)

    server_devices = _resolve_server_devices(args)
    worker_slots = _build_worker_slots(server_devices, args.num_workers)
    total_worker_slots = len(worker_slots)
    print(
        f"[eval_vla_suite_parallel] total_jobs={len(all_jobs)} workers_per_device={args.num_workers} "
        f"total_worker_slots={total_worker_slots} models={len(jobs_by_model)} server_devices={server_devices} "
        f"server_port_base={args.server_port_base} server_port_max={args.server_port_max or min(65535, int(args.server_port_base) + 511)} "
        f"persistent_sim={bool(args.persistent_sim)}"
    )

    args_dict = vars(args).copy()
    model_task_spec_lists: list[list[dict]] = []
    for model_index, (model_path, model_jobs) in enumerate(jobs_by_model.items(), start=1):
        pending_jobs = [
            job for job in model_jobs
            if _result_key(job["model_path"], job["seed"], job["repeat_idx"]) not in existing_results
        ]
        completed_jobs = len(model_jobs) - len(pending_jobs)
        model_label = model_jobs[0]["model_label"]
        print(
            f"[eval_vla_suite_parallel] model={model_index}/{len(jobs_by_model)} label={model_label} "
            f"total_jobs={len(model_jobs)} completed={completed_jobs} pending={len(pending_jobs)}"
        )
        if not pending_jobs:
            continue
        model_task_spec_lists.append(
            _build_task_specs_for_model_jobs(
                pending_jobs,
                persistent_sim=bool(args.persistent_sim),
                max_tasks=total_worker_slots,
            )
        )

    pending_task_specs = _interleave_task_specs_by_model(model_task_spec_lists)
    print(f"[eval_vla_suite_parallel] global_pending_task_specs={len(pending_task_specs)}")

    if pending_task_specs:
        pending_queue = collections.deque(pending_task_specs)
        with concurrent.futures.ProcessPoolExecutor(max_workers=total_worker_slots) as executor:
            future_to_task: dict[concurrent.futures.Future, dict] = {}

            def submit_next(worker_slot: dict) -> None:
                if not pending_queue:
                    return
                task_spec = _attach_worker_slot(pending_queue.popleft(), worker_slot)
                worker_id = task_spec["task_id"]
                task_model_label = task_spec["model_label"]
                task_seed = task_spec.get("seed", "chunk")
                print(
                    f"[eval_vla_suite_parallel] dispatch worker={worker_id} "
                    f"model={task_model_label} seed={task_seed} "
                    f"remaining_task_specs={len(pending_queue)}"
                )
                future = executor.submit(_worker_run, task_spec, args_dict, str(run_dir))
                future_to_task[future] = task_spec

            for worker_slot in worker_slots:
                submit_next(worker_slot)

            last_live_summary_write = 0.0
            live_summary_interval_s = 30.0
            while future_to_task:
                done, _ = concurrent.futures.wait(
                    set(future_to_task.keys()),
                    timeout=live_summary_interval_s,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    now = time.monotonic()
                    if now - last_live_summary_write >= live_summary_interval_s:
                        partial_results = _merged_summary_results(run_dir, existing_results)
                        _write_summary(run_dir, partial_results)
                        last_live_summary_write = now
                        print(
                            f"[eval_vla_suite_parallel] live summary refreshed completed_episodes={len(partial_results)}"
                        )
                    continue
                for future in done:
                    task_spec = future_to_task.pop(future)
                    worker_id = task_spec["task_id"]
                    worker_slot = {
                        "task_id": worker_id,
                        "server_device": task_spec["server_device"],
                        "device_worker_index": task_spec["device_worker_index"],
                    }
                    try:
                        task_results = future.result()
                    except Exception as exc:
                        print(
                            f"[eval_vla_suite_parallel] worker={worker_id} crashed before returning results: {exc}"
                        )
                        task_results = [
                            _build_failure_result(
                                args,
                                f"{args.server_scheme}://{args.server_host}:{args.server_port_base + worker_id}",
                                run_dir,
                                task_spec,
                                job,
                                exc,
                            )
                            for job in task_spec["jobs"]
                        ]
                    print(
                        f"[eval_vla_suite_parallel] worker={worker_id} completed episodes={len(task_results)}"
                    )
                    for row in task_results:
                        existing_results[_result_key(row["model_path"], row["seed"], row["repeat_idx"])] = row
                    partial_results = _merged_summary_results(run_dir, existing_results)
                    _write_summary(run_dir, partial_results)
                    last_live_summary_write = time.monotonic()
                    submit_next(worker_slot)

    results = _merged_summary_results(run_dir, existing_results)
    _write_summary(run_dir, results)
    total_successes = sum(int(bool(row.get('success'))) for row in results)
    total_episodes = len(results)
    total_failures = max(0, total_episodes - total_successes)
    overall_rate = total_successes / total_episodes if total_episodes else 0.0
    result_reason_counts: dict[str, int] = {}
    failure_reason_counts: dict[str, int] = {}
    for row in results:
        reason = str(row.get('failure_reason', '') or '').strip()
        if not reason:
            reason = 'success' if bool(row.get('success')) else 'unknown'
        result_reason_counts[reason] = int(result_reason_counts.get(reason, 0)) + 1
        if not bool(row.get('success')):
            failure_reason_counts[reason] = int(failure_reason_counts.get(reason, 0)) + 1
    summary = {
        'total_jobs': len(all_jobs),
        'completed_results': len(results),
        'pending_results': max(0, len(all_jobs) - len(results)),
        'total_successes': total_successes,
        'total_failures': total_failures,
        'workers_per_device': args.num_workers,
        'total_worker_slots': total_worker_slots,
        'models_total': len(jobs_by_model),
        'models_completed': sum(
            1 for model_jobs in jobs_by_model.values()
            if all(_result_key(job['model_path'], job['seed'], job['repeat_idx']) in existing_results for job in model_jobs)
        ),
        'server_devices': server_devices,
        'server_port_base': args.server_port_base,
        'server_port_max': args.server_port_max,
        'overall_success_rate': overall_rate,
        'result_reason_counts': {key: result_reason_counts[key] for key in sorted(result_reason_counts)},
        'failure_reason_counts': {key: failure_reason_counts[key] for key in sorted(failure_reason_counts)},
        'persistent_sim': bool(args.persistent_sim),
    }
    (run_dir / 'parallel_config.json').write_text(json.dumps(summary, ensure_ascii=True, indent=2) + '\n')
    print(
        f"[eval_vla_suite_parallel] completed episodes={total_episodes} successes={total_successes} "
        f"success_rate={overall_rate:.4f} results_dir={run_dir}"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
