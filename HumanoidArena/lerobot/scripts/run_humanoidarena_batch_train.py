#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import signal
import socket
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SPEC_FILENAME = "scheduler_job_spec.json"
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "summary.json"
EVENTS_FILENAME = "scheduler_events.jsonl"
ACTIVE_LOG_SUFFIX = "__active.log"
LOCK_FILENAME = "scheduler_job.lock"
SCHEDULER_METADATA_FILENAMES = {SPEC_FILENAME, LOCK_FILENAME}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return str(value)


def nested_get(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def classify_dataset_kind(dataset_name: str, rules: list[dict[str, str]]) -> str:
    lowered = dataset_name.lower()
    for rule in rules:
        token = str(rule["token"]).lower()
        if token in lowered:
            return str(rule["kind"])
    head = lowered.split("_", 1)[0].strip()
    return head or lowered


def verify_train_config_compatibility(train_cfg_path: Path, job_spec: dict[str, Any]) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    try:
        train_cfg = load_json(train_cfg_path)
    except Exception as exc:
        return False, [f"failed_to_read_train_config:{exc}"]

    ignored_keys = {"output_dir", "job_name"}
    for dotted_key, expected in job_spec["cli_args"].items():
        if dotted_key in ignored_keys:
            continue
        actual = nested_get(train_cfg, dotted_key)
        if actual != expected:
            mismatches.append(f"{dotted_key}: expected={expected!r}, actual={actual!r}")
    return (len(mismatches) == 0), mismatches


def build_completed_train_config_index(search_roots: list[str]) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_root in search_roots:
        root = Path(raw_root)
        if not root.is_dir():
            continue
        for path in root.glob("**/checkpoints/*/pretrained_model/train_config.json"):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            step_dir = path.parent.parent
            step_name = step_dir.name
            step = int(step_name) if step_name.isdigit() else None
            index.append(
                {
                    "train_config_path": path,
                    "output_dir": path.parents[3],
                    "checkpoint_dir": path.parent,
                    "step": step,
                }
            )
    return sorted(index, key=lambda item: str(item["train_config_path"]))


def build_output_dir(results_root: Path, job_name: str) -> Path:
    return results_root / job_name


def build_repo_id(dataset_dir: Path) -> str:
    return f"local/{dataset_dir.name}"


def normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return spec


def find_latest_checkpoint_step(output_dir: Path) -> tuple[int | None, Path | None]:
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None, None

    best_step: int | None = None
    best_path: Path | None = None
    for step_dir in checkpoints_dir.iterdir():
        if not step_dir.is_dir() or not step_dir.name.isdigit():
            continue
        train_cfg_path = step_dir / "pretrained_model" / "train_config.json"
        if not train_cfg_path.exists():
            continue
        step = int(step_dir.name)
        if best_step is None or step > best_step:
            best_step = step
            best_path = train_cfg_path
    return best_step, best_path


def should_resume_incomplete(job: Job, train_cfg_path: Path | None, resume_incomplete_jobs: bool) -> tuple[str, bool, list[str]] | None:
    if train_cfg_path is None:
        return None
    compatible, mismatches = verify_train_config_compatibility(train_cfg_path, job.spec())
    if not compatible:
        return "conflict", False, ["checkpoint_config_mismatch", *mismatches]

    latest_step, _ = find_latest_checkpoint_step(job.output_dir)
    if latest_step is None:
        if resume_incomplete_jobs:
            return "resume", True, ["resume_without_numeric_checkpoint_step"]
        return "skip", False, ["skip_without_numeric_checkpoint_step"]
    if latest_step >= job.steps:
        return "skip", False, [f"completed_or_target_reached:{latest_step}"]
    if resume_incomplete_jobs:
        return "resume", True, [f"resume_incomplete_checkpoint_step:{latest_step}/{job.steps}"]
    return "skip", False, [f"skip_incomplete_checkpoint_step:{latest_step}/{job.steps}"]


@dataclass
class Job:
    model_alias: str
    task_name: str
    dataset_kind: str
    dataset_dir_name: str
    dataset_root: Path
    output_dir: Path
    job_name: str
    model_config_path: Path
    model_config_sha256: str
    launcher: str
    priority: int
    policy_type: str
    cli_args: dict[str, Any]
    steps: int
    expected_checkpoint_dir: Path
    status: str = "pending"
    resume: bool = False
    assigned_gpus: list[int] | None = None
    log_path: str | None = None
    pid: int | None = None
    returncode: int | None = None
    notes: list[str] | None = None

    def spec(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "job_name": self.job_name,
            "task_name": self.task_name,
            "dataset_kind": self.dataset_kind,
            "dataset_dir_name": self.dataset_dir_name,
            "dataset_root": str(self.dataset_root),
            "dataset_repo_id": build_repo_id(self.dataset_root),
            "model_alias": self.model_alias,
            "policy_type": self.policy_type,
            "model_config_path": str(self.model_config_path),
            "model_config_sha256": self.model_config_sha256,
            "launcher": self.launcher,
            "cli_args": self.cli_args,
            "steps": self.steps,
            "expected_checkpoint_dir": str(self.expected_checkpoint_dir),
        }

    def manifest_row(self) -> dict[str, Any]:
        return {
            "job_name": self.job_name,
            "model_alias": self.model_alias,
            "task_name": self.task_name,
            "dataset_kind": self.dataset_kind,
            "dataset_dir_name": self.dataset_dir_name,
            "dataset_root": str(self.dataset_root),
            "output_dir": str(self.output_dir),
            "status": self.status,
            "resume": self.resume,
            "assigned_gpus": self.assigned_gpus or [],
            "log_path": self.log_path,
            "pid": self.pid,
            "returncode": self.returncode,
            "notes": self.notes or [],
        }


def determine_existing_state(
    job: Job,
    completed_train_configs: list[dict[str, Any]],
    resume_incomplete_jobs: bool = False,
) -> tuple[str, bool, list[str]]:
    output_dir = job.output_dir
    spec_path = output_dir / SPEC_FILENAME
    checkpoints_last = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    latest_step, latest_train_cfg_path = find_latest_checkpoint_step(output_dir)

    if spec_path.exists():
        try:
            existing_spec = load_json(spec_path)
        except Exception as exc:
            return "conflict", False, [f"existing_spec_unreadable:{exc}"]
        has_only_scheduler_metadata = all(
            path.name in SCHEDULER_METADATA_FILENAMES for path in output_dir.iterdir()
        )
        if normalize_spec(existing_spec) != normalize_spec(job.spec()):
            if has_only_scheduler_metadata:
                return "pending", False, ["metadata_only_output_dir_with_stale_scheduler_spec"]
            return "conflict", False, ["existing_spec_mismatch"]
        if job.expected_checkpoint_dir.is_dir():
            return "skip", False, ["completed_with_matching_scheduler_spec"]
        if checkpoints_last.exists():
            resume_state = should_resume_incomplete(job, checkpoints_last, resume_incomplete_jobs)
            if resume_state is not None:
                return resume_state
        if latest_train_cfg_path is not None:
            resume_state = should_resume_incomplete(job, latest_train_cfg_path, resume_incomplete_jobs)
            if resume_state is not None:
                return resume_state
        has_non_spec_contents = any(path.name not in {SPEC_FILENAME, "wandb", LOCK_FILENAME} for path in output_dir.iterdir())
        if not has_non_spec_contents:
            return "pending", False, ["scheduler_spec_without_checkpoint"]
        if resume_incomplete_jobs:
            return "resume", True, ["resume_with_matching_scheduler_spec_no_checkpoint_step"]
        return "skip", False, ["skip_with_matching_scheduler_spec_no_checkpoint_step"]

    if job.expected_checkpoint_dir.is_dir():
        train_cfg_path = job.expected_checkpoint_dir / "train_config.json"
        compatible, mismatches = verify_train_config_compatibility(train_cfg_path, job.spec())
        if compatible:
            return "skip", False, ["completed_with_matching_train_config"]
        return "conflict", False, ["completed_checkpoint_config_mismatch", *mismatches]

    if checkpoints_last.exists():
        resume_state = should_resume_incomplete(job, checkpoints_last, resume_incomplete_jobs)
        if resume_state is not None:
            return resume_state

    if latest_train_cfg_path is not None:
        resume_state = should_resume_incomplete(job, latest_train_cfg_path, resume_incomplete_jobs)
        if resume_state is not None:
            return resume_state

    if not output_dir.exists() or not any(output_dir.iterdir()):
        for entry in completed_train_configs:
            train_cfg_path = entry["train_config_path"]
            compatible, mismatches = verify_train_config_compatibility(train_cfg_path, job.spec())
            if not compatible:
                continue

            checkpoint_step = entry.get("step")
            if checkpoint_step is not None and checkpoint_step >= job.steps:
                return "skip", False, [f"completed_elsewhere:{train_cfg_path}"]

            if resume_incomplete_jobs:
                source_output_dir = Path(entry["output_dir"])
                job.output_dir = source_output_dir
                job.cli_args["output_dir"] = str(source_output_dir)
                job.expected_checkpoint_dir = source_output_dir / "checkpoints" / step_identifier(job.steps, job.steps) / "pretrained_model"
                return "resume", True, [f"resume_elsewhere:{train_cfg_path}"]

        return "pending", False, []

    try:
        has_contents = any(output_dir.iterdir())
    except Exception as exc:
        return "conflict", False, [f"output_dir_iter_failed:{exc}"]
    if has_contents:
        return "conflict", False, ["output_dir_exists_without_recognized_spec_or_checkpoint"]
    return "pending", False, []


def build_job(task_dir: Path, dataset_dir: Path, model_alias: str, model_cfg: dict[str, Any], results_root: Path) -> Job:
    task_name = task_dir.name
    dataset_kind = classify_dataset_kind(dataset_dir.name, model_cfg["dataset_kind_rules"])
    job_name = f"{model_alias}_{task_name}_{dataset_kind}"
    output_dir = build_output_dir(results_root, job_name)
    cli_args = dict(model_cfg["args"])
    cli_args["dataset.repo_id"] = build_repo_id(dataset_dir)
    cli_args["dataset.root"] = str(dataset_dir)
    cli_args["output_dir"] = str(output_dir)
    cli_args["job_name"] = job_name
    cli_args["wandb.project"] = model_cfg["wandb_project"]
    cli_args["wandb.notes"] = f"task={task_name},dataset_kind={dataset_kind},dataset={dataset_dir.name},model={model_alias}"
    steps = int(cli_args["steps"])
    checkpoint_dir = output_dir / "checkpoints" / step_identifier(steps, steps) / "pretrained_model"
    return Job(
        model_alias=model_alias,
        task_name=task_name,
        dataset_kind=dataset_kind,
        dataset_dir_name=dataset_dir.name,
        dataset_root=dataset_dir,
        output_dir=output_dir,
        job_name=job_name,
        model_config_path=Path(model_cfg["_config_path"]),
        model_config_sha256=model_cfg["_config_sha256"],
        launcher=str(model_cfg["launcher"]),
        priority=int(model_cfg["priority"]),
        policy_type=str(model_cfg["args"]["policy.type"]),
        cli_args=cli_args,
        steps=steps,
        expected_checkpoint_dir=checkpoint_dir,
    )


def build_command(job: Job, scheduler_cfg: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    lerobot_root = Path(scheduler_cfg["lerobot_root"])
    python_bin = str(scheduler_cfg.get("python_bin") or sys.executable)
    if not os.path.isabs(python_bin):
        python_bin = sys.executable
    train_script = str((lerobot_root / scheduler_cfg["train_entrypoint"]).resolve())

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in scheduler_cfg.get("env", {}).items()})
    wandb_api_key = str(scheduler_cfg.get("wandb_api_key", "") or "").strip()
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key
    env["TORCH_HOME"] = str(scheduler_cfg["torch_home"])
    env["PYTHONPATH"] = str(scheduler_cfg.get("pythonpath", "src"))
    python_lib_dir = str((Path(python_bin).resolve().parent.parent / "lib"))
    existing_ld_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{python_lib_dir}:{existing_ld_library_path}" if existing_ld_library_path else python_lib_dir
    )
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in (job.assigned_gpus or []))
    env["PYTHONUNBUFFERED"] = "1"

    if job.launcher == "torchrun":
        cmd = [
            python_bin,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={len(job.assigned_gpus or [])}",
            train_script,
        ]
    elif job.launcher == "python":
        cmd = [python_bin, train_script]
    else:
        raise ValueError(f"Unsupported launcher {job.launcher!r} for {job.job_name}")

    arg_items = list(job.cli_args.items())
    if job.resume:
        arg_items.append(("resume", True))
    for key, value in arg_items:
        cmd.append(f"--{key}={cli_value(value)}")
    return cmd, env


def process_exists(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_active_log_metadata(path: Path) -> dict[str, int]:
    metadata: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(8):
                line = f.readline()
                if not line:
                    break
                if not line.startswith("# ") or "=" not in line:
                    continue
                key, value = line[2:].strip().split("=", 1)
                if key in {"scheduler_pid", "training_pid"}:
                    try:
                        metadata[key] = int(value)
                    except ValueError:
                        continue
    except FileNotFoundError:
        return {}
    return metadata


def job_active_log_path(job: Job, log_root: Path) -> Path:
    return log_root / f"{job.job_name}{ACTIVE_LOG_SUFFIX}"


def cleanup_metadata_only_output_dir(output_dir: Path) -> None:
    if not output_dir.is_dir():
        return
    entries = list(output_dir.iterdir())
    if not entries or any(path.name not in SCHEDULER_METADATA_FILENAMES for path in entries):
        return
    for path in entries:
        path.unlink()
    try:
        output_dir.rmdir()
    except OSError:
        pass


def job_lock_path(job: Job) -> Path:
    return job.output_dir.parent / "scheduler_logs" / "locks" / f"{job.job_name}.lock"


def _archive_existing_file(path: Path, suffix: str) -> None:
    if not path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = path.with_name(f"{path.name}.{suffix}.{timestamp}")
    counter = 1
    while archived.exists():
        archived = path.with_name(f"{path.name}.{suffix}.{timestamp}_{counter}")
        counter += 1
    path.rename(archived)


def _build_job_lock_payload(job: Job, training_pid: int | None = None) -> dict[str, Any]:
    now = time.time()
    now_iso = datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": SCHEMA_VERSION,
        "job_name": job.job_name,
        "hostname": socket.gethostname(),
        "scheduler_pid": os.getpid(),
        "training_pid": training_pid,
        "assigned_gpus": list(job.assigned_gpus or []),
        "created_at": now_iso,
        "created_ts": now,
        "heartbeat_at": now_iso,
        "heartbeat_ts": now,
    }


def _write_job_lock(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def _update_job_lock(job: Job, training_pid: int | None = None) -> None:
    lock_path = job_lock_path(job)
    if not lock_path.exists():
        return
    payload = _build_job_lock_payload(job, training_pid=training_pid)
    _write_job_lock(lock_path, payload)


def _is_job_lock_stale(lock_path: Path, stale_seconds: int) -> bool:
    try:
        payload = load_json(lock_path)
    except Exception:
        try:
            age = time.time() - lock_path.stat().st_mtime
        except FileNotFoundError:
            return False
        return age > stale_seconds

    heartbeat_ts = payload.get("heartbeat_ts") or payload.get("created_ts")
    try:
        heartbeat_ts_f = float(heartbeat_ts)
    except (TypeError, ValueError):
        return True

    age = time.time() - heartbeat_ts_f
    if age <= stale_seconds:
        return False

    if str(payload.get("hostname", "")) == socket.gethostname():
        if process_exists(payload.get("training_pid")) or process_exists(payload.get("scheduler_pid")):
            return False

    return True


def _clear_stale_job_claim(job: Job, log_root: Path, stale_seconds: int) -> bool:
    lock_path = job_lock_path(job)
    active_log_path = job_active_log_path(job, log_root)

    if lock_path.exists():
        if not _is_job_lock_stale(lock_path, stale_seconds):
            return False
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        if active_log_path.exists():
            _archive_existing_file(active_log_path, "stale")
        return True

    if active_log_path.exists():
        metadata = read_active_log_metadata(active_log_path)
        if process_exists(metadata.get("training_pid")) or process_exists(metadata.get("scheduler_pid")):
            return False
        try:
            age = time.time() - active_log_path.stat().st_mtime
        except FileNotFoundError:
            return True
        if age <= stale_seconds:
            return False
        _archive_existing_file(active_log_path, "stale")

    return True


def is_job_claimed_by_lock(job: Job, log_root: Path, stale_seconds: int) -> bool:
    return not _clear_stale_job_claim(job, log_root, stale_seconds)


def finalize_job_log(job: Job, process: subprocess.Popen, log_root: Path) -> None:
    log_fp = getattr(process, "_scheduler_log_fp", None)
    active_log_path_raw = getattr(process, "_scheduler_active_log_path", None)
    timestamp = getattr(process, "_scheduler_log_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))

    if log_fp is not None:
        try:
            log_fp.flush()
        except Exception:
            pass
        try:
            log_fp.close()
        except Exception:
            pass

    if not active_log_path_raw:
        return

    active_log_path = Path(active_log_path_raw)
    if not active_log_path.exists():
        return

    final_log_path = log_root / f"{job.job_name}__{timestamp}.log"
    suffix = 1
    while final_log_path.exists():
        final_log_path = log_root / f"{job.job_name}__{timestamp}_{suffix}.log"
        suffix += 1
    active_log_path.rename(final_log_path)
    job.log_path = str(final_log_path)


def query_gpu_memory_used_mb() -> dict[int, int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return {}

    usage: dict[int, int] = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            usage[int(parts[0])] = int(float(parts[1]))
        except ValueError:
            continue
    return usage


def gpus_are_idle(gpu_ids: list[int], max_used_memory_mb: int) -> bool:
    if not gpu_ids:
        return False
    usage = query_gpu_memory_used_mb()
    if not usage:
        return True
    return all(usage.get(int(gpu_id), max_used_memory_mb + 1) <= max_used_memory_mb for gpu_id in gpu_ids)


def launch_job(job: Job, scheduler_cfg: dict[str, Any], log_root: Path, lock_stale_seconds: int) -> subprocess.Popen:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    active_log_path = job_active_log_path(job, log_root)
    lock_path = job_lock_path(job)

    cleanup_metadata_only_output_dir(job.output_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if not _clear_stale_job_claim(job, log_root, lock_stale_seconds):
        raise FileExistsError(f"Job lock already exists for {job.job_name}: {lock_path}")

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(f"Job lock already exists for {job.job_name}: {lock_path}") from exc

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as lock_fp:
            json.dump(_build_job_lock_payload(job), lock_fp, ensure_ascii=True, indent=2)
            lock_fp.write("\n")

        if active_log_path.exists():
            _archive_existing_file(active_log_path, "stale")
        active_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = active_log_path.open("w", encoding="utf-8", buffering=1)
        log_fp.write(f"# scheduler_pid={os.getpid()}\n")
        log_fp.write(f"# created_at={datetime.now().isoformat(timespec='seconds')}\n")
        log_fp.write(f"# hostname={socket.gethostname()}\n")

        dump_json(log_root / "specs" / f"{job.job_name}.json", job.spec())

        cmd, env = build_command(job, scheduler_cfg)
        process = subprocess.Popen(
            cmd,
            cwd=str(scheduler_cfg["lerobot_root"]),
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_fp.write(f"# training_pid={process.pid}\n")
        log_fp.flush()
        _update_job_lock(job, training_pid=process.pid)
    except Exception:
        try:
            log_fp.close()
        except Exception:
            pass
        try:
            active_log_path.unlink()
        except FileNotFoundError:
            pass
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        raise

    setattr(process, "_scheduler_log_fp", log_fp)
    setattr(process, "_scheduler_active_log_path", str(active_log_path))
    setattr(process, "_scheduler_log_timestamp", timestamp)
    job.log_path = str(active_log_path)
    job.pid = process.pid
    job.status = "running"
    return process


def save_manifest(path: Path, jobs: list[Job]) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "jobs": [job.manifest_row() for job in jobs],
    }
    dump_json(path, payload)


def save_summary(path: Path, jobs: list[Job]) -> None:
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.status] = counts.get(job.status, 0) + 1
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "counts": counts,
        "jobs": [job.manifest_row() for job in jobs],
    }
    dump_json(path, payload)


def scan_jobs(scheduler_cfg: dict[str, Any], model_cfgs: dict[str, dict[str, Any]], completed_train_configs: list[dict[str, Any]]) -> list[Job]:
    dataset_root = Path(scheduler_cfg["dataset_root"])
    results_root = Path(scheduler_cfg["results_root"])
    ignore = set(str(x) for x in scheduler_cfg.get("ignore_task_names", []))
    jobs: list[Job] = []
    seen_job_names: dict[str, Path] = {}

    for task_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if task_dir.name in ignore:
            continue
        dataset_dirs = [p for p in sorted(task_dir.iterdir()) if p.is_dir() and not p.name.startswith(".")]
        for dataset_dir in dataset_dirs:
            for model_alias in scheduler_cfg["model_order"]:
                job = build_job(task_dir, dataset_dir, model_alias, model_cfgs[model_alias], results_root)
                if job.job_name in seen_job_names:
                    raise RuntimeError(
                        f"Duplicate job name {job.job_name!r} for dataset {dataset_dir} and {seen_job_names[job.job_name]}. "
                        "Adjust dataset naming or classification rules."
                    )
                seen_job_names[job.job_name] = dataset_dir
                state, resume, notes = determine_existing_state(
                    job,
                    completed_train_configs,
                    bool(scheduler_cfg.get("resume_incomplete_jobs", False)),
                )
                job.status = state
                job.resume = resume
                job.notes = notes
                jobs.append(job)
    return jobs


def sort_jobs_for_queue(
    jobs: list[Job],
    dataset_kind_order: list[str] | None = None,
    model_order: list[str] | None = None,
) -> list[Job]:
    order = {str(kind): index for index, kind in enumerate(dataset_kind_order or [])}
    model_rank = {str(alias): index for index, alias in enumerate(model_order or [])}

    def sort_key(job: Job) -> tuple[str, int, str, int, str, int, str]:
        kind_rank = order.get(job.dataset_kind, len(order))
        alias_rank = model_rank.get(job.model_alias, len(model_rank))
        return (job.task_name, kind_rank, job.dataset_kind, alias_rank, job.model_alias, -job.priority, job.job_name)

    return sorted(jobs, key=sort_key)


def print_plan(jobs: list[Job], scheduler_cfg: dict[str, Any]) -> None:
    pi05_gpu_groups = scheduler_cfg.get("pi05_gpu_groups")
    if pi05_gpu_groups is None:
        legacy_pool = scheduler_cfg.get("pi05_gpu_pool", scheduler_cfg.get("pi05_gpu_group", []))
        pi05_gpu_groups = [[gpu_id] for gpu_id in legacy_pool]
    pi05_model_aliases = scheduler_cfg.get("pi05_model_aliases", ["pi05"])
    single_gpu_model_aliases = scheduler_cfg.get("single_gpu_model_aliases")
    if single_gpu_model_aliases is None:
        single_gpu_model_aliases = [alias for alias in scheduler_cfg.get("model_order", []) if alias not in pi05_model_aliases]
    single_gpu_capacity = int(scheduler_cfg.get("single_gpu_capacity", 1))
    single_gpu_capacity_by_gpu = {int(k): int(v) for k, v in scheduler_cfg.get("single_gpu_capacity_by_gpu", {}).items()}
    print("=" * 100)
    print("HumanoidArena LeRobot Batch Train Plan")
    print("=" * 100)
    print(f"dataset_root      : {scheduler_cfg['dataset_root']}")
    print(f"results_root      : {scheduler_cfg['results_root']}")
    print(f"scheduler_log_root: {scheduler_cfg['scheduler_log_root']}")
    print(f"pi05_model_aliases: {pi05_model_aliases}")
    print(f"pi05_gpu_groups   : {pi05_gpu_groups}")
    single_gpu_pool = sorted(single_gpu_capacity_by_gpu)
    print(f"single_gpu_model_aliases: {single_gpu_model_aliases}")
    print(f"single_gpu_pool   : {single_gpu_pool}")
    print(f"single_gpu_capacity: {single_gpu_capacity}")
    print(f"single_gpu_capacity_by_gpu: {single_gpu_capacity_by_gpu}")
    print(f"wandb_project     : {scheduler_cfg.get('wandb_project', '')}")
    print("-" * 100)
    dataset_kind_order = [str(kind) for kind in scheduler_cfg.get("dataset_kind_order", [])]
    model_order = [str(alias) for alias in scheduler_cfg.get("model_order", [])]
    for job in sort_jobs_for_queue(jobs, dataset_kind_order, model_order):
        note = f" notes={job.notes}" if job.notes else ""
        print(f"{job.status:>8} | {job.job_name:<40} | dataset={job.dataset_root}{note}")
    print("=" * 100)


def terminate_process_tree(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch scheduler for HumanoidArena LeRobot training jobs")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/ai/Yichi/taowen/HumanoidArena/lerobot/configs/humanoidarena_batch_train/scheduler.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scheduler_cfg = load_json(args.config)
    model_cfgs: dict[str, dict[str, Any]] = {}
    model_config_dir = Path(scheduler_cfg["model_config_dir"])
    for model_alias in scheduler_cfg["model_order"]:
        cfg_path = model_config_dir / f"{model_alias}.json"
        model_cfg = load_json(cfg_path)
        model_cfg["_config_path"] = str(cfg_path)
        model_cfg["_config_sha256"] = file_sha256(cfg_path)
        model_cfg["dataset_kind_rules"] = scheduler_cfg["dataset_kind_rules"]
        model_cfg["wandb_project"] = scheduler_cfg.get("wandb_project", "lerobot")
        model_cfgs[model_alias] = model_cfg

    results_root = Path(scheduler_cfg["results_root"])
    log_root = Path(scheduler_cfg["scheduler_log_root"])
    results_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    manifest_path = results_root / MANIFEST_FILENAME
    summary_path = results_root / SUMMARY_FILENAME
    events_path = log_root / EVENTS_FILENAME

    completed_train_configs = build_completed_train_config_index(scheduler_cfg.get("completed_search_roots", []))
    jobs = scan_jobs(scheduler_cfg, model_cfgs, completed_train_configs)
    save_manifest(manifest_path, jobs)
    save_summary(summary_path, jobs)
    print_plan(jobs, scheduler_cfg)

    conflicts = [job for job in jobs if job.status == "conflict"]
    if conflicts:
        print("[scheduler] conflict jobs detected. Refusing to start.", file=sys.stderr)
        return 2

    if args.dry_run:
        return 0

    pi05_model_aliases = set(str(alias) for alias in scheduler_cfg.get("pi05_model_aliases", ["pi05"]))
    single_gpu_model_aliases_cfg = scheduler_cfg.get("single_gpu_model_aliases")
    if single_gpu_model_aliases_cfg is None:
        single_gpu_model_aliases_cfg = [alias for alias in scheduler_cfg.get("model_order", []) if alias not in pi05_model_aliases]
    single_gpu_model_aliases = set(str(alias) for alias in single_gpu_model_aliases_cfg)

    dataset_kind_order = [str(kind) for kind in scheduler_cfg.get("dataset_kind_order", [])]
    model_order = [str(alias) for alias in scheduler_cfg.get("model_order", [])]
    pi_queue = sort_jobs_for_queue(
        [job for job in jobs if job.model_alias in pi05_model_aliases and job.status in {"pending", "resume"}],
        dataset_kind_order,
        model_order,
    )
    single_queue = sort_jobs_for_queue(
        [job for job in jobs if job.model_alias in single_gpu_model_aliases and job.status in {"pending", "resume"}],
        dataset_kind_order,
        model_order,
    )

    poll_seconds = int(scheduler_cfg.get("poll_interval_seconds", 15))
    raw_pi05_gpu_groups = scheduler_cfg.get("pi05_gpu_groups")
    if raw_pi05_gpu_groups is None:
        legacy_pool = scheduler_cfg.get("pi05_gpu_pool", scheduler_cfg.get("pi05_gpu_group", []))
        raw_pi05_gpu_groups = [[gpu_id] for gpu_id in legacy_pool]
    pi05_gpu_groups = [tuple(int(gpu_id) for gpu_id in group) for group in raw_pi05_gpu_groups]

    single_gpu_capacity = int(scheduler_cfg.get("single_gpu_capacity", 1))
    raw_single_gpu_pool = scheduler_cfg.get("single_gpu_pool")
    if raw_single_gpu_pool is not None:
        single_gpu_pool = [int(gpu_id) for gpu_id in raw_single_gpu_pool]
    else:
        single_gpu_pool = []
    single_gpu_capacity_by_gpu = {gpu_id: single_gpu_capacity for gpu_id in single_gpu_pool}
    for raw_gpu_id, capacity in scheduler_cfg.get("single_gpu_capacity_by_gpu", {}).items():
        single_gpu_capacity_by_gpu[int(raw_gpu_id)] = int(capacity)
    if not single_gpu_pool:
        single_gpu_pool = sorted(single_gpu_capacity_by_gpu)

    gpu_free_max_used_memory_mb = int(scheduler_cfg.get("gpu_free_max_used_memory_mb", 1024))
    job_lock_stale_seconds = int(scheduler_cfg.get("job_lock_stale_seconds", 1800))
    job_lock_cleanup_grace_seconds = int(scheduler_cfg.get("job_lock_cleanup_grace_seconds", 20))

    pi_processes: dict[tuple[int, ...], tuple[Job, subprocess.Popen]] = {}
    single_processes: dict[int, list[tuple[Job, subprocess.Popen]]] = {gpu_id: [] for gpu_id in single_gpu_pool}

    def refresh_job_from_disk(job: Job) -> bool:
        state, resume, notes = determine_existing_state(
            job,
            [],
            bool(scheduler_cfg.get("resume_incomplete_jobs", False)),
        )
        changed = job.status != state or job.resume != resume or (job.notes or []) != (notes or [])
        job.status = state
        job.resume = resume
        job.notes = notes
        return changed

    def finalize_job(job: Job, process: subprocess.Popen) -> bool:
        returncode = process.poll()
        if returncode is None:
            _update_job_lock(job, training_pid=process.pid)
            return False
        finalize_job_log(job, process, log_root)
        if job.output_dir.exists():
            dump_json(job.output_dir / SPEC_FILENAME, job.spec())
        try:
            job_lock_path(job).unlink()
        except FileNotFoundError:
            pass
        job.returncode = returncode
        if returncode == 0 and job.expected_checkpoint_dir.is_dir():
            job.status = "completed"
        elif returncode == 0:
            job.status = "failed"
            job.notes = (job.notes or []) + ["process_exited_zero_without_final_checkpoint"]
        else:
            job.status = "failed"
        append_jsonl(events_path, {"event": "job_finished", **job.manifest_row()})
        return True

    def launch_next_job(queue: list[Job], assigned_gpus: list[int]) -> tuple[tuple[Job, subprocess.Popen] | None, bool]:
        state_changed_local = False
        index = 0
        while index < len(queue):
            job = queue[index]
            if refresh_job_from_disk(job):
                state_changed_local = True
            if job.status not in {"pending", "resume"}:
                queue.pop(index)
                state_changed_local = True
                continue
            if is_job_claimed_by_lock(job, log_root, job_lock_stale_seconds):
                index += 1
                continue
            if not gpus_are_idle(assigned_gpus, gpu_free_max_used_memory_mb):
                return None, state_changed_local
            job.assigned_gpus = list(assigned_gpus)
            try:
                process = launch_job(job, scheduler_cfg, log_root, job_lock_stale_seconds)
            except FileExistsError:
                index += 1
                continue
            queue.pop(index)
            append_jsonl(events_path, {"event": "job_started", **job.manifest_row()})
            return (job, process), True
        return None, state_changed_local

    try:
        while pi_queue or single_queue or pi_processes or any(single_processes.values()):
            state_changed = False

            for gpu_group, payload in list(pi_processes.items()):
                job, process = payload
                if finalize_job(job, process):
                    del pi_processes[gpu_group]
                    state_changed = True

            for gpu_id, payloads in single_processes.items():
                remaining: list[tuple[Job, subprocess.Popen]] = []
                for job, process in payloads:
                    if finalize_job(job, process):
                        state_changed = True
                    else:
                        remaining.append((job, process))
                single_processes[gpu_id] = remaining

            for gpu_group in pi05_gpu_groups:
                if gpu_group in pi_processes or not pi_queue:
                    continue
                payload, changed = launch_next_job(pi_queue, list(gpu_group))
                state_changed = state_changed or changed
                if payload is None:
                    continue
                job, process = payload
                pi_processes[gpu_group] = (job, process)

            for gpu_id in single_gpu_pool:
                gpu_capacity = int(single_gpu_capacity_by_gpu.get(gpu_id, single_gpu_capacity))
                while len(single_processes[gpu_id]) < gpu_capacity and single_queue:
                    payload, changed = launch_next_job(single_queue, [gpu_id])
                    state_changed = state_changed or changed
                    if payload is None:
                        break
                    job, process = payload
                    single_processes[gpu_id].append((job, process))

            if state_changed:
                save_manifest(manifest_path, jobs)
                save_summary(summary_path, jobs)
                continue

            time.sleep(poll_seconds)

    except KeyboardInterrupt:
        print("[scheduler] interrupted, terminating child processes...", file=sys.stderr)
        tracked_processes: list[tuple[Job, subprocess.Popen]] = []
        for job, process in pi_processes.values():
            terminate_process_tree(process)
            tracked_processes.append((job, process))
        for payloads in single_processes.values():
            for job, process in payloads:
                terminate_process_tree(process)
                tracked_processes.append((job, process))

        deadline = time.time() + job_lock_cleanup_grace_seconds
        while tracked_processes and time.time() < deadline:
            remaining: list[tuple[Job, subprocess.Popen]] = []
            for job, process in tracked_processes:
                if finalize_job(job, process):
                    continue
                remaining.append((job, process))
            tracked_processes = remaining
            if tracked_processes:
                time.sleep(1.0)
        return 130

    save_manifest(manifest_path, jobs)
    save_summary(summary_path, jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
