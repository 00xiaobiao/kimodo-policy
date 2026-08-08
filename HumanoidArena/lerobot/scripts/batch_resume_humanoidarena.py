#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_DATASET_ROOT = Path("/ai/Yichi/taowen/dataset")
DEFAULT_CONDA_PREFIX = Path("/ai/Yichi/0_Systems/miniconda3/envs/lerobot")
DEFAULT_CLIP_PATH = Path("/ai/Yichi/taowen/ckpts/checkpoints/clip-vit-base-patch16")
DEFAULT_MEM_MIB = {"act": 4000, "dp": 7400, "mtp": 14000}
OLD_DATASET_PREFIXES = [
    "/mnt/workspace/users/xujunzhe/yunhengwang/lerobot/lerobot/datasets/HumanoidArena_datasets",
]
OLD_CHECKPOINT_PREFIXES = [
    "/mnt/workspace/users/xujunzhe/yunhengwang/lerobot/lerobot/checkpoints",
]
TRAIN_CONFIG_REL = Path("checkpoints/last/pretrained_model/train_config.json")
LAST_CHECKPOINT_LINK = "last"
EVENTS_FILENAME = "resume_scheduler_events.jsonl"
MANIFEST_FILENAME = "resume_manifest.json"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".before_batch_resume")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_gpu_ids(raw: str) -> list[int]:
    gpus: list[int] = []
    for item in parse_csv(raw):
        if "-" in item:
            start, end = item.split("-", 1)
            gpus.extend(range(int(start), int(end) + 1))
        else:
            gpus.append(int(item))
    return sorted(dict.fromkeys(gpus))


def recursive_replace(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        out = value
        for old, new in replacements:
            out = out.replace(old, new)
        return out
    if isinstance(value, list):
        return [recursive_replace(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: recursive_replace(item, replacements) for key, item in value.items()}
    return value


def infer_model_kind(job_name: str, policy_type: str | None) -> str:
    text = f"{job_name} {policy_type or ''}".lower()
    if "multi_task" in text or "mtp" in text:
        return "mtp"
    if "diffusion" in text or re.search(r"(^|[_-])dp([_-]|$)", text):
        return "dp"
    if "act" in text:
        return "act"
    return (policy_type or job_name).lower()


def rel_job_dir_from_train_config(input_root: Path, train_cfg: Path) -> Path:
    model_dir = train_cfg.parent.parent.parent.parent
    return model_dir.relative_to(input_root)


def task_name_from_rel(rel_job_dir: Path) -> str:
    parts = rel_job_dir.parts
    if not parts:
        return "unknown_task"
    if parts[0] == "merges" and len(parts) >= 2:
        return parts[1]
    return parts[0]


def read_dataset_features(dataset_dir: Path) -> set[str]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return set()
    try:
        info = load_json(info_path)
    except Exception:
        return set()
    features = info.get("features")
    if not isinstance(features, dict):
        return set()
    return set(features.keys())


def find_dataset_dir(dataset_root: Path, task_name: str, old_root: str | None) -> tuple[Path | None, list[str]]:
    notes: list[str] = []
    if not old_root:
        return None, ["dataset.root missing in train_config"]
    basename = Path(old_root).name
    candidates = [dataset_root / task_name / basename, dataset_root / basename]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve(), notes
    matches = sorted(path for path in dataset_root.glob(f"**/{basename}") if path.is_dir())
    if len(matches) == 1:
        return matches[0].resolve(), notes
    if len(matches) > 1:
        notes.append("multiple dataset candidates: " + ", ".join(str(p) for p in matches[:5]))
        return matches[0].resolve(), notes
    notes.append(f"dataset not found for basename={basename!r} under {dataset_root}")
    return None, notes


def maybe_clear_rename_map(train_cfg: dict[str, Any], dataset_dir: Path | None) -> tuple[dict[str, str], list[str]]:
    notes: list[str] = []
    rename_map = train_cfg.get("rename_map")
    if not isinstance(rename_map, dict):
        rename_map = {}
    if not rename_map or dataset_dir is None:
        return dict(rename_map), notes

    dataset_features = read_dataset_features(dataset_dir)
    policy = train_cfg.get("policy", {}) if isinstance(train_cfg.get("policy"), dict) else {}
    input_features = policy.get("input_features", {}) if isinstance(policy.get("input_features"), dict) else {}
    expected_obs = {key for key in input_features if str(key).startswith("observation.")}
    if expected_obs and expected_obs.issubset(dataset_features):
        notes.append("cleared rename_map because dataset already contains policy observation keys")
        return {}, notes
    return dict(rename_map), notes


def patch_json_file(path: Path, replacements: list[tuple[str, str]]) -> bool:
    if not path.exists():
        return False
    old = load_json(path)
    new = recursive_replace(old, replacements)
    if new != old:
        backup_once(path)
        dump_json(path, new)
        return True
    return False


def update_preprocessor(path: Path, *, rename_map: dict[str, str], clip_path: Path | None, replacements: list[tuple[str, str]]) -> bool:
    if not path.exists():
        return False
    old_payload = load_json(path)
    payload = recursive_replace(old_payload, replacements)
    changed = payload != old_payload
    for step in payload.get("steps", []):
        if not isinstance(step, dict):
            continue
        cfg = step.setdefault("config", {})
        if step.get("registry_name") == "rename_observations_processor":
            if cfg.get("rename_map") != rename_map:
                cfg["rename_map"] = rename_map
                changed = True
        if clip_path is not None and step.get("registry_name") == "tokenizer_processor":
            if cfg.get("tokenizer_name") != str(clip_path):
                cfg["tokenizer_name"] = str(clip_path)
                changed = True
    if changed:
        backup_once(path)
        dump_json(path, payload)
        return True
    return False


def patch_checkpoint(
    *,
    output_train_config: Path,
    output_job_dir: Path,
    dataset_root: Path,
    clip_path: Path,
    steps: int,
    rel_job_dir: Path,
    model_kind: str,
    dry_run: bool,
) -> dict[str, Any]:
    train_cfg = load_json(output_train_config)
    notes: list[str] = []
    task_name = task_name_from_rel(rel_job_dir)
    dataset_cfg = train_cfg.setdefault("dataset", {})
    old_dataset_root = dataset_cfg.get("root") if isinstance(dataset_cfg, dict) else None
    dataset_dir, dataset_notes = find_dataset_dir(dataset_root, task_name, old_dataset_root)
    notes.extend(dataset_notes)

    replacements: list[tuple[str, str]] = []
    for old in OLD_DATASET_PREFIXES:
        replacements.append((old, str(dataset_root)))
    for old in OLD_CHECKPOINT_PREFIXES:
        replacements.append((old, str(clip_path.parent)))
    replacements.append((
        "/mnt/workspace/users/xujunzhe/yunhengwang/lerobot/lerobot/checkpoints/clip-vit-base-patch16",
        str(clip_path),
    ))

    train_cfg = recursive_replace(train_cfg, replacements)
    dataset_cfg = train_cfg.setdefault("dataset", {})
    if dataset_dir is not None:
        dataset_cfg["root"] = str(dataset_dir)
        dataset_cfg["repo_id"] = f"local/{dataset_dir.name}"
    train_cfg["output_dir"] = str(output_job_dir)
    train_cfg["job_name"] = output_job_dir.name
    train_cfg["resume"] = True
    train_cfg["steps"] = int(steps)

    rename_map, rename_notes = maybe_clear_rename_map(train_cfg, dataset_dir)
    notes.extend(rename_notes)
    train_cfg["rename_map"] = rename_map

    policy = train_cfg.get("policy")
    if isinstance(policy, dict) and model_kind == "mtp":
        for key in ("vision_encoder_name", "text_encoder_name"):
            if key in policy:
                policy[key] = str(clip_path)

    if not dry_run:
        backup_once(output_train_config)
        dump_json(output_train_config, train_cfg)

        model_config = output_train_config.parent / "config.json"
        if model_config.exists():
            cfg_payload = recursive_replace(load_json(model_config), replacements)
            if model_kind == "mtp":
                for key in ("vision_encoder_name", "text_encoder_name"):
                    if key in cfg_payload:
                        cfg_payload[key] = str(clip_path)
            backup_once(model_config)
            dump_json(model_config, cfg_payload)

        preprocessor = output_train_config.parent / "policy_preprocessor.json"
        update_preprocessor(
            preprocessor,
            rename_map=rename_map,
            clip_path=clip_path if model_kind == "mtp" else None,
            replacements=replacements,
        )
        patch_json_file(output_train_config.parent / "policy_postprocessor.json", replacements)

    return {
        "task_name": task_name,
        "dataset_root": str(dataset_dir) if dataset_dir else None,
        "model_kind": model_kind,
        "output_dir": str(output_job_dir),
        "train_config": str(output_train_config),
        "rename_map": rename_map,
        "notes": notes,
    }


def query_gpus() -> dict[int, dict[str, int]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to query nvidia-smi: {exc}") from exc

    gpus: dict[int, dict[str, int]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        idx, total, used = (int(float(part)) for part in parts)
        gpus[idx] = {"total": total, "used": used}
    return gpus


def read_training_step(job_dir: Path) -> int | None:
    path = job_dir / "checkpoints" / "last" / "training_state" / "training_step.json"
    if not path.exists():
        return None
    try:
        return int(load_json(path).get("step"))
    except Exception:
        return None


def unique_backup_path(path: Path) -> Path:
    backup = path.with_name(path.name + ".dir_backup")
    idx = 0
    while backup.exists() or backup.is_symlink():
        idx += 1
        backup = path.with_name(f"{path.name}.dir_backup_{idx}")
    return backup


def ensure_last_checkpoint_symlink(job_dir: Path) -> str | None:
    checkpoints_dir = job_dir / "checkpoints"
    last_dir = checkpoints_dir / LAST_CHECKPOINT_LINK
    if not last_dir.exists() and not last_dir.is_symlink():
        return None
    if last_dir.is_symlink():
        return None
    if not last_dir.is_dir():
        return f"skip non-directory checkpoints/last: {last_dir}"

    numeric_dirs = sorted(
        path for path in checkpoints_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if numeric_dirs:
        target = numeric_dirs[-1]
        backup = unique_backup_path(last_dir)
        last_dir.rename(backup)
    else:
        step = read_training_step(job_dir)
        if step is None:
            return f"skip checkpoints/last directory without training_step: {last_dir}"
        target = checkpoints_dir / f"{step:06d}"
        if target.exists() or target.is_symlink():
            backup = unique_backup_path(last_dir)
            last_dir.rename(backup)
        else:
            last_dir.rename(target)

    last_dir.symlink_to(target.name, target_is_directory=True)
    return f"normalized checkpoints/last -> {target.name}"


@dataclass
class ResumeJob:
    name: str
    rel_job_dir: str
    model_kind: str
    mem_mib: int
    output_dir: Path
    train_config: Path
    dataset_root: str | None
    notes: list[str] = field(default_factory=list)
    gpu: int | None = None
    log_path: Path | None = None
    process: subprocess.Popen | None = None
    status: str = "pending"
    returncode: int | None = None


def discover_and_prepare(args: argparse.Namespace) -> list[ResumeJob]:
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    jobs: list[ResumeJob] = []
    include_models = set(parse_csv(args.models)) if args.models else None
    include_tasks = set(parse_csv(args.tasks)) if args.tasks else None
    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None
    manifest_rows: list[dict[str, Any]] = []

    train_configs = sorted(input_root.glob("**/checkpoints/last/pretrained_model/train_config.json"))
    for src_train_config in train_configs:
        rel_job_dir = rel_job_dir_from_train_config(input_root, src_train_config)
        if exclude_re and exclude_re.search(str(rel_job_dir)):
            continue
        src_job_dir = input_root / rel_job_dir
        dst_job_dir = output_root / rel_job_dir
        src_cfg = load_json(src_train_config)
        policy = src_cfg.get("policy") if isinstance(src_cfg.get("policy"), dict) else {}
        model_kind = infer_model_kind(src_job_dir.name, policy.get("type") if isinstance(policy, dict) else None)
        task_name = task_name_from_rel(rel_job_dir)

        if include_models and model_kind not in include_models:
            continue
        if include_tasks and task_name not in include_tasks:
            continue
        if model_kind not in args.mem_mib:
            print(f"[WARN] skip {rel_job_dir}: unknown model kind {model_kind!r}", file=sys.stderr)
            continue

        dst_train_config = dst_job_dir / TRAIN_CONFIG_REL
        if not args.dry_run:
            if dst_job_dir.exists() and args.recopy:
                shutil.rmtree(dst_job_dir)
            if not dst_train_config.exists():
                dst_job_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_job_dir, dst_job_dir, dirs_exist_ok=True, symlinks=True)
            last_note = ensure_last_checkpoint_symlink(dst_job_dir)
        else:
            last_note = None

        train_config_to_patch = dst_train_config if not args.dry_run else src_train_config
        prepared = patch_checkpoint(
            output_train_config=train_config_to_patch,
            output_job_dir=dst_job_dir,
            dataset_root=args.dataset_root.resolve(),
            clip_path=args.clip_path.resolve(),
            steps=args.steps,
            rel_job_dir=rel_job_dir,
            model_kind=model_kind,
            dry_run=args.dry_run,
        )

        step = read_training_step(dst_job_dir) if dst_job_dir.exists() else None
        status = "pending"
        notes = list(prepared["notes"])
        if not args.dry_run and last_note:
            notes.append(last_note)
        if step is not None and step >= args.steps:
            status = "completed"
            notes.append(f"already at step {step}")

        job = ResumeJob(
            name=dst_job_dir.name,
            rel_job_dir=str(rel_job_dir),
            model_kind=model_kind,
            mem_mib=args.mem_mib[model_kind],
            output_dir=dst_job_dir,
            train_config=dst_train_config,
            dataset_root=prepared["dataset_root"],
            notes=notes,
            status=status,
        )
        jobs.append(job)
        manifest_rows.append(job_to_dict(job))

    if not args.dry_run:
        dump_json(output_root / MANIFEST_FILENAME, {"created_at": now(), "jobs": manifest_rows})
    return jobs


def command_for_job(args: argparse.Namespace, job: ResumeJob, gpu: int) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = args.pythonpath
    conda_lib = str(args.conda_prefix.resolve() / "lib")
    env["LD_LIBRARY_PATH"] = conda_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    cmd = [
        str(args.python_bin),
        str(args.train_entrypoint),
        f"--config_path={job.train_config}",
        "--resume=true",
        f"--steps={args.steps}",
    ]
    if job.dataset_root:
        cmd.append(f"--dataset.root={job.dataset_root}")
    return cmd, env


def launch_job(args: argparse.Namespace, job: ResumeJob, gpu: int, events_path: Path) -> None:
    logs_dir = args.output_root / "_resume_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    job.log_path = logs_dir / f"{job.rel_job_dir.replace('/', '__')}__gpu{gpu}.log"
    cmd, env = command_for_job(args, job, gpu)
    log_f = job.log_path.open("a", encoding="utf-8")
    log_f.write(f"\n===== launch {now()} gpu={gpu} mem_mib={job.mem_mib} =====\n")
    log_f.write("cmd: " + " ".join(cmd) + "\n")
    log_f.flush()
    job.process = subprocess.Popen(
        cmd,
        cwd=args.lerobot_root,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    job.gpu = gpu
    job.status = "running"
    append_jsonl(events_path, {"time": now(), "event": "launch", "job": job.name, "gpu": gpu, "pid": job.process.pid, "log": str(job.log_path)})


def choose_job_for_gpu(pending: list[ResumeJob], available_mib: int) -> ResumeJob | None:
    fitting = [job for job in pending if job.status == "pending" and job.mem_mib <= available_mib]
    if not fitting:
        return None
    fitting.sort(key=lambda job: (-job.mem_mib, job.name))
    return fitting[0]


def schedule(args: argparse.Namespace, jobs: list[ResumeJob]) -> int:
    gpus_info = query_gpus()
    gpu_ids = args.gpu_ids or sorted(gpus_info)
    missing = [gpu for gpu in gpu_ids if gpu not in gpus_info]
    if missing:
        raise ValueError(f"GPU ids not found by nvidia-smi: {missing}")

    events_path = args.output_root / EVENTS_FILENAME
    running: list[ResumeJob] = []
    reservations = {gpu: 0 for gpu in gpu_ids}
    baseline_used = {gpu: gpus_info[gpu]["used"] for gpu in gpu_ids}
    totals = {gpu: gpus_info[gpu]["total"] for gpu in gpu_ids}

    def stop_all(signum: int, _frame: Any) -> None:
        append_jsonl(events_path, {"time": now(), "event": "signal", "signal": signum})
        for job in running:
            if job.process and job.process.poll() is None:
                try:
                    os.killpg(os.getpgid(job.process.pid), signal.SIGTERM)
                except Exception:
                    job.process.terminate()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    pending = [job for job in jobs if job.status == "pending"]
    completed = [job for job in jobs if job.status == "completed"]
    append_jsonl(events_path, {"time": now(), "event": "start", "pending": len(pending), "completed": len(completed), "gpus": gpus_info})

    while pending or running:
        for job in list(running):
            assert job.process is not None
            rc = job.process.poll()
            if rc is None:
                continue
            job.returncode = rc
            job.status = "succeeded" if rc == 0 else "failed"
            if job.gpu is not None:
                reservations[job.gpu] = max(0, reservations[job.gpu] - job.mem_mib)
            running.remove(job)
            append_jsonl(events_path, {"time": now(), "event": "finish", "job": job.name, "returncode": rc, "log": str(job.log_path)})

        launched_any = False
        gpu_snapshot = query_gpus()
        for gpu in gpu_ids:
            actual_used = gpu_snapshot.get(gpu, {}).get("used", baseline_used[gpu])
            modeled_used = baseline_used[gpu] + reservations[gpu]
            used_for_decision = max(actual_used, modeled_used)
            available = totals[gpu] - used_for_decision - args.reserve_mib
            while available > 0:
                job = choose_job_for_gpu(pending, available)
                if job is None:
                    break
                pending.remove(job)
                reservations[gpu] += job.mem_mib
                launch_job(args, job, gpu, events_path)
                running.append(job)
                launched_any = True
                available -= job.mem_mib
                if args.launch_interval_s > 0:
                    time.sleep(args.launch_interval_s)

        if not launched_any and running:
            time.sleep(args.poll_interval_s)
        elif not running and pending:
            print("No pending job fits available GPU memory. Waiting for external GPU memory to free up.", file=sys.stderr)
            time.sleep(args.poll_interval_s)

    failed = [job for job in jobs if job.status == "failed"]
    append_jsonl(events_path, {"time": now(), "event": "done", "failed": len(failed)})
    dump_json(args.output_root / MANIFEST_FILENAME, {"updated_at": now(), "jobs": [job_to_dict(job) for job in jobs]})
    return 1 if failed else 0


def job_to_dict(job: ResumeJob) -> dict[str, Any]:
    return {
        "name": job.name,
        "rel_job_dir": job.rel_job_dir,
        "model_kind": job.model_kind,
        "mem_mib": job.mem_mib,
        "output_dir": str(job.output_dir),
        "train_config": str(job.train_config),
        "dataset_root": job.dataset_root,
        "gpu": job.gpu,
        "log_path": str(job.log_path) if job.log_path else None,
        "status": job.status,
        "returncode": job.returncode,
        "notes": job.notes,
    }


def parse_mem_mib(raw: str | None) -> dict[str, int]:
    mem = dict(DEFAULT_MEM_MIB)
    if not raw:
        return mem
    for item in parse_csv(raw):
        if "=" not in item:
            raise ValueError(f"invalid --mem-mib item {item!r}, expected model=mib")
        key, value = item.split("=", 1)
        mem[key.strip()] = int(value)
    return mem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patch and resume HumanoidArena LeRobot checkpoints with GPU memory scheduling.")
    parser.add_argument("--input-root", required=True, type=Path, help="Source checkpoint root, e.g. /ai/Yichi/taowen/ckpts/0424_new")
    parser.add_argument("--output-root", required=True, type=Path, help="Destination root for patched resumable checkpoints; input is not modified")
    parser.add_argument("--steps", required=True, type=int, help="Target total training steps for resume")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--clip-path", type=Path, default=DEFAULT_CLIP_PATH)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="GPU ids or ranges, e.g. 0,1,2,3 or 0-7")
    parser.add_argument("--reserve-mib", type=int, default=0, help="Per-GPU memory reserve before launching jobs")
    parser.add_argument("--mem-mib", default=None, help="Override model memory estimates, e.g. act=3700,dp=7100,mtp=13000")
    parser.add_argument("--models", default=None, help="Comma-separated model kinds to include: act,dp,mtp")
    parser.add_argument("--tasks", default=None, help="Comma-separated task dirs to include, e.g. HOI_football,HOI_double_desk")
    parser.add_argument("--exclude-regex", default=None, help="Skip jobs whose relative path matches this regex, e.g. '_resume(/|$)'")
    parser.add_argument("--prepare-only", action="store_true", help="Only copy and patch configs; do not launch training")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without copying, patching, or launching")
    parser.add_argument("--recopy", action="store_true", help="Delete and recopy each output job directory from input before patching")
    parser.add_argument("--lerobot-root", type=Path, default=Path("/ai/Yichi/taowen/HumanoidArena/lerobot"))
    parser.add_argument("--train-entrypoint", default="src/lerobot/scripts/lerobot_train.py")
    parser.add_argument("--python-bin", default=str(DEFAULT_CONDA_PREFIX / "bin/python"))
    parser.add_argument("--pythonpath", default="src")
    parser.add_argument("--conda-prefix", type=Path, default=DEFAULT_CONDA_PREFIX)
    parser.add_argument("--poll-interval-s", type=float, default=30.0)
    parser.add_argument("--launch-interval-s", type=float, default=5.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.gpu_ids = parse_gpu_ids(args.gpus)
    args.mem_mib = parse_mem_mib(args.mem_mib)
    args.output_root = args.output_root.resolve()
    args.input_root = args.input_root.resolve()
    args.lerobot_root = args.lerobot_root.resolve()
    args.train_entrypoint = Path(args.train_entrypoint)
    if not args.train_entrypoint.is_absolute():
        args.train_entrypoint = args.lerobot_root / args.train_entrypoint

    jobs = discover_and_prepare(args)
    print(f"Prepared {len(jobs)} jobs under {args.output_root}")
    for job in jobs:
        print(f"[{job.status}] {job.rel_job_dir} kind={job.model_kind} mem={job.mem_mib}MiB dataset={job.dataset_root}")
        for note in job.notes:
            print(f"  note: {note}")

    if args.dry_run or args.prepare_only:
        return 0
    return schedule(args, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
