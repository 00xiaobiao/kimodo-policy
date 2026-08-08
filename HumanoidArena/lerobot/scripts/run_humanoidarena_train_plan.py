#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PLAN_COLUMNS = [
    'enabled',
    'task_name',
    'dataset_kind',
    'dataset_dir_name',
    'dataset_root',
    'model_alias',
    'job_name',
    'output_dir',
    'target_steps',
    'latest_checkpoint_step',
    'latest_checkpoint_train_config',
    'train_status',
    'resume',
    'gpu_ids',
    'runtime_status',
    'log_path',
    'last_error',
    'notes',
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def dump_plan(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=PLAN_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, '') for column in PLAN_COLUMNS})


def load_plan(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def parse_model_aliases(raw_models: list[str] | None) -> set[str] | None:
    if not raw_models:
        return None
    aliases: set[str] = set()
    for item in raw_models:
        for part in str(item).split(','):
            alias = part.strip()
            if alias:
                aliases.add(alias)
    return aliases or None


def load_model_configs(model_config_dir: Path, allowed_aliases: set[str] | None = None) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    available_aliases: set[str] = set()
    for path in sorted(model_config_dir.glob('*.json')):
        if path.name.startswith('scheduler'):
            continue
        cfg = load_json(path)
        alias = str(cfg.get('alias') or path.stem)
        available_aliases.add(alias)
        if allowed_aliases is not None and alias not in allowed_aliases:
            continue
        cfg['_config_path'] = str(path)
        configs[alias] = cfg
    if allowed_aliases is not None:
        missing = sorted(alias for alias in allowed_aliases if alias not in available_aliases)
        if missing:
            raise ValueError(f'Unknown model aliases: {", ".join(missing)}')
    if not configs:
        raise FileNotFoundError(f'No model configs found in {model_config_dir}')
    return configs


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {'1', 'true', 'yes', 'y'}:
        return True
    if lowered in {'0', 'false', 'no', 'n', ''}:
        return False
    return default


def parse_gpu_ids(value: str) -> list[int]:
    value = (value or '').strip()
    if not value:
        return []

    if ',' not in value and value.isdigit() and len(value) > 1:
        # Support shorthand like `01` -> [0, 1] for single-digit GPU ids.
        return [int(ch) for ch in value]

    gpu_ids: list[int] = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        gpu_ids.append(int(part))
    return gpu_ids


def query_gpu_utilization() -> dict[int, int]:
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,utilization.gpu',
                '--format=csv,noheader,nounits',
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
        parts = [part.strip() for part in line.split(',', 1)]
        if len(parts) != 2:
            continue
        try:
            usage[int(parts[0])] = int(float(parts[1]))
        except ValueError:
            continue
    return usage


def gpus_under_utilization_limit(gpu_ids: list[int], max_utilization_percent: int) -> bool:
    if not gpu_ids:
        return False
    usage = query_gpu_utilization()
    if not usage:
        return True
    return all(usage.get(int(gpu_id), max_utilization_percent + 1) <= max_utilization_percent for gpu_id in gpu_ids)


def cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return 'null'
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    return str(value)


def build_repo_id(dataset_root: Path) -> str:
    return f'local/{dataset_root.name}'


def find_latest_checkpoint(output_dir: Path) -> tuple[int | None, Path | None]:
    checkpoints_dir = output_dir / 'checkpoints'
    if not checkpoints_dir.is_dir():
        return None, None
    best_step: int | None = None
    best_cfg: Path | None = None
    for step_dir in checkpoints_dir.iterdir():
        if not step_dir.is_dir() or not step_dir.name.isdigit():
            continue
        cfg_path = step_dir / 'pretrained_model' / 'train_config.json'
        if not cfg_path.exists():
            continue
        step = int(step_dir.name)
        if best_step is None or step > best_step:
            best_step = step
            best_cfg = cfg_path
    if best_cfg is not None:
        return best_step, best_cfg
    last_cfg = checkpoints_dir / 'last' / 'pretrained_model' / 'train_config.json'
    if last_cfg.exists():
        return None, last_cfg
    return None, None


def refresh_row_state(row: dict[str, str]) -> None:
    output_dir = Path(row['output_dir'])
    target_steps = int(row['target_steps'])
    latest_step, latest_cfg = find_latest_checkpoint(output_dir)
    row['latest_checkpoint_step'] = '' if latest_step is None else str(latest_step)
    row['latest_checkpoint_train_config'] = '' if latest_cfg is None else str(latest_cfg.resolve())
    completed = latest_step is not None and latest_step >= target_steps
    row['train_status'] = 'completed' if completed else 'incomplete'


def build_runtime_defaults(model_config_dir: Path) -> dict[str, Any]:
    scheduler_cfg_path = model_config_dir / 'scheduler.json'
    scheduler_cfg = load_json(scheduler_cfg_path) if scheduler_cfg_path.exists() else {}
    return {
        'lerobot_root': scheduler_cfg.get('lerobot_root', '/ai/Yichi/taowen/HumanoidArena/lerobot'),
        'train_entrypoint': scheduler_cfg.get('train_entrypoint', 'src/lerobot/scripts/lerobot_train.py'),
        'python_bin': scheduler_cfg.get('python_bin', sys.executable),
        'pythonpath': scheduler_cfg.get('pythonpath', 'src'),
        'torch_home': scheduler_cfg.get('torch_home', os.environ.get('TORCH_HOME', '')),
        'env': scheduler_cfg.get('env', {}),
        'wandb_api_key': scheduler_cfg.get('wandb_api_key', ''),
        'wandb_project': scheduler_cfg.get('wandb_project', 'HumanoidArena'),
    }


def output_dir_allows_fresh_start(output_dir: Path) -> bool:
    if not output_dir.exists():
        return True
    allowed_entries = {'scheduler_job_spec.json', 'wandb'}
    existing_entries = {path.name for path in output_dir.iterdir()}
    return existing_entries.issubset(allowed_entries)


def build_command(
    row: dict[str, str],
    model_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    assigned_gpus: list[int],
) -> tuple[list[str], dict[str, str]]:
    lerobot_root = Path(runtime_cfg['lerobot_root']).resolve()
    train_script = str((lerobot_root / runtime_cfg['train_entrypoint']).resolve())
    python_bin = str(runtime_cfg['python_bin'])
    if not os.path.isabs(python_bin):
        python_bin = sys.executable

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in runtime_cfg.get('env', {}).items()})
    if runtime_cfg.get('wandb_api_key'):
        env['WANDB_API_KEY'] = str(runtime_cfg['wandb_api_key'])
    if runtime_cfg.get('torch_home'):
        env['TORCH_HOME'] = str(runtime_cfg['torch_home'])
    env['PYTHONPATH'] = str(runtime_cfg.get('pythonpath', 'src'))
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in assigned_gpus)
    env['PYTHONUNBUFFERED'] = '1'

    args = dict(model_cfg['args'])
    args['steps'] = int(row['target_steps'])
    args['dataset.repo_id'] = build_repo_id(Path(row['dataset_root']))
    args['dataset.root'] = row['dataset_root']
    args['output_dir'] = row['output_dir']
    args['job_name'] = row['job_name']
    args['wandb.project'] = runtime_cfg.get('wandb_project', 'HumanoidArena')
    args["wandb.disable_artifact"] = True
    args['wandb.notes'] = (
        f"task={row['task_name']},dataset_kind={row['dataset_kind']},"
        f"dataset={row['dataset_dir_name']},model={row['model_alias']}"
    )

    launcher = str(model_cfg.get('launcher', 'python'))
    if launcher == 'torchrun':
        if not assigned_gpus:
            raise ValueError(f"torchrun job requires gpu_ids: {row['job_name']}")
        cmd = [
            python_bin,
            '-m',
            'torch.distributed.run',
            '--standalone',
            '--nnodes=1',
            f'--nproc_per_node={len(assigned_gpus)}',
            train_script,
        ]
    elif launcher == 'python':
        cmd = [python_bin, train_script]
    else:
        raise ValueError(f"Unsupported launcher: {launcher}")

    if parse_bool(row.get('resume')):
        latest_cfg = row.get('latest_checkpoint_train_config', '').strip()
        if not latest_cfg:
            raise ValueError(f"resume=true but no checkpoint found for {row['job_name']}")
        args['resume'] = True
        args['config_path'] = latest_cfg
    else:
        output_dir = Path(row['output_dir'])
        if not output_dir_allows_fresh_start(output_dir):
            raise ValueError(
                f"output_dir already contains checkpoints or other artifacts for {row['job_name']}; "
                'set resume=true or change output_dir in CSV'
            )

    for key, value in args.items():
        cmd.append(f'--{key}={cli_value(value)}')
    return cmd, env


def maybe_completed_after_run(row: dict[str, str]) -> bool:
    refresh_row_state(row)
    return row['train_status'] == 'completed'


def main() -> int:
    parser = argparse.ArgumentParser(description='Run HumanoidArena training plan from editable CSV.')
    parser.add_argument('--plan-csv', required=True)
    parser.add_argument('--model-config-dir', default='/ai/Yichi/taowen/HumanoidArena/lerobot/configs/humanoidarena_batch_train')
    parser.add_argument('--models', nargs='*', default=None, help='Optional model aliases to run, e.g. --models pi05 mtp or --models pi05,mtp')
    parser.add_argument('--poll-interval-seconds', type=int, default=15)
    parser.add_argument('--max-gpu-utilization-percent', type=int, default=100)
    parser.add_argument('--log-dir', default=None)
    args = parser.parse_args()

    plan_csv = Path(args.plan_csv).resolve()
    model_config_dir = Path(args.model_config_dir).resolve()
    rows = load_plan(plan_csv)
    allowed_aliases = parse_model_aliases(args.models)
    model_configs = load_model_configs(model_config_dir, allowed_aliases=allowed_aliases)
    runtime_cfg = build_runtime_defaults(model_config_dir)
    log_dir = Path(args.log_dir).resolve() if args.log_dir else plan_csv.parent / f'{plan_csv.stem}_logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        refresh_row_state(row)
        if allowed_aliases is not None and row['model_alias'] not in allowed_aliases:
            row['runtime_status'] = row.get('runtime_status', '') or 'filtered_out'
            continue
        if row['train_status'] == 'completed':
            row['runtime_status'] = 'completed'
            row['last_error'] = ''
        elif row.get('runtime_status') == 'running':
            row['runtime_status'] = 'queued'
    dump_plan(plan_csv, rows)

    running: list[dict[str, Any]] = []

    def used_gpu_ids() -> set[int]:
        used: set[int] = set()
        for item in running:
            used.update(item['gpu_ids'])
        return used

    def try_launch_pending_jobs() -> bool:
        launched = False
        # Allow multiple jobs to share the same GPU id; gpu_ids only controls CUDA_VISIBLE_DEVICES.
        for row in rows:
            if allowed_aliases is not None and row['model_alias'] not in allowed_aliases:
                continue
            if not parse_bool(row.get('enabled', '1'), default=True):
                continue
            if row['train_status'] == 'completed':
                row['runtime_status'] = 'completed'
                continue
            if row.get('runtime_status') == 'running':
                continue
            gpu_ids = parse_gpu_ids(row.get('gpu_ids', ''))
            if not gpu_ids:
                row['runtime_status'] = 'waiting_gpu_ids'
                continue
            if not gpus_under_utilization_limit(gpu_ids, args.max_gpu_utilization_percent):
                row['runtime_status'] = 'waiting_gpu_utilization'
                continue
            model_alias = row['model_alias']
            model_cfg = model_configs.get(model_alias)
            if model_cfg is None:
                row['runtime_status'] = 'error'
                row['last_error'] = f'missing model config: {model_alias}'
                continue
            refresh_row_state(row)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = log_dir / f"{row['job_name']}__{timestamp}.log"
            try:
                cmd, env = build_command(row, model_cfg, runtime_cfg, gpu_ids)
            except Exception as exc:
                row['runtime_status'] = 'error'
                row['last_error'] = str(exc)
                continue
            log_fp = log_path.open('w', encoding='utf-8', buffering=1)
            log_fp.write('# command=' + ' '.join(cmd) + '\n')
            log_fp.write('# started_at=' + datetime.now().isoformat(timespec='seconds') + '\n')
            process = subprocess.Popen(
                cmd,
                cwd=str(runtime_cfg['lerobot_root']),
                env=env,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running.append({
                'row': row,
                'gpu_ids': gpu_ids,
                'process': process,
                'log_fp': log_fp,
            })
            row['runtime_status'] = 'running'
            row['log_path'] = str(log_path)
            row['last_error'] = ''
            launched = True
        if launched:
            dump_plan(plan_csv, rows)
        return launched

    while True:
        still_running: list[dict[str, Any]] = []
        for item in running:
            process: subprocess.Popen = item['process']
            row = item['row']
            rc = process.poll()
            if rc is None:
                still_running.append(item)
                continue
            item['log_fp'].flush()
            item['log_fp'].close()
            if maybe_completed_after_run(row):
                row['runtime_status'] = 'completed'
                row['last_error'] = ''
            else:
                row['runtime_status'] = 'failed'
                row['last_error'] = f'process exited with code {rc}'
        running = still_running
        dump_plan(plan_csv, rows)

        launched = try_launch_pending_jobs()
        if not running and not launched:
            break
        time.sleep(args.poll_interval_seconds)

    completed = sum(1 for row in rows if row['train_status'] == 'completed')
    incomplete = len(rows) - completed
    print(f'Plan finished: completed={completed} incomplete={incomplete}')
    print(f'Updated CSV: {plan_csv}')
    if allowed_aliases is not None:
        print('models=' + ','.join(sorted(allowed_aliases)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
