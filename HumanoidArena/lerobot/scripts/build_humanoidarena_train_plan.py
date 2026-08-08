#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
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

DEFAULT_DATASET_KIND_RULES = [
    ('twist2', 'twist2'),
    ('sonic', 'sonic'),
    ('merged', 'merge'),
    ('merge', 'merge'),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


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
        raise FileNotFoundError(f'No model config JSON files found in {model_config_dir}')
    return configs


def load_existing_plan(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            job_name = (row.get('job_name') or '').strip()
            if job_name:
                rows[job_name] = row
        return rows


def classify_dataset_kind(name: str) -> str:
    lowered = name.lower()
    for token, kind in DEFAULT_DATASET_KIND_RULES:
        if token in lowered:
            return kind
    head = lowered.split('_', 1)[0].strip()
    return head or lowered


def is_lerobot_dataset_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / 'meta' / 'info.json').is_file()
        and (path / 'meta' / 'tasks.parquet').is_file()
        and (path / 'data').is_dir()
    )


def iter_dataset_dirs(dataset_root: Path):
    if is_lerobot_dataset_dir(dataset_root):
        yield dataset_root.parent, dataset_root
        return

    for task_dir in sorted(dataset_root.iterdir()):
        if not task_dir.is_dir() or task_dir.name == '.cache':
            continue

        if is_lerobot_dataset_dir(task_dir):
            yield dataset_root, task_dir
            continue

        for dataset_dir in sorted(task_dir.iterdir()):
            if is_lerobot_dataset_dir(dataset_dir):
                yield task_dir, dataset_dir


def find_latest_checkpoint_step(output_dir: Path) -> tuple[int | None, Path | None]:
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
    last_cfg = checkpoints_dir / 'last' / 'pretrained_model' / 'train_config.json'
    if best_cfg is None and last_cfg.exists():
        return None, last_cfg
    return best_step, best_cfg


def default_resume_value(latest_step: int | None, latest_cfg: Path | None, target_steps: int) -> str:
    if latest_cfg is None:
        return 'false'
    if latest_step is None:
        return 'true'
    return 'true' if latest_step < target_steps else 'false'


def main() -> int:
    parser = argparse.ArgumentParser(description='Build editable HumanoidArena training plan CSV.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--results-root', required=True)
    parser.add_argument('--model-config-dir', default='/ai/Yichi/taowen/HumanoidArena/lerobot/configs/humanoidarena_batch_train')
    parser.add_argument('--output-csv', default=None)
    parser.add_argument('--models', nargs='*', default=None, help='Optional model aliases to include, e.g. --models pi05 mtp or --models pi05,mtp')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    results_root = Path(args.results_root).resolve()
    model_config_dir = Path(args.model_config_dir).resolve()
    output_csv = Path(args.output_csv).resolve() if args.output_csv else results_root / 'train_plan.csv'

    if not dataset_root.is_dir():
        raise NotADirectoryError(f'dataset_root not found: {dataset_root}')
    results_root.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    allowed_aliases = parse_model_aliases(args.models)
    model_configs = load_model_configs(model_config_dir, allowed_aliases=allowed_aliases)
    existing_rows = load_existing_plan(output_csv)

    rows: list[dict[str, str]] = []
    for task_dir, dataset_dir in iter_dataset_dirs(dataset_root):
        dataset_kind = classify_dataset_kind(dataset_dir.name)
        for model_alias, model_cfg in model_configs.items():
            job_name = f'{model_alias}_{task_dir.name}_{dataset_kind}'
            previous = existing_rows.get(job_name, {})
            output_dir = Path(previous.get('output_dir') or (results_root / job_name)).resolve()
            target_steps = int(model_cfg['args']['steps'])
            latest_step, latest_cfg = find_latest_checkpoint_step(output_dir)
            completed = latest_step is not None and latest_step >= target_steps
            train_status = 'completed' if completed else 'incomplete'
            row = {
                'enabled': previous.get('enabled', '1'),
                'task_name': task_dir.name,
                'dataset_kind': dataset_kind,
                'dataset_dir_name': dataset_dir.name,
                'dataset_root': str(dataset_dir.resolve()),
                'model_alias': model_alias,
                'job_name': job_name,
                'output_dir': str(output_dir),
                'target_steps': str(target_steps),
                'latest_checkpoint_step': '' if latest_step is None else str(latest_step),
                'latest_checkpoint_train_config': '' if latest_cfg is None else str(latest_cfg.resolve()),
                'train_status': train_status,
                'resume': previous.get('resume', default_resume_value(latest_step, latest_cfg, target_steps)),
                'gpu_ids': previous.get('gpu_ids', ''),
                'runtime_status': previous.get('runtime_status', ''),
                'log_path': previous.get('log_path', ''),
                'last_error': previous.get('last_error', ''),
                'notes': previous.get('notes', ''),
            }
            rows.append(row)

    with output_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=PLAN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    completed_count = sum(1 for row in rows if row['train_status'] == 'completed')
    incomplete_count = len(rows) - completed_count
    print(f'Wrote {len(rows)} rows to {output_csv}')
    print(f'completed={completed_count} incomplete={incomplete_count}')
    if allowed_aliases is not None:
        print('models=' + ','.join(sorted(allowed_aliases)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
