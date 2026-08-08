#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

EVAL_ROOT = Path('/ai/Yichi/taowen/HumanoidArena/isaaclab_twist2_g1/script/eval_scripts')
MODE_ROOTS = {
    'sonic': EVAL_ROOT / 'sonic' / 'eval_results',
    'twist2': EVAL_ROOT / 'twist2' / 'eval_results',
}
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
PATH_RE = re.compile(r'(?<![A-Za-z0-9_.-])/(?:[^\s:]+/)+[^\s:]+')
FLOAT_TIME_RE = re.compile(r'\b\d+\.\d+\b')
INT_RE = re.compile(r'\b\d+\b')

ERROR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ('ort_provider', re.compile(r'(ONNXRuntimeError|loaded .* on CPU instead of cuda|requested GPU .* available_providers|requested GPU ONNX session|WARNING: CUDAExecutionProvider unavailable|failed to load .* on cuda|failed to load .*\[ONNXRuntimeError\])', re.IGNORECASE)),
    ('cuda_runtime', re.compile(r'(CUDNN failure|CUBLAS failure|CUBLAS_STATUS|CUDNN_STATUS|libcublas|libonnxruntime_providers_|cuda.*error|cudnnCreate\(|cublasCreate\()', re.IGNORECASE)),
    ('policy_server', re.compile(r'(predict_action_chunk|infer_chunk|stack expects a non-empty TensorList|LeRobot server not ready|HTTP Error 500|Connection refused|RemoteDisconnected|Read timed out)', re.IGNORECASE)),
    ('provider_guard', re.compile(r'(encoder/decoder not loaded|Encoder/Decoder missing during runtime|refusing to fall back to default pose)', re.IGNORECASE)),
    ('worker_failure', re.compile(r'(episode failed:|worker=\d+ failed|crashed before returning results|Can.t instantiate abstract class|Traceback \(most recent call last\):)', re.IGNORECASE)),
]

TRACEBACK_FINAL_RE = re.compile(r'^(?:[A-Za-z_][\w.]*Error|Exception|RuntimeError|ValueError|TypeError|KeyError|AssertionError|FileNotFoundError): .+')


@dataclass(frozen=True)
class Issue:
    category: str
    signature: str
    line: str
    file_path: str
    mode: str
    batch: str
    line_no: int
    log_kind: str


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub('', text)


def normalize_signature(text: str) -> str:
    text = strip_ansi(text).strip()
    text = PATH_RE.sub('<PATH>', text)
    text = re.sub(r'GPU=\d+', 'GPU=<N>', text)
    text = re.sub(r'worker=\d+', 'worker=<N>', text)
    text = re.sub(r'port=\d+', 'port=<N>', text)
    text = re.sub(r'peer=\([^)]*\)', 'peer=(...)', text)
    text = FLOAT_TIME_RE.sub('<NUM>', text)
    text = INT_RE.sub('<N>', text)
    text = re.sub(r'\s+', ' ', text)
    return text[:280]


def detect_mode(path: Path) -> str:
    for part in path.parts:
        if part in ('sonic', 'twist2'):
            return part
    return 'unknown'


def detect_batch(path: Path) -> str:
    for part in path.parts:
        if part.startswith('sonic_batch_') or part.startswith('twist2_batch_'):
            return part
    return 'unknown_batch'


def detect_log_kind(path: Path) -> str:
    name = path.name
    if name.startswith('server__worker_'):
        return 'server'
    return 'episode'


def iter_batches(mode: str, latest_only: bool) -> list[Path]:
    root = MODE_ROOTS[mode]
    if not root.is_dir():
        return []
    batches = sorted([p for p in root.iterdir() if p.is_dir()])
    if latest_only and batches:
        return [batches[-1]]
    return batches


def iter_log_files(batches: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for batch in batches:
        files.extend(sorted(batch.rglob('*.log')))
    return files


def extract_traceback_final(lines: list[str], start_idx: int) -> tuple[str | None, int]:
    final_line: str | None = None
    idx = start_idx + 1
    while idx < len(lines):
        candidate = strip_ansi(lines[idx]).rstrip('\n')
        if TRACEBACK_FINAL_RE.match(candidate.strip()):
            final_line = candidate.strip()
        if idx > start_idx + 1 and candidate.strip() == '':
            break
        idx += 1
    return final_line, idx


def scan_log(path: Path) -> list[Issue]:
    raw_lines = path.read_text(errors='replace').splitlines()
    issues: list[Issue] = []
    seen: set[tuple[str, str]] = set()
    mode = detect_mode(path)
    batch = detect_batch(path)
    log_kind = detect_log_kind(path)

    idx = 0
    while idx < len(raw_lines):
        line = strip_ansi(raw_lines[idx]).rstrip('\n')
        stripped = line.strip()
        if not stripped:
            idx += 1
            continue

        if 'Traceback (most recent call last):' in stripped:
            final_line, next_idx = extract_traceback_final(raw_lines, idx)
            signature_source = final_line or stripped
            signature = normalize_signature(signature_source)
            key = ('traceback', signature)
            if key not in seen:
                seen.add(key)
                issues.append(Issue('traceback', signature, signature_source, str(path), mode, batch, idx + 1, log_kind))
            idx = next_idx
            continue

        for category, pattern in ERROR_PATTERNS:
            if pattern.search(stripped):
                signature = normalize_signature(stripped)
                key = ('traceback', signature)
                if key not in seen:
                    seen.add(key)
                    issues.append(Issue(category, signature, stripped, str(path), mode, batch, idx + 1, log_kind))
                break
        idx += 1

    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Scan sonic/twist2 eval logs and group actionable errors.')
    parser.add_argument('--mode', choices=['sonic', 'twist2', 'both'], default='both')
    parser.add_argument('--batch-dir', action='append', default=[], help='Explicit batch dir(s) to scan.')
    parser.add_argument('--all-batches', action='store_true', help='Scan all batches under each mode instead of only the latest one.')
    parser.add_argument('--max-issues', type=int, default=20, help='Max grouped issues to print.')
    parser.add_argument('--sample-files', type=int, default=3, help='Sample files to print for each grouped issue.')
    parser.add_argument('--json-out', type=str, default='', help='Optional JSON output path.')
    parser.add_argument('--fail-on-issues', action='store_true', help='Exit non-zero if any issues are found.')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_dir:
        batches = [Path(p).expanduser().resolve() for p in args.batch_dir]
    else:
        modes = ['sonic', 'twist2'] if args.mode == 'both' else [args.mode]
        batches = []
        for mode in modes:
            batches.extend(iter_batches(mode, latest_only=not args.all_batches))

    batches = [b for b in batches if b.is_dir()]
    log_files = iter_log_files(batches)
    issues: list[Issue] = []
    for log_file in log_files:
        issues.extend(scan_log(log_file))

    grouped: dict[tuple[str, str], list[Issue]] = defaultdict(list)
    for issue in issues:
        grouped[(issue.category, issue.signature)].append(issue)

    print(f'batches_scanned={len(batches)} log_files_scanned={len(log_files)} unique_file_issues={len(issues)}')
    for batch in batches:
        print(f'  batch={batch}')

    per_mode = Counter(issue.mode for issue in issues)
    per_kind = Counter(issue.log_kind for issue in issues)
    if issues:
        print(f'issues_by_mode={dict(per_mode)}')
        print(f'issues_by_log_kind={dict(per_kind)}')

    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            -len({issue.file_path for issue in item[1]}),
            -len(item[1]),
            item[0][0],
            item[0][1],
        ),
    )
    for rank, ((category, signature), bucket) in enumerate(ranked[: args.max_issues], start=1):
        sample_files = []
        for issue in bucket:
            loc = f"{issue.file_path}:{issue.line_no}"
            if loc not in sample_files:
                sample_files.append(loc)
            if len(sample_files) >= args.sample_files:
                break
        modes = sorted({issue.mode for issue in bucket})
        batches_for_issue = sorted({issue.batch for issue in bucket})[:3]
        file_count = len({issue.file_path for issue in bucket})
        print(f'[{rank}] files={file_count} occurrences={len(bucket)} category={category} modes={modes} batches={batches_for_issue}')
        print(f'    signature={signature}')
        print(f'    sample_line={bucket[0].line}')
        for loc in sample_files:
            print(f'    file={loc}')

    if args.json_out:
        payload = {
            'batches': [str(b) for b in batches],
            'log_files_scanned': len(log_files),
            'unique_file_issues': len(issues),
            'issues_by_mode': dict(per_mode),
            'issues_by_log_kind': dict(per_kind),
            'grouped_issues': [
                {
                    'category': category,
                    'signature': signature,
                    'file_count': len({issue.file_path for issue in bucket}),
                    'occurrences': len(bucket),
                    'sample_line': bucket[0].line,
                    'modes': sorted({issue.mode for issue in bucket}),
                    'batches': sorted({issue.batch for issue in bucket}),
                    'sample_files': [f"{issue.file_path}:{issue.line_no}" for issue in bucket[: args.sample_files]],
                }
                for (category, signature), bucket in ranked
            ],
        }
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        print(f'json_written={out_path}')

    return 1 if (args.fail_on_issues and issues) else 0


if __name__ == '__main__':
    raise SystemExit(main())
