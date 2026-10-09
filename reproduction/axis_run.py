"""Run prepared Figure 3 collection only inside a verified live GPU Slurm job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import zipfile

from .axis_collection import verify_asset
from .axis_figure import read_inputs
from .checkpoint_download import digest_file
from .datasets import validate_dataset


def require_live_gpu_allocation() -> dict:
    job = os.environ.get('SLURM_JOB_ID', '')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not re.fullmatch(r'\d+', job) or visible in {'', '-1', 'NoDevFiles'}:
        raise RuntimeError('Figure 3 GPU collection requires an actual GPU Slurm allocation')
    result = subprocess.run(['scontrol', 'show', 'job', job, '-o'], capture_output=True,
                            text=True, check=True, timeout=20)
    fields = dict(token.split('=', 1) for token in result.stdout.split() if '=' in token)
    tres = fields.get('AllocTRES', '')
    allocated = re.search(r'(?:^|,)gres/gpu(?:\:[^,=]+)?=(\d+)(?:,|$)', tres)
    if fields.get('JobId') != job or fields.get('JobState') != 'RUNNING' or not allocated or int(allocated[1]) < 1:
        raise RuntimeError('Slurm job is not a running GPU allocation')
    nodes = subprocess.run(['scontrol', 'show', 'hostnames', fields['NodeList']],
                           capture_output=True, text=True, check=True, timeout=20).stdout.split()
    hostname = socket.gethostname().split('.')[0]
    if hostname not in {name.split('.')[0] for name in nodes}:
        raise RuntimeError('Current host is not allocated to this Slurm job')
    listing = subprocess.run(['scontrol', 'listpids', job], capture_output=True,
                             text=True, check=True, timeout=20).stdout
    memberships = [line.split() for line in listing.splitlines()[1:]]
    current = [row for row in memberships if len(row) >= 3 and row[0] == str(os.getpid()) and row[1] == job]
    if len(current) != 1:
        raise RuntimeError('Current process is not tracked inside the Slurm allocation')
    return {'job_id': job, 'hostname': hostname, 'cuda_visible_devices': visible,
            'state': fields['JobState'], 'allocated_tres': tres,
            'reservation': fields.get('Reservation'), 'end_time': fields.get('EndTime'),
            'pid': os.getpid(), 'step_id': current[0][2]}


def validate_prepared(root: Path, workspace: Path) -> dict:
    workspace = workspace.resolve()
    manifest, _, _ = read_inputs(root)
    spec = manifest['fresh_collection']
    plan = json.loads((workspace / 'figure3-plan.json').read_text())
    source = workspace / 'source'
    if plan['cwd'] != str(source) or plan['source_bank_policy'] != 'fresh_source_only':
        raise ValueError('Figure 3 runtime/source-bank policy differs')
    for name in ['results', 'bank-cache']:
        folder = workspace / name
        if folder.is_symlink() or not folder.is_dir() or any(folder.iterdir()):
            raise ValueError('Figure 3 output and bank cache must be empty new directories')
    if (workspace / 'figure3-execution.json').exists():
        raise FileExistsError('Figure 3 execution already exists; preserve it')
    mapping = {
        '/mnt/data/pilab-kingjinyoung/ted-reproduce-20261008/snapshot': str(source),
        '/mnt/data/pilab-kingjinyoung/ADPretrain-CLIP': str(source),
        '/mnt/data/hf-cache/anomalyclip': str(workspace / 'clip-cache'),
    }
    if plan['path_mapping'] != mapping:
        raise ValueError('Figure 3 path relocation differs')
    index = json.loads((root / 'source-manifest.json').read_text())['files']
    with zipfile.ZipFile(root / 'source.zip') as archive:
        for entry in index:
            raw = archive.read(entry['path'])
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('Figure 3 archived source changed')
            if entry['path'].endswith('.py'):
                text = raw.decode('utf-8')
                for old, new in mapping.items():
                    text = text.replace(old, new)
                raw = text.encode('utf-8')
            if digest_file(source / entry['path']) != hashlib.sha256(raw).hexdigest():
                raise ValueError('Figure 3 prepared source differs beyond path relocation')
    for name in ['checkpoint', 'backbone']:
        asset = plan['assets'][name]
        verify_asset(Path(asset['path']), spec[name])
    checkpoint = source / spec['checkpoint']['path']
    backbone = workspace / 'clip-cache' / spec['backbone']['filename']
    for name, path in [('checkpoint', checkpoint), ('backbone', backbone)]:
        if not path.is_symlink() or path.resolve() != Path(plan['assets'][name]['path']).resolve():
            raise ValueError('Figure 3 asset link differs')
    for name in ['mvtec', 'visa']:
        prepared = plan['datasets'][name]
        metadata = workspace / 'data' / name / 'meta.json'
        if prepared['prepared_metadata'] != str(metadata) or digest_file(metadata) != prepared['prepared_metadata_sha256']:
            raise ValueError('Figure 3 dataset metadata differs')
        checked = validate_dataset(root / 'datasets' / name, Path(prepared['roots']['images']),
                                   Path(prepared['roots']['masks']))
        if checked['file_errors']:
            raise ValueError('Figure 3 dataset input changed')
    command = plan['command']
    if command[:3] != ['{python}', '-u', str(source / manifest['collector']['path'])]:
        raise ValueError('Figure 3 collector command differs')
    summary = spec['observed_summary']
    meta = summary['bank_cache_meta']
    helper = spec['model_helper']
    expected = {
        '--source_root': workspace / 'data/mvtec', '--source_dataset': 'mvtec',
        '--target_root': workspace / 'data/visa', '--target_dataset': 'visa',
        '--target_mode': summary['target_mode'], '--target_class': summary['target_class'],
        '--checkpoint_path': checkpoint, '--save_dir': workspace / 'results', '--device': 'cuda:0',
        '--image_size': meta['image_size'], '--depth': helper['depth'], '--n_ctx': helper['n_ctx'],
        '--t_n_ctx': helper['t_n_ctx'], '--hard_frac': summary['hard_frac'], '--tau': summary['tau'],
        '--max_good_per_class': meta['max_good_per_class'], '--max_defect_per_class': meta['max_defect_per_class'],
        '--max_bank_per_layer': meta['max_bank_per_layer'], '--max_fp_per_image': meta['max_fp_per_image'],
        '--max_defect_per_image': meta['max_defect_per_image'],
        '--max_normal_eval_images': summary['max_normal_eval_images'],
        '--max_anomaly_eval_images': summary['max_anomaly_eval_images'],
        '--bank_cache_path': workspace / 'bank-cache/fresh-source-bank.pt',
    }
    rebuilt = ['{python}', '-u', str(source / manifest['collector']['path'])]
    for flag, value in expected.items():
        rebuilt.extend([flag, str(value)])
    rebuilt.extend(['--features_list', *map(str, meta['features_list']), '--refresh_bank_cache'])
    if command != rebuilt:
        raise ValueError('Figure 3 numerical or path arguments differ')
    return plan


def compare_arrays(root: Path, fresh: dict) -> dict:
    import numpy as np
    manifest, prior, _ = read_inputs(root)
    expected_summary = prior['summary']
    for key in ['source_dataset','target_dataset','target_mode','target_class',
                'hard_frac','tau','max_normal_eval_images','max_anomaly_eval_images']:
        if fresh['summary'][key] != expected_summary[key]:
            raise ValueError('Fresh Figure 3 diagnostic setting differs: ' + key)
    for key, value in expected_summary['bank_cache_meta'].items():
        if key not in {'checkpoint_path', 'source_root'} and fresh['summary']['bank_cache_meta'][key] != value:
            raise ValueError('Fresh Figure 3 source-bank setting differs: ' + key)
    if set(fresh['groups']) != set(prior['groups']):
        raise ValueError('Fresh Figure 3 group coverage differs')
    arrays = []
    for name, original in prior['groups'].items():
        expected = np.asarray(original, dtype=np.float32)
        actual = np.asarray(fresh['groups'][name], dtype=np.float32)
        if actual.ndim != 1 or not np.isfinite(actual).all():
            raise ValueError('Invalid Figure 3 fresh group')
        same_shape = actual.shape == expected.shape
        arrays.append({'group': name, 'actual_count': int(actual.size), 'expected_count': int(expected.size),
                       'exact_match': same_shape and bool(np.array_equal(actual, expected)),
                       'max_absolute_error': float(np.max(np.abs(actual - expected))) if same_shape and actual.size else None})
    annotations = []
    for panel in manifest['panels']:
        actual = fresh['panel_summary'][panel['stored_panel']]
        for key, printed in [('auc_inside_vs_hardfp', panel['printed_inside_vs_hard_fp']),
                             ('auc_inside_vs_outside', panel['printed_inside_vs_outside'])]:
            value = float(actual[key])
            annotations.append({'panel': panel['group_prefix'], 'metric': key, 'actual': value,
                                'printed': printed, 'matches_printed_3dp': f'{value:.3f}' == f'{printed:.3f}'})
    return {'scope': 'Fresh collector arrays and six AUC annotations versus pinned archival diagnostics; no final PDF layout claim',
            'arrays': arrays, 'annotations': annotations,
            'all_arrays_exact': all(row['exact_match'] for row in arrays),
            'all_annotations_match_3dp': all(row['matches_printed_3dp'] for row in annotations),
            'fresh_bank_stats': fresh['summary']['bank_stats'], 'archived_bank_stats': prior['summary']['bank_stats']}


def run(root: Path, workspace: Path) -> dict:
    allocation = require_live_gpu_allocation()
    workspace = workspace.resolve()
    plan = validate_prepared(root, workspace)
    allocation = require_live_gpu_allocation()  # Inputs may take time to hash.
    record = workspace / 'figure3-execution.json'
    execution = {'status': 'running', 'started': datetime.now(timezone.utc).isoformat(),
                 'plan_sha256': digest_file(workspace / 'figure3-plan.json'), 'allocation': allocation,
                 'argument_provenance': plan['argument_provenance'], 'historical_seed': plan['historical_seed']}
    with record.open('x', encoding='utf-8') as stream:
        json.dump(execution, stream, indent=2)
    try:
        command = [sys.executable, str(root / 'axis_worker.py'), plan['command'][2],
                   str(workspace / 'figure3-initial-rng.json'), *plan['command'][3:]]
        with (workspace / 'figure3-execution.log').open('x') as log:
            environment = dict(os.environ, ANOMALYCLIP_CACHE_DIR=str(workspace / 'clip-cache'),
                               MPLCONFIGDIR=str(workspace / 'mpl-cache'))
            execution['environment_path_overrides'] = {key: environment[key] for key in
                                                       ['ANOMALYCLIP_CACHE_DIR', 'MPLCONFIGDIR']}
            result = subprocess.run(command, cwd=plan['cwd'], env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        execution.update(returncode=result.returncode, status='failed' if result.returncode else 'completed')
        if result.returncode == 0:
            fresh_path = workspace / 'results/axis_entanglement_mvtec2visa.json'
            comparison = compare_arrays(root, json.loads(fresh_path.read_text()))
            with (workspace / 'figure3-comparison.json').open('x', encoding='utf-8') as stream:
                json.dump(comparison, stream, indent=2)
            execution.update(comparison=comparison, fresh_sha256=digest_file(fresh_path),
                             status='matched' if comparison['all_arrays_exact'] and comparison['all_annotations_match_3dp'] else 'mismatch')
    except Exception as error:
        execution.update(status='failed', error=f'{type(error).__name__}: {error}')
    execution['finished'] = datetime.now(timezone.utc).isoformat()
    record.write_text(json.dumps(execution, indent=2) + '\n', encoding='utf-8')
    return execution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--validate-only', action='store_true', help='CPU input/command validation; creates no model or output')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.validate_only:
        validate_prepared(root, args.workspace)
        print(json.dumps({'status':'validated', 'gpu_execution':False}))
        return 0
    execution = run(root, args.workspace)
    print(json.dumps(execution, indent=2))
    return 0 if execution['status'] == 'matched' else 1


if __name__ == '__main__':
    raise SystemExit(main())
