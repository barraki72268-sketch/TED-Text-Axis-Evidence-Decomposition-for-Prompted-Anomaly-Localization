"""Validate and run reconstructed Figure 4 inference inside a live GPU Slurm job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

from .axis_collection import verify_asset
from .axis_run import require_live_gpu_allocation
from .checkpoint_download import digest_file
from .datasets import load_protocol, validate_dataset
from .rawclip_focus import read_inputs, original_functions
from .rawclip_focus_collection import collector_command
from .source import verify_source


def validate_prepared(root: Path, workspace: Path) -> dict:
    workspace = workspace.resolve()
    manifest, payload = read_inputs(root)
    plan = json.loads((workspace / 'figure4-plan.json').read_text())
    source = workspace / 'source'
    if plan['cwd'] != str(source) or plan['source_bank_policy'] != 'current_recorded_path_bank_preserved_bytes':
        raise ValueError('Figure 4 source/bank policy differs')
    if (workspace / 'figure4-execution.json').exists():
        raise FileExistsError('Figure 4 execution already exists; preserve it')
    output = workspace / 'results'
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
        raise ValueError('Figure 4 results must be an empty normal directory')
    mapping = {
        '/mnt/data/pilab-kingjinyoung/ted-reproduce-20261008/snapshot': str(source),
        '/mnt/data/pilab-kingjinyoung/ADPretrain-CLIP': str(source),
        '/mnt/data/pilab-kingjinyoung/.cache/clip': str(workspace / 'clip-cache'),
        '/home/jinyoung/.cache/clip': str(workspace / 'clip-cache'),
        '~/.cache/clip': str(workspace / 'clip-cache'),
    }
    if plan['path_mapping'] != mapping:
        raise ValueError('Figure 4 source path mapping differs')
    pattern = re.compile('|'.join(re.escape(key) for key in sorted(mapping, key=len, reverse=True)))
    index = verify_source(root / 'source.zip', root / 'source-manifest.json')
    changes = []
    with zipfile.ZipFile(root / 'source.zip') as archive:
        for entry in index['files']:
            raw = archive.read(entry['path'])
            relocated = pattern.sub(lambda match: mapping[match.group()], raw.decode('utf-8')).encode('utf-8') if entry['path'].endswith('.py') else raw
            path = source / entry['path']
            if path.is_symlink() or not path.resolve().is_relative_to(source) or digest_file(path) != hashlib.sha256(relocated).hexdigest():
                raise ValueError('Figure 4 source differs beyond path relocation: ' + entry['path'])
            if relocated != raw:
                changes.append({'path': entry['path'], 'original_sha256': entry['sha256'],
                                'relocated_sha256': hashlib.sha256(relocated).hexdigest()})
    if changes != plan['source_path_changes'] or plan['source']['archive_sha256'] != index['archive_sha256']:
        raise ValueError('Figure 4 source relocation record differs')
    binding = manifest['current_original_bank_trace']
    trace_path = root / binding['path']
    if trace_path.stat().st_size != binding['bytes'] or digest_file(trace_path) != binding['sha256']:
        raise ValueError('Figure 4 current bank trace differs')
    trace = json.loads(trace_path.read_text())
    backbone = next(row for row in json.loads((root / 'backbones.json').read_text())['artifacts']
                    if row['sha256'] == '3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02')
    for name, expected, link in [
        ('bank', trace, workspace / 'bank-cache/recorded-source-bank.pt'),
        ('backbone', backbone, workspace / 'clip-cache' / backbone['filename']),
    ]:
        asset = plan['assets'][name]
        actual = verify_asset(Path(asset['path']), expected)
        if actual != asset or not link.is_symlink() or link.resolve() != Path(asset['path']).resolve():
            raise ValueError('Figure 4 asset/link differs: ' + name)
    prepared = plan['datasets']['mvtec']
    protocol = root / 'datasets/mvtec'
    checked = validate_dataset(protocol, Path(prepared['roots']['images']), Path(prepared['roots']['masks']))
    if checked['file_errors'] or any(checked[key] != prepared[key] for key in checked):
        raise ValueError('Figure 4 full target inputs differ')
    _, metadata = load_protocol(protocol)
    for classes in metadata.values():
        for rows in classes.values():
            for row in rows:
                for field, role in [('img_path', 'images'), ('mask_path', 'masks')]:
                    if row.get(field):
                        row[field] = (Path(checked['roots'][role]) / row[field]).as_posix()
    meta_path = workspace / 'data/mvtec/meta.json'
    if (meta_path.is_symlink() or prepared['prepared_metadata'] != str(meta_path)
            or digest_file(meta_path) != prepared['prepared_metadata_sha256']
            or meta_path.read_bytes() != (json.dumps(metadata, indent=2) + '\n').encode('utf-8')):
        raise ValueError('Figure 4 prepared metadata/path binding differs')
    summary = json.loads(payload['rawclip_aggregated_distribution.json'])
    if plan['command'] != collector_command(workspace, manifest, summary, trace['cache_meta']):
        raise ValueError('Figure 4 original numerical/path arguments differ')
    return plan


def compare_results(root: Path, summary: dict, arrays: Path) -> dict:
    import numpy as np
    manifest, payload = read_inputs(root)
    prior = json.loads(payload['rawclip_aggregated_distribution.json'])
    for key in ['backbone', 'pretrained_dataset', 'target_split', 'class_name', 'hard_fp_frac']:
        if summary[key] != prior[key]:
            raise ValueError('Figure 4 diagnostic setting differs: ' + key)
    with np.load(io.BytesIO(payload['rawclip_aggregated_distribution_arrays.npz']), allow_pickle=False) as archive:
        originals = {key: archive[key] for key in archive.files}
    with np.load(arrays, allow_pickle=False) as archive:
        groups = {key: archive[key] for key in archive.files}
    if set(groups) != set(originals):
        raise ValueError('Figure 4 fresh group coverage differs')
    spec = next(row for row in manifest['source_scripts'] if row['path'].endswith('generate_rawclip_aggregated_distribution.py'))
    original = original_functions(root, spec, {'auc_from_scores'}, np)
    array_checks, statistics, aucs = [], [], []
    for key, values in groups.items():
        if values.ndim != 1 or values.dtype != np.float32 or not values.size or not np.isfinite(values).all():
            raise ValueError('Figure 4 fresh groups must be finite nonempty float32 vectors')
        expected = originals[key]
        same = values.shape == expected.shape
        array_checks.append({'group': key, 'actual_count': int(values.size), 'expected_count': int(expected.size),
                             'exact_match': same and bool(np.array_equal(values, expected)),
                             'max_absolute_error': float(np.max(np.abs(values - expected))) if same else None})
        method, kind = key.split('_', 1)
        stats = {'count': int(values.size), 'mean': float(values.mean()), 'std': float(values.std()),
                 'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95))}
        for metric, value in stats.items():
            stored, archived = summary['summary'][method][kind][metric], prior['summary'][method][kind][metric]
            statistics.append({'group': key, 'metric': metric, 'recomputed': value, 'fresh_stored': stored,
                               'archived': archived, 'matches_fresh_exact': value == stored,
                               'matches_archive_exact': value == archived})
    for method in ['baseline', 'ours']:
        for pos, neg in [('abnormal', 'hard_fp'), ('abnormal', 'other'), ('hard_fp', 'other')]:
            key = pos + '_vs_' + neg
            value = original['auc_from_scores'](groups[method + '_' + pos], groups[method + '_' + neg])
            stored, archived = summary['pairwise_auc'][method][key], prior['pairwise_auc'][method][key]
            aucs.append({'method': method, 'comparison': key, 'recomputed': value, 'fresh_stored': stored,
                         'archived': archived, 'matches_fresh_exact': value == stored,
                         'matches_archive_exact': value == archived,
                         'matches_archive_3dp': f'{value:.3f}' == f'{archived:.3f}'})
    counts = [{'count': key, 'actual': summary['counts'][key], 'archived': value,
               'matches_exact': summary['counts'][key] == value} for key, value in prior['counts'].items()]
    verified = all(row['matches_fresh_exact'] for row in statistics + aucs)
    matched = verified and all(row['exact_match'] for row in array_checks) and all(row['matches_exact'] for row in counts) and all(row['matches_archive_exact'] for row in statistics + aucs)
    return {'status': 'matched' if matched else 'mismatch',
            'scope': 'Current recorded-path bank replay; six arrays, full counts, 30 statistics and six full-array AUCs. Not final compact plot sampling or fresh source-bank rebuild.',
            'numpy': np.__version__, 'arrays': array_checks, 'counts': counts,
            'statistics': statistics, 'aucs': aucs, 'fresh_summary_independently_verified': verified}


def run(root: Path, workspace: Path) -> dict:
    allocation = require_live_gpu_allocation()
    workspace = workspace.resolve()
    plan = validate_prepared(root, workspace)
    allocation = require_live_gpu_allocation()
    record = workspace / 'figure4-execution.json'
    execution = {'status': 'running', 'started': datetime.now(timezone.utc).isoformat(),
                 'plan_sha256': digest_file(workspace / 'figure4-plan.json'), 'allocation': allocation,
                 'argument_provenance': plan['argument_provenance'], 'seed_provenance': plan['seed_provenance'],
                 'source_bank_policy': plan['source_bank_policy']}
    with record.open('x', encoding='utf-8') as stream:
        json.dump(execution, stream, indent=2)
    try:
        command = [sys.executable, str(root / 'rawclip_focus_worker.py'), plan['command'][2],
                   str(workspace / 'figure4-initial-rng.json'), *plan['command'][3:]]
        with (workspace / 'figure4-execution.log').open('x') as log:
            environment = dict(os.environ, MPLCONFIGDIR=str(workspace / 'mpl-cache'))
            execution['environment_path_overrides'] = {'MPLCONFIGDIR': environment['MPLCONFIGDIR']}
            result = subprocess.run(command, cwd=plan['cwd'], env=environment, stdout=log, stderr=subprocess.STDOUT)
        execution.update(returncode=result.returncode, status='failed' if result.returncode else 'completed')
        bank = plan['assets']['bank']
        execution['post_run_bank_sha256'] = digest_file(Path(bank['path']))
        if execution['post_run_bank_sha256'] != bank['sha256']:
            raise ValueError('Figure 4 recorded bank bytes changed during inference')
        if result.returncode == 0:
            output = workspace / 'results'
            summary = output / 'rawclip_aggregated_distribution.json'
            comparison = compare_results(root, json.loads(summary.read_text()), output / 'rawclip_aggregated_distribution_arrays.npz')
            with (workspace / 'figure4-comparison.json').open('x', encoding='utf-8') as stream:
                json.dump(comparison, stream, indent=2)
            execution.update(comparison=comparison, status=comparison['status'],
                             output_files=[{'name': path.name, 'bytes': path.stat().st_size, 'sha256': digest_file(path)} for path in sorted(output.iterdir()) if path.is_file()])
            if len(list(output.glob('*.pdf'))) != 4:
                raise ValueError('Figure 4 collector did not produce its four original PDFs')
    except Exception as error:
        execution.update(status='failed', error=f'{type(error).__name__}: {error}')
    execution['finished'] = datetime.now(timezone.utc).isoformat()
    record.write_text(json.dumps(execution, indent=2) + '\n', encoding='utf-8')
    return execution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.validate_only:
        validate_prepared(root, args.workspace)
        print(json.dumps({'status': 'validated', 'gpu_execution': False}))
        return 0
    execution = run(root, args.workspace)
    print(json.dumps(execution, indent=2))
    return 0 if execution['status'] == 'matched' else 1


if __name__ == '__main__':
    raise SystemExit(main())
