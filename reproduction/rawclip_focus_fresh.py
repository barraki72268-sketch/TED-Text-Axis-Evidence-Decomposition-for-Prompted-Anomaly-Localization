"""CPU audit of fresh Figure 4 evidence; no model execution or plotting."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

from .checkpoint_download import digest_file
from .rawclip_focus import read_inputs, original_functions
from .rawclip_focus_run import compare_results


def read_evidence(root: Path) -> tuple[dict, dict]:
    manifest = json.loads((root / 'validation/a10-20261009/figure4-fresh-v7/manifest.json').read_text())
    spec = manifest['archive']
    path = root / spec['path']
    if path.stat().st_size != spec['bytes'] or digest_file(path) != spec['sha256']:
        raise ValueError('Fresh Figure 4 archive hash/size mismatch')
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != list(spec['members']):
            raise ValueError('Fresh Figure 4 member identity mismatch')
        payload = {}
        for name, expected in spec['members'].items():
            raw = archive.read(name)
            if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
                raise ValueError('Fresh Figure 4 member hash/size mismatch')
            payload[name] = raw
    execution = json.loads(payload['figure4-execution.json'])
    plan = json.loads(payload['figure4-plan.json'])
    if execution['plan_sha256'] != hashlib.sha256(payload['figure4-plan.json']).hexdigest():
        raise ValueError('Fresh Figure 4 execution plan binding differs')
    prepared = json.loads((root / 'figures/rawclip-focus/linux-preparation.json').read_text())
    if execution['plan_sha256'] != prepared['plan_sha256']:
        raise ValueError('Fresh Figure 4 plan differs from independently validated preparation')
    policy = 'current_recorded_path_bank_preserved_bytes'
    if plan['source_bank_policy'] != policy or execution['source_bank_policy'] != policy:
        raise ValueError('Fresh Figure 4 bank policy differs')
    if execution.get('returncode') != 0 or not execution.get('finished') or execution['status'] not in {'matched', 'mismatch'}:
        raise ValueError('Fresh Figure 4 collection is not terminal successful')
    if execution['post_run_bank_sha256'] != plan['assets']['bank']['sha256']:
        raise ValueError('Fresh Figure 4 bank bytes changed')
    for item in execution['output_files']:
        raw = payload['results/' + item['name']]
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Fresh Figure 4 output binding differs')
    if len([name for name in payload if name.endswith('.pdf')]) != 4:
        raise ValueError('Fresh Figure 4 PDF coverage differs')
    progress = []
    for line in payload['figure4-execution.log'].decode('utf-8').splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get('stage') == 'aggregated_distribution_progress':
            progress.append(value)
    if not progress or progress[-1]['images_done'] != 1725 or progress[-1]['images_total'] != 1725:
        raise ValueError('Fresh Figure 4 terminal coverage progress is missing')
    return manifest, payload


def audit(root: Path) -> dict:
    import numpy as np
    evidence, payload = read_evidence(root)
    summary = json.loads(payload['results/rawclip_aggregated_distribution.json'])
    arrays = payload['results/rawclip_aggregated_distribution_arrays.npz']
    comparison = compare_results(root, summary, io.BytesIO(arrays))
    execution = json.loads(payload['figure4-execution.json'])
    if comparison != execution['comparison'] or comparison != json.loads(payload['figure4-comparison.json']):
        raise ValueError('Fresh Figure 4 comparison differs; use the recorded NumPy evaluation dependency')
    if not comparison['fresh_summary_independently_verified'] or execution['status'] != comparison['status']:
        raise ValueError('Fresh Figure 4 summary/status binding differs')
    manifest, _ = read_inputs(root)
    plotter_spec = next(row for row in manifest['source_scripts'] if row['path'].endswith('make_rawclip_focus_bigfont_20260505.py'))
    plotter = original_functions(root, plotter_spec, {'subsample', 'auc_from_scores'}, np)
    with np.load(io.BytesIO(arrays), allow_pickle=False) as archive:
        groups = {key: archive[key] for key in archive.files}
    rng = np.random.default_rng(0)
    order = ['baseline_abnormal', 'baseline_hard_fp', 'ours_abnormal', 'ours_hard_fp']
    for key in order:
        plotter['subsample'](groups[key], rng)
    compact = {key: plotter['subsample'](groups[key], rng) for key in order}
    sampled = [{'method': method, 'patches_per_group': len(compact[method + '_abnormal']),
                'auc': plotter['auc_from_scores'](compact[method + '_abnormal'], compact[method + '_hard_fp']),
                'annotation_in_final_graphic': False} for method in ['baseline', 'ours']]
    return {'status': comparison['status'], 'scope': evidence['scope'],
            'evidence_bindings_verified': True, 'fresh_gpu_execution_verified': False,
            'gpu_proof_limit': 'Offline audit binds recorded execution; live Slurm/PID allocation was observed separately.',
            'recorded_allocation': execution['allocation'], 'comparison': comparison,
            'compact_plot_sampling': sampled, 'pdf_visual_review': evidence['pdf_visual_review'],
            'historical_gaps': evidence['historical_gaps']}


def main() -> int:
    report = audit(Path(__file__).resolve().parent)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'matched' else 1


if __name__ == '__main__':
    raise SystemExit(main())
