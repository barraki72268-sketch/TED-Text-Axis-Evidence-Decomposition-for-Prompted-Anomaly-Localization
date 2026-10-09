"""Audit Figure 4 archived arrays and statistics without model inference."""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

from .checkpoint_download import digest_file


def read_inputs(root: Path) -> tuple[dict, dict]:
    manifest = json.loads((root / 'figures/rawclip-focus/manifest.json').read_text())
    spec = manifest['archive']
    archive = root / spec['path']
    if archive.stat().st_size != spec['bytes'] or digest_file(archive) != spec['sha256']:
        raise ValueError('Figure 4 archive hash/size mismatch')
    with zipfile.ZipFile(archive) as source:
        if source.namelist() != list(spec['members']):
            raise ValueError('Figure 4 archive member identity mismatch')
        payload = {}
        for name, expected in spec['members'].items():
            raw = source.read(name)
            if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
                raise ValueError('Figure 4 member hash/size mismatch')
            payload[name] = raw
    paper = manifest['original_paper_graphic']
    pdf = payload[Path(paper['path']).name]
    if len(pdf) != paper['bytes'] or hashlib.sha256(pdf).hexdigest() != paper['sha256']:
        raise ValueError('Figure 4 original paper graphic differs')
    return manifest, payload


def original_functions(root: Path, spec: dict, names: set[str], np) -> dict:
    entries = json.loads((root / 'source-manifest.json').read_text())['files']
    indexed = [row for row in entries if row['path'] == spec['path']]
    if len(indexed) != 1 or indexed[0]['sha256'] != spec['sha256']:
        raise ValueError('Figure 4 source manifest binding differs')
    with zipfile.ZipFile(root / 'source.zip') as archive:
        raw = archive.read(spec['path'])
    if hashlib.sha256(raw).hexdigest() != spec['sha256']:
        raise ValueError('Figure 4 original function source hash mismatch')
    tree = ast.parse(raw.decode('utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names or len(nodes) != len(names):
        raise ValueError('Figure 4 original function coverage differs')
    scope = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), spec['path'], 'exec'), scope)
    return scope


def audit(root: Path) -> dict:
    import numpy as np

    manifest, payload = read_inputs(root)
    summary = json.loads(payload['rawclip_aggregated_distribution.json'])
    with np.load(io.BytesIO(payload['rawclip_aggregated_distribution_arrays.npz']), allow_pickle=False) as archive:
        groups = {key: archive[key] for key in archive.files}
    methods, kinds = ['baseline', 'ours'], ['abnormal', 'hard_fp', 'other']
    if set(groups) != {method + '_' + kind for method in methods for kind in kinds}:
        raise ValueError('Figure 4 array group coverage differs')
    collector_spec = next(row for row in manifest['source_scripts'] if row['path'].endswith('generate_rawclip_aggregated_distribution.py'))
    plotter_spec = next(row for row in manifest['source_scripts'] if row['path'].endswith('make_rawclip_focus_bigfont_20260505.py'))
    collector = original_functions(root, collector_spec, {'auc_from_scores'}, np)
    plotter = original_functions(root, plotter_spec, {'auc_from_scores', 'subsample'}, np)
    stats, aucs = [], []
    for method in methods:
        for kind in kinds:
            values = groups[method + '_' + kind]
            if values.ndim != 1 or values.dtype != np.float32 or values.size == 0 or not np.isfinite(values).all():
                raise ValueError('Figure 4 groups must be finite nonempty float32 vectors')
            actual = {'count': int(values.size), 'mean': float(values.mean()), 'std': float(values.std()),
                      'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95))}
            for key, value in actual.items():
                stored = summary['summary'][method][kind][key]
                stats.append({'group': method + '_' + kind, 'metric': key, 'actual': value,
                              'stored': stored, 'matches_exact': value == stored})
        for positive, negative in [('abnormal', 'hard_fp'), ('abnormal', 'other'), ('hard_fp', 'other')]:
            value = collector['auc_from_scores'](groups[method + '_' + positive], groups[method + '_' + negative])
            key = positive + '_vs_' + negative
            stored = summary['pairwise_auc'][method][key]
            aucs.append({'method': method, 'comparison': key, 'actual': value,
                         'stored': stored, 'matches_exact': value == stored})
    # Original main draws four groups for its horizontal plot before the compact
    # plot. Replaying only compact with a reset RNG would change its samples.
    rng = np.random.default_rng(0)
    order = ['baseline_abnormal', 'baseline_hard_fp', 'ours_abnormal', 'ours_hard_fp']
    for key in order:
        plotter['subsample'](groups[key], rng)
    compact = {key: plotter['subsample'](groups[key], rng) for key in order}
    plotted = []
    for method in methods:
        pos, neg = compact[method + '_abnormal'], compact[method + '_hard_fp']
        plotted.append({'method': method, 'patches_per_group': int(pos.size),
                        'auc': plotter['auc_from_scores'](pos, neg),
                        'median_gap': float(np.median(pos) - np.median(neg)),
                        'annotation_in_final_graphic': False})
    passed = all(row['matches_exact'] for row in stats + aucs)
    return {'status': 'matched' if passed else 'mismatch',
            'scope': 'Archival six arrays, 30 summary statistics and six pairwise AUCs; no fresh inference',
            'numpy': np.__version__, 'statistics': stats, 'aucs': aucs,
            'compact_plot_sampling': plotted, 'original_pdf_hash_verified': True,
            'reported_image_counts': summary['counts'],
            'fresh_gpu_collection': 'pending', 'historical_command_and_bank_provenance': 'pending',
            'final_layout_rerender': 'pending'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(Path(__file__).resolve().parent)
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text, end='')
    return 0 if report['status'] == 'matched' else 1


if __name__ == '__main__':
    raise SystemExit(main())
