"""Audit Figure 3 archival arrays with both original, hash-pinned AUC functions.

This CPU audit does not collect new model features or recreate final PDF layout.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

from .checkpoint_download import digest_file


def read_inputs(root: Path) -> tuple[dict, dict, dict]:
    manifest = json.loads((root / 'figures/axis-entanglement.json').read_text())
    specification = manifest['archive']
    archive = root / specification['path']
    if archive.stat().st_size != specification['bytes'] or digest_file(archive) != specification['sha256']:
        raise ValueError('Figure 3 archive hash/size mismatch')
    with zipfile.ZipFile(archive) as source:
        if source.namelist() != list(specification['members']):
            raise ValueError('Figure 3 archive member identity mismatch')
        contents = {}
        for name, expected in specification['members'].items():
            raw = source.read(name)
            if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
                raise ValueError('Figure 3 member hash/size mismatch')
            contents[name] = json.loads(raw)
    return manifest, contents['axis_entanglement_mvtec2visa.json'], contents['figure3-axis-data-trace-20261009.json']


def original_functions(root: Path, specification: dict, namespace: dict) -> dict:
    """Execute only two pure functions from the verified archival source member."""
    source_index = json.loads((root / 'source-manifest.json').read_text())['files']
    indexed = [row for row in source_index if row['path'] == specification['path']]
    if len(indexed) != 1 or indexed[0]['sha256'] != specification['sha256']:
        raise ValueError('Figure 3 function source differs from source manifest')
    with zipfile.ZipFile(root / 'source.zip') as archive:
        raw = archive.read(specification['path'])
    if hashlib.sha256(raw).hexdigest() != specification['sha256']:
        raise ValueError('Figure 3 function source hash mismatch')
    tree = ast.parse(raw.decode('utf-8'), filename=specification['path'])
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {'safe_auc', 'sample_array'}]
    if len(functions) != 2 or {node.name for node in functions} != {'safe_auc', 'sample_array'}:
        raise ValueError('Figure 3 function coverage differs')
    module = ast.Module(body=functions, type_ignores=[])
    scope = dict(namespace)
    exec(compile(module, specification['path'], 'exec'), scope)
    return scope


def audit(root: Path) -> dict:
    import numpy as np
    import sklearn
    from sklearn.metrics import roc_auc_score

    manifest, data, prior = read_inputs(root)
    namespace = {'np': np, 'roc_auc_score': roc_auc_score}
    collector = original_functions(root, manifest['collector'], namespace)['safe_auc']
    plotter = original_functions(root, manifest['plotter'], namespace)['safe_auc']
    rows = []
    for panel in manifest['panels']:
        prefix = panel['group_prefix']
        groups = {key: np.asarray(data['groups'][prefix + '_' + key], dtype=np.float32)
                  for key in ['inside', 'hard_fp', 'outside']}
        stored = data['panel_summary'][panel['stored_panel']]
        for negative, summary_key, printed in [
            ('hard_fp', 'auc_inside_vs_hardfp', panel['printed_inside_vs_hard_fp']),
            ('outside', 'auc_inside_vs_outside', panel['printed_inside_vs_outside']),
        ]:
            actual = plotter(groups['inside'], groups[negative])
            collected = collector(groups['inside'], groups[negative])
            previous = [row for row in prior['rows'] if row['panel'] == prefix
                        and row['comparison'] == 'inside_vs_' + negative]
            if len(previous) != 1:
                raise ValueError('Figure 3 prior comparison coverage differs')
            rows.append({
                'panel': prefix, 'comparison': 'inside_vs_' + negative,
                'plotter_auc': actual, 'printed': printed,
                'matches_printed_3dp': f'{actual:.3f}' == f'{printed:.3f}',
                'collector_auc': collected, 'stored_panel_auc': stored[summary_key],
                'matches_stored_exact': collected == stored[summary_key],
                'matches_prior_plotter_exact': actual == previous[0]['actual'],
                'plotter_minus_collector': actual - collected,
                'positive_patches': int(groups['inside'].size),
                'negative_patches': int(groups[negative].size),
            })
    passed = all(row['matches_printed_3dp'] and row['matches_stored_exact']
                 and row['matches_prior_plotter_exact'] for row in rows)
    return {'status': 'matched' if passed else 'mismatch', 'scope': manifest['scope'],
            'input_sha256': manifest['archive']['members']['axis_entanglement_mvtec2visa.json']['sha256'],
            'numpy': np.__version__, 'sklearn': sklearn.__version__, 'rows': rows,
            'fresh_model_collection': 'pending', 'final_pdf_layout_provenance': 'pending'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', action='store_true', required=True,
                        help='Audit archived Figure 3 arrays; does not run a model')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(Path(__file__).resolve().parent)
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        # Preserve prior audit evidence on a repeated invocation.
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text, end='')
    return 0 if report['status'] == 'matched' else 1


if __name__ == '__main__':
    raise SystemExit(main())
