"""CPU audit of published fresh Figure 3 evidence; does not run a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from .axis_figure import read_inputs, original_functions
from .axis_run import compare_arrays
from .checkpoint_download import digest_file


def read_evidence(root: Path) -> tuple[dict, dict]:
    manifest = json.loads((root / 'validation/a10-20261009/figure3-fresh-v6/manifest.json').read_text())
    spec = manifest['archive']
    path = root / spec['path']
    if path.stat().st_size != spec['bytes'] or digest_file(path) != spec['sha256']:
        raise ValueError('Fresh Figure 3 archive hash/size mismatch')
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != list(spec['members']):
            raise ValueError('Fresh Figure 3 archive member identity mismatch')
        payload = {}
        for name, entry in spec['members'].items():
            raw = archive.read(name)
            if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('Fresh Figure 3 member hash/size mismatch')
            payload[name] = raw
    execution = json.loads(payload['figure3-execution.json'])
    plan = json.loads(payload['figure3-plan.json'])
    if execution['plan_sha256'] != hashlib.sha256(payload['figure3-plan.json']).hexdigest():
        raise ValueError('Fresh Figure 3 execution plan binding differs')
    if plan['source_bank_policy'] != 'fresh_source_only':
        raise ValueError('Fresh Figure 3 source-bank policy differs')
    if execution.get('returncode') != 0 or not execution.get('finished') or execution['status'] not in {'matched', 'mismatch'}:
        raise ValueError('Fresh Figure 3 evaluation is not terminal successful')
    if execution['fresh_sha256'] != hashlib.sha256(payload['results/axis_entanglement_mvtec2visa.json']).hexdigest():
        raise ValueError('Fresh Figure 3 result binding differs')
    return manifest, payload


def audit(root: Path) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    evidence, payload = read_evidence(root)
    manifest, _, _ = read_inputs(root)
    fresh = json.loads(payload['results/axis_entanglement_mvtec2visa.json'])
    comparison = compare_arrays(root, fresh)
    execution = json.loads(payload['figure3-execution.json'])
    if comparison != execution['comparison'] or comparison != json.loads(payload['figure3-comparison.json']):
        raise ValueError('Fresh Figure 3 comparison record differs')
    collector = original_functions(root, manifest['collector'], {'np': np, 'roc_auc_score': roc_auc_score})['safe_auc']
    rows = []
    for panel in manifest['panels']:
        prefix = panel['group_prefix']
        inside = np.asarray(fresh['groups'][prefix + '_inside'], dtype=np.float32)
        for negative, key in [('hard_fp', 'auc_inside_vs_hardfp'), ('outside', 'auc_inside_vs_outside')]:
            value = collector(inside, np.asarray(fresh['groups'][prefix + '_' + negative], dtype=np.float32))
            stored = fresh['panel_summary'][panel['stored_panel']][key]
            if value != stored:
                raise ValueError('Fresh Figure 3 stored AUC differs from array recalculation')
            rows.append({'panel': prefix, 'metric': key, 'recomputed_auc': value, 'stored_auc': stored})
    expected_status = 'matched' if comparison['all_arrays_exact'] and comparison['all_annotations_match_3dp'] else 'mismatch'
    if execution['status'] != expected_status:
        raise ValueError('Fresh Figure 3 execution status differs')
    return {'status': expected_status, 'scope': evidence['scope'],
            'evidence_bindings_verified': True, 'fresh_gpu_execution_verified': False,
            'gpu_proof_limit': 'Offline CPU audit binds records; live allocation was observed separately.',
            'recorded_allocation': execution['allocation'], 'rows': rows,
            'comparison': comparison, 'pdf_visual_review': evidence['pdf_visual_review'],
            'historical_gaps': evidence['historical_gaps']}


def main() -> int:
    report = audit(Path(__file__).resolve().parent)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'matched' else 1


if __name__ == '__main__':
    raise SystemExit(main())
