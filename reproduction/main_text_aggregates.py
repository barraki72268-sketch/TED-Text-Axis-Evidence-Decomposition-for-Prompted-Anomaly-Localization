"""Verify preserved inputs and reassemble two main-text archival aggregates."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean
import zipfile


def audit(root: Path) -> dict:
    manifest = json.loads((root / 'ablations/main-text-aggregates/manifest.json').read_text(encoding='utf-8'))
    inputs = {}
    evidence = []
    for entry in manifest['inputs']:
        raw = (root / 'ablations/main-text-aggregates' / entry['file']).read_bytes()
        if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Main-text aggregate input hash/size mismatch')
        rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
        if len(rows) != entry['rows']:
            raise ValueError('Main-text aggregate row coverage differs')
        inputs[entry['table']] = rows
        evidence.append(entry)
    with zipfile.ZipFile(root / 'source.zip') as source:
        for entry in manifest['source_scripts']:
            if hashlib.sha256(source.read(entry['path'])).hexdigest() != entry['sha256']:
                raise ValueError('Original aggregate source hash mismatch')
    cells = []

    def compare(table, row, metric, actual, expected, decimals=1):
        if not math.isfinite(float(actual)):
            raise ValueError('Non-finite main-text aggregate')
        cells.append({'table': table, 'row': row, 'metric': metric, 'actual': actual,
                      'printed': expected, 'decimals': decimals,
                      'matches': f'{actual:.{decimals}f}' == f'{expected:.{decimals}f}'})

    rows = inputs['tab:failure_conditioned_gains']
    if {r['group'] for r in rows} != {'Low', 'Mid', 'High'}:
        raise ValueError('Failure-conditioned group coverage differs')
    for group, expected in manifest['failure_conditioned_printed'].items():
        selected = [r for r in rows if r['group'] == group]
        if len({r['setting'] for r in selected}) != 3 or len(selected) != 3:
            raise ValueError('Failure-conditioned setting coverage differs')
        compare('tab:failure_conditioned_gains', group, 'bins', len(selected), 3, 0)
        for metric, key in [('severity', 'severity'), ('PRO', 'delta_PRO'), ('AP', 'delta_AP'), ('Loc', 'delta_Loc')]:
            compare('tab:failure_conditioned_gains', group, metric,
                    mean(float(r[key]) for r in selected), expected[metric], 2 if metric == 'severity' else 1)
    rows = inputs['tab:source_memory_controls:b']
    if len({r['setting'] for r in rows}) != 16:
        raise ValueError('Scalar-source setting coverage differs')
    for mode, expected in manifest['scalar_printed'].items():
        compare('tab:source_memory_controls:b', mode, 'settings', len(rows), 16, 0)
        for metric in ['PRO', 'AP', 'Loc']:
            compare('tab:source_memory_controls:b', mode, metric,
                    mean(float(r[f'{mode}_delta_{metric.lower()}']) for r in rows), expected[metric])
    return {'status': 'matched' if all(c['matches'] for c in cells) else 'mismatch',
            'comparison_scope': 'Table 2 and Table 4(b), preserved CSV aggregation versus printed cells',
            'input_kind': 'archived_aggregate_csv', 'fresh_gpu_execution_verified': False,
            'cells_compared': len(cells), 'cells_matched': sum(c['matches'] for c in cells),
            'evidence': evidence, 'source_scripts': manifest['source_scripts'], 'cells': cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(Path(__file__).resolve().parent)
    text = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        if args.output.exists():
            raise FileExistsError('Preserve previous evidence: choose a new output path')
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')
    return 0 if report['status'] == 'matched' else 1


if __name__ == '__main__':
    raise SystemExit(main())
