"""Audit the fixed-protocol AA residual-strength table; never select an alpha."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

from .checkpoint_download import digest_file
from .coverage import validate_coverage


def compare_residual_strength(root: Path, runs: Path | None = None) -> dict:
    manifest = json.loads((root / 'ablations/residual-strength.json').read_text(encoding='utf8'))
    archive = root / 'ablations' / manifest['archive']
    if digest_file(archive) != manifest['archive_sha256']:
        raise ValueError('Residual-strength reference archive changed')
    sources = {s['path']: s['sha256'] for s in json.loads((root / 'source-manifest.json').read_text())['files']}
    if sources[manifest['launcher']['path']] != manifest['launcher']['sha256']:
        raise ValueError('Residual-strength launcher hash differs from original source')
    recipes = manifest['recipes']
    if len(recipes) != 4 or {r['preset'] for r in recipes} != {'visa2mvtec','mvtec2visa','mvtec2mpdd','mvtec2btad'}:
        raise ValueError('Residual-strength table requires all four AA transfers')
    report = dict(table_label=manifest['table_label'],
                  input='archived_references' if runs is None else 'provided_summaries',
                  fresh_gpu_execution_verified=False,
                  aggregation=manifest['aggregation'])
    if runs is not None:
        missing = [r['id'] for r in recipes if not (runs / r['id'] / 'summary.json').is_file()]
        if missing:
            return dict(report, status='incomplete', missing=missing)
    grouped = {alpha: [] for alpha in manifest['alphas']}
    evidence = []
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != 4 or set(bundle.namelist()) != {r['reference'] for r in recipes}:
            raise ValueError('Unexpected residual-strength reference coverage')
        for recipe in recipes:
            original = bundle.read(recipe['reference'])
            if hashlib.sha256(original).hexdigest() != recipe['reference_sha256']:
                raise ValueError('Residual-strength reference bytes changed')
            payload = original if runs is None else (runs / recipe['id'] / 'summary.json').read_bytes()
            summary = json.loads(payload)
            coverage = validate_coverage(root, dict(host='AA-CLIP', transfer=recipe['preset']), summary)
            if coverage['status'] != 'reported_coverage_matches':
                raise ValueError('Full target coverage is not established')
            if summary['alphas'] != manifest['alphas'] or summary['img_size'] != 224:
                raise ValueError('Residual-strength numerical configuration differs')
            baseline = summary['mean']['baseline']
            for alpha in grouped:
                key = manifest['candidate_prefix'] + f'{alpha:g}'
                candidate = summary['mean']['candidates'][key]
                dp = candidate['pixel_pro'] - baseline['pixel_pro']
                da = candidate['pixel_ap'] - baseline['pixel_ap']
                grouped[alpha].append((dp + da) / 2)
            evidence.append(dict(recipe=recipe['id'], summary_sha256=hashlib.sha256(payload).hexdigest(),
                                 coverage=coverage))
    cells = []
    for alpha, printed in zip(manifest['alphas'], manifest['printed_delta_loc']):
        actual = statistics.mean(grouped[alpha])
        cells.append(dict(alpha=alpha, transfers=4, actual_delta_loc_pp=actual,
                          printed=printed, matches_1dp=f'{actual:+.1f}' == printed))
    return dict(report, status='matched' if all(r['matches_1dp'] for r in cells) else 'mismatch',
                cells=cells, evidence=evidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--references', action='store_true')
    mode.add_argument('--runs', type=Path)
    args = parser.parse_args()
    result = compare_residual_strength(Path(__file__).resolve().parent, args.runs)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'matched' else 2)
