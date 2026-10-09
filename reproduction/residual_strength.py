"""Audit the fixed-protocol AA residual-strength table; never select an alpha."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

from .checkpoint_download import digest_file
from .coverage import validate_coverage
from .metrics import compare, extract_recipe
from .recipe_lookup import execution_recipe


def terminal_evidence(root: Path, workspace: Path, recipe: dict, payload: bytes, reference: bytes) -> dict:
    execution = json.loads((workspace / 'execution.json').read_text())
    plan = json.loads((workspace / 'run.json').read_text())
    recorded = json.loads((workspace / 'comparison.json').read_text())
    if execution.get('status') not in {'matched', 'mismatch'} or not execution.get('finished') or execution.get('returncode') != 0:
        raise ValueError('Residual-strength execution is not a successful terminal evaluation')
    if execution['recipe'] != recipe['id'] or plan['recipe'] != recipe['id']:
        raise ValueError('Residual-strength execution recipe binding differs')
    if execution['plan_sha256'] != digest_file(workspace / 'run.json'):
        raise ValueError('Residual-strength execution plan hash differs')
    if plan.get('source_bank_policy') != 'fresh_source_only' or plan['bank_path_changes']:
        raise ValueError('Residual-strength fresh execution substituted a historical bank')
    contract = execution_recipe(root, recipe['id'])
    if len(plan['argv']) != len(contract['argv']):
        raise ValueError('Residual-strength execution argument count differs')
    for expected, actual in zip(contract['argv'], plan['argv']):
        if expected != '{save_dir}' and expected != actual:
            raise ValueError('Residual-strength execution arguments differ')
    cells = compare(extract_recipe(json.loads(payload), contract), extract_recipe(json.loads(reference), contract))
    if recorded != execution['comparison'] or recorded['cells'] != cells:
        raise ValueError('Residual-strength full metric comparison differs')
    if recorded['actual_sha256'] != hashlib.sha256(payload).hexdigest() or recorded['reference_sha256'] != recipe['reference_sha256']:
        raise ValueError('Residual-strength summary/reference binding differs')
    matched = sum(row['matches_printed_precision'] for row in cells)
    if len(cells) != 44 or execution['status'] != ('matched' if matched == 44 else 'mismatch'):
        raise ValueError('Residual-strength terminal metric status differs')
    job = str(execution.get('slurm_job_id', ''))
    if not job.isdigit():
        raise ValueError('Residual-strength execution has no recorded Slurm job')
    return {'status':execution['status'], 'finished':execution['finished'], 'recorded_slurm_job_id':job,
            'execution_sha256':digest_file(workspace / 'execution.json'),
            'plan_sha256':execution['plan_sha256'], 'comparison_sha256':digest_file(workspace / 'comparison.json'),
            'metrics_checked':44, 'metrics_matched_2dp':matched,
            'verification_scope':'Terminal record, plan and all metric bindings; offline audit cannot independently prove past GPU allocation'}


def compare_residual_strength(root: Path, runs: Path | None = None, execution_runs: Path | None = None) -> dict:
    if runs is not None and execution_runs is not None:
        raise ValueError('Choose summary inputs or execution workspaces')
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
                  input='terminal_execution_workspaces' if execution_runs is not None else ('archived_references' if runs is None else 'provided_summaries'),
                  fresh_gpu_execution_verified=False,
                  terminal_execution_records_verified=False,
                  matching_scope='Five printed aggregate cells at one decimal; individual 44-metric comparisons are separate',
                  aggregation=manifest['aggregation'])
    if execution_runs is not None:
        pending = []
        for recipe in recipes:
            workspace = execution_runs / recipe['id']
            required = ['run.json','execution.json','comparison.json','results/summary.json']
            if not all((workspace / name).is_file() for name in required):
                pending.append(recipe['id'])
                continue
            execution = json.loads((workspace / 'execution.json').read_text())
            if not execution.get('finished') or execution.get('status') == 'running':
                pending.append(recipe['id'])
        if pending:
            return dict(report, status='incomplete', pending=pending)
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
            payload = ((execution_runs / recipe['id'] / 'results/summary.json').read_bytes() if execution_runs is not None
                       else (original if runs is None else (runs / recipe['id'] / 'summary.json').read_bytes()))
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
            if execution_runs is not None:
                evidence[-1]['execution'] = terminal_evidence(root, execution_runs / recipe['id'], recipe, payload, original)
    cells = []
    for alpha, printed in zip(manifest['alphas'], manifest['printed_delta_loc']):
        actual = statistics.mean(grouped[alpha])
        cells.append(dict(alpha=alpha, transfers=4, actual_delta_loc_pp=actual,
                          printed=printed, matches_1dp=f'{actual:+.1f}' == printed))
    report['terminal_execution_records_verified'] = execution_runs is not None
    return dict(report, status='matched' if all(r['matches_1dp'] for r in cells) else 'mismatch',
                cells=cells, evidence=evidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--references', action='store_true')
    mode.add_argument('--runs', type=Path)
    mode.add_argument('--executions', type=Path, help='Audit all four terminal execution workspaces and fresh summaries')
    args = parser.parse_args()
    result = compare_residual_strength(Path(__file__).resolve().parent, args.runs, args.executions)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'matched' else 2)
