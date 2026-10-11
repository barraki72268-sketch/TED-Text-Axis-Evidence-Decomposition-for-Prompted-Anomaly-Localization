"""Prepare evaluation with a separately collected BayesPFL source bank.

Preparation retains the full historical-input preflight. This is not bank-free
preparation, and successful preparation does not establish target metrics.
"""
import argparse
import json
from pathlib import Path

from .bayes_bank_build import recorded_arguments
from .checkpoint_download import digest_file
from .recipe_lookup import execution_recipe
from .runtime import prepare_run, read_json

PATH_ARGUMENTS = {'data_path', 'target_data_path', 'checkpoint_path', 'config_path',
                  'pretrained_path', 'bank_cache_path', 'save_dir'}


def verify_collection_settings(proof: dict, plan: dict, arguments: dict):
    if (proof['status'] != 'completed' or proof['recipe'] != plan['recipe']
            or proof.get('historical_bank_loaded') is not False
            or proof.get('target_evaluation_performed') is not False
            or proof.get('target_and_historical_bank_reads_denied') is not True):
        raise ValueError('Fresh source collection provenance differs')
    expected = proof['recorded_arguments']
    if set(expected) != set(arguments) or any(expected[k] != arguments[k]
                                            for k in expected if k not in PATH_ARGUMENTS):
        raise ValueError('Fresh bank collection settings differ from evaluation')
    if Path(expected['checkpoint_path']).name != Path(arguments['checkpoint_path']).name:
        raise ValueError('Fresh bank checkpoint stage differs')


def validate_fresh_binding(root: Path, workspace: Path, plan: dict):
    evidence = plan['fresh_bank_evaluation']
    folder = workspace / 'fresh-bank-evidence'
    for name, sha in evidence['files'].items():
        if digest_file(folder / name) != sha:
            raise ValueError('Fresh bank provenance changed after preparation')
    proof = read_json(folder / 'construction.json')
    collection_plan = read_json(folder / 'collection-run.json')
    if proof['plan_sha256'] != evidence['files']['collection-run.json']:
        raise ValueError('Fresh collection plan is not bound to its proof')
    if collection_plan['recipe'] != plan['recipe']:
        raise ValueError('Fresh collection recipe differs')
    recipe = execution_recipe(root, plan['recipe'])
    if recipe['host'] != 'BayesPFL':
        raise ValueError('Fresh evaluation currently supports BayesPFL only')
    verify_collection_settings(proof, plan, vars(recorded_arguments(Path(plan['evaluator']), plan['argv'])))
    expected_models = {b['sha256'] for b in recipe['path_bindings'].values()
                       if b.get('argument') in {'--checkpoint_path', '--pretrained_path'}}
    collected_objects = {o['sha256'] for o in collection_plan['verified_objects']}
    current_objects = {o['sha256'] for o in plan['verified_objects']}
    if not expected_models <= collected_objects & current_objects:
        raise ValueError('Fresh bank model objects differ')
    if len(plan['bank_path_changes']) != 1:
        raise ValueError('Fresh evaluation requires one explicit bank binding')
    bank = plan['bank_path_changes'][0]
    if (bank['derived_sha256'] != proof['bank_sha256']
            or digest_file(Path(bank['path'])) != proof['bank_sha256']
            or Path(bank['path']).stat().st_size != proof['bank_bytes']):
        raise ValueError('Fresh source bank bytes differ')
    if evidence.get('historical_inputs_required_for_preparation') is not True:
        raise ValueError('Historical preparation scope must be retained')


def prepare_fresh_evaluation(root: Path, recipe: str, destination: Path,
                             object_roots: list[Path], datasets: Path,
                             collection: Path, collection_plan: Path):
    from .run import validate_prepared
    import torch
    bank = (collection / 'source-bank.pt').resolve()
    proof_path = collection / 'construction.json'
    proof = read_json(proof_path)
    if proof['recipe'] != recipe or proof['status'] != 'completed':
        raise ValueError('Collection must have completed for this recipe')
    if digest_file(collection_plan) != proof['plan_sha256']:
        raise ValueError('Collection plan hash differs')
    if digest_file(bank) != proof['bank_sha256'] or bank.stat().st_size != proof['bank_bytes']:
        raise ValueError('Fresh bank hash/size differs')
    payload = torch.load(bank, map_location='cpu', weights_only=True)
    for group in ('fp_banks', 'def_banks'):
        if not payload[group] or any(not torch.isfinite(x).all() for x in payload[group].values()):
            raise ValueError('Fresh bank tensors are invalid')
    plan = prepare_run(root, recipe, destination, object_roots, datasets)
    workspace = destination.absolute()
    validate_prepared(root, workspace)
    verify_collection_settings(proof, plan, vars(recorded_arguments(Path(plan['evaluator']), plan['argv'])))
    folder = workspace / 'fresh-bank-evidence'
    folder.mkdir()
    files = {'construction.json': proof_path.read_bytes(), 'collection-run.json': collection_plan.read_bytes(),
             'historical-preparation-run.json': (workspace / 'run.json').read_bytes()}
    for name, data in files.items():
        (folder / name).write_bytes(data)
    if len(plan['bank_path_changes']) != 1:
        raise ValueError('Expected one BayesPFL historical bank binding')
    prior = plan['bank_path_changes'][0]
    fresh = workspace / 'source/neurips2026/results/bank_cache/fresh-source-bank.pt'
    fresh.symlink_to(bank)
    pos = plan['argv'].index('--bank_cache_path') + 1
    if Path(plan['argv'][pos]) != Path(prior['path']):
        raise ValueError('Historical bank argument binding differs')
    plan['argv'][pos] = str(fresh)
    plan['bank_path_changes'] = [dict(original_sha256=prior['original_sha256'],
        derived_sha256=proof['bank_sha256'], path=str(fresh),
        transformation='New source-only collection; historical preparation retained separately')]
    plan['fresh_bank_evaluation'] = dict(
        files={name: digest_file(folder / name) for name in files},
        historical_inputs_required_for_preparation=True, target_evaluation_performed=False)
    (workspace / 'run.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    validate_prepared(root, workspace)
    return plan


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('recipe')
    prepare.add_argument('destination', type=Path)
    prepare.add_argument('--objects', type=Path, action='append', required=True)
    prepare.add_argument('--datasets', type=Path, required=True)
    prepare.add_argument('--collection', type=Path, required=True)
    prepare.add_argument('--collection-plan', type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('workspace', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.action == 'prepare':
        result = prepare_fresh_evaluation(root, args.recipe, args.destination, args.objects,
                                        args.datasets, args.collection, args.collection_plan)
        print(json.dumps(dict(recipe=result['recipe'], status='prepared_requires_fresh_execution',
                              target_evaluation_performed=False)))
    else:
        from .axis_run import require_live_gpu_allocation
        from .run import run_prepared
        allocation = require_live_gpu_allocation()
        plan = read_json(args.workspace / 'run.json')
        if 'fresh_bank_evaluation' not in plan:
            raise ValueError('Not a fresh-bank evaluation workspace')
        marker = args.workspace / 'fresh-bank-allocation.json'
        if marker.exists():
            raise FileExistsError('Preserve the prior allocation/execution attempt')
        marker.write_text(json.dumps(allocation, indent=2) + '\n', encoding='utf-8')
        result = run_prepared(root, args.workspace, require_slurm=True)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'matched' else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
