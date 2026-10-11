import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

base = Path('/home/jinyoungkim/ted-reproduce-20261009')
repo = base / 'github-bayes-fresh-target-bankfree-20261011-v4'
sys.path.insert(0, str(repo))
from reproduction.recipe_lookup import execution_recipe
recipe_id = 'bayespfl-vitb_plus-mvtec2btad-seed0'
recipe = execution_recipe(repo / 'reproduction', recipe_id)
models = {b['sha256'] for b in recipe['path_bindings'].values()
          if b.get('argument') in {'--checkpoint_path', '--pretrained_path'}}
historical = next(b['sha256'] for b in recipe['path_bindings'].values()
                  if b.get('argument') == '--bank_cache_path')
inputs = base / 'bayes-fresh-target-bankfree-inputs-20261011-v4'
(inputs / 'objects').mkdir(parents=True, exist_ok=False)
for sha in models:
    (inputs / 'objects' / sha).symlink_to(base / 'inputs/objects' / sha)
assert set(p.name for p in (inputs / 'objects').iterdir()) == models
denied = []
historical_path = (base / 'inputs/objects' / historical).resolve()
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0]))
        if path.name == historical or path.resolve() == historical_path:
            denied.append(str(path))
            raise PermissionError('Archived bank object read denied')
sys.addaudithook(audit)
try:
    historical_path.open('rb')
except PermissionError:
    pass
else:
    raise AssertionError('Historical bank denial selfcheck failed')
selfcheck_count = len(denied)
workspace = base / 'bayes-fresh-target-bankfree-runs-20261011-v4' / recipe_id
collection_plan = repo / 'reproduction/validation/fresh-bank-20261010/bayes-bplus-public-checkout-v3/run.json'
sys.argv = ['bayes_fresh_evaluation', 'prepare', recipe_id, str(workspace),
            '--objects', str(inputs), '--datasets', str(base / 'reports/bayes-fresh-target-datasets-20261011-v1.json'),
            '--collection', str(base / 'fresh-bank-historical-audit-20261011-v1'),
            '--collection-plan', str(collection_plan)]
try:
    runpy.run_module('reproduction.bayes_fresh_evaluation', run_name='__main__')
except SystemExit as e:
    assert e.code == 0
from reproduction.run import validate_prepared
plan, _ = validate_prepared(repo / 'reproduction', workspace)
assert {o['sha256'] for o in plan['verified_objects']} == models
assert plan['fresh_bank_evaluation']['historical_inputs_required_for_preparation'] is False
assert len(denied) == selfcheck_count
assert not os.environ.get('CUDA_VISIBLE_DEVICES')
sys.argv = ['bayes_fresh_evaluation', 'run', str(workspace)]
try:
    runpy.run_module('reproduction.bayes_fresh_evaluation', run_name='__main__')
except RuntimeError as e:
    assert 'actual GPU Slurm allocation' in str(e)
else:
    raise AssertionError('Unallocated GPU run was accepted')
assert not (workspace / 'execution.json').exists() and not (workspace / 'results').exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
proof = dict(status='prepared_and_validated_cpu_only', actual_cli_executed=True,
             prototype_overlay=True, clean_public_checkout=False,
             workspace=str(workspace), recipe=recipe_id, plan_sha256=sha(workspace/'run.json'),
             helper_sha256=sha(Path(__file__)),
             code_sha256={n:sha(repo/n) for n in ['reproduction/runtime.py','reproduction/run.py','reproduction/bayes_fresh_evaluation.py']},
             model_only_object_sha256=sorted(models), historical_bank_input_sha256=historical,
             historical_bank_read_denial_selfcheck_passed=True, historical_bank_reads_attempted_after_selfcheck=0,
             historical_inputs_required_for_preparation=False, unallocated_gpu_run_rejected=True,
             fresh_bank_sha256=plan['bank_path_changes'][0]['derived_sha256'],
             target_evaluation_performed=False, gpu_execution_performed=False)
out = base / 'reports/bayes-fresh-target-bankfree-preparation-20261011-v4.json'
with out.open('x') as f:
    f.write(json.dumps(proof, indent=2)+'\n')
print(json.dumps(proof), flush=True)
