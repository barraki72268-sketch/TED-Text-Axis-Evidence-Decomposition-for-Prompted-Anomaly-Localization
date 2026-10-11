import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

base = Path('/home/jinyoungkim/ted-reproduce-20261009')
repo = base / 'github-bayes-fresh-target-cli-20261011-v2'
sys.path.insert(0, str(repo))
workspace = base / 'bayes-fresh-target-cli-runs-20261011-v2/bayespfl-vitb_plus-mvtec2btad-seed0'
collection_plan = repo / 'reproduction/validation/fresh-bank-20261010/bayes-bplus-public-checkout-v3/run.json'
sys.argv = ['bayes_fresh_evaluation', 'prepare', 'bayespfl-vitb_plus-mvtec2btad-seed0', str(workspace),
            '--objects', str(base / 'inputs'), '--datasets', str(base / 'reports/bayes-fresh-target-datasets-20261011-v1.json'),
            '--collection', str(base / 'fresh-bank-historical-audit-20261011-v1'),
            '--collection-plan', str(collection_plan)]
try:
    runpy.run_module('reproduction.bayes_fresh_evaluation', run_name='__main__')
except SystemExit as e:
    assert e.code == 0
from reproduction.run import validate_prepared
plan, recipe = validate_prepared(repo / 'reproduction', workspace)
assert not os.environ.get('CUDA_VISIBLE_DEVICES')
sys.argv = ['bayes_fresh_evaluation', 'run', str(workspace)]
try:
    runpy.run_module('reproduction.bayes_fresh_evaluation', run_name='__main__')
except RuntimeError as e:
    assert 'actual GPU Slurm allocation' in str(e)
else:
    raise AssertionError('Unallocated run was not rejected')
assert not (workspace / 'execution.json').exists()
assert not (workspace / 'results').exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
proof = dict(status='prepared_and_validated_cpu_only', actual_cli_executed=True,
             workspace=str(workspace), recipe=plan['recipe'], plan_sha256=sha(workspace / 'run.json'),
             fresh_bank_sha256=plan['bank_path_changes'][0]['derived_sha256'],
             evidence_files=plan['fresh_bank_evaluation']['files'],
             helper_sha256=sha(Path(__file__)), entrypoint_sha256=sha(repo / 'reproduction/bayes_fresh_evaluation.py'),
             run_validator_sha256=sha(repo / 'reproduction/run.py'),
             unallocated_gpu_run_rejected=True, historical_inputs_required_for_preparation=True,
             target_evaluation_performed=False, gpu_execution_performed=False)
out = base / 'reports/bayes-fresh-target-cli-preparation-20261011-v2.json'
assert not out.exists()
out.write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof), flush=True)
