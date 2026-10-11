import json,os,sys,hashlib
from pathlib import Path
root=Path('/home/jinyoungkim/ted-reproduce-20261009')
repo=root/'github-bayes-cold-public-20261011-v2'
sys.path.insert(0,str(repo))
import subprocess
public_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip()
assert public_commit=='86dfdb4f5ca72707494853289a4276a3fa1996ac'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo)
from reproduction.runtime import prepare_run
from reproduction.run import validate_prepared
from reproduction.bayes_bank_build import recorded_arguments
old=json.loads((root/'bayes-filename-runs-20261009-v1/bayespfl-vitb_plus-mvtec2btad-seed0/run.json').read_text())
source=old['datasets']['mvtec']
inputs=root/'bayes-cold-public-inputs-20261011-v2'
(inputs/'objects').mkdir(parents=True,exist_ok=False)
needed={'b222592c36efdc3ad7e397608bc575bf5a750aa9d7e6563c966367101c863bb4','699c4b843885d82733517f36f0911d7e1b360bcc1314dda81d8c56c76fe9524d'}
for a in old['verified_objects']:
 if a['sha256'] in needed:(inputs/'objects'/a['sha256']).symlink_to(a['path'])
assert {p.name for p in (inputs/'objects').iterdir()}==needed
config=inputs/'datasets.json'
config.write_text(json.dumps({'mvtec':source['roots']}))
denied_files={Path(x['path']).resolve() for x in old['bank_path_changes']}
denied_files.update(Path(x['path']).resolve() for x in old['verified_objects'] if x['sha256'] not in needed)
target=old['datasets']['btad']
denied_roots={Path(target['prepared_metadata']).parent.resolve(),*(Path(p).resolve() for p in target['roots'].values())}
counts={'denied_bank_selfchecks':0,'denied_target_selfchecks':0}
def audit(event,args):
 if event in ('socket.connect','socket.getaddrinfo'):raise PermissionError('No network in cold preparation')
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).resolve()
  if p in denied_files or any(p==d or d in p.parents for d in denied_roots):raise PermissionError('No historical bank/target access in cold preparation')
sys.addaudithook(audit)
for p in denied_files:
 try:p.open('rb')
 except PermissionError:counts['denied_bank_selfchecks']+=1
 else:raise RuntimeError('Bank denial failed')
for p in denied_roots:
 try:(p/'denial-selfcheck').open('rb')
 except PermissionError:counts['denied_target_selfchecks']+=1
 else:raise RuntimeError('Target denial failed')
dest=root/'bayes-cold-public-runs-20261011-v2/bayespfl-vitb_plus-mvtec2btad-seed0'
import runpy
sys.argv=['reproduction','prepare-run',old['recipe'],str(dest),'--objects',str(inputs),'--datasets',str(config),'--bank-collection-only']
try:runpy.run_module('reproduction',run_name='__main__')
except SystemExit as e:assert e.code==0
plan=json.loads((dest/'run.json').read_text())
validate_prepared(repo/'reproduction',dest,allow_bank_collection=True)
try:validate_prepared(repo/'reproduction',dest)
except ValueError as e:
 assert 'cannot execute target evaluation' in str(e)
else:raise RuntimeError('Collection plan accepted as target evaluation')
newargs=vars(recorded_arguments(Path(plan['evaluator']),plan['argv']))
oldargs=vars(recorded_arguments(Path(old['evaluator']),old['argv']))
pathargs={'data_path','target_data_path','checkpoint_path','config_path','pretrained_path','bank_cache_path','save_dir'}
assert {k:v for k,v in newargs.items() if k not in pathargs}=={k:v for k,v in oldargs.items() if k not in pathargs}
r=dict(public_commit=public_commit,public_checkout_clean=True,actual_cli_executed=True,status='matched',cpu_only=True,recipe=plan['recipe'],verified_objects=plan['verified_objects'],datasets=list(plan['datasets']),source_dataset_only=True,historical_bank_inputs_required=False,target_dataset_inputs_required=False,numerical_args_equal_original=True,original_checkpoint_basename=Path(newargs['checkpoint_path']).name,denial_selfchecks=counts,ordinary_evaluation_rejected=True,gpu_collection_verified=False,target_evaluation_verified=False,plan_sha256=hashlib.sha256((dest/'run.json').read_bytes()).hexdigest(),scope='CPU preparation from source data and two model weight objects only, with historical bank and target reads/network blocked. GPU collection on this new preparation not yet executed.')
(root/'reports/bayes-cold-public-prepare-20261011-v2.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r),flush=True)
