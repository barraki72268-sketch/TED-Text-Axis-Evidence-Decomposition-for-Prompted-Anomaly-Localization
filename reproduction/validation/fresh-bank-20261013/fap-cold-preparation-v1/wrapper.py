import hashlib,json,os,runpy,sys
from pathlib import Path
base=Path('/home/jinyoungkim/ted-reproduce-20261009')
repo=base/'github-fap-cold-prepare-20261013-v1'
sys.path.insert(0,str(repo))
from reproduction.recipe_lookup import execution_recipe
from reproduction.backbone_download import required_backbone_assets
recipe_id='faprompt-vitb16plus-mvtec2btad-seed1'
r=execution_recipe(repo/'reproduction',recipe_id)
read=lambda p:json.loads(p.read_text())
bb=read(repo/'reproduction/backbones.json')
models={a['sha256'] for a in required_backbone_assets(bb,recipe_id)}
ck=read(repo/'reproduction/host-checkpoints.json')
models|={a['sha256'] for b in ck['bindings'] if b['recipe']==recipe_id for a in b['checkpoint_assets']}
inputs=base/'fap-cold-inputs-20261013-v1'
(inputs/'objects').mkdir(parents=True,exist_ok=False)
for h in models:
 source=base/'inputs/objects'/h
 assert source.is_file(),str(source)
 (inputs/'objects'/h).symlink_to(source)
datasets=read(base/'reports/bayes-fresh-target-datasets-20261011-v1.json')
(inputs/'source-roots.json').write_text(json.dumps({'mvtec':datasets['mvtec']}))
banks={a['sha256'] for a in read(repo/'reproduction/host-source-banks.json')['artifacts']}
target_roots={Path(v).resolve() for v in datasets['btad'].values() if v}
blocked=[]
def audit(event,args):
 if event in {'socket.connect','socket.getaddrinfo'}: raise PermissionError('Network disabled')
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).resolve()
  if p.name in banks or any(p==d or d in p.parents for d in target_roots):
   blocked.append(str(p));raise PermissionError('Historical bank and target reads disabled')
sys.addaudithook(audit)
for path in [base/'inputs/objects'/next(iter(banks)),*target_roots]:
 try: path.open('rb')
 except PermissionError: pass
 else: raise AssertionError('Denial selfcheck failed')
count=len(blocked)
w=base/'fap-cold-runs-20261013-v1'/recipe_id
sys.argv=['reproduction','prepare-run',recipe_id,str(w),'--objects',str(inputs),'--datasets',str(inputs/'source-roots.json'),'--bank-collection-only']
try: runpy.run_module('reproduction',run_name='__main__')
except SystemExit as e: assert e.code==0
from reproduction.run import validate_prepared
plan,_=validate_prepared(repo/'reproduction',w,allow_bank_collection=True)
try: validate_prepared(repo/'reproduction',w)
except ValueError as e: assert 'cannot execute target evaluation' in str(e)
else: raise AssertionError('Target evaluation allowed')
assert len(blocked)==count and set(plan['datasets'])=={'mvtec'} and not plan['bank_path_changes']
assert {o['sha256'] for o in plan['verified_objects']}==models
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof=dict(status='prepared_and_validated_cpu_only',recipe=recipe_id,actual_CLI_executed=True,
 prototype_overlay=True,clean_public_checkout=False,source_only=True,model_objects=sorted(models),
 historical_bank_inputs_required=False,target_dataset_inputs_required=False,
 denial_selfchecks=count,blocked_reads_after_selfcheck=0,ordinary_target_evaluation_rejected=True,
 bank_collection_performed=False,GPU_submitted=False,target_evaluation_performed=False,
 plan_sha256=sha(w/'run.json'),helper_sha256=sha(Path(__file__)),
 code_sha256={n:sha(repo/n) for n in ['reproduction/runtime.py','reproduction/run.py','reproduction/__main__.py']})
out=base/'reports/fap-cold-prepare-proof-20261013-v1.json'
with out.open('x') as f:f.write(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof),flush=True)
