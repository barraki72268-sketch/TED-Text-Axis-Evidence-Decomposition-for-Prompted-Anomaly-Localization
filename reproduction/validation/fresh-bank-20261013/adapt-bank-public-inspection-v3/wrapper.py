"""CPU inspection and synthetic cache-format compatibility, not bank mining."""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import torch

base=Path('/home/jinyoungkim/ted-reproduce-20261009')
repo=base/'github-adapt-bank-public-20261013-v2'
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='4cc102cfab7be89980621ad8776771490ab0ef0c'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
assert not os.environ.get('CUDA_VISIBLE_DEVICES')
sys.path.insert(0,str(repo))
workspace=base/'adapt-cold-runs-20261013-v3/adaptclip-vitl14openai-mvtec2btad-seed0'
sys.argv=['adapt_bank_build',str(workspace),'--inspect']
output=io.StringIO()
with contextlib.redirect_stdout(output):
    runpy.run_module('reproduction.adapt_bank_build',run_name='__main__')
inspection=json.loads(output.getvalue().splitlines()[-1])
assert not inspection['bank_collection_performed'] and not inspection['GPU_execution_performed']
destination=base/'fresh-banks/adapt-public-no-allocation-20261013-v3'
assert not destination.exists()
sys.argv=['adapt_bank_build',str(workspace),str(destination)]
try:runpy.run_module('reproduction.adapt_bank_build',run_name='__main__')
except RuntimeError as e:assert 'actual GPU Slurm allocation' in str(e)
else:raise AssertionError('Unallocated collection accepted')
assert not destination.exists()

# Load only original cache/protocol AST definitions; no model imports or images.
scripts=workspace/'source/neurips2026/scripts'
plain=ast.parse((scripts/'official_parallel_test_adaptclip.py').read_bytes())
branch=ast.parse((scripts/'analyze_adaptclip_vl_score_forms.py').read_bytes())
scope=dict(Path=Path,torch=torch)
definitions=[n for n in plain.body if isinstance(n,ast.FunctionDef) and n.name in {'save_bank_cache','load_bank_cache'}]
definitions += [n for n in branch.body if isinstance(n,ast.FunctionDef) and n.name=='get_or_collect_vl_banks']
assert len(definitions)==3
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*definitions],type_ignores=[])),'<original-cache-functions>','exec'),scope)
def no_mining(**kwargs):raise AssertionError('Synthetic consumer test attempted source collection')
scope['collect_vl_source_banks']=no_mining
getter=next(n for n in definitions if n.name=='get_or_collect_vl_banks')
meta=next(n for n in getter.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='requested_meta' for t in n.targets))
fixture=base/'synthetic-cache-format/adapt-20261013-v3';fixture.mkdir(parents=True,exist_ok=False)
a=inspection['recorded_arguments']
results=[]
for name in ['vl','tl']:
    kwargs={k:a[k] for k in ['source_root','source_dataset','image_size','features_list','sigma','fusion_type','device','hard_frac','max_good_per_class','max_defect_per_class','max_bank_points','max_fp_per_image','max_defect_per_image','checkpoint_path']}
    kwargs.update(model=None,textual_learner=None,visual_learner=None,learned_text_features=None,dpam_layer=0,
                  bank_mining_source=name,bank_feature_source=name,bank_cache_path=str(fixture/(name+'.pt')))
    meta_scope=dict(kwargs)
    exec(compile(ast.Module(body=[meta],type_ignores=[]),'<original-meta>','exec'),meta_scope)
    fp=torch.arange(7*768,dtype=torch.float32).reshape(7,768)/1000
    defect=torch.arange(5*768,dtype=torch.float32).reshape(5,768)/1000
    stats=dict(fixture='synthetic-cache-format-only',num_fp=7,num_def=5)
    path=Path(kwargs['bank_cache_path'])
    scope['save_bank_cache'](path,{0:fp},{0:defect},{0:stats},meta_scope['requested_meta'])
    got_fp,got_def,got_stats,got_path,got_meta=scope['get_or_collect_vl_banks'](**kwargs)
    assert torch.equal(fp,got_fp) and torch.equal(defect,got_def)
    assert got_stats==stats and got_meta==meta_scope['requested_meta'] and got_path==path
    results.append(dict(branch=name,exact_tensors=True,metadata_equal=True,original_cache_getter_reused_without_mining=True,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof=dict(status='CPU_public_inspection_and_synthetic_cache_consumer_verified',clean_public_checkout=True,
           public_commit='4cc102cfab7be89980621ad8776771490ab0ef0c',recipe=inspection['recipe'],
           actual_CLI_inspection=True,code_sha256=sha(repo/'reproduction/adapt_bank_build.py'),
           helper_sha256=sha(Path(__file__)),plan_sha256=sha(workspace/'run.json'),
           unallocated_collection_rejected=True,destination_created=False,
           bank_collection_performed=False,GPU_submitted=False,target_evaluation_performed=False,
           synthetic_cache_consumer_results=results,
           synthetic_test_is_not_source_bank_reconstruction=True,inspection=inspection)
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
with (base/'reports/adapt-bank-public-check-20261013-v3.json').open('x') as f:f.write(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof),flush=True)
