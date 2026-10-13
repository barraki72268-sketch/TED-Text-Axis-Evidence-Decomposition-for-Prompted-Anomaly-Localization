"""Inspect or collect original RawCLIP source patch banks without cache lookup.

CPU inspection does not initialize a model. Collection preserves recorded GPU
settings and requires actual live Slurm process membership.
"""
import argparse
import ast
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

from .adapt_bank_build import evaluate_expression
from .checkpoint_download import digest_file
from .import_paths import isolated_paths
from .run import validate_prepared
from .runtime import DATA_ROOTS,read_json


def original_structure(script):
    tree=ast.parse(script.read_bytes())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    collector=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='collect_source_patch_banks')
    getter=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_or_collect_source_patch_banks')
    parser_end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
        and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='parse_args')
    calls=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id=='get_or_collect_source_patch_banks']
    if len(calls)!=1: raise ValueError('Original bank call structure differs')
    fields={k.arg:k.value for k in calls[0].keywords}
    parameters={p.arg for p in collector.args.args}
    direct={k:v for k,v in fields.items() if k in parameters}
    if set(direct)!=parameters: raise ValueError('Original direct collector signature differs')
    collection_start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and n.value is calls[0])
    initialization=main.body[parser_end+1:collection_start]
    names={n.func.id for node in initialization for n in ast.walk(node)
           if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    if not {'create_winclip_raw_backbone','resolve_features_list','build_transforms'}<=names:
        raise ValueError('Original backbone/transform boundary differs')
    metadata=next(n for n in getter.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='requested_cache_meta' for t in n.targets))
    return main.body[:parser_end+1],initialization,direct,fields,metadata


def parser_arguments(script,argv,torch_module):
    prefix,_,_,_,_=original_structure(script)
    scope=dict(argparse=argparse,torch=torch_module)
    previous=sys.argv
    try:
        sys.argv=[str(script),*argv]
        exec(compile(ast.Module(body=prefix,type_ignores=[]),str(script),'exec'),scope)
    finally: sys.argv=previous
    return scope['args']


def inspect_plan(root,workspace,torch_module):
    plan,recipe=validate_prepared(root,workspace,allow_bank_collection=True)
    if recipe['host']!='RawCLIP' or plan.get('preparation_scope')!='source_bank_collection_only':
        raise ValueError('Use source-only RawCLIP preparation')
    script=Path(plan['evaluator']);args=parser_arguments(script,plan['argv'],torch_module)
    source=recipe['transfer'].split('2',1)[0]
    if args.source_dataset!=source or Path(args.source_root).resolve()!=Path(plan['datasets'][source]['prepared_metadata']).parent.resolve():
        raise ValueError('Prepared source binding differs')
    _,initialization,direct,fields,metadata=original_structure(script)
    return plan,recipe,args,initialization,direct,fields,metadata


def collect(root,workspace,destination):
    from .axis_run import require_live_gpu_allocation
    allocation=require_live_gpu_allocation()
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Allocated GPU unavailable')
    plan,recipe,args,initialization,direct,fields,metadata=inspect_plan(root,workspace,torch)
    if not args.device.startswith('cuda'): raise ValueError('Preserve the recorded GPU device')
    destination.mkdir(parents=True,exist_ok=False);script=Path(plan['evaluator'])
    proof=dict(recipe=plan['recipe'],status='running',allocation=allocation,
        started=datetime.now(timezone.utc).isoformat(),recorded_arguments=vars(args),
        plan_sha256=digest_file(workspace/'run.json'),entrypoint_sha256=digest_file(Path(__file__)),
        collector_sha256=digest_file(script),historical_bank_loaded=False,target_evaluation_performed=False)
    record=destination/'construction.json'
    record.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    try:
        source,target=recipe['transfer'].split('2',1)
        targets={(workspace/'unavailable-target').resolve(),
            (workspace/'source/neurips2026/results/bank_cache').resolve(),
            *(Path(p).resolve() for p in DATA_ROOTS[target])}
        sources={Path(args.source_root).resolve(),
            *(Path(p).resolve() for p in plan['datasets'][source]['roots'].values() if p)}
        if any(s==t or s in t.parents or t in s.parents for s in sources for t in targets):
            raise ValueError('Source and denied roots overlap')
        hashes={a['sha256'] for a in read_json(root/'raw-source-banks.json')['artifacts']}
        sentinel=(workspace/'unavailable-historical-bank.pt').resolve()
        def audit(event,values):
            if event in {'socket.connect','socket.getaddrinfo'}:
                raise PermissionError('Network disabled during source collection')
            if event=='open' and isinstance(values[0],(str,bytes,os.PathLike)):
                p=Path(os.fsdecode(values[0])).resolve()
                if p.name in hashes or p==sentinel or any(p==t or t in p.parents for t in targets):
                    raise PermissionError('Target data and cached banks disabled')
        sys.addaudithook(audit)
        for p in [sentinel,workspace/'unavailable-target/metadata.json',
                  workspace/'source/neurips2026/results/bank_cache/denial-selfcheck.pt']:
            try: p.open('rb')
            except PermissionError: continue
            raise RuntimeError('Source isolation selfcheck failed')
        proof['target_and_cached_bank_reads_denied']=True
        for key,value in plan['environment'].items():
            if key!='CUDA_VISIBLE_DEVICES': os.environ[key]=value
        sys.path[:]=isolated_paths(sys.path,script,plan['cwd'],root)
        spec=importlib.util.spec_from_file_location('ted_original_rawclip_collector',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        scope=dict(module.__dict__,args=args)
        # Original class-name/image-size resolution, backbone, layers and transforms.
        # No additional seed, cache lookup, calibration or target evaluation.
        exec(compile(ast.Module(body=initialization,type_ignores=[]),str(script),'exec'),scope)
        kwargs={k:evaluate_expression(v,scope) for k,v in direct.items()}
        fp,defect,stats=module.collect_source_patch_banks(**kwargs)
        if any(not torch.isfinite(t).all() for bank in [fp,defect] for t in bank.values()):
            raise ValueError('Nonfinite source-bank tensors')
        meta_scope={k:evaluate_expression(v,scope) for k,v in fields.items()
                    if k not in {'bank_cache_path','refresh_bank_cache'}}
        exec(compile(ast.Module(body=[metadata],type_ignores=[]),str(script),'exec'),meta_scope)
        bank=destination/'source-bank.pt'
        module.save_bank_cache(bank,fp,defect,stats,meta_scope['requested_cache_meta'])
        proof.update(status='completed',bank=dict(path=str(bank),sha256=digest_file(bank),bytes=bank.stat().st_size,
            fp_shapes={str(k):list(v.shape) for k,v in fp.items()},defect_shapes={str(k):list(v.shape) for k,v in defect.items()},
            metadata=meta_scope['requested_cache_meta'],stats=stats))
    except Exception as error:
        proof.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        proof['finished']=datetime.now(timezone.utc).isoformat()
        record.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('workspace',type=Path)
    p.add_argument('destination',type=Path,nargs='?');p.add_argument('--inspect',action='store_true')
    opts=p.parse_args();root=Path(__file__).resolve().parent
    if opts.inspect:
        import torch
        plan,recipe,args,initialization,direct,fields,metadata=inspect_plan(root,opts.workspace.resolve(),torch)
        print(json.dumps(dict(recipe=plan['recipe'],recorded_arguments=vars(args),
            original_initialization=ast.unparse(ast.Module(body=initialization,type_ignores=[])),
            direct_collection_call={k:ast.unparse(v) for k,v in direct.items()},
            original_cache_metadata=ast.unparse(metadata),bank_collection_performed=False,GPU_execution_performed=False)))
    else:
        if opts.destination is None: p.error('Collection requires a new destination')
        collect(root,opts.workspace.resolve(),opts.destination.resolve())


if __name__=='__main__': main()
