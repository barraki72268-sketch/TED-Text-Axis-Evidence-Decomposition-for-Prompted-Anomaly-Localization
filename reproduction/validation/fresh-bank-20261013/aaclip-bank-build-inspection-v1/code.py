"""Inspect or collect original AA-CLIP source banks without cache lookup.

GPU collection requires actual live Slurm process membership. Inspection only
parses recorded settings and original AST calls; it never initializes a model.
"""
import argparse
import ast
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

from .adapt_bank_build import evaluate_expression
from .checkpoint_download import digest_file
from .import_paths import isolated_paths
from .run import validate_prepared
from .runtime import DATA_ROOTS, read_json


def original_structure(script):
    tree=ast.parse(script.read_bytes())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    collector=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='collect_source_banks')
    getters=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
             and n.func.id=='get_or_collect_source_banks']
    if len(getters)!=1: raise ValueError('Original bank call structure differs')
    direct={k.arg:k.value for k in getters[0].keywords if k.arg not in {'preset','ckpt_dir','cache_dir'}}
    if set(direct)!={p.arg for p in collector.args.args}:
        raise ValueError('Original direct collector signature differs')
    parser_end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
                    and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
                    and n.value.func.attr=='parse_args')
    model_end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='save_dir' for t in n.targets))
    initialization=main.body[parser_end+1:model_end]
    names={n.func.id for node in initialization for n in ast.walk(node)
           if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    if not {'setup_seed','_add_aaclip_to_syspath','ensure_aaclip_pretrained_weight','create_model','AdaptedCLIP'}<=names:
        raise ValueError('Original model initialization boundary differs')
    return tree,main.body[:parser_end+1],initialization,direct


def parser_arguments(script,argv,torch_module):
    tree,prefix,_,_=original_structure(script)
    presets=next(n for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='PRESET_CONFIGS' for t in n.targets))
    scope=dict(argparse=argparse,torch=torch_module)
    exec(compile(ast.Module(body=[presets],type_ignores=[]),str(script),'exec'),scope)
    previous=sys.argv
    try:
        sys.argv=[str(script),*argv]
        exec(compile(ast.Module(body=prefix,type_ignores=[]),str(script),'exec'),scope)
    finally: sys.argv=previous
    args=scope['args'];cfg=dict(scope['PRESET_CONFIGS'][args.preset])
    if args.ckpt_dir: cfg['ckpt_dir']=args.ckpt_dir
    return args,cfg


def inspect_plan(root,workspace,torch_module):
    plan,recipe=validate_prepared(root,workspace,allow_bank_collection=True)
    if recipe['host']!='AA-CLIP' or plan.get('preparation_scope')!='source_bank_collection_only':
        raise ValueError('Use source-only AA-CLIP preparation')
    script=Path(plan['evaluator']);args,cfg=parser_arguments(script,plan['argv'],torch_module)
    source=recipe['transfer'].split('2',1)[0]
    if args.preset!=recipe['transfer'] or cfg['source_dataset']!={'mvtec':'MVTec','visa':'VisA'}.get(source,source):
        raise ValueError('Recorded source/preset differs')
    if Path(cfg['source_root']).resolve()!=Path(plan['datasets'][source]['prepared_metadata']).parent.resolve():
        raise ValueError('Prepared source binding differs')
    verified={Path(x['path']).resolve() for x in plan['verified_objects']}
    for name in ['image_adapter.pth','text_adapter.pth']:
        if (Path(cfg['ckpt_dir'])/name).resolve() not in verified:
            raise ValueError('Original adapter checkpoint is not verified')
    bootstrap=workspace/'source/neurips2026/AA-CLIP/model/ViT-L-14-336px.pt'
    if bootstrap.resolve() not in verified:
        raise ValueError('Original bootstrap weight is not verified')
    _,_,initialization,direct=original_structure(script)
    return plan,recipe,args,cfg,initialization,direct


def collect(root,workspace,destination):
    from .axis_run import require_live_gpu_allocation
    allocation=require_live_gpu_allocation()
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Allocated GPU unavailable')
    plan,recipe,args,cfg,initialization,direct=inspect_plan(root,workspace,torch)
    if not args.device.startswith('cuda'): raise ValueError('Preserve the recorded GPU device')
    destination.mkdir(parents=True,exist_ok=False)
    script=Path(plan['evaluator'])
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
        sources={Path(cfg['source_root']).resolve(),
            *(Path(p).resolve() for p in plan['datasets'][source]['roots'].values() if p)}
        if any(s==t or s in t.parents or t in s.parents for s in sources for t in targets):
            raise ValueError('Source and denied roots overlap')
        bank_hashes={a['sha256'] for a in read_json(root/'host-source-banks.json')['artifacts']}
        sentinel=(workspace/'unavailable-historical-bank.pt').resolve()
        def audit(event,values):
            if event in {'socket.connect','socket.getaddrinfo'}:
                raise PermissionError('Network disabled during source collection')
            if event=='open' and isinstance(values[0],(str,bytes,os.PathLike)):
                p=Path(os.fsdecode(values[0])).resolve()
                if p.name in bank_hashes or p==sentinel or any(p==t or t in p.parents for t in targets):
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
        spec=importlib.util.spec_from_file_location('ted_original_aaclip_collector',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        scope=dict(module.__dict__,args=args)
        # Execute the complete original setup span, including both adapter loads,
        # eval calls and parameter freezing; stop before output/cache/target work.
        exec(compile(ast.Module(body=initialization,type_ignores=[]),str(script),'exec'),scope)
        for name in ['source_root','ckpt_dir']:
            if Path(scope['cfg'][name]).resolve()!=Path(cfg[name]).resolve():
                raise ValueError('Imported original preset binding differs')
        kwargs={k:evaluate_expression(v,scope) for k,v in direct.items()}
        banks=module.collect_source_banks(**kwargs)
        def tensors(value):
            if isinstance(value,torch.Tensor): yield value
            elif isinstance(value,dict):
                for v in value.values(): yield from tensors(v)
            elif isinstance(value,(list,tuple)):
                for v in value: yield from tensors(v)
        if not isinstance(banks,dict) or banks.get('protocol_version')!=module.BANK_PROTOCOL_VERSION:
            raise ValueError('Original bank protocol differs')
        if any(not torch.isfinite(t).all() for t in tensors(banks)):
            raise ValueError('Nonfinite source-bank tensors')
        bank=destination/'source-bank.pt';torch.save(banks,bank)
        proof.update(status='completed',bank=dict(path=str(bank),sha256=digest_file(bank),bytes=bank.stat().st_size,
            fp_shapes=[list(t.shape) for t in banks['fp']],defect_shapes=[list(t.shape) for t in banks['defect']],
            protocol_version=banks['protocol_version'],levels=banks['levels']))
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
        plan,recipe,args,cfg,initialization,direct=inspect_plan(root,opts.workspace.resolve(),torch)
        print(json.dumps(dict(recipe=plan['recipe'],recorded_arguments=vars(args),source_root=cfg['source_root'],
            ckpt_dir=cfg['ckpt_dir'],original_initialization=ast.unparse(ast.Module(body=initialization,type_ignores=[])),
            direct_collection_call={k:ast.unparse(v) for k,v in direct.items()},
            bank_collection_performed=False,GPU_execution_performed=False)))
    else:
        if opts.destination is None: p.error('Collection requires a new destination')
        collect(root,opts.workspace.resolve(),opts.destination.resolve())


if __name__=='__main__': main()
