"""Inspect or collect original FAPrompt source banks without a cache getter.

Actual collection requires live Slurm GPU membership. CPU inspection does not
initialize a model, mine source patches, calibrate, or evaluate target images.
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


def original_calls(script):
    tree = ast.parse(script.read_bytes())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect_source_banks')
    def calls(name):
        return [n for n in ast.walk(main) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == name]
    getters, builds = calls('get_or_collect_source_banks'), calls('build_model')
    if len(getters) != 1 or len(builds) != 1:
        raise ValueError('Original FAPrompt model/collector call structure differs')
    direct = {k.arg:k.value for k in getters[0].keywords
              if k.arg not in {'preset', 'checkpoint_path', 'cache_dir', 'force_rebuild_bank'}}
    if set(direct) != {p.arg for p in fn.args.args}:
        raise ValueError('Original direct collector signature differs')
    return tree, main, direct, builds[0]


def parser_arguments(script, argv, torch_module, dataset_roots=None):
    tree, main, _, _ = original_calls(script)
    source = script.parents[2]
    roots = dataset_roots or {}
    scope = dict(argparse=argparse, Path=Path, torch=torch_module, ROOT=source,
                 SCRIPT_ROOT=script.parent, FAP_ROOT=source/'neurips2026/FAPrompt')
    for key, name in [('MVTEC_ROOT','mvtec'), ('OFFICIAL_VISA_ROOT','visa'),
                      ('BTAD_ROOT','btad'), ('MPDD_ROOT','mpdd')]:
        scope[key] = roots.get(name, source/'unavailable-dataset'/name)
    presets = next(n for n in tree.body if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='PRESET_CONFIGS' for t in n.targets))
    exec(compile(ast.Module(body=[presets],type_ignores=[]),str(script),'exec'),scope)
    prefix = []
    for node in main.body:
        prefix.append(node)
        if (isinstance(node,ast.Assign) and isinstance(node.value,ast.Call)
                and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='parse_args'):
            break
    else:
        raise ValueError('Original parser boundary missing')
    previous = sys.argv
    try:
        sys.argv = [str(script), *argv]
        exec(compile(ast.Module(body=prefix,type_ignores=[]),str(script),'exec'),scope)
    finally:
        sys.argv = previous
    args = scope['args']
    cfg = dict(scope['PRESET_CONFIGS'][args.preset])
    if args.checkpoint_path:
        cfg['checkpoint_path'] = args.checkpoint_path
    return args, cfg


def inspect_plan(root, workspace, torch_module):
    plan, recipe = validate_prepared(root,workspace,allow_bank_collection=True)
    if recipe['host'] != 'FAPrompt' or plan.get('preparation_scope') != 'source_bank_collection_only':
        raise ValueError('Use source-only FAPrompt preparation')
    script = Path(plan['evaluator'])
    roots = {n:Path(d['prepared_metadata']).parent for n,d in plan['datasets'].items()}
    args,cfg = parser_arguments(script,plan['argv'],torch_module,roots)
    source = recipe['transfer'].split('2',1)[0]
    if args.preset != recipe['transfer'] or cfg['source_dataset'] != {'mvtec':'mvtecad'}.get(source,source):
        raise ValueError('Recorded source/preset differs')
    if Path(cfg['source_root']).resolve() != roots[source].resolve():
        raise ValueError('Source-root binding differs')
    checkpoint = Path(cfg['checkpoint_path'])
    if not any(checkpoint.resolve() == Path(x['path']).resolve() for x in plan['verified_objects']):
        raise ValueError('Preset checkpoint is not a verified model object')
    _,_,direct,model = original_calls(script)
    return plan,recipe,args,cfg,direct,model


def collect(root,workspace,destination):
    from .axis_run import require_live_gpu_allocation
    allocation = require_live_gpu_allocation()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated GPU unavailable')
    plan,recipe,args,cfg,direct,model_call = inspect_plan(root,workspace,torch)
    if not args.device.startswith('cuda'):
        raise ValueError('Preserve the recorded GPU device')
    destination.mkdir(parents=True,exist_ok=False)
    script = Path(plan['evaluator'])
    proof = dict(recipe=plan['recipe'],status='running',allocation=allocation,
                 started=datetime.now(timezone.utc).isoformat(),recorded_arguments=vars(args),
                 plan_sha256=digest_file(workspace/'run.json'),entrypoint_sha256=digest_file(Path(__file__)),
                 collector_sha256=digest_file(script),historical_bank_loaded=False,
                 target_evaluation_performed=False)
    record = destination/'construction.json'
    record.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    try:
        source,target = recipe['transfer'].split('2',1)
        targets = {(workspace/'unavailable-target').resolve(),
                   (workspace/'source/neurips2026/results/bank_cache').resolve(),
                   *(Path(p).resolve() for p in DATA_ROOTS[target])}
        sources = {Path(cfg['source_root']).resolve(),
                   *(Path(p).resolve() for p in plan['datasets'][source]['roots'].values() if p)}
        if any(s==t or s in t.parents or t in s.parents for s in sources for t in targets):
            raise ValueError('Source and denied roots overlap')
        bank_hashes = {a['sha256'] for a in read_json(root/'host-source-banks.json')['artifacts']}
        sentinels = {(workspace/'unavailable-historical-bank.pt').resolve()}
        def audit(event,values):
            if event in {'socket.connect','socket.getaddrinfo'}:
                raise PermissionError('Network disabled during source collection')
            if event=='open' and isinstance(values[0],(str,bytes,os.PathLike)):
                p = Path(os.fsdecode(values[0])).resolve()
                if p.name in bank_hashes or p in sentinels or any(p==t or t in p.parents for t in targets):
                    raise PermissionError('Target data and cached banks disabled')
        sys.addaudithook(audit)
        for p in [*sentinels,workspace/'unavailable-target/metadata.json',
                  workspace/'source/neurips2026/results/bank_cache/denial-selfcheck.pt']:
            try:
                p.open('rb')
            except PermissionError:
                continue
            raise RuntimeError('Source isolation selfcheck failed')
        proof['target_and_cached_bank_reads_denied'] = True
        for key,value in plan['environment'].items():
            if key!='CUDA_VISIBLE_DEVICES': os.environ[key]=value
        sys.path[:] = isolated_paths(sys.path,script,plan['cwd'],root)
        spec = importlib.util.spec_from_file_location('ted_original_fap_collector',script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        actual_cfg = dict(module.PRESET_CONFIGS[args.preset])
        if args.checkpoint_path: actual_cfg['checkpoint_path']=args.checkpoint_path
        for name in ['source_root','checkpoint_path']:
            if Path(actual_cfg[name]).resolve()!=Path(cfg[name]).resolve():
                raise ValueError('Imported original preset binding differs')
        module.setup_seed(args.seed)
        module.validate_visa_root(cfg['source_root'],cfg['source_dataset'])
        scope = dict(module.__dict__,args=args,cfg=actual_cfg,device=torch.device(args.device))
        model,prompt_learner,prompt_load_info = evaluate_expression(model_call,scope)
        scope.update(model=model,prompt_learner=prompt_learner,prompt_load_info=prompt_load_info)
        kwargs = {k:evaluate_expression(v,scope) for k,v in direct.items()}
        banks = module.collect_source_banks(**kwargs)
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
        bank = destination/'source-bank.pt'
        torch.save(banks,bank)  # Preserve the complete original returned dictionary.
        proof.update(status='completed',bank=dict(path=str(bank),sha256=digest_file(bank),bytes=bank.stat().st_size,
                     fp_shape=list(banks['fp'].shape),defect_shape=list(banks['defect'].shape),
                     protocol_version=banks['protocol_version']),effective_dpam_layer=prompt_load_info['effective_dpam_layer'])
    except Exception as error:
        proof.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        proof['finished']=datetime.now(timezone.utc).isoformat()
        record.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('workspace',type=Path)
    p.add_argument('destination',type=Path,nargs='?')
    p.add_argument('--inspect',action='store_true')
    opts = p.parse_args(); root = Path(__file__).resolve().parent
    if opts.inspect:
        import torch
        plan,recipe,args,cfg,direct,model = inspect_plan(root,opts.workspace.resolve(),torch)
        print(json.dumps(dict(recipe=plan['recipe'],recorded_arguments=vars(args),source_root=cfg['source_root'],
                              checkpoint_path=cfg['checkpoint_path'],original_model_call=ast.unparse(model),
                              direct_collection_call={k:ast.unparse(v) for k,v in direct.items()},
                              bank_collection_performed=False,GPU_execution_performed=False)))
    else:
        if opts.destination is None: p.error('Collection requires a new destination')
        collect(root,opts.workspace.resolve(),opts.destination.resolve())


if __name__=='__main__': main()
