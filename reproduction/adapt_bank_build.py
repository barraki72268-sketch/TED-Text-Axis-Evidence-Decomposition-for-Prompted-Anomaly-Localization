"""Inspect or collect original AdaptCLIP VL/TL source banks.

GPU construction requires live Slurm membership. Inspection executes only the
original argument-parser prefix; it does not construct a model or a bank.
"""
import argparse
import ast
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

from .checkpoint_download import digest_file
from .import_paths import isolated_paths
from .run import validate_prepared
from .runtime import DATA_ROOTS, read_json


def parser_arguments(script, argv, torch_module):
    tree = ast.parse(script.read_bytes())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    prefix = []
    for node in main.body:
        prefix.append(node)
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'parse_args'):
            break
    else:
        raise ValueError('Original argument-parser boundary missing')
    source = script.parents[2]
    scope = dict(argparse=argparse, Path=Path, torch=torch_module, ROOT=source,
                 SCRIPT_ROOT=script.parent, ADAPT_ROOT=source/'neurips2026/AdaptCLIP')
    previous = sys.argv
    try:
        sys.argv = [str(script), *argv]
        exec(compile(ast.Module(body=prefix, type_ignores=[]), str(script), 'exec'), scope)
    finally:
        sys.argv = previous
    return scope['args']


def source_calls(evaluator, collector):
    et, ct = ast.parse(evaluator.read_bytes()), ast.parse(collector.read_bytes())
    main = next(n for n in et.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    fn = next(n for n in ct.body if isinstance(n, ast.FunctionDef) and n.name == 'collect_vl_source_banks')
    parameters = {p.arg for p in fn.args.args}
    getters = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == 'get_or_collect_vl_banks']
    builds = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id == 'build_model']
    if len(getters) != 2 or len(builds) != 1:
        raise ValueError('Archived dual-branch/model call structure differs')
    calls = {}
    for call in getters:
        fields = {k.arg: k.value for k in call.keywords}
        branch = ast.literal_eval(fields['bank_feature_source'])
        direct = {k: v for k,v in fields.items() if k not in {'checkpoint_path', 'bank_cache_path'}}
        if set(direct) != parameters or branch not in {'vl','tl'} or branch in calls:
            raise ValueError('Original collector signature differs')
        calls[branch] = direct
    if set(calls) != {'vl','tl'}:
        raise ValueError('Both original bank branches are required')
    return calls, builds[0]


def inspect_plan(root, workspace, torch_module):
    plan, recipe = validate_prepared(root, workspace, allow_bank_collection=True)
    if recipe['host'] != 'AdaptCLIP' or plan.get('preparation_scope') != 'source_bank_collection_only':
        raise ValueError('Use source-only AdaptCLIP preparation')
    evaluator = Path(plan['evaluator'])
    collector = workspace/'source/neurips2026/scripts/analyze_adaptclip_vl_score_forms.py'
    args = parser_arguments(evaluator, plan['argv'], torch_module)
    if args.ted_insertion != 'dual_branch' or args.source_dataset != recipe['transfer'].split('2',1)[0]:
        raise ValueError('Recorded source/branch settings differ')
    calls, model_call = source_calls(evaluator, collector)
    return plan, recipe, args, collector, calls, model_call


def evaluate_expression(node, scope):
    return eval(compile(ast.Expression(node), '<original-collection-expression>', 'eval'), scope)


def collect(root, workspace, destination):
    from .axis_run import require_live_gpu_allocation
    allocation = require_live_gpu_allocation()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated GPU is unavailable')
    plan, recipe, args, script, calls, model_call = inspect_plan(root, workspace, torch)
    if not args.device.startswith('cuda'):
        raise ValueError('Preserve the recorded GPU device')
    destination.mkdir(parents=True, exist_ok=False)
    proof = dict(recipe=plan['recipe'], status='running', allocation=allocation,
                 started=datetime.now(timezone.utc).isoformat(), recorded_arguments=vars(args),
                 plan_sha256=digest_file(workspace/'run.json'), entrypoint_sha256=digest_file(Path(__file__)),
                 collector_sha256=digest_file(script), historical_bank_loaded=False,
                 target_evaluation_performed=False, bank_branches={})
    record = destination/'construction.json'
    record.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    try:
        target = recipe['transfer'].split('2',1)[1]
        targets = {(workspace/'unavailable-target').resolve(), *(Path(p).resolve() for p in DATA_ROOTS[target])}
        source = recipe['transfer'].split('2',1)[0]
        sources = {Path(plan['datasets'][source]['prepared_metadata']).parent.resolve(),
                   *(Path(p).resolve() for p in plan['datasets'][source]['roots'].values() if p)}
        if any(s == t or s in t.parents or t in s.parents for s in sources for t in targets):
            raise ValueError('Source and target roots overlap')
        banks = {a['sha256'] for a in read_json(root/'host-source-banks.json')['artifacts']}
        sentinels = {(workspace/n).resolve() for n in ['unavailable-historical-bank.pt','unavailable-historical-tl-bank.pt']}
        def audit(event, values):
            if event in {'socket.connect','socket.getaddrinfo'}:
                raise PermissionError('Network disabled during source collection')
            if event == 'open' and isinstance(values[0], (str,bytes,os.PathLike)):
                p = Path(os.fsdecode(values[0])).resolve()
                if p.name in banks or p in sentinels or any(p == t or t in p.parents for t in targets):
                    raise PermissionError('Historical banks and target data disabled')
        sys.addaudithook(audit)
        for p in [*sentinels, workspace/'unavailable-target/metadata.json']:
            try:
                p.open('rb')
            except PermissionError:
                continue
            raise RuntimeError('Source isolation selfcheck failed')
        proof['target_and_historical_bank_reads_denied'] = True
        for key,value in plan['environment'].items():
            if key != 'CUDA_VISIBLE_DEVICES': os.environ[key] = value
        sys.path[:] = isolated_paths(sys.path, Path(plan['evaluator']), plan['cwd'], root)
        spec = importlib.util.spec_from_file_location('ted_original_adapt_branch_collector', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import random
        import numpy as np
        random.seed(int(args.seed)); np.random.seed(int(args.seed)); torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))
        scope = dict(module.__dict__, args=args)
        values = evaluate_expression(model_call, scope)
        scope.update(zip(['model','textual_learner','visual_learner','learned_text_features','dpam_layer'], values))
        scope['vl_bank_mining_source'] = 'vl'  # Original dual_branch conditional.
        for branch in ['vl','tl']:
            kwargs = {k:evaluate_expression(v, scope) for k,v in calls[branch].items()}
            if kwargs['bank_mining_source'] != branch or kwargs['bank_feature_source'] != branch:
                raise ValueError('Recorded dual-branch mining/features differ')
            fp, defect, stats = module.collect_vl_source_banks(**kwargs)
            if any(not torch.isfinite(t).all() for t in [fp,defect]):
                raise ValueError('Nonfinite source-bank tensors')
            # Execute only the archived getter's metadata assignment, never its cache path.
            tree = ast.parse(script.read_bytes())
            getter = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'get_or_collect_vl_banks')
            meta_node = next(n for n in getter.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='requested_meta' for t in n.targets))
            meta_scope = dict(kwargs, checkpoint_path=args.checkpoint_path)
            exec(compile(ast.Module(body=[meta_node],type_ignores=[]),str(script),'exec'),meta_scope)
            bank = destination/(branch+'-source-bank.pt')
            module.save_bank_cache(bank,{0:fp},{0:defect},{0:stats},meta_scope['requested_meta'])
            proof['bank_branches'][branch] = dict(path=str(bank),sha256=digest_file(bank),bytes=bank.stat().st_size,
                                                  fp_shape=list(fp.shape),def_shape=list(defect.shape),stats=stats)
        proof['status'] = 'completed'
    except Exception as error:
        proof.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        proof['finished'] = datetime.now(timezone.utc).isoformat()
        record.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('workspace',type=Path)
    p.add_argument('destination',type=Path,nargs='?')
    p.add_argument('--inspect',action='store_true')
    opts = p.parse_args(); root = Path(__file__).resolve().parent
    if opts.inspect:
        import torch
        plan,recipe,args,script,calls,model = inspect_plan(root,opts.workspace.resolve(),torch)
        print(json.dumps(dict(recipe=plan['recipe'],recorded_arguments=vars(args),
                              collector_sha256=digest_file(script),original_model_call=ast.unparse(model),
                              direct_branch_calls={b:{k:ast.unparse(v) for k,v in kw.items()} for b,kw in calls.items()},
                              bank_collection_performed=False,GPU_execution_performed=False)))
    else:
        if opts.destination is None: p.error('Collection requires a new destination')
        collect(root,opts.workspace.resolve(),opts.destination.resolve())


if __name__ == '__main__':
    main()
