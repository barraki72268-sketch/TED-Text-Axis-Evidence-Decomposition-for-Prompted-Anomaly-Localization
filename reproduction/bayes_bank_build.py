"""Build a new source-only BayesPFL bank inside a live GPU Slurm allocation.

The original collector is called directly. Historical bank getters and target
evaluation are never called. This command preserves the recipe's source budget.
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


def recorded_arguments(script: Path, argv: list[str]):
    """Execute only the original main's argument-parser prefix, not evaluation."""
    tree = ast.parse(script.read_text(encoding='utf-8'))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    prefix = []
    for node in main.body:
        prefix.append(node)
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == 'parse_args'):
            break
    else:
        raise ValueError('Original parser prefix not found')
    root = script.parents[2]
    namespace = {'argparse': argparse, 'Path': Path, 'ROOT': root,
                 'BAYES_ROOT': root / 'neurips2026/Bayes-PFL'}
    previous = sys.argv
    try:
        sys.argv = [str(script), *argv]
        exec(compile(ast.Module(body=prefix, type_ignores=[]), str(script), 'exec'), namespace)
    finally:
        sys.argv = previous
    return namespace['args']


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('destination', type=Path)
    opts = parser.parse_args()
    from .axis_run import require_live_gpu_allocation
    allocation = require_live_gpu_allocation()
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    root = Path(__file__).resolve().parent
    workspace = opts.workspace.resolve()
    plan, recipe = validate_prepared(root, workspace, require_slurm=True)
    if recipe['host'] != 'BayesPFL':
        raise ValueError('Only BayesPFL recipes are supported')
    args = recorded_arguments(Path(plan['evaluator']), plan['argv'])
    source = recipe['transfer'].split('2', 1)[0]
    if args.dataset != source:
        raise ValueError('Collector source dataset differs from recorded transfer')
    destination = opts.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    proof = {'recipe': plan['recipe'], 'status': 'running',
             'started': datetime.now(timezone.utc).isoformat(), 'device': 'cuda',
             'allocation': allocation, 'entrypoint_sha256': digest_file(Path(__file__)),
             'plan_sha256': digest_file(workspace / 'run.json'),
             'collection_function': 'collect_source_patch_banks',
             'historical_bank_loaded': False, 'target_evaluation_performed': False,
             'source_dataset': source, 'recorded_arguments': vars(args).copy(),
             'paper_metric_reproduction_verified': False}
    record = destination / 'construction.json'
    record.write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    try:
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Allocated GPU is unavailable')
        if not hasattr(np, 'trapezoid'):
            np.trapezoid = np.trapz
        script = workspace / 'source/neurips2026/scripts/collect_bayespfl_source_banks.py'
        sys.path[:] = isolated_paths(sys.path, script, plan['cwd'], root)
        for key, value in plan['environment'].items():
            if key != 'CUDA_VISIBLE_DEVICES':
                os.environ[key] = value
        spec = importlib.util.spec_from_file_location('ted_fresh_bayes_collector', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = recipe['transfer'].split('2', 1)[1]
        target_info = plan['datasets'][target]
        denied_roots = {Path(target_info['prepared_metadata']).parent.resolve(),
                        *(Path(v).resolve() for v in target_info['roots'].values())}
        source_roots = {Path(plan['datasets'][source]['prepared_metadata']).parent.resolve(),
                        *(Path(v).resolve() for v in plan['datasets'][source]['roots'].values())}
        if any(s == t or t in s.parents or s in t.parents for s in source_roots for t in denied_roots):
            raise ValueError('Cannot isolate overlapping source/target data roots')
        denied_banks = {Path(b['path']).resolve() for b in plan['bank_path_changes']}

        def source_only_audit(event, values):
            if event in {'socket.connect', 'socket.getaddrinfo'}:
                raise PermissionError('Network disabled during source-only collection')
            if event == 'open' and isinstance(values[0], (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(values[0])).resolve()
                if path in denied_banks or any(path == d or d in path.parents for d in denied_roots):
                    raise PermissionError('Historical bank and target inputs disabled during collection')

        sys.addaudithook(source_only_audit)
        # Exercise the actual hook before collection, rather than infer isolation
        # from the absence of a target-evaluation function call.
        for blocked in [Path(target_info['prepared_metadata']), *denied_banks]:
            try:
                with blocked.open('rb'):
                    pass
            except PermissionError:
                continue
            raise RuntimeError('Input-isolation self-check failed')
        proof.update(target_and_historical_bank_reads_denied=True,
                     denied_target_roots=sorted(map(str, denied_roots)),
                     denied_historical_bank_paths=sorted(map(str, denied_banks)))
        # The original collector builds models, sets the recorded seed, reads
        # source normal/defect images and mines new tensors without cache reads.
        fp, defect, stats = module.collect_source_patch_banks(args)
        payload = {'fp_banks': {int(k): v.detach().cpu() for k, v in fp.items()},
                   'def_banks': {int(k): v.detach().cpu() for k, v in defect.items()},
                   'bank_stats': stats,
                   'meta': {'host': 'Bayes-PFL', **{k: getattr(args, k) for k in (
                       'dataset', 'data_path', 'checkpoint_path', 'checkpoint_load_mode',
                       'features_list', 'hard_frac', 'max_good_per_class',
                       'max_defect_per_class', 'max_bank_per_layer', 'max_fp_per_image',
                       'max_defect_per_image', 'source_class', 'exclude_class')}}}
        for group in ('fp_banks', 'def_banks'):
            if not payload[group] or any(not torch.isfinite(v).all() for v in payload[group].values()):
                raise ValueError('Invalid source bank tensors')
        bank = destination / 'source-bank.pt'
        torch.save(payload, bank)
        proof.update(status='completed', collector_sha256=digest_file(script),
                     bank_sha256=digest_file(bank), bank_bytes=bank.stat().st_size,
                     bank_stats=stats,
                     tensor_shapes={g: {str(k): list(v.shape) for k, v in payload[g].items()}
                                    for g in ('fp_banks', 'def_banks')},
                     environment={'python': sys.version, 'torch': torch.__version__,
                                  'numpy': np.__version__})
    except Exception as error:
        proof.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        proof['finished'] = datetime.now(timezone.utc).isoformat()
        record.write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
