"""Compare captured AdaptCLIP image inference to the original evaluator on CPU."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .checkpoint_download import digest_file
from .datasets import load_protocol


def original_output_block(script: Path):
    tree = ast.parse(script.read_text(encoding='utf-8'))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    blocks = [node for node in ast.walk(main) if isinstance(node, ast.If)
              and ast.unparse(node.test) == 'source_vl_calibrator is not None'
              and isinstance(node.body[0], ast.Assign)
              and isinstance(node.body[0].targets[0], ast.Name)
              and node.body[0].targets[0].id == 'calibrator_token']
    if len(blocks) != 1:
        raise ValueError('Original AdaptCLIP calibrated output block is ambiguous')
    return compile(ast.Module(body=blocks, type_ignores=[]), str(script), 'exec')


def run(exported: Path, workspace: Path, fixture_workspace: Path | None = None) -> dict:
    import numpy as np
    from PIL import Image
    import torch
    from ted.inference.adaptclip_engine import CapturedAdaptCLIPEngine

    engine = CapturedAdaptCLIPEngine(export_directory=exported, workspace=workspace, device='cpu')
    plan = json.loads(((fixture_workspace or workspace) / 'run.json').read_text())
    if digest_file((fixture_workspace or workspace) / 'run.json') != digest_file(workspace / 'run.json'):
        raise ValueError('Fixture workspace must share the exact original plan')
    dataset = engine.summary['target_dataset']
    prepared = plan['datasets'][dataset]
    metadata_path = Path(prepared['prepared_metadata'])
    if digest_file(metadata_path) != prepared['prepared_metadata_sha256']:
        raise ValueError('Prepared target metadata changed')
    root = Path(__file__).resolve().parent
    protocol = root / 'datasets' / dataset
    manifest, _ = load_protocol(protocol)
    if digest_file(protocol / 'manifest.json') != prepared['manifest_sha256']:
        raise ValueError('Target input manifest changed')
    inputs = {(row['role'], row['path']): row for row in manifest['files']}
    classes = json.loads(metadata_path.read_text())['test']
    capture_manifest = json.loads((exported / 'manifest.json').read_text())
    calibrators = {}
    for entry in capture_manifest['captured_state']:
        value = torch.load(exported / entry['object_path'], map_location='cpu', weights_only=True)
        observed = {key: v for key, v in value.items() if key not in {'basis', 'fp_coords', 'def_coords'}}
        roles = [role for role in ['vl', 'tl'] if observed == engine.summary['source_' + role + '_calibrator']]
        if len(roles) != 1 or roles[0] in calibrators:
            raise ValueError('Reference capture branch differs')
        calibrators[roles[0]] = value
    code = original_output_block(workspace / 'source/neurips2026/scripts/official_parallel_test_adaptclip_vlrefine.py')
    rows = []
    # First recorded test image of every class, fixed before observing outputs.
    for category, items in classes.items():
        path = Path(items[0]['img_path'])
        relative = path.relative_to(Path(prepared['roots']['images'])).as_posix()
        binding = inputs[('images', relative)]
        if digest_file(path) != binding['sha256'] or path.stat().st_size != binding['bytes']:
            raise ValueError('Target fixture bytes changed')
        with Image.open(path) as opened:
            image = opened.convert('RGB')
        actual = engine.predict(image)
        with torch.inference_mode():
            tensor = engine.transform(image).unsqueeze(0)
            cfg, host = engine.summary, engine.host
            baseline = host.compute_baseline_outputs(engine.model, engine.textual, engine.visual, engine.text,
                tensor, cfg['features_list'], engine.dpam_layer, cfg['image_size'], cfg['sigma'], cfg['fusion_type'])
            vl = host.vl_patch_bank_features(engine.visual, baseline['query_patch_feats'])
            tl = host.branch_patch_bank_features(engine.visual, baseline['query_patch_feats'], 'tl')
            vl_map = host.resize_map(host.smooth_map(baseline['local_vl_map'], cfg['sigma']), cfg['image_size'])
            tl_map = host.resize_map(host.smooth_map(baseline['local_tl_map'], cfg['sigma']), cfg['image_size'])
            namespace = dict(vars(host), args=SimpleNamespace(**cfg, device='cpu'), image=tensor,
                baseline=baseline, vl_patch=vl, tl_patch=tl, h=math.isqrt(vl.shape[0]),
                vl_axis=None, tl_axis=None, fp_coeff=None, def_coeff=None, tl_fp_coeff=None, tl_def_coeff=None,
                source_vl_calibrator=calibrators['vl'], source_tl_calibrator=calibrators['tl'],
                vl_map=vl_map, tl_map=tl_map, raw_tl_map=baseline['local_tl_map'], outputs_to_append=[])
            exec(code, namespace)
            _, expected_map, expected_score = namespace['outputs_to_append'][0]
            rows.append({'category': category, 'image': relative, 'image_sha256': binding['sha256'],
                'decoded_rgb_sha256': __import__('hashlib').sha256(image.tobytes()).hexdigest(),
                'map_shape': list(actual['host_map'].shape),
                'host_max_absolute_error': float(np.max(np.abs(actual['host_map'] - baseline['baseline_map'].numpy()))),
                'cted_max_absolute_error': float(np.max(np.abs(actual['cted_map'] - expected_map.numpy()))),
                'host_image_score_error': abs(actual['image_score'] - float(baseline['baseline_img_score'][0])),
                'cted_image_score_error': abs(actual['cted_image_score'] - float(expected_score[0]))})
    passed = all(row[key] == 0 for row in rows for key in
                 ['host_max_absolute_error', 'cted_max_absolute_error', 'host_image_score_error', 'cted_image_score_error'])
    return {'status': 'matched' if passed else 'mismatch', 'device': 'cpu', 'engine': engine.info(),
            'scope': 'First recorded test image of each class, same CPU model versus original baseline functions and original calibrated-output AST block. Not a full-dataset metric replay or independent GPU image parity.',
            'model_fitting': False, 'rows': rows, 'portable_bundle_verified': False, 'docker_http_verified': False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export_directory', type=Path)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture-workspace', type=Path, help='Original dataset metadata for a portable bundle test; never used by the engine')
    args = parser.parse_args()
    report = {'status': 'running', 'started': datetime.now(timezone.utc).isoformat()}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    try:
        report.update(run(args.export_directory.resolve(), args.workspace.resolve(),
                          args.fixture_workspace.resolve() if args.fixture_workspace else None))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    report['finished'] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'matched' else 1


if __name__ == '__main__':
    raise SystemExit(main())
