"""Prepare a Figure 3 source-bank rebuild without running a model or using a GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .axis_figure import read_inputs
from .checkpoint_download import digest_file
from .datasets import prepare_dataset
from .source import unpack_source


def verify_asset(path: Path, specification: dict) -> dict:
    path = path.resolve()
    if path.stat().st_size != specification['bytes'] or digest_file(path) != specification['sha256']:
        raise ValueError('Figure 3 asset hash/size mismatch')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': specification['sha256']}


def prepare(root: Path, destination: Path, checkpoint: Path, backbone: Path, datasets: Path) -> dict:
    if sys.platform != 'linux':
        raise RuntimeError('Figure 3 collection preparation requires Linux')
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError('Figure 3 workspace must be new')
    manifest, data, _ = read_inputs(root)
    specification = manifest['fresh_collection']
    assets = {'checkpoint': verify_asset(checkpoint, specification['checkpoint']),
              'backbone': verify_asset(backbone, specification['backbone'])}
    roots = json.loads(datasets.read_text())
    summary = specification['observed_summary']
    if summary != data['summary']:
        raise ValueError('Figure 3 reconstructed summary differs from pinned arrays')
    destination.mkdir()
    source = destination / 'source'
    source_report = unpack_source(root / 'source.zip', root / 'source-manifest.json', source)
    prepared = {}
    for name in ['mvtec', 'visa']:
        inputs = roots[name]
        folder = destination / 'data' / name
        prepared[name] = prepare_dataset(root / 'datasets' / name, Path(inputs['images']),
                                         Path(inputs['masks']) if inputs.get('masks') else None, folder)
        if prepared[name]['file_errors']:
            raise ValueError('Figure 3 dataset verification failed: ' + name)
    cache = destination / 'clip-cache'
    cache.mkdir()
    (cache / specification['backbone']['filename']).symlink_to(Path(assets['backbone']['path']))
    checkpoint_link = source / summary['checkpoint_path']
    checkpoint_link.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_link.symlink_to(Path(assets['checkpoint']['path']))
    mapping = {
        '/mnt/data/pilab-kingjinyoung/ted-reproduce-20261008/snapshot': str(source),
        '/mnt/data/pilab-kingjinyoung/ADPretrain-CLIP': str(source),
        '/mnt/data/hf-cache/anomalyclip': str(cache),
    }
    changes = []
    for entry in json.loads((root / 'source-manifest.json').read_text())['files']:
        if not entry['path'].endswith('.py'):
            continue
        path = source / entry['path']
        before = path.read_text(encoding='utf-8')
        after = before
        for original, relocated in mapping.items():
            after = after.replace(original, relocated)
        if before != after:
            path.write_text(after, encoding='utf-8')
            changes.append({'path': entry['path'], 'original_sha256': entry['sha256'],
                            'relocated_sha256': digest_file(path)})
    bank = destination / 'bank-cache'
    bank.mkdir()
    output = destination / 'results'
    output.mkdir()
    helper = specification['model_helper']
    meta = summary['bank_cache_meta']
    values = {
        '--source_root': destination / 'data/mvtec', '--source_dataset': 'mvtec',
        '--target_root': destination / 'data/visa', '--target_dataset': 'visa',
        '--target_mode': summary['target_mode'], '--target_class': summary['target_class'],
        '--checkpoint_path': checkpoint_link, '--save_dir': output, '--device': 'cuda:0',
        '--image_size': meta['image_size'], '--depth': helper['depth'],
        '--n_ctx': helper['n_ctx'], '--t_n_ctx': helper['t_n_ctx'],
        '--hard_frac': summary['hard_frac'], '--tau': summary['tau'],
        '--max_good_per_class': meta['max_good_per_class'],
        '--max_defect_per_class': meta['max_defect_per_class'],
        '--max_bank_per_layer': meta['max_bank_per_layer'],
        '--max_fp_per_image': meta['max_fp_per_image'],
        '--max_defect_per_image': meta['max_defect_per_image'],
        '--max_normal_eval_images': summary['max_normal_eval_images'],
        '--max_anomaly_eval_images': summary['max_anomaly_eval_images'],
        '--bank_cache_path': bank / 'fresh-source-bank.pt',
    }
    command = ['{python}', '-u', str(source / manifest['collector']['path'])]
    for flag, value in values.items():
        command.extend([flag, str(value)])
    command.extend(['--features_list', *map(str, meta['features_list']), '--refresh_bank_cache'])
    report = {'status': 'prepared', 'scope': specification['scope'], 'gpu_execution': False,
              'source': source_report, 'assets': assets, 'datasets': prepared,
              'source_path_changes': changes, 'path_mapping': mapping, 'command': command,
              'cwd': str(source), 'bank_directory': str(bank), 'pre_run_bank_entries': 0,
              'source_bank_policy': 'fresh_source_only', 'historical_argv': 'not recovered',
              'historical_seed': specification['historical_seed'],
              'argument_provenance': 'Pinned diagnostic summary plus original parser/model defaults; not observed historical argv',
              'unresolved': specification['unresolved'],
              'execution_requirements': 'A verified valid Slurm allocation and a recorded guarded runner; no GPU execution by prepare'}
    with (destination / 'figure3-plan.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--datasets', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(Path(__file__).resolve().parent, args.destination, args.checkpoint,
                     args.backbone, args.datasets)
    print(json.dumps({'status': report['status'], 'gpu_execution': False,
                      'plan': str(args.destination / 'figure3-plan.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
