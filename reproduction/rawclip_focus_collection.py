"""Prepare Figure 4 inference with the recovered recorded-path bank on Linux."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from .axis_collection import verify_asset
from .checkpoint_download import digest_file
from .datasets import prepare_dataset
from .rawclip_focus import read_inputs
from .source import unpack_source


def prepare(root: Path, destination: Path, bank: Path, backbone: Path, datasets: Path) -> dict:
    if sys.platform != 'linux':
        raise RuntimeError('Figure 4 collection preparation requires Linux')
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError('Figure 4 workspace must be new; preserve earlier attempts')
    manifest, payload = read_inputs(root)
    binding = manifest['current_original_bank_trace']
    trace_path = root / binding['path']
    if trace_path.stat().st_size != binding['bytes'] or digest_file(trace_path) != binding['sha256']:
        raise ValueError('Figure 4 bank trace hash/size mismatch')
    trace = json.loads(trace_path.read_text())
    meta = trace['cache_meta']
    registry = json.loads((root / 'backbones.json').read_text())
    # This existing public backbone catalog binds the official OpenAI download.
    asset_rows = registry['artifacts']
    asset = next(row for row in asset_rows if row['sha256'] == '3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02')
    assets = {'bank': verify_asset(bank, trace), 'backbone': verify_asset(backbone, asset)}
    summary = json.loads(payload['rawclip_aggregated_distribution.json'])
    if (meta['backbone_name'] != summary['backbone'] or meta['pretrained_dataset'] != summary['pretrained_dataset']
            or meta['source_dataset'] != 'visa' or meta['image_size'] != 336):
        raise ValueError('Figure 4 bank/configuration binding differs')
    destination.mkdir()
    source = destination / 'source'
    source_report = unpack_source(root / 'source.zip', root / 'source-manifest.json', source)
    inputs = json.loads(datasets.read_text())['mvtec']
    prepared = prepare_dataset(root / 'datasets/mvtec', Path(inputs['images']),
                               Path(inputs['masks']) if inputs.get('masks') else None, destination / 'data/mvtec')
    if prepared['file_errors']:
        raise ValueError('Figure 4 target dataset verification failed')
    cache = destination / 'clip-cache'
    cache.mkdir()
    (cache / asset['filename']).symlink_to(Path(assets['backbone']['path']))
    bank_link = destination / 'bank-cache/recorded-source-bank.pt'
    bank_link.parent.mkdir()
    bank_link.symlink_to(Path(assets['bank']['path']))
    mapping = {
        '/mnt/data/pilab-kingjinyoung/ted-reproduce-20261008/snapshot': str(source),
        '/mnt/data/pilab-kingjinyoung/ADPretrain-CLIP': str(source),
        '/mnt/data/pilab-kingjinyoung/.cache/clip': str(cache),
        '/home/jinyoung/.cache/clip': str(cache), '~/.cache/clip': str(cache),
    }
    pattern = re.compile('|'.join(re.escape(key) for key in sorted(mapping, key=len, reverse=True)))
    changes = []
    for entry in json.loads((root / 'source-manifest.json').read_text())['files']:
        if not entry['path'].endswith('.py'):
            continue
        path = source / entry['path']
        before = path.read_text(encoding='utf-8')
        after = pattern.sub(lambda match: mapping[match.group()], before)
        if after != before:
            path.write_text(after, encoding='utf-8')
            changes.append({'path': entry['path'], 'original_sha256': entry['sha256'],
                            'relocated_sha256': digest_file(path)})
    output = destination / 'results'
    output.mkdir()
    collector = next(row for row in manifest['source_scripts'] if row['path'].endswith('generate_rawclip_aggregated_distribution.py'))
    command = ['{python}', '-u', str(source / collector['path'])]
    settings = {'--target_root': destination / 'data/mvtec', '--target_split': summary['target_split'],
                '--backbone': summary['backbone'], '--pretrained_dataset': summary['pretrained_dataset'],
                '--image_size': meta['image_size'], '--prompt_mode': meta['prompt_mode'],
                '--bank_cache_path': bank_link, '--sigma': 4.0, '--tau': 20.0,
                '--image_score_topk': 0.01, '--hard_fp_frac': summary['hard_fp_frac'],
                '--max_points_per_group': 200000, '--per_image_cap_abnormal': 4096,
                '--per_image_cap_hard_fp': 4096, '--per_image_cap_other': 4096,
                '--bins': 220, '--smooth_sigma': 2.0, '--dist_fig_width': 13.5, '--dist_fig_height': 4.6,
                '--focus_fig_width': 12.8, '--focus_fig_height': 7.2,
                '--combo_fig_width': 20.0, '--combo_fig_height': 4.8,
                '--device': 'cuda:0', '--seed': 0, '--output_dir': output}
    for flag, value in settings.items():
        command.extend([flag, str(value)])
    command.extend(['--features_list', *map(str, meta['features_list'])])
    report = {'status': 'prepared_requires_guarded_slurm_execution', 'gpu_execution': False,
              'scope': 'Reconstructed Figure4 full-target inference with recovered current recorded-path bank; not a fresh bank rebuild',
              'source': source_report, 'assets': assets, 'datasets': {'mvtec': prepared},
              'source_path_changes': changes, 'path_mapping': mapping, 'command': command, 'cwd': str(source),
              'source_bank_policy': 'current_recorded_path_bank_preserved_bytes', 'historical_argv': 'not recovered',
              'argument_provenance': 'Pinned diagnostic summary, recovered bank metadata and original parser defaults; not observed historical argv',
              'seed_provenance': 'Original parser default0; historical seed not recorded in summary',
              'pre_run_result_entries': 0, 'expected_collector_pdf_files': 4,
              'next_gate': 'Verified Slurm/PID guard and output/RNG recording; separate fresh-source bank rebuild still required'}
    with (destination / 'figure4-plan.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--datasets', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(Path(__file__).resolve().parent, args.destination, args.bank, args.backbone, args.datasets)
    print(json.dumps({'status': report['status'], 'gpu_execution': False,
                      'plan': str(args.destination / 'figure4-plan.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
