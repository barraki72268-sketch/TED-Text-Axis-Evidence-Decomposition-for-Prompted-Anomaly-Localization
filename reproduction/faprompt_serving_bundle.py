"""Package a matched FAPrompt execution without fitting or altering research source."""
import argparse
import json
from pathlib import Path
import shutil

from .captured_export import export_captured_run
from .checkpoint_download import digest_file
from .recipe_lookup import execution_recipe


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def build_faprompt_serving_bundle(root, workspace, destination):
    root, workspace, destination = Path(root).resolve(), Path(workspace).resolve(), Path(destination).absolute()
    if destination.exists():
        raise FileExistsError('Bundle destination must be new; preserve earlier attempts')
    recipe = execution_recipe(root, read(workspace / 'run.json')['recipe'])
    if recipe['host'] != 'FAPrompt':
        raise ValueError('Expected a FAPrompt recipe')
    manifest = export_captured_run(root, workspace, destination / 'export')
    summary = read(destination / 'export/summary.json')
    inventory = []

    def copy(original, relative, expected):
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            raise ValueError('Invalid bundle path')
        if digest_file(original) != expected:
            raise ValueError('Original serving input changed: ' + relative)
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        if digest_file(target) != expected:
            raise ValueError('Copied serving input changed: ' + relative)
        inventory.append(dict(path=relative, bytes=target.stat().st_size, sha256=expected))

    for entry in manifest['prepared_source_files']:
        copy(workspace / 'source' / entry['path'], 'source/' + entry['path'], entry['sha256'])
    execution = read(workspace / 'execution.json')
    copy(workspace / 'run.json', 'run.json', execution['plan_sha256'])
    binding = next(b for b in read(root / 'host-checkpoints.json')['bindings'] if b['recipe'] == recipe['id'])
    checkpoints = {b['sha256'] for b in binding['checkpoint_assets'] if b['role'] == 'checkpoint_path'}
    if len(checkpoints) != 1:
        raise ValueError('Ambiguous FAPrompt checkpoint')
    copy(Path(summary['checkpoint_path']), 'checkpoints/epoch_15.pth', checkpoints.pop())
    backbones = read(root / 'backbones.json')
    selected = next(b for b in backbones['bindings'] if b['recipe'] == recipe['id'])
    backbone = next(b for b in backbones['artifacts'] if b['sha256'] == selected['sha256'])
    copy(workspace / 'clip-cache' / backbone['filename'], 'clip-cache/' + backbone['filename'], backbone['sha256'])
    report = dict(schema_version=1, host='FAPrompt', recipe=recipe['id'],
        export_sha256=digest_file(destination / 'export/manifest.json'),
        original_plan_sha256=execution['plan_sha256'], files=inventory,
        status='packaged_requires_captured_state_engine_and_relocated_inference_parity',
        fitting_performed=False,
        limitations=['No dataset images included; original third-party licenses apply.',
                     'Exact prepared source and fitted state retained; no numerical arguments changed.',
                     'This is not the earlier engineering export that fits fresh source calibrators.',
                     'A captured-state inference engine, original-math image parity, anonymous acquisition and Docker checks are still required.'])
    (destination / 'serving-bundle.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    report = build_faprompt_serving_bundle(Path(__file__).resolve().parent, args.workspace, args.destination)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
