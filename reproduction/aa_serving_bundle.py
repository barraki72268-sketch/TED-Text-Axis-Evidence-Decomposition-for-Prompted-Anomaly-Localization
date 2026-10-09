"""Package a passing AA capture with exact source and weights for relocation."""
import json
from pathlib import Path
import shutil

from .captured_export import export_captured_run
from .checkpoint_download import digest_file
from .recipe_lookup import execution_recipe


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def build_aa_serving_bundle(root: Path, workspace: Path, destination: Path) -> dict:
    workspace, destination = workspace.resolve(), destination.absolute()
    if destination.exists():
        raise FileExistsError('Serving bundle destination must be new')
    plan = read(workspace / 'run.json')
    recipe = execution_recipe(root, plan['recipe'])
    if recipe['host'] != 'AA-CLIP':
        raise ValueError('This builder supports AA-CLIP only')
    # Recompute original metric, coverage, source, input and capture checks.
    manifest = export_captured_run(root, workspace, destination / 'export')
    inventory = []

    def copy(original, relative, expected):
        if digest_file(original) != expected:
            raise ValueError(f'Original serving input changed: {relative}')
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        if digest_file(target) != expected:
            raise ValueError(f'Serving copy differs: {relative}')
        inventory.append(dict(path=relative, sha256=expected, bytes=target.stat().st_size))

    for entry in manifest['prepared_source_files']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid prepared source path')
        copy(workspace / 'source' / relative, 'source/' + entry['path'], entry['sha256'])
    execution = read(workspace / 'execution.json')
    copy(workspace / 'run.json', 'run.json', execution['plan_sha256'])
    binding_id = recipe.get('dependency_recipe', recipe['id'])
    binding = next(b for b in read(root / 'host-checkpoints.json')['bindings'] if b['recipe'] == binding_id)
    summary = read(destination / 'export/summary.json')
    for entry in binding['checkpoint_assets']:
        name = entry['checkpoint_filename']
        if Path(name).name != name:
            raise ValueError('Invalid checkpoint filename')
        copy(Path(summary['ckpt_dir']) / name, 'checkpoints/' + name, entry['sha256'])
    backbones = read(root / 'backbones.json')
    selected = next(b for b in backbones['bindings'] if b['recipe'] == binding_id)
    backbone = next(b for b in backbones['artifacts'] if b['sha256'] == selected['sha256'])
    copy(workspace / 'clip-cache' / backbone['filename'], 'clip-cache/' + backbone['filename'], backbone['sha256'])
    bootstrap = next(b for b in backbones['artifacts'] if b['id'] == 'openai_vit_l14_336')
    relative = 'source/neurips2026/AA-CLIP/model/ViT-L-14-336px.pt'
    copy(workspace / relative, relative, bootstrap['sha256'])
    report = dict(schema_version=1, host='AA-CLIP', recipe=recipe['id'],
        export_sha256=digest_file(destination / 'export/manifest.json'),
        original_plan_sha256=execution['plan_sha256'], files=inventory,
        relocation='Exact source bytes; runtime checkpoint path binding only. No numerical arguments changed.',
        status='packaged_requires_relocated_inference_parity',
        limitations=['No dataset images included.', 'Original third-party licenses apply.',
                    'Per-image parity and container validation remain required.'])
    (destination / 'serving-bundle.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    return report
