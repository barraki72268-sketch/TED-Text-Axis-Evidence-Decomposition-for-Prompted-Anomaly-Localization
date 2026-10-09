"""Export fitted evaluator state without importing a model or fitting anything."""
import json
from pathlib import Path
import re
import shutil

from .checkpoint_download import digest_file
from .coverage import validate_coverage
from .metrics import compare, extract
from .recipe_lookup import execution_recipe, reference_summary


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def capture_inventory(directory: Path) -> tuple[str, list[dict]]:
    index = directory / 'index.json'
    entries = read(index)
    if not isinstance(entries, list) or not entries:
        raise ValueError('No fitted-state capture inventory')
    names = set()
    for entry in entries:
        name = entry['file']
        if not re.fullmatch(r'[A-Za-z0-9_-]+\.pt', name) or name in names:
            raise ValueError('Invalid or duplicate captured filename')
        names.add(name)
        path = directory / name
        if path.is_symlink() or not path.is_file() or digest_file(path) != entry['sha256']:
            raise ValueError('Captured state bytes differ from recorded inventory')
    return digest_file(index), entries


def export_captured_run(root: Path, workspace: Path, destination: Path) -> dict:
    """Create a state/evidence bundle, not an inference-parity certificate."""
    workspace, destination = workspace.resolve(), destination.absolute()
    if destination.exists():
        raise FileExistsError('Export destination must be new; preserve earlier exports')
    execution = read(workspace / 'execution.json')
    if execution.get('status') != 'matched' or execution.get('returncode') != 0 or not execution.get('finished'):
        raise ValueError('Only terminal, numerically matching full-recipe executions may be exported')
    plan = read(workspace / 'run.json')
    if digest_file(workspace / 'run.json') != execution['plan_sha256'] or plan['recipe'] != execution['recipe']:
        raise ValueError('Execution plan changed after evaluation')
    recipe = execution_recipe(root, plan['recipe'])
    reference, expected = reference_summary(root, plan['recipe'])
    summary_path = workspace / 'results/summary.json'
    summary = read(summary_path)
    cells = compare(extract(summary, recipe['host']), extract(expected, recipe['host']))
    comparison = read(workspace / 'comparison.json')
    if (comparison != execution['comparison'] or comparison['cells'] != cells or
            comparison['reference_sha256'] != reference['reference_sha256'] or
            comparison['actual_sha256'] != digest_file(summary_path) or
            not cells or not all(c['matches_printed_precision'] for c in cells)):
        raise ValueError('Recomputed metrics or summary bytes differ from recorded passing execution')
    coverage = validate_coverage(root, reference, summary)
    changes = {r['path']: r['after_sha256'] for r in plan['source_path_changes']}
    prepared_source = []
    for entry in read(root / 'source-manifest.json')['files']:
        expected_sha = changes.get(entry['path'], entry['sha256'])
        if digest_file(workspace / 'source' / entry['path']) != expected_sha:
            raise ValueError('Evaluator source changed after evaluation')
        prepared_source.append({'path': entry['path'], 'sha256': expected_sha})
    dependencies = []
    for item in plan['verified_objects']:
        if digest_file(Path(item['path'])) != item['sha256']:
            raise ValueError('Model input object changed after evaluation')
        dependencies.append({'sha256': item['sha256'], 'bytes': item['bytes']})
    captures = workspace / 'results/artifacts'
    index_sha, entries = capture_inventory(captures)
    if not any('calibrator' in e['function'] for e in entries):
        raise ValueError('Execution has no captured fitted calibrator')
    bound = execution.get('capture_index_sha256')
    if bound is not None and bound != index_sha:
        raise ValueError('Capture index differs from terminal execution record')
    if execution.get('captured_files') is not None and execution['captured_files'] != entries:
        raise ValueError('Captured file inventory differs from terminal execution record')
    destination.mkdir(parents=True, exist_ok=False)
    objects = destination / 'objects'
    objects.mkdir()
    exported = []
    for entry in entries:
        target = objects / entry['sha256']
        if not target.exists():
            shutil.copyfile(captures / entry['file'], target)
        if digest_file(target) != entry['sha256']:
            raise ValueError('Exported capture bytes do not match the original capture')
        exported.append(dict(entry, object_path='objects/' + entry['sha256'], bytes=target.stat().st_size))
    evidence = []
    for name, original in [('execution.json', workspace / 'execution.json'),
                           ('comparison.json', workspace / 'comparison.json'),
                           ('summary.json', summary_path), ('capture-index.json', captures / 'index.json')]:
        shutil.copyfile(original, destination / name)
        evidence.append({'file': name, 'sha256': digest_file(destination / name)})
    report = dict(schema_version=1, recipe=recipe['id'], host=recipe['host'], backbone=recipe['backbone'],
        transfer=recipe['transfer'], seed=recipe['seed'], scope=recipe.get('scope', 'main-or-host'),
        status='captured_state_exported_requires_inference_parity_and_host_adapter',
        numerical_arguments=recipe['argv'], target_coverage=coverage,
        reference_sha256=reference['reference_sha256'], original_model_inputs=dependencies,
        prepared_source_files=prepared_source,
        captured_state=exported, evidence=evidence,
        capture_binding='terminal_execution_record' if bound else 'inventory_verified_at_export_only',
        limitations=['No fitting, image loading, GPU work, or pickle deserialization occurs during export.',
                     'This bundle does not include target/source images, original model weights, or an inference engine.',
                     'Download pinned original dependencies separately; their original licenses still apply.',
                     'Captured tensors may retain historical source-path metadata; they are not executed as instructions.',
                     'Per-image inference parity and a working host adapter are still required before deployment.'])
    (destination / 'manifest.json').write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    return report
