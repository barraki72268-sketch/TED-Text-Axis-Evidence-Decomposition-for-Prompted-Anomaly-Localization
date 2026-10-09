import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from reproduction.axis_collection import verify_asset
from reproduction.axis_figure import read_inputs


ROOT = Path(__file__).resolve().parents[1] / 'reproduction'


class AxisCollectionTests(unittest.TestCase):
    def test_linux_preparation_preserves_source_except_recorded_paths_and_numeric_arguments(self):
        manifest, data, _ = read_inputs(ROOT)
        report = json.loads((ROOT / 'validation/a10-20261009/figure3-linux-preparation.json').read_text())
        original = {key:value for key,value in report.items() if key not in {'plan_sha256','public_runtime_commit'}}
        raw = (json.dumps(original, indent=2) + '\n').encode()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), report['plan_sha256'])
        self.assertEqual(report['status'], 'prepared')
        self.assertFalse(report['gpu_execution'])
        self.assertEqual(report['pre_run_bank_entries'], 0)
        self.assertEqual(report['source_bank_policy'], 'fresh_source_only')
        self.assertEqual(report['datasets']['mvtec']['files_verified'], 6612)
        self.assertEqual(report['datasets']['visa']['files_verified'], 12021)
        with zipfile.ZipFile(ROOT / 'source.zip') as source:
            for change in report['source_path_changes']:
                raw = source.read(change['path'])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), change['original_sha256'])
                text = raw.decode()
                for old, new in report['path_mapping'].items():
                    text = text.replace(old, new)
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), change['relocated_sha256'])
        command = report['command']
        for flag,value in [('--image_size',518),('--depth',9),('--n_ctx',12),('--t_n_ctx',4),
                           ('--hard_frac',0.01),('--tau',20.0),('--max_good_per_class',2),
                           ('--max_defect_per_class',4),('--max_bank_per_layer',512),
                           ('--max_normal_eval_images',24),('--max_anomaly_eval_images',24)]:
            self.assertEqual(command.count(flag), 1)
            self.assertEqual(command[command.index(flag)+1], str(value))
        self.assertEqual(command[command.index('--features_list')+1:-1], ['6','12','18','24'])
        self.assertEqual(command[-1], '--refresh_bank_cache')
        self.assertNotIn('--seed', command)

    def test_reconstructed_collection_is_bound_to_summary_and_original_defaults(self):
        manifest, data, _ = read_inputs(ROOT)
        specification = manifest['fresh_collection']
        self.assertEqual(specification['observed_summary'], data['summary'])
        self.assertFalse(specification['historical_bank']['exists_at_original_path'])
        self.assertEqual(specification['historical_argv'], 'not recovered')
        self.assertEqual(specification['checkpoint']['path'], data['summary']['checkpoint_path'])
        catalog = json.loads((ROOT / 'backbones.json').read_text())['artifacts']
        self.assertIn(specification['backbone'], catalog)
        helper = specification['model_helper']
        with zipfile.ZipFile(ROOT / 'source.zip') as source:
            raw = source.read(helper['path'])
            collector = ast.parse(source.read(manifest['collector']['path']).decode())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), helper['sha256'])
        build = next(node for node in ast.parse(raw.decode()).body
                     if isinstance(node, ast.FunctionDef) and node.name == 'build_model')
        self.assertEqual(ast.literal_eval(build.args.defaults[0]), helper['model'])
        self.assertEqual(ast.literal_eval(build.args.defaults[1]), helper['dpam_layer'])
        defaults = {}
        for node in ast.walk(collector):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                for keyword in node.keywords:
                    if keyword.arg == 'default':
                        try:
                            defaults[ast.literal_eval(node.args[0])] = ast.literal_eval(keyword.value)
                        except (ValueError, TypeError):
                            pass
        for key in ['depth', 'n_ctx', 't_n_ctx']:
            self.assertEqual(helper[key], defaults['--' + key])
        self.assertNotIn('--seed', defaults)

    def test_reject_wrong_checkpoint_or_backbone_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights'
            path.write_bytes(b'wrong bytes')
            with self.assertRaisesRegex(ValueError, 'asset hash/size'):
                verify_asset(path, {'bytes':11, 'sha256':'0' * 64})
