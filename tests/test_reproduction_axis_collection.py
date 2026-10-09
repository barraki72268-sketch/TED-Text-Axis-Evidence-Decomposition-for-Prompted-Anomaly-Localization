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
