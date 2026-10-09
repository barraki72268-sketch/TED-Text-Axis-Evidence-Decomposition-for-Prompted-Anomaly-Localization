import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from reproduction.aa_serving_bundle import build_aa_serving_bundle
from reproduction.checkpoint_download import digest_file


class AAServingBundleTests(unittest.TestCase):
    def fixture(self, base):
        work, catalog = base / 'original', base / 'catalog'
        work.mkdir()
        catalog.mkdir()
        recipe = dict(id='aa-test', host='AA-CLIP')
        def write(relative, payload):
            path = work / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return path
        write('run.json', b'{"recipe":"aa-test"}')
        write('execution.json', json.dumps({'plan_sha256': digest_file(work / 'run.json')}).encode())
        source = write('source/neurips2026/scripts/official_parallel_test_aaclip.py', b'# original absolute paths preserved\n')
        bootstrap = write('source/neurips2026/AA-CLIP/model/ViT-L-14-336px.pt', b'opaque-backbone')
        write('clip-cache/ViT-L-14-336px.pt', bootstrap.read_bytes())
        image = write('checkpoints/image_adapter.pth', b'opaque-image-adapter')
        text = write('checkpoints/text_adapter.pth', b'opaque-text-adapter')
        (catalog / 'host-checkpoints.json').write_text(json.dumps({'bindings': [dict(recipe='aa-test',
            checkpoint_assets=[dict(checkpoint_filename=p.name, sha256=digest_file(p)) for p in [image, text]])]}))
        (catalog / 'backbones.json').write_text(json.dumps({'bindings': [dict(recipe='aa-test', sha256=digest_file(bootstrap))],
            'artifacts': [dict(id='openai_vit_l14_336', filename='ViT-L-14-336px.pt', sha256=digest_file(bootstrap))]}))
        def verified_export(root, workspace, destination):
            destination.mkdir(parents=True)
            manifest = dict(prepared_source_files=[dict(path=str(source.relative_to(work / 'source')).replace('\\', '/'), sha256=digest_file(source))])
            (destination / 'manifest.json').write_text(json.dumps(manifest))
            (destination / 'summary.json').write_text(json.dumps({'ckpt_dir': str(work / 'checkpoints')}))
            return manifest
        return work, catalog, recipe, verified_export

    def test_relocation_preserves_exact_bytes_and_original_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            work, catalog, recipe, verified_export = self.fixture(base)
            with patch('reproduction.aa_serving_bundle.execution_recipe', return_value=recipe), \
                    patch('reproduction.aa_serving_bundle.export_captured_run', side_effect=verified_export):
                report = build_aa_serving_bundle(catalog, work, base / 'bundle')
                with self.assertRaises(FileExistsError):
                    build_aa_serving_bundle(catalog, work, base / 'bundle')
            moved = base / 'unrelated-location'
            shutil.move(base / 'bundle', moved)
            for entry in report['files']:
                self.assertEqual(digest_file(moved / entry['path']), entry['sha256'])
                self.assertFalse((moved / entry['path']).is_symlink())
            self.assertEqual((moved / 'run.json').read_bytes(), (work / 'run.json').read_bytes())
            self.assertEqual(report['export_sha256'], digest_file(moved / 'export/manifest.json'))
            self.assertEqual(report['status'], 'packaged_requires_relocated_inference_parity')
            self.assertFalse((moved / 'data').exists())

    def test_changed_adapter_is_rejected_and_failed_attempt_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            work, catalog, recipe, verified_export = self.fixture(base)
            (work / 'checkpoints/image_adapter.pth').write_bytes(b'changed')
            with patch('reproduction.aa_serving_bundle.execution_recipe', return_value=recipe), \
                    patch('reproduction.aa_serving_bundle.export_captured_run', side_effect=verified_export):
                with self.assertRaisesRegex(ValueError, 'Original serving input changed'):
                    build_aa_serving_bundle(catalog, work, base / 'failed-bundle')
            self.assertTrue((base / 'failed-bundle/export/manifest.json').exists())
            self.assertFalse((base / 'failed-bundle/serving-bundle.json').exists())
