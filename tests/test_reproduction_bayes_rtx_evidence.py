import hashlib
import json
from pathlib import Path
import unittest

from reproduction.coverage import validate_coverage
from reproduction.metrics import compare, extract
from reproduction.recipe_lookup import reference_summary


class BayesRTXEvidenceTests(unittest.TestCase):
    def test_captured_readout_proof_binds_actual_code_and_keeps_image_gate(self):
        repo = Path(__file__).resolve().parents[1]
        base = repo / 'reproduction/validation/rtx-20261010/bayes-readout-v1'
        read = lambda name: json.loads((base / name).read_text(encoding='utf-8'))
        for item in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((base / item['file']).read_bytes()).hexdigest(), item['sha256'])
        proof = read('bayes-readout-parity-20261010-v1.json')
        self.assertEqual(proof['adapter_sha256'], hashlib.sha256((repo / 'ted/inference/bayespfl_captured.py').read_bytes()).hexdigest())
        self.assertEqual(proof['export_manifest_sha256'], hashlib.sha256((base / 'export-manifest.json').read_bytes()).hexdigest())
        manifest = read('export-manifest.json')
        self.assertEqual(manifest['host'], 'BayesPFL')
        script = 'neurips2026/scripts/probe_bayespfl_ted_smoke_20260501.py'
        bound = next(x for x in manifest['prepared_source_files'] if x['path'] == script)
        self.assertEqual(bound['sha256'], proof['original_script_sha256'])
        self.assertEqual(proof['max_absolute_map_error'], 0.0)
        self.assertEqual(proof['layers'], 4)
        self.assertFalse(proof['image_inference_verified'])
        self.assertFalse(proof['deployment_ready'])

    def test_original_checkpoint_binding_and_reported_metric_scope(self):
        root = Path(__file__).resolve().parents[1] / 'reproduction'
        base = root / 'validation/rtx-20261010'
        report = json.loads((base / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual(len(report['results']), 13)
        for result in report['results']:
            for item in result['evidence']:
                self.assertEqual(hashlib.sha256((base / item['file']).read_bytes()).hexdigest(), item['sha256'])
            folder = base / Path(result['evidence'][0]['file']).parent
            execution = json.loads((folder / 'execution.json').read_text(encoding='utf-8'))
            run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
            actual = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
            reference, expected = reference_summary(root, result['recipe'])
            cells = compare(extract(actual, reference['host']), extract(expected, reference['host']))
            self.assertEqual(cells, execution['comparison']['cells'])
            self.assertTrue(all(c['matches_printed_precision'] for c in cells))
            self.assertEqual(len(cells), 8)
            coverage = validate_coverage(root, reference, actual)
            self.assertEqual(coverage, result['target_coverage'])
            self.assertEqual(coverage['classes_checked'], {'btad': 3, 'mpdd': 6, 'mvtec': 15}[result['recipe'].split('2', 1)[1].split('-seed', 1)[0]])
            self.assertFalse(coverage['reported_image_counts_checked'])
            self.assertFalse(coverage['per_image_execution_verified'])
            checkpoint = run['argv'][run['argv'].index('--checkpoint_path') + 1]
            self.assertEqual(Path(checkpoint).name, 'epoch_post_15_1.pth')
            self.assertEqual(hashlib.sha256((folder / 'run.json').read_bytes()).hexdigest(), execution['plan_sha256'])
            self.assertEqual(hashlib.sha256((folder / 'summary.json').read_bytes()).hexdigest(), execution['comparison']['actual_sha256'])


if __name__ == '__main__':
    unittest.main()
