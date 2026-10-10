import tempfile
import json
import hashlib
from pathlib import Path
import unittest
import zipfile

from reproduction.bayes_bank_build import recorded_arguments


class BayesBankBuildTests(unittest.TestCase):
    def test_live_source_collection_evidence_and_cpu_tensor_audit(self):
        base = Path(__file__).resolve().parents[1] / 'reproduction/validation/fresh-bank-20261010/bayes-bplus-mvtec-s0'
        read = lambda n: json.loads((base / n).read_text(encoding='utf-8'))
        for item in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((base / item['file']).read_bytes()).hexdigest(), item['sha256'])
        proof, audit = read('construction.json'), read('cpu-audit.json')
        self.assertEqual(proof['status'], 'completed')
        self.assertEqual(proof['allocation']['job_id'], '13805')
        self.assertEqual(proof['entrypoint_sha256'], hashlib.sha256((base / 'entrypoint.py').read_bytes()).hexdigest())
        self.assertEqual(proof['plan_sha256'], hashlib.sha256((base / 'run.json').read_bytes()).hexdigest())
        self.assertFalse(proof['historical_bank_loaded'])
        self.assertFalse(proof['target_evaluation_performed'])
        self.assertFalse(proof['paper_metric_reproduction_verified'])
        self.assertTrue(proof['target_and_historical_bank_reads_denied'])
        self.assertTrue(audit['weights_only_cpu_load'])
        self.assertEqual(audit['bank_sha256'], proof['bank_sha256'])
        self.assertEqual(len(audit['layers']), 4)
        for layer in audit['layers'].values():
            self.assertEqual(layer['fp_shape'], [240, 640])
            self.assertEqual(layer['defect_shape'], [1024, 640])
            self.assertLess(layer['fp_norm_max_error'], 3e-7)
            self.assertLess(layer['defect_norm_max_error'], 3e-7)

    def test_original_parser_preserves_source_budget_without_calling_evaluation(self):
        root = Path(__file__).resolve().parents[1] / 'reproduction'
        with zipfile.ZipFile(root / 'source.zip') as archive:
            script = archive.read('neurips2026/scripts/probe_bayespfl_ted_smoke_20260501.py')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source/neurips2026/scripts/probe.py'
            path.parent.mkdir(parents=True)
            path.write_bytes(script)
            args = recorded_arguments(path, ['--dataset', 'mvtec', '--seed', '1',
                                             '--max_bank_per_layer', '512', '--image_size', '240'])
            self.assertEqual(args.max_good_per_class, 2)
            self.assertEqual(args.max_defect_per_class, 4)
            self.assertEqual(args.max_defect_per_image, 64)
            self.assertEqual(args.seed, 1)
            self.assertEqual(args.image_size, 240)
            self.assertEqual(args.dataset, 'mvtec')
            # No torch/model/data dependencies are imported by parsing. The
            # original main's evaluate() and output writes cannot run here.
            self.assertFalse((Path(directory) / 'source/neurips2026/results').exists())


if __name__ == '__main__':
    unittest.main()
