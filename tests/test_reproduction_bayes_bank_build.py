import tempfile
import json
import hashlib
from pathlib import Path
import unittest
import zipfile
from unittest.mock import patch

from reproduction.bayes_bank_build import recorded_arguments


class BayesBankBuildTests(unittest.TestCase):
    def test_clean_public_checkout_executes_cold_preparation_cli(self):
        base = Path(__file__).resolve().parents[1] / 'reproduction/validation/fresh-bank-20261011/bayes-cold-public-checkout-v2'
        read = lambda n: json.loads((base / n).read_text(encoding='utf-8'))
        for item in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((base / item['file']).read_bytes()).hexdigest(), item['sha256'])
        proof, plan = read('proof.json'), read('run.json')
        self.assertTrue(proof['public_checkout_clean'] and proof['actual_cli_executed'])
        self.assertEqual(proof['public_commit'], '86dfdb4f5ca72707494853289a4276a3fa1996ac')
        self.assertEqual(proof['plan_sha256'], hashlib.sha256((base / 'run.json').read_bytes()).hexdigest())
        self.assertEqual(set(plan['datasets']), {'mvtec'})
        self.assertEqual(len(plan['verified_objects']), 2)
        self.assertFalse(plan['bank_path_changes'])
        self.assertTrue(proof['ordinary_evaluation_rejected'] and proof['numerical_args_equal_original'])
        self.assertFalse(proof['gpu_collection_verified'] or proof['target_evaluation_verified'])

    def test_cold_cpu_preparation_excludes_bank_and_target_inputs(self):
        base = Path(__file__).resolve().parents[1] / 'reproduction/validation/fresh-bank-20261011/bayes-cold-preparation-v1'
        read = lambda n: json.loads((base / n).read_text(encoding='utf-8'))
        for item in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((base / item['file']).read_bytes()).hexdigest(), item['sha256'])
        proof, plan = read('proof.json'), read('run.json')
        self.assertEqual(set(plan['datasets']), {'mvtec'})
        self.assertEqual(plan['bank_path_changes'], [])
        self.assertEqual(len(plan['verified_objects']), 2)
        self.assertTrue(proof['cpu_only'] and proof['numerical_args_equal_original'])
        self.assertTrue(proof['ordinary_evaluation_rejected'])
        self.assertEqual(proof['denial_selfchecks']['denied_bank_selfchecks'], 2)
        self.assertEqual(proof['denial_selfchecks']['denied_target_selfchecks'], 3)
        self.assertFalse(proof['gpu_collection_verified'])
        self.assertFalse(proof['target_evaluation_verified'])

    def test_source_collection_plan_cannot_run_target_evaluation(self):
        from reproduction.run import validate_prepared
        plan = {'recipe': 'test', 'preparation_scope': 'source_bank_collection_only'}
        recipe = {'host': 'BayesPFL', 'transfer': 'mvtec2btad'}
        with tempfile.TemporaryDirectory() as directory, patch('reproduction.run.read_json', return_value=plan), patch('reproduction.run.execution_recipe', return_value=recipe):
            with self.assertRaisesRegex(ValueError, 'cannot execute target evaluation'):
                validate_prepared(Path(directory), Path(directory))
            plan.update(datasets={'mvtec': {}}, bank_path_changes=[{'path': 'bank'}])
            with self.assertRaisesRegex(ValueError, 'target data or historical banks'):
                validate_prepared(Path(directory), Path(directory), allow_bank_collection=True)

    def test_public_checkout_independently_rebuilds_identical_source_tensors(self):
        base = Path(__file__).resolve().parents[1] / 'reproduction/validation/fresh-bank-20261010/bayes-bplus-public-checkout-v3'
        read = lambda n: json.loads((base / n).read_text(encoding='utf-8'))
        for item in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((base / item['file']).read_bytes()).hexdigest(), item['sha256'])
        proof, comparison = read('construction.json'), read('cpu-comparison.json')
        self.assertEqual(proof['allocation']['job_id'], '13808')
        self.assertEqual(proof['plan_sha256'], hashlib.sha256((base / 'run.json').read_bytes()).hexdigest())
        self.assertTrue(comparison['public_checkout_clean'])
        self.assertTrue(comparison['weights_only_cpu_load'])
        self.assertEqual(comparison['first_bank_sha256'], comparison['second_bank_sha256'])
        self.assertEqual(comparison['second_bank_sha256'], proof['bank_sha256'])
        self.assertEqual(len(comparison['tensor_rows']), 8)
        for row in comparison['tensor_rows']:
            self.assertTrue(row['exact'] and row['finite'])
            self.assertEqual(row['max_absolute_error'], 0)
        self.assertFalse(proof['target_evaluation_performed'])
        self.assertFalse(read('index.json')['paper_metric_reproduction_verified'])

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
