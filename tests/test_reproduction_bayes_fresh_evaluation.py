import unittest
import hashlib
import json
from pathlib import Path

from reproduction.bayes_fresh_evaluation import verify_collection_settings


class FreshBankEvaluationGuards(unittest.TestCase):
    def test_actual_cpu_cli_proof_binds_code_plan_and_unallocated_run_rejection(self):
        base = Path(__file__).resolve().parents[1] / 'reproduction/validation/fresh-bank-20261011/bayes-fresh-evaluation-preparation-v2'
        read = lambda n: json.loads((base/n).read_bytes())
        for item in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((base/item['file']).read_bytes()).hexdigest(), item['sha256'])
        proof, plan, old = read('proof.json'), read('run.json'), read('historical-preparation-run.json')
        self.assertEqual(proof['plan_sha256'], hashlib.sha256((base/'run.json').read_bytes()).hexdigest())
        self.assertTrue(proof['actual_cli_executed'] and proof['unallocated_gpu_run_rejected'])
        self.assertFalse(proof['target_evaluation_performed'] or proof['gpu_execution_performed'])
        self.assertTrue(proof['historical_inputs_required_for_preparation'])
        differences = [i for i, (a,b) in enumerate(zip(plan['argv'], old['argv'])) if a != b]
        self.assertEqual(differences, [plan['argv'].index('--bank_cache_path')+1])
        for name, sha in plan['fresh_bank_evaluation']['files'].items():
            self.assertEqual(hashlib.sha256((base/name).read_bytes()).hexdigest(), sha)

    def setUp(self):
        self.arguments = dict(seed=0, max_good_per_class=8, image_size=240,
                              checkpoint_path='/new/epoch_post_15_1.pth', data_path='/new/mvtec')
        self.proof = dict(status='completed', recipe='bayes-test', historical_bank_loaded=False,
                          target_evaluation_performed=False, target_and_historical_bank_reads_denied=True,
                          recorded_arguments={**self.arguments, 'data_path': '/old/mvtec'})
        self.plan = dict(recipe='bayes-test')

    def test_paths_can_relocate_without_changing_collection_settings(self):
        verify_collection_settings(self.proof, self.plan, self.arguments)

    def test_bank_from_another_seed_budget_or_resolution_is_rejected(self):
        for key, value in [('seed', 1), ('max_good_per_class', 2), ('image_size', 224)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'settings differ'):
                verify_collection_settings(self.proof, self.plan, {**self.arguments, key: value})

    def test_stage_filename_must_be_preserved(self):
        with self.assertRaisesRegex(ValueError, 'stage differs'):
            verify_collection_settings(self.proof, self.plan,
                                       {**self.arguments, 'checkpoint_path': '/new/epoch_post_15_2.pth'})

    def test_target_access_or_historical_cache_collection_is_rejected(self):
        for field, value in [('historical_bank_loaded', True), ('target_evaluation_performed', True),
                             ('target_and_historical_bank_reads_denied', False), ('status', 'failed')]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'provenance differs'):
                verify_collection_settings({**self.proof, field: value}, self.plan, self.arguments)


if __name__ == '__main__':
    unittest.main()
