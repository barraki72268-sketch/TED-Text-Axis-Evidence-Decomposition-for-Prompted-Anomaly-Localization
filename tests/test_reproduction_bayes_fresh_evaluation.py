import unittest
import hashlib
import json
from pathlib import Path

from reproduction.bayes_fresh_evaluation import verify_collection_settings, verify_preparation_scope


class FreshBankEvaluationGuards(unittest.TestCase):
    def test_actual_bank_independent_cpu_preparation_binds_proof_and_no_gpu_claim(self):
        base = Path(__file__).resolve().parents[1] / 'reproduction/validation/fresh-bank-20261011/bayes-fresh-evaluation-bankfree-v4'
        read = lambda n: json.loads((base/n).read_bytes())
        proof, plan = read('proof.json'), read('run.json')
        for item in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((base/item['file']).read_bytes()).hexdigest(), item['sha256'])
        self.assertTrue(proof['historical_bank_read_denial_selfcheck_passed'])
        self.assertEqual(proof['historical_bank_reads_attempted_after_selfcheck'], 0)
        self.assertFalse(proof['historical_inputs_required_for_preparation'])
        self.assertTrue(proof['unallocated_gpu_run_rejected'] and proof['prototype_overlay'])
        self.assertFalse(proof['target_evaluation_performed'] or proof['gpu_execution_performed'])
        self.assertEqual(set(proof['model_only_object_sha256']), {o['sha256'] for o in plan['verified_objects']})
        self.assertEqual(proof['plan_sha256'], hashlib.sha256((base/'run.json').read_bytes()).hexdigest())
        for name, sha in plan['fresh_bank_evaluation']['files'].items():
            self.assertEqual(hashlib.sha256((base/name).read_bytes()).hexdigest(), sha)

    def test_bank_independent_scope_requires_target_data_and_excludes_historical_object(self):
        recipe = dict(transfer='mvtec2btad', path_bindings={'bank': dict(argument='--bank_cache_path', sha256='historical')})
        plan = dict(fresh_bank_evaluation=dict(historical_inputs_required_for_preparation=False),
                    preparation_scope='fresh_bank_target_evaluation', historical_bank_inputs_required=False,
                    target_dataset_inputs_required=True, datasets=dict(mvtec={}, btad={}),
                    verified_objects=[dict(sha256='model')])
        verify_preparation_scope(recipe, plan)
        for change in [dict(verified_objects=[dict(sha256='historical')]), dict(datasets=dict(mvtec={})),
                       dict(historical_bank_inputs_required=True), dict(target_dataset_inputs_required=False),
                       dict(preparation_scope='source_bank_collection_only')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_preparation_scope(recipe, {**plan, **change})

    def test_clean_public_checkout_cli_preparation_is_bound_and_not_gpu_evaluation(self):
        base = Path(__file__).resolve().parents[1] / 'reproduction/validation/fresh-bank-20261011/bayes-fresh-evaluation-public-v3'
        read = lambda n: json.loads((base/n).read_bytes())
        index, proof = read('index.json'), read('proof.json')
        for item in index['evidence']:
            self.assertEqual(hashlib.sha256((base/item['file']).read_bytes()).hexdigest(), item['sha256'])
        self.assertTrue(proof['public_checkout_clean'] and proof['actual_cli_executed'])
        self.assertEqual(proof['public_commit'], 'ca4d6f944413f51607674a9b39febfd4932a1d44')
        self.assertEqual(proof['plan_sha256'], hashlib.sha256((base/'run.json').read_bytes()).hexdigest())
        self.assertEqual(proof['helper_sha256'], hashlib.sha256((base/'wrapper.py').read_bytes()).hexdigest())
        self.assertTrue(proof['unallocated_gpu_run_rejected'])
        self.assertFalse(proof['target_evaluation_performed'] or proof['gpu_execution_performed'])
        for name, sha in proof['evidence_files'].items():
            self.assertEqual(hashlib.sha256((base/name).read_bytes()).hexdigest(), sha)

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
