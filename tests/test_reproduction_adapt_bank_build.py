import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.adapt_bank_build import parser_arguments, source_calls, collect

ROOT = Path(__file__).resolve().parents[1]


class AdaptOriginalCollectionCalls(unittest.TestCase):
    def test_clean_public_inspection_and_synthetic_protocol_are_scoped(self):
        folder = ROOT/'reproduction/validation/fresh-bank-20261013/adapt-bank-public-inspection-v3'
        read = lambda n: json.loads((folder/n).read_bytes())
        proof = read('proof.json')
        for e in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((folder/e['file']).read_bytes()).hexdigest(), e['sha256'])
        self.assertTrue(proof['clean_public_checkout'] and proof['actual_CLI_inspection'])
        self.assertTrue(proof['unallocated_collection_rejected'] and proof['synthetic_test_is_not_source_bank_reconstruction'])
        self.assertFalse(proof['bank_collection_performed'] or proof['GPU_submitted'] or proof['target_evaluation_performed'])
        self.assertEqual(proof['helper_sha256'],hashlib.sha256((folder/'wrapper.py').read_bytes()).hexdigest())
        self.assertEqual(proof['plan_sha256'],hashlib.sha256((folder/'run.json').read_bytes()).hexdigest())
        self.assertEqual({x['branch'] for x in proof['synthetic_cache_consumer_results']},{'vl','tl'})
        self.assertTrue(all(x['exact_tensors'] and x['metadata_equal'] and x['original_cache_getter_reused_without_mining']
                            for x in proof['synthetic_cache_consumer_results']))

    def test_actual_cpu_inspection_evidence_claims_no_collection(self):
        folder = ROOT/'reproduction/validation/fresh-bank-20261013/adapt-bank-build-inspection-v1'
        read = lambda n: json.loads((folder/n).read_bytes())
        proof = read('proof.json')
        for e in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((folder/e['file']).read_bytes()).hexdigest(), e['sha256'])
        self.assertEqual(proof['code_sha256'],hashlib.sha256((folder/'code.py').read_bytes()).hexdigest())
        self.assertEqual(proof['plan_sha256'],hashlib.sha256((folder/'run.json').read_bytes()).hexdigest())
        self.assertTrue(proof['unallocated_collection_rejected'])
        self.assertFalse(proof['destination_created'] or proof['bank_collection_performed'] or proof['GPU_submitted'])
        self.assertEqual(read('static-call-audit.json')['recipe_count'],46)

    def test_original_parser_and_calls_preserve_every_recorded_adapt_recipe(self):
        scripts = ROOT/'research/bank_construction'
        evaluator = scripts/'official_parallel_test_adaptclip_vlrefine.py'
        calls, model = source_calls(evaluator,scripts/'analyze_adaptclip_vl_score_forms.py')
        self.assertEqual(set(calls), {'vl','tl'})
        self.assertNotIn('checkpoint_path',calls['vl'])
        self.assertNotIn('bank_cache_path',calls['tl'])
        # This stub serves only the original parser's device-default expression.
        # Every audited recipe explicitly supplies its actual recorded device.
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        recipes = json.loads((ROOT/'reproduction/execution-recipes.json').read_bytes())['recipes']
        count = 0
        for r in recipes:
            if r['host'] != 'AdaptCLIP': continue
            a = parser_arguments(evaluator,r['argv'],torch)
            self.assertEqual(a.ted_insertion,'dual_branch')
            self.assertEqual(a.seed,r['seed'])
            for flag,name in [('--image_size','image_size'),('--max_bank_points','max_bank_points'),
                              ('--max_good_per_class','max_good_per_class'),('--max_defect_per_class','max_defect_per_class')]:
                self.assertEqual(getattr(a,name),int(r['argv'][r['argv'].index(flag)+1]))
            self.assertEqual(a.device,r['argv'][r['argv'].index('--device')+1])
            count += 1
        self.assertEqual(count,46)

    def test_no_allocation_rejects_before_model_import_or_output_creation(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)/'new-bank'
            with patch('reproduction.axis_run.require_live_gpu_allocation',side_effect=RuntimeError('No allocation')):
                with self.assertRaisesRegex(RuntimeError,'No allocation'):
                    collect(ROOT/'reproduction',Path(d)/'not-prepared',dest)
            self.assertFalse(dest.exists())


if __name__ == '__main__':
    unittest.main()
