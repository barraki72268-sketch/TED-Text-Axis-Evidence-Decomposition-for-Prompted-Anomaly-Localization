from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

from reproduction.run import validate_prepared


class AASourceOnlyPreparation(unittest.TestCase):
    def test_actual_preparation_proof_binds_model_dependencies_and_scope(self):
        root=Path(__file__).resolve().parents[1]
        folder=root/'reproduction/validation/fresh-bank-20261013/aaclip-cold-preparation-v1'
        read=lambda n:json.loads((folder/n).read_bytes())
        proof,plan=read('proof.json'),read('run.json')
        for e in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((folder/e['file']).read_bytes()).hexdigest(),e['sha256'])
        self.assertTrue(proof['actual_CLI_executed'] and proof['ordinary_target_evaluation_rejected'])
        self.assertFalse(proof['bank_collection_performed'] or proof['GPU_submitted'] or proof['target_evaluation_performed'])
        self.assertEqual(proof['denial_selfchecks'],3)
        self.assertEqual(set(plan['datasets']),{'mvtec'})
        self.assertEqual({x['sha256'] for x in plan['verified_objects']},set(proof['model_objects']))
        self.assertEqual(len(proof['model_objects']),4)
        self.assertEqual(plan['bank_path_changes'],[])
        self.assertEqual(read('static-call-audit.json')['recipe_count'],48)

    def test_old_cache_and_target_execution_rejected(self):
        recipe=dict(host='AA-CLIP',transfer='mvtec2btad')
        plan=dict(recipe='test-aa',preparation_scope='source_bank_collection_only',
                  datasets=dict(mvtec={}),bank_path_changes=[],historical_bank_inputs_required=False,
                  target_dataset_inputs_required=False)
        with tempfile.TemporaryDirectory() as d:
            w=Path(d);cache=w/'source/neurips2026/results/bank_cache';cache.mkdir(parents=True)
            (cache/'old-bank.pt').write_bytes(b'prior bank')
            with patch('reproduction.run.read_json',return_value=plan), patch('reproduction.run.execution_recipe',return_value=recipe):
                with self.assertRaisesRegex(ValueError,'AA-CLIP source-only bank cache must remain empty'):
                    validate_prepared(w,w,allow_bank_collection=True)
                with self.assertRaisesRegex(ValueError,'cannot execute target evaluation'):
                    validate_prepared(w,w)
