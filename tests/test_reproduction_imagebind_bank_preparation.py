from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

from reproduction.run import validate_prepared


class ImageBindSourceOnlyPreparation(unittest.TestCase):
    def test_actual_source_only_preparation_preserves_model_and_scope(self):
        root=Path(__file__).resolve().parents[1]
        folder=root/'reproduction/validation/fresh-bank-20261013/imagebind-cold-preparation-v1'
        read=lambda n:json.loads((folder/n).read_bytes())
        proof,plan=read('proof.json'),read('run.json')
        for e in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((folder/e['file']).read_bytes()).hexdigest(),e['sha256'])
        self.assertTrue(proof['actual_CLI_executed'] and proof['ordinary_target_evaluation_rejected'])
        self.assertFalse(proof['bank_collection_performed'] or proof['GPU_submitted'] or proof['target_evaluation_performed'])
        self.assertEqual(proof['model_objects'],['d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157'])
        self.assertEqual(set(plan['datasets']),{'mvtec'})
        self.assertEqual(plan['bank_path_changes'],[])
        self.assertEqual(Path(plan['argv'][plan['argv'].index('--bank_cache_path')+1]).name,'unavailable-historical-bank.pt')
        self.assertEqual(Path(plan['argv'][plan['argv'].index('--target_root')+1]).name,'unavailable-target')
        self.assertEqual(proof['denial_selfchecks'],3)
        self.assertEqual(proof['blocked_reads_after_selfcheck'],0)
        self.assertEqual(read('static-call-audit.json')['recipe_count'],5)

    def test_old_cache_and_target_execution_rejected(self):
        recipe=dict(host='RawImageBind',transfer='mvtec2btad')
        plan=dict(recipe='test-imagebind',preparation_scope='source_bank_collection_only',
                  datasets=dict(mvtec={}),bank_path_changes=[],historical_bank_inputs_required=False,
                  target_dataset_inputs_required=False)
        with tempfile.TemporaryDirectory() as d:
            w=Path(d);cache=w/'source/neurips2026/results/bank_cache';cache.mkdir(parents=True)
            (cache/'old-bank.pt').write_bytes(b'prior bank')
            with patch('reproduction.run.read_json',return_value=plan),patch('reproduction.run.execution_recipe',return_value=recipe):
                with self.assertRaisesRegex(ValueError,'RawImageBind source-only bank cache must remain empty'):
                    validate_prepared(w,w,allow_bank_collection=True)
                with self.assertRaisesRegex(ValueError,'cannot execute target evaluation'):
                    validate_prepared(w,w)
