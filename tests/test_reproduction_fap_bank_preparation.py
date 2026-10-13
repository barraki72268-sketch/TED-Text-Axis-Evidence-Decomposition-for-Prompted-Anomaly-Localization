import tempfile
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from reproduction.run import validate_prepared


class FAPSourceOnlyPreparation(unittest.TestCase):
    def test_actual_cpu_cli_evidence_is_not_bank_construction(self):
        root=Path(__file__).resolve().parents[1]
        folder=root/'reproduction/validation/fresh-bank-20261013/fap-cold-preparation-v1'
        read=lambda n:json.loads((folder/n).read_bytes())
        proof,plan=read('proof.json'),read('run.json')
        for e in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((folder/e['file']).read_bytes()).hexdigest(),e['sha256'])
        self.assertTrue(proof['actual_CLI_executed'] and proof['ordinary_target_evaluation_rejected'])
        self.assertFalse(proof['historical_bank_inputs_required'] or proof['target_dataset_inputs_required'])
        self.assertFalse(proof['bank_collection_performed'] or proof['GPU_submitted'] or proof['target_evaluation_performed'])
        self.assertEqual(proof['denial_selfchecks'],3)
        self.assertEqual(proof['blocked_reads_after_selfcheck'],0)
        self.assertEqual(set(plan['datasets']),{'mvtec'})
        self.assertEqual(plan['bank_path_changes'],[])
        self.assertEqual(proof['plan_sha256'],hashlib.sha256((folder/'run.json').read_bytes()).hexdigest())
        self.assertEqual(proof['helper_sha256'],hashlib.sha256((folder/'wrapper.py').read_bytes()).hexdigest())

    def test_existing_bank_cache_rejected_before_source_collection(self):
        recipe=dict(host='FAPrompt',transfer='mvtec2btad')
        plan=dict(recipe='test-fap',preparation_scope='source_bank_collection_only',
                  datasets=dict(mvtec={}),bank_path_changes=[],historical_bank_inputs_required=False,
                  target_dataset_inputs_required=False)
        with tempfile.TemporaryDirectory() as d:
            w=Path(d);cache=w/'source/neurips2026/results/bank_cache';cache.mkdir(parents=True)
            (cache/'old-bank.pt').write_bytes(b'prior bank')
            with patch('reproduction.run.read_json',return_value=plan), patch('reproduction.run.execution_recipe',return_value=recipe):
                with self.assertRaisesRegex(ValueError,'FAPrompt source-only bank cache must remain empty'):
                    validate_prepared(w,w,allow_bank_collection=True)
                with self.assertRaisesRegex(ValueError,'cannot execute target evaluation'):
                    validate_prepared(w,w)


if __name__=='__main__':unittest.main()
