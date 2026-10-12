import ast
import hashlib
import json
from pathlib import Path
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class AdaptBankSourceEvidence(unittest.TestCase):
    def test_recorded_branch_collector_and_evaluator_preserve_original_bytes(self):
        folder = ROOT/'research/bank_construction'
        manifest = json.loads((folder/'manifest.json').read_bytes())
        with zipfile.ZipFile(ROOT/'reproduction/source.zip') as archive:
            for name in ['analyze_adaptclip_vl_score_forms.py', 'official_parallel_test_adaptclip_vlrefine.py']:
                item = next(x for x in manifest['files'] if x['file'] == name)
                data = (folder/name).read_bytes()
                self.assertEqual(data, archive.read(item['archive_member']))
                self.assertEqual(hashlib.sha256(data).hexdigest(), item['sha256'])
        tree = ast.parse((folder/'official_parallel_test_adaptclip_vlrefine.py').read_bytes())
        self.assertTrue(any(isinstance(n, ast.ImportFrom) and n.module == 'analyze_adaptclip_vl_score_forms'
                            and any(a.name == 'get_or_collect_vl_banks' for a in n.names) for n in tree.body))

    def test_clean_public_cpu_preparation_claims_no_bank_generation(self):
        folder = ROOT/'reproduction/validation/fresh-bank-20261013/adapt-cold-public-v3'
        read = lambda n: json.loads((folder/n).read_bytes())
        proof, plan = read('proof.json'), read('run.json')
        for e in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((folder/e['file']).read_bytes()).hexdigest(), e['sha256'])
        self.assertTrue(proof['actual_CLI_executed'] and proof['clean_public_checkout'])
        self.assertEqual(proof['public_commit'], '61de3cbef1732d03ae69f25296ecad3fbf326ed3')
        self.assertFalse(proof['bank_collection_performed'] or proof['target_evaluation_performed'] or proof['GPU_submitted'])
        self.assertEqual(proof['blocked_reads_after_selfcheck'], 0)
        self.assertEqual(proof['plan_sha256'], hashlib.sha256((folder/'run.json').read_bytes()).hexdigest())
        self.assertEqual(set(plan['datasets']), {'mvtec'})


if __name__ == '__main__':
    unittest.main()
