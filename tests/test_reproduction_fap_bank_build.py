import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.fap_bank_build import original_calls, parser_arguments, collect

ROOT = Path(__file__).resolve().parents[1]


class OriginalFapConstruction(unittest.TestCase):
    def test_clean_public_inspection_and_scope(self):
        folder=ROOT/'reproduction/validation/fresh-bank-20261013/fap-bank-public-inspection-v2'
        read=lambda n:json.loads((folder/n).read_bytes())
        proof=read('proof.json')
        for item in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((folder/item['file']).read_bytes()).hexdigest(),item['sha256'])
        self.assertTrue(proof['clean_public_checkout'] and proof['actual_CPU_CLI_inspection'])
        self.assertFalse(proof['prototype_overlay'] or proof['bank_collection_performed'] or proof['GPU_submitted'])
        self.assertEqual(proof['public_commit'],'13316b8317661735504b504f759fa7fd27a23b67')
        self.assertEqual(proof['plan_sha256'],hashlib.sha256((folder/'run.json').read_bytes()).hexdigest())
        self.assertEqual(proof['helper_sha256'],hashlib.sha256((folder/'wrapper.py').read_bytes()).hexdigest())
        self.assertTrue(proof['unallocated_collection_rejected'])
        self.assertFalse(proof['destination_created'] or proof['target_evaluation_performed'])

    def test_actual_inspection_evidence_is_scoped_and_bound(self):
        folder=ROOT/'reproduction/validation/fresh-bank-20261013/fap-bank-build-inspection-v1'
        read=lambda n:json.loads((folder/n).read_bytes())
        proof=read('proof.json')
        for item in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((folder/item['file']).read_bytes()).hexdigest(),item['sha256'])
        self.assertEqual(proof['code_sha256'],hashlib.sha256((folder/'code.py').read_bytes()).hexdigest())
        self.assertEqual(proof['helper_sha256'],hashlib.sha256((folder/'wrapper.py').read_bytes()).hexdigest())
        self.assertTrue(proof['actual_CPU_CLI_inspection'] and proof['unallocated_collection_rejected'])
        self.assertFalse(proof['destination_created'] or proof['bank_collection_performed'] or proof['GPU_submitted'])
        self.assertEqual(proof['denial_selfchecks'],3)
        self.assertEqual(proof['blocked_reads_after_selfcheck'],[])
        self.assertEqual(read('static-call-audit.json')['recipe_count'],20)

    def test_recorded_parser_presets_and_direct_model_call(self):
        script=ROOT/'research/bank_construction/official_parallel_test_faprompt.py'
        _,_,direct,model=original_calls(script)
        self.assertEqual(ast.unparse(direct['dpam_layer']), "prompt_load_info['effective_dpam_layer']")
        self.assertEqual(ast.unparse(model.func),'build_model')
        self.assertFalse({'cache_dir','force_rebuild_bank','checkpoint_path','preset'} & set(direct))
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        recipes=json.loads((ROOT/'reproduction/execution-recipes.json').read_bytes())['recipes']
        count=0
        for recipe in recipes:
            if recipe['host']!='FAPrompt': continue
            args,cfg=parser_arguments(script,recipe['argv'],torch)
            self.assertEqual(args.preset,recipe['transfer'])
            self.assertEqual(args.seed,recipe['seed'])
            for name in ['image_size','max_fp_per_class','max_defect_per_class','max_fp_total','max_defect_total']:
                self.assertEqual(getattr(args,name),int(recipe['argv'][recipe['argv'].index('--'+name)+1]))
            self.assertEqual(args.device,recipe['argv'][recipe['argv'].index('--device')+1])
            source=recipe['transfer'].split('2',1)[0]
            self.assertEqual(cfg['source_dataset'],{'mvtec':'mvtecad'}.get(source,source))
            count+=1
        self.assertEqual(count,20)

    def test_no_allocation_rejects_before_model_import_or_output_creation(self):
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/'new-bank'
            with patch('reproduction.axis_run.require_live_gpu_allocation',side_effect=RuntimeError('No allocation')):
                with self.assertRaisesRegex(RuntimeError,'No allocation'):
                    collect(ROOT/'reproduction',Path(d)/'not-prepared',dest)
            self.assertFalse(dest.exists())


if __name__=='__main__': unittest.main()
