import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.aaclip_bank_build import original_structure,parser_arguments,collect

ROOT=Path(__file__).resolve().parents[1]


class OriginalAAConstruction(unittest.TestCase):
    def test_clean_public_inspection_preserves_scope(self):
        folder=ROOT/'reproduction/validation/fresh-bank-20261013/aaclip-bank-public-inspection-v2'
        read=lambda n:json.loads((folder/n).read_bytes())
        proof=read('proof.json')
        for e in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((folder/e['file']).read_bytes()).hexdigest(),e['sha256'])
        self.assertTrue(proof['actual_CPU_CLI_inspection'] and proof['clean_public_checkout'])
        self.assertFalse(proof['prototype_overlay'] or proof['bank_collection_performed'] or proof['GPU_submitted'])
        self.assertEqual(proof['plan_sha256'],hashlib.sha256((folder/'run.json').read_bytes()).hexdigest())
        self.assertEqual(proof['helper_sha256'],hashlib.sha256((folder/'wrapper.py').read_bytes()).hexdigest())
        self.assertEqual(proof['denial_selfchecks'],3)
        self.assertEqual(proof['blocked_reads_after_selfcheck'],[])
        self.assertTrue(proof['unallocated_collection_rejected'])
        self.assertFalse(proof['destination_created'] or proof['target_evaluation_performed'])

    def test_actual_cpu_inspection_evidence_is_bound_and_scoped(self):
        folder=ROOT/'reproduction/validation/fresh-bank-20261013/aaclip-bank-build-inspection-v1'
        read=lambda n:json.loads((folder/n).read_bytes())
        proof=read('proof.json')
        for e in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((folder/e['file']).read_bytes()).hexdigest(),e['sha256'])
        self.assertEqual(proof['code_sha256'],hashlib.sha256((folder/'code.py').read_bytes()).hexdigest())
        self.assertEqual(proof['plan_sha256'],hashlib.sha256((folder/'run.json').read_bytes()).hexdigest())
        self.assertEqual(proof['helper_sha256'],hashlib.sha256((folder/'wrapper.py').read_bytes()).hexdigest())
        self.assertTrue(proof['actual_CPU_CLI_inspection'] and proof['unallocated_collection_rejected'])
        self.assertFalse(proof['destination_created'] or proof['bank_collection_performed'] or proof['GPU_submitted'])
        self.assertEqual(proof['denial_selfchecks'],3)
        self.assertEqual(proof['blocked_reads_after_selfcheck'],[])
        self.assertEqual(read('static-call-audit.json')['recipe_count'],48)

    def test_recorded_parser_and_complete_original_model_setup(self):
        script=ROOT/'research/bank_construction/official_parallel_test_aaclip.py'
        _,_,initialization,direct=original_structure(script)
        text=ast.unparse(ast.Module(body=initialization,type_ignores=[]))
        for marker in ['ensure_aaclip_pretrained_weight()', 'model.image_adapter', 'model.text_adapter',
                       'model.eval()', 'p.requires_grad = False']:
            self.assertIn(marker,text)
        self.assertNotIn('get_or_collect_source_banks',text)
        self.assertNotIn('evaluate_target_dataset',text)
        self.assertFalse({'preset','ckpt_dir','cache_dir'} & set(direct))
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        recipes=json.loads((ROOT/'reproduction/execution-recipes.json').read_bytes())['recipes']
        count=0
        for recipe in recipes:
            if recipe['host']!='AA-CLIP':continue
            args,cfg=parser_arguments(script,recipe['argv'],torch)
            self.assertEqual(args.preset,recipe['transfer'])
            self.assertEqual(args.seed,recipe['seed'])
            for name in ['img_size','max_fp_per_class','max_defect_per_class']:
                self.assertEqual(getattr(args,name),int(recipe['argv'][recipe['argv'].index('--'+name)+1]))
            self.assertEqual(args.device,recipe['argv'][recipe['argv'].index('--device')+1])
            self.assertEqual(args.model_name,recipe['argv'][recipe['argv'].index('--model_name')+1])
            count+=1
        self.assertEqual(count,48)

    def test_no_allocation_rejects_before_model_import_or_output_creation(self):
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/'new-bank'
            with patch('reproduction.axis_run.require_live_gpu_allocation',side_effect=RuntimeError('No allocation')):
                with self.assertRaisesRegex(RuntimeError,'No allocation'):
                    collect(ROOT/'reproduction',Path(d)/'not-prepared',dest)
            self.assertFalse(dest.exists())
