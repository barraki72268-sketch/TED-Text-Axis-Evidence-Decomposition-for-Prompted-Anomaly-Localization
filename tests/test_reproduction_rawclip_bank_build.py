import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.rawclip_bank_build import original_structure,parser_arguments,collect

ROOT=Path(__file__).resolve().parents[1]


class OriginalRawConstruction(unittest.TestCase):
    def test_actual_cpu_inspection_proof_is_bound_and_scoped(self):
        folder=ROOT/'reproduction/validation/fresh-bank-20261013/rawclip-bank-build-inspection-v1'
        read=lambda n:json.loads((folder/n).read_bytes())
        proof=read('proof.json')
        for e in read('index.json')['evidence']:
            self.assertEqual(hashlib.sha256((folder/e['file']).read_bytes()).hexdigest(),e['sha256'])
        for key,name in [('code_sha256','code.py'),('plan_sha256','run.json'),('helper_sha256','wrapper.py')]:
            self.assertEqual(proof[key],hashlib.sha256((folder/name).read_bytes()).hexdigest())
        self.assertTrue(proof['actual_CPU_CLI_inspection'] and proof['unallocated_collection_rejected'])
        self.assertFalse(proof['destination_created'] or proof['bank_collection_performed'] or proof['GPU_submitted'])
        self.assertEqual(proof['denial_selfchecks'],3)
        self.assertEqual(proof['blocked_reads_after_selfcheck'],[])
        self.assertEqual(read('static-call-audit.json')['recipe_count'],20)

    def test_recorded_parser_and_original_initialization(self):
        script=ROOT/'research/bank_construction/official_parallel_test_rawclip.py'
        _,initialization,direct,fields,metadata=original_structure(script)
        text=ast.unparse(ast.Module(body=initialization,type_ignores=[]))
        for marker in ['default_image_size', 'create_winclip_raw_backbone', 'resolve_features_list', 'build_transforms']:
            self.assertIn(marker,text)
        for marker in ['get_or_collect', 'calibrator', 'manual_seed']:
            self.assertNotIn(marker,text)
        self.assertNotIn('bank_cache_path',direct)
        self.assertEqual(ast.unparse(metadata.targets[0]),'requested_cache_meta')
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        recipes=json.loads((ROOT/'reproduction/execution-recipes.json').read_bytes())['recipes']
        count=0
        for recipe in recipes:
            if recipe['host']!='RawCLIP':continue
            args=parser_arguments(script,recipe['argv'],torch)
            self.assertEqual(args.source_dataset,recipe['transfer'].split('2',1)[0])
            for name in ['image_size','max_bank_per_layer','max_good_per_class','max_defect_per_class',
                         'max_fp_per_image','max_defect_per_image']:
                self.assertEqual(getattr(args,name),int(recipe['argv'][recipe['argv'].index('--'+name)+1]))
            self.assertEqual(args.device,recipe['argv'][recipe['argv'].index('--device')+1])
            self.assertFalse(hasattr(args,'seed'))
            count+=1
        self.assertEqual(count,20)

    def test_no_allocation_rejects_before_model_import_or_output_creation(self):
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/'new-bank'
            with patch('reproduction.axis_run.require_live_gpu_allocation',side_effect=RuntimeError('No allocation')):
                with self.assertRaisesRegex(RuntimeError,'No allocation'):
                    collect(ROOT/'reproduction',Path(d)/'not-prepared',dest)
            self.assertFalse(dest.exists())
