import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.rawclip_bank_build import original_structure,parser_arguments,collect

ROOT=Path(__file__).resolve().parents[1]


class OriginalRawConstruction(unittest.TestCase):
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
