import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.fap_bank_build import original_calls, parser_arguments, collect

ROOT = Path(__file__).resolve().parents[1]


class OriginalFapConstruction(unittest.TestCase):
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
