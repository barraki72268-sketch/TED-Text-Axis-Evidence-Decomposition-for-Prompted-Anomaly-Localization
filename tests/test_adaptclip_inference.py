import ast
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import zipfile

import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from ted.inference.adaptclip import CapturedAdaptDensityReadout


ROOT = Path(__file__).resolve().parents[1]


class CapturedAdaptDensityTests(unittest.TestCase):
    def captured(self, name):
        path = ROOT / 'reproduction/validation/a10-20261009' / (name + '-export.zip')
        with zipfile.ZipFile(path) as bundle:
            manifest = json.loads(bundle.read('manifest.json'))
            summary = json.loads(bundle.read('summary.json'))['summary']
            captures = {}
            for item in manifest['captured_state']:
                raw = bundle.read(item['object_path'])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item['sha256'])
                state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
                observed = {k: v for k, v in state.items() if k not in {'basis', 'fp_coords', 'def_coords'}}
                roles = [role for role in ['vl', 'tl'] if observed == summary['source_' + role + '_calibrator']]
                self.assertEqual(len(roles), 1)
                self.assertNotIn(roles[0], captures)
                captures[roles[0]] = state
        return captures, summary

    def original(self):
        files = {
            'neurips2026/scripts/official_parallel_test_adaptclip_vlrefine.py': {
                'spatial_tanh_zscore', 'logmeanexp_negative_sqdist_nd', 'calibrated_vl_density_tokens',
                'host_response_gate', 'token_scores_to_map', 'calibrated_branch_tokens'},
            'neurips2026/scripts/official_parallel_test_adaptclip.py': {
                'smooth_map', 'resize_map', 'compute_baseline_outputs'},
            'neurips2026/AdaptCLIP/adaptcliplib/adaptclip.py': {'average_mean', 'fusion_fun'},
        }
        namespace = dict(torch=torch, F=F, math=math, gaussian_filter=gaussian_filter)
        with zipfile.ZipFile(ROOT / 'reproduction/source.zip') as bundle:
            for filename, names in files.items():
                tree = ast.parse(bundle.read(filename))
                nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
                self.assertEqual({node.name for node in nodes}, names)
                exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), namespace)
                if filename.endswith('vlrefine.py'):
                    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
                    # Execute the exact archived calibrated-output block. This
                    # also tests the fusion/smoothing/resize and image score,
                    # rather than reproducing those statements in the test.
                    blocks = [node for node in ast.walk(main) if isinstance(node, ast.If)
                              and ast.unparse(node.test) == 'source_vl_calibrator is not None'
                              and isinstance(node.body[0], ast.Assign)
                              and isinstance(node.body[0].targets[0], ast.Name)
                              and node.body[0].targets[0].id == 'calibrator_token']
                    self.assertEqual(len(blocks), 1)
                    block = compile(ast.Module(body=blocks, type_ignores=[]), filename, 'exec')
        return namespace, block

    def test_two_exported_recipes_match_archived_output_block_without_fitting(self):
        torch.manual_seed(37)
        for name in ['adaptclip-openai-btad-20261009', 'adaptclip-l336-btad-20261009']:
            with self.subTest(recipe=name):
                captures, summary = self.captured(name)
                namespace, block = self.original()
                # Keep the recorded 518px output and captured 768-D basis.
                vl = F.normalize(torch.randn(16, 768), dim=-1)
                tl = F.normalize(torch.randn(16, 768), dim=-1)
                raw_vl, raw_tl = torch.rand(1, 9, 9), torch.rand(1, 9, 9)
                global_vl, global_tl = torch.tensor([.7]), torch.tensor([.4])
                model = SimpleNamespace(encode_image=lambda *a, **k: (None, []))
                visual = lambda *a: (torch.stack([1-global_vl, global_vl], -1).log(), torch.stack([1-raw_vl, raw_vl], 1))
                textual = SimpleNamespace(static_text_features=None, compute_global_local_score=lambda *a:
                    (torch.stack([1-global_tl, global_tl], -1).log(), torch.stack([1-raw_tl, raw_tl], 1)))
                baseline = namespace['compute_baseline_outputs'](model, textual, visual, None,
                    torch.zeros(1, 3, 2, 2), [24], 20, summary['image_size'], summary['sigma'], summary['fusion_type'])
                vl_map = namespace['resize_map'](namespace['smooth_map'](raw_vl, summary['sigma']), summary['image_size'])
                tl_map = namespace['resize_map'](namespace['smooth_map'](raw_tl, summary['sigma']), summary['image_size'])
                namespace.update(args=SimpleNamespace(**summary, device='cpu'), vl_patch=vl, tl_patch=tl,
                    vl_axis=None, tl_axis=None, fp_coeff=None, def_coeff=None, tl_fp_coeff=None, tl_def_coeff=None,
                    source_vl_calibrator=captures['vl'], source_tl_calibrator=captures['tl'],
                    h=4, image=torch.zeros(1, 3, 2, 2), baseline=baseline,
                    vl_map=vl_map, tl_map=tl_map, raw_tl_map=raw_tl, outputs_to_append=[])
                exec(block, namespace)
                _, expected_map, expected_score = namespace['outputs_to_append'][0]
                runtime = CapturedAdaptDensityReadout(captures, summary)
                actual = runtime(vl, tl, raw_vl, raw_tl, baseline['global_vl_score'], baseline['global_tl_score'])
                for observed, expected in zip(actual, [baseline['baseline_map'], expected_map, baseline['baseline_img_score'], expected_score]):
                    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
                    self.assertFalse(observed.requires_grad)

    def test_swapped_branches_and_non_square_token_grids_are_rejected(self):
        captures, summary = self.captured('adaptclip-openai-btad-20261009')
        with self.assertRaisesRegex(ValueError, 'branch summary'):
            CapturedAdaptDensityReadout({'vl': captures['tl'], 'tl': captures['vl']}, summary)
        runtime = CapturedAdaptDensityReadout(captures, summary)
        with self.assertRaisesRegex(ValueError, 'branch tokens'):
            runtime(torch.randn(15, 768), torch.randn(15, 768), torch.rand(1, 4, 4),
                    torch.rand(1, 4, 4), torch.rand(1), torch.rand(1))
