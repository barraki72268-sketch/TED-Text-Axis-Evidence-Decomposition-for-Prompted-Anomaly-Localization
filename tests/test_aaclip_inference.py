import ast
import math
from pathlib import Path
import unittest
import zipfile

import torch
import torch.nn.functional as F

from ted.inference.aaclip import CapturedAAReadout


class CapturedAAReadoutTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(51)
        axis = F.normalize(torch.randn(8), dim=0)
        banks = {k: [F.normalize(torch.randn(7, 8), dim=-1) for _ in range(2)] for k in ['fp', 'defect']}
        calibrators = [dict(basis=torch.linalg.qr(torch.randn(8, 3)).Q, transport_mode='subspace_tanh',
            eta=.2, transport_direction=[.4, -.1, .5], transport_b=.1, fp_weight=1.2,
            subspace_score_w=[.3, .7], readout_gamma=.15) for _ in range(2)]
        tokens = [F.normalize(torch.randn(2, 16, 8), dim=-1) for _ in range(2)]
        target = F.normalize(torch.randn(8), dim=0)
        return axis, banks, calibrators, tokens, target

    def original_functions(self):
        archive = Path(__file__).resolve().parents[1] / 'reproduction/source.zip'
        with zipfile.ZipFile(archive) as bundle:
            tree = ast.parse(bundle.read('neurips2026/scripts/official_parallel_test_aaclip.py'))
        names = {'spatial_tanh_zscore', 'logmeanexp_negative_sqdist_1d',
                 'prescore_subspace_transport_seg_tokens', 'subspace_host_residual_map_from_tokens'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual(len(nodes), len(names))
        namespace = dict(torch=torch, F=F, math=math, Sequence=list)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<archived-AA-math>', 'exec'), namespace)
        return namespace

    def test_captured_readout_matches_archived_functions_for_both_output_modes(self):
        axis, banks, calibrators, tokens, target = self.fixture()
        original = self.original_functions()
        for mode in ['host_residual', 'host_residual_zscore']:
            with self.subTest(mode=mode):
                expected = []
                for i, (t, cal) in enumerate(zip(tokens, calibrators)):
                    qd = original['logmeanexp_negative_sqdist_1d'](t @ axis, F.normalize(banks['defect'][i], dim=-1) @ axis, .1)
                    qf = original['logmeanexp_negative_sqdist_1d'](t @ axis, F.normalize(banks['fp'][i], dim=-1) @ axis, .1)
                    rect = original['prescore_subspace_transport_seg_tokens'](t, qd, qf, cal['basis'], cal['eta'],
                        cal['transport_direction'], cal['transport_b'], cal['fp_weight'])
                    expected.append(original['subspace_host_residual_map_from_tokens'](t, rect, cal['basis'],
                        cal['subspace_score_w'], target, cal['readout_gamma'], 24, mode))
                runtime = CapturedAAReadout(banks, calibrators, axis, image_size=24, output_mode=mode)
                actual = runtime(tokens, target, .1)
                torch.testing.assert_close(actual, torch.stack(expected, dim=1).sum(1), atol=0, rtol=0)
                self.assertFalse(actual.requires_grad)

    def test_missing_calibrator_or_non_square_tokens_are_rejected(self):
        axis, banks, calibrators, tokens, target = self.fixture()
        with self.assertRaisesRegex(ValueError, 'layer counts'):
            CapturedAAReadout(banks, calibrators[:1], axis, image_size=24, output_mode='host_residual')
        runtime = CapturedAAReadout(banks, calibrators, axis, image_size=24, output_mode='host_residual')
        with self.assertRaisesRegex(ValueError, 'segmentation tokens'):
            runtime([t[:, :15] for t in tokens], target, .1)
