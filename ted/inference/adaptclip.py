"""Frozen dual-branch AdaptCLIP density readout for captured passing recipes."""
import math

import torch

from ted.cted import adaptclip as core


class CapturedAdaptDensityReadout:
    def __init__(self, calibrators, summary, *, device='cpu'):
        required = {'ted_insertion': 'dual_branch', 'calib_score_mode': 'rankr_density',
                    'calib_apply_mode': 'add', 'fusion_type': 'average_mean', 'vl_gate_mode': 'host'}
        if any(summary.get(key) != value for key, value in required.items()):
            raise ValueError('Expected captured dual-branch rankr-density add recipe')
        if set(calibrators) != {'vl', 'tl'}:
            raise ValueError('Both captured VL and TL branches are required')
        self.device = torch.device(device)
        self.image_size = int(summary['image_size'])
        self.sigma = float(summary['sigma'])
        self.strength = float(summary['calib_lambda'])
        self.quantile = float(summary['vl_gate_quantile'])
        self.temperature = float(summary['vl_gate_temp'])
        if (self.image_size < 2 or not all(math.isfinite(value) for value in
                [self.sigma, self.strength, self.quantile, self.temperature])
                or self.sigma < 0 or self.strength < 0 or not 0 <= self.quantile <= 1 or self.temperature <= 0):
            raise ValueError('Invalid captured map/gate configuration')
        self.calibrators = {}
        for role, original in calibrators.items():
            if original.get('calibration') != 'adaptclip_vl_source_calibrated_rankr_density_no_target':
                raise ValueError('Wrong captured calibrator type')
            cal = dict(original)
            for key in ['basis', 'fp_coords', 'def_coords']:
                value = cal[key]
                if not isinstance(value, torch.Tensor) or value.ndim != 2 or not value.numel() or not torch.isfinite(value).all():
                    raise ValueError('Invalid captured density tensor')
                cal[key] = value.detach().clone().float().to(self.device)
            dimension, rank = cal['basis'].shape
            if (rank != cal['subspace_rank'] or cal['fp_coords'].shape[1] != rank
                    or cal['def_coords'].shape[1] != rank or len(cal['coord_mean']) != rank
                    or len(cal['coord_std']) != rank or not math.isfinite(float(cal['rankr_tau']))
                    or float(cal['rankr_tau']) <= 0):
                raise ValueError('Captured density dimensions or bandwidth differ')
            if any(not math.isfinite(float(value)) for value in cal['coord_mean'] + cal['coord_std']):
                raise ValueError('Nonfinite captured coordinate parameters')
            # The branch names cannot be guessed from two identical function
            # names: bind each capture to its recorded summary before use.
            observed = {key: value for key, value in original.items() if key not in {'basis', 'fp_coords', 'def_coords'}}
            if observed != summary['source_' + role + '_calibrator']:
                raise ValueError('Captured state does not match its recorded branch summary')
            self.calibrators[role] = cal
        if self.calibrators['vl']['basis'].shape[0] != self.calibrators['tl']['basis'].shape[0]:
            raise ValueError('Captured branch feature dimensions differ')

    @torch.inference_mode()
    def __call__(self, vl_tokens, tl_tokens, raw_vl_map, raw_tl_map, global_vl_score, global_tl_score):
        tokens = {'vl': vl_tokens, 'tl': tl_tokens}
        for role, value in tokens.items():
            if (value.ndim != 2 or value.shape[1] != self.calibrators[role]['basis'].shape[0]
                    or value.shape[0] < 4 or math.isqrt(value.shape[0]) ** 2 != value.shape[0]
                    or not torch.isfinite(value).all() or value.device != self.device):
                raise ValueError('Invalid captured AdaptCLIP branch tokens')
        if vl_tokens.shape != tl_tokens.shape:
            raise ValueError('VL and TL patch grids differ')
        if (raw_vl_map.ndim != 3 or raw_vl_map.shape[0] != 1 or raw_vl_map.shape != raw_tl_map.shape
                or not torch.isfinite(raw_vl_map).all() or not torch.isfinite(raw_tl_map).all()
                or global_vl_score.shape != (1,) or global_tl_score.shape != (1,)
                or not torch.isfinite(global_vl_score).all() or not torch.isfinite(global_tl_score).all()):
            raise ValueError('Expected one finite AdaptCLIP host image')
        baseline_before_resize = core.smooth_map(core.average_mean([raw_vl_map, raw_tl_map]), self.sigma).to(self.device)
        baseline_score = core.average_mean([global_vl_score, global_tl_score, baseline_before_resize.flatten(1).max(1).values])
        baseline = core.resize_map(baseline_before_resize, self.image_size)
        refined = []
        side = math.isqrt(vl_tokens.shape[0])
        for role, host_map in [('vl', raw_vl_map), ('tl', raw_tl_map)]:
            host_map = core.resize_map(core.smooth_map(host_map, self.sigma).to(self.device), self.image_size)
            density = core.calibrated_vl_density_tokens(tokens[role], self.calibrators[role])
            residual = core.spatial_tanh_zscore(core.token_scores_to_map(density, side, self.image_size, self.sigma, str(self.device)))
            gate = core.host_response_gate(host_map, self.quantile, self.temperature)
            refined.append(torch.clamp(host_map + self.strength * gate * residual, 0., 1.))
        corrected = core.resize_map(core.smooth_map(core.average_mean(refined), self.sigma).to(self.device), self.image_size)
        corrected_score = core.average_mean([global_vl_score, global_tl_score, corrected.flatten(1).max(1).values])
        return baseline, corrected, baseline_score, corrected_score
