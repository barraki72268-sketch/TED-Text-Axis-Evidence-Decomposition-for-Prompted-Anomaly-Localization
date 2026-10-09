"""Frozen AA-CLIP readout; no fitting or dataset access during prediction."""
import math

import torch
import torch.nn.functional as F

from ted.cted import aaclip as core


class CapturedAAReadout:
    def __init__(self, banks, calibrators, source_axis, *, image_size, output_mode, device='cpu'):
        self.device = torch.device(device)
        if output_mode not in {'host_residual', 'host_residual_zscore'}:
            raise ValueError('Only captured AA host-residual readouts are supported')
        self.mode, self.image_size = output_mode, int(image_size)
        self.axis = source_axis.detach().float().to(self.device)
        if self.axis.ndim != 1 or not torch.isfinite(self.axis).all() or not torch.isclose(self.axis.norm(), self.axis.new_tensor(1.), atol=1e-4):
            raise ValueError('Invalid source text axis')
        if not calibrators or len(calibrators) != len(banks['fp']) or len(calibrators) != len(banks['defect']):
            raise ValueError('Captured layer counts differ')
        self.fp = [F.normalize(t.detach().float().to(self.device), dim=-1) @ self.axis for t in banks['fp']]
        self.defect = [F.normalize(t.detach().float().to(self.device), dim=-1) @ self.axis for t in banks['defect']]
        self.calibrators = []
        for cal in calibrators:
            if cal is None or cal.get('transport_mode') != 'subspace_tanh':
                raise ValueError('Missing captured source subspace calibrator')
            cal = dict(cal, basis=cal['basis'].detach().float().to(self.device))
            basis = cal['basis']
            if (basis.ndim != 2 or basis.shape[0] != self.axis.numel() or
                    len(cal['transport_direction']) != basis.shape[1] or
                    len(cal['subspace_score_w']) != basis.shape[1] - 1 or
                    not torch.isfinite(basis).all()):
                raise ValueError('Invalid captured source subspace dimensions')
            for key in ['eta', 'transport_b', 'fp_weight', 'readout_gamma']:
                if not math.isfinite(float(cal[key])):
                    raise ValueError('Nonfinite captured readout parameter')
            self.calibrators.append(cal)

    @torch.inference_mode()
    def __call__(self, seg_tokens, target_axis, tau):
        if len(seg_tokens) != len(self.calibrators) or not math.isfinite(tau) or tau <= 0:
            raise ValueError('Wrong number of layers or invalid kernel bandwidth')
        maps = []
        for tokens, fp, defect, cal in zip(seg_tokens, self.fp, self.defect, self.calibrators):
            if (tokens.ndim != 3 or tokens.shape[-1] != self.axis.numel() or
                    tokens.shape[1] < 2 or math.isqrt(tokens.shape[1]) ** 2 != tokens.shape[1] or
                    not torch.isfinite(tokens).all()):
                raise ValueError('Invalid AA segmentation tokens')
            query = tokens @ self.axis
            q_def = core.logmeanexp_negative_sqdist_1d(query, defect, tau)
            q_fp = core.logmeanexp_negative_sqdist_1d(query, fp, tau)
            rectified = core.prescore_subspace_transport_seg_tokens(tokens, q_def, q_fp,
                cal['basis'], cal['eta'], cal['transport_direction'], cal['transport_b'], cal['fp_weight'])
            maps.append(core.subspace_host_residual_map_from_tokens(tokens, rectified, cal['basis'],
                cal['subspace_score_w'], target_axis, cal['readout_gamma'], self.image_size, self.mode))
        return torch.stack(maps, dim=1).sum(1)
