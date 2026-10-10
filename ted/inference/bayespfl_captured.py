"""Captured BayesPFL layer readout; no bank collection or calibration fitting."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from reproduction.checkpoint_download import digest_file


class CapturedBayesPFLReadout:
    def __init__(self, export_directory, device='cpu'):
        if torch.device(device).type == 'cuda':
            from reproduction.axis_run import require_live_gpu_allocation
            require_live_gpu_allocation()
        root = Path(export_directory).resolve()
        read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
        manifest = read('manifest.json')
        if manifest['host'] != 'BayesPFL' or manifest['schema_version'] != 1:
            raise ValueError('Expected BayesPFL captured export')
        for entry in manifest['evidence']:
            if Path(entry['file']).name != entry['file'] or digest_file(root / entry['file']) != entry['sha256']:
                raise ValueError('Captured evidence changed')
        execution = read('execution.json')
        if execution['status'] != 'matched' or execution['returncode'] != 0 or not execution.get('finished') or execution['recipe'] != manifest['recipe']:
            raise ValueError('Expected terminal matching execution')
        if not execution['comparison']['all_match_2dp']:
            raise ValueError('Expected all recorded metrics to match')
        summary = read('summary.json')['summary']
        inventory = read('capture-index.json')
        if digest_file(root / 'capture-index.json') != execution['capture_index_sha256']:
            raise ValueError('Capture inventory changed')
        states = {}
        for entry in manifest['captured_state']:
            binding = {k: entry[k] for k in ('function', 'file', 'sha256')}
            if binding not in inventory or binding not in execution['captured_files']:
                raise ValueError('Capture differs from terminal binding')
            if entry['object_path'] != 'objects/' + entry['sha256'] or digest_file(root / entry['object_path']) != entry['sha256']:
                raise ValueError('Captured object changed')
            if entry['function'] in states:
                raise ValueError('Duplicate captured function')
            states[entry['function']] = torch.load(root / entry['object_path'], map_location='cpu', weights_only=True)
        if set(states) != {'get_or_collect_banks', 'train_source_calibrators'}:
            raise ValueError('Unexpected captured state roles')
        fp, defect, _, _ = states['get_or_collect_banks']
        calibrators = states['train_source_calibrators']
        if {str(k): v for k, v in calibrators.items()} != summary['source_calibrators']:
            raise ValueError('Calibrator metadata differs from recorded summary')
        if set(fp) != set(defect) or set(fp) != set(calibrators):
            raise ValueError('Layer bindings differ')
        for bank in (fp, defect):
            if any(v.ndim != 2 or not torch.isfinite(v).all() for v in bank.values()):
                raise ValueError('Invalid captured bank')
        self.fp, self.defect, self.calibrators = fp, defect, calibrators
        self.args = SimpleNamespace(**{k: summary[k] for k in ('tau', 'fp_weight', 'margin_sign', 'map_activation', 'image_size')})
        self.device = torch.device(device)
        self.manifest = manifest
        self.artifact_sha256 = digest_file(root / 'manifest.json')

    def __call__(self, base_map, outputs):
        args, device = self.args, self.device
        base = torch.as_tensor(np.asarray(base_map, dtype=np.float32), device=device)
        base_scale = base.std().clamp_min(1e-6)
        residual_maps = []
        for layer, out in enumerate(outputs):
            feat, axis = out['dense_feature'].to(device), out['axis'].to(device)
            coeff = feat @ axis
            fp, defect = self.fp[layer].to(device), self.defect[layer].to(device)
            def support(bank):
                bank_coeff = bank @ axis if bank.numel() else torch.empty((0,), device=device)
                if bank_coeff.numel() == 0:
                    return coeff.new_zeros((coeff.shape[0],))
                dist2 = (coeff.unsqueeze(1) - bank_coeff.unsqueeze(0)) ** 2
                return torch.logsumexp(-dist2 / args.tau, dim=-1) - np.log(float(bank_coeff.shape[0]))
            q_fp, q_def = support(fp), support(defect)
            score = q_def - args.fp_weight * q_fp if args.margin_sign == 'def_minus_fp' else q_fp - args.fp_weight * q_def
            if args.map_activation == 'sigmoid':
                score = torch.sigmoid((score - score.mean()) / score.std().clamp_min(1e-6))
            h = int(round(score.shape[0] ** 0.5))
            score_map = F.interpolate(score.reshape(1, 1, h, score.shape[0] // h), size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)[0, 0]
            score_map = (score_map - score_map.mean()) / score_map.std().clamp_min(1e-6)
            residual_maps.append(float(self.calibrators.get(layer, {}).get('eta', 0.0)) * score_map)
        if not residual_maps:
            return np.asarray(base_map, dtype=np.float32)
        residual = torch.stack(residual_maps, dim=0).mean(dim=0)
        return (base + base_scale * residual).detach().float().cpu().numpy().astype(np.float32)
