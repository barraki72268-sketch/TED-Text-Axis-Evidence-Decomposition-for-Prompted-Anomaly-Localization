"""Frozen AdaptCLIP density and map equations from the archived evaluator.

No source-bank construction, fitting or dataset loading.
"""
import math
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


# Archived source: neurips2026/scripts/official_parallel_test_adaptclip_vlrefine.py
# SHA256: c06ade64ed14339dce1fcfdfdb168bd0370ab23c6928134274001c1fb9cee762

def spatial_tanh_zscore(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    if x.ndim <= 1:
        return torch.tanh((x - x.mean()) / x.std().clamp_min(1e-6))
    flat = x.reshape(x.shape[0], -1)
    mean = flat.mean(dim=1).view(-1, *([1] * (x.ndim - 1)))
    std = flat.std(dim=1).view(-1, *([1] * (x.ndim - 1))).clamp_min(1e-6)
    return torch.tanh((x - mean) / std)

def logmeanexp_negative_sqdist_nd(query: torch.Tensor, bank: torch.Tensor, tau: float) -> torch.Tensor:
    if bank.numel() == 0:
        return torch.full((query.shape[0],), -20.0, device=query.device, dtype=query.dtype)
    dist = (query[:, None, :] - bank[None, :, :]).pow(2).sum(dim=-1)
    return torch.logsumexp(-dist / max(float(tau), 1e-6), dim=1) - math.log(bank.shape[0])

def calibrated_vl_density_tokens(tokens: torch.Tensor, calibrator: dict) -> torch.Tensor:
    basis = calibrator["basis"].to(tokens.device).float()
    coord_mean = torch.tensor(calibrator["coord_mean"], device=tokens.device, dtype=torch.float32)
    coord_std = torch.tensor(calibrator["coord_std"], device=tokens.device, dtype=torch.float32).clamp_min(1e-6)
    fp_coords = calibrator["fp_coords"].to(tokens.device).float()
    def_coords = calibrator["def_coords"].to(tokens.device).float()
    coords = (tokens.float() @ basis - coord_mean) / coord_std
    q_fp = logmeanexp_negative_sqdist_nd(coords, fp_coords, tau=float(calibrator["rankr_tau"]))
    q_def = logmeanexp_negative_sqdist_nd(coords, def_coords, tau=float(calibrator["rankr_tau"]))
    return q_def - q_fp

def host_response_gate(batch_map: torch.Tensor, quantile: float, temperature: float) -> torch.Tensor:
    flat = batch_map.reshape(batch_map.shape[0], -1)
    threshold = torch.quantile(flat, float(quantile), dim=1).view(-1, 1, 1)
    scale = flat.std(dim=1).view(-1, 1, 1).clamp_min(1e-6)
    return torch.sigmoid((batch_map - threshold) / (scale * max(float(temperature), 1e-6)))

def token_scores_to_map(token_scores: torch.Tensor, h: int, image_size: int, sigma: float, device: str) -> torch.Tensor:
    score_map = token_scores.reshape(1, 1, h, h)
    score_map = F.interpolate(score_map, size=(image_size, image_size), mode="bilinear", align_corners=False)[:, 0]
    score_map = smooth_map(score_map, sigma=sigma).to(device)
    return score_map

# Archived source: neurips2026/scripts/official_parallel_test_adaptclip.py
# SHA256: b9777fc16604ddc1b6625c73048ca0222154417bc78825b2fa3912debdeab4a3

def smooth_map(batch_map: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.stack(
        [torch.from_numpy(gaussian_filter(arr, sigma=sigma)) for arr in batch_map.detach().cpu().numpy()],
        dim=0,
    )

def resize_map(batch_map: torch.Tensor, image_size: int) -> torch.Tensor:
    return F.interpolate(
        batch_map[:, None],
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[:, 0]

# Archived source: neurips2026/AdaptCLIP/adaptcliplib/adaptclip.py
# SHA256: 01541f9104e269e307b593d8029c1fd5219ba02b5c4751e20b3bde7bebd1c797

def average_mean(tensor_list):
    stacked_tensors = torch.stack(tensor_list)  # shape: (N, B, C, H, W)

    a_mean = stacked_tensors.mean(dim=0)  # shape: (B, C, H, W)

    return a_mean
