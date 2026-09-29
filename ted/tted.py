"""Feature-level T-TED, corresponding to the paper's support and margin.

No backbone, bank mining, image aggregation, or target calibration is included.
Inputs for one image/layer must already share a host feature coordinate system.
"""
import math

import torch
import torch.nn.functional as F


def _finite_float(value, name, ndim):
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-D tensor")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")


def text_axis(normal, anomaly):
    """Build a unit normal-to-anomaly axis from host text embeddings [D].

Host projection/prompt aggregation must happen before calling this function.
"""
    _finite_float(normal, "normal", 1)
    _finite_float(anomaly, "anomaly", 1)
    if normal.shape != anomaly.shape or normal.numel() == 0:
        raise ValueError("Text embeddings must have the same nonempty shape")
    if normal.device != anomaly.device:
        raise ValueError("Text embeddings must be on the same device")
    if normal.norm() == 0 or anomaly.norm() == 0:
        raise ValueError("Text embeddings must be nonzero")
    direction = F.normalize(anomaly.float(), dim=0) - F.normalize(normal.float(), dim=0)
    if direction.norm() <= 1e-12:
        raise ValueError("Normal and anomaly directions must differ")
    return F.normalize(direction, dim=0)


def logmeanexp_support(coeff, bank_coeff, *, tau, chunk_size=1024):
    """log(mean(exp(-(c-b)^2/tau))) for query [N] and bank [B].

Chunks query points to bound the pairwise allocation by chunk_size * B.
An empty bank contributes zero, matching the research implementation's
ablation fallback; the normal two-bank protocol expects nonempty banks.
"""
    _finite_float(coeff, "coeff", 1)
    _finite_float(bank_coeff, "bank_coeff", 1)
    if coeff.device != bank_coeff.device or coeff.dtype != bank_coeff.dtype:
        raise ValueError("Coefficients must share device and dtype")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if bank_coeff.numel() == 0 or coeff.numel() == 0:
        return torch.zeros_like(coeff)
    outputs = []
    for query in coeff.split(chunk_size):
        dist2 = (query.unsqueeze(1) - bank_coeff.unsqueeze(0)).square()
        outputs.append(torch.logsumexp(-dist2 / tau, dim=-1) - math.log(bank_coeff.numel()))
    return torch.cat(outputs)


def tted_score(patches, defect_bank, hard_fp_bank, axis, *, tau, fp_weight=1.0, chunk_size=1024):
    """Return signed T-TED margins [N] for one image and one layer.

Features/banks have shape [N,D]/[B,D]. They are normalized in float32,
as in the original extraction + scoring pipeline. axis is a unit vector
[D] in the same feature space. Banks are reprojected on every call.
No sigmoid, map resizing, smoothing, or layer aggregation is applied.
tau is deliberately required rather than advertised as a universal default.
"""
    for name, tensor in (("patches", patches), ("defect_bank", defect_bank), ("hard_fp_bank", hard_fp_bank)):
        _finite_float(tensor, name, 2)
    _finite_float(axis, "axis", 1)
    d = patches.shape[1]
    if d == 0 or axis.shape != (d,) or defect_bank.shape[1] != d or hard_fp_bank.shape[1] != d:
        raise ValueError("All features and the axis must share dimension D > 0")
    if axis.device != patches.device:
        raise ValueError("Axis and patches must share a device")
    if not torch.isclose(axis.float().norm(), axis.new_tensor(1.0).float(), atol=1e-5, rtol=1e-5):
        raise ValueError("axis must be unit length")
    if not math.isfinite(fp_weight) or fp_weight < 0:
        raise ValueError("fp_weight must be finite and nonnegative")
    patch = F.normalize(patches.float(), dim=-1)
    direction = axis.float()
    defect = F.normalize(defect_bank.to(device=patch.device, dtype=torch.float32), dim=-1)
    fp = F.normalize(hard_fp_bank.to(device=patch.device, dtype=torch.float32), dim=-1)
    coeff = patch @ direction
    q_def = logmeanexp_support(coeff, defect @ direction, tau=tau, chunk_size=chunk_size)
    q_fp = logmeanexp_support(coeff, fp @ direction, tau=tau, chunk_size=chunk_size)
    return q_def - fp_weight * q_fp
