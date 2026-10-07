"""Source-only artifact export and cached FAPrompt branch-calibrated inference.

This module does not import the host or datasets. A fixed-axis artifact is tied
to one checkpoint/interface; changing either requires a new export.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ted.cted import faprompt as core


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tensor(value, name, ndim):
    if not isinstance(value, torch.Tensor) or value.ndim != ndim or not value.numel():
        raise ValueError(f"{name} must be a nonempty {ndim}-D tensor")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")
    return value.detach().float()


def export_artifact(banks, *, checkpoint_sha256, research_sha256, alpha,
                    seed=0, epochs=30, max_train_points=2048):
    """Fit two source-only branches; never pass target data to this function.

    alpha is explicit because the research evaluator swept several strengths.
    This configuration is an engineering example, not a canonical table recipe.
    """
    if not math.isfinite(alpha) or alpha < 0 or epochs < 1 or max_train_points < 4:
        raise ValueError("Invalid strength or training budget")
    axis = finite_tensor(banks["axis"], "axis", 1)
    if not torch.isclose(axis.norm(), torch.tensor(1.), atol=1e-4):
        raise ValueError("Source axis must be unit normalized")
    for role in ("fp", "defect"):
        features = finite_tensor(banks[role], role, 2)
        if features.shape[1] != len(axis):
            raise ValueError("Bank and axis dimensions differ")
        for branch in (1, 2):
            scores = finite_tensor(banks[f"{role}_branch{branch}_score"], "host scores", 1)
            if len(scores) != len(features):
                raise ValueError("Bank features and host scores must remain aligned")
    required = ("model_name", "prompt_load_mode", "features_list", "image_size",
                "dap_token_mode", "dpam_layer", "protocol_version")
    interface = {key: banks[key] for key in required}
    recipe = dict(tau=0.1, bank_chunk=2048, max_train_points=max_train_points,
                  epochs=epochs, lr=0.02, rank_margin=0.05, rank_temp=0.05,
                  eta_init=0.05, eta_max=0.2, eta_reg=0.001,
                  fp_weight_init=1.0, learn_fp_weight=False, hardpair_frac=0.75,
                  subspace_rank=4, subspace_basis_control="source",
                  pair_label_control="correct", preserve_host_margin_weight=1.0)
    calibrators = {}
    for branch, offset in (("branch1", 101), ("branch2", 202)):
        torch.manual_seed(seed + offset)
        cal = core.train_subspace_host_residual_calibrator(
            banks=banks, axis=axis, seed=seed + offset, device=torch.device("cpu"),
            host_score_key=f"{branch}_score", **recipe)
        if cal is None:
            raise ValueError("Source bank cannot produce a calibrator")
        calibrators[branch] = cal
    return dict(schema_version=1, host="faprompt", checkpoint_sha256=checkpoint_sha256,
                research_sha256=research_sha256, interface=interface, axis=axis.cpu(),
                fp_projection=(banks["fp"].float() @ axis).cpu(),
                defect_projection=(banks["defect"].float() @ axis).cpu(),
                calibrators=calibrators, recipe=recipe, seed=seed,
                inference=dict(alpha=float(alpha), gate_quantile=0.9, gate_temp=0.25,
                               sigma=10, depth=9, n_ctx=12, t_n_ctx=4))


class BranchCalibratedReadout:
    """Read-only inference state; no bank mining, fitting, or target labels."""

    def __init__(self, artifact, device="cpu", bank_chunk=2048):
        if artifact.get("schema_version") != 1 or artifact.get("host") != "faprompt":
            raise ValueError("Unsupported artifact schema or host")
        if bank_chunk < 1:
            raise ValueError("bank_chunk must be positive")
        self.device = torch.device(device)
        self.axis = finite_tensor(artifact["axis"], "axis", 1).to(self.device)
        self.fp = finite_tensor(artifact["fp_projection"], "fp_projection", 1).to(self.device)
        self.defect = finite_tensor(artifact["defect_projection"], "defect_projection", 1).to(self.device)
        self.tau = float(artifact["recipe"]["tau"])
        self.options = dict(artifact["inference"])
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("tau must be positive and finite")
        for key in ("alpha", "gate_quantile", "gate_temp"):
            if not math.isfinite(self.options[key]):
                raise ValueError(f"Nonfinite {key}")
        if self.options["alpha"] < 0 or not 0 <= self.options["gate_quantile"] < 1 or self.options["gate_temp"] <= 0:
            raise ValueError("Invalid residual strength or gate settings")
        self.bank_chunk = bank_chunk
        self.calibrators = {}
        for name in ("branch1", "branch2"):
            cal = dict(artifact["calibrators"][name])
            basis = finite_tensor(cal["basis"], "basis", 2).to(self.device)
            direction = torch.as_tensor(cal["transport_direction"], device=self.device, dtype=torch.float32)
            weights = torch.as_tensor(cal["subspace_score_w"], device=self.device, dtype=torch.float32)
            if basis.shape[0] != len(self.axis) or direction.shape != (basis.shape[1],) or weights.shape != (basis.shape[1]-1,):
                raise ValueError("Expected a host-residual source subspace of rank >= 2")
            if weights.numel() == 0 or not torch.isfinite(direction).all() or not torch.isfinite(weights).all():
                raise ValueError("Invalid residual parameters")
            for key in ("eta", "transport_b", "fp_weight", "readout_gamma"):
                if not math.isfinite(float(cal[key])):
                    raise ValueError(f"Invalid {key}")
            cal.update(basis=basis, move=F.normalize(basis @ F.normalize(direction, dim=0), dim=0),
                       weights=F.normalize(weights, dim=0))
            self.calibrators[name] = cal

    @torch.inference_mode()
    def __call__(self, tokens, baseline, branch1, branch2):
        tokens = finite_tensor(tokens, "tokens", 3).to(self.device)
        if tokens.shape[-1] != len(self.axis) or tokens.shape[1] < 2:
            raise ValueError("Token dimension mismatch or fewer than two tokens")
        scores = []
        for name, value in (("baseline", baseline), ("branch1", branch1), ("branch2", branch2)):
            value = finite_tensor(value, name, 2).to(self.device)
            if value.shape != tokens.shape[:2]:
                raise ValueError("Token and host-score shapes differ")
            scores.append(value)
        baseline, branch1, branch2 = scores
        query = tokens @ self.axis
        q_def = core.logmeanexp_negative_sqdist_1d(query, self.defect, self.tau, self.bank_chunk)
        q_fp = core.logmeanexp_negative_sqdist_1d(query, self.fp, self.tau, self.bank_chunk)
        residuals = []
        for name, anchor in (("branch1", branch1), ("branch2", branch2)):
            cal = self.calibrators[name]
            margin = core.spatial_tanh_zscore(q_def - cal["fp_weight"] * q_fp)
            delta = cal["eta"] * torch.tanh(margin + cal["transport_b"])
            rect = F.normalize(tokens + delta.unsqueeze(-1) * cal["move"], dim=-1)
            corrected = anchor + cal["readout_gamma"] * ((rect @ cal["basis"])[..., 1:] @ cal["weights"])
            residuals.append(corrected - anchor)
        z = core.spatial_tanh_zscore(baseline)
        k = max(1, math.ceil(z.shape[1] * max(1e-6, 1.0 - self.options["gate_quantile"])))
        threshold = torch.topk(z, k=k, dim=1).values[:, -1:]
        gate = torch.sigmoid((z - threshold) / z.std(dim=1, keepdim=True).clamp_min(1e-6)
                             / max(self.options["gate_temp"], 1e-6))
        return baseline + self.options["alpha"] * gate * 0.5 * (residuals[0] + residuals[1])
