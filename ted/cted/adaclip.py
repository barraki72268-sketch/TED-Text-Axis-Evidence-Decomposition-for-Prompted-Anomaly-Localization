"""adaclip source-calibration primitives extracted unchanged from the research evaluator.
See docs/CTED.md for host-specific interfaces and release limitations.
"""
from __future__ import annotations
import math
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F

def farthest_point_subsample_indices(feat: torch.Tensor, max_points: int) -> torch.Tensor:
    if feat.shape[0] <= max_points:
        return torch.arange(feat.shape[0], dtype=torch.long)
    feat = F.normalize(feat.float(), dim=-1)
    mean_dir = F.normalize(feat.mean(dim=0, keepdim=True), dim=-1)
    min_dist = 1.0 - (feat @ mean_dir.t()).squeeze(1)
    selected = []
    for _ in range(max_points):
        idx = int(torch.argmax(min_dist).item())
        selected.append(idx)
        candidate_dist = 1.0 - (feat @ feat[idx : idx + 1].t()).squeeze(1)
        min_dist = torch.minimum(min_dist, candidate_dist)
        min_dist[idx] = -1.0
    return torch.tensor(selected, dtype=torch.long)


def logmeanexp_negative_sqdist_1d(coeff: torch.Tensor, bank_coeff: torch.Tensor, tau: float) -> torch.Tensor:
    if bank_coeff.numel() == 0:
        return coeff.new_zeros((coeff.shape[0],))
    dist2 = (coeff.unsqueeze(1) - bank_coeff.unsqueeze(0)) ** 2
    logits = -dist2 / tau
    return torch.logsumexp(logits, dim=-1) - math.log(bank_coeff.shape[0])


def add_to_logit_difference(layer_logits: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    if delta.ndim == 3:
        delta = delta[:, None]
    logit_center = 0.5 * (layer_logits[:, 1:2] + layer_logits[:, 0:1])
    logit_diff = layer_logits[:, 1:2] - layer_logits[:, 0:1]
    refined_diff = logit_diff + delta
    refined_norm = logit_center - 0.5 * refined_diff
    refined_anom = logit_center + 0.5 * refined_diff
    return torch.cat([refined_norm, refined_anom], dim=1)


def inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value)) if value < 20 else value


def sample_source_hardpair_indices(
    fp_scores: torch.Tensor,
    defect_scores: torch.Tensor,
    n: int,
    hardpair_frac: float,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    hardpair_frac = float(np.clip(float(hardpair_frac), 0.0, 1.0))
    n_hard = int(round(n * hardpair_frac))
    n_rand = max(0, n - n_hard)

    def mixed(scores: torch.Tensor, hard_high: bool) -> torch.Tensor:
        scores_cpu = scores.detach().cpu()
        hard_idx = torch.empty(0, dtype=torch.long)
        rand_idx = torch.empty(0, dtype=torch.long)
        if n_hard > 0:
            k = min(n_hard, scores_cpu.numel())
            hard_idx = torch.topk(scores_cpu, k=k, largest=hard_high).indices
        if n_rand > 0:
            perm = torch.randperm(scores_cpu.numel(), generator=rng)
            if hard_idx.numel() > 0:
                mask = torch.ones(scores_cpu.numel(), dtype=torch.bool)
                mask[hard_idx] = False
                perm = perm[mask[perm]]
            rand_idx = perm[: min(n_rand, perm.numel())]
        idx = torch.cat([hard_idx, rand_idx], dim=0)
        if idx.numel() < n:
            fill = torch.randperm(scores_cpu.numel(), generator=rng)[: n - idx.numel()]
            idx = torch.cat([idx, fill], dim=0)
        return idx[:n]

    return mixed(fp_scores, hard_high=True), mixed(defect_scores, hard_high=False)


def _orthonormalize_columns(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    basis = []
    for i in range(mat.shape[1]):
        v = mat[:, i].float()
        for u in basis:
            v = v - torch.dot(v, u) * u
        n = torch.norm(v)
        if float(n.item()) > eps:
            basis.append(v / n)
    if not basis:
        return F.normalize(mat[:, :1].float(), dim=0)
    return torch.stack(basis, dim=1)


def build_defect_parallel_basis(
    fp: torch.Tensor,
    defect: torch.Tensor,
    axis: torch.Tensor,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    fp = F.normalize(fp.float().to(device), dim=-1)
    defect = F.normalize(defect.float().to(device), dim=-1)
    axis = F.normalize(axis.float().to(device), dim=0)
    cols = [axis]
    if fp.numel() > 0 and defect.numel() > 0:
        mu_fp = F.normalize(fp.mean(dim=0), dim=0)
        mu_def = F.normalize(defect.mean(dim=0), dim=0)
        cols.append(F.normalize(mu_def - mu_fp, dim=0))
        n = min(fp.shape[0], defect.shape[0], 4096)
        if n > 4 and rank > 2:
            pair_diff = defect[:n] - fp[:n]
            pair_diff = pair_diff - pair_diff.mean(dim=0, keepdim=True)
            try:
                _, _, vh = torch.linalg.svd(pair_diff.float(), full_matrices=False)
                for pc in vh[: max(0, int(rank) - len(cols))]:
                    cols.append(F.normalize(pc, dim=0))
            except RuntimeError:
                pass
    mat = torch.stack(cols, dim=1)
    return _orthonormalize_columns(mat)[:, : max(1, int(rank))]


def build_random_axis_basis(
    axis: torch.Tensor,
    feature_dim: int,
    rank: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    axis = F.normalize(axis.float().to(device), dim=0)
    cols = [axis]
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    for _ in range(max(0, int(rank) - 1) * 4 + 8):
        v = torch.randn(feature_dim, generator=rng).to(device)
        v = v - torch.dot(v.float(), axis) * axis
        cols.append(F.normalize(v.float(), dim=0))
        basis = _orthonormalize_columns(torch.stack(cols, dim=1))
        if basis.shape[1] >= max(1, int(rank)):
            return basis[:, : max(1, int(rank))]
    return _orthonormalize_columns(torch.stack(cols, dim=1))[:, : max(1, int(rank))]


def apply_pair_label_control_to_tensors(
    fp_items: list[torch.Tensor],
    defect_items: list[torch.Tensor],
    mode: str,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    mode = str(mode)
    if mode == "correct":
        return list(fp_items), list(defect_items)
    if mode == "swap":
        return list(defect_items), list(fp_items)
    if mode == "shuffle":
        n = min(fp_items[0].shape[0], defect_items[0].shape[0])
        if n <= 0:
            return list(fp_items), list(defect_items)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(int(seed))
        perm = torch.randperm(2 * n, generator=rng)
        fp_out: list[torch.Tensor] = []
        def_out: list[torch.Tensor] = []
        for fp_t, def_t in zip(fp_items, defect_items):
            merged = torch.cat([fp_t[:n], def_t[:n]], dim=0)
            fp_out.append(merged[perm[:n]])
            def_out.append(merged[perm[n : 2 * n]])
        return fp_out, def_out
    raise ValueError(f"Unsupported pair label control: {mode}")


def spatial_tanh_zscore_token(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh((x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6))


def prescore_subspace_transport_tokens(
    tokens: torch.Tensor,
    q_def: torch.Tensor,
    q_fp: torch.Tensor,
    basis: torch.Tensor,
    eta: float,
    direction: list[float],
    transport_b: float,
    fp_weight: float,
) -> torch.Tensor:
    basis = basis.to(tokens.device).float()
    direction_t = F.normalize(torch.tensor(direction, device=tokens.device, dtype=torch.float32), dim=0)
    move_dir = F.normalize(basis @ direction_t, dim=0)
    raw_margin = q_def - float(fp_weight) * q_fp
    margin = spatial_tanh_zscore_token(raw_margin)
    delta = float(eta) * torch.tanh(margin + float(transport_b))
    return F.normalize(tokens.float() + delta.unsqueeze(-1) * move_dir.view(1, 1, -1), dim=-1)


def subspace_host_residual_token_score(
    host_score: torch.Tensor,
    rect_tokens: torch.Tensor,
    basis: torch.Tensor,
    score_w: list[float],
    readout_gamma: float,
) -> torch.Tensor:
    basis = basis.to(rect_tokens.device).float()
    score_w_t = F.normalize(torch.tensor(score_w, device=rect_tokens.device, dtype=torch.float32), dim=0)
    coords = rect_tokens.float() @ basis
    if score_w_t.numel() == max(0, coords.shape[-1] - 1):
        residual = coords[..., 1:] @ score_w_t
    else:
        residual = coords @ score_w_t - coords[..., 0]
    return host_score.float() + float(readout_gamma) * residual


def train_subspace_host_residual_calibrator(
    banks: dict,
    axis: torch.Tensor,
    tau: float,
    max_train_points: int,
    epochs: int,
    lr: float,
    rank_margin: float,
    rank_temp: float,
    eta_init: float,
    eta_max: float,
    eta_reg: float,
    fp_weight_init: float,
    learn_fp_weight: bool,
    hardpair_frac: float,
    subspace_rank: int,
    subspace_basis_control: str,
    pair_label_control: str,
    preserve_host_margin_weight: float,
    seed: int,
    device: torch.device,
) -> dict | None:
    if epochs <= 0:
        return None
    fp = F.normalize(banks["fp"].float(), dim=-1)
    defect = F.normalize(banks["defect"].float(), dim=-1)
    if fp.numel() == 0 or defect.numel() == 0:
        return None
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    n = min(fp.shape[0], defect.shape[0], max_train_points // 2 if max_train_points > 0 else 10**9)
    if n <= 0:
        return None

    axis = F.normalize(axis.to(device).float(), dim=0)
    if subspace_basis_control == "source":
        basis = build_defect_parallel_basis(fp, defect, axis, subspace_rank, device)
    elif subspace_basis_control == "random":
        basis = build_random_axis_basis(axis, fp.shape[1], subspace_rank, seed, device)
    else:
        raise ValueError(f"Unsupported subspace basis control: {subspace_basis_control}")
    fp_all = fp.to(device)
    defect_all = defect.to(device)
    fp_proj_all = fp_all @ axis
    defect_proj_all = defect_all @ axis
    fp_host_all = banks.get("fp_score", fp_proj_all.detach().cpu()).float().to(device)
    defect_host_all = banks.get("defect_score", defect_proj_all.detach().cpu()).float().to(device)

    fp_idx, def_idx = sample_source_hardpair_indices(
        fp_scores=fp_host_all,
        defect_scores=defect_host_all,
        n=n,
        hardpair_frac=hardpair_frac,
        rng=rng,
    )
    fp_train = fp[fp_idx].to(device)
    defect_train = defect[def_idx].to(device)
    fp_host = fp_host_all[fp_idx]
    defect_host = defect_host_all[def_idx]
    fp_q_axis = fp_train @ axis
    defect_q_axis = defect_train @ axis
    fp_q_def = logmeanexp_negative_sqdist_1d(fp_q_axis, defect_proj_all, tau=tau)
    fp_q_fp = logmeanexp_negative_sqdist_1d(fp_q_axis, fp_proj_all, tau=tau)
    defect_q_def = logmeanexp_negative_sqdist_1d(defect_q_axis, defect_proj_all, tau=tau)
    defect_q_fp = logmeanexp_negative_sqdist_1d(defect_q_axis, fp_proj_all, tau=tau)
    (
        fp_train,
        fp_host,
        fp_q_def,
        fp_q_fp,
    ), (
        defect_train,
        defect_host,
        defect_q_def,
        defect_q_fp,
    ) = apply_pair_label_control_to_tensors(
        [fp_train, fp_host, fp_q_def, fp_q_fp],
        [defect_train, defect_host, defect_q_def, defect_q_fp],
        pair_label_control,
        seed + 10007,
    )

    eta_max = max(float(eta_max), 1e-6)
    eta_init = float(np.clip(abs(float(eta_init)), 1e-6, 0.99 * eta_max))
    eta_raw = torch.tensor(inverse_softplus(eta_init), device=device, requires_grad=True)
    fp_weight_raw = torch.tensor(inverse_softplus(fp_weight_init), device=device, requires_grad=bool(learn_fp_weight))
    raw_direction = torch.zeros(basis.shape[1], device=device, requires_grad=True)
    raw_direction.data[0] = 1.0
    transport_b = torch.tensor(0.0, device=device, requires_grad=True)
    scorer_dim = max(1, basis.shape[1] - 1)
    scorer = torch.zeros(scorer_dim, device=device, requires_grad=True)
    scorer.data[0] = 1.0
    readout_gamma_raw = torch.tensor(inverse_softplus(eta_init), device=device, requires_grad=True)
    params = [eta_raw, raw_direction, transport_b, scorer, readout_gamma_raw]
    if fp_weight_raw.requires_grad:
        params.append(fp_weight_raw)
    opt = torch.optim.Adam(params, lr=lr)
    pair_index = torch.arange(n, device=device)
    rank_temp = max(float(rank_temp), 1e-6)

    def current_params():
        eta = torch.clamp(F.softplus(eta_raw), max=eta_max)
        readout_gamma = torch.clamp(F.softplus(readout_gamma_raw), max=eta_max)
        fp_weight = F.softplus(fp_weight_raw) + 1e-6
        direction = F.normalize(raw_direction, dim=0)
        score_w = F.normalize(scorer, dim=0)
        return eta, direction, transport_b, score_w, readout_gamma, fp_weight

    def transport_scores(features, host_scores, q_def, q_fp, eta, direction, b, score_w, readout_gamma, fp_weight):
        raw_margin = q_def - fp_weight * q_fp
        margin = torch.tanh((raw_margin - raw_margin.mean()) / raw_margin.std().clamp_min(1e-6))
        delta = eta * torch.tanh(margin + b)
        move_dir = basis @ direction
        rect = F.normalize(features.float() + delta.unsqueeze(-1) * move_dir.view(1, -1), dim=-1)
        coords = rect @ basis
        if basis.shape[1] > 1:
            residual = coords[:, 1:] @ score_w
        else:
            residual = coords[:, :1].squeeze(1)
        scores = host_scores.float() + readout_gamma * residual
        return scores, delta

    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        perm = pair_index[torch.randperm(n, device=device)]
        eta, direction, b, score_w, readout_gamma, fp_weight = current_params()
        features = torch.cat([fp_train[perm], defect_train], dim=0)
        host_scores = torch.cat([fp_host[perm], defect_host], dim=0)
        q_def = torch.cat([fp_q_def[perm], defect_q_def], dim=0)
        q_fp = torch.cat([fp_q_fp[perm], defect_q_fp], dim=0)
        scores, _ = transport_scores(features, host_scores, q_def, q_fp, eta, direction, b, score_w, readout_gamma, fp_weight)
        s_fp, s_def = scores[:n], scores[n:]
        loss = F.softplus((s_fp - s_def + rank_margin) / rank_temp).mean()
        if preserve_host_margin_weight > 0:
            base_margin = defect_host - fp_host[perm]
            final_margin = s_def - s_fp
            easy_weight = torch.sigmoid((base_margin - rank_margin) / rank_temp)
            preserve = F.relu(base_margin - final_margin).mul(easy_weight).sum()
            preserve = preserve / easy_weight.sum().clamp_min(1e-6)
            loss = loss + float(preserve_host_margin_weight) * preserve
        if eta_reg > 0:
            loss = loss + float(eta_reg) * (eta.pow(2) + readout_gamma.pow(2))
        loss.backward()
        opt.step()

    with torch.no_grad():
        eta, direction, b, score_w, readout_gamma, fp_weight = current_params()
        features = torch.cat([fp_train, defect_train], dim=0)
        host_scores = torch.cat([fp_host, defect_host], dim=0)
        q_def = torch.cat([fp_q_def, defect_q_def], dim=0)
        q_fp = torch.cat([fp_q_fp, defect_q_fp], dim=0)
        scores, delta = transport_scores(features, host_scores, q_def, q_fp, eta, direction, b, score_w, readout_gamma, fp_weight)
        s_fp, s_def = scores[:n], scores[n:]
        pair_acc = (s_def > s_fp).float().mean().item()
        base_pair_acc = (defect_host > fp_host).float().mean().item()

    return {
        "calibration": "source_calibrated_host_residual_subspace_rank_no_target",
        "host_branch": "adaclip_multilayer_visual_text_similarity",
        "eta": float(eta.item()),
        "transport_direction": direction.detach().cpu().tolist(),
        "transport_b": float(b.item()),
        "subspace_score_w": score_w.detach().cpu().tolist(),
        "readout_gamma": float(readout_gamma.item()),
        "basis": basis.detach().cpu(),
        "subspace_rank": int(basis.shape[1]),
        "subspace_basis_control": str(subspace_basis_control),
        "pair_label_control": str(pair_label_control),
        "fp_weight": float(fp_weight.item()),
        "train_points_per_class": int(n),
        "source_pair_rank_accuracy": float(pair_acc),
        "source_base_pair_rank_accuracy": float(base_pair_acc),
        "source_delta_fp_mean": float(delta[:n].mean().item()),
        "source_delta_def_mean": float(delta[n:].mean().item()),
        "hardpair_frac": float(hardpair_frac),
        "preserve_host_margin_weight": float(preserve_host_margin_weight),
    }
