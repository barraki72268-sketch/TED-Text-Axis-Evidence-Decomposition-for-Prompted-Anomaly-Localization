from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/mnt/data/pilab-kingjinyoung/ted-reproduce-20261008/snapshot")
SCRIPT_ROOT = ROOT / "neurips2026" / "scripts"
ADAPT_ROOT = ROOT / "neurips2026" / "AdaptCLIP"
if str(ADAPT_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(1, str(SCRIPT_ROOT))

from official_parallel_test_adaptclip import (  # noqa: E402
    Evaluator,
    activate_margin_map,
    aggregate_results,
    build_dataset,
    build_model,
    calculate_fast_aupro,
    compute_baseline_outputs,
    decision_axis,
    fusion_fun,
    logmeanexp_negative_sqdist_1d,
    map_group_separability,
    precompute_aupro_regions,
    prepare_aupro_arrays,
    resize_map,
    save_results,
    smooth_map,
)
from analyze_adaptclip_vl_score_forms import (  # noqa: E402
    branch_patch_bank_features,
    get_or_collect_vl_banks,
    vl_patch_bank_features,
)


def jsonable(obj):
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.detach().cpu().item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value)) if value < 20 else value


def spatial_tanh_zscore(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    if x.ndim <= 1:
        return torch.tanh((x - x.mean()) / x.std().clamp_min(1e-6))
    flat = x.reshape(x.shape[0], -1)
    mean = flat.mean(dim=1).view(-1, *([1] * (x.ndim - 1)))
    std = flat.std(dim=1).view(-1, *([1] * (x.ndim - 1))).clamp_min(1e-6)
    return torch.tanh((x - mean) / std)


def build_scalar_plugin_features(query_proj: torch.Tensor, q_def: torch.Tensor, q_fp: torch.Tensor) -> torch.Tensor:
    return torch.stack([query_proj, q_def, q_fp, q_def - q_fp, q_def + q_fp], dim=-1)


def apply_scalar_plugin(query_proj: torch.Tensor, q_def: torch.Tensor, q_fp: torch.Tensor, plugin: dict) -> torch.Tensor:
    feats = build_scalar_plugin_features(query_proj, q_def, q_fp)
    mean = plugin["mean"].to(query_proj.device)
    std = plugin["std"].to(query_proj.device)
    x = ((feats - mean) / std).float()
    model_type = plugin.get("model_type", "linear")
    if model_type == "mlp":
        hidden = F.relu(x @ plugin["w1"].to(query_proj.device) + plugin["b1"].to(query_proj.device))
        return hidden @ plugin["w2"].to(query_proj.device) + plugin["b2"].to(query_proj.device)
    if model_type == "support_affine":
        defect_weight = float(plugin["defect_weight"])
        fp_weight = float(plugin["fp_weight"])
        query_weight = float(plugin.get("query_weight", 0.0))
        bias = float(plugin["bias"])
        return query_weight * query_proj.float() + defect_weight * q_def.float() - fp_weight * q_fp.float() + bias
    return x @ plugin["weight"].to(query_proj.device) + plugin["bias"].to(query_proj.device)


def apply_hostscore_plugin(host_score: torch.Tensor, plugin: dict) -> torch.Tensor:
    x = ((host_score.float().unsqueeze(-1) - plugin["mean"].to(host_score.device)) / plugin["std"].to(host_score.device)).float()
    if plugin.get("model_type") == "host_mlp":
        hidden = F.relu(x @ plugin["w1"].to(host_score.device) + plugin["b1"].to(host_score.device))
        return hidden @ plugin["w2"].to(host_score.device) + plugin["b2"].to(host_score.device)
    return x.squeeze(-1) * plugin["weight"].to(host_score.device) + plugin["bias"].to(host_score.device)


def train_hostscore_plugin_from_scores(
    fp_scores: torch.Tensor,
    def_scores: torch.Tensor,
    max_train_points: int,
    epochs: int,
    lr: float,
    hardpair_frac: float,
    loss_type: str,
    hidden_dim: int,
    rank_margin: float,
    rank_temp: float,
    seed: int,
    device: torch.device,
) -> dict | None:
    fp_all = fp_scores.detach().float().flatten()
    def_all = def_scores.detach().float().flatten()
    if epochs <= 0 or fp_all.numel() == 0 or def_all.numel() == 0:
        return None
    n = min(int(max_train_points), int(fp_all.numel()), int(def_all.numel()))
    if n <= 0:
        return None
    hardpair_frac = float(np.clip(float(hardpair_frac), 0.0, 1.0))
    n_hard = int(round(n * hardpair_frac))
    n_rand = max(0, n - n_hard)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    def mixed_indices(scores: torch.Tensor, hard_high: bool) -> torch.Tensor:
        scores_cpu = scores.detach().cpu()
        hard_idx = torch.empty(0, dtype=torch.long)
        rand_idx = torch.empty(0, dtype=torch.long)
        if n_hard > 0:
            hard_idx = torch.topk(scores_cpu, k=min(n_hard, scores_cpu.numel()), largest=hard_high).indices
        if n_rand > 0:
            perm = torch.randperm(scores_cpu.numel(), generator=rng)
            if hard_idx.numel() > 0:
                mask = torch.ones(scores_cpu.numel(), dtype=torch.bool)
                mask[hard_idx] = False
                perm = perm[mask[perm]]
            rand_idx = perm[: min(n_rand, perm.numel())]
        idx = torch.cat([hard_idx, rand_idx], dim=0)
        if idx.numel() < n:
            idx = torch.cat([idx, torch.randperm(scores_cpu.numel(), generator=rng)[: n - idx.numel()]], dim=0)
        return idx[:n]

    fp = fp_all[mixed_indices(fp_all, hard_high=True)].to(device).view(-1, 1)
    defect = def_all[mixed_indices(def_all, hard_high=False)].to(device).view(-1, 1)
    x = torch.cat([fp, defect], dim=0).float()
    y = torch.cat([torch.zeros(fp.shape[0], device=device), torch.ones(defect.shape[0], device=device)], dim=0)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    xz = (x - mean) / std
    loss_type = str(loss_type)
    if loss_type == "pair_rank":
        hidden_dim = max(1, int(hidden_dim))
        w1 = torch.empty((1, hidden_dim), device=device, requires_grad=True)
        b1 = torch.zeros((hidden_dim,), device=device, requires_grad=True)
        w2 = torch.empty((hidden_dim,), device=device, requires_grad=True)
        b2 = torch.zeros((), device=device, requires_grad=True)
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(w1)
            torch.nn.init.zeros_(b1)
            torch.nn.init.normal_(w2, std=0.02)
        opt = torch.optim.Adam([w1, b1, w2, b2], lr=lr)

        def mlp(feats: torch.Tensor) -> torch.Tensor:
            z = (feats.float() - mean) / std
            return F.relu(z @ w1 + b1) @ w2 + b2

        for _ in range(int(epochs)):
            opt.zero_grad(set_to_none=True)
            logit_fp = mlp(fp)
            logit_def = mlp(defect)
            loss = F.softplus(-(logit_def - logit_fp - float(rank_margin)) / max(float(rank_temp), 1e-6)).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            logit = mlp(x)
            rank_acc = (mlp(defect) > mlp(fp)).float().mean().item()
            acc = ((logit.sigmoid() >= 0.5).float() == y).float().mean().item()
        return {
            "model_type": "host_mlp",
            "feature_space": "hostscore",
            "w1": w1.detach(),
            "b1": b1.detach(),
            "w2": w2.detach(),
            "b2": b2.detach(),
            "mean": mean.detach(),
            "std": std.detach(),
            "train_points_per_class": int(n),
            "train_accuracy": float(acc),
            "source_pair_rank_accuracy": float(rank_acc),
            "loss_type": loss_type,
        }

    weight = torch.zeros((), device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.Adam([weight, bias], lr=lr)
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logit = xz.squeeze(-1) * weight + bias
        if loss_type == "support_rank":
            loss = F.softplus(-((xz[fp.shape[0] :, 0] * weight + bias) - (xz[: fp.shape[0], 0] * weight + bias) - float(rank_margin)) / max(float(rank_temp), 1e-6)).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(logit, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logit = xz.squeeze(-1) * weight + bias
        rank_acc = ((defect - mean).squeeze(-1) / std.squeeze(-1) * weight + bias > (fp - mean).squeeze(-1) / std.squeeze(-1) * weight + bias).float().mean().item()
        acc = ((logit.sigmoid() >= 0.5).float() == y).float().mean().item()
    return {
        "model_type": "host_linear",
        "feature_space": "hostscore",
        "weight": weight.detach(),
        "bias": bias.detach(),
        "mean": mean.detach(),
        "std": std.detach(),
        "train_points_per_class": int(n),
        "train_accuracy": float(acc),
        "source_pair_rank_accuracy": float(rank_acc),
        "loss_type": loss_type,
    }


def train_scalar_plugin_for_bank(
    fp_bank: torch.Tensor,
    def_bank: torch.Tensor,
    axis: torch.Tensor,
    tau: float,
    max_train_points: int,
    epochs: int,
    lr: float,
    hardpair_frac: float,
    loss_type: str,
    hidden_dim: int,
    rank_margin: float,
    rank_temp: float,
    seed: int,
    device: torch.device,
) -> dict | None:
    if epochs <= 0 or fp_bank.numel() == 0 or def_bank.numel() == 0:
        return None
    fp = F.normalize(fp_bank.detach().float(), dim=-1)
    defect = F.normalize(def_bank.detach().float(), dim=-1)
    n = min(fp.shape[0], defect.shape[0], max_train_points // 2 if max_train_points > 0 else 10**9)
    if n <= 0:
        return None
    axis = F.normalize(axis.detach().to(device).float(), dim=0)
    fp_all = fp.to(device)
    def_all = defect.to(device)
    fp_proj_all = fp_all @ axis
    def_proj_all = def_all @ axis
    hardpair_frac = float(np.clip(float(hardpair_frac), 0.0, 1.0))
    n_hard = int(round(n * hardpair_frac))
    n_rand = max(0, n - n_hard)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    def mixed_indices(scores: torch.Tensor, hard_high: bool) -> torch.Tensor:
        scores_cpu = scores.detach().cpu()
        hard_idx = torch.empty(0, dtype=torch.long)
        rand_idx = torch.empty(0, dtype=torch.long)
        if n_hard > 0:
            hard_idx = torch.topk(scores_cpu, k=min(n_hard, scores_cpu.numel()), largest=hard_high).indices
        if n_rand > 0:
            perm = torch.randperm(scores_cpu.numel(), generator=rng)
            if hard_idx.numel() > 0:
                mask = torch.ones(scores_cpu.numel(), dtype=torch.bool)
                mask[hard_idx] = False
                perm = perm[mask[perm]]
            rand_idx = perm[: min(n_rand, perm.numel())]
        idx = torch.cat([hard_idx, rand_idx], dim=0)
        if idx.numel() < n:
            idx = torch.cat([idx, torch.randperm(scores_cpu.numel(), generator=rng)[: n - idx.numel()]], dim=0)
        return idx[:n]

    fp_train = fp[mixed_indices(fp_proj_all, hard_high=True)].to(device)
    def_train = defect[mixed_indices(def_proj_all, hard_high=False)].to(device)
    fp_q = fp_train @ axis
    def_q = def_train @ axis
    x_fp = build_scalar_plugin_features(
        fp_q,
        logmeanexp_negative_sqdist_1d(fp_q, def_proj_all, tau=tau),
        logmeanexp_negative_sqdist_1d(fp_q, fp_proj_all, tau=tau),
    )
    x_def = build_scalar_plugin_features(
        def_q,
        logmeanexp_negative_sqdist_1d(def_q, def_proj_all, tau=tau),
        logmeanexp_negative_sqdist_1d(def_q, fp_proj_all, tau=tau),
    )
    x = torch.cat([x_fp, x_def], dim=0).float()
    y = torch.cat([torch.zeros(x_fp.shape[0], device=device), torch.ones(x_def.shape[0], device=device)], dim=0)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    xz = (x - mean) / std
    loss_type = str(loss_type)
    if loss_type == "support_rank":
        defect_raw = torch.zeros((), device=device, requires_grad=True)
        fp_raw = torch.zeros((), device=device, requires_grad=True)
        query_weight = torch.zeros((), device=device, requires_grad=True)
        bias = torch.zeros((), device=device, requires_grad=True)
        opt = torch.optim.Adam([defect_raw, fp_raw, query_weight, bias], lr=lr)

        def score_from_feats(feats: torch.Tensor) -> torch.Tensor:
            defect_weight = F.softplus(defect_raw)
            fp_weight = F.softplus(fp_raw)
            return query_weight * feats[..., 0] + defect_weight * feats[..., 1] - fp_weight * feats[..., 2] + bias

        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            logit_fp = score_from_feats(x_fp.float())
            logit_def = score_from_feats(x_def.float())
            loss = F.softplus(-(logit_def - logit_fp - float(rank_margin)) / max(float(rank_temp), 1e-6)).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            logit = score_from_feats(x)
            logit_fp = score_from_feats(x_fp.float())
            logit_def = score_from_feats(x_def.float())
            rank_acc = (logit_def > logit_fp).float().mean().item()
            acc = ((logit.sigmoid() >= 0.5).float() == y).float().mean().item()
        return {
            "model_type": "support_affine",
            "defect_weight": float(F.softplus(defect_raw).item()),
            "fp_weight": float(F.softplus(fp_raw).item()),
            "query_weight": float(query_weight.item()),
            "bias": float(bias.item()),
            "mean": mean.detach(),
            "std": std.detach(),
            "train_points_per_class": int(n),
            "train_accuracy": float(acc),
            "source_pair_rank_accuracy": float(rank_acc),
            "loss_type": loss_type,
        }
    if loss_type == "pair_rank":
        hidden_dim = max(1, int(hidden_dim))
        w1 = torch.empty((xz.shape[1], hidden_dim), device=device, requires_grad=True)
        b1 = torch.zeros((hidden_dim,), device=device, requires_grad=True)
        w2 = torch.empty((hidden_dim,), device=device, requires_grad=True)
        b2 = torch.zeros((), device=device, requires_grad=True)
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(w1)
            torch.nn.init.zeros_(b1)
            torch.nn.init.normal_(w2, std=0.02)
        opt = torch.optim.Adam([w1, b1, w2, b2], lr=lr)

        def mlp(feats: torch.Tensor) -> torch.Tensor:
            z = (feats.float() - mean) / std
            return F.relu(z @ w1 + b1) @ w2 + b2

        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            logit_fp = mlp(x_fp)
            logit_def = mlp(x_def)
            loss = F.softplus(-(logit_def - logit_fp - float(rank_margin)) / max(float(rank_temp), 1e-6)).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            logit = mlp(x)
            logit_fp = mlp(x_fp)
            logit_def = mlp(x_def)
            rank_acc = (logit_def > logit_fp).float().mean().item()
            acc = ((logit.sigmoid() >= 0.5).float() == y).float().mean().item()
        return {
            "model_type": "mlp",
            "w1": w1.detach(),
            "b1": b1.detach(),
            "w2": w2.detach(),
            "b2": b2.detach(),
            "mean": mean.detach(),
            "std": std.detach(),
            "train_points_per_class": int(n),
            "train_accuracy": float(acc),
            "source_pair_rank_accuracy": float(rank_acc),
            "loss_type": loss_type,
        }
    if loss_type != "bce":
        raise ValueError(f"Unsupported plugin_loss={loss_type}")
    weight = torch.zeros(xz.shape[1], device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.Adam([weight, bias], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(xz @ weight + bias, y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logit = xz @ weight + bias
        xz_fp = (x_fp.float() - mean) / std
        xz_def = (x_def.float() - mean) / std
        rank_acc = ((xz_def @ weight + bias) > (xz_fp @ weight + bias)).float().mean().item()
        acc = ((logit.sigmoid() >= 0.5).float() == y).float().mean().item()
    return {
        "model_type": "linear",
        "weight": weight.detach(),
        "bias": bias.detach(),
        "mean": mean.detach(),
        "std": std.detach(),
        "train_points_per_class": int(n),
        "train_accuracy": float(acc),
        "source_pair_rank_accuracy": float(rank_acc),
        "loss_type": loss_type,
    }


def sample_source_hardpair_indices(
    fp_scores: torch.Tensor,
    defect_scores: torch.Tensor,
    n: int,
    hardpair_frac: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hardpair_frac = float(np.clip(float(hardpair_frac), 0.0, 1.0))
    n_hard = int(round(n * hardpair_frac))
    n_rand = max(0, n - n_hard)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

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


def apply_pair_label_control_to_tensors(
    fp_tensors: list[torch.Tensor],
    defect_tensors: list[torch.Tensor],
    mode: str,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Corrupt source pair labels while keeping capacity and budget fixed."""
    if mode == "correct":
        return fp_tensors, defect_tensors
    if mode == "swap":
        return defect_tensors, fp_tensors
    if mode == "shuffle":
        if not fp_tensors or not defect_tensors:
            return fp_tensors, defect_tensors
        n = min(fp_tensors[0].shape[0], defect_tensors[0].shape[0])
        if n <= 1:
            return fp_tensors, defect_tensors
        rng = torch.Generator(device="cpu")
        rng.manual_seed(int(seed))
        swap_mask = torch.rand(n, generator=rng) < 0.5
        new_fp: list[torch.Tensor] = []
        new_def: list[torch.Tensor] = []
        for fp, defect in zip(fp_tensors, defect_tensors):
            fp_out = fp.clone()
            def_out = defect.clone()
            fp_out[swap_mask] = defect[swap_mask]
            def_out[swap_mask] = fp[swap_mask]
            new_fp.append(fp_out)
            new_def.append(def_out)
        return new_fp, new_def
    raise ValueError(f"Unknown pair label control: {mode}")


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


def train_vl_subspace_calibrator(
    fp_bank: torch.Tensor,
    def_bank: torch.Tensor,
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
    preserve_host_margin_weight: float,
    pair_label_control: str,
    seed: int,
    device: torch.device,
) -> dict | None:
    if epochs <= 0 or fp_bank.numel() == 0 or def_bank.numel() == 0:
        return None
    fp = F.normalize(fp_bank.float(), dim=-1)
    defect = F.normalize(def_bank.float(), dim=-1)
    n = min(fp.shape[0], defect.shape[0], max_train_points // 2 if max_train_points > 0 else 10**9)
    if n <= 0:
        return None

    axis = F.normalize(axis.to(device).float(), dim=0)
    basis = build_defect_parallel_basis(fp, defect, axis, subspace_rank, device)
    fp_all = fp.to(device)
    defect_all = defect.to(device)
    fp_proj_all = fp_all @ axis
    defect_proj_all = defect_all @ axis
    fp_host_all = fp_proj_all
    defect_host_all = defect_proj_all

    fp_idx, def_idx = sample_source_hardpair_indices(
        fp_scores=fp_host_all,
        defect_scores=defect_host_all,
        n=n,
        hardpair_frac=hardpair_frac,
        seed=seed,
    )
    fp_train = fp[fp_idx].to(device)
    defect_train = defect[def_idx].to(device)
    fp_host = fp_host_all[fp_idx]
    defect_host = defect_host_all[def_idx]
    fp_axis = fp_train @ axis
    defect_axis = defect_train @ axis
    fp_q_def = logmeanexp_negative_sqdist_1d(fp_axis, defect_proj_all, tau=tau)
    fp_q_fp = logmeanexp_negative_sqdist_1d(fp_axis, fp_proj_all, tau=tau)
    defect_q_def = logmeanexp_negative_sqdist_1d(defect_axis, defect_proj_all, tau=tau)
    defect_q_fp = logmeanexp_negative_sqdist_1d(defect_axis, fp_proj_all, tau=tau)
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
    fp_train = fp_train.detach()
    defect_train = defect_train.detach()
    fp_host = fp_host.detach()
    defect_host = defect_host.detach()
    fp_q_def = fp_q_def.detach()
    fp_q_fp = fp_q_fp.detach()
    defect_q_def = defect_q_def.detach()
    defect_q_fp = defect_q_fp.detach()
    basis = basis.detach()

    eta_max = max(float(eta_max), 1e-6)
    eta_init = float(np.clip(abs(float(eta_init)), 1e-6, 0.99 * eta_max))
    eta_raw = torch.tensor(inverse_softplus(eta_init), device=device, requires_grad=True)
    gamma_raw = torch.tensor(inverse_softplus(eta_init), device=device, requires_grad=True)
    fp_weight_raw = torch.tensor(inverse_softplus(fp_weight_init), device=device, requires_grad=bool(learn_fp_weight))
    raw_direction = torch.zeros(basis.shape[1], device=device, requires_grad=True)
    raw_direction.data[0] = 1.0
    transport_b = torch.tensor(0.0, device=device, requires_grad=True)
    scorer_dim = max(1, basis.shape[1] - 1)
    scorer = torch.zeros(scorer_dim, device=device, requires_grad=True)
    scorer.data[0] = 1.0
    params = [eta_raw, gamma_raw, raw_direction, transport_b, scorer]
    if fp_weight_raw.requires_grad:
        params.append(fp_weight_raw)
    opt = torch.optim.Adam(params, lr=lr)
    pair_index = torch.arange(n, device=device)
    rank_temp = max(float(rank_temp), 1e-6)

    def current_params():
        eta = torch.clamp(F.softplus(eta_raw), max=eta_max)
        gamma = torch.clamp(F.softplus(gamma_raw), max=eta_max)
        fp_weight = F.softplus(fp_weight_raw) + 1e-6
        direction = F.normalize(raw_direction, dim=0)
        score_w = F.normalize(scorer, dim=0)
        return eta, gamma, fp_weight, direction, score_w

    def score_features(features, host_scores, q_def, q_fp, eta, gamma, fp_weight, direction, score_w):
        raw_margin = q_def - fp_weight * q_fp
        margin = spatial_tanh_zscore(raw_margin)
        move_dir = F.normalize(basis @ direction, dim=0)
        delta = eta * torch.tanh(margin + transport_b)
        rect = F.normalize(features.float() + delta.unsqueeze(-1) * move_dir.view(1, -1), dim=-1)
        coords = rect @ basis
        if basis.shape[1] > 1:
            residual = coords[:, 1:] @ score_w
        else:
            residual = coords[:, 0]
        return host_scores.float() + gamma * residual, delta

    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        perm = pair_index[torch.randperm(n, device=device)]
        eta, gamma, fp_weight, direction, score_w = current_params()
        features = torch.cat([fp_train[perm], defect_train], dim=0)
        host_scores = torch.cat([fp_host[perm], defect_host], dim=0)
        q_def = torch.cat([fp_q_def[perm], defect_q_def], dim=0)
        q_fp = torch.cat([fp_q_fp[perm], defect_q_fp], dim=0)
        scores, _ = score_features(features, host_scores, q_def, q_fp, eta, gamma, fp_weight, direction, score_w)
        s_fp, s_def = scores[:n], scores[n:]
        loss = F.softplus((s_fp - s_def + rank_margin) / rank_temp).mean()
        if preserve_host_margin_weight > 0:
            base_margin = defect_host - fp_host[perm]
            final_margin = s_def - s_fp
            easy_weight = torch.sigmoid((base_margin - rank_margin) / rank_temp)
            preserve = F.relu(base_margin - final_margin).mul(easy_weight).sum()
            loss = loss + float(preserve_host_margin_weight) * preserve / easy_weight.sum().clamp_min(1e-6)
        if eta_reg > 0:
            loss = loss + float(eta_reg) * (eta.pow(2) + gamma.pow(2))
        loss.backward()
        opt.step()

    with torch.no_grad():
        eta, gamma, fp_weight, direction, score_w = current_params()
        features = torch.cat([fp_train, defect_train], dim=0)
        host_scores = torch.cat([fp_host, defect_host], dim=0)
        q_def = torch.cat([fp_q_def, defect_q_def], dim=0)
        q_fp = torch.cat([fp_q_fp, defect_q_fp], dim=0)
        scores, delta = score_features(features, host_scores, q_def, q_fp, eta, gamma, fp_weight, direction, score_w)
        s_fp, s_def = scores[:n], scores[n:]
        pair_acc = (s_def > s_fp).float().mean().item()
        base_pair_acc = (defect_host > fp_host).float().mean().item()

    return {
        "calibration": "adaptclip_vl_source_calibrated_residual_subspace_no_target",
        "host_branch": "visual_local_adapter",
        "host_score_source": "static_text_axis_projection",
        "basis": basis.detach().cpu(),
        "subspace_rank": int(basis.shape[1]),
        "eta": float(eta.item()),
        "gamma": float(gamma.item()),
        "fp_weight": float(fp_weight.item()),
        "transport_direction": direction.detach().cpu().tolist(),
        "transport_b": float(transport_b.item()),
        "subspace_score_w": score_w.detach().cpu().tolist(),
        "train_points_per_class": int(n),
        "source_pair_rank_accuracy": float(pair_acc),
        "source_base_pair_rank_accuracy": float(base_pair_acc),
        "source_delta_fp_mean": float(delta[:n].mean().item()),
        "source_delta_def_mean": float(delta[n:].mean().item()),
        "hardpair_frac": float(hardpair_frac),
        "preserve_host_margin_weight": float(preserve_host_margin_weight),
        "pair_label_control": str(pair_label_control),
    }


def train_vl_rankr_readout_calibrator(
    fp_bank: torch.Tensor,
    def_bank: torch.Tensor,
    axis: torch.Tensor,
    max_train_points: int,
    epochs: int,
    lr: float,
    rank_margin: float,
    rank_temp: float,
    hardpair_frac: float,
    subspace_rank: int,
    seed: int,
    device: torch.device,
) -> dict | None:
    if epochs <= 0 or fp_bank.numel() == 0 or def_bank.numel() == 0:
        return None
    fp = F.normalize(fp_bank.float(), dim=-1)
    defect = F.normalize(def_bank.float(), dim=-1)
    n = min(fp.shape[0], defect.shape[0], max_train_points // 2 if max_train_points > 0 else 10**9)
    if n <= 0:
        return None

    axis = F.normalize(axis.to(device).float(), dim=0)
    basis = build_defect_parallel_basis(fp, defect, axis, subspace_rank, device)
    fp_all = fp.to(device)
    defect_all = defect.to(device)
    fp_host_all = fp_all @ axis
    defect_host_all = defect_all @ axis
    fp_idx, def_idx = sample_source_hardpair_indices(
        fp_scores=fp_host_all,
        defect_scores=defect_host_all,
        n=n,
        hardpair_frac=hardpair_frac,
        seed=seed,
    )

    fp_coords_all = fp_all @ basis
    defect_coords_all = defect_all @ basis
    train_coords_all = torch.cat([fp_coords_all[fp_idx], defect_coords_all[def_idx]], dim=0)
    coord_mean = train_coords_all.mean(dim=0)
    coord_std = train_coords_all.std(dim=0).clamp_min(1e-6)
    fp_coords = (fp_coords_all[fp_idx] - coord_mean) / coord_std
    defect_coords = (defect_coords_all[def_idx] - coord_mean) / coord_std

    raw_w = torch.zeros(basis.shape[1], device=device, requires_grad=True)
    if basis.shape[1] > 1:
        raw_w.data[1] = 1.0
    else:
        raw_w.data[0] = 1.0
    bias = torch.tensor(0.0, device=device, requires_grad=True)
    opt = torch.optim.Adam([raw_w, bias], lr=lr)
    pair_index = torch.arange(n, device=device)
    rank_temp = max(float(rank_temp), 1e-6)

    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        perm = pair_index[torch.randperm(n, device=device)]
        w = F.normalize(raw_w, dim=0)
        s_fp = fp_coords[perm] @ w + bias
        s_def = defect_coords @ w + bias
        loss = F.softplus((s_fp - s_def + rank_margin) / rank_temp).mean()
        loss = loss + 1e-4 * raw_w.pow(2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        w = F.normalize(raw_w, dim=0)
        s_fp = fp_coords @ w + bias
        s_def = defect_coords @ w + bias
        pair_acc = (s_def > s_fp).float().mean().item()
        base_pair_acc = (defect_host_all[def_idx] > fp_host_all[fp_idx]).float().mean().item()
        all_scores = torch.cat(
            [
                ((fp_coords_all - coord_mean) / coord_std) @ w + bias,
                ((defect_coords_all - coord_mean) / coord_std) @ w + bias,
            ],
            dim=0,
        )

    return {
        "calibration": "adaptclip_vl_source_calibrated_rankr_readout_no_target",
        "host_branch": "visual_local_adapter",
        "host_score_source": "rankr_subspace_readout",
        "basis": basis.detach().cpu(),
        "subspace_rank": int(basis.shape[1]),
        "readout_w": w.detach().cpu().tolist(),
        "readout_b": float(bias.item()),
        "coord_mean": coord_mean.detach().cpu().tolist(),
        "coord_std": coord_std.detach().cpu().tolist(),
        "train_points_per_class": int(n),
        "source_pair_rank_accuracy": float(pair_acc),
        "source_base_pair_rank_accuracy": float(base_pair_acc),
        "source_score_fp_mean": float(s_fp.mean().item()),
        "source_score_def_mean": float(s_def.mean().item()),
        "source_score_mean": float(all_scores.mean().item()),
        "source_score_std": float(all_scores.std().clamp_min(1e-6).item()),
        "hardpair_frac": float(hardpair_frac),
    }


def logmeanexp_negative_sqdist_nd(query: torch.Tensor, bank: torch.Tensor, tau: float) -> torch.Tensor:
    if bank.numel() == 0:
        return torch.full((query.shape[0],), -20.0, device=query.device, dtype=query.dtype)
    dist = (query[:, None, :] - bank[None, :, :]).pow(2).sum(dim=-1)
    return torch.logsumexp(-dist / max(float(tau), 1e-6), dim=1) - math.log(bank.shape[0])


def build_vl_rankr_density_calibrator(
    fp_bank: torch.Tensor,
    def_bank: torch.Tensor,
    axis: torch.Tensor,
    hardpair_frac: float,
    subspace_rank: int,
    rankr_tau: float,
    seed: int,
    device: torch.device,
) -> dict | None:
    if fp_bank.numel() == 0 or def_bank.numel() == 0:
        return None
    fp = F.normalize(fp_bank.float(), dim=-1)
    defect = F.normalize(def_bank.float(), dim=-1)
    axis = F.normalize(axis.to(device).float(), dim=0)
    basis = build_defect_parallel_basis(fp, defect, axis, subspace_rank, device)
    fp_all = fp.to(device)
    defect_all = defect.to(device)
    fp_host_all = fp_all @ axis
    defect_host_all = defect_all @ axis
    n = min(fp.shape[0], defect.shape[0])
    fp_idx, def_idx = sample_source_hardpair_indices(
        fp_scores=fp_host_all,
        defect_scores=defect_host_all,
        n=n,
        hardpair_frac=hardpair_frac,
        seed=seed,
    )

    fp_coords_raw = fp_all @ basis
    defect_coords_raw = defect_all @ basis
    all_coords = torch.cat([fp_coords_raw, defect_coords_raw], dim=0)
    coord_mean = all_coords.mean(dim=0)
    coord_std = all_coords.std(dim=0).clamp_min(1e-6)
    fp_coords = (fp_coords_raw - coord_mean) / coord_std
    defect_coords = (defect_coords_raw - coord_mean) / coord_std
    q_fp_on_fp = logmeanexp_negative_sqdist_nd(fp_coords, fp_coords, tau=rankr_tau)
    q_def_on_fp = logmeanexp_negative_sqdist_nd(fp_coords, defect_coords, tau=rankr_tau)
    q_fp_on_def = logmeanexp_negative_sqdist_nd(defect_coords, fp_coords, tau=rankr_tau)
    q_def_on_def = logmeanexp_negative_sqdist_nd(defect_coords, defect_coords, tau=rankr_tau)
    fp_score = q_def_on_fp - q_fp_on_fp
    defect_score = q_def_on_def - q_fp_on_def
    pair_acc = (defect_score[def_idx] > fp_score[fp_idx]).float().mean().item()
    base_pair_acc = (defect_host_all[def_idx] > fp_host_all[fp_idx]).float().mean().item()

    return {
        "calibration": "adaptclip_vl_source_calibrated_rankr_density_no_target",
        "host_branch": "visual_local_adapter",
        "host_score_source": "rankr_bank_density_margin",
        "basis": basis.detach().cpu(),
        "subspace_rank": int(basis.shape[1]),
        "rankr_tau": float(rankr_tau),
        "coord_mean": coord_mean.detach().cpu().tolist(),
        "coord_std": coord_std.detach().cpu().tolist(),
        "fp_coords": fp_coords.detach().cpu(),
        "def_coords": defect_coords.detach().cpu(),
        "train_points_per_class": int(n),
        "source_pair_rank_accuracy": float(pair_acc),
        "source_base_pair_rank_accuracy": float(base_pair_acc),
        "source_score_fp_mean": float(fp_score.mean().item()),
        "source_score_def_mean": float(defect_score.mean().item()),
        "hardpair_frac": float(hardpair_frac),
    }


def calibrated_vl_residual_tokens(
    tokens: torch.Tensor,
    axis: torch.Tensor,
    fp_coeff: torch.Tensor,
    def_coeff: torch.Tensor,
    tau: float,
    calibrator: dict,
) -> torch.Tensor:
    axis = F.normalize(axis.to(tokens.device).float(), dim=0)
    basis = calibrator["basis"].to(tokens.device).float()
    coeff = tokens @ axis
    q_fp = logmeanexp_negative_sqdist_1d(coeff, fp_coeff, tau=tau)
    q_def = logmeanexp_negative_sqdist_1d(coeff, def_coeff, tau=tau)
    direction = F.normalize(torch.tensor(calibrator["transport_direction"], device=tokens.device, dtype=torch.float32), dim=0)
    score_w = F.normalize(torch.tensor(calibrator["subspace_score_w"], device=tokens.device, dtype=torch.float32), dim=0)
    raw_margin = q_def - float(calibrator["fp_weight"]) * q_fp
    margin = spatial_tanh_zscore(raw_margin)
    move_dir = F.normalize(basis @ direction, dim=0)
    delta = float(calibrator["eta"]) * torch.tanh(margin + float(calibrator["transport_b"]))
    rect = F.normalize(tokens.float() + delta.unsqueeze(-1) * move_dir.view(1, -1), dim=-1)
    coords = rect @ basis
    if basis.shape[1] > 1:
        residual = coords[:, 1:] @ score_w
    else:
        residual = coords[:, 0]
    corrected = coeff.float() + float(calibrator["gamma"]) * residual
    return corrected - coeff.float()


def calibrated_vl_readout_tokens(tokens: torch.Tensor, calibrator: dict) -> torch.Tensor:
    basis = calibrator["basis"].to(tokens.device).float()
    w = torch.tensor(calibrator["readout_w"], device=tokens.device, dtype=torch.float32)
    w = F.normalize(w, dim=0)
    coord_mean = torch.tensor(calibrator["coord_mean"], device=tokens.device, dtype=torch.float32)
    coord_std = torch.tensor(calibrator["coord_std"], device=tokens.device, dtype=torch.float32).clamp_min(1e-6)
    coords = (tokens.float() @ basis - coord_mean) / coord_std
    return coords @ w + float(calibrator["readout_b"])


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


def calibrated_branch_tokens(
    tokens: torch.Tensor,
    axis: torch.Tensor,
    fp_coeff: torch.Tensor,
    def_coeff: torch.Tensor,
    tau: float,
    calibrator: dict,
) -> torch.Tensor:
    if calibrator.get("calibration") == "adaptclip_vl_source_calibrated_rankr_readout_no_target":
        return calibrated_vl_readout_tokens(tokens=tokens, calibrator=calibrator)
    if calibrator.get("calibration") == "adaptclip_vl_source_calibrated_rankr_density_no_target":
        return calibrated_vl_density_tokens(tokens=tokens, calibrator=calibrator)
    return calibrated_vl_residual_tokens(
        tokens=tokens,
        axis=axis,
        fp_coeff=fp_coeff,
        def_coeff=def_coeff,
        tau=tau,
        calibrator=calibrator,
    )


def safe_calibrator_summary(calibrator: dict | None) -> dict | None:
    if calibrator is None:
        return None
    return {k: v for k, v in calibrator.items() if k not in {"basis", "fp_coords", "def_coords"}}


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


def trainfree_margin_map(
    patch: torch.Tensor,
    axis: torch.Tensor,
    fp_coeff: torch.Tensor,
    def_coeff: torch.Tensor,
    tau: float,
    fp_weight: float,
    image_size: int,
    sigma: float,
    activation: str,
    temperature: float,
    device: str,
) -> tuple[torch.Tensor, int]:
    coeff = patch @ axis
    q_fp = logmeanexp_negative_sqdist_1d(coeff, fp_coeff, tau=tau)
    q_def = logmeanexp_negative_sqdist_1d(coeff, def_coeff, tau=tau)
    h = int(round(patch.shape[0] ** 0.5))
    margin_map = token_scores_to_map(q_def - float(fp_weight) * q_fp, h, image_size, sigma, device)
    margin_map = activate_margin_map(margin_map, activation=activation, temperature=temperature)
    return margin_map, h


def trainfree_rankr_margin_map(
    patch: torch.Tensor,
    calibrator: dict,
    image_size: int,
    sigma: float,
    activation: str,
    temperature: float,
    device: str,
) -> tuple[torch.Tensor, int]:
    token_scores = calibrated_vl_density_tokens(patch, calibrator)
    h = int(round(patch.shape[0] ** 0.5))
    margin_map = token_scores_to_map(token_scores, h, image_size, sigma, device)
    margin_map = activate_margin_map(margin_map, activation=activation, temperature=temperature)
    return margin_map, h


def append_result(
    results: dict,
    mode: str,
    sample_id: str,
    cls_name: str,
    query_path: str,
    gt_mask: torch.Tensor,
    gt_anomaly: torch.Tensor,
    pr_map: torch.Tensor,
    pr_score: torch.Tensor,
) -> None:
    results[mode]["sample_ids"].append(sample_id)
    results[mode]["cls_names"].append(cls_name)
    results[mode]["query_paths"].append(query_path)
    results[mode]["gt_masks"].append(gt_mask.detach().cpu().int())
    results[mode]["pr_masks"].append(pr_map.detach().cpu())
    results[mode]["gt_anomalys"].append(gt_anomaly.detach().cpu())
    results[mode]["pr_anomalys"].append(pr_score.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser("Official AdaptCLIP VL-refine + TL re-fusion test runner")
    parser.add_argument("--source_root", type=str, required=True)
    parser.add_argument("--source_dataset", type=str, required=True)
    parser.add_argument("--target_root", type=str, required=True)
    parser.add_argument("--target_dataset", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--bank_cache_path", type=str, required=True)
    parser.add_argument("--tl_bank_cache_path", type=str, default=None)
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--pretrained_model", type=str, default="ViT-L/14@336px")
    parser.add_argument(
        "--checkpoint_load_mode",
        type=str,
        default="strict",
        choices=["strict", "compatible"],
        help="Use compatible for backbone-swap stress; only shape-compatible adapter tensors are loaded.",
    )
    parser.add_argument("--features_list", type=int, nargs="+", default=[24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--vl_reduction", type=int, default=4)
    parser.add_argument("--fusion_type", type=str, default="average_mean")
    parser.add_argument("--hard_frac", type=float, default=0.01)
    parser.add_argument("--max_good_per_class", type=int, default=2)
    parser.add_argument("--max_defect_per_class", type=int, default=4)
    parser.add_argument("--max_bank_points", type=int, default=512)
    parser.add_argument("--max_fp_per_image", type=int, default=64)
    parser.add_argument("--max_defect_per_image", type=int, default=64)
    parser.add_argument("--bank_mining_source", type=str, choices=["vl", "tl", "baseline"], default="vl")
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--fp_weight", type=float, default=0.3)
    parser.add_argument("--corr_alpha", type=float, default=0.3)
    parser.add_argument("--map_activation", type=str, default="sigmoid")
    parser.add_argument("--map_temperature", type=float, default=1.0)
    parser.add_argument("--trainfree_score_mode", type=str, choices=["axis", "rankr_density"], default="axis")
    parser.add_argument("--vl_gate_mode", type=str, choices=["host", "none"], default="host")
    parser.add_argument("--vl_gate_quantile", type=float, default=0.9)
    parser.add_argument("--vl_gate_temp", type=float, default=0.25)
    parser.add_argument(
        "--ted_insertion",
        type=str,
        choices=["branch", "fused", "dual_branch"],
        default="branch",
        help="branch edits VL before TL fusion; fused edits official fused map; dual_branch edits VL and TL separately before official fusion.",
    )
    parser.add_argument("--calib_epochs", type=int, default=0)
    parser.add_argument("--calib_lr", type=float, default=0.02)
    parser.add_argument("--calib_max_train_points", type=int, default=8192)
    parser.add_argument("--calib_rank_margin", type=float, default=0.05)
    parser.add_argument("--calib_rank_temp", type=float, default=0.05)
    parser.add_argument("--calib_eta_init", type=float, default=0.2)
    parser.add_argument("--calib_eta_max", type=float, default=1.0)
    parser.add_argument("--calib_eta_reg", type=float, default=0.001)
    parser.add_argument("--calib_fp_weight_init", type=float, default=1.0)
    parser.add_argument("--calib_learn_fp_weight", action="store_true")
    parser.add_argument("--calib_hardpair_frac", type=float, default=0.75)
    parser.add_argument("--calib_subspace_rank", type=int, default=8)
    parser.add_argument("--calib_preserve_host_margin_weight", type=float, default=1.0)
    parser.add_argument("--calib_lambda", type=float, default=0.3)
    parser.add_argument("--calib_pair_label_control", type=str, default="correct", choices=["correct", "shuffle", "swap"])
    parser.add_argument(
        "--calib_score_mode",
        type=str,
        choices=["residual_subspace", "rankr_readout", "rankr_density"],
        default="residual_subspace",
    )
    parser.add_argument("--plugin_epochs", type=int, default=0)
    parser.add_argument("--plugin_lr", type=float, default=0.02)
    parser.add_argument("--plugin_max_train_points", type=int, default=2048)
    parser.add_argument("--plugin_hardpair_frac", type=float, default=0.0)
    parser.add_argument("--plugin_alpha", type=float, default=0.5)
    parser.add_argument("--plugin_loss", type=str, default="bce", choices=["bce", "pair_rank", "support_rank"])
    parser.add_argument("--plugin_feature_space", type=str, default="axis", choices=["axis", "hostscore"])
    parser.add_argument("--plugin_hidden_dim", type=int, default=32)
    parser.add_argument("--plugin_rank_margin", type=float, default=0.05)
    parser.add_argument("--plugin_rank_temp", type=float, default=0.05)
    parser.add_argument(
        "--calib_apply_mode",
        type=str,
        choices=["add", "replace"],
        default="add",
    )
    parser.add_argument("--calib_rankr_tau", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_eval_images", type=int, default=0)
    parser.add_argument(
        "--eval_metrics",
        nargs="+",
        default=["I-AUROC", "I-AP", "P-AUROC", "P-AP", "P-AUPRO"],
    )
    parser.add_argument("--aupro_max_step", type=int, default=200)
    parser.add_argument("--aupro_expect_fpr", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--report_group_separability", action="store_true")
    parser.add_argument("--separability_hard_frac", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model, textual_learner, visual_learner, learned_text_features, dpam_layer = build_model(
        args.device,
        args.checkpoint_path,
        args.image_size,
        args.n_ctx,
        args.vl_reduction,
        args.pretrained_model,
        checkpoint_load_mode=args.checkpoint_load_mode,
    )
    vl_axis = decision_axis(textual_learner.static_text_features).to(args.device)
    tl_axis = decision_axis(learned_text_features).to(args.device)

    vl_bank_mining_source = "vl" if args.ted_insertion == "dual_branch" else args.bank_mining_source
    fp_bank, def_bank, bank_stats, bank_cache_path, bank_cache_meta = get_or_collect_vl_banks(
        model=model,
        textual_learner=textual_learner,
        visual_learner=visual_learner,
        learned_text_features=learned_text_features,
        source_root=args.source_root,
        source_dataset=args.source_dataset,
        image_size=args.image_size,
        features_list=args.features_list,
        dpam_layer=dpam_layer,
        sigma=args.sigma,
        fusion_type=args.fusion_type,
        device=args.device,
        hard_frac=args.hard_frac,
        max_good_per_class=args.max_good_per_class,
        max_defect_per_class=args.max_defect_per_class,
        max_bank_points=args.max_bank_points,
        max_fp_per_image=args.max_fp_per_image,
        max_defect_per_image=args.max_defect_per_image,
        checkpoint_path=args.checkpoint_path,
        bank_cache_path=args.bank_cache_path,
        bank_mining_source=vl_bank_mining_source,
        bank_feature_source="vl",
    )
    fp_coeff = fp_bank.to(args.device) @ vl_axis if fp_bank.numel() else torch.empty((0,), device=args.device)
    def_coeff = def_bank.to(args.device) @ vl_axis if def_bank.numel() else torch.empty((0,), device=args.device)
    if args.plugin_feature_space == "hostscore":
        source_vl_plugin = train_hostscore_plugin_from_scores(
            fp_scores=fp_coeff,
            def_scores=def_coeff,
            max_train_points=args.plugin_max_train_points,
            epochs=args.plugin_epochs,
            lr=args.plugin_lr,
            hardpair_frac=args.plugin_hardpair_frac,
            loss_type=args.plugin_loss,
            hidden_dim=args.plugin_hidden_dim,
            rank_margin=args.plugin_rank_margin,
            rank_temp=args.plugin_rank_temp,
            seed=args.seed,
            device=torch.device(args.device),
        )
    else:
        source_vl_plugin = train_scalar_plugin_for_bank(
            fp_bank=fp_bank,
            def_bank=def_bank,
            axis=vl_axis,
            tau=args.tau,
            max_train_points=args.plugin_max_train_points,
            epochs=args.plugin_epochs,
            lr=args.plugin_lr,
            hardpair_frac=args.plugin_hardpair_frac,
            loss_type=args.plugin_loss,
            hidden_dim=args.plugin_hidden_dim,
            rank_margin=args.plugin_rank_margin,
            rank_temp=args.plugin_rank_temp,
            seed=args.seed,
            device=torch.device(args.device),
        )
    if source_vl_plugin is not None:
        print(
            json.dumps(
                {
                    "stage": "source_vl_scalar_plugin_trained",
                    "train_accuracy": source_vl_plugin.get("train_accuracy"),
                    "source_pair_rank_accuracy": source_vl_plugin.get("source_pair_rank_accuracy"),
                    "train_points_per_class": source_vl_plugin.get("train_points_per_class"),
                    "plugin_loss": args.plugin_loss,
                    "plugin_feature_space": args.plugin_feature_space,
                }
            ),
            flush=True,
        )

    tl_fp_bank = tl_def_bank = torch.empty((0, fp_bank.shape[-1] if fp_bank.numel() else 1))
    tl_bank_stats: dict = {}
    tl_bank_cache_path: Path | None = None
    tl_bank_cache_meta: dict = {}
    tl_fp_coeff = torch.empty((0,), device=args.device)
    tl_def_coeff = torch.empty((0,), device=args.device)
    source_tl_plugin = None
    if args.ted_insertion == "dual_branch":
        tl_cache_path = args.tl_bank_cache_path
        if tl_cache_path is None:
            base_path = Path(args.bank_cache_path)
            tl_cache_path = str(base_path.with_name(base_path.stem + "_tl" + base_path.suffix))
        tl_fp_bank, tl_def_bank, tl_bank_stats, tl_bank_cache_path, tl_bank_cache_meta = get_or_collect_vl_banks(
            model=model,
            textual_learner=textual_learner,
            visual_learner=visual_learner,
            learned_text_features=learned_text_features,
            source_root=args.source_root,
            source_dataset=args.source_dataset,
            image_size=args.image_size,
            features_list=args.features_list,
            dpam_layer=dpam_layer,
            sigma=args.sigma,
            fusion_type=args.fusion_type,
            device=args.device,
            hard_frac=args.hard_frac,
            max_good_per_class=args.max_good_per_class,
            max_defect_per_class=args.max_defect_per_class,
            max_bank_points=args.max_bank_points,
            max_fp_per_image=args.max_fp_per_image,
            max_defect_per_image=args.max_defect_per_image,
            checkpoint_path=args.checkpoint_path,
            bank_cache_path=tl_cache_path,
            bank_mining_source="tl",
            bank_feature_source="tl",
        )
        tl_fp_coeff = tl_fp_bank.to(args.device) @ tl_axis if tl_fp_bank.numel() else torch.empty((0,), device=args.device)
        tl_def_coeff = tl_def_bank.to(args.device) @ tl_axis if tl_def_bank.numel() else torch.empty((0,), device=args.device)
        if args.plugin_feature_space == "hostscore":
            source_tl_plugin = train_hostscore_plugin_from_scores(
                fp_scores=tl_fp_coeff,
                def_scores=tl_def_coeff,
                max_train_points=args.plugin_max_train_points,
                epochs=args.plugin_epochs,
                lr=args.plugin_lr,
                hardpair_frac=args.plugin_hardpair_frac,
                loss_type=args.plugin_loss,
                hidden_dim=args.plugin_hidden_dim,
                rank_margin=args.plugin_rank_margin,
                rank_temp=args.plugin_rank_temp,
                seed=args.seed + 17,
                device=torch.device(args.device),
            )
        else:
            source_tl_plugin = train_scalar_plugin_for_bank(
                fp_bank=tl_fp_bank,
                def_bank=tl_def_bank,
                axis=tl_axis,
                tau=args.tau,
                max_train_points=args.plugin_max_train_points,
                epochs=args.plugin_epochs,
                lr=args.plugin_lr,
                hardpair_frac=args.plugin_hardpair_frac,
                loss_type=args.plugin_loss,
                hidden_dim=args.plugin_hidden_dim,
                rank_margin=args.plugin_rank_margin,
                rank_temp=args.plugin_rank_temp,
                seed=args.seed + 17,
                device=torch.device(args.device),
            )
    if args.calib_score_mode == "rankr_density":
        source_vl_calibrator = build_vl_rankr_density_calibrator(
            fp_bank=fp_bank,
            def_bank=def_bank,
            axis=vl_axis,
            hardpair_frac=args.calib_hardpair_frac,
            subspace_rank=args.calib_subspace_rank,
            rankr_tau=args.calib_rankr_tau,
            seed=args.seed,
            device=torch.device(args.device),
        )
    elif args.calib_score_mode == "rankr_readout":
        source_vl_calibrator = train_vl_rankr_readout_calibrator(
            fp_bank=fp_bank,
            def_bank=def_bank,
            axis=vl_axis,
            max_train_points=args.calib_max_train_points,
            epochs=args.calib_epochs,
            lr=args.calib_lr,
            rank_margin=args.calib_rank_margin,
            rank_temp=args.calib_rank_temp,
            hardpair_frac=args.calib_hardpair_frac,
            subspace_rank=args.calib_subspace_rank,
            seed=args.seed,
            device=torch.device(args.device),
        )
    else:
        source_vl_calibrator = train_vl_subspace_calibrator(
            fp_bank=fp_bank,
            def_bank=def_bank,
            axis=vl_axis,
            tau=args.tau,
            max_train_points=args.calib_max_train_points,
            epochs=args.calib_epochs,
            lr=args.calib_lr,
            rank_margin=args.calib_rank_margin,
            rank_temp=args.calib_rank_temp,
            eta_init=args.calib_eta_init,
            eta_max=args.calib_eta_max,
            eta_reg=args.calib_eta_reg,
            fp_weight_init=args.calib_fp_weight_init,
            learn_fp_weight=args.calib_learn_fp_weight,
            hardpair_frac=args.calib_hardpair_frac,
            subspace_rank=args.calib_subspace_rank,
            preserve_host_margin_weight=args.calib_preserve_host_margin_weight,
            pair_label_control=args.calib_pair_label_control,
            seed=args.seed,
            device=torch.device(args.device),
        )
    source_tl_calibrator = None
    if args.ted_insertion == "dual_branch":
        if args.calib_score_mode == "rankr_density":
            source_tl_calibrator = build_vl_rankr_density_calibrator(
                fp_bank=tl_fp_bank,
                def_bank=tl_def_bank,
                axis=tl_axis,
                hardpair_frac=args.calib_hardpair_frac,
                subspace_rank=args.calib_subspace_rank,
                rankr_tau=args.calib_rankr_tau,
                seed=args.seed,
                device=torch.device(args.device),
            )
        elif args.calib_score_mode == "rankr_readout":
            source_tl_calibrator = train_vl_rankr_readout_calibrator(
                fp_bank=tl_fp_bank,
                def_bank=tl_def_bank,
                axis=tl_axis,
                max_train_points=args.calib_max_train_points,
                epochs=args.calib_epochs,
                lr=args.calib_lr,
                rank_margin=args.calib_rank_margin,
                rank_temp=args.calib_rank_temp,
                hardpair_frac=args.calib_hardpair_frac,
                subspace_rank=args.calib_subspace_rank,
                seed=args.seed,
                device=torch.device(args.device),
            )
        else:
            source_tl_calibrator = train_vl_subspace_calibrator(
                fp_bank=tl_fp_bank,
                def_bank=tl_def_bank,
                axis=tl_axis,
                tau=args.tau,
                max_train_points=args.calib_max_train_points,
                epochs=args.calib_epochs,
                lr=args.calib_lr,
                rank_margin=args.calib_rank_margin,
                rank_temp=args.calib_rank_temp,
                eta_init=args.calib_eta_init,
                eta_max=args.calib_eta_max,
                eta_reg=args.calib_eta_reg,
                fp_weight_init=args.calib_fp_weight_init,
                learn_fp_weight=args.calib_learn_fp_weight,
                hardpair_frac=args.calib_hardpair_frac,
                subspace_rank=args.calib_subspace_rank,
                preserve_host_margin_weight=args.calib_preserve_host_margin_weight,
                pair_label_control=args.calib_pair_label_control,
                seed=args.seed,
                device=torch.device(args.device),
            )
    if source_vl_calibrator is not None:
        print(
            json.dumps(
                {
                    "stage": "source_vl_calibrator_trained",
                    **safe_calibrator_summary(source_vl_calibrator),
                }
            ),
            flush=True,
        )
    if source_tl_calibrator is not None:
        print(
            json.dumps(
                {
                    "stage": "source_tl_calibrator_trained",
                    **safe_calibrator_summary(source_tl_calibrator),
                }
            ),
            flush=True,
        )

    dataset = build_dataset(
        args.target_root,
        args.image_size,
        args.target_dataset,
        "test",
        str(save_dir),
        class_name=args.class_name,
    )
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["pin_memory"] = args.device.startswith("cuda")
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    mode_names = ["baseline_txt", "ted_trainfree_vl"]
    if source_vl_plugin is not None:
        mode_names.extend(["scalar_replace_vl", "scalar_residual_vl"])
    if source_vl_calibrator is not None:
        mode_names.append("ted_calibrated_vl")
    results = {
        mode: {"sample_ids": [], "gt_masks": [], "pr_masks": [], "cls_names": [], "gt_anomalys": [], "pr_anomalys": [], "query_paths": []}
        for mode in mode_names
    }

    with torch.no_grad():
        for idx, items in enumerate(loader, start=1):
            if args.max_eval_images > 0 and idx > args.max_eval_images:
                break
            image = items["img"].to(args.device)
            cls_name = items["cls_name"][0]
            sample_id = items["sample_id"][0]
            query_path = items["img_path"][0]
            gt_mask = items["img_mask"][:, 0].clone().to(args.device)
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            gt_mask = gt_mask.int()
            gt_anomaly = items["anomaly"].to(args.device).int()

            baseline = compute_baseline_outputs(
                model,
                textual_learner,
                visual_learner,
                learned_text_features,
                image,
                args.features_list,
                dpam_layer,
                args.image_size,
                args.sigma,
                args.fusion_type,
            )

            raw_vl_map = baseline["local_vl_map"]
            raw_tl_map = baseline["local_tl_map"]
            vl_map = smooth_map(raw_vl_map, sigma=args.sigma).to(args.device)
            vl_map = resize_map(vl_map, args.image_size)
            tl_map = smooth_map(raw_tl_map, sigma=args.sigma).to(args.device)
            tl_map = resize_map(tl_map, args.image_size)
            host_map = baseline["baseline_map"]
            insertion_ref_map = host_map if args.ted_insertion == "fused" else vl_map
            ted_gate = (
                host_response_gate(insertion_ref_map, args.vl_gate_quantile, args.vl_gate_temp)
                if args.vl_gate_mode == "host"
                else torch.ones_like(insertion_ref_map)
            )

            vl_patch = vl_patch_bank_features(visual_learner, baseline["query_patch_feats"])
            if args.trainfree_score_mode == "rankr_density":
                margin_map, h = trainfree_rankr_margin_map(
                    vl_patch,
                    source_vl_calibrator,
                    args.image_size,
                    args.sigma,
                    args.map_activation,
                    args.map_temperature,
                    args.device,
                )
            else:
                margin_map, h = trainfree_margin_map(
                    vl_patch,
                    vl_axis,
                    fp_coeff,
                    def_coeff,
                    args.tau,
                    args.fp_weight,
                    args.image_size,
                    args.sigma,
                    args.map_activation,
                    args.map_temperature,
                    args.device,
                )
            scalar_plugin_map = None
            if source_vl_plugin is not None:
                vl_query_coeff = vl_patch @ vl_axis
                if source_vl_plugin.get("feature_space") == "hostscore":
                    scalar_token = spatial_tanh_zscore(apply_hostscore_plugin(vl_query_coeff, source_vl_plugin))
                else:
                    vl_q_fp = logmeanexp_negative_sqdist_1d(vl_query_coeff, fp_coeff, tau=args.tau)
                    vl_q_def = logmeanexp_negative_sqdist_1d(vl_query_coeff, def_coeff, tau=args.tau)
                    scalar_token = spatial_tanh_zscore(
                        apply_scalar_plugin(vl_query_coeff, vl_q_def, vl_q_fp, source_vl_plugin)
                    )
                scalar_plugin_map = token_scores_to_map(scalar_token, h, args.image_size, args.sigma, args.device)

            if args.ted_insertion == "fused":
                parallel_map = torch.clamp(host_map + args.corr_alpha * ted_gate * (margin_map - 0.5), 0.0, 1.0)
            elif args.ted_insertion == "dual_branch":
                tl_patch = branch_patch_bank_features(visual_learner, baseline["query_patch_feats"], "tl")
                if args.trainfree_score_mode == "rankr_density":
                    tl_margin_map, _ = trainfree_rankr_margin_map(
                        tl_patch,
                        source_tl_calibrator,
                        args.image_size,
                        args.sigma,
                        args.map_activation,
                        args.map_temperature,
                        args.device,
                    )
                else:
                    tl_margin_map, _ = trainfree_margin_map(
                        tl_patch,
                        tl_axis,
                        tl_fp_coeff,
                        tl_def_coeff,
                        args.tau,
                        args.fp_weight,
                        args.image_size,
                        args.sigma,
                        args.map_activation,
                        args.map_temperature,
                        args.device,
                    )
                vl_gate = host_response_gate(vl_map, args.vl_gate_quantile, args.vl_gate_temp) if args.vl_gate_mode == "host" else torch.ones_like(vl_map)
                tl_gate = host_response_gate(tl_map, args.vl_gate_quantile, args.vl_gate_temp) if args.vl_gate_mode == "host" else torch.ones_like(tl_map)
                refined_vl_map = torch.clamp(vl_map + args.corr_alpha * vl_gate * (margin_map - 0.5), 0.0, 1.0)
                refined_tl_map = torch.clamp(tl_map + args.corr_alpha * tl_gate * (tl_margin_map - 0.5), 0.0, 1.0)
                parallel_map = smooth_map(fusion_fun([refined_vl_map, refined_tl_map], fusion_type=args.fusion_type), sigma=args.sigma).to(args.device)
                parallel_map = resize_map(parallel_map, args.image_size)
            else:
                refined_vl_map = torch.clamp(vl_map + args.corr_alpha * ted_gate * (margin_map - 0.5), 0.0, 1.0)
                parallel_map = smooth_map(fusion_fun([raw_tl_map, refined_vl_map], fusion_type=args.fusion_type), sigma=args.sigma).to(args.device)
                parallel_map = resize_map(parallel_map, args.image_size)

            anomaly_map_max_parallel, _ = torch.max(parallel_map.view(image.shape[0], -1), dim=1)
            parallel_img_score = fusion_fun(
                [baseline["global_vl_score"], baseline["global_tl_score"], anomaly_map_max_parallel],
                fusion_type=args.fusion_type,
            )

            outputs_to_append = [
                ("baseline_txt", baseline["baseline_map"], baseline["baseline_img_score"]),
                ("ted_trainfree_vl", parallel_map, parallel_img_score),
            ]
            if scalar_plugin_map is not None:
                scalar_replace_vl = torch.clamp((scalar_plugin_map + 1.0) * 0.5, 0.0, 1.0)
                scalar_residual_vl = torch.clamp(
                    vl_map + float(args.plugin_alpha) * ted_gate * scalar_plugin_map,
                    0.0,
                    1.0,
                )
                if args.ted_insertion == "fused":
                    scalar_replace_map = scalar_replace_vl
                    scalar_residual_map = torch.clamp(
                        host_map + float(args.plugin_alpha) * ted_gate * scalar_plugin_map,
                        0.0,
                        1.0,
                    )
                elif args.ted_insertion == "dual_branch" and source_tl_plugin is not None:
                    tl_query_coeff = tl_patch @ tl_axis
                    if source_tl_plugin.get("feature_space") == "hostscore":
                        tl_scalar_token = spatial_tanh_zscore(apply_hostscore_plugin(tl_query_coeff, source_tl_plugin))
                    else:
                        tl_q_fp = logmeanexp_negative_sqdist_1d(tl_query_coeff, tl_fp_coeff, tau=args.tau)
                        tl_q_def = logmeanexp_negative_sqdist_1d(tl_query_coeff, tl_def_coeff, tau=args.tau)
                        tl_scalar_token = spatial_tanh_zscore(
                            apply_scalar_plugin(tl_query_coeff, tl_q_def, tl_q_fp, source_tl_plugin)
                        )
                    tl_scalar_map = token_scores_to_map(tl_scalar_token, h, args.image_size, args.sigma, args.device)
                    scalar_replace_map = smooth_map(
                        fusion_fun([scalar_replace_vl, torch.clamp((tl_scalar_map + 1.0) * 0.5, 0.0, 1.0)], fusion_type=args.fusion_type),
                        sigma=args.sigma,
                    ).to(args.device)
                    scalar_replace_map = resize_map(scalar_replace_map, args.image_size)
                    tl_gate = host_response_gate(tl_map, args.vl_gate_quantile, args.vl_gate_temp) if args.vl_gate_mode == "host" else torch.ones_like(tl_map)
                    scalar_residual_map = smooth_map(
                        fusion_fun(
                            [
                                scalar_residual_vl,
                                torch.clamp(tl_map + float(args.plugin_alpha) * tl_gate * tl_scalar_map, 0.0, 1.0),
                            ],
                            fusion_type=args.fusion_type,
                        ),
                        sigma=args.sigma,
                    ).to(args.device)
                    scalar_residual_map = resize_map(scalar_residual_map, args.image_size)
                else:
                    scalar_replace_map = smooth_map(
                        fusion_fun([raw_tl_map, scalar_replace_vl], fusion_type=args.fusion_type),
                        sigma=args.sigma,
                    ).to(args.device)
                    scalar_replace_map = resize_map(scalar_replace_map, args.image_size)
                    scalar_residual_map = smooth_map(
                        fusion_fun([raw_tl_map, scalar_residual_vl], fusion_type=args.fusion_type),
                        sigma=args.sigma,
                    ).to(args.device)
                    scalar_residual_map = resize_map(scalar_residual_map, args.image_size)
                scalar_replace_img, _ = torch.max(scalar_replace_map.view(image.shape[0], -1), dim=1)
                scalar_residual_img, _ = torch.max(scalar_residual_map.view(image.shape[0], -1), dim=1)
                outputs_to_append.append(
                    (
                        "scalar_replace_vl",
                        scalar_replace_map,
                        fusion_fun([baseline["global_vl_score"], baseline["global_tl_score"], scalar_replace_img], fusion_type=args.fusion_type),
                    )
                )
                outputs_to_append.append(
                    (
                        "scalar_residual_vl",
                        scalar_residual_map,
                        fusion_fun([baseline["global_vl_score"], baseline["global_tl_score"], scalar_residual_img], fusion_type=args.fusion_type),
                    )
                )

            if source_vl_calibrator is not None:
                calibrator_token = calibrated_branch_tokens(
                    tokens=vl_patch,
                    axis=vl_axis,
                    fp_coeff=fp_coeff,
                    def_coeff=def_coeff,
                    tau=args.tau,
                    calibrator=source_vl_calibrator,
                )
                residual_map = token_scores_to_map(calibrator_token, h, args.image_size, args.sigma, args.device)
                residual_map = spatial_tanh_zscore(residual_map)
                if args.calib_apply_mode == "replace":
                    replacement_map = torch.clamp((residual_map + 1.0) * 0.5, 0.0, 1.0)
                    if args.ted_insertion == "fused":
                        calibrated_map = replacement_map
                    elif args.ted_insertion == "dual_branch" and source_tl_calibrator is not None:
                        tl_token = calibrated_branch_tokens(
                            tokens=tl_patch,
                            axis=tl_axis,
                            fp_coeff=tl_fp_coeff,
                            def_coeff=tl_def_coeff,
                            tau=args.tau,
                            calibrator=source_tl_calibrator,
                        )
                        tl_residual_map = spatial_tanh_zscore(
                            token_scores_to_map(tl_token, h, args.image_size, args.sigma, args.device)
                        )
                        calibrated_vl_map = replacement_map
                        calibrated_tl_map = torch.clamp((tl_residual_map + 1.0) * 0.5, 0.0, 1.0)
                        calibrated_map = smooth_map(
                            fusion_fun([calibrated_vl_map, calibrated_tl_map], fusion_type=args.fusion_type),
                            sigma=args.sigma,
                        ).to(args.device)
                        calibrated_map = resize_map(calibrated_map, args.image_size)
                    else:
                        calibrated_vl_map = replacement_map
                        calibrated_map = smooth_map(
                            fusion_fun([raw_tl_map, calibrated_vl_map], fusion_type=args.fusion_type),
                            sigma=args.sigma,
                        ).to(args.device)
                        calibrated_map = resize_map(calibrated_map, args.image_size)
                else:
                    if args.ted_insertion == "fused":
                        calibrated_map = torch.clamp(host_map + float(args.calib_lambda) * ted_gate * residual_map, 0.0, 1.0)
                    elif args.ted_insertion == "dual_branch" and source_tl_calibrator is not None:
                        tl_token = calibrated_branch_tokens(
                            tokens=tl_patch,
                            axis=tl_axis,
                            fp_coeff=tl_fp_coeff,
                            def_coeff=tl_def_coeff,
                            tau=args.tau,
                            calibrator=source_tl_calibrator,
                        )
                        tl_residual_map = spatial_tanh_zscore(
                            token_scores_to_map(tl_token, h, args.image_size, args.sigma, args.device)
                        )
                        vl_gate = host_response_gate(vl_map, args.vl_gate_quantile, args.vl_gate_temp) if args.vl_gate_mode == "host" else torch.ones_like(vl_map)
                        tl_gate = host_response_gate(tl_map, args.vl_gate_quantile, args.vl_gate_temp) if args.vl_gate_mode == "host" else torch.ones_like(tl_map)
                        calibrated_vl_map = torch.clamp(vl_map + float(args.calib_lambda) * vl_gate * residual_map, 0.0, 1.0)
                        calibrated_tl_map = torch.clamp(tl_map + float(args.calib_lambda) * tl_gate * tl_residual_map, 0.0, 1.0)
                        calibrated_map = smooth_map(
                            fusion_fun([calibrated_vl_map, calibrated_tl_map], fusion_type=args.fusion_type),
                            sigma=args.sigma,
                        ).to(args.device)
                        calibrated_map = resize_map(calibrated_map, args.image_size)
                    else:
                        calibrated_vl_map = torch.clamp(vl_map + float(args.calib_lambda) * ted_gate * residual_map, 0.0, 1.0)
                        calibrated_map = smooth_map(
                            fusion_fun([raw_tl_map, calibrated_vl_map], fusion_type=args.fusion_type),
                            sigma=args.sigma,
                        ).to(args.device)
                        calibrated_map = resize_map(calibrated_map, args.image_size)
                anomaly_map_max_calibrated, _ = torch.max(calibrated_map.view(image.shape[0], -1), dim=1)
                calibrated_img_score = fusion_fun(
                    [baseline["global_vl_score"], baseline["global_tl_score"], anomaly_map_max_calibrated],
                    fusion_type=args.fusion_type,
                )
                outputs_to_append.append(("ted_calibrated_vl", calibrated_map, calibrated_img_score))

            for mode, map_tensor, image_score in outputs_to_append:
                append_result(results, mode, sample_id, cls_name, query_path, gt_mask, gt_anomaly, map_tensor, image_score)

            if idx % 8 == 0:
                print(json.dumps({"stage": "eval_progress", "images_done": idx, "current_class": cls_name}), flush=True)

    rows = []
    separability_mean = {}
    class_order = dataset.obj_list if args.class_name is None else [args.class_name]
    for mode in mode_names:
        result_mode = aggregate_results(results[mode])
        if args.report_group_separability:
            masks_np_all, maps_np_all = prepare_aupro_arrays(result_mode["gt_masks"], result_mode["pr_masks"])
            labels_np_all = result_mode["gt_anomalys"].detach().cpu().numpy().astype(np.int64).reshape(-1)
            separability_mean[mode] = map_group_separability(
                masks_np_all,
                labels_np_all,
                maps_np_all,
                hard_frac=args.separability_hard_frac,
            )
        image_aurocs, image_aps, pixel_aurocs, pixel_aps, pixel_aupros = [], [], [], [], []
        evaluator_metrics = [m for m in args.eval_metrics if not m.startswith("P-AUPRO")]
        evaluator = Evaluator("cpu", metrics=evaluator_metrics, sample_level=False) if evaluator_metrics else None
        aupro_region_cache: dict[str, dict] = {}
        for cls_name in class_order:
            cls_mask = result_mode["cls_names"] == cls_name
            if not np.any(cls_mask):
                continue
            metric_results = evaluator.run(result_mode, cls_name) if evaluator is not None else {}
            if any(m.startswith("P-AUPRO") for m in args.eval_metrics):
                masks_np, maps_np = prepare_aupro_arrays(result_mode["gt_masks"][cls_mask], result_mode["pr_masks"][cls_mask])
                precomputed = aupro_region_cache.get(cls_name)
                if precomputed is None or precomputed.get("masks_shape") != tuple(masks_np.shape):
                    precomputed = precompute_aupro_regions(masks_np)
                    aupro_region_cache[cls_name] = precomputed
                pixel_aupro = calculate_fast_aupro(
                    maps_np,
                    precomputed=precomputed,
                    max_step=args.aupro_max_step,
                    expect_fpr=args.aupro_expect_fpr,
                )
            else:
                pixel_aupro = float("nan")
            row = {
                "mode": mode,
                "target_label": cls_name,
                "image_auroc": float(metric_results.get("I-AUROC", float("nan"))),
                "image_ap": float(metric_results.get("I-AP", float("nan"))),
                "pixel_auroc": float(metric_results.get("P-AUROC", float("nan"))),
                "pixel_ap": float(metric_results.get("P-AP", float("nan"))),
                "pixel_aupro": pixel_aupro,
            }
            rows.append(row)
            image_aurocs.append(row["image_auroc"])
            image_aps.append(row["image_ap"])
            pixel_aurocs.append(row["pixel_auroc"])
            pixel_aps.append(row["pixel_ap"])
            if any(m.startswith("P-AUPRO") for m in args.eval_metrics):
                pixel_aupros.append(row["pixel_aupro"])

        rows.append(
            {
                "mode": mode,
                "target_label": "mean",
                "image_auroc": float(np.mean(image_aurocs)),
                "image_ap": float(np.mean(image_aps)),
                "pixel_auroc": float(np.mean(pixel_aurocs)),
                "pixel_ap": float(np.mean(pixel_aps)),
                "pixel_aupro": float(np.mean(pixel_aupros)) if pixel_aupros else float("nan"),
            }
        )

    summary = {
        "checkpoint_path": args.checkpoint_path,
        "source_root": args.source_root,
        "source_dataset": args.source_dataset,
        "target_root": args.target_root,
        "target_dataset": args.target_dataset,
        "class_name": args.class_name,
        "pretrained_model": args.pretrained_model,
        "checkpoint_load_mode": args.checkpoint_load_mode,
        "checkpoint_load_info": getattr(build_model, "last_checkpoint_load_info", None),
        "features_list": args.features_list,
        "image_size": args.image_size,
        "n_ctx": args.n_ctx,
        "sigma": args.sigma,
        "fusion_type": args.fusion_type,
        "bank_mining_source": args.bank_mining_source,
        "tau": args.tau,
        "fp_weight": args.fp_weight,
        "corr_alpha": args.corr_alpha,
        "map_activation": args.map_activation,
        "map_temperature": args.map_temperature,
        "trainfree_score_mode": args.trainfree_score_mode,
        "vl_gate_mode": args.vl_gate_mode,
        "vl_gate_quantile": args.vl_gate_quantile,
        "vl_gate_temp": args.vl_gate_temp,
        "ted_insertion": args.ted_insertion,
        "calib_epochs": args.calib_epochs,
        "calib_lr": args.calib_lr,
        "calib_max_train_points": args.calib_max_train_points,
        "calib_rank_margin": args.calib_rank_margin,
        "calib_rank_temp": args.calib_rank_temp,
        "calib_eta_init": args.calib_eta_init,
        "calib_eta_max": args.calib_eta_max,
        "calib_eta_reg": args.calib_eta_reg,
        "calib_fp_weight_init": args.calib_fp_weight_init,
        "calib_learn_fp_weight": args.calib_learn_fp_weight,
        "calib_hardpair_frac": args.calib_hardpair_frac,
        "calib_subspace_rank": args.calib_subspace_rank,
        "calib_preserve_host_margin_weight": args.calib_preserve_host_margin_weight,
        "calib_lambda": args.calib_lambda,
        "calib_pair_label_control": args.calib_pair_label_control,
        "calib_score_mode": args.calib_score_mode,
        "calib_apply_mode": args.calib_apply_mode,
        "calib_rankr_tau": args.calib_rankr_tau,
        "plugin_epochs": args.plugin_epochs,
        "plugin_lr": args.plugin_lr,
        "plugin_max_train_points": args.plugin_max_train_points,
        "plugin_hardpair_frac": args.plugin_hardpair_frac,
        "plugin_alpha": args.plugin_alpha,
        "plugin_loss": args.plugin_loss,
        "plugin_feature_space": args.plugin_feature_space,
        "plugin_hidden_dim": args.plugin_hidden_dim,
        "plugin_rank_margin": args.plugin_rank_margin,
        "plugin_rank_temp": args.plugin_rank_temp,
        "seed": args.seed,
        "max_eval_images": args.max_eval_images,
        "num_workers": args.num_workers,
        "eval_metrics": args.eval_metrics,
        "aupro_max_step": args.aupro_max_step,
        "aupro_expect_fpr": args.aupro_expect_fpr,
        "report_group_separability": args.report_group_separability,
        "separability_hard_frac": args.separability_hard_frac,
        "aupro_backend": "fast_threshold_cpu",
        "bank_stats": bank_stats,
        "bank_cache_path": str(bank_cache_path),
        "bank_cache_meta": bank_cache_meta,
        "tl_bank_stats": tl_bank_stats,
        "tl_bank_cache_path": str(tl_bank_cache_path) if tl_bank_cache_path is not None else None,
        "tl_bank_cache_meta": tl_bank_cache_meta,
        "source_vl_calibrator": safe_calibrator_summary(source_vl_calibrator),
        "source_tl_calibrator": safe_calibrator_summary(source_tl_calibrator),
        "source_vl_scalar_plugin": {
            kk: vv
            for kk, vv in (source_vl_plugin or {}).items()
            if kk not in {"weight", "bias", "mean", "std"}
        },
        "source_tl_scalar_plugin": {
            kk: vv
            for kk, vv in (source_tl_plugin or {}).items()
            if kk not in {"weight", "bias", "mean", "std"}
        },
        "score_definition": {
            "baseline_txt": "official AdaptCLIP zero-shot baseline",
            "ted_trainfree_vl": "train-free TED margin inserted at the requested AdaptCLIP local evidence point",
            "scalar_replace_vl": "source-only scalar readout used directly at the requested AdaptCLIP local evidence point",
            "scalar_residual_vl": "source-only scalar readout inserted as a host-anchored residual at the requested AdaptCLIP local evidence point",
            "ted_calibrated_vl": "source-calibrated TED score inserted at the requested AdaptCLIP local evidence point",
        },
    }
    if args.report_group_separability:
        summary["separability_mean"] = separability_mean
    save_results(rows, summary, save_dir)
    print(json.dumps(jsonable({"summary": summary, "rows": rows}), indent=2))


if __name__ == "__main__":
    main()
