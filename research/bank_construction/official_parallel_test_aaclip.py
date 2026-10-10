#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import average_precision_score, roc_auc_score
from skimage import measure


ROOT = Path(__file__).resolve().parents[2]
AACLP_ROOT = ROOT / "neurips2026" / "AA-CLIP"
AACLP_WEIGHT_FALLBACK = Path(
    "/mnt/data/pilab-kingjinyoung/ADPretrain/AA-CLIP/model/ViT-L-14-336px.pt"
)
BANK_PROTOCOL_VERSION = "ours_readme_v4_prescore_segtoken_backboneswap_meta_trainnormal_testanomaly_masktop_fps_v1"


PRESET_CONFIGS = {
    "visa2mvtec": {
        "source_dataset": "VisA",
        "target_dataset": "MVTec",
        "source_root": "/mnt/data/pilab-kingjinyoung/ADPretrain-CLIP/neurips2026/data/VisA_pytorch_official/1cls",
        "target_root": "/mnt/data/pilab-kingjinyoung/ijepa/data/mvtec",
        "ckpt_dir": "/mnt/data/pilab-kingjinyoung/ADPretrain/results/aaclip_ckpt/visa_fullshot_bs32_4",
    },
    "visa2mpdd": {
        "source_dataset": "VisA",
        "target_dataset": "MPDD",
        "source_root": "/mnt/data/pilab-kingjinyoung/ADPretrain-CLIP/neurips2026/data/VisA_pytorch_official/1cls",
        "target_root": "/mnt/data/pilab-kingjinyoung/ADPretrain/datasets/MPDD",
        "ckpt_dir": "/mnt/data/pilab-kingjinyoung/ADPretrain/results/aaclip_ckpt/visa_fullshot_bs32_4",
    },
    "visa2btad": {
        "source_dataset": "VisA",
        "target_dataset": "BTAD",
        "source_root": "/mnt/data/pilab-kingjinyoung/ADPretrain-CLIP/neurips2026/data/VisA_pytorch_official/1cls",
        "target_root": "/mnt/data/pilab-kingjinyoung/ADPretrain-CLIP/neurips2026/data/BTAD_official",
        "ckpt_dir": "/mnt/data/pilab-kingjinyoung/ADPretrain/results/aaclip_ckpt/visa_fullshot_bs32_4",
    },
    "mvtec2visa": {
        "source_dataset": "MVTec",
        "target_dataset": "VisA",
        "source_root": "/mnt/data/pilab-kingjinyoung/ijepa/data/mvtec",
        "target_root": "/mnt/data/pilab-kingjinyoung/ADPretrain-CLIP/neurips2026/data/VisA_pytorch_official/1cls",
        "ckpt_dir": "/mnt/data/pilab-kingjinyoung/ADPretrain/results/aaclip_ckpt/mvtec_fullshot",
    },
    "mvtec2btad": {
        "source_dataset": "MVTec",
        "target_dataset": "BTAD",
        "source_root": "/mnt/data/pilab-kingjinyoung/ijepa/data/mvtec",
        "target_root": "/mnt/data/pilab-kingjinyoung/ADPretrain-CLIP/neurips2026/data/BTAD_official",
        "ckpt_dir": "/mnt/data/pilab-kingjinyoung/ADPretrain/results/aaclip_ckpt/mvtec_fullshot",
    },
    "mvtec2mpdd": {
        "source_dataset": "MVTec",
        "target_dataset": "MPDD",
        "source_root": "/mnt/data/pilab-kingjinyoung/ijepa/data/mvtec",
        "target_root": "/mnt/data/pilab-kingjinyoung/ADPretrain/datasets/MPDD",
        "ckpt_dir": "/mnt/data/pilab-kingjinyoung/ADPretrain/results/aaclip_ckpt/mvtec_fullshot",
    },
}


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _add_aaclip_to_syspath() -> None:
    repo = str(AACLP_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def ensure_aaclip_pretrained_weight() -> None:
    local_weight = AACLP_ROOT / "model" / "ViT-L-14-336px.pt"
    if local_weight.exists():
        return
    if not AACLP_WEIGHT_FALLBACK.exists():
        raise FileNotFoundError(
            f"AA-CLIP pretrained weight not found: {local_weight} or {AACLP_WEIGHT_FALLBACK}"
        )
    local_weight.parent.mkdir(parents=True, exist_ok=True)
    if local_weight.exists() or local_weight.is_symlink():
        return
    os.symlink(AACLP_WEIGHT_FALLBACK, local_weight)


def patch_score_grid(
    patch_features: torch.Tensor,
    text_feature: torch.Tensor,
) -> torch.Tensor:
    patch_scores = 100.0 * torch.matmul(patch_features, text_feature)
    return (patch_scores[..., 1] + 1.0 - patch_scores[..., 0]) / 2.0


def spatial_tanh_zscore(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=1, keepdim=True)
    sigma = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.tanh((x - mu) / sigma)


def spatial_tanh_zscore_map(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(1)
    mu = flat.mean(dim=1, keepdim=True).view(-1, 1, 1)
    sigma = flat.std(dim=1, keepdim=True).clamp_min(1e-6).view(-1, 1, 1)
    return torch.tanh((x - mu) / sigma)


def spatial_zscore_map(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(1)
    mu = flat.mean(dim=1, keepdim=True).view(-1, 1, 1)
    sigma = flat.std(dim=1, keepdim=True).clamp_min(1e-6).view(-1, 1, 1)
    return (x - mu) / sigma


def high_score_gate_map(x: torch.Tensor, topk_frac: float, sharpness: float) -> torch.Tensor:
    flat = x.flatten(1)
    k = max(1, int(math.ceil(flat.shape[1] * topk_frac)))
    threshold = torch.topk(flat, k=k, dim=1, largest=True).values[:, -1].view(-1, 1, 1)
    return torch.sigmoid(sharpness * (x - threshold))


def high_score_hard_gate_map(x: torch.Tensor, topk_frac: float) -> torch.Tensor:
    flat = x.flatten(1)
    k = max(1, int(math.ceil(flat.shape[1] * topk_frac)))
    threshold = torch.topk(flat, k=k, dim=1, largest=True).values[:, -1].view(-1, 1, 1)
    return (x >= threshold).to(dtype=x.dtype)


def rerank_topk_preserve_values(
    baseline_map: torch.Tensor,
    rank_map: torch.Tensor,
    topk_frac: float,
) -> torch.Tensor:
    flat_base = baseline_map.flatten(1)
    flat_rank = rank_map.flatten(1)
    flat_out = flat_base.clone()
    k = max(1, int(math.ceil(flat_base.shape[1] * topk_frac)))

    top_values, top_indices = torch.topk(flat_base, k=k, dim=1, largest=True, sorted=True)
    rank_subset = torch.gather(flat_rank, dim=1, index=top_indices)
    rank_order = torch.argsort(rank_subset, dim=1, descending=True)
    reranked_indices = torch.gather(top_indices, dim=1, index=rank_order)
    flat_out.scatter_(dim=1, index=reranked_indices, src=top_values)
    return flat_out.view_as(baseline_map)


def safe_image_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.uint8).reshape(-1)
    scores = scores.astype(np.float32).reshape(-1)
    if labels.size == 0 or labels.min() == labels.max():
        return 0.0
    return float(roc_auc_score(labels, scores) * 100.0)


def safe_image_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.uint8).reshape(-1)
    scores = scores.astype(np.float32).reshape(-1)
    if labels.size == 0 or labels.min() == labels.max():
        return 0.0
    return float(average_precision_score(labels, scores) * 100.0)


def _auc_np(pos: np.ndarray, neg: np.ndarray, max_points: int = 100000) -> float:
    pos = pos.reshape(-1).astype(np.float32)
    neg = neg.reshape(-1).astype(np.float32)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    if pos.size > max_points:
        rng = np.random.default_rng(0)
        pos = pos[rng.choice(pos.size, max_points, replace=False)]
    if neg.size > max_points:
        rng = np.random.default_rng(1)
        neg = neg[rng.choice(neg.size, max_points, replace=False)]
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size, dtype=np.uint8), np.zeros(neg.size, dtype=np.uint8)])
    if scores.size == 0 or np.all(scores == scores[0]):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _sample_np(values: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    values = values.reshape(-1).astype(np.float32, copy=False)
    if values.size <= max_points:
        return values
    rng = np.random.default_rng(seed)
    idx = rng.choice(values.size, max_points, replace=False)
    return values[idx]


def _top_fraction_values(arr: np.ndarray, frac: float) -> np.ndarray:
    flat = arr.reshape(-1).astype(np.float32, copy=False)
    if flat.size == 0:
        return flat
    k = max(1, int(flat.size * float(frac)))
    idx = np.argpartition(flat, -k)[-k:]
    return flat[idx]


def align_masks_to_maps(masks: np.ndarray, maps: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    if masks.shape[-2:] == maps.shape[-2:]:
        return masks
    raise ValueError(f"Mask/map shape mismatch: masks={masks.shape}, maps={maps.shape}")


def map_group_separability(
    masks: np.ndarray,
    labels: np.ndarray,
    maps: np.ndarray,
    hard_frac: float = 0.01,
    max_points: int = 100000,
) -> dict[str, float]:
    masks = align_masks_to_maps(masks, maps).astype(bool)
    maps = maps.astype(np.float32, copy=False)
    labels = labels.astype(np.int64).reshape(-1)
    true_defect: list[np.ndarray] = []
    outside: list[np.ndarray] = []
    hard_fp: list[np.ndarray] = []
    normal: list[np.ndarray] = []
    for i in range(maps.shape[0]):
        score = maps[i]
        mask = masks[i]
        if labels[i] == 1 and mask.any():
            true_defect.append(score[mask].reshape(-1))
            outside.append(_sample_np(score[~mask].reshape(-1), max_points, i + 17))
        else:
            hard_fp.append(_top_fraction_values(score, hard_frac))
            normal.append(_sample_np(score.reshape(-1), max_points, i + 29))
    groups = {
        "true_defect": np.concatenate(true_defect, axis=0) if true_defect else np.empty((0,), dtype=np.float32),
        "outside": np.concatenate(outside, axis=0) if outside else np.empty((0,), dtype=np.float32),
        "hard_fp": np.concatenate(hard_fp, axis=0) if hard_fp else np.empty((0,), dtype=np.float32),
        "normal": np.concatenate(normal, axis=0) if normal else np.empty((0,), dtype=np.float32),
    }
    return {
        "hard_frac": float(hard_frac),
        "count_true_defect": float(groups["true_defect"].size),
        "count_hard_fp": float(groups["hard_fp"].size),
        "auc_defect_vs_hardfp": _auc_np(groups["true_defect"], groups["hard_fp"], max_points=max_points),
        "auc_defect_vs_outside": _auc_np(groups["true_defect"], groups["outside"], max_points=max_points),
        "mean_true_defect": float(np.mean(groups["true_defect"])) if groups["true_defect"].size else float("nan"),
        "mean_hard_fp": float(np.mean(groups["hard_fp"])) if groups["hard_fp"].size else float("nan"),
        "mean_outside": float(np.mean(groups["outside"])) if groups["outside"].size else float("nan"),
    }


def mean_separability(per_class: list[dict], mode: str, alpha_key: str | None = None) -> dict[str, float]:
    keys = [
        "auc_defect_vs_hardfp",
        "auc_defect_vs_outside",
        "mean_true_defect",
        "mean_hard_fp",
        "mean_outside",
    ]
    out: dict[str, float] = {}
    for key in keys:
        vals = []
        for row in per_class:
            sep = row.get("separability", {})
            if mode == "parallel":
                item = sep.get("parallel", {}).get(alpha_key, {})
            elif mode == "candidates":
                item = sep.get("candidates", {}).get(alpha_key, {})
            else:
                item = sep.get(mode, {})
            value = item.get(key) if isinstance(item, dict) else None
            if value is not None and np.isfinite(value):
                vals.append(float(value))
        out[key] = float(np.nanmean(vals)) if vals else float("nan")
    return out


def normalize_1d_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32).reshape(-1)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def topk_mean_maps_np(maps: np.ndarray, topk_frac: float) -> np.ndarray:
    flat = maps.reshape(maps.shape[0], -1).astype(np.float32)
    k = max(1, int(math.ceil(flat.shape[1] * float(topk_frac))))
    idx = np.argpartition(flat, -k, axis=1)[:, -k:]
    return np.take_along_axis(flat, idx, axis=1).mean(axis=1)


def image_scores_from_maps(
    mode: str,
    baseline_image_scores: np.ndarray,
    baseline_maps: np.ndarray,
    candidate_maps: np.ndarray,
    topk_frac: float,
) -> np.ndarray | None:
    if mode == "fused":
        return None
    baseline_image_scores = baseline_image_scores.reshape(-1).astype(np.float32)
    if mode == "baseline":
        return baseline_image_scores
    if mode == "map_max":
        return candidate_maps.reshape(candidate_maps.shape[0], -1).max(axis=1).astype(np.float32)
    if mode == "map_topk":
        return topk_mean_maps_np(candidate_maps, topk_frac)
    candidate_topk = topk_mean_maps_np(candidate_maps, topk_frac)
    baseline_topk = topk_mean_maps_np(baseline_maps, topk_frac)
    if mode == "baseline_plus_delta_topk":
        return normalize_1d_np(baseline_image_scores) + normalize_1d_np(candidate_topk - baseline_topk)
    if mode == "baseline_plus_map_topk":
        return normalize_1d_np(baseline_image_scores) + normalize_1d_np(candidate_topk)
    raise ValueError(f"Unsupported candidate_image_mode: {mode}")


def override_image_metrics_from_scores(metrics: dict, labels: np.ndarray, scores: np.ndarray | None) -> None:
    if scores is None:
        return
    metrics["image AUC"] = safe_image_auc(labels, scores)
    metrics["image AP"] = safe_image_ap(labels, scores)


def prepare_aupro_arrays(masks: np.ndarray, maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masks = masks.astype(np.float32)
    maps = maps.astype(np.float32)
    if masks.ndim == 4:
        masks = masks[:, 0]
    if maps.ndim == 4:
        maps = maps[:, 0]
    if masks.shape[0] != maps.shape[0]:
        raise ValueError(f"AUPRO image count mismatch: masks={masks.shape}, maps={maps.shape}")
    if masks.shape[-2:] != maps.shape[-2:]:
        masks_t = torch.from_numpy(masks[:, None])
        masks = F.interpolate(masks_t, size=maps.shape[-2:], mode="nearest")[:, 0].numpy()
    return (masks > 0.5).astype(np.uint8), maps.astype(np.float32)


def precompute_aupro_regions(masks: np.ndarray) -> dict:
    masks = (masks > 0.5).astype(np.uint8)
    inverse_masks = 1 - masks
    inverse_total = int(inverse_masks.sum())
    regions_by_image: list[list[np.ndarray]] = []
    for mask in masks:
        regions = []
        labeled = measure.label(mask)
        for region in measure.regionprops(labeled):
            regions.append(region.coords)
        regions_by_image.append(regions)
    return {
        "masks": masks,
        "inverse_masks": inverse_masks,
        "inverse_total": inverse_total,
        "regions_by_image": regions_by_image,
        "has_foreground": bool(masks.sum() > 0),
    }


def calculate_aupro(
    masks: np.ndarray,
    maps: np.ndarray,
    max_step: int,
    expect_fpr: float = 0.3,
) -> float:
    masks, maps = prepare_aupro_arrays(masks, maps)
    pre = precompute_aupro_regions(masks)
    if not pre["has_foreground"]:
        return 0.0
    min_th = float(np.min(maps))
    max_th = float(np.max(maps))
    if not np.isfinite(min_th) or not np.isfinite(max_th) or max_th <= min_th:
        return 50.0
    inverse_total = int(pre["inverse_total"])
    if inverse_total == 0:
        return 0.0
    delta = (max_th - min_th) / max(1, int(max_step))
    pros: list[float] = []
    fprs: list[float] = []
    for th in np.arange(min_th, max_th, delta):
        binary_maps = maps > th
        pro_vals = []
        for binary_map, regions in zip(binary_maps, pre["regions_by_image"]):
            for coords in regions:
                tp_pixels = binary_map[coords[:, 0], coords[:, 1]].sum()
                pro_vals.append(float(tp_pixels / max(coords.shape[0], 1)))
        if not pro_vals:
            continue
        fp_pixels = np.logical_and(pre["inverse_masks"], binary_maps).sum()
        pros.append(float(np.mean(pro_vals)))
        fprs.append(float(fp_pixels / inverse_total))
    if len(pros) <= 2:
        return 50.0
    pros_np = np.asarray(pros, dtype=np.float32)
    fprs_np = np.asarray(fprs, dtype=np.float32)
    keep = fprs_np < float(expect_fpr)
    if int(keep.sum()) <= 2:
        return 50.0
    fprs_keep = fprs_np[keep]
    pros_keep = pros_np[keep]
    order = np.argsort(fprs_keep)
    fprs_keep = fprs_keep[order]
    pros_keep = pros_keep[order]
    denom = float(fprs_keep.max() - fprs_keep.min())
    if denom <= 1e-12:
        return 50.0
    fprs_keep = (fprs_keep - fprs_keep.min()) / denom
    return float(np.trapezoid(pros_keep, fprs_keep) * 100.0)


def maybe_add_pixel_pro(metrics: dict, masks: np.ndarray, maps: np.ndarray, cal_pro: bool, max_step: int) -> None:
    if not cal_pro:
        metrics.setdefault("pixel PRO", 0.0)
        return
    metrics["pixel PRO"] = calculate_aupro(masks, maps, max_step=max_step)


def farthest_point_subsample(feat: torch.Tensor, max_points: int) -> torch.Tensor:
    if feat.shape[0] <= max_points:
        return feat
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
    return feat[torch.tensor(selected, dtype=torch.long)]


def logmeanexp_negative_sqdist_1d(
    query_z: torch.Tensor,
    bank_z: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    if bank_z.numel() == 0:
        return torch.full_like(query_z, fill_value=-1e6)
    diff2 = (query_z.unsqueeze(-1) - bank_z.view(1, 1, -1)).pow(2)
    return torch.logsumexp(-diff2 / tau, dim=-1) - math.log(bank_z.numel())


def logmeanexp_cosine_support(
    query: torch.Tensor,
    bank: torch.Tensor,
    tau: float,
    chunk: int = 512,
) -> torch.Tensor:
    if bank.numel() == 0:
        return torch.full(query.shape[:2], -1e6, device=query.device, dtype=query.dtype)
    query = F.normalize(query.float(), dim=-1)
    bank = F.normalize(bank.float(), dim=-1)
    running = None
    count = 0
    for start in range(0, bank.shape[0], chunk):
        part = bank[start : start + chunk].to(query.device)
        sim = query @ part.t()
        val = torch.logsumexp(sim / tau, dim=-1)
        running = val if running is None else torch.logaddexp(running, val)
        count += int(part.shape[0])
    return running - math.log(max(count, 1))


def build_plugin_features_from_scores(
    query_proj: torch.Tensor,
    q_def: torch.Tensor,
    q_fp: torch.Tensor,
) -> torch.Tensor:
    return torch.stack([query_proj, q_def, q_fp, q_def - q_fp, q_def + q_fp], dim=-1)


def apply_plugin_from_scores(
    query_proj: torch.Tensor,
    q_def: torch.Tensor,
    q_fp: torch.Tensor,
    plugin: Dict[str, torch.Tensor],
) -> torch.Tensor:
    if plugin.get("plugin_type", "linear") == "support_affine":
        defect_weight = float(plugin.get("defect_weight", 1.0))
        fp_weight = float(plugin.get("fp_weight", 1.0))
        query_weight = float(plugin.get("query_weight", 0.0))
        bias = float(plugin.get("bias", 0.0))
        return query_weight * query_proj.float() + defect_weight * q_def.float() - fp_weight * q_fp.float() + bias
    if plugin.get("feature_space") == "hostscore":
        feats = query_proj.unsqueeze(-1)
    else:
        feats = build_plugin_features_from_scores(query_proj, q_def, q_fp)
    mean = plugin["mean"].to(query_proj.device)
    std = plugin["std"].to(query_proj.device)
    x = ((feats - mean) / std).float()
    if plugin.get("plugin_type", "linear") == "rank_mlp":
        w1 = plugin["w1"].to(query_proj.device)
        b1 = plugin["b1"].to(query_proj.device)
        w2 = plugin["w2"].to(query_proj.device)
        b2 = plugin["b2"].to(query_proj.device)
        hidden = torch.tanh(x @ w1 + b1)
        return hidden @ w2 + b2
    weight = plugin["weight"].to(query_proj.device)
    bias = plugin["bias"].to(query_proj.device)
    return x @ weight + bias


def train_one_layer_plugin(
    fp: torch.Tensor,
    defect: torch.Tensor,
    axis: torch.Tensor,
    tau: float,
    max_train_points: int,
    epochs: int,
    lr: float,
    loss_type: str,
    margin: float,
    rank_temp: float,
    contrast_weight: float,
    contrast_margin: float,
    hidden_dim: int,
    learn_alpha: bool,
    alpha_init: float,
    alpha_max: float,
    alpha_reg: float,
    hardpair_frac: float,
    seed: int,
    device: torch.device,
) -> Dict[str, object] | None:
    if epochs <= 0 or fp.numel() == 0 or defect.numel() == 0:
        return None
    fp = F.normalize(fp.float(), dim=-1)
    defect = F.normalize(defect.float(), dim=-1)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    n = min(fp.shape[0], defect.shape[0], max_train_points // 2 if max_train_points > 0 else 10**9)
    if n <= 0:
        return None
    axis = axis.to(device)
    fp_proj = fp.to(device) @ axis
    def_proj = defect.to(device) @ axis
    hardpair_frac = float(np.clip(float(hardpair_frac), 0.0, 1.0))
    n_hard = int(round(n * hardpair_frac))
    n_rand = max(0, n - n_hard)

    def mixed_indices(scores: torch.Tensor, hard_high: bool, total: int) -> torch.Tensor:
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
        if idx.numel() < total:
            fill = torch.randperm(scores_cpu.numel(), generator=rng)[: total - idx.numel()]
            idx = torch.cat([idx, fill], dim=0)
        return idx[:total]

    # Hard pairs are the source patches the base text axis already confuses:
    # high-scoring normal FP patches and low-scoring defect patches.
    fp_idx = mixed_indices(fp_proj, hard_high=True, total=n)
    def_idx = mixed_indices(def_proj, hard_high=False, total=n)
    fp_train = fp[fp_idx].to(device)
    def_train = defect[def_idx].to(device)

    fp_q = fp_train @ axis
    def_q = def_train @ axis
    x_fp = build_plugin_features_from_scores(
        fp_q,
        logmeanexp_negative_sqdist_1d(fp_q.unsqueeze(0), def_proj, tau=tau)[0],
        logmeanexp_negative_sqdist_1d(fp_q.unsqueeze(0), fp_proj, tau=tau)[0],
    )
    x_def = build_plugin_features_from_scores(
        def_q,
        logmeanexp_negative_sqdist_1d(def_q.unsqueeze(0), def_proj, tau=tau)[0],
        logmeanexp_negative_sqdist_1d(def_q.unsqueeze(0), fp_proj, tau=tau)[0],
    )
    if loss_type.lower() == "host_rank":
        x_fp = fp_q.unsqueeze(-1)
        x_def = def_q.unsqueeze(-1)
    x = torch.cat([x_fp, x_def], dim=0).float()
    y = torch.cat([torch.zeros(x_fp.shape[0], device=device), torch.ones(x_def.shape[0], device=device)], dim=0)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    xz_fp = (x_fp.float() - mean) / std
    xz_def = (x_def.float() - mean) / std
    xz = torch.cat([xz_fp, xz_def], dim=0)

    loss_type = loss_type.lower()
    rank_temp = max(float(rank_temp), 1e-6)
    alpha_max = max(float(alpha_max), 1e-6)
    alpha_init = float(np.clip(float(alpha_init), -0.99 * alpha_max, 0.99 * alpha_max))

    def init_alpha_raw() -> torch.Tensor:
        ratio = alpha_init / alpha_max
        raw = 0.5 * math.log((1.0 + ratio) / max(1.0 - ratio, 1e-6))
        return torch.tensor(raw, device=device, requires_grad=True)

    def bounded_alpha(alpha_raw: torch.Tensor) -> torch.Tensor:
        return alpha_max * torch.tanh(alpha_raw)

    train_loss_type = "pair_rank" if loss_type == "host_rank" else loss_type

    if train_loss_type == "bce":
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
            acc = ((logit.sigmoid() >= 0.5).float() == y).float().mean().item()
            rank_acc = ((xz_def @ weight + bias) > (xz_fp @ weight + bias)).float().mean().item()
            pos_mean = (xz_def @ weight + bias).mean().item()
            neg_mean = (xz_fp @ weight + bias).mean().item()
        plugin = {
            "plugin_type": "linear",
            "weight": weight.detach(),
            "bias": bias.detach(),
            "learned_alpha": torch.tensor(1.0),
        }
    elif train_loss_type == "support_rank":
        defect_raw = torch.zeros((), device=device, requires_grad=True)
        fp_raw = torch.zeros((), device=device, requires_grad=True)
        query_weight = torch.zeros((), device=device, requires_grad=True)
        bias = torch.zeros((), device=device, requires_grad=True)
        opt = torch.optim.Adam([defect_raw, fp_raw, query_weight, bias], lr=lr)

        def score_from_feats(feats: torch.Tensor) -> torch.Tensor:
            defect_weight = F.softplus(defect_raw)
            fp_weight = F.softplus(fp_raw)
            return query_weight * feats[..., 0] + defect_weight * feats[..., 1] - fp_weight * feats[..., 2] + bias

        pair_index = torch.arange(n, device=device)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            perm = pair_index[torch.randperm(n, device=device)]
            s_fp = score_from_feats(x_fp.float()[perm])
            s_def = score_from_feats(x_def.float())
            loss = F.softplus((s_fp - s_def + margin) / rank_temp).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            final_fp = score_from_feats(x_fp.float())
            final_def = score_from_feats(x_def.float())
            logit = torch.cat([final_fp, final_def], dim=0)
            acc = ((logit >= logit.median()).float() == y).float().mean().item()
            rank_acc = (final_def > final_fp).float().mean().item()
            pos_mean = final_def.mean().item()
            neg_mean = final_fp.mean().item()
        plugin = {
            "plugin_type": "support_affine",
            "defect_weight": torch.tensor(float(F.softplus(defect_raw).item())),
            "fp_weight": torch.tensor(float(F.softplus(fp_raw).item())),
            "query_weight": torch.tensor(float(query_weight.item())),
            "bias": torch.tensor(float(bias.item())),
            "learned_alpha": torch.tensor(1.0),
        }
    elif train_loss_type in {"pair_rank", "rank_contrast"}:
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            hidden_dim = 16
        w1 = torch.empty(xz.shape[1], hidden_dim, device=device, requires_grad=True)
        b1 = torch.zeros(hidden_dim, device=device, requires_grad=True)
        w2 = torch.empty(hidden_dim, device=device, requires_grad=True)
        b2 = torch.zeros((), device=device, requires_grad=True)
        torch.nn.init.xavier_uniform_(w1)
        torch.nn.init.normal_(w2, mean=0.0, std=0.02)
        alpha_raw = init_alpha_raw() if learn_alpha else None

        def forward_rank(x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = torch.tanh(x_in @ w1 + b1)
            score = hidden @ w2 + b2
            z = F.normalize(hidden, dim=-1)
            return score, z

        params = [w1, b1, w2, b2]
        if alpha_raw is not None:
            params.append(alpha_raw)
        opt = torch.optim.Adam(params, lr=lr)
        pair_index = torch.arange(n, device=device)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            perm = pair_index[torch.randperm(n, device=device)]
            s_fp, z_fp = forward_rank(xz_fp[perm])
            s_def, z_def = forward_rank(xz_def)
            if alpha_raw is not None:
                alpha = bounded_alpha(alpha_raw)
                # Source-side analogue of target fusion: preserve the base text-axis
                # score unless the learned correction improves defect-vs-hardFP ranking.
                final_fp = xz_fp[perm, 0] + alpha * s_fp
                final_def = xz_def[:, 0] + alpha * s_def
            else:
                alpha = torch.tensor(1.0, device=device)
                final_fp = s_fp
                final_def = s_def
            loss_rank = F.softplus((final_fp - final_def + margin) / rank_temp).mean()
            loss = loss_rank
            if train_loss_type == "rank_contrast" and contrast_weight > 0:
                roll = torch.roll(pair_index, shifts=1)
                neg_perm = pair_index[torch.randperm(n, device=device)]
                pos_def = (z_def * z_def[roll]).sum(dim=-1)
                pos_fp = (z_fp * z_fp[roll]).sum(dim=-1)
                neg_def = (z_def * z_fp[neg_perm]).sum(dim=-1)
                neg_fp = (z_fp * z_def[neg_perm]).sum(dim=-1)
                loss_contrast = 0.5 * (
                    F.softplus((neg_def - pos_def + contrast_margin) / rank_temp).mean()
                    + F.softplus((neg_fp - pos_fp + contrast_margin) / rank_temp).mean()
                )
                loss = loss + contrast_weight * loss_contrast
            if alpha_raw is not None and alpha_reg > 0:
                loss = loss + float(alpha_reg) * alpha.pow(2)
            loss.backward()
            opt.step()

        with torch.no_grad():
            s_fp, _ = forward_rank(xz_fp)
            s_def, _ = forward_rank(xz_def)
            if alpha_raw is not None:
                alpha = bounded_alpha(alpha_raw)
                final_fp = xz_fp[:, 0] + alpha * s_fp
                final_def = xz_def[:, 0] + alpha * s_def
                learned_alpha = float(alpha.item())
            else:
                final_fp = s_fp
                final_def = s_def
                learned_alpha = 1.0
            logit = torch.cat([final_fp, final_def], dim=0)
            acc = ((logit >= logit.median()).float() == y).float().mean().item()
            rank_acc = (final_def > final_fp).float().mean().item()
            pos_mean = final_def.mean().item()
            neg_mean = final_fp.mean().item()
        plugin = {
            "plugin_type": "rank_mlp",
            "w1": w1.detach(),
            "b1": b1.detach(),
            "w2": w2.detach(),
            "b2": b2.detach(),
            "learned_alpha": torch.tensor(learned_alpha),
        }
    else:
        raise ValueError(f"Unknown plugin_loss: {loss_type}")

    plugin.update({
        "mean": mean.detach(),
        "std": std.detach(),
        "train_points_per_class": int(n),
        "train_accuracy": float(acc),
        "source_pair_rank_accuracy": float(rank_acc),
        "source_defect_logit_mean": float(pos_mean),
        "source_hardfp_logit_mean": float(neg_mean),
        "loss_type": loss_type,
        "rank_margin": float(margin),
        "rank_temp": float(rank_temp),
        "contrast_weight": float(contrast_weight),
        "contrast_margin": float(contrast_margin),
        "hidden_dim": int(hidden_dim) if train_loss_type in {"pair_rank", "rank_contrast"} else 0,
        "learn_alpha": bool(learn_alpha),
        "alpha_init": float(alpha_init),
        "alpha_max": float(alpha_max),
        "alpha_reg": float(alpha_reg),
        "hardpair_frac": float(hardpair_frac),
        "learned_alpha": float(plugin["learned_alpha"].item()),
        "feature_space": "hostscore" if loss_type == "host_rank" else "support",
    })
    return plugin


def train_layer_plugins(
    banks: Dict[str, List[torch.Tensor]],
    axis: torch.Tensor,
    tau: float,
    max_train_points: int,
    epochs: int,
    lr: float,
    loss_type: str,
    margin: float,
    rank_temp: float,
    contrast_weight: float,
    contrast_margin: float,
    hidden_dim: int,
    learn_alpha: bool,
    alpha_init: float,
    alpha_max: float,
    alpha_reg: float,
    hardpair_frac: float,
    seed: int,
    device: torch.device,
) -> List[Dict[str, object] | None]:
    plugins: List[Dict[str, object] | None] = []
    for li, (fp, defect) in enumerate(zip(banks["fp"], banks["defect"])):
        plugins.append(
            train_one_layer_plugin(
                fp=fp,
                defect=defect,
                axis=axis,
                tau=tau,
                max_train_points=max_train_points,
                epochs=epochs,
                lr=lr,
                loss_type=loss_type,
                margin=margin,
                rank_temp=rank_temp,
                contrast_weight=contrast_weight,
                contrast_margin=contrast_margin,
                hidden_dim=hidden_dim,
                learn_alpha=learn_alpha,
                alpha_init=alpha_init,
                alpha_max=alpha_max,
                alpha_reg=alpha_reg,
                hardpair_frac=hardpair_frac,
                seed=seed + 100 * li,
                device=device,
            )
        )
    return plugins


def mean_text_anomaly_axis(text_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
    axes = []
    for text_feature in text_embeddings.values():
        axes.append(F.normalize(text_feature[:, 1] - text_feature[:, 0], dim=0))
    if not axes:
        raise ValueError("No text embeddings available to build a source-global anomaly axis")
    return F.normalize(torch.stack(axes, dim=0).mean(dim=0), dim=0)


def prescore_rectify_seg_tokens(
    tokens: torch.Tensor,
    q_def: torch.Tensor,
    q_fp: torch.Tensor,
    target_axis: torch.Tensor,
    alpha: float,
    fp_weight: float,
    gate_sharpness: float,
) -> torch.Tensor:
    margin = spatial_tanh_zscore(q_def - fp_weight * q_fp)
    support_mass = spatial_tanh_zscore(q_def + q_fp)
    gate = torch.sigmoid(float(gate_sharpness) * support_mass)
    target_axis = F.normalize(target_axis.to(tokens.device).float(), dim=0)
    step = gate.unsqueeze(-1) * margin.unsqueeze(-1) * target_axis.view(1, 1, -1)
    return F.normalize(tokens.float() + float(alpha) * step, dim=-1)


def prescore_parallel_transport_seg_tokens(
    tokens: torch.Tensor,
    q_def: torch.Tensor,
    q_fp: torch.Tensor,
    target_axis: torch.Tensor,
    eta: float,
    transport_w: float,
    transport_b: float,
    fp_weight: float,
) -> torch.Tensor:
    raw_margin = q_def - float(fp_weight) * q_fp
    margin = spatial_tanh_zscore(raw_margin)
    delta = float(eta) * torch.tanh(float(transport_w) * margin + float(transport_b))
    target_axis = F.normalize(target_axis.to(tokens.device).float(), dim=0)
    step = delta.unsqueeze(-1) * target_axis.view(1, 1, -1)
    return F.normalize(tokens.float() + step, dim=-1)


def prescore_subspace_transport_seg_tokens(
    tokens: torch.Tensor,
    q_def: torch.Tensor,
    q_fp: torch.Tensor,
    basis: torch.Tensor,
    eta: float,
    direction: Sequence[float],
    transport_b: float,
    fp_weight: float,
) -> torch.Tensor:
    basis = basis.to(tokens.device).float()
    direction_t = F.normalize(torch.tensor(direction, device=tokens.device, dtype=torch.float32), dim=0)
    move_dir = F.normalize(basis @ direction_t, dim=0)
    raw_margin = q_def - float(fp_weight) * q_fp
    margin = spatial_tanh_zscore(raw_margin)
    delta = float(eta) * torch.tanh(margin + float(transport_b))
    return F.normalize(tokens.float() + delta.unsqueeze(-1) * move_dir.view(1, 1, -1), dim=-1)


def subspace_score_map_from_tokens(
    tokens: torch.Tensor,
    basis: torch.Tensor,
    score_w: Sequence[float],
    img_size: int,
    normalize_mode: str,
) -> torch.Tensor:
    basis = basis.to(tokens.device).float()
    score_w_t = F.normalize(torch.tensor(score_w, device=tokens.device, dtype=torch.float32), dim=0)
    score = (tokens.float() @ basis) @ score_w_t
    if normalize_mode == "subspace_zscore":
        score = spatial_tanh_zscore(score)
    side = int(round(math.sqrt(tokens.shape[1])))
    score_map = score.view(tokens.shape[0], 1, side, side)
    score_map = F.interpolate(
        score_map,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=True,
    )
    return score_map[:, 0]


def subspace_host_residual_map_from_tokens(
    tokens: torch.Tensor,
    rect_tokens: torch.Tensor,
    basis: torch.Tensor,
    score_w: Sequence[float],
    target_axis: torch.Tensor,
    readout_gamma: float,
    img_size: int,
    normalize_mode: str,
) -> torch.Tensor:
    basis = basis.to(tokens.device).float()
    score_w_t = F.normalize(torch.tensor(score_w, device=tokens.device, dtype=torch.float32), dim=0)
    target_axis = F.normalize(target_axis.to(tokens.device).float(), dim=0)
    host_score = tokens.float() @ target_axis
    coords = rect_tokens.float() @ basis
    if score_w_t.numel() == max(0, coords.shape[-1] - 1):
        residual = coords[..., 1:] @ score_w_t
    else:
        residual = coords @ score_w_t - coords[..., 0]
    score = host_score + float(readout_gamma) * residual
    if normalize_mode == "host_residual_zscore":
        score = spatial_tanh_zscore(score)
    side = int(round(math.sqrt(tokens.shape[1])))
    score_map = score.view(tokens.shape[0], 1, side, side)
    score_map = F.interpolate(
        score_map,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=True,
    )
    return score_map[:, 0]


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


def train_one_layer_prescore_calibrator(
    fp: torch.Tensor,
    defect: torch.Tensor,
    axis: torch.Tensor,
    tau: float,
    max_train_points: int,
    epochs: int,
    lr: float,
    rank_margin: float,
    rank_temp: float,
    alpha_init: float,
    alpha_max: float,
    alpha_reg: float,
    fp_weight_init: float,
    gate_sharpness_init: float,
    learn_fp_weight: bool,
    learn_gate_sharpness: bool,
    nonnegative_alpha: bool,
    hardpair_frac: float,
    seed: int,
    device: torch.device,
) -> Dict[str, object] | None:
    if epochs <= 0 or fp.numel() == 0 or defect.numel() == 0:
        return None
    fp = F.normalize(fp.float(), dim=-1)
    defect = F.normalize(defect.float(), dim=-1)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    n = min(fp.shape[0], defect.shape[0], max_train_points // 2 if max_train_points > 0 else 10**9)
    if n <= 0:
        return None

    axis = F.normalize(axis.to(device).float(), dim=0)
    fp_all = fp.to(device)
    def_all = defect.to(device)
    fp_proj_all = fp_all @ axis
    def_proj_all = def_all @ axis
    fp_idx, def_idx = sample_source_hardpair_indices(
        fp_scores=fp_proj_all,
        defect_scores=def_proj_all,
        n=n,
        hardpair_frac=hardpair_frac,
        rng=rng,
    )
    fp_train = fp[fp_idx].to(device)
    def_train = defect[def_idx].to(device)
    fp_q_axis = fp_train @ axis
    def_q_axis = def_train @ axis
    fp_q_def = logmeanexp_negative_sqdist_1d(fp_q_axis.unsqueeze(0), def_proj_all, tau=tau)[0]
    fp_q_fp = logmeanexp_negative_sqdist_1d(fp_q_axis.unsqueeze(0), fp_proj_all, tau=tau)[0]
    def_q_def = logmeanexp_negative_sqdist_1d(def_q_axis.unsqueeze(0), def_proj_all, tau=tau)[0]
    def_q_fp = logmeanexp_negative_sqdist_1d(def_q_axis.unsqueeze(0), fp_proj_all, tau=tau)[0]

    alpha_max = max(float(alpha_max), 1e-6)
    if nonnegative_alpha:
        alpha_init = float(np.clip(float(alpha_init), 1e-6, 0.99 * alpha_max))
        alpha_raw = torch.tensor(
            inverse_softplus(alpha_init),
            device=device,
            requires_grad=True,
        )
    else:
        alpha_init = float(np.clip(float(alpha_init), -0.99 * alpha_max, 0.99 * alpha_max))
        ratio = alpha_init / alpha_max
        alpha_raw = torch.tensor(
            0.5 * math.log((1.0 + ratio) / max(1.0 - ratio, 1e-6)),
            device=device,
            requires_grad=True,
        )
    fp_weight_raw = torch.tensor(
        inverse_softplus(fp_weight_init),
        device=device,
        requires_grad=bool(learn_fp_weight),
    )
    gate_raw = torch.tensor(
        inverse_softplus(gate_sharpness_init),
        device=device,
        requires_grad=bool(learn_gate_sharpness),
    )
    params = [alpha_raw]
    if fp_weight_raw.requires_grad:
        params.append(fp_weight_raw)
    if gate_raw.requires_grad:
        params.append(gate_raw)
    opt = torch.optim.Adam(params, lr=lr)
    pair_index = torch.arange(n, device=device)
    rank_temp = max(float(rank_temp), 1e-6)

    def current_params() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = torch.clamp(F.softplus(alpha_raw), max=alpha_max) if nonnegative_alpha else alpha_max * torch.tanh(alpha_raw)
        fp_weight = F.softplus(fp_weight_raw) + 1e-6
        gate_sharpness = F.softplus(gate_raw) + 1e-6
        return alpha, fp_weight, gate_sharpness

    def rectified_axis_scores(
        features: torch.Tensor,
        q_def: torch.Tensor,
        q_fp: torch.Tensor,
        alpha: torch.Tensor,
        fp_weight: torch.Tensor,
        gate_sharpness: torch.Tensor,
    ) -> torch.Tensor:
        raw_margin = q_def - fp_weight * q_fp
        raw_mass = q_def + q_fp
        margin = torch.tanh((raw_margin - raw_margin.mean()) / raw_margin.std().clamp_min(1e-6))
        support_mass = torch.tanh((raw_mass - raw_mass.mean()) / raw_mass.std().clamp_min(1e-6))
        gate = torch.sigmoid(gate_sharpness * support_mass)
        step = gate.unsqueeze(-1) * margin.unsqueeze(-1) * axis.view(1, -1)
        rect = F.normalize(features.float() + alpha * step, dim=-1)
        return rect @ axis

    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        perm = pair_index[torch.randperm(n, device=device)]
        alpha, fp_weight, gate_sharpness = current_params()
        features = torch.cat([fp_train[perm], def_train], dim=0)
        q_def = torch.cat([fp_q_def[perm], def_q_def], dim=0)
        q_fp = torch.cat([fp_q_fp[perm], def_q_fp], dim=0)
        scores = rectified_axis_scores(features, q_def, q_fp, alpha, fp_weight, gate_sharpness)
        s_fp, s_def = scores[:n], scores[n:]
        loss = F.softplus((s_fp - s_def + rank_margin) / rank_temp).mean()
        if alpha_reg > 0:
            loss = loss + float(alpha_reg) * alpha.pow(2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        alpha, fp_weight, gate_sharpness = current_params()
        features = torch.cat([fp_train, def_train], dim=0)
        q_def = torch.cat([fp_q_def, def_q_def], dim=0)
        q_fp = torch.cat([fp_q_fp, def_q_fp], dim=0)
        scores = rectified_axis_scores(features, q_def, q_fp, alpha, fp_weight, gate_sharpness)
        s_fp, s_def = scores[:n], scores[n:]
        pair_acc = (s_def > s_fp).float().mean().item()
        base_pair_acc = (def_q_axis > fp_q_axis).float().mean().item()
    return {
        "alpha": float(alpha.item()),
        "fp_weight": float(fp_weight.item()),
        "gate_sharpness": float(gate_sharpness.item()),
        "train_points_per_class": int(n),
        "source_pair_rank_accuracy": float(pair_acc),
        "source_base_pair_rank_accuracy": float(base_pair_acc),
        "nonnegative_alpha": bool(nonnegative_alpha),
        "hardpair_frac": float(hardpair_frac),
    }


def train_one_layer_parallel_transport_calibrator(
    fp: torch.Tensor,
    defect: torch.Tensor,
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
    seed: int,
    device: torch.device,
) -> Dict[str, object] | None:
    if epochs <= 0 or fp.numel() == 0 or defect.numel() == 0:
        return None
    fp = F.normalize(fp.float(), dim=-1)
    defect = F.normalize(defect.float(), dim=-1)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    n = min(fp.shape[0], defect.shape[0], max_train_points // 2 if max_train_points > 0 else 10**9)
    if n <= 0:
        return None

    axis = F.normalize(axis.to(device).float(), dim=0)
    fp_all = fp.to(device)
    def_all = defect.to(device)
    fp_proj_all = fp_all @ axis
    def_proj_all = def_all @ axis
    fp_idx, def_idx = sample_source_hardpair_indices(
        fp_scores=fp_proj_all,
        defect_scores=def_proj_all,
        n=n,
        hardpair_frac=hardpair_frac,
        rng=rng,
    )
    fp_train = fp[fp_idx].to(device)
    def_train = defect[def_idx].to(device)
    fp_q_axis = fp_train @ axis
    def_q_axis = def_train @ axis
    fp_q_def = logmeanexp_negative_sqdist_1d(fp_q_axis.unsqueeze(0), def_proj_all, tau=tau)[0]
    fp_q_fp = logmeanexp_negative_sqdist_1d(fp_q_axis.unsqueeze(0), fp_proj_all, tau=tau)[0]
    def_q_def = logmeanexp_negative_sqdist_1d(def_q_axis.unsqueeze(0), def_proj_all, tau=tau)[0]
    def_q_fp = logmeanexp_negative_sqdist_1d(def_q_axis.unsqueeze(0), fp_proj_all, tau=tau)[0]

    eta_max = max(float(eta_max), 1e-6)
    eta_init = float(np.clip(abs(float(eta_init)), 1e-6, 0.99 * eta_max))
    eta_raw = torch.tensor(inverse_softplus(eta_init), device=device, requires_grad=True)
    transport_w = torch.tensor(1.0, device=device, requires_grad=True)
    transport_b = torch.tensor(0.0, device=device, requires_grad=True)
    fp_weight_raw = torch.tensor(
        inverse_softplus(fp_weight_init),
        device=device,
        requires_grad=bool(learn_fp_weight),
    )
    params = [eta_raw, transport_w, transport_b]
    if fp_weight_raw.requires_grad:
        params.append(fp_weight_raw)
    opt = torch.optim.Adam(params, lr=lr)
    pair_index = torch.arange(n, device=device)
    rank_temp = max(float(rank_temp), 1e-6)

    def current_params() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eta = torch.clamp(F.softplus(eta_raw), max=eta_max)
        fp_weight = F.softplus(fp_weight_raw) + 1e-6
        return eta, transport_w, transport_b, fp_weight

    def transport_scores(
        features: torch.Tensor,
        q_def: torch.Tensor,
        q_fp: torch.Tensor,
        eta: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        fp_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_margin = q_def - fp_weight * q_fp
        margin = torch.tanh((raw_margin - raw_margin.mean()) / raw_margin.std().clamp_min(1e-6))
        delta = eta * torch.tanh(w * margin + b)
        rect = F.normalize(features.float() + delta.unsqueeze(-1) * axis.view(1, -1), dim=-1)
        return rect @ axis, delta

    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        perm = pair_index[torch.randperm(n, device=device)]
        eta, w, b, fp_weight = current_params()
        features = torch.cat([fp_train[perm], def_train], dim=0)
        q_def = torch.cat([fp_q_def[perm], def_q_def], dim=0)
        q_fp = torch.cat([fp_q_fp[perm], def_q_fp], dim=0)
        scores, _ = transport_scores(features, q_def, q_fp, eta, w, b, fp_weight)
        s_fp, s_def = scores[:n], scores[n:]
        loss = F.softplus((s_fp - s_def + rank_margin) / rank_temp).mean()
        if eta_reg > 0:
            loss = loss + float(eta_reg) * eta.pow(2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        eta, w, b, fp_weight = current_params()
        features = torch.cat([fp_train, def_train], dim=0)
        q_def = torch.cat([fp_q_def, def_q_def], dim=0)
        q_fp = torch.cat([fp_q_fp, def_q_fp], dim=0)
        scores, delta = transport_scores(features, q_def, q_fp, eta, w, b, fp_weight)
        s_fp, s_def = scores[:n], scores[n:]
        pair_acc = (s_def > s_fp).float().mean().item()
        base_pair_acc = (def_q_axis > fp_q_axis).float().mean().item()
        delta_fp = delta[:n].mean().item()
        delta_def = delta[n:].mean().item()
    return {
        "alpha": float(eta.item()),
        "effective_alpha": float(eta.item()),
        "eta": float(eta.item()),
        "transport_w": float(w.item()),
        "transport_b": float(b.item()),
        "fp_weight": float(fp_weight.item()),
        "gate_sharpness": 0.0,
        "transport_mode": "parallel_tanh",
        "train_points_per_class": int(n),
        "source_pair_rank_accuracy": float(pair_acc),
        "source_base_pair_rank_accuracy": float(base_pair_acc),
        "source_delta_fp_mean": float(delta_fp),
        "source_delta_def_mean": float(delta_def),
        "nonnegative_alpha": True,
        "hardpair_frac": float(hardpair_frac),
    }


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
            fp_s = fp[:n]
            def_s = defect[:n]
            pair_diff = def_s - fp_s
            pair_diff = pair_diff - pair_diff.mean(dim=0, keepdim=True)
            try:
                _, _, vh = torch.linalg.svd(pair_diff.float(), full_matrices=False)
                for pc in vh[: max(0, rank - len(cols))]:
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


def apply_pair_label_control(
    fp: torch.Tensor,
    defect: torch.Tensor,
    mode: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "correct":
        return fp, defect
    if mode == "swap":
        return defect, fp
    if mode == "shuffle":
        n = min(fp.shape[0], defect.shape[0])
        if n <= 0:
            return fp, defect
        merged = torch.cat([fp[:n], defect[:n]], dim=0)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(int(seed))
        perm = torch.randperm(merged.shape[0], generator=rng)
        return merged[perm[:n]], merged[perm[n : 2 * n]]
    raise ValueError(f"Unsupported pair label control: {mode}")


def train_one_layer_subspace_transport_calibrator(
    fp: torch.Tensor,
    defect: torch.Tensor,
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
    subspace_readout_mode: str,
    subspace_basis_control: str,
    preserve_host_margin_weight: float,
    seed: int,
    device: torch.device,
) -> Dict[str, object] | None:
    if epochs <= 0 or fp.numel() == 0 or defect.numel() == 0:
        return None
    fp = F.normalize(fp.float(), dim=-1)
    defect = F.normalize(defect.float(), dim=-1)
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
    def_all = defect.to(device)
    fp_proj_all = fp_all @ axis
    def_proj_all = def_all @ axis
    fp_idx, def_idx = sample_source_hardpair_indices(
        fp_scores=fp_proj_all,
        defect_scores=def_proj_all,
        n=n,
        hardpair_frac=hardpair_frac,
        rng=rng,
    )
    fp_train = fp[fp_idx].to(device)
    def_train = defect[def_idx].to(device)
    fp_q_axis = fp_train @ axis
    def_q_axis = def_train @ axis
    fp_q_def = logmeanexp_negative_sqdist_1d(fp_q_axis.unsqueeze(0), def_proj_all, tau=tau)[0]
    fp_q_fp = logmeanexp_negative_sqdist_1d(fp_q_axis.unsqueeze(0), fp_proj_all, tau=tau)[0]
    def_q_def = logmeanexp_negative_sqdist_1d(def_q_axis.unsqueeze(0), def_proj_all, tau=tau)[0]
    def_q_fp = logmeanexp_negative_sqdist_1d(def_q_axis.unsqueeze(0), fp_proj_all, tau=tau)[0]

    eta_max = max(float(eta_max), 1e-6)
    eta_init = float(np.clip(abs(float(eta_init)), 1e-6, 0.99 * eta_max))
    eta_raw = torch.tensor(inverse_softplus(eta_init), device=device, requires_grad=True)
    fp_weight_raw = torch.tensor(
        inverse_softplus(fp_weight_init),
        device=device,
        requires_grad=bool(learn_fp_weight),
    )
    raw_w = torch.zeros(basis.shape[1], device=device, requires_grad=True)
    raw_w.data[0] = 1.0
    transport_b = torch.tensor(0.0, device=device, requires_grad=True)
    residual_readout = subspace_readout_mode == "host_residual" and basis.shape[1] > 1
    scorer_dim = basis.shape[1] - 1 if residual_readout else basis.shape[1]
    scorer = torch.zeros(scorer_dim, device=device, requires_grad=True)
    scorer.data[0] = 1.0
    readout_gamma_raw = torch.tensor(inverse_softplus(eta_init), device=device, requires_grad=True)
    params = [eta_raw, raw_w, transport_b, scorer, readout_gamma_raw]
    if fp_weight_raw.requires_grad:
        params.append(fp_weight_raw)
    opt = torch.optim.Adam(params, lr=lr)
    pair_index = torch.arange(n, device=device)
    rank_temp = max(float(rank_temp), 1e-6)

    def current_params() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eta = torch.clamp(F.softplus(eta_raw), max=eta_max)
        readout_gamma = torch.clamp(F.softplus(readout_gamma_raw), max=eta_max)
        fp_weight = F.softplus(fp_weight_raw) + 1e-6
        direction = F.normalize(raw_w, dim=0)
        score_w = F.normalize(scorer, dim=0)
        return eta, direction, transport_b, score_w, readout_gamma, fp_weight

    def transport_scores(
        features: torch.Tensor,
        q_def: torch.Tensor,
        q_fp: torch.Tensor,
        eta: torch.Tensor,
        direction: torch.Tensor,
        b: torch.Tensor,
        score_w: torch.Tensor,
        readout_gamma: torch.Tensor,
        fp_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_margin = q_def - fp_weight * q_fp
        margin = torch.tanh((raw_margin - raw_margin.mean()) / raw_margin.std().clamp_min(1e-6))
        delta = eta * torch.tanh(margin + b)
        move_dir = basis @ direction
        rect = F.normalize(features.float() + delta.unsqueeze(-1) * move_dir.view(1, -1), dim=-1)
        coords = rect @ basis
        if residual_readout:
            scores = (features.float() @ axis) + readout_gamma * (coords[:, 1:] @ score_w)
        else:
            scores = coords @ score_w
        return scores, delta

    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        perm = pair_index[torch.randperm(n, device=device)]
        eta, direction, b, score_w, readout_gamma, fp_weight = current_params()
        features = torch.cat([fp_train[perm], def_train], dim=0)
        q_def = torch.cat([fp_q_def[perm], def_q_def], dim=0)
        q_fp = torch.cat([fp_q_fp[perm], def_q_fp], dim=0)
        scores, _ = transport_scores(features, q_def, q_fp, eta, direction, b, score_w, readout_gamma, fp_weight)
        s_fp, s_def = scores[:n], scores[n:]
        loss = F.softplus((s_fp - s_def + rank_margin) / rank_temp).mean()
        if preserve_host_margin_weight > 0:
            base_margin = def_q_axis - fp_q_axis[perm]
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
        features = torch.cat([fp_train, def_train], dim=0)
        q_def = torch.cat([fp_q_def, def_q_def], dim=0)
        q_fp = torch.cat([fp_q_fp, def_q_fp], dim=0)
        scores, delta = transport_scores(features, q_def, q_fp, eta, direction, b, score_w, readout_gamma, fp_weight)
        s_fp, s_def = scores[:n], scores[n:]
        pair_acc = (s_def > s_fp).float().mean().item()
        base_pair_acc = (def_q_axis > fp_q_axis).float().mean().item()
        delta_fp = delta[:n].mean().item()
        delta_def = delta[n:].mean().item()
    return {
        "alpha": float(eta.item()),
        "effective_alpha": float(eta.item()),
        "eta": float(eta.item()),
        "transport_direction": direction.detach().cpu().tolist(),
        "transport_b": float(b.item()),
        "subspace_score_w": score_w.detach().cpu().tolist(),
        "subspace_readout_mode": str(subspace_readout_mode),
        "subspace_basis_control": str(subspace_basis_control),
        "readout_gamma": float(readout_gamma.item()),
        "preserve_host_margin_weight": float(preserve_host_margin_weight),
        "basis": basis.detach().cpu(),
        "subspace_rank": int(basis.shape[1]),
        "fp_weight": float(fp_weight.item()),
        "gate_sharpness": 0.0,
        "transport_mode": "subspace_tanh",
        "train_points_per_class": int(n),
        "source_pair_rank_accuracy": float(pair_acc),
        "source_base_pair_rank_accuracy": float(base_pair_acc),
        "source_delta_fp_mean": float(delta_fp),
        "source_delta_def_mean": float(delta_def),
        "nonnegative_alpha": True,
        "hardpair_frac": float(hardpair_frac),
    }


def train_prescore_calibrators(
    banks: Dict[str, List[torch.Tensor]],
    axis: torch.Tensor,
    tau: float,
    max_train_points: int,
    epochs: int,
    lr: float,
    rank_margin: float,
    rank_temp: float,
    alpha_init: float,
    alpha_max: float,
    alpha_reg: float,
    fp_weight_init: float,
    gate_sharpness_init: float,
    learn_fp_weight: bool,
    learn_gate_sharpness: bool,
    nonnegative_alpha: bool,
    transport_mode: str,
    subspace_rank: int,
    subspace_readout_mode: str,
    subspace_basis_control: str,
    preserve_host_margin_weight: float,
    hardpair_frac: float,
    pair_label_control: str,
    seed: int,
    device: torch.device,
) -> List[Dict[str, object] | None]:
    calibrators: List[Dict[str, object] | None] = []
    for li, (fp, defect) in enumerate(zip(banks["fp"], banks["defect"])):
        train_fp, train_defect = apply_pair_label_control(fp, defect, pair_label_control, seed + 1000 * li)
        if transport_mode == "parallel_tanh":
            calibrators.append(
                train_one_layer_parallel_transport_calibrator(
                    fp=train_fp,
                    defect=train_defect,
                    axis=axis,
                    tau=tau,
                    max_train_points=max_train_points,
                    epochs=epochs,
                    lr=lr,
                    rank_margin=rank_margin,
                    rank_temp=rank_temp,
                    eta_init=abs(alpha_init),
                    eta_max=alpha_max,
                    eta_reg=alpha_reg,
                    fp_weight_init=fp_weight_init,
                    learn_fp_weight=learn_fp_weight,
                    hardpair_frac=hardpair_frac,
                    seed=seed + 100 * li,
                    device=device,
                )
            )
            continue
        if transport_mode == "subspace_tanh":
            calibrators.append(
                train_one_layer_subspace_transport_calibrator(
                    fp=train_fp,
                    defect=train_defect,
                    axis=axis,
                    tau=tau,
                    max_train_points=max_train_points,
                    epochs=epochs,
                    lr=lr,
                    rank_margin=rank_margin,
                    rank_temp=rank_temp,
                    eta_init=abs(alpha_init),
                    eta_max=alpha_max,
                    eta_reg=alpha_reg,
                    fp_weight_init=fp_weight_init,
                    learn_fp_weight=learn_fp_weight,
                    hardpair_frac=hardpair_frac,
                    subspace_rank=subspace_rank,
                    subspace_readout_mode=subspace_readout_mode,
                    subspace_basis_control=subspace_basis_control,
                    preserve_host_margin_weight=preserve_host_margin_weight,
                    seed=seed + 100 * li,
                    device=device,
                )
            )
            continue
        calibrators.append(
            train_one_layer_prescore_calibrator(
                fp=train_fp,
                defect=train_defect,
                axis=axis,
                tau=tau,
                max_train_points=max_train_points,
                epochs=epochs,
                lr=lr,
                rank_margin=rank_margin,
                rank_temp=rank_temp,
                alpha_init=alpha_init,
                alpha_max=alpha_max,
                alpha_reg=alpha_reg,
                fp_weight_init=fp_weight_init,
                gate_sharpness_init=gate_sharpness_init,
                learn_fp_weight=learn_fp_weight,
                learn_gate_sharpness=learn_gate_sharpness,
                nonnegative_alpha=nonnegative_alpha,
                hardpair_frac=hardpair_frac,
                seed=seed + 100 * li,
                device=device,
            )
        )
    return calibrators


def spatial_zscore_tokens(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=1, keepdim=True)
    sigma = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (x - mu) / sigma


def batch_patch_score_grid(
    patch_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    normal = text_features[:, :, 0]
    abnormal = text_features[:, :, 1]
    normal_score = 100.0 * (patch_features * normal[:, None, :]).sum(dim=-1)
    abnormal_score = 100.0 * (patch_features * abnormal[:, None, :]).sum(dim=-1)
    return (abnormal_score + 1.0 - normal_score) / 2.0


def masked_mean_abs(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vals = []
    for xi, mi in zip(x, mask):
        if bool(mi.any()):
            vals.append(xi[mi].abs().mean())
    if not vals:
        return x.new_tensor(0.0)
    return torch.stack(vals).mean()


def masked_topk_mean_by_rank(
    values: torch.Tensor,
    region_mask: torch.Tensor,
    rank_values: torch.Tensor,
    topk_frac: float,
) -> torch.Tensor | None:
    vals = []
    for vi, mi, ri in zip(values, region_mask, rank_values):
        idx = torch.nonzero(mi, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        k = max(1, int(math.ceil(idx.numel() * float(topk_frac))))
        k = min(k, idx.numel())
        top = torch.topk(ri[idx].detach(), k=k, largest=True).indices
        vals.append(vi[idx[top]].mean())
    if not vals:
        return None
    return torch.stack(vals).mean()


def train_prescore_map_calibrators(
    model,
    source_dataset: str,
    source_root: str,
    banks: Dict[str, List[torch.Tensor]],
    source_text_embeddings: Dict[str, torch.Tensor],
    source_axis: torch.Tensor,
    img_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    tau: float,
    fp_weight_init: float,
    gate_sharpness_init: float,
    epochs: int,
    lr: float,
    max_images: int,
    sep_margin: float,
    sep_temp: float,
    hardfp_mode: str,
    aggregate_loss: bool,
    preserve_weight: float,
    preserve_mode: str,
    topk_frac: float,
    alpha_init: float,
    alpha_max: float,
    alpha_reg: float,
    learn_layer_beta: bool,
    beta_init: float,
    beta_reg: float,
    learn_fp_weight: bool,
    learn_gate_sharpness: bool,
    nonnegative_alpha: bool,
    seed: int,
) -> List[Dict[str, object] | None]:
    if epochs <= 0:
        return [None] * len(banks["fp"])

    image_transform, mask_transform = build_transforms(img_size)
    normal_ds = GenericMetaDataset(
        root=source_root,
        image_transform=image_transform,
        mask_transform=mask_transform,
        split="train",
        normal_only=True,
    )
    anomaly_ds = GenericMetaDataset(
        root=source_root,
        image_transform=image_transform,
        mask_transform=mask_transform,
        split="test",
        anomaly_only=True,
    )
    source_ds = torch.utils.data.ConcatDataset([normal_ds, anomaly_ds])
    if max_images > 0 and max_images < len(source_ds):
        rng = random.Random(seed)
        indices = list(range(len(source_ds)))
        rng.shuffle(indices)
        source_ds = torch.utils.data.Subset(source_ds, indices[:max_images])
    loader = DataLoader(
        source_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    num_layers = len(banks["fp"])
    alpha_max = max(float(alpha_max), 1e-6)
    alpha_raws = []
    beta_raws = []
    fp_weight_raws = []
    gate_raws = []
    params: list[torch.Tensor] = []
    for _ in range(num_layers):
        if nonnegative_alpha:
            init = float(np.clip(float(alpha_init), 1e-6, 0.99 * alpha_max))
            alpha_raw = torch.tensor(inverse_softplus(init), device=device, requires_grad=True)
        else:
            init = float(np.clip(float(alpha_init), -0.99 * alpha_max, 0.99 * alpha_max))
            ratio = init / alpha_max
            alpha_raw = torch.tensor(
                0.5 * math.log((1.0 + ratio) / max(1.0 - ratio, 1e-6)),
                device=device,
                requires_grad=True,
            )
        beta_init_clamped = float(np.clip(float(beta_init), 1e-4, 1.0 - 1e-4))
        beta_raw = torch.tensor(
            math.log(beta_init_clamped / (1.0 - beta_init_clamped)),
            device=device,
            requires_grad=bool(learn_layer_beta),
        )
        fp_raw = torch.tensor(
            inverse_softplus(fp_weight_init),
            device=device,
            requires_grad=bool(learn_fp_weight),
        )
        gate_raw = torch.tensor(
            inverse_softplus(gate_sharpness_init),
            device=device,
                requires_grad=bool(learn_gate_sharpness),
        )
        alpha_raws.append(alpha_raw)
        beta_raws.append(beta_raw)
        fp_weight_raws.append(fp_raw)
        gate_raws.append(gate_raw)
        params.append(alpha_raw)
        if beta_raw.requires_grad:
            params.append(beta_raw)
        if fp_raw.requires_grad:
            params.append(fp_raw)
        if gate_raw.requires_grad:
            params.append(gate_raw)
    opt = torch.optim.Adam(params, lr=lr)
    source_axis = F.normalize(source_axis.to(device).float(), dim=0)
    fp_proj = [F.normalize(t.to(device), dim=-1) @ source_axis for t in banks["fp"]]
    def_proj = [F.normalize(t.to(device), dim=-1) @ source_axis for t in banks["defect"]]
    sep_temp = max(float(sep_temp), 1e-6)

    def current_layer_params(li: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if nonnegative_alpha:
            alpha = torch.clamp(F.softplus(alpha_raws[li]), max=alpha_max)
        else:
            alpha = alpha_max * torch.tanh(alpha_raws[li])
        beta = torch.sigmoid(beta_raws[li]) if learn_layer_beta else alpha.new_tensor(1.0)
        effective_alpha = alpha * beta
        fp_weight = F.softplus(fp_weight_raws[li]) + 1e-6
        gate = F.softplus(gate_raws[li]) + 1e-6
        return alpha, beta, effective_alpha, fp_weight, gate

    for _ in range(epochs):
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            class_names_batch = list(batch["class_name"])
            text_batch = torch.stack([source_text_embeddings[c].to(device) for c in class_names_batch], dim=0)
            axis_batch = F.normalize(text_batch[:, :, 1] - text_batch[:, :, 0], dim=-1)
            with torch.no_grad():
                seg_tokens, _ = model(image)
                seg_tokens = [t.detach() for t in seg_tokens]
            losses = []
            sum_base_maps: list[torch.Tensor] = []
            sum_new_maps: list[torch.Tensor] = []
            sum_ref_side: int | None = None
            for li, tokens in enumerate(seg_tokens):
                if fp_proj[li].numel() == 0 or def_proj[li].numel() == 0:
                    continue
                side = int(round(math.sqrt(tokens.shape[1])))
                patch_mask = downsample_mask_to_patch(mask, side).to(device)
                background_mask = ~patch_mask
                query_proj = tokens @ source_axis
                q_def = logmeanexp_negative_sqdist_1d(query_proj, def_proj[li], tau=tau)
                q_fp = logmeanexp_negative_sqdist_1d(query_proj, fp_proj[li], tau=tau)
                alpha, beta, effective_alpha, fp_weight_layer, gate_sharpness_layer = current_layer_params(li)
                raw_margin = q_def - fp_weight_layer * q_fp
                raw_mass = q_def + q_fp
                margin = spatial_tanh_zscore(raw_margin)
                support_mass = spatial_tanh_zscore(raw_mass)
                gate = torch.sigmoid(gate_sharpness_layer * support_mass)
                step = gate.unsqueeze(-1) * margin.unsqueeze(-1) * axis_batch[:, None, :]
                rect_tokens = F.normalize(tokens.float() + effective_alpha * step, dim=-1)
                base_score = spatial_zscore_tokens(batch_patch_score_grid(tokens, text_batch))
                new_score = spatial_zscore_tokens(batch_patch_score_grid(rect_tokens, text_batch))
                if aggregate_loss:
                    score_side = int(round(math.sqrt(base_score.shape[1])))
                    if sum_ref_side is None:
                        sum_ref_side = score_side

                    def to_ref_grid(score: torch.Tensor) -> torch.Tensor:
                        grid = score.view(score.shape[0], 1, score_side, score_side)
                        if score_side != sum_ref_side:
                            grid = F.interpolate(
                                grid,
                                size=(sum_ref_side, sum_ref_side),
                                mode="bilinear",
                                align_corners=True,
                            )
                        return grid[:, 0].reshape(score.shape[0], -1)

                    sum_base_maps.append(to_ref_grid(base_score))
                    sum_new_maps.append(to_ref_grid(new_score))
                q_fp_z = spatial_zscore_tokens(q_fp)
                if not aggregate_loss:
                    if hardfp_mode == "online_rank":
                        hard_rank = new_score
                    else:
                        hard_rank = base_score + q_fp_z
                    inside = masked_topk_mean_by_rank(new_score, patch_mask, new_score, topk_frac)
                    outside = masked_topk_mean_by_rank(new_score, background_mask, hard_rank, topk_frac)
                    if inside is not None and outside is not None:
                        losses.append(F.softplus((outside - inside + sep_margin) / sep_temp))
                    if preserve_mode == "online_hardfp":
                        inside_base = masked_topk_mean_by_rank(base_score, patch_mask, base_score, topk_frac)
                        inside_new = masked_topk_mean_by_rank(new_score, patch_mask, base_score, topk_frac)
                        outside_new = masked_topk_mean_by_rank(new_score, background_mask, new_score, topk_frac)
                        outside_base_at_new = masked_topk_mean_by_rank(base_score, background_mask, new_score, topk_frac)
                        preserve_terms = []
                        if inside_base is not None and inside_new is not None:
                            preserve_terms.append(F.relu(inside_base - inside_new))
                        if outside_new is not None and outside_base_at_new is not None:
                            preserve_terms.append(F.relu(outside_new - outside_base_at_new))
                        preserve = torch.stack(preserve_terms).mean() if preserve_terms else new_score.new_tensor(0.0)
                    elif preserve_mode == "monotonic_topk":
                        inside_base = masked_topk_mean_by_rank(base_score, patch_mask, base_score, topk_frac)
                        outside_base = masked_topk_mean_by_rank(base_score, background_mask, hard_rank, topk_frac)
                        inside_new = masked_topk_mean_by_rank(new_score, patch_mask, base_score, topk_frac)
                        outside_new = masked_topk_mean_by_rank(new_score, background_mask, hard_rank, topk_frac)
                        preserve_terms = []
                        if inside_base is not None and inside_new is not None:
                            preserve_terms.append(F.relu(inside_base - inside_new))
                        if outside_base is not None and outside_new is not None:
                            preserve_terms.append(F.relu(outside_new - outside_base))
                        preserve = torch.stack(preserve_terms).mean() if preserve_terms else new_score.new_tensor(0.0)
                    else:
                        preserve = masked_mean_abs(new_score - base_score, background_mask)
                    losses.append(float(preserve_weight) * preserve)
                if alpha_reg > 0:
                    losses.append(float(alpha_reg) * effective_alpha.pow(2))
                if beta_reg > 0 and learn_layer_beta:
                    losses.append(float(beta_reg) * beta.pow(2))
            if aggregate_loss and sum_base_maps and sum_new_maps and sum_ref_side is not None:
                base_sum = torch.stack(sum_base_maps, dim=0).sum(0)
                new_sum = torch.stack(sum_new_maps, dim=0).sum(0)
                patch_mask = downsample_mask_to_patch(mask, sum_ref_side).to(device)
                background_mask = ~patch_mask
                hard_rank = new_sum if hardfp_mode == "online_rank" else base_sum
                inside = masked_topk_mean_by_rank(new_sum, patch_mask, new_sum, topk_frac)
                outside = masked_topk_mean_by_rank(new_sum, background_mask, hard_rank, topk_frac)
                if inside is not None and outside is not None:
                    losses.append(F.softplus((outside - inside + sep_margin) / sep_temp))
                if preserve_mode == "online_hardfp":
                    inside_base = masked_topk_mean_by_rank(base_sum, patch_mask, base_sum, topk_frac)
                    inside_new = masked_topk_mean_by_rank(new_sum, patch_mask, base_sum, topk_frac)
                    outside_new = masked_topk_mean_by_rank(new_sum, background_mask, new_sum, topk_frac)
                    outside_base_at_new = masked_topk_mean_by_rank(base_sum, background_mask, new_sum, topk_frac)
                    preserve_terms = []
                    if inside_base is not None and inside_new is not None:
                        preserve_terms.append(F.relu(inside_base - inside_new))
                    if outside_new is not None and outside_base_at_new is not None:
                        preserve_terms.append(F.relu(outside_new - outside_base_at_new))
                    preserve = torch.stack(preserve_terms).mean() if preserve_terms else new_sum.new_tensor(0.0)
                elif preserve_mode == "monotonic_topk":
                    inside_base = masked_topk_mean_by_rank(base_sum, patch_mask, base_sum, topk_frac)
                    outside_base = masked_topk_mean_by_rank(base_sum, background_mask, hard_rank, topk_frac)
                    inside_new = masked_topk_mean_by_rank(new_sum, patch_mask, base_sum, topk_frac)
                    outside_new = masked_topk_mean_by_rank(new_sum, background_mask, hard_rank, topk_frac)
                    preserve_terms = []
                    if inside_base is not None and inside_new is not None:
                        preserve_terms.append(F.relu(inside_base - inside_new))
                    if outside_base is not None and outside_new is not None:
                        preserve_terms.append(F.relu(outside_new - outside_base))
                    preserve = torch.stack(preserve_terms).mean() if preserve_terms else new_sum.new_tensor(0.0)
                else:
                    preserve = masked_mean_abs(new_sum - base_sum, background_mask)
                losses.append(float(preserve_weight) * preserve)
            if not losses:
                continue
            opt.zero_grad(set_to_none=True)
            loss = torch.stack(losses).mean()
            loss.backward()
            opt.step()

    calibrators: List[Dict[str, object] | None] = []
    for li in range(num_layers):
        with torch.no_grad():
            alpha, beta, effective_alpha, fp_weight_layer, gate_layer = current_layer_params(li)
        calibrators.append(
            {
                "alpha": float(alpha.item()),
                "beta": float(beta.item()),
                "effective_alpha": float(effective_alpha.item()),
                "fp_weight": float(fp_weight_layer.item()),
                "gate_sharpness": float(gate_layer.item()),
                "calibration": "source_map_sep_preserve_no_target",
                "preserve_weight": float(preserve_weight),
                "preserve_mode": str(preserve_mode),
                "hardfp_mode": str(hardfp_mode),
                "aggregate_loss": bool(aggregate_loss),
                "topk_frac": float(topk_frac),
                "learn_layer_beta": bool(learn_layer_beta),
                "nonnegative_alpha": bool(nonnegative_alpha),
            }
        )
    return calibrators


def downsample_mask_to_patch(mask: torch.Tensor, side: int) -> torch.Tensor:
    ds = F.interpolate(mask.float(), size=(side, side), mode="nearest")
    return ds[:, 0].reshape(mask.shape[0], -1) > 0.5


class GenericMetaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        image_transform,
        mask_transform,
        split: str,
        class_name: str | None = None,
        anomaly_only: bool | None = None,
        normal_only: bool | None = None,
    ):
        self.root = Path(root)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        meta = json.loads((self.root / "meta.json").read_text())
        split_meta = meta[split]
        if class_name is not None:
            split_meta = {class_name: split_meta[class_name]}
        self.obj_list = sorted(split_meta.keys())
        self.items = []
        for cls_name in self.obj_list:
            for item in split_meta[cls_name]:
                anomaly = int(item["anomaly"])
                if anomaly_only is True and anomaly == 0:
                    continue
                if normal_only is True and anomaly != 0:
                    continue
                self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        img_path = self.root / item["img_path"]
        img = Image.open(img_path).convert("RGB")
        img = self.image_transform(img)
        anomaly = int(item["anomaly"])
        mask_path = item.get("mask_path")
        if anomaly == 0 or not mask_path:
            mask = Image.fromarray(np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8), mode="L")
        else:
            raw_mask = np.array(Image.open(self.root / mask_path).convert("L")) > 0
            mask = Image.fromarray(raw_mask.astype(np.uint8) * 255, mode="L")
        mask = self.mask_transform(mask)
        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(anomaly).to(torch.int64),
            "file_name": item["img_path"],
            "class_name": item["cls_name"],
            "anomaly": anomaly,
        }


def build_transforms(image_size: int):
    image_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    return image_transform, mask_transform


def get_bank_cache_path(
    preset: str,
    source_root: str,
    ckpt_dir: str,
    cache_dir: Path,
    model_name: str,
    adapter_load_mode: str,
    levels: Sequence[int],
    img_size: int,
    max_fp_per_class: int,
    max_defect_per_class: int,
    normal_topk_frac: float,
    defect_topk_frac: float,
    seed: int,
    source_limit_per_class: int | None,
    source_fp_mode: str = "hardfp",
) -> Path:
    stem = Path(ckpt_dir).name
    frac_tag = str(normal_topk_frac).replace(".", "p")
    source_tag = Path(source_root).name.replace("/", "_")
    model_tag = model_name.replace("/", "-").replace("@", "-").replace(" ", "")
    levels_tag = "-".join(str(int(level)) for level in levels)
    limit_tag = "all" if source_limit_per_class is None else str(int(source_limit_per_class))
    short_mode = {"random_normal": "rn", "easy_normal": "en"}.get(source_fp_mode, source_fp_mode)
    mode_tag = "" if source_fp_mode == "hardfp" else f"_fp{short_mode}"
    return cache_dir / (
        f"aaclip_{preset}_{source_tag}_{stem}_{BANK_PROTOCOL_VERSION}_{model_tag}_{adapter_load_mode}_l{levels_tag}_"
        f"img{int(img_size)}_fp{max_fp_per_class}_def{max_defect_per_class}_"
        f"topk{frac_tag}_deftop{str(defect_topk_frac).replace('.', 'p')}_src{limit_tag}_seed{seed}{mode_tag}.pt"
    )


def collect_source_banks(
    model,
    model_name: str,
    adapter_load_mode: str,
    source_dataset: str,
    source_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_fp_per_class: int,
    max_defect_per_class: int,
    normal_topk_frac: float,
    defect_topk_frac: float,
    seed: int,
    source_limit_per_class: int | None,
    source_fp_mode: str = "hardfp",
) -> Dict[str, object]:
    from forward_utils import get_adapted_text_embedding

    image_transform, mask_transform = build_transforms(img_size)
    text_embeddings = get_adapted_text_embedding(model, source_dataset, device)
    source_meta = json.loads((Path(source_root) / "meta.json").read_text())
    class_names = sorted(source_meta["train"].keys())
    num_layers = len(getattr(model, "levels", [0, 1, 2, 3]))
    feature_dim = int(getattr(model, "output_dim", 768))
    bank_fp: List[List[torch.Tensor]] = [list() for _ in range(num_layers)]
    bank_def: List[List[torch.Tensor]] = [list() for _ in range(num_layers)]

    rng = random.Random(seed)

    for class_name in class_names:
        normal_ds = GenericMetaDataset(
            root=source_root,
            image_transform=image_transform,
            mask_transform=mask_transform,
            split="train",
            class_name=class_name,
            normal_only=True,
        )
        anomaly_ds = GenericMetaDataset(
            root=source_root,
            image_transform=image_transform,
            mask_transform=mask_transform,
            split="test",
            class_name=class_name,
            anomaly_only=True,
        )
        if source_limit_per_class is not None and source_limit_per_class < len(normal_ds):
            indices = list(range(len(normal_ds)))
            rng.shuffle(indices)
            indices = indices[:source_limit_per_class]
            normal_ds = torch.utils.data.Subset(normal_ds, indices)
        if source_limit_per_class is not None and source_limit_per_class < len(anomaly_ds):
            indices = list(range(len(anomaly_ds)))
            rng.shuffle(indices)
            indices = indices[:source_limit_per_class]
            anomaly_ds = torch.utils.data.Subset(anomaly_ds, indices)
        normal_dl = DataLoader(
            normal_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        anomaly_dl = DataLoader(
            anomaly_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        text_feature = text_embeddings[class_name]
        fp_per_layer: List[List[torch.Tensor]] = [list() for _ in range(num_layers)]
        def_per_layer: List[List[torch.Tensor]] = [list() for _ in range(num_layers)]

        for batch in normal_dl:
            image = batch["image"].to(device)
            seg_tokens, _ = model(image)

            for li, tokens in enumerate(seg_tokens):
                scores = patch_score_grid(tokens, text_feature)
                for bi in range(tokens.shape[0]):
                    feats = tokens[bi]
                    k = max(1, int(normal_topk_frac * feats.shape[0]))
                    if source_fp_mode == "hardfp":
                        top_idx = torch.topk(scores[bi], k=k, largest=True).indices
                    elif source_fp_mode == "easy_normal":
                        top_idx = torch.topk(scores[bi], k=k, largest=False).indices
                    elif source_fp_mode == "random_normal":
                        top_idx = torch.randperm(feats.shape[0], device=feats.device)[:k]
                    else:
                        raise ValueError(f"Unsupported source_fp_mode: {source_fp_mode}")
                    selected = feats[top_idx]
                    if selected.shape[0] > max_fp_per_class:
                        selected = farthest_point_subsample(selected, max_fp_per_class)
                    fp_per_layer[li].append(selected.detach().cpu())

        for batch in anomaly_dl:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            seg_tokens, _ = model(image)

            for li, tokens in enumerate(seg_tokens):
                side = int(round(math.sqrt(tokens.shape[1])))
                assert side * side == tokens.shape[1]
                patch_mask = downsample_mask_to_patch(mask, side)
                scores = patch_score_grid(tokens, text_feature)

                for bi in range(tokens.shape[0]):
                    feats = tokens[bi]
                    pos_idx = torch.nonzero(patch_mask[bi], as_tuple=False).flatten()
                    if pos_idx.numel() == 0:
                        continue
                    pos_scores = scores[bi][pos_idx]
                    k = max(1, int(math.ceil(pos_idx.numel() * defect_topk_frac)))
                    k = min(k, max_defect_per_class, pos_idx.numel())
                    top_pos = torch.topk(pos_scores, k=k, largest=True).indices
                    selected = feats[pos_idx[top_pos]]
                    def_per_layer[li].append(selected.detach().cpu())

        for li in range(num_layers):
            fp_cat = torch.cat(fp_per_layer[li], dim=0) if fp_per_layer[li] else torch.empty(0, feature_dim)
            def_cat = torch.cat(def_per_layer[li], dim=0) if def_per_layer[li] else torch.empty(0, feature_dim)
            if fp_cat.shape[0] > max_fp_per_class:
                fp_cat = farthest_point_subsample(fp_cat, max_fp_per_class)
            if def_cat.shape[0] > max_defect_per_class:
                def_cat = farthest_point_subsample(def_cat, max_defect_per_class)
            if fp_cat.numel() > 0:
                bank_fp[li].append(fp_cat)
            if def_cat.numel() > 0:
                bank_def[li].append(def_cat)

    bank_fp_t = [torch.cat(v, dim=0) if v else torch.empty(0, feature_dim) for v in bank_fp]
    bank_def_t = [torch.cat(v, dim=0) if v else torch.empty(0, feature_dim) for v in bank_def]
    return {
        "fp": bank_fp_t,
        "defect": bank_def_t,
        "protocol_version": BANK_PROTOCOL_VERSION,
        "model_name": model_name,
        "adapter_load_mode": adapter_load_mode,
        "levels": list(getattr(model, "levels", [])),
        "feature_dim": feature_dim,
        "img_size": img_size,
        "source_dataset": source_dataset,
        "source_root": source_root,
        "max_fp_per_class": max_fp_per_class,
        "max_defect_per_class": max_defect_per_class,
        "normal_topk_frac": normal_topk_frac,
        "defect_topk_frac": defect_topk_frac,
        "source_fp_mode": source_fp_mode,
        "source_limit_per_class": source_limit_per_class,
    }


def get_or_collect_source_banks(
    model,
    preset: str,
    model_name: str,
    adapter_load_mode: str,
    source_dataset: str,
    source_root: str,
    ckpt_dir: str,
    cache_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_fp_per_class: int,
    max_defect_per_class: int,
    normal_topk_frac: float,
    defect_topk_frac: float,
    seed: int,
    source_limit_per_class: int | None,
    source_fp_mode: str = "hardfp",
) -> Dict[str, object]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    levels = list(getattr(model, "levels", []))
    feature_dim = int(getattr(model, "output_dim", 768))
    requested_meta = {
        "protocol_version": BANK_PROTOCOL_VERSION,
        "model_name": model_name,
        "adapter_load_mode": adapter_load_mode,
        "levels": levels,
        "feature_dim": feature_dim,
        "img_size": img_size,
        "source_dataset": source_dataset,
        "source_root": source_root,
        "max_fp_per_class": max_fp_per_class,
        "max_defect_per_class": max_defect_per_class,
        "normal_topk_frac": normal_topk_frac,
        "defect_topk_frac": defect_topk_frac,
        "source_limit_per_class": source_limit_per_class,
        "source_fp_mode": source_fp_mode,
    }
    cache_path = get_bank_cache_path(
        preset=preset,
        source_root=source_root,
        ckpt_dir=ckpt_dir,
        cache_dir=cache_dir,
        model_name=model_name,
        adapter_load_mode=adapter_load_mode,
        levels=levels,
        img_size=img_size,
        max_fp_per_class=max_fp_per_class,
        max_defect_per_class=max_defect_per_class,
        normal_topk_frac=normal_topk_frac,
        defect_topk_frac=defect_topk_frac,
        seed=seed,
        source_limit_per_class=source_limit_per_class,
        source_fp_mode=source_fp_mode,
    )
    if cache_path.exists():
        banks = torch.load(cache_path, map_location="cpu")
        cached_meta = {
            "protocol_version": banks.get("protocol_version"),
            "model_name": banks.get("model_name"),
            "adapter_load_mode": banks.get("adapter_load_mode"),
            "levels": banks.get("levels"),
            "feature_dim": banks.get("feature_dim"),
            "img_size": banks.get("img_size"),
            "source_dataset": banks.get("source_dataset"),
            "source_root": banks.get("source_root"),
            "max_fp_per_class": banks.get("max_fp_per_class"),
            "max_defect_per_class": banks.get("max_defect_per_class"),
            "normal_topk_frac": banks.get("normal_topk_frac"),
            "defect_topk_frac": banks.get("defect_topk_frac"),
            "source_limit_per_class": banks.get("source_limit_per_class"),
            "source_fp_mode": banks.get("source_fp_mode", "hardfp"),
        }
        if cached_meta == requested_meta:
            return banks

    for candidate_path in ([] if source_fp_mode != "hardfp" else sorted(cache_dir.glob("aaclip_*.pt"))):
        if candidate_path == cache_path:
            continue
        try:
            banks = torch.load(candidate_path, map_location="cpu")
        except Exception:
            continue
        cached_meta = {
            "protocol_version": banks.get("protocol_version"),
            "model_name": banks.get("model_name"),
            "adapter_load_mode": banks.get("adapter_load_mode"),
            "levels": banks.get("levels"),
            "feature_dim": banks.get("feature_dim"),
            "img_size": banks.get("img_size"),
            "source_dataset": banks.get("source_dataset"),
            "source_root": banks.get("source_root"),
            "max_fp_per_class": banks.get("max_fp_per_class"),
            "max_defect_per_class": banks.get("max_defect_per_class"),
            "normal_topk_frac": banks.get("normal_topk_frac"),
            "defect_topk_frac": banks.get("defect_topk_frac"),
            "source_limit_per_class": banks.get("source_limit_per_class"),
            "source_fp_mode": banks.get("source_fp_mode", "hardfp"),
        }
        if cached_meta == requested_meta:
            print(f"[BANK] reuse compatible cache: {candidate_path}", flush=True)
            return banks

    banks = collect_source_banks(
        model=model,
        model_name=model_name,
        adapter_load_mode=adapter_load_mode,
        source_dataset=source_dataset,
        source_root=source_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        max_fp_per_class=max_fp_per_class,
        max_defect_per_class=max_defect_per_class,
        normal_topk_frac=normal_topk_frac,
        defect_topk_frac=defect_topk_frac,
        seed=seed,
        source_limit_per_class=source_limit_per_class,
        source_fp_mode=source_fp_mode,
    )
    torch.save(banks, cache_path)
    return banks


def evaluate_target_dataset(
    model,
    source_dataset: str,
    source_root: str,
    target_dataset: str,
    target_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    banks: Dict[str, object],
    tau: float,
    fp_weight: float,
    alphas: Sequence[float],
    blend_modes: Sequence[str],
    candidate_topk_frac: float,
    gate_sharpness: float,
    candidate_image_mode: str,
    cal_pro: bool,
    aupro_max_step: int,
    report_group_separability: bool,
    separability_hard_frac: float,
    class_name_filter: str | None,
    target_limit_per_class: int | None,
    seed: int,
    plugin_epochs: int,
    plugin_lr: float,
    plugin_max_train_points: int,
    plugin_loss: str,
    plugin_rank_margin: float,
    plugin_rank_temp: float,
    plugin_contrast_weight: float,
    plugin_contrast_margin: float,
    plugin_hidden_dim: int,
    plugin_learn_alpha: bool,
    plugin_alpha_init: float,
    plugin_alpha_max: float,
    plugin_alpha_reg: float,
    plugin_hardpair_frac: float,
    prescore_calib_epochs: int,
    prescore_calib_lr: float,
    prescore_calib_max_train_points: int,
    prescore_calib_rank_margin: float,
    prescore_calib_rank_temp: float,
    prescore_calib_alpha_init: float,
    prescore_calib_alpha_max: float,
    prescore_calib_alpha_reg: float,
    prescore_calib_fp_weight_init: float,
    prescore_calib_hardpair_frac: float,
    prescore_calib_learn_fp_weight: bool,
    prescore_calib_learn_gate: bool,
    prescore_calib_nonnegative_alpha: bool,
    prescore_calib_transport_mode: str,
    prescore_calib_subspace_rank: int,
    prescore_calib_subspace_readout_mode: str,
    prescore_calib_subspace_basis_control: str,
    prescore_calib_pair_label_control: str,
    prescore_calib_preserve_host_margin_weight: float,
    prescore_calib_output_mode: str,
    prescore_map_calib_epochs: int,
    prescore_map_calib_lr: float,
    prescore_map_calib_max_images: int,
    prescore_map_calib_sep_margin: float,
    prescore_map_calib_sep_temp: float,
    prescore_map_calib_hardfp_mode: str,
    prescore_map_calib_aggregate_loss: bool,
    prescore_map_calib_preserve_weight: float,
    prescore_map_calib_preserve_mode: str,
    prescore_map_calib_topk_frac: float,
    prescore_map_calib_learn_layer_beta: bool,
    prescore_map_calib_beta_init: float,
    prescore_map_calib_beta_reg: float,
) -> Dict[str, object]:
    from dataset.constants import DOMAINS
    from forward_utils import (
        calculate_similarity_map,
        get_adapted_text_embedding,
        metrics_eval,
    )
    metrics_supports_pro = "cal_pro" in getattr(metrics_eval, "__code__").co_varnames

    def run_metrics(pixel_label, image_label, pixel_preds, image_preds, class_name):
        kwargs = {"domain": DOMAINS[target_dataset]}
        if metrics_supports_pro:
            kwargs["cal_pro"] = cal_pro
        out = metrics_eval(pixel_label, image_label, pixel_preds, image_preds, class_name, **kwargs)
        maybe_add_pixel_pro(out, pixel_label, pixel_preds, cal_pro=cal_pro, max_step=aupro_max_step)
        return out

    image_transform, mask_transform = build_transforms(img_size)
    text_embeddings = get_adapted_text_embedding(model, target_dataset, device)
    source_text_embeddings = get_adapted_text_embedding(model, source_dataset, device)
    source_axis = mean_text_anomaly_axis(source_text_embeddings)
    fp_proj = [F.normalize(t.to(device), dim=-1) @ source_axis for t in banks["fp"]]
    def_proj = [F.normalize(t.to(device), dim=-1) @ source_axis for t in banks["defect"]]
    use_plugin = plugin_epochs > 0 and any(mode.startswith("plugin_") for mode in blend_modes)
    use_prescore_calibrated = any(mode.startswith("prescore_calibrated") for mode in blend_modes)
    layer_plugins = (
        train_layer_plugins(
            banks=banks,
            axis=source_axis,
            tau=tau,
            max_train_points=plugin_max_train_points,
            epochs=plugin_epochs,
            lr=plugin_lr,
            loss_type=plugin_loss,
            margin=plugin_rank_margin,
            rank_temp=plugin_rank_temp,
            contrast_weight=plugin_contrast_weight,
            contrast_margin=plugin_contrast_margin,
            hidden_dim=plugin_hidden_dim,
            learn_alpha=plugin_learn_alpha,
            alpha_init=plugin_alpha_init,
            alpha_max=plugin_alpha_max,
            alpha_reg=plugin_alpha_reg,
            hardpair_frac=plugin_hardpair_frac,
            seed=seed,
            device=device,
        )
        if use_plugin
        else [None] * len(banks["fp"])
    )
    if use_prescore_calibrated and prescore_map_calib_epochs > 0:
        prescore_calibrators = train_prescore_map_calibrators(
            model=model,
            source_dataset=source_dataset,
            source_root=source_root,
            banks=banks,
            source_text_embeddings=source_text_embeddings,
            source_axis=source_axis,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            tau=tau,
            fp_weight_init=fp_weight,
            gate_sharpness_init=gate_sharpness,
            epochs=prescore_map_calib_epochs,
            lr=prescore_map_calib_lr,
            max_images=prescore_map_calib_max_images,
            sep_margin=prescore_map_calib_sep_margin,
            sep_temp=prescore_map_calib_sep_temp,
            hardfp_mode=prescore_map_calib_hardfp_mode,
            aggregate_loss=prescore_map_calib_aggregate_loss,
            preserve_weight=prescore_map_calib_preserve_weight,
            preserve_mode=prescore_map_calib_preserve_mode,
            topk_frac=prescore_map_calib_topk_frac,
            alpha_init=prescore_calib_alpha_init,
            alpha_max=prescore_calib_alpha_max,
            alpha_reg=prescore_calib_alpha_reg,
            learn_layer_beta=prescore_map_calib_learn_layer_beta,
            beta_init=prescore_map_calib_beta_init,
            beta_reg=prescore_map_calib_beta_reg,
            learn_fp_weight=prescore_calib_learn_fp_weight,
            learn_gate_sharpness=prescore_calib_learn_gate,
            nonnegative_alpha=prescore_calib_nonnegative_alpha,
            seed=seed,
        )
    elif use_prescore_calibrated:
        prescore_calibrators = train_prescore_calibrators(
            banks=banks,
            axis=source_axis,
            tau=tau,
            max_train_points=prescore_calib_max_train_points,
            epochs=prescore_calib_epochs,
            lr=prescore_calib_lr,
            rank_margin=prescore_calib_rank_margin,
            rank_temp=prescore_calib_rank_temp,
            alpha_init=prescore_calib_alpha_init,
            alpha_max=prescore_calib_alpha_max,
            alpha_reg=prescore_calib_alpha_reg,
            fp_weight_init=prescore_calib_fp_weight_init,
            gate_sharpness_init=gate_sharpness,
            learn_fp_weight=prescore_calib_learn_fp_weight,
            learn_gate_sharpness=prescore_calib_learn_gate,
            nonnegative_alpha=prescore_calib_nonnegative_alpha,
            transport_mode=prescore_calib_transport_mode,
            subspace_rank=prescore_calib_subspace_rank,
            subspace_readout_mode=prescore_calib_subspace_readout_mode,
            subspace_basis_control=prescore_calib_subspace_basis_control,
            preserve_host_margin_weight=prescore_calib_preserve_host_margin_weight,
            hardpair_frac=prescore_calib_hardpair_frac,
            pair_label_control=prescore_calib_pair_label_control,
            seed=seed,
            device=device,
        )
    else:
        prescore_calibrators = [None] * len(banks["fp"])
    source_plugin_summary = None
    source_prescore_calib_summary = None
    if use_plugin:
        source_plugin_summary = {
            "source_dataset": source_dataset,
            "plugin_loss": plugin_loss,
            "learn_alpha": bool(plugin_learn_alpha),
            "hardpair_frac": float(plugin_hardpair_frac),
            "layer_train_accuracy": [
                None if p is None else p["train_accuracy"] for p in layer_plugins
            ],
            "layer_pair_rank_accuracy": [
                None if p is None else p["source_pair_rank_accuracy"] for p in layer_plugins
            ],
            "layer_learned_alpha": [
                None if p is None else p.get("learned_alpha", 1.0) for p in layer_plugins
            ],
        }
        print(
            json.dumps({"stage": "source_global_plugin_trained", **source_plugin_summary}),
            flush=True,
        )
    if use_prescore_calibrated:
        source_prescore_calib_summary = {
            "source_dataset": source_dataset,
            "calibration": "source_map_sep_preserve_no_target"
            if prescore_map_calib_epochs > 0
            else "source_prescore_feature_rank_no_target",
            "hardpair_frac": float(prescore_calib_hardpair_frac),
            "map_calib_epochs": int(prescore_map_calib_epochs),
            "map_calib_hardfp_mode": str(prescore_map_calib_hardfp_mode),
            "map_calib_aggregate_loss": bool(prescore_map_calib_aggregate_loss),
            "map_calib_preserve_weight": float(prescore_map_calib_preserve_weight),
            "map_calib_preserve_mode": str(prescore_map_calib_preserve_mode),
            "map_calib_topk_frac": float(prescore_map_calib_topk_frac),
            "map_calib_learn_layer_beta": bool(prescore_map_calib_learn_layer_beta),
            "map_calib_beta_init": float(prescore_map_calib_beta_init),
            "map_calib_beta_reg": float(prescore_map_calib_beta_reg),
            "nonnegative_alpha": bool(prescore_calib_nonnegative_alpha),
            "transport_mode": str(prescore_calib_transport_mode),
            "subspace_rank": int(prescore_calib_subspace_rank),
            "subspace_readout_mode": str(prescore_calib_subspace_readout_mode),
            "subspace_basis_control": str(prescore_calib_subspace_basis_control),
            "pair_label_control": str(prescore_calib_pair_label_control),
            "preserve_host_margin_weight": float(prescore_calib_preserve_host_margin_weight),
            "output_mode": str(prescore_calib_output_mode),
            "layer_alpha": [
                None if c is None else c["alpha"] for c in prescore_calibrators
            ],
            "layer_beta": [
                None if c is None else c.get("beta", 1.0) for c in prescore_calibrators
            ],
            "layer_effective_alpha": [
                None if c is None else c.get("effective_alpha", c["alpha"]) for c in prescore_calibrators
            ],
            "layer_fp_weight": [
                None if c is None else c["fp_weight"] for c in prescore_calibrators
            ],
            "layer_gate_sharpness": [
                None if c is None else c["gate_sharpness"] for c in prescore_calibrators
            ],
            "layer_transport_w": [
                None if c is None else c.get("transport_w") for c in prescore_calibrators
            ],
            "layer_transport_b": [
                None if c is None else c.get("transport_b") for c in prescore_calibrators
            ],
            "layer_transport_direction": [
                None if c is None else c.get("transport_direction") for c in prescore_calibrators
            ],
            "layer_subspace_score_w": [
                None if c is None else c.get("subspace_score_w") for c in prescore_calibrators
            ],
            "layer_readout_gamma": [
                None if c is None else c.get("readout_gamma") for c in prescore_calibrators
            ],
            "layer_source_delta_fp_mean": [
                None if c is None else c.get("source_delta_fp_mean") for c in prescore_calibrators
            ],
            "layer_source_delta_def_mean": [
                None if c is None else c.get("source_delta_def_mean") for c in prescore_calibrators
            ],
            "layer_source_base_pair_rank_accuracy": [
                None if c is None else c.get("source_base_pair_rank_accuracy") for c in prescore_calibrators
            ],
            "layer_source_pair_rank_accuracy": [
                None if c is None else c.get("source_pair_rank_accuracy") for c in prescore_calibrators
            ],
        }
        print(
            json.dumps({"stage": "source_prescore_calibrator_trained", **source_prescore_calib_summary}),
            flush=True,
        )
    target_meta = json.loads((Path(target_root) / "meta.json").read_text())
    class_names = sorted(target_meta["test"].keys())
    if class_name_filter is not None:
        class_names = [c for c in class_names if c == class_name_filter]
    rng = random.Random(seed)

    results = []
    for class_name in class_names:
        ds = GenericMetaDataset(
            root=target_root,
            image_transform=image_transform,
            mask_transform=mask_transform,
            split="test",
            class_name=class_name,
        )
        if target_limit_per_class is not None and target_limit_per_class < len(ds):
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            indices = indices[:target_limit_per_class]
            ds = torch.utils.data.Subset(ds, indices)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        text_feature = text_embeddings[class_name]
        target_axis = F.normalize(text_feature[:, 1] - text_feature[:, 0], dim=0)
        use_prescore_rectify = "prescore_rectify" in blend_modes

        masks_all: List[np.ndarray] = []
        labels_all: List[np.ndarray] = []
        baseline_maps_all: List[np.ndarray] = []
        ours_maps_all: List[np.ndarray] = []
        candidate_maps_all: Dict[str, List[np.ndarray]] = {}
        baseline_img_all: List[np.ndarray] = []

        for batch in dl:
            image = batch["image"].to(device)
            mask = batch["mask"].cpu().numpy()
            label = batch["label"].cpu().numpy()
            seg_tokens, det_token = model(image)

            pred = det_token @ text_feature
            pred = (pred[:, 1] + 1.0) / 2.0
            baseline_img_all.append(pred.detach().cpu().numpy())

            layer_baseline = []
            layer_parallel = []
            layer_fullfeat = []
            layer_plugin = []
            layer_prescore_calibrated = []
            layer_prescore_by_alpha: Dict[float, List[torch.Tensor]] = {
                float(alpha): [] for alpha in alphas
            } if use_prescore_rectify else {}
            for li, tokens in enumerate(seg_tokens):
                baseline_map = calculate_similarity_map(
                    tokens,
                    text_feature,
                    img_size,
                    test=True,
                    domain=DOMAINS[target_dataset],
                )
                layer_baseline.append(baseline_map[:, 0])

                query_proj = tokens @ source_axis
                q_def = logmeanexp_negative_sqdist_1d(query_proj, def_proj[li], tau=tau)
                q_fp = logmeanexp_negative_sqdist_1d(query_proj, fp_proj[li], tau=tau)
                margin = spatial_tanh_zscore(q_def - fp_weight * q_fp)
                q_def_full = logmeanexp_cosine_support(tokens, banks["defect"][li].to(device), tau=0.1)
                q_fp_full = logmeanexp_cosine_support(tokens, banks["fp"][li].to(device), tau=0.1)
                full_margin = spatial_tanh_zscore(q_def_full - fp_weight * q_fp_full)
                if use_prescore_rectify:
                    for alpha in alphas:
                        rect_tokens = prescore_rectify_seg_tokens(
                            tokens=tokens,
                            q_def=q_def,
                            q_fp=q_fp,
                            target_axis=target_axis,
                            alpha=float(alpha),
                            fp_weight=fp_weight,
                            gate_sharpness=gate_sharpness,
                        )
                        rect_map = calculate_similarity_map(
                            rect_tokens,
                            text_feature,
                            img_size,
                            test=True,
                            domain=DOMAINS[target_dataset],
                        )
                        layer_prescore_by_alpha[float(alpha)].append(rect_map[:, 0])
                if use_prescore_calibrated:
                    calibrator = prescore_calibrators[li]
                    if calibrator is None:
                        raise RuntimeError("prescore_calibrated requires source calibrators")
                    if calibrator.get("transport_mode") == "parallel_tanh":
                        rect_tokens = prescore_parallel_transport_seg_tokens(
                            tokens=tokens,
                            q_def=q_def,
                            q_fp=q_fp,
                            target_axis=target_axis,
                            eta=float(calibrator["eta"]),
                            transport_w=float(calibrator["transport_w"]),
                            transport_b=float(calibrator["transport_b"]),
                            fp_weight=float(calibrator["fp_weight"]),
                        )
                    elif calibrator.get("transport_mode") == "subspace_tanh":
                        rect_tokens = prescore_subspace_transport_seg_tokens(
                            tokens=tokens,
                            q_def=q_def,
                            q_fp=q_fp,
                            basis=calibrator["basis"],
                            eta=float(calibrator["eta"]),
                            direction=calibrator["transport_direction"],
                            transport_b=float(calibrator["transport_b"]),
                            fp_weight=float(calibrator["fp_weight"]),
                        )
                        if prescore_calib_output_mode in {"subspace_score", "subspace_zscore"}:
                            rect_map = subspace_score_map_from_tokens(
                                tokens=rect_tokens,
                                basis=calibrator["basis"],
                                score_w=calibrator["subspace_score_w"],
                                img_size=img_size,
                                normalize_mode=prescore_calib_output_mode,
                            )
                        elif prescore_calib_output_mode in {"host_residual", "host_residual_zscore"}:
                            rect_map = subspace_host_residual_map_from_tokens(
                                tokens=tokens,
                                rect_tokens=rect_tokens,
                                basis=calibrator["basis"],
                                score_w=calibrator["subspace_score_w"],
                                target_axis=target_axis,
                                readout_gamma=float(calibrator.get("readout_gamma", 1.0)),
                                img_size=img_size,
                                normalize_mode=prescore_calib_output_mode,
                            )
                        else:
                            rect_map = None
                        if rect_map is not None:
                            side = int(round(math.sqrt(tokens.shape[1])))
                            margin_map = margin.view(tokens.shape[0], 1, side, side)
                            margin_map = F.interpolate(
                                margin_map,
                                size=(img_size, img_size),
                                mode="bilinear",
                                align_corners=True,
                            )
                            layer_parallel.append(margin_map[:, 0])
                            full_margin_map = full_margin.view(tokens.shape[0], 1, side, side)
                            full_margin_map = F.interpolate(
                                full_margin_map,
                                size=(img_size, img_size),
                                mode="bilinear",
                                align_corners=True,
                            )
                            layer_fullfeat.append(full_margin_map[:, 0])
                            if layer_plugins[li] is not None:
                                plugin_logit = apply_plugin_from_scores(query_proj, q_def, q_fp, layer_plugins[li])
                                plugin_token = spatial_tanh_zscore(plugin_logit)
                                plugin_map = plugin_token.view(tokens.shape[0], 1, side, side)
                                plugin_map = F.interpolate(
                                    plugin_map,
                                    size=(img_size, img_size),
                                    mode="bilinear",
                                    align_corners=True,
                                )
                                layer_plugin.append(plugin_map[:, 0])
                            layer_prescore_calibrated.append(rect_map)
                            continue
                    else:
                        rect_tokens = prescore_rectify_seg_tokens(
                            tokens=tokens,
                            q_def=q_def,
                            q_fp=q_fp,
                            target_axis=target_axis,
                            alpha=float(calibrator.get("effective_alpha", calibrator["alpha"])),
                            fp_weight=float(calibrator["fp_weight"]),
                            gate_sharpness=float(calibrator["gate_sharpness"]),
                        )
                    rect_map = calculate_similarity_map(
                        rect_tokens,
                        text_feature,
                        img_size,
                        test=True,
                        domain=DOMAINS[target_dataset],
                    )
                    layer_prescore_calibrated.append(rect_map[:, 0])
                side = int(round(math.sqrt(tokens.shape[1])))
                margin_map = margin.view(tokens.shape[0], 1, side, side)
                margin_map = F.interpolate(
                    margin_map,
                    size=(img_size, img_size),
                    mode="bilinear",
                    align_corners=True,
                )
                layer_parallel.append(margin_map[:, 0])
                full_margin_map = full_margin.view(tokens.shape[0], 1, side, side)
                full_margin_map = F.interpolate(
                    full_margin_map,
                    size=(img_size, img_size),
                    mode="bilinear",
                    align_corners=True,
                )
                layer_fullfeat.append(full_margin_map[:, 0])
                if layer_plugins[li] is not None:
                    plugin_logit = apply_plugin_from_scores(query_proj, q_def, q_fp, layer_plugins[li])
                    plugin_token = spatial_tanh_zscore(plugin_logit)
                    plugin_map = plugin_token.view(tokens.shape[0], 1, side, side)
                    plugin_map = F.interpolate(
                        plugin_map,
                        size=(img_size, img_size),
                        mode="bilinear",
                        align_corners=True,
                    )
                    layer_plugin.append(plugin_map[:, 0])

            baseline_map = torch.stack(layer_baseline, dim=1).sum(1)
            parallel_map = torch.stack(layer_parallel, dim=1).mean(1)
            fullfeat_map = torch.stack(layer_fullfeat, dim=1).mean(1) if layer_fullfeat else None
            plugin_map = torch.stack(layer_plugin, dim=1).mean(1) if layer_plugin else None
            plugin_alpha = None
            if plugin_map is not None:
                alpha_vals = [
                    float(p.get("learned_alpha", 1.0))
                    for p in layer_plugins
                    if p is not None
                ]
                plugin_alpha = torch.tensor(
                    float(np.mean(alpha_vals)) if alpha_vals else 1.0,
                    device=device,
                    dtype=baseline_map.dtype,
                )
            prescore_maps_by_alpha = {
                float(alpha): torch.stack(layer_maps, dim=1).sum(1)
                for alpha, layer_maps in layer_prescore_by_alpha.items()
                if layer_maps
            }
            prescore_calibrated_map = (
                torch.stack(layer_prescore_calibrated, dim=1).sum(1)
                if layer_prescore_calibrated
                else None
            )
            baseline_z = spatial_tanh_zscore_map(baseline_map)
            baseline_zt = baseline_z
            baseline_z_raw = spatial_zscore_map(baseline_map)
            parallel_z_raw = spatial_zscore_map(parallel_map)
            baseline_scale = baseline_map.flatten(1).std(dim=1, keepdim=True).clamp_min(1e-6).view(-1, 1, 1)
            candidate_gate = high_score_gate_map(baseline_z, candidate_topk_frac, gate_sharpness)
            hard_candidate_gate = high_score_hard_gate_map(baseline_z, candidate_topk_frac)

            for mode in blend_modes:
                for alpha in alphas:
                    key = f"{mode}_alpha_{alpha:g}"
                    if key not in candidate_maps_all:
                        candidate_maps_all[key] = []
                    if mode == "prescore_rectify":
                        candidate_map = prescore_maps_by_alpha[float(alpha)]
                    elif mode == "prescore_calibrated":
                        if prescore_calibrated_map is None:
                            raise RuntimeError("prescore_calibrated requires --prescore_calib_epochs > 0")
                        candidate_map = prescore_calibrated_map
                    elif mode == "prescore_calibrated_residual":
                        if prescore_calibrated_map is None:
                            raise RuntimeError("prescore_calibrated_residual requires --prescore_calib_epochs > 0")
                        # Strict C-TED rule: preserve the official host map and
                        # insert only a bounded source-calibrated residual.
                        candidate_map = baseline_map + alpha * candidate_gate * (prescore_calibrated_map - baseline_map)
                    elif mode == "prescore_calibrated_boost":
                        if prescore_calibrated_map is None:
                            raise RuntimeError("prescore_calibrated_boost requires --prescore_calib_epochs > 0")
                        candidate_map = baseline_map + alpha * F.relu(prescore_calibrated_map - baseline_map)
                    elif mode == "replace":
                        candidate_map = parallel_map
                    elif mode == "fullfeat_replace":
                        if fullfeat_map is None:
                            raise RuntimeError("fullfeat_replace requires source visual banks")
                        candidate_map = fullfeat_map
                    elif mode == "fullfeat_residual":
                        if fullfeat_map is None:
                            raise RuntimeError("fullfeat_residual requires source visual banks")
                        candidate_map = baseline_z_raw + alpha * candidate_gate * spatial_zscore_map(fullfeat_map)
                    elif mode == "blend":
                        candidate_map = baseline_map + alpha * baseline_scale * (parallel_map - baseline_z)
                    elif mode == "gate":
                        candidate_map = baseline_map * (1.0 + alpha * candidate_gate * parallel_map)
                    elif mode == "residual":
                        candidate_map = baseline_map + alpha * baseline_scale * candidate_gate * parallel_map
                    elif mode == "suppress":
                        candidate_map = baseline_map - alpha * baseline_scale * candidate_gate * F.relu(-parallel_map)
                    elif mode == "boost_suppress":
                        candidate_map = baseline_map + alpha * baseline_scale * candidate_gate * F.relu(parallel_map)
                        candidate_map = candidate_map - alpha * baseline_scale * candidate_gate * F.relu(-parallel_map)
                    elif mode == "z_global_residual":
                        candidate_map = baseline_z_raw + alpha * parallel_z_raw
                    elif mode == "z_residual":
                        candidate_map = baseline_z_raw + alpha * candidate_gate * parallel_z_raw
                    elif mode == "z_hard_residual":
                        candidate_map = baseline_z_raw + alpha * hard_candidate_gate * parallel_z_raw
                    elif mode == "zt_residual":
                        candidate_map = baseline_zt + alpha * candidate_gate * parallel_map
                    elif mode == "zt_hard_residual":
                        candidate_map = baseline_zt + alpha * hard_candidate_gate * parallel_map
                    elif mode == "rank_residual":
                        candidate_map = rerank_topk_preserve_values(
                            baseline_map=baseline_map,
                            rank_map=parallel_z_raw,
                            topk_frac=candidate_topk_frac,
                        )
                    elif mode == "rank_fused":
                        candidate_map = rerank_topk_preserve_values(
                            baseline_map=baseline_map,
                            rank_map=baseline_z_raw + alpha * parallel_z_raw,
                            topk_frac=candidate_topk_frac,
                        )
                    elif mode == "rank_positive":
                        candidate_map = rerank_topk_preserve_values(
                            baseline_map=baseline_map,
                            rank_map=baseline_z_raw + alpha * F.relu(parallel_z_raw),
                            topk_frac=candidate_topk_frac,
                        )
                    elif mode == "plugin_replace":
                        if plugin_map is None:
                            raise RuntimeError("plugin_replace requires --plugin_epochs > 0")
                        candidate_map = plugin_map
                    elif mode == "plugin_residual":
                        if plugin_map is None:
                            raise RuntimeError("plugin_residual requires --plugin_epochs > 0")
                        plugin_z = spatial_zscore_map(plugin_map)
                        candidate_map = baseline_z_raw + alpha * candidate_gate * plugin_z
                    elif mode == "plugin_rank_fused":
                        if plugin_map is None:
                            raise RuntimeError("plugin_rank_fused requires --plugin_epochs > 0")
                        plugin_z = spatial_zscore_map(plugin_map)
                        candidate_map = rerank_topk_preserve_values(
                            baseline_map=baseline_map,
                            rank_map=baseline_z_raw + alpha * plugin_z,
                            topk_frac=candidate_topk_frac,
                        )
                    elif mode == "plugin_learned_residual":
                        if plugin_map is None or plugin_alpha is None:
                            raise RuntimeError("plugin_learned_residual requires --plugin_epochs > 0")
                        plugin_z = spatial_zscore_map(plugin_map)
                        candidate_map = baseline_z_raw + plugin_alpha * candidate_gate * plugin_z
                    elif mode == "plugin_learned_rank_fused":
                        if plugin_map is None or plugin_alpha is None:
                            raise RuntimeError("plugin_learned_rank_fused requires --plugin_epochs > 0")
                        plugin_z = spatial_zscore_map(plugin_map)
                        candidate_map = rerank_topk_preserve_values(
                            baseline_map=baseline_map,
                            rank_map=baseline_z_raw + plugin_alpha * plugin_z,
                            topk_frac=candidate_topk_frac,
                        )
                    else:
                        raise ValueError(f"Unsupported blend mode: {mode}")
                    candidate_maps_all[key].append(candidate_map.detach().cpu().numpy())

            baseline_maps_all.append(baseline_map.detach().cpu().numpy())
            ours_maps_all.append(parallel_map.detach().cpu().numpy())
            masks_all.append(mask)
            labels_all.append(label)

        masks_np = np.concatenate(masks_all, axis=0)
        labels_np = np.concatenate(labels_all, axis=0)
        baseline_maps_np = np.concatenate(baseline_maps_all, axis=0)
        ours_maps_np = np.concatenate(ours_maps_all, axis=0)
        baseline_img_np = np.concatenate(baseline_img_all, axis=0)

        baseline_metrics = run_metrics(
            masks_np,
            labels_np,
            baseline_maps_np,
            baseline_img_np,
            class_name,
        )
        ours_metrics = run_metrics(
            masks_np,
            labels_np,
            ours_maps_np,
            baseline_img_np,
            class_name,
        )
        override_image_metrics_from_scores(
            ours_metrics,
            labels_np,
            image_scores_from_maps(
                candidate_image_mode,
                baseline_img_np,
                baseline_maps_np,
                ours_maps_np,
                candidate_topk_frac,
            ),
        )
        candidate_metrics = {}
        candidate_separability = {}
        for key, maps_list in candidate_maps_all.items():
            maps_np = np.concatenate(maps_list, axis=0)
            metrics = run_metrics(
                masks_np,
                labels_np,
                maps_np,
                baseline_img_np,
                class_name,
            )
            override_image_metrics_from_scores(
                metrics,
                labels_np,
                image_scores_from_maps(
                    candidate_image_mode,
                    baseline_img_np,
                    baseline_maps_np,
                    maps_np,
                    candidate_topk_frac,
                ),
            )
            candidate_metrics[key] = {
                "pixel_auc": float(metrics["pixel AUC"]),
                "pixel_ap": float(metrics["pixel AP"]),
                "pixel_pro": float(metrics.get("pixel PRO", 0.0)),
                "image_auc": float(metrics["image AUC"]),
                "image_ap": float(metrics["image AP"]),
            }
            if report_group_separability:
                candidate_separability[key] = map_group_separability(
                    masks_np,
                    labels_np,
                    maps_np,
                    hard_frac=separability_hard_frac,
                )
        separability = None
        if report_group_separability:
            separability = {
                "baseline": map_group_separability(
                    masks_np,
                    labels_np,
                    baseline_maps_np,
                    hard_frac=separability_hard_frac,
                ),
                "parallel": {
                    "replace": map_group_separability(
                        masks_np,
                        labels_np,
                        ours_maps_np,
                        hard_frac=separability_hard_frac,
                    )
                },
                "candidates": candidate_separability,
            }
        row = {
            "class_name": class_name,
            "num_images": int(labels_np.shape[0]),
            "baseline": {
                "pixel_auc": float(baseline_metrics["pixel AUC"]),
                "pixel_ap": float(baseline_metrics["pixel AP"]),
                "pixel_pro": float(baseline_metrics.get("pixel PRO", 0.0)),
                "image_auc": float(baseline_metrics["image AUC"]),
                "image_ap": float(baseline_metrics["image AP"]),
            },
            "parallel": {
                "pixel_auc": float(ours_metrics["pixel AUC"]),
                "pixel_ap": float(ours_metrics["pixel AP"]),
                "pixel_pro": float(ours_metrics.get("pixel PRO", 0.0)),
                "image_auc": float(ours_metrics["image AUC"]),
                "image_ap": float(ours_metrics["image AP"]),
            },
            "candidates": candidate_metrics,
        }
        if separability is not None:
            row["separability"] = separability
        results.append(row)

    def mean_of(mode: str, key: str) -> float:
        vals = [float(r[mode][key]) for r in results]
        return float(np.mean(vals)) if vals else float("nan")

    candidate_keys = sorted({key for row in results for key in row.get("candidates", {})})

    def mean_candidate(candidate_key: str, key: str) -> float:
        vals = [float(r["candidates"][candidate_key][key]) for r in results if candidate_key in r["candidates"]]
        return float(np.mean(vals)) if vals else float("nan")

    output = {
        "per_class": results,
        "source_plugin_summary": source_plugin_summary,
        "source_prescore_calib_summary": source_prescore_calib_summary,
        "mean": {
            "baseline": {
                "pixel_auc": mean_of("baseline", "pixel_auc"),
                "pixel_ap": mean_of("baseline", "pixel_ap"),
                "pixel_pro": mean_of("baseline", "pixel_pro"),
                "image_auc": mean_of("baseline", "image_auc"),
                "image_ap": mean_of("baseline", "image_ap"),
            },
            "parallel": {
                "pixel_auc": mean_of("parallel", "pixel_auc"),
                "pixel_ap": mean_of("parallel", "pixel_ap"),
                "pixel_pro": mean_of("parallel", "pixel_pro"),
                "image_auc": mean_of("parallel", "image_auc"),
                "image_ap": mean_of("parallel", "image_ap"),
            },
            "candidates": {
                candidate_key: {
                    "pixel_auc": mean_candidate(candidate_key, "pixel_auc"),
                    "pixel_ap": mean_candidate(candidate_key, "pixel_ap"),
                    "pixel_pro": mean_candidate(candidate_key, "pixel_pro"),
                    "image_auc": mean_candidate(candidate_key, "image_auc"),
                    "image_ap": mean_candidate(candidate_key, "image_ap"),
                }
                for candidate_key in candidate_keys
            },
        },
    }
    if report_group_separability:
        output["separability_mean"] = {
            "baseline": mean_separability(results, "baseline"),
            "parallel": {
                "replace": mean_separability(results, "parallel", "replace"),
            },
            "candidates": {
                candidate_key: mean_separability(results, "candidates", candidate_key)
                for candidate_key in candidate_keys
            },
        }
    return output


def parse_float_list(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def load_state_project_crop(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor], mode: str) -> Dict[str, int]:
    if mode == "strict":
        module.load_state_dict(state_dict, strict=True)
        return {"loaded": len(state_dict), "cropped": 0, "skipped": 0, "missing": 0}
    if mode == "skip":
        return {"loaded": 0, "cropped": 0, "skipped": len(state_dict), "missing": len(module.state_dict())}
    if mode != "project_crop":
        raise ValueError(f"unknown adapter_load_mode: {mode}")

    target = module.state_dict()
    patched: Dict[str, torch.Tensor] = {}
    loaded = 0
    cropped = 0
    skipped = 0
    for key, value in state_dict.items():
        if key not in target:
            skipped += 1
            continue
        dst = target[key]
        if tuple(value.shape) == tuple(dst.shape):
            patched[key] = value.to(dtype=dst.dtype)
            loaded += 1
            continue
        if not torch.is_floating_point(value) or value.ndim != dst.ndim:
            skipped += 1
            continue
        out = dst.clone()
        slices = tuple(slice(0, min(int(src), int(tgt))) for src, tgt in zip(value.shape, dst.shape))
        out[slices] = value[slices].to(dtype=dst.dtype)
        patched[key] = out
        cropped += 1

    incompatible = module.load_state_dict(patched, strict=False)
    return {
        "loaded": loaded,
        "cropped": cropped,
        "skipped": skipped,
        "missing": len(incompatible.missing_keys),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, required=True, choices=sorted(PRESET_CONFIGS))
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_name", type=str, default="ViT-L-14-336")
    ap.add_argument("--pretrained", type=str, default="openai")
    ap.add_argument("--adapter_load_mode", type=str, default="strict", choices=["strict", "project_crop", "skip"])
    ap.add_argument("--levels", type=str, default="6,12,18,24")
    ap.add_argument("--text_adapt_until", type=int, default=3)
    ap.add_argument("--image_adapt_until", type=int, default=6)
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--class_name", type=str, default="")
    ap.add_argument("--source_limit_per_class", type=int, default=0)
    ap.add_argument("--target_limit_per_class", type=int, default=0)
    ap.add_argument("--max_fp_per_class", type=int, default=2048)
    ap.add_argument("--max_defect_per_class", type=int, default=2048)
    ap.add_argument("--normal_topk_frac", type=float, default=0.01)
    ap.add_argument("--source_fp_mode", type=str, default="hardfp", choices=["hardfp", "random_normal", "easy_normal"])
    ap.add_argument("--defect_topk_frac", type=float, default=0.25)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--fp_weight", type=float, default=1.0)
    ap.add_argument("--alphas", type=str, default="1.0")
    ap.add_argument("--blend_modes", type=str, default="replace")
    ap.add_argument("--candidate_topk_frac", type=float, default=0.10)
    ap.add_argument("--gate_sharpness", type=float, default=6.0)
    ap.add_argument(
        "--candidate_image_mode",
        type=str,
        default="fused",
        choices=[
            "fused",
            "baseline",
            "map_max",
            "map_topk",
            "baseline_plus_delta_topk",
            "baseline_plus_map_topk",
        ],
    )
    ap.add_argument("--cal_pro", action="store_true")
    ap.add_argument("--aupro_max_step", type=int, default=50)
    ap.add_argument("--report_group_separability", action="store_true")
    ap.add_argument("--separability_hard_frac", type=float, default=0.01)
    ap.add_argument("--plugin_epochs", type=int, default=0)
    ap.add_argument("--plugin_lr", type=float, default=0.05)
    ap.add_argument("--plugin_max_train_points", type=int, default=4096)
    ap.add_argument("--plugin_loss", type=str, default="bce", choices=["bce", "pair_rank", "rank_contrast", "support_rank", "host_rank"])
    ap.add_argument("--plugin_rank_margin", type=float, default=0.25)
    ap.add_argument("--plugin_rank_temp", type=float, default=0.1)
    ap.add_argument("--plugin_contrast_weight", type=float, default=0.1)
    ap.add_argument("--plugin_contrast_margin", type=float, default=0.2)
    ap.add_argument("--plugin_hidden_dim", type=int, default=16)
    ap.add_argument("--plugin_learn_alpha", action="store_true")
    ap.add_argument("--plugin_alpha_init", type=float, default=0.1)
    ap.add_argument("--plugin_alpha_max", type=float, default=2.0)
    ap.add_argument("--plugin_alpha_reg", type=float, default=0.0)
    ap.add_argument("--plugin_hardpair_frac", type=float, default=0.0)
    ap.add_argument("--prescore_calib_epochs", type=int, default=0)
    ap.add_argument("--prescore_calib_lr", type=float, default=0.02)
    ap.add_argument("--prescore_calib_max_train_points", type=int, default=8192)
    ap.add_argument("--prescore_calib_rank_margin", type=float, default=0.05)
    ap.add_argument("--prescore_calib_rank_temp", type=float, default=0.05)
    ap.add_argument("--prescore_calib_alpha_init", type=float, default=0.2)
    ap.add_argument("--prescore_calib_alpha_max", type=float, default=4.0)
    ap.add_argument("--prescore_calib_alpha_reg", type=float, default=0.0)
    ap.add_argument("--prescore_calib_fp_weight_init", type=float, default=None)
    ap.add_argument("--prescore_calib_hardpair_frac", type=float, default=0.75)
    ap.add_argument("--prescore_calib_learn_fp_weight", action="store_true")
    ap.add_argument("--prescore_calib_learn_gate", action="store_true")
    ap.add_argument("--prescore_calib_nonnegative_alpha", action="store_true")
    ap.add_argument(
        "--prescore_calib_transport_mode",
        type=str,
        default="additive_margin",
        choices=["additive_margin", "parallel_tanh", "subspace_tanh"],
    )
    ap.add_argument("--prescore_calib_subspace_rank", type=int, default=4)
    ap.add_argument(
        "--prescore_calib_subspace_readout_mode",
        type=str,
        default="replacement",
        choices=["replacement", "host_residual"],
    )
    ap.add_argument(
        "--prescore_calib_subspace_basis_control",
        type=str,
        default="source",
        choices=["source", "random"],
    )
    ap.add_argument(
        "--prescore_calib_pair_label_control",
        type=str,
        default="correct",
        choices=["correct", "shuffle", "swap"],
    )
    ap.add_argument("--prescore_calib_preserve_host_margin_weight", type=float, default=0.0)
    ap.add_argument(
        "--prescore_calib_output_mode",
        type=str,
        default="host_text",
        choices=["host_text", "subspace_score", "subspace_zscore", "host_residual", "host_residual_zscore"],
    )
    ap.add_argument("--prescore_map_calib_epochs", type=int, default=0)
    ap.add_argument("--prescore_map_calib_lr", type=float, default=0.01)
    ap.add_argument("--prescore_map_calib_max_images", type=int, default=256)
    ap.add_argument("--prescore_map_calib_sep_margin", type=float, default=0.25)
    ap.add_argument("--prescore_map_calib_sep_temp", type=float, default=0.1)
    ap.add_argument(
        "--prescore_map_calib_hardfp_mode",
        type=str,
        default="baseline_rank",
        choices=["baseline_rank", "online_rank"],
    )
    ap.add_argument("--prescore_map_calib_aggregate_loss", action="store_true")
    ap.add_argument("--prescore_map_calib_preserve_weight", type=float, default=0.25)
    ap.add_argument(
        "--prescore_map_calib_preserve_mode",
        type=str,
        default="l1_background",
        choices=["l1_background", "monotonic_topk", "online_hardfp"],
    )
    ap.add_argument("--prescore_map_calib_topk_frac", type=float, default=0.05)
    ap.add_argument("--prescore_map_calib_learn_layer_beta", action="store_true")
    ap.add_argument("--prescore_map_calib_beta_init", type=float, default=0.5)
    ap.add_argument("--prescore_map_calib_beta_reg", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    setup_seed(args.seed)
    _add_aaclip_to_syspath()
    ensure_aaclip_pretrained_weight()

    from model.adapter import AdaptedCLIP
    from model.clip import create_model

    cfg = dict(PRESET_CONFIGS[args.preset])
    if args.ckpt_dir:
        cfg["ckpt_dir"] = args.ckpt_dir
    ckpt_dir = cfg["ckpt_dir"]
    device = torch.device(args.device)
    levels = parse_int_list(args.levels)

    clip_model = create_model(
        model_name=args.model_name,
        img_size=int(args.img_size),
        device=device,
        pretrained=args.pretrained,
        force_image_size=int(args.img_size),
        require_pretrained=True,
    )
    clip_model.eval()
    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=0.1,
        image_adapt_weight=0.1,
        text_adapt_until=int(args.text_adapt_until),
        image_adapt_until=int(args.image_adapt_until),
        levels=levels,
        relu=False,
    ).to(device)
    model.eval()

    image_ckpt = os.path.join(ckpt_dir, "image_adapter.pth")
    if not os.path.exists(image_ckpt):
        raise FileNotFoundError(image_ckpt)
    ckpt = torch.load(image_ckpt, map_location="cpu")
    image_load_info = load_state_project_crop(
        model.image_adapter,
        ckpt["image_adapter"],
        mode=args.adapter_load_mode,
    )

    text_ckpt = os.path.join(ckpt_dir, "text_adapter.pth")
    text_load_info = {"loaded": 0, "cropped": 0, "skipped": 0, "missing": 0}
    if os.path.exists(text_ckpt):
        ckpt_t = torch.load(text_ckpt, map_location="cpu")
        text_load_info = load_state_project_crop(
            model.text_adapter,
            ckpt_t["text_adapter"],
            mode=args.adapter_load_mode,
        )
    print(
        json.dumps(
            {
                "stage": "aaclip_model_ready",
                "model_name": args.model_name,
                "pretrained": args.pretrained,
                "adapter_load_mode": args.adapter_load_mode,
                "levels": list(model.levels),
                "vision_width": int(model.vision_width),
                "vision_depth": int(model.vision_depth),
                "text_width": int(model.text_width),
                "output_dim": int(model.output_dim),
                "image_adapter_load": image_load_info,
                "text_adapter_load": text_load_info,
            }
        ),
        flush=True,
    )

    for p in model.parameters():
        p.requires_grad = False

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / "neurips2026" / "results" / "bank_cache"
    banks = get_or_collect_source_banks(
        model=model,
        preset=args.preset,
        model_name=f"{args.model_name}:{args.pretrained}",
        adapter_load_mode=args.adapter_load_mode,
        source_dataset=cfg["source_dataset"],
        source_root=cfg["source_root"],
        ckpt_dir=ckpt_dir,
        cache_dir=cache_dir,
        img_size=int(args.img_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
        max_fp_per_class=int(args.max_fp_per_class),
        max_defect_per_class=int(args.max_defect_per_class),
        normal_topk_frac=float(args.normal_topk_frac),
        defect_topk_frac=float(args.defect_topk_frac),
        seed=int(args.seed),
        source_limit_per_class=int(args.source_limit_per_class) or None,
        source_fp_mode=args.source_fp_mode,
    )

    summary = evaluate_target_dataset(
        model=model,
        source_dataset=cfg["source_dataset"],
        source_root=cfg["source_root"],
        target_dataset=cfg["target_dataset"],
        target_root=cfg["target_root"],
        img_size=int(args.img_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
        banks=banks,
        tau=float(args.tau),
        fp_weight=float(args.fp_weight),
        alphas=parse_float_list(args.alphas),
        blend_modes=parse_str_list(args.blend_modes),
        candidate_topk_frac=float(args.candidate_topk_frac),
        gate_sharpness=float(args.gate_sharpness),
        candidate_image_mode=args.candidate_image_mode,
        cal_pro=bool(args.cal_pro),
        aupro_max_step=int(args.aupro_max_step),
        report_group_separability=bool(args.report_group_separability),
        separability_hard_frac=float(args.separability_hard_frac),
        class_name_filter=args.class_name or None,
        target_limit_per_class=int(args.target_limit_per_class) or None,
        seed=int(args.seed),
        plugin_epochs=int(args.plugin_epochs),
        plugin_lr=float(args.plugin_lr),
        plugin_max_train_points=int(args.plugin_max_train_points),
        plugin_loss=args.plugin_loss,
        plugin_rank_margin=float(args.plugin_rank_margin),
        plugin_rank_temp=float(args.plugin_rank_temp),
        plugin_contrast_weight=float(args.plugin_contrast_weight),
        plugin_contrast_margin=float(args.plugin_contrast_margin),
        plugin_hidden_dim=int(args.plugin_hidden_dim),
        plugin_learn_alpha=bool(args.plugin_learn_alpha),
        plugin_alpha_init=float(args.plugin_alpha_init),
        plugin_alpha_max=float(args.plugin_alpha_max),
        plugin_alpha_reg=float(args.plugin_alpha_reg),
        plugin_hardpair_frac=float(args.plugin_hardpair_frac),
        prescore_calib_epochs=int(args.prescore_calib_epochs),
        prescore_calib_lr=float(args.prescore_calib_lr),
        prescore_calib_max_train_points=int(args.prescore_calib_max_train_points),
        prescore_calib_rank_margin=float(args.prescore_calib_rank_margin),
        prescore_calib_rank_temp=float(args.prescore_calib_rank_temp),
        prescore_calib_alpha_init=float(args.prescore_calib_alpha_init),
        prescore_calib_alpha_max=float(args.prescore_calib_alpha_max),
        prescore_calib_alpha_reg=float(args.prescore_calib_alpha_reg),
        prescore_calib_fp_weight_init=float(args.fp_weight if args.prescore_calib_fp_weight_init is None else args.prescore_calib_fp_weight_init),
        prescore_calib_hardpair_frac=float(args.prescore_calib_hardpair_frac),
        prescore_calib_learn_fp_weight=bool(args.prescore_calib_learn_fp_weight),
        prescore_calib_learn_gate=bool(args.prescore_calib_learn_gate),
        prescore_calib_nonnegative_alpha=bool(args.prescore_calib_nonnegative_alpha),
        prescore_calib_transport_mode=args.prescore_calib_transport_mode,
        prescore_calib_subspace_rank=int(args.prescore_calib_subspace_rank),
        prescore_calib_subspace_readout_mode=args.prescore_calib_subspace_readout_mode,
        prescore_calib_subspace_basis_control=args.prescore_calib_subspace_basis_control,
        prescore_calib_pair_label_control=args.prescore_calib_pair_label_control,
        prescore_calib_preserve_host_margin_weight=float(args.prescore_calib_preserve_host_margin_weight),
        prescore_calib_output_mode=args.prescore_calib_output_mode,
        prescore_map_calib_epochs=int(args.prescore_map_calib_epochs),
        prescore_map_calib_lr=float(args.prescore_map_calib_lr),
        prescore_map_calib_max_images=int(args.prescore_map_calib_max_images),
        prescore_map_calib_sep_margin=float(args.prescore_map_calib_sep_margin),
        prescore_map_calib_sep_temp=float(args.prescore_map_calib_sep_temp),
        prescore_map_calib_hardfp_mode=args.prescore_map_calib_hardfp_mode,
        prescore_map_calib_aggregate_loss=bool(args.prescore_map_calib_aggregate_loss),
        prescore_map_calib_preserve_weight=float(args.prescore_map_calib_preserve_weight),
        prescore_map_calib_preserve_mode=args.prescore_map_calib_preserve_mode,
        prescore_map_calib_topk_frac=float(args.prescore_map_calib_topk_frac),
        prescore_map_calib_learn_layer_beta=bool(args.prescore_map_calib_learn_layer_beta),
        prescore_map_calib_beta_init=float(args.prescore_map_calib_beta_init),
        prescore_map_calib_beta_reg=float(args.prescore_map_calib_beta_reg),
    )
    def _bank_count(value):
        if value is None:
            return None
        if hasattr(value, "shape"):
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            counts = [_bank_count(item) for item in value]
            counts = [count for count in counts if count is not None]
            return int(sum(counts)) if counts else 0
        return None

    summary.update(
        {
            "preset": args.preset,
            "source_dataset": cfg["source_dataset"],
            "target_dataset": cfg["target_dataset"],
            "source_root": cfg["source_root"],
            "target_root": cfg["target_root"],
            "ckpt_dir": ckpt_dir,
            "class_name": args.class_name or None,
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "adapter_load_mode": args.adapter_load_mode,
            "levels": list(model.levels),
            "vision_width": int(model.vision_width),
            "vision_depth": int(model.vision_depth),
            "text_width": int(model.text_width),
            "output_dim": int(model.output_dim),
            "img_size": int(args.img_size),
            "tau": float(args.tau),
            "fp_weight": float(args.fp_weight),
            "alphas": parse_float_list(args.alphas),
            "blend_modes": parse_str_list(args.blend_modes),
            "max_fp_per_class": int(args.max_fp_per_class),
            "max_defect_per_class": int(args.max_defect_per_class),
            "bank": {
                "num_fp": _bank_count(banks.get("fp")),
                "num_defect": _bank_count(banks.get("defect")),
                "max_fp_per_class": int(args.max_fp_per_class),
                "max_defect_per_class": int(args.max_defect_per_class),
                "normal_topk_frac": float(args.normal_topk_frac),
                "source_fp_mode": args.source_fp_mode,
            },
            "normal_topk_frac": float(args.normal_topk_frac),
            "source_fp_mode": args.source_fp_mode,
            "defect_topk_frac": float(args.defect_topk_frac),
            "candidate_topk_frac": float(args.candidate_topk_frac),
            "gate_sharpness": float(args.gate_sharpness),
            "candidate_image_mode": args.candidate_image_mode,
            "cal_pro": bool(args.cal_pro),
            "aupro_max_step": int(args.aupro_max_step),
            "source_plugin": {
                "calibration": f"source_bank_{args.plugin_loss}_no_target",
                "axis": "source_global_mean_text_axis",
                "training_scope": "once_per_source_host_run",
                "feature_names": ["axis_projection", "defect_support", "hardfp_support", "support_margin", "support_mass"],
                "plugin_epochs": int(args.plugin_epochs),
                "plugin_lr": float(args.plugin_lr),
                "plugin_max_train_points": int(args.plugin_max_train_points),
                "plugin_loss": args.plugin_loss,
                "plugin_rank_margin": float(args.plugin_rank_margin),
                "plugin_rank_temp": float(args.plugin_rank_temp),
                "plugin_contrast_weight": float(args.plugin_contrast_weight),
                "plugin_contrast_margin": float(args.plugin_contrast_margin),
                "plugin_hidden_dim": int(args.plugin_hidden_dim),
                "plugin_learn_alpha": bool(args.plugin_learn_alpha),
                "plugin_alpha_init": float(args.plugin_alpha_init),
                "plugin_alpha_max": float(args.plugin_alpha_max),
                "plugin_alpha_reg": float(args.plugin_alpha_reg),
                "plugin_hardpair_frac": float(args.plugin_hardpair_frac),
            }
            if int(args.plugin_epochs) > 0
            else None,
            "source_prescore_calibration": {
                "calibration": "source_map_sep_preserve_no_target"
                if int(args.prescore_map_calib_epochs) > 0
                else "source_prescore_feature_rank_no_target",
                "training_scope": "once_per_source_host_backbone_run",
                "feature_names": ["axis_projection", "defect_support", "hardfp_support"],
                "prescore_calib_epochs": int(args.prescore_calib_epochs),
                "prescore_calib_lr": float(args.prescore_calib_lr),
                "prescore_calib_max_train_points": int(args.prescore_calib_max_train_points),
                "prescore_calib_rank_margin": float(args.prescore_calib_rank_margin),
                "prescore_calib_rank_temp": float(args.prescore_calib_rank_temp),
                "prescore_calib_alpha_init": float(args.prescore_calib_alpha_init),
                "prescore_calib_alpha_max": float(args.prescore_calib_alpha_max),
                "prescore_calib_alpha_reg": float(args.prescore_calib_alpha_reg),
                "prescore_calib_fp_weight_init": float(args.fp_weight if args.prescore_calib_fp_weight_init is None else args.prescore_calib_fp_weight_init),
                "prescore_calib_hardpair_frac": float(args.prescore_calib_hardpair_frac),
                "prescore_calib_learn_fp_weight": bool(args.prescore_calib_learn_fp_weight),
                "prescore_calib_learn_gate": bool(args.prescore_calib_learn_gate),
                "prescore_calib_nonnegative_alpha": bool(args.prescore_calib_nonnegative_alpha),
                "prescore_calib_transport_mode": args.prescore_calib_transport_mode,
                "prescore_calib_subspace_rank": int(args.prescore_calib_subspace_rank),
                "prescore_calib_subspace_readout_mode": args.prescore_calib_subspace_readout_mode,
                "prescore_calib_subspace_basis_control": args.prescore_calib_subspace_basis_control,
                "prescore_calib_pair_label_control": args.prescore_calib_pair_label_control,
                "prescore_calib_preserve_host_margin_weight": float(args.prescore_calib_preserve_host_margin_weight),
                "prescore_calib_output_mode": args.prescore_calib_output_mode,
                "prescore_map_calib_epochs": int(args.prescore_map_calib_epochs),
                "prescore_map_calib_lr": float(args.prescore_map_calib_lr),
                "prescore_map_calib_max_images": int(args.prescore_map_calib_max_images),
                "prescore_map_calib_sep_margin": float(args.prescore_map_calib_sep_margin),
                "prescore_map_calib_sep_temp": float(args.prescore_map_calib_sep_temp),
                "prescore_map_calib_hardfp_mode": args.prescore_map_calib_hardfp_mode,
                "prescore_map_calib_aggregate_loss": bool(args.prescore_map_calib_aggregate_loss),
                "prescore_map_calib_preserve_weight": float(args.prescore_map_calib_preserve_weight),
                "prescore_map_calib_preserve_mode": args.prescore_map_calib_preserve_mode,
                "prescore_map_calib_topk_frac": float(args.prescore_map_calib_topk_frac),
                "prescore_map_calib_learn_layer_beta": bool(args.prescore_map_calib_learn_layer_beta),
                "prescore_map_calib_beta_init": float(args.prescore_map_calib_beta_init),
                "prescore_map_calib_beta_reg": float(args.prescore_map_calib_beta_reg),
            }
            if int(args.prescore_calib_epochs) > 0 or int(args.prescore_map_calib_epochs) > 0
            else None,
        }
    )
    out_path = save_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["mean"], indent=2))
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
