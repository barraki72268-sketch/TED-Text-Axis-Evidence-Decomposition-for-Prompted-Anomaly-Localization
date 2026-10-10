from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/mnt/data/pilab-kingjinyoung/ted-reproduce-20261008/snapshot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurips2026.common_backbones import create_imagebind_raw_backbone  # noqa: E402
from neurips2026.scripts.official_parallel_test_rawclip import (  # noqa: E402
    activate_margin_map,
    build_dataset,
    build_prompt_groups,
    build_transforms,
    calculate_fast_aupro,
    farthest_point_subsample,
    logmeanexp_negative_sqdist_1d,
    pixel_average_precision,
    precompute_aupro_regions,
    save_results,
    smooth_map,
    topk_mean,
    zscore_tensor,
)
from metrics import image_level_metrics, pixel_level_metrics  # noqa: E402


def encode_prompt_pair(backbone, cls_name: str, prompt_mode: str) -> dict[str, torch.Tensor]:
    normal_prompts, abnormal_prompts = build_prompt_groups(prompt_mode, cls_name)
    with torch.no_grad():
        e_norm = F.normalize(backbone.encode_text(backbone.tokenizer(normal_prompts)).float(), dim=-1).mean(dim=0)
        e_anom = F.normalize(backbone.encode_text(backbone.tokenizer(abnormal_prompts)).float(), dim=-1).mean(dim=0)
    e_norm = F.normalize(e_norm, dim=0)
    e_anom = F.normalize(e_anom, dim=0)
    axis = F.normalize(e_anom - e_norm, dim=0)
    return {
        "norm_joint": e_norm,
        "anom_joint": e_anom,
        "axis_joint": axis,
    }


def compute_layer_host_and_ted(
    patch_tokens: torch.Tensor,
    text_pair: dict[str, torch.Tensor],
    fp_bank: torch.Tensor,
    def_bank: torch.Tensor,
    tau: float,
    fp_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    patch = F.normalize(patch_tokens.float(), dim=-1)
    logits = patch @ torch.stack([text_pair["norm_joint"], text_pair["anom_joint"]], dim=1)
    host_score = (logits / 0.07).softmax(dim=-1)[:, 1]
    axis = text_pair["axis_joint"]
    coeff = patch @ axis
    fp_bank = fp_bank.to(patch.device)
    def_bank = def_bank.to(patch.device)
    fp_coeff = fp_bank @ axis if fp_bank.numel() else torch.empty((0,), device=patch.device)
    def_coeff = def_bank @ axis if def_bank.numel() else torch.empty((0,), device=patch.device)
    q_fp = logmeanexp_negative_sqdist_1d(coeff, fp_coeff, tau=tau)
    q_def = logmeanexp_negative_sqdist_1d(coeff, def_coeff, tau=tau)
    return host_score, q_def - fp_weight * q_fp


def _sample_indices(mask: torch.Tensor, max_points: int) -> torch.Tensor:
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    if idx.numel() <= max_points:
        return idx
    perm = torch.randperm(idx.numel(), device=idx.device)[:max_points]
    return idx[perm]


def _collect_calibration_pairs(args, backbone, image_transform, mask_transform, features_list, fp_banks, def_banks):
    train_ds = build_dataset(args.source_root, image_transform, mask_transform, "train")
    test_ds = build_dataset(args.source_root, image_transform, mask_transform, "test")
    per_good = {cls_name: 0 for cls_name in train_ds.obj_list}
    per_defect = {cls_name: 0 for cls_name in test_ds.obj_list}
    data = {out_idx: {"fp_host": [], "fp_res": [], "def_host": [], "def_res": []} for out_idx in range(len(features_list))}
    with torch.no_grad():
        for idx in range(len(train_ds)):
            if all(v >= args.calib_max_good_per_class for v in per_good.values()):
                break
            item = train_ds[idx]
            cls_name = item["cls_name"]
            if args.source_exclude_class is not None and cls_name == args.source_exclude_class:
                continue
            if per_good[cls_name] >= args.calib_max_good_per_class:
                continue
            per_good[cls_name] += 1
            text_pair = encode_prompt_pair(backbone, cls_name, args.prompt_mode)
            image = item["img"].unsqueeze(0).to(backbone.device)
            patch_dict = backbone.extract_layer_patch_tokens(image, target_layers=features_list, project_to_joint=True)
            for out_idx, layer in enumerate(features_list):
                host_score, margin = compute_layer_host_and_ted(
                    patch_dict[layer][0], text_pair, fp_banks[out_idx], def_banks[out_idx], args.tau, args.fp_weight
                )
                k = max(1, int(round(host_score.numel() * args.calib_hard_frac)))
                idx_top = torch.topk(host_score, k=k, largest=True).indices
                if idx_top.numel() > args.calib_max_fp_per_image:
                    idx_top = idx_top[torch.randperm(idx_top.numel(), device=idx_top.device)[: args.calib_max_fp_per_image]]
                data[out_idx]["fp_host"].append(host_score[idx_top].detach().cpu())
                data[out_idx]["fp_res"].append(zscore_tensor(margin)[idx_top].detach().cpu())

        for idx in range(len(test_ds)):
            if all(v >= args.calib_max_defect_per_class for v in per_defect.values()):
                break
            item = test_ds[idx]
            cls_name = item["cls_name"]
            if int(item["anomaly"]) == 0:
                continue
            if args.source_exclude_class is not None and cls_name == args.source_exclude_class:
                continue
            if per_defect[cls_name] >= args.calib_max_defect_per_class:
                continue
            per_defect[cls_name] += 1
            text_pair = encode_prompt_pair(backbone, cls_name, args.prompt_mode)
            image = item["img"].unsqueeze(0).to(backbone.device)
            gt_mask = item["img_mask"].unsqueeze(0).to(backbone.device).float()
            patch_dict = backbone.extract_layer_patch_tokens(image, target_layers=features_list, project_to_joint=True)
            for out_idx, layer in enumerate(features_list):
                host_score, margin = compute_layer_host_and_ted(
                    patch_dict[layer][0], text_pair, fp_banks[out_idx], def_banks[out_idx], args.tau, args.fp_weight
                )
                side = int(round(math.sqrt(host_score.numel())))
                gt_small = F.interpolate(gt_mask, size=(side, side), mode="nearest")[0, 0].reshape(-1) > 0.5
                idx_def = _sample_indices(gt_small, args.calib_max_defect_per_image)
                if idx_def.numel() == 0:
                    continue
                data[out_idx]["def_host"].append(host_score[idx_def].detach().cpu())
                data[out_idx]["def_res"].append(zscore_tensor(margin)[idx_def].detach().cpu())
    packed = {}
    for out_idx, chunks in data.items():
        packed[out_idx] = {}
        for key, vals in chunks.items():
            tensor = torch.cat(vals, dim=0).float() if vals else torch.empty((0,), dtype=torch.float32)
            if tensor.numel() > args.calib_max_train_points:
                perm = torch.randperm(tensor.numel())[: args.calib_max_train_points]
                tensor = tensor[perm]
            packed[out_idx][key] = tensor
    return packed


def train_source_calibrators(args, backbone, image_transform, mask_transform, features_list, fp_banks, def_banks):
    pairs = _collect_calibration_pairs(args, backbone, image_transform, mask_transform, features_list, fp_banks, def_banks)
    calibrators = {}
    for out_idx, pack in pairs.items():
        fp_host = pack["fp_host"].to(backbone.device)
        fp_res = pack["fp_res"].to(backbone.device)
        def_host = pack["def_host"].to(backbone.device)
        def_res = pack["def_res"].to(backbone.device)
        if fp_host.numel() == 0 or def_host.numel() == 0:
            calibrators[out_idx] = {"eta": 0.0, "num_fp": int(fp_host.numel()), "num_defect": int(def_host.numel())}
            continue
        init = max(-args.calib_eta_max + 1e-6, min(args.calib_eta_max - 1e-6, args.calib_eta_init))
        raw_eta = torch.nn.Parameter(torch.atanh(torch.tensor(init / args.calib_eta_max, device=backbone.device)))
        opt = torch.optim.Adam([raw_eta], lr=args.calib_lr)
        n_pairs = min(fp_host.numel(), def_host.numel())
        for _ in range(args.calib_epochs):
            fp_idx = torch.randint(0, fp_host.numel(), (n_pairs,), device=backbone.device)
            def_idx = torch.randint(0, def_host.numel(), (n_pairs,), device=backbone.device)
            eta = args.calib_eta_max * torch.tanh(raw_eta)
            fp_score = fp_host[fp_idx] + eta * fp_host.std().clamp_min(1e-6) * fp_res[fp_idx]
            def_score = def_host[def_idx] + eta * def_host.std().clamp_min(1e-6) * def_res[def_idx]
            rank_loss = F.softplus(-(def_score - fp_score - args.calib_rank_margin) / args.calib_rank_temp).mean()
            loss = rank_loss + args.calib_eta_reg * eta.pow(2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            eta = float((args.calib_eta_max * torch.tanh(raw_eta)).detach().cpu())
        calibrators[out_idx] = {
            "eta": eta,
            "num_fp": int(fp_host.numel()),
            "num_defect": int(def_host.numel()),
            "fp_host_mean": float(fp_host.mean().detach().cpu()),
            "def_host_mean": float(def_host.mean().detach().cpu()),
            "fp_res_mean": float(fp_res.mean().detach().cpu()),
            "def_res_mean": float(def_res.mean().detach().cpu()),
        }
    print(json.dumps({"stage": "rawimagebind_source_calibrators_trained", "calibrators": calibrators}, indent=2), flush=True)
    return calibrators


def collect_source_patch_banks(
    backbone,
    source_root: str,
    image_transform,
    mask_transform,
    features_list: list[int],
    prompt_mode: str,
    heldout_class: str | None,
    max_good_per_class: int,
    max_defect_per_class: int,
    hard_frac: float,
    max_bank_per_layer: int,
    max_fp_per_image: int,
    max_defect_per_image: int,
):
    train_ds = build_dataset(source_root, image_transform, mask_transform, "train")
    test_ds = build_dataset(source_root, image_transform, mask_transform, "test")

    fp_chunks: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(features_list))}
    def_chunks: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(features_list))}
    per_good = {cls_name: 0 for cls_name in train_ds.obj_list}
    per_defect = {cls_name: 0 for cls_name in test_ds.obj_list}

    with torch.no_grad():
        for idx in range(len(train_ds)):
            item = train_ds[idx]
            cls_name = item["cls_name"]
            if heldout_class is not None and cls_name == heldout_class:
                continue
            if per_good[cls_name] >= max_good_per_class:
                continue
            per_good[cls_name] += 1

            text_pair = encode_prompt_pair(backbone, cls_name, prompt_mode)
            image = item["img"].unsqueeze(0).to(backbone.device)
            patch_dict = backbone.extract_layer_patch_tokens(image, target_layers=features_list, project_to_joint=True)

            for out_idx, layer in enumerate(features_list):
                patch = F.normalize(patch_dict[layer][0].float(), dim=-1)
                logits = patch @ torch.stack([text_pair["norm_joint"], text_pair["anom_joint"]], dim=1)
                patch_scores = (logits / 0.07).softmax(dim=-1)[:, 1]
                k = max(1, int(patch_scores.numel() * hard_frac))
                top_idx = torch.topk(patch_scores, k=k, largest=True).indices
                selected = patch[top_idx]
                if selected.shape[0] > max_fp_per_image:
                    selected = farthest_point_subsample(selected, max_fp_per_image)
                fp_chunks[out_idx].append(selected.cpu())

        for idx in range(len(test_ds)):
            item = test_ds[idx]
            cls_name = item["cls_name"]
            anomaly_flag = int(item["anomaly"])
            if anomaly_flag == 0:
                continue
            if heldout_class is not None and cls_name == heldout_class:
                continue
            if per_defect[cls_name] >= max_defect_per_class:
                continue
            per_defect[cls_name] += 1

            image = item["img"].unsqueeze(0).to(backbone.device)
            gt_mask = item["img_mask"].unsqueeze(0).to(backbone.device).float()
            patch_dict = backbone.extract_layer_patch_tokens(image, target_layers=features_list, project_to_joint=True)
            for out_idx, layer in enumerate(features_list):
                patch = F.normalize(patch_dict[layer][0].float(), dim=-1)
                side = int(round(math.sqrt(patch.shape[0])))
                gt_small = F.interpolate(gt_mask, size=(side, side), mode="nearest")[0, 0].reshape(-1) > 0.5
                if int(gt_small.sum().item()) == 0:
                    continue
                selected = patch[gt_small]
                if selected.shape[0] > max_defect_per_image:
                    selected = farthest_point_subsample(selected, max_defect_per_image)
                def_chunks[out_idx].append(selected.cpu())

    fp_banks: dict[int, torch.Tensor] = {}
    def_banks: dict[int, torch.Tensor] = {}
    bank_stats: dict[int, dict] = {}
    token_dim = backbone.joint_feature_dimension
    for out_idx in range(len(features_list)):
        fp_bank = torch.cat(fp_chunks[out_idx], dim=0) if fp_chunks[out_idx] else torch.empty((0, token_dim))
        def_bank = torch.cat(def_chunks[out_idx], dim=0) if def_chunks[out_idx] else torch.empty((0, token_dim))
        if fp_bank.shape[0] > max_bank_per_layer:
            fp_bank = farthest_point_subsample(fp_bank, max_bank_per_layer)
        if def_bank.shape[0] > max_bank_per_layer:
            def_bank = farthest_point_subsample(def_bank, max_bank_per_layer)
        fp_banks[out_idx] = F.normalize(fp_bank.float(), dim=-1) if fp_bank.numel() else fp_bank.float()
        def_banks[out_idx] = F.normalize(def_bank.float(), dim=-1) if def_bank.numel() else def_bank.float()
        bank_stats[out_idx] = {"num_fp": int(fp_banks[out_idx].shape[0]), "num_defect": int(def_banks[out_idx].shape[0])}
    return fp_banks, def_banks, bank_stats


def default_bank_cache_path(
    source_root: str,
    source_dataset: str,
    heldout_class: str | None,
    image_size: int,
    features_list: list[int],
    prompt_mode: str,
    hard_frac: float,
    max_good_per_class: int,
    max_defect_per_class: int,
    max_bank_per_layer: int,
    max_fp_per_image: int,
    max_defect_per_image: int,
) -> Path:
    cache_cfg = {
        "source_root": str(source_root),
        "source_dataset": source_dataset,
        "heldout_class": heldout_class,
        "image_size": int(image_size),
        "features_list": list(features_list),
        "prompt_mode": prompt_mode,
        "hard_frac": float(hard_frac),
        "max_good_per_class": int(max_good_per_class),
        "max_defect_per_class": int(max_defect_per_class),
        "max_bank_per_layer": int(max_bank_per_layer),
        "max_fp_per_image": int(max_fp_per_image),
        "max_defect_per_image": int(max_defect_per_image),
    }
    digest = hashlib.md5(json.dumps(cache_cfg, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    source_tag = f"{source_dataset}_{Path(source_root).name.replace('/', '_')}"
    heldout_tag = heldout_class or "none"
    return ROOT / "neurips2026" / "results" / "bank_cache" / f"rawimagebind_{source_tag}_{prompt_mode}_exclude-{heldout_tag}_{digest}.pt"


def load_bank_cache(cache_path: Path):
    payload = torch.load(cache_path, map_location="cpu")
    fp_banks = {int(k): v.float() for k, v in payload["fp_banks"].items()}
    def_banks = {int(k): v.float() for k, v in payload["def_banks"].items()}
    bank_stats = {int(k): v for k, v in payload["bank_stats"].items()}
    cache_meta = payload.get("cache_meta", {})
    return fp_banks, def_banks, bank_stats, cache_meta


def save_bank_cache(cache_path: Path, fp_banks, def_banks, bank_stats, cache_meta):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fp_banks": {int(k): v.detach().cpu() for k, v in fp_banks.items()},
            "def_banks": {int(k): v.detach().cpu() for k, v in def_banks.items()},
            "bank_stats": bank_stats,
            "cache_meta": cache_meta,
        },
        cache_path,
    )


def get_or_collect_source_patch_banks(
    backbone,
    source_root: str,
    source_dataset: str,
    image_transform,
    mask_transform,
    image_size: int,
    features_list: list[int],
    prompt_mode: str,
    heldout_class: str | None,
    max_good_per_class: int,
    max_defect_per_class: int,
    hard_frac: float,
    max_bank_per_layer: int,
    max_fp_per_image: int,
    max_defect_per_image: int,
    bank_cache_path: str | None = None,
    refresh_bank_cache: bool = False,
):
    requested_cache_meta = {
        "source_root": str(source_root),
        "source_dataset": source_dataset,
        "heldout_class": heldout_class,
        "image_size": int(image_size),
        "features_list": list(features_list),
        "prompt_mode": prompt_mode,
        "hard_frac": float(hard_frac),
        "max_good_per_class": int(max_good_per_class),
        "max_defect_per_class": int(max_defect_per_class),
        "max_bank_per_layer": int(max_bank_per_layer),
        "max_fp_per_image": int(max_fp_per_image),
        "max_defect_per_image": int(max_defect_per_image),
    }
    cache_path = Path(bank_cache_path) if bank_cache_path else default_bank_cache_path(
        source_root=source_root,
        source_dataset=source_dataset,
        heldout_class=heldout_class,
        image_size=image_size,
        features_list=features_list,
        prompt_mode=prompt_mode,
        hard_frac=hard_frac,
        max_good_per_class=max_good_per_class,
        max_defect_per_class=max_defect_per_class,
        max_bank_per_layer=max_bank_per_layer,
        max_fp_per_image=max_fp_per_image,
        max_defect_per_image=max_defect_per_image,
    )
    if cache_path.exists() and not refresh_bank_cache:
        fp_banks, def_banks, bank_stats, cache_meta = load_bank_cache(cache_path)
        if cache_meta == requested_cache_meta:
            return fp_banks, def_banks, bank_stats, cache_path, cache_meta

    fp_banks, def_banks, bank_stats = collect_source_patch_banks(
        backbone=backbone,
        source_root=source_root,
        image_transform=image_transform,
        mask_transform=mask_transform,
        features_list=features_list,
        prompt_mode=prompt_mode,
        heldout_class=heldout_class,
        max_good_per_class=max_good_per_class,
        max_defect_per_class=max_defect_per_class,
        hard_frac=hard_frac,
        max_bank_per_layer=max_bank_per_layer,
        max_fp_per_image=max_fp_per_image,
        max_defect_per_image=max_defect_per_image,
    )
    save_bank_cache(cache_path, fp_banks, def_banks, bank_stats, requested_cache_meta)
    return fp_banks, def_banks, bank_stats, cache_path, requested_cache_meta


def main():
    parser = argparse.ArgumentParser("Raw ImageBind baseline + decomposition evaluation")
    parser.add_argument("--source_root", type=str, required=True)
    parser.add_argument("--source_dataset", type=str, required=True)
    parser.add_argument("--target_root", type=str, required=True)
    parser.add_argument("--target_dataset", type=str, required=True)
    parser.add_argument("--target_mode", type=str, default="test")
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--bank_cache_path", type=str, default=None)
    parser.add_argument("--refresh_bank_cache", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--features_list", type=int, nargs="+", default=None)
    parser.add_argument("--prompt_mode", type=str, choices=["class", "object"], default="class")
    parser.add_argument("--source_exclude_class", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--hard_frac", type=float, default=0.01)
    parser.add_argument("--max_good_per_class", type=int, default=2)
    parser.add_argument("--max_defect_per_class", type=int, default=4)
    parser.add_argument("--max_bank_per_layer", type=int, default=512)
    parser.add_argument("--max_fp_per_image", type=int, default=64)
    parser.add_argument("--max_defect_per_image", type=int, default=64)
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--fp_weight", type=float, default=1.0)
    parser.add_argument("--map_activation", type=str, choices=["identity", "sigmoid"], default="identity")
    parser.add_argument("--map_temperature", type=float, default=1.0)
    parser.add_argument("--hybrid_source", type=str, choices=["none", "baseline"], default="none")
    parser.add_argument("--hybrid_weight", type=float, default=0.0)
    parser.add_argument("--image_score_topk", type=float, default=0.01)
    parser.add_argument("--image_fusion_weight", type=float, default=0.5)
    parser.add_argument("--enable_calibrator", action="store_true")
    parser.add_argument("--calib_epochs", type=int, default=60)
    parser.add_argument("--calib_lr", type=float, default=0.02)
    parser.add_argument("--calib_eta_init", type=float, default=0.02)
    parser.add_argument("--calib_eta_max", type=float, default=0.2)
    parser.add_argument("--calib_eta_reg", type=float, default=1e-3)
    parser.add_argument("--calib_rank_margin", type=float, default=0.05)
    parser.add_argument("--calib_rank_temp", type=float, default=0.05)
    parser.add_argument("--calib_hard_frac", type=float, default=0.01)
    parser.add_argument("--calib_max_good_per_class", type=int, default=8)
    parser.add_argument("--calib_max_defect_per_class", type=int, default=8)
    parser.add_argument("--calib_max_fp_per_image", type=int, default=64)
    parser.add_argument("--calib_max_defect_per_image", type=int, default=128)
    parser.add_argument("--calib_max_train_points", type=int, default=8192)
    parser.add_argument("--calib_residual_scale", type=float, default=1.0)
    parser.add_argument("--calib_anchor", type=str, choices=["baseline", "trainfree"], default="baseline")
    parser.add_argument("--max_normal_eval_images", type=int, default=None)
    parser.add_argument("--max_anomaly_eval_images", type=int, default=None)
    parser.add_argument("--aupro_max_step", type=int, default=200)
    args = parser.parse_args()

    backbone = create_imagebind_raw_backbone(device=args.device, image_size=args.image_size)
    features_list = args.features_list or backbone.layer_ids
    image_transform, mask_transform = build_transforms(args.image_size)

    fp_banks, def_banks, bank_stats, bank_cache_path, bank_cache_meta = get_or_collect_source_patch_banks(
        backbone=backbone,
        source_root=args.source_root,
        source_dataset=args.source_dataset,
        image_transform=image_transform,
        mask_transform=mask_transform,
        image_size=args.image_size,
        features_list=features_list,
        prompt_mode=args.prompt_mode,
        heldout_class=args.source_exclude_class,
        max_good_per_class=args.max_good_per_class,
        max_defect_per_class=args.max_defect_per_class,
        hard_frac=args.hard_frac,
        max_bank_per_layer=args.max_bank_per_layer,
        max_fp_per_image=args.max_fp_per_image,
        max_defect_per_image=args.max_defect_per_image,
        bank_cache_path=args.bank_cache_path,
        refresh_bank_cache=args.refresh_bank_cache,
    )
    source_calibrators = {}
    if args.enable_calibrator:
        source_calibrators = train_source_calibrators(
            args,
            backbone,
            image_transform,
            mask_transform,
            features_list,
            fp_banks,
            def_banks,
        )

    target_ds = build_dataset(args.target_root, image_transform, mask_transform, args.target_mode, class_name=args.class_name)
    loader = torch.utils.data.DataLoader(target_ds, batch_size=1, shuffle=False, num_workers=0)

    results = {}
    modes = ["baseline_txt", "parallel_margin"]
    if args.enable_calibrator:
        modes.append("ted_calibrated")
    for mode in modes:
        results[mode] = {}
        for obj in target_ds.obj_list:
            results[mode][obj] = {"gt_sp": [], "pr_sp": [], "imgs_masks": [], "anomaly_maps": []}

    with torch.no_grad():
        num_normal_done = 0
        num_anomaly_done = 0
        for idx, items in enumerate(loader, start=1):
            image = items["img"].to(backbone.device)
            cls_name = items["cls_name"][0]
            anomaly_flag = int(items["anomaly"].item()) if hasattr(items["anomaly"], "item") else int(items["anomaly"])
            if anomaly_flag == 0 and args.max_normal_eval_images is not None and num_normal_done >= args.max_normal_eval_images:
                continue
            if anomaly_flag == 1 and args.max_anomaly_eval_images is not None and num_anomaly_done >= args.max_anomaly_eval_images:
                continue
            if anomaly_flag == 0:
                num_normal_done += 1
            else:
                num_anomaly_done += 1

            gt_mask = items["img_mask"].clone()
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

            results["baseline_txt"][cls_name]["gt_sp"].append(anomaly_flag)
            results["parallel_margin"][cls_name]["gt_sp"].append(anomaly_flag)
            results["baseline_txt"][cls_name]["imgs_masks"].append(gt_mask)
            results["parallel_margin"][cls_name]["imgs_masks"].append(gt_mask)
            if args.enable_calibrator:
                results["ted_calibrated"][cls_name]["gt_sp"].append(anomaly_flag)
                results["ted_calibrated"][cls_name]["imgs_masks"].append(gt_mask)

            text_pair = encode_prompt_pair(backbone, cls_name, args.prompt_mode)
            image_feature = backbone.encode_global_image(image)
            img_logits = image_feature @ torch.stack([text_pair["norm_joint"], text_pair["anom_joint"]], dim=1)
            global_anom_prob = (img_logits / 0.07).softmax(dim=-1)[0, 1].item()

            patch_dict = backbone.extract_layer_patch_tokens(image, target_layers=features_list, project_to_joint=True)
            baseline_layer_maps = []
            parallel_layer_maps = []
            calibrated_residual_maps = []
            for out_idx, layer in enumerate(features_list):
                probs, margin = compute_layer_host_and_ted(
                    patch_dict[layer][0],
                    text_pair,
                    fp_banks[out_idx],
                    def_banks[out_idx],
                    args.tau,
                    args.fp_weight,
                )
                side = int(round(math.sqrt(probs.shape[0])))
                baseline_map = probs.reshape(1, 1, side, side)
                baseline_map = F.interpolate(
                    baseline_map, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False
                )[:, 0]
                baseline_layer_maps.append(baseline_map)

                parallel_map = margin.reshape(1, 1, side, side)
                parallel_map = F.interpolate(
                    parallel_map, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False
                )[:, 0]
                parallel_layer_maps.append(parallel_map)
                if args.enable_calibrator:
                    eta = float(source_calibrators.get(out_idx, {}).get("eta", 0.0))
                    calibrated_residual_maps.append(eta * zscore_tensor(parallel_map))

            baseline_map = smooth_map(torch.stack(baseline_layer_maps).mean(dim=0), sigma=args.sigma)
            parallel_map = smooth_map(torch.stack(parallel_layer_maps).mean(dim=0), sigma=args.sigma)
            parallel_map = activate_margin_map(parallel_map, activation=args.map_activation, temperature=args.map_temperature)
            if args.hybrid_source != "none" and args.hybrid_weight > 0:
                parallel_map = (1.0 - args.hybrid_weight) * parallel_map + args.hybrid_weight * baseline_map

            baseline_img_score = args.image_fusion_weight * global_anom_prob + (1.0 - args.image_fusion_weight) * topk_mean(
                baseline_map[0].numpy(), args.image_score_topk
            )
            ours_img_score = args.image_fusion_weight * global_anom_prob + (1.0 - args.image_fusion_weight) * topk_mean(
                parallel_map[0].numpy(), args.image_score_topk
            )
            if args.enable_calibrator:
                residual = smooth_map(torch.stack(calibrated_residual_maps).mean(dim=0), sigma=args.sigma)
                calib_anchor_map = baseline_map if args.calib_anchor == "baseline" else parallel_map
                calibrated_map = calib_anchor_map + args.calib_residual_scale * calib_anchor_map.std().clamp_min(1e-6) * residual
                calibrated_img_score = args.image_fusion_weight * global_anom_prob + (1.0 - args.image_fusion_weight) * topk_mean(
                    calibrated_map[0].numpy(), args.image_score_topk
                )

            results["baseline_txt"][cls_name]["pr_sp"].append(baseline_img_score)
            results["parallel_margin"][cls_name]["pr_sp"].append(ours_img_score)
            results["baseline_txt"][cls_name]["anomaly_maps"].append(baseline_map)
            results["parallel_margin"][cls_name]["anomaly_maps"].append(parallel_map)
            if args.enable_calibrator:
                results["ted_calibrated"][cls_name]["pr_sp"].append(calibrated_img_score)
                results["ted_calibrated"][cls_name]["anomaly_maps"].append(calibrated_map)

            if idx % 8 == 0:
                print(json.dumps({"stage": "eval_progress", "images_done": idx, "current_class": cls_name}), flush=True)

    rows = []
    for mode in modes:
        image_aurocs, image_aps, pixel_aurocs, pixel_aupros, pixel_aps = [], [], [], [], []
        aupro_cache: dict[str, dict] = {}
        for obj in target_ds.obj_list:
            if not results[mode][obj]["gt_sp"]:
                continue
            results[mode][obj]["imgs_masks"] = torch.cat(results[mode][obj]["imgs_masks"])
            results[mode][obj]["anomaly_maps"] = torch.cat(results[mode][obj]["anomaly_maps"]).detach().cpu().numpy()
            if obj not in aupro_cache:
                aupro_cache[obj] = precompute_aupro_regions(results[mode][obj]["imgs_masks"])
            row = {
                "mode": mode,
                "target_label": obj,
                "image_auroc": float(image_level_metrics(results[mode], obj, "image-auroc")),
                "image_ap": float(image_level_metrics(results[mode], obj, "image-ap")),
                "pixel_auroc": float(pixel_level_metrics(results[mode], obj, "pixel-auroc")),
                "pixel_aupro": float(
                    calculate_fast_aupro(
                        results[mode][obj]["imgs_masks"],
                        results[mode][obj]["anomaly_maps"],
                        max_step=args.aupro_max_step,
                        precomputed=aupro_cache[obj],
                    )
                ),
                "pixel_ap": float(pixel_average_precision(results[mode], obj)),
            }
            rows.append(row)
            image_aurocs.append(row["image_auroc"])
            image_aps.append(row["image_ap"])
            pixel_aurocs.append(row["pixel_auroc"])
            pixel_aupros.append(row["pixel_aupro"])
            pixel_aps.append(row["pixel_ap"])
        rows.append(
            {
                "mode": mode,
                "target_label": "mean",
                "image_auroc": float(np.mean(image_aurocs)),
                "image_ap": float(np.mean(image_aps)),
                "pixel_auroc": float(np.mean(pixel_aurocs)),
                "pixel_aupro": float(np.mean(pixel_aupros)),
                "pixel_ap": float(np.mean(pixel_aps)),
            }
        )

    summary = {
        "backbone": "ImageBind-Huge",
        "pretrained_dataset": "imagebind_official",
        "prompt_mode": args.prompt_mode,
        "class_name": args.class_name,
        "features_list": features_list,
        "image_size": args.image_size,
        "sigma": args.sigma,
        "tau": args.tau,
        "fp_weight": args.fp_weight,
        "map_activation": args.map_activation,
        "map_temperature": args.map_temperature,
        "hybrid_source": args.hybrid_source,
        "hybrid_weight": args.hybrid_weight,
        "image_score_topk": args.image_score_topk,
        "image_fusion_weight": args.image_fusion_weight,
        "bank_cache_path": str(bank_cache_path),
        "bank_cache_meta": bank_cache_meta,
        "bank_stats": bank_stats,
        "enable_calibrator": args.enable_calibrator,
        "source_calibrators": source_calibrators,
        "calib_anchor": args.calib_anchor,
        "calib_residual_scale": args.calib_residual_scale,
        "aupro_max_step": args.aupro_max_step,
        "aupro_backend": "precomputed_region_fast_cpu",
    }
    save_results(rows, summary, Path(args.save_dir))
    print(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
