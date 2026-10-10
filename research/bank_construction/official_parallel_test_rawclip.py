from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import measure
from sklearn.metrics import auc, average_precision_score
from torchvision import transforms

ROOT = Path("/mnt/data/pilab-kingjinyoung/ted-reproduce-20261008/snapshot")
ANO_ROOT = ROOT / "neurips2026" / "AnomalyCLIP"
if str(ANO_ROOT) not in sys.path:
    sys.path.insert(0, str(ANO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import image_level_metrics, pixel_level_metrics  # noqa: E402
from neurips2026.common_backbones import create_winclip_raw_backbone  # noqa: E402
from neurips2026.WinClip.WinCLIP.ad_prompts import (  # noqa: E402
    state_level_abnormal_prompts,
    state_level_normal_prompts,
    template_level_prompts,
)


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


def precompute_aupro_regions(masks: np.ndarray | torch.Tensor) -> dict:
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    masks = (masks > 0.5).astype(np.uint8)
    inverse_masks = 1 - masks
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
        "inverse_total": int(inverse_masks.sum()),
        "regions_by_image": regions_by_image,
        "has_foreground": bool(masks.sum() > 0),
    }


def calculate_fast_aupro(
    masks: np.ndarray | torch.Tensor,
    maps: np.ndarray | torch.Tensor,
    max_step: int = 200,
    expect_fpr: float = 0.3,
    precomputed: dict | None = None,
) -> float:
    pre = precomputed if precomputed is not None else precompute_aupro_regions(masks)
    if not pre["has_foreground"] or pre["inverse_total"] <= 0:
        return float("nan")
    if isinstance(maps, torch.Tensor):
        maps = maps.detach().cpu().numpy()
    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim == 4:
        maps = maps.squeeze(1)
    min_th = float(np.min(maps))
    max_th = float(np.max(maps))
    if not np.isfinite(min_th) or not np.isfinite(max_th) or max_th <= min_th:
        return 0.5
    thresholds = np.linspace(min_th, max_th, num=max(2, int(max_step)), endpoint=False, dtype=np.float32)
    pros: list[float] = []
    fprs: list[float] = []
    inverse_masks = pre["inverse_masks"]
    inverse_total = int(pre["inverse_total"])
    for th in thresholds:
        binary_maps = maps > float(th)
        pro_vals = []
        for binary_map, regions in zip(binary_maps, pre["regions_by_image"]):
            for coords in regions:
                pro_vals.append(float(binary_map[coords[:, 0], coords[:, 1]].sum() / max(coords.shape[0], 1)))
        if not pro_vals:
            continue
        fp_pixels = np.logical_and(inverse_masks, binary_maps).sum()
        pros.append(float(np.mean(pro_vals)))
        fprs.append(float(fp_pixels / inverse_total))
    if not pros:
        return float("nan")
    pros_np = np.asarray(pros, dtype=np.float32)
    fprs_np = np.asarray(fprs, dtype=np.float32)
    keep = fprs_np < float(expect_fpr)
    if int(keep.sum()) <= 2:
        return 0.5
    fprs_keep = fprs_np[keep]
    denom = float(fprs_keep.max() - fprs_keep.min())
    if denom <= 1e-12:
        return 0.5
    fprs_keep = (fprs_keep - fprs_keep.min()) / denom
    return float(auc(fprs_keep, pros_np[keep]))


def logmeanexp_negative_sqdist_1d(coeff: torch.Tensor, bank_coeff: torch.Tensor, tau: float) -> torch.Tensor:
    if bank_coeff.numel() == 0:
        return coeff.new_zeros((coeff.shape[0],))
    dist2 = (coeff.unsqueeze(1) - bank_coeff.unsqueeze(0)) ** 2
    logits = -dist2 / tau
    return torch.logsumexp(logits, dim=-1) - math.log(bank_coeff.shape[0])


def smooth_map(batch_map: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.stack(
        [torch.from_numpy(gaussian_filter(arr, sigma=sigma)) for arr in batch_map.detach().cpu().numpy()],
        dim=0,
    )


def activate_margin_map(batch_map: torch.Tensor, activation: str, temperature: float) -> torch.Tensor:
    if activation == "identity":
        return batch_map
    if activation == "sigmoid":
        return torch.sigmoid(batch_map / max(temperature, 1e-6))
    raise ValueError(f"Unsupported map activation: {activation}")


def topk_mean(vals: np.ndarray, ratio: float) -> float:
    vals = np.asarray(vals).reshape(-1)
    if vals.size == 0:
        return float("nan")
    k = max(1, int(round(vals.size * ratio)))
    idx = np.argpartition(vals, -k)[-k:]
    return float(vals[idx].mean())


def zscore_tensor(vals: torch.Tensor) -> torch.Tensor:
    return (vals - vals.mean()) / vals.std().clamp_min(1e-6)


def pixel_average_precision(results: dict, obj: str) -> float:
    gt = np.array(results[obj]["imgs_masks"])
    pr = np.array(results[obj]["anomaly_maps"])
    return float(average_precision_score(gt.ravel(), pr.ravel()))


def save_results(rows: list[dict], summary: dict, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "summary.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    if rows:
        with open(save_dir / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


class GenericMetaDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, image_transform, mask_transform, split: str, class_name: str | None = None):
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
            self.items.extend(split_meta[cls_name])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        img_path = self.root / item["img_path"]
        img = Image.open(img_path).convert("RGB")
        anomaly = int(item["anomaly"])
        mask_path = item.get("mask_path")
        if anomaly == 0 or not mask_path:
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
        else:
            raw_mask = np.array(Image.open(self.root / mask_path).convert("L")) > 0
            mask = Image.fromarray(raw_mask.astype(np.uint8) * 255, mode="L")

        return {
            "img": self.image_transform(img),
            "img_mask": self.mask_transform(mask),
            "cls_name": item["cls_name"],
            "anomaly": anomaly,
            "img_path": str(img_path),
        }


def build_prompt_groups(prompt_mode: str, cls_name: str) -> tuple[list[str], list[str]]:
    subject = "object" if prompt_mode == "object" else cls_name.replace("_", " ")
    normal_prompts = [
        template.format(state_prompt.format(subject))
        for template in template_level_prompts
        for state_prompt in state_level_normal_prompts
    ]
    abnormal_prompts = [
        template.format(state_prompt.format(subject))
        for template in template_level_prompts
        for state_prompt in state_level_abnormal_prompts
    ]
    return normal_prompts, abnormal_prompts


def encode_prompt_pair(backbone, cls_name: str, prompt_mode: str) -> dict[str, torch.Tensor]:
    normal_prompts, abnormal_prompts = build_prompt_groups(prompt_mode, cls_name)
    text_normal = backbone.tokenizer(normal_prompts).to(backbone.device)
    text_abnormal = backbone.tokenizer(abnormal_prompts).to(backbone.device)

    with torch.no_grad():
        e_norm_img = F.normalize(backbone.encode_text(text_normal).float(), dim=-1).mean(dim=0)
        e_anom_img = F.normalize(backbone.encode_text(text_abnormal).float(), dim=-1).mean(dim=0)
    e_norm_img = F.normalize(e_norm_img, dim=0)
    e_anom_img = F.normalize(e_anom_img, dim=0)

    proj = backbone.visual.proj.detach().float().to(backbone.device)
    e_norm_tok = F.normalize(proj @ e_norm_img, dim=0)
    e_anom_tok = F.normalize(proj @ e_anom_img, dim=0)
    axis_tok = F.normalize(e_anom_tok - e_norm_tok, dim=0)

    return {
        "norm_img": e_norm_img,
        "anom_img": e_anom_img,
        "norm_tok": e_norm_tok,
        "anom_tok": e_anom_tok,
        "axis_tok": axis_tok,
    }


def encode_global_image_feature(backbone, image: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        sequence = backbone.prepare_sequence(image)
        x = backbone.run_blocks(sequence)
        x = x.permute(1, 0, 2)
        cls = backbone.visual.ln_post(x[:, 0, :]).float()
        proj = getattr(backbone.visual, "proj", None)
        if proj is not None:
            cls = cls @ proj.float()
    return F.normalize(cls, dim=-1)


def compute_layer_host_and_ted(
    patch_tokens: torch.Tensor,
    text_pair: dict[str, torch.Tensor],
    fp_bank: torch.Tensor,
    def_bank: torch.Tensor,
    tau: float,
    fp_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    patch = F.normalize(patch_tokens.float(), dim=-1)
    logits = patch @ torch.stack([text_pair["norm_tok"], text_pair["anom_tok"]], dim=1)
    host_score = (logits / 0.07).softmax(dim=-1)[:, 1]
    axis = text_pair["axis_tok"]
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


def _collect_rawclip_calibration_pairs(
    args,
    backbone,
    image_transform,
    mask_transform,
    features_list: list[int],
    image_size: int,
    fp_banks: dict[int, torch.Tensor],
    def_banks: dict[int, torch.Tensor],
) -> dict[int, dict[str, torch.Tensor]]:
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
            patch_dict = backbone.extract_layer_patch_tokens(image, target_layers=features_list)
            for out_idx, layer in enumerate(features_list):
                host_score, margin = compute_layer_host_and_ted(
                    patch_dict[layer][0],
                    text_pair,
                    fp_banks[out_idx],
                    def_banks[out_idx],
                    tau=args.tau,
                    fp_weight=args.fp_weight,
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
            patch_dict = backbone.extract_layer_patch_tokens(image, target_layers=features_list)
            for out_idx, layer in enumerate(features_list):
                host_score, margin = compute_layer_host_and_ted(
                    patch_dict[layer][0],
                    text_pair,
                    fp_banks[out_idx],
                    def_banks[out_idx],
                    tau=args.tau,
                    fp_weight=args.fp_weight,
                )
                h = backbone.grid_size[0] if host_score.numel() == backbone.grid_size[0] * backbone.grid_size[1] else int(round(host_score.numel() ** 0.5))
                w = host_score.numel() // h
                gt_small = F.interpolate(gt_mask, size=(h, w), mode="nearest")[0, 0].reshape(-1) > 0.5
                idx_def = _sample_indices(gt_small, args.calib_max_defect_per_image)
                if idx_def.numel() == 0:
                    continue
                data[out_idx]["def_host"].append(host_score[idx_def].detach().cpu())
                data[out_idx]["def_res"].append(zscore_tensor(margin)[idx_def].detach().cpu())

    packed: dict[int, dict[str, torch.Tensor]] = {}
    for out_idx, chunks in data.items():
        packed[out_idx] = {}
        for key, vals in chunks.items():
            tensor = torch.cat(vals, dim=0).float() if vals else torch.empty((0,), dtype=torch.float32)
            if tensor.numel() > args.calib_max_train_points:
                perm = torch.randperm(tensor.numel())[: args.calib_max_train_points]
                tensor = tensor[perm]
            packed[out_idx][key] = tensor
    return packed


def train_rawclip_source_calibrators(
    args,
    backbone,
    image_transform,
    mask_transform,
    features_list: list[int],
    image_size: int,
    fp_banks: dict[int, torch.Tensor],
    def_banks: dict[int, torch.Tensor],
) -> dict[int, dict]:
    pairs = _collect_rawclip_calibration_pairs(
        args=args,
        backbone=backbone,
        image_transform=image_transform,
        mask_transform=mask_transform,
        features_list=features_list,
        image_size=image_size,
        fp_banks=fp_banks,
        def_banks=def_banks,
    )
    calibrators: dict[int, dict] = {}
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
    print(json.dumps({"stage": "rawclip_source_calibrators_trained", "calibrators": calibrators}, indent=2), flush=True)
    return calibrators


def build_transforms(image_size: int):
    image_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
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


def build_dataset(root: str, image_transform, mask_transform, split: str, class_name: str | None = None):
    return GenericMetaDataset(root=root, image_transform=image_transform, mask_transform=mask_transform, split=split, class_name=class_name)


def collect_source_patch_banks(
    backbone,
    source_root: str,
    image_transform,
    mask_transform,
    features_list: list[int],
    image_size: int,
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
            patch_dict = backbone.extract_layer_patch_tokens(image, target_layers=features_list)

            for out_idx, layer in enumerate(features_list):
                patch = F.normalize(patch_dict[layer][0].float(), dim=-1)
                logits = patch @ torch.stack([text_pair["norm_tok"], text_pair["anom_tok"]], dim=1)
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
            patch_dict = backbone.extract_layer_patch_tokens(image, target_layers=features_list)
            for out_idx, layer in enumerate(features_list):
                patch = F.normalize(patch_dict[layer][0].float(), dim=-1)
                h = backbone.grid_size[0] if patch.shape[0] == backbone.grid_size[0] * backbone.grid_size[1] else int(round(patch.shape[0] ** 0.5))
                w = patch.shape[0] // h
                gt_small = F.interpolate(gt_mask, size=(h, w), mode="nearest")[0, 0].reshape(-1) > 0.5
                if int(gt_small.sum().item()) == 0:
                    continue
                selected = patch[gt_small]
                if selected.shape[0] > max_defect_per_image:
                    selected = farthest_point_subsample(selected, max_defect_per_image)
                def_chunks[out_idx].append(selected.cpu())

    fp_banks: dict[int, torch.Tensor] = {}
    def_banks: dict[int, torch.Tensor] = {}
    bank_stats: dict[int, dict] = {}
    token_dim = backbone.visual.class_embedding.shape[0]
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
    backbone_name: str,
    pretrained_dataset: str,
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
        "backbone_name": backbone_name,
        "pretrained_dataset": pretrained_dataset,
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
    backbone_tag = backbone_name.replace("/", "-")
    return ROOT / "neurips2026" / "results" / "bank_cache" / f"rawclip_{backbone_tag}_{pretrained_dataset}_{source_tag}_{prompt_mode}_exclude-{heldout_tag}_{digest}.pt"


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
    backbone_name: str,
    pretrained_dataset: str,
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
        "backbone_name": backbone_name,
        "pretrained_dataset": pretrained_dataset,
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
        backbone_name=backbone_name,
        pretrained_dataset=pretrained_dataset,
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
        print(json.dumps({"stage": "bank_cache_mismatch", "cache_path": str(cache_path), "cached_meta": cache_meta, "requested_meta": requested_cache_meta}), flush=True)

    fp_banks, def_banks, bank_stats = collect_source_patch_banks(
        backbone=backbone,
        source_root=source_root,
        image_transform=image_transform,
        mask_transform=mask_transform,
        features_list=features_list,
        image_size=image_size,
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


def resolve_features_list(num_layers: int, requested: list[int] | None) -> list[int]:
    if requested:
        return requested
    if num_layers == 12:
        return [3, 6, 9, 12]
    if num_layers == 24:
        return [6, 12, 18, 24]
    step = max(1, num_layers // 4)
    return sorted({step, min(num_layers, step * 2), min(num_layers, step * 3), num_layers})


def main():
    parser = argparse.ArgumentParser("Raw CLIP baseline + multilayer parallel_margin evaluation")
    parser.add_argument("--source_root", type=str, required=True)
    parser.add_argument("--source_dataset", type=str, required=True)
    parser.add_argument("--target_root", type=str, required=True)
    parser.add_argument("--target_dataset", type=str, required=True)
    parser.add_argument("--target_mode", type=str, default="test")
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--bank_cache_path", type=str, default=None)
    parser.add_argument("--refresh_bank_cache", action="store_true")
    parser.add_argument("--backbone", type=str, default="ViT-L-14-336")
    parser.add_argument("--pretrained_dataset", type=str, default="openai")
    parser.add_argument("--image_size", type=int, default=None)
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
    if args.class_name is not None and args.class_name.lower() in {"", "none", "null", "all"}:
        args.class_name = None

    default_image_size = 336 if "336" in args.backbone else 240 if "240" in args.backbone else 224
    image_size = args.image_size or default_image_size

    backbone = create_winclip_raw_backbone(
        backbone=args.backbone,
        pretrained_dataset=args.pretrained_dataset,
        device=args.device,
        img_resize=image_size,
        img_cropsize=image_size,
    )
    features_list = resolve_features_list(backbone.num_layers, args.features_list)
    image_transform, mask_transform = build_transforms(image_size)

    fp_banks, def_banks, bank_stats, bank_cache_path, bank_cache_meta = get_or_collect_source_patch_banks(
        backbone=backbone,
        backbone_name=args.backbone,
        pretrained_dataset=args.pretrained_dataset,
        source_root=args.source_root,
        source_dataset=args.source_dataset,
        image_transform=image_transform,
        mask_transform=mask_transform,
        image_size=image_size,
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
        source_calibrators = train_rawclip_source_calibrators(
            args=args,
            backbone=backbone,
            image_transform=image_transform,
            mask_transform=mask_transform,
            features_list=features_list,
            image_size=image_size,
            fp_banks=fp_banks,
            def_banks=def_banks,
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
            image_feature = encode_global_image_feature(backbone, image)
            img_logits = image_feature @ torch.stack([text_pair["norm_img"], text_pair["anom_img"]], dim=1)
            global_anom_prob = (img_logits / 0.07).softmax(dim=-1)[0, 1].item()

            patch_dict = backbone.extract_layer_patch_tokens(image, target_layers=features_list)
            baseline_layer_maps = []
            parallel_layer_maps = []
            calibrated_residual_maps = []
            for out_idx, layer in enumerate(features_list):
                probs, margin = compute_layer_host_and_ted(
                    patch_dict[layer][0],
                    text_pair,
                    fp_banks[out_idx],
                    def_banks[out_idx],
                    tau=args.tau,
                    fp_weight=args.fp_weight,
                )
                h, w = backbone.grid_size
                if probs.shape[0] != h * w:
                    h = int(round(probs.shape[0] ** 0.5))
                    w = probs.shape[0] // h
                baseline_map = probs.reshape(1, 1, h, w)
                baseline_map = F.interpolate(baseline_map, size=(image_size, image_size), mode="bilinear", align_corners=False)[:, 0]
                baseline_layer_maps.append(baseline_map)

                parallel_map = margin.reshape(1, 1, h, w)
                parallel_map = F.interpolate(parallel_map, size=(image_size, image_size), mode="bilinear", align_corners=False)[:, 0]
                parallel_layer_maps.append(parallel_map)
                if args.enable_calibrator:
                    eta = float(source_calibrators.get(out_idx, {}).get("eta", 0.0))
                    calibrated_residual_maps.append(eta * zscore_tensor(parallel_map))

            baseline_map = smooth_map(torch.stack(baseline_layer_maps).mean(dim=0), sigma=args.sigma)
            parallel_map = smooth_map(torch.stack(parallel_layer_maps).mean(dim=0), sigma=args.sigma)
            parallel_map = activate_margin_map(parallel_map, activation=args.map_activation, temperature=args.map_temperature)
            if args.hybrid_source != "none" and args.hybrid_weight > 0:
                if args.hybrid_source == "baseline":
                    parallel_map = (1.0 - args.hybrid_weight) * parallel_map + args.hybrid_weight * baseline_map
                else:
                    raise ValueError(f"Unsupported hybrid_source: {args.hybrid_source}")
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
        "backbone": args.backbone,
        "pretrained_dataset": args.pretrained_dataset,
        "prompt_mode": args.prompt_mode,
        "class_name": args.class_name,
        "features_list": features_list,
        "image_size": image_size,
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
