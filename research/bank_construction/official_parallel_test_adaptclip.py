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
from scipy.ndimage import gaussian_filter
from skimage import measure

ROOT = Path("/mnt/data/pilab-kingjinyoung/ted-reproduce-20261008/snapshot")
SCRIPT_ROOT = ROOT / "neurips2026" / "scripts"
ADAPT_ROOT = ROOT / "neurips2026" / "AdaptCLIP"
if str(ADAPT_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(1, str(SCRIPT_ROOT))

import adaptcliplib  # noqa: E402
from adaptcliplib import TextualAdapter, VisualAdapter, fusion_fun  # noqa: E402
from dataset import Dataset  # noqa: E402
from tools import Evaluator, get_transform  # noqa: E402


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


def decision_axis(text_pair: torch.Tensor) -> torch.Tensor:
    axis = text_pair[1] - text_pair[0]
    return F.normalize(axis.float(), dim=-1)


def decompose_with_axis(feat: torch.Tensor, axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    coeff = feat @ axis
    parallel = coeff.unsqueeze(-1) * axis.unsqueeze(0)
    perp = feat - parallel
    return parallel, perp


def logmeanexp_negative_sqdist_1d(coeff: torch.Tensor, bank_coeff: torch.Tensor, tau: float) -> torch.Tensor:
    if bank_coeff.numel() == 0:
        return coeff.new_zeros((coeff.shape[0],))
    dist2 = (coeff.unsqueeze(1) - bank_coeff.unsqueeze(0)) ** 2
    logits = -dist2 / tau
    return torch.logsumexp(logits, dim=-1) - math.log(bank_coeff.shape[0])


def logmeanexp_cosine_support(query: torch.Tensor, bank: torch.Tensor, tau: float, chunk: int) -> torch.Tensor:
    if bank.numel() == 0:
        return torch.full((query.shape[0],), -1e6, device=query.device, dtype=query.dtype)
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


def project_bank(bank: torch.Tensor, axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    coeff = bank @ axis
    _, perp = decompose_with_axis(bank, axis)
    return coeff, perp


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


def resize_map(batch_map: torch.Tensor, image_size: int) -> torch.Tensor:
    return F.interpolate(
        batch_map[:, None],
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def topk_mean(arr: np.ndarray, frac: float) -> float:
    flat = arr.reshape(-1)
    k = max(1, int(math.ceil(flat.size * frac)))
    idx = np.argpartition(flat, -k)[-k:]
    return float(flat[idx].mean())


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


def save_results(rows: list[dict], summary: dict, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "summary.json", "w") as f:
        json.dump(jsonable({"summary": summary, "rows": rows}), f, indent=2)
    if rows:
        with open(save_dir / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def get_model_spec(pretrained_model: str) -> tuple[int, int, int]:
    model_name = Path(pretrained_model).name
    if pretrained_model == "ViT-L/14@336px":
        return 20, 768, 14
    if pretrained_model == "ViT-L-14-CLIPA-336":
        return 20, 768, 14
    if model_name in {"ViT-L-14.pt", "vit_l_14-laion400m_e32-3d133497.pt"}:
        return 20, 768, 14
    if pretrained_model == "VITB16_PLUS_240" or model_name in {
        "vit_b_16_plus_240-laion400m_e31-8fb26589.pt",
        "vit_b_16_plus_240-laion400m_e32-699c4b84.pt",
    }:
        return 10, 640, 16
    if pretrained_model in {"ViT-B/16", "ViT-B-16"} or model_name in {"ViT-B-16.pt", "vit_b_16-laion400m_e32-55e67d44.pt"}:
        return 10, 512, 16
    if pretrained_model in {"ViT-H/14", "ViT-H-14"} or model_name in {"ViT-H-14.pt", "open_clip_pytorch_model.bin", "open_clip_model.safetensors"}:
        return 30, 1024, 14
    raise ValueError(f"Unsupported pretrained model: {pretrained_model}")


def load_module_state(module: torch.nn.Module, source_state: dict, mode: str) -> dict:
    if mode == "strict":
        module.load_state_dict(source_state, strict=True)
        return {"mode": "strict", "loaded_keys": len(source_state), "skipped_keys": 0, "missing_keys": 0, "unexpected_keys": 0}
    if mode != "compatible":
        raise ValueError(f"Unsupported checkpoint_load_mode: {mode}")
    target_state = module.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in source_state.items():
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape):
            compatible_state[key] = value
        else:
            skipped.append(key)
    incompatible = module.load_state_dict(compatible_state, strict=False)
    return {
        "mode": "compatible",
        "loaded_keys": len(compatible_state),
        "skipped_keys": len(skipped),
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
    }


def build_dataset(root: str, image_size: int, dataset_name: str, mode: str, save_dir: str, class_name: str | None = None):
    preprocess, target_transform = get_transform(image_size=image_size)
    return Dataset(
        root=root,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=dataset_name,
        k_shots=0,
        save_dir=save_dir,
        mode=mode,
        seed=10,
        class_name=class_name,
    )


def build_model(
    device: str,
    checkpoint_path: str,
    image_size: int,
    n_ctx: int,
    vl_reduction: int,
    pretrained_model: str,
    checkpoint_load_mode: str = "strict",
):
    dpam_layer, input_dim, patch_size = get_model_spec(pretrained_model)
    model, _ = adaptcliplib.load(pretrained_model, device=device)
    model.visual.DAPM_replace(DPAM_layer=dpam_layer)

    textual_learner = TextualAdapter(model.to("cpu"), image_size, n_ctx)
    visual_learner = VisualAdapter(image_size, patch_size, input_dim=input_dim, reduction=vl_reduction)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_info = {
        "checkpoint_load_mode": checkpoint_load_mode,
        "textual_learner": load_module_state(textual_learner, checkpoint["textual_learner"], checkpoint_load_mode),
        "visual_learner": load_module_state(visual_learner, checkpoint["visual_learner"], checkpoint_load_mode),
    }

    model.to(device).eval()
    textual_learner.to(device).eval()
    visual_learner.to(device).eval()

    textual_learner.prepare_static_text_feature(model)
    learned_prompts, tokenized_prompts = textual_learner()
    learned_text_features = model.encode_text_learn(learned_prompts, tokenized_prompts).float()
    learned_text_features = F.normalize(learned_text_features, dim=-1)

    textual_learner._ted_checkpoint_load_info = load_info["textual_learner"]
    visual_learner._ted_checkpoint_load_info = load_info["visual_learner"]
    build_model.last_checkpoint_load_info = load_info
    return model, textual_learner, visual_learner, learned_text_features, dpam_layer


def compute_baseline_outputs(
    model,
    textual_learner,
    visual_learner,
    learned_text_features: torch.Tensor,
    image: torch.Tensor,
    features_list: list[int],
    dpam_layer: int,
    image_size: int,
    sigma: float,
    fusion_type: str,
):
    query_feats, query_patch_feats = model.encode_image(image, features_list, DPAM_layer=dpam_layer)

    global_vl_logit, local_vl_map = visual_learner(query_feats, query_patch_feats, textual_learner.static_text_features)
    global_tl_logit, local_tl_map = textual_learner.compute_global_local_score(query_feats, query_patch_feats, learned_text_features)

    global_vl_score = global_vl_logit.softmax(-1)[:, 1].detach()
    global_tl_score = global_tl_logit.softmax(-1)[:, 1].detach()
    local_vl_map = local_vl_map[:, 1].detach()
    local_tl_map = local_tl_map[:, 1].detach()

    baseline_map = fusion_fun([local_vl_map, local_tl_map], fusion_type=fusion_type)
    baseline_map = smooth_map(baseline_map, sigma=sigma).to(image.device)
    anomaly_map_max, _ = torch.max(baseline_map.view(image.shape[0], -1), dim=1)
    baseline_img_score = fusion_fun([global_vl_score, global_tl_score, anomaly_map_max], fusion_type=fusion_type)

    baseline_map = resize_map(baseline_map, image_size)
    return {
        "query_feats": query_feats,
        "query_patch_feats": query_patch_feats,
        "global_vl_score": global_vl_score,
        "global_tl_score": global_tl_score,
        "local_vl_map": local_vl_map,
        "local_tl_map": local_tl_map,
        "baseline_map": baseline_map,
        "baseline_img_score": baseline_img_score,
    }


def default_bank_cache_path(
    checkpoint_path: str,
    source_root: str,
    source_dataset: str,
    source_exclude_class: str | None,
    image_size: int,
    features_list: list[int],
    hard_frac: float,
    max_good_per_class: int,
    max_defect_per_class: int,
    max_bank_per_layer: int,
    max_fp_per_image: int,
    max_defect_per_image: int,
) -> Path:
    meta = {
        "checkpoint_path": checkpoint_path,
        "source_root": source_root,
        "source_dataset": source_dataset,
        "source_exclude_class": source_exclude_class,
        "image_size": image_size,
        "features_list": list(features_list),
        "hard_frac": hard_frac,
        "max_good_per_class": max_good_per_class,
        "max_defect_per_class": max_defect_per_class,
        "max_bank_per_layer": max_bank_per_layer,
        "max_fp_per_image": max_fp_per_image,
        "max_defect_per_image": max_defect_per_image,
    }
    digest = hashlib.md5(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    source_tag = f"{source_dataset}_{Path(source_root).name.replace('/', '_')}"
    exclude_tag = source_exclude_class or "none"
    return ROOT / "neurips2026" / "results" / "bank_cache" / f"adaptclip_{source_tag}_exclude-{exclude_tag}_{digest}.pt"


def load_bank_cache(cache_path: Path):
    payload = torch.load(cache_path, map_location="cpu")
    fp_banks = {int(k): v.float() for k, v in payload["fp_banks"].items()}
    def_banks = {int(k): v.float() for k, v in payload["def_banks"].items()}
    bank_stats = {int(k): v for k, v in payload["bank_stats"].items()}
    cache_meta = payload.get("cache_meta", {})
    return fp_banks, def_banks, bank_stats, cache_meta


def save_bank_cache(cache_path: Path, fp_banks: dict[int, torch.Tensor], def_banks: dict[int, torch.Tensor], bank_stats: dict[int, dict], cache_meta: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fp_banks": {int(k): v.detach().cpu() for k, v in fp_banks.items()},
        "def_banks": {int(k): v.detach().cpu() for k, v in def_banks.items()},
        "bank_stats": bank_stats,
        "cache_meta": cache_meta,
    }
    torch.save(payload, cache_path)


def collect_source_patch_banks(
    model,
    textual_learner,
    visual_learner,
    learned_text_features: torch.Tensor,
    source_root: str,
    source_dataset: str,
    image_size: int,
    features_list: list[int],
    dpam_layer: int,
    sigma: float,
    fusion_type: str,
    device: str,
    source_exclude_class: str | None,
    bank_score_source: str,
    hard_frac: float,
    max_good_per_class: int,
    max_defect_per_class: int,
    max_bank_per_layer: int,
    max_fp_per_image: int,
    max_defect_per_image: int,
):
    train_ds = build_dataset(source_root, image_size, source_dataset, "train", str(ROOT / "neurips2026" / "results" / "_tmp"))
    test_ds = build_dataset(source_root, image_size, source_dataset, "test", str(ROOT / "neurips2026" / "results" / "_tmp"))

    fp_chunks = {i: [] for i in range(len(features_list))}
    def_chunks = {i: [] for i in range(len(features_list))}
    per_good = {cls_name: 0 for cls_name in train_ds.obj_list}
    per_defect = {cls_name: 0 for cls_name in test_ds.obj_list}

    with torch.no_grad():
        for idx in range(len(train_ds)):
            item = train_ds[idx]
            cls_name = item["cls_name"]
            if source_exclude_class is not None and cls_name == source_exclude_class:
                continue
            if per_good[cls_name] >= max_good_per_class:
                continue
            per_good[cls_name] += 1

            image = item["img"].unsqueeze(0).to(device)
            outputs = compute_baseline_outputs(
                model,
                textual_learner,
                visual_learner,
                learned_text_features,
                image,
                features_list,
                dpam_layer,
                image_size,
                sigma,
                fusion_type,
            )
            if bank_score_source == "baseline":
                mining_map = outputs["baseline_map"][0]
            elif bank_score_source == "tl":
                mining_map = smooth_map(outputs["local_tl_map"], sigma=sigma)[0].to(device)
            else:
                raise ValueError(f"Unsupported bank_score_source: {bank_score_source}")
            for layer_idx, patch_feature in enumerate(outputs["query_patch_feats"]):
                patch = F.normalize(patch_feature[:, 1:, :].reshape(-1, patch_feature.shape[-1]).float(), dim=-1)
                h = int(round(patch.shape[0] ** 0.5))
                score_small = F.interpolate(
                    mining_map[None, None],
                    size=(h, h),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].reshape(-1)
                k = max(1, int(score_small.numel() * hard_frac))
                top_idx = torch.topk(score_small, k=k, largest=True).indices
                selected = patch[top_idx]
                if selected.shape[0] > max_fp_per_image:
                    selected = farthest_point_subsample(selected, max_fp_per_image)
                fp_chunks[layer_idx].append(selected.cpu())

        for idx in range(len(test_ds)):
            item = test_ds[idx]
            cls_name = item["cls_name"]
            anomaly_flag = int(item["anomaly"])
            if anomaly_flag == 0:
                continue
            if source_exclude_class is not None and cls_name == source_exclude_class:
                continue
            if per_defect[cls_name] >= max_defect_per_class:
                continue
            per_defect[cls_name] += 1

            image = item["img"].unsqueeze(0).to(device)
            gt_mask = item["img_mask"].unsqueeze(0).to(device).float()
            query_feats, query_patch_feats = model.encode_image(image, features_list, DPAM_layer=dpam_layer)
            for layer_idx, patch_feature in enumerate(query_patch_feats):
                patch = F.normalize(patch_feature[:, 1:, :].reshape(-1, patch_feature.shape[-1]).float(), dim=-1)
                h = int(round(patch.shape[0] ** 0.5))
                gt_small = F.interpolate(gt_mask, size=(h, h), mode="nearest")[0, 0].reshape(-1) > 0.5
                if int(gt_small.sum().item()) == 0:
                    continue
                selected = patch[gt_small]
                if selected.shape[0] > max_defect_per_image:
                    selected = farthest_point_subsample(selected, max_defect_per_image)
                def_chunks[layer_idx].append(selected.cpu())

    fp_banks = {}
    def_banks = {}
    bank_stats = {}
    feat_dim = next(iter(outputs["query_patch_feats"])).shape[-1] if "outputs" in locals() else 768
    for layer_idx in range(len(features_list)):
        fp_bank = torch.cat(fp_chunks[layer_idx], dim=0) if fp_chunks[layer_idx] else torch.empty((0, feat_dim))
        def_bank = torch.cat(def_chunks[layer_idx], dim=0) if def_chunks[layer_idx] else torch.empty((0, feat_dim))
        if fp_bank.shape[0] > max_bank_per_layer:
            fp_bank = farthest_point_subsample(fp_bank, max_bank_per_layer)
        if def_bank.shape[0] > max_bank_per_layer:
            def_bank = farthest_point_subsample(def_bank, max_bank_per_layer)
        fp_banks[layer_idx] = F.normalize(fp_bank.float(), dim=-1) if fp_bank.numel() else fp_bank.float()
        def_banks[layer_idx] = F.normalize(def_bank.float(), dim=-1) if def_bank.numel() else def_bank.float()
        bank_stats[layer_idx] = {
            "num_fp": int(fp_banks[layer_idx].shape[0]),
            "num_defect": int(def_banks[layer_idx].shape[0]),
        }
    return fp_banks, def_banks, bank_stats


def get_or_collect_source_patch_banks(
    model,
    textual_learner,
    visual_learner,
    learned_text_features: torch.Tensor,
    source_root: str,
    source_dataset: str,
    image_size: int,
    features_list: list[int],
    dpam_layer: int,
    sigma: float,
    fusion_type: str,
    device: str,
    source_exclude_class: str | None,
    bank_score_source: str,
    hard_frac: float,
    max_good_per_class: int,
    max_defect_per_class: int,
    max_bank_per_layer: int,
    max_fp_per_image: int,
    max_defect_per_image: int,
    checkpoint_path: str,
    bank_cache_path: str | None,
    refresh_bank_cache: bool,
):
    requested_meta = {
        "checkpoint_path": checkpoint_path,
        "source_root": source_root,
        "source_dataset": source_dataset,
        "source_exclude_class": source_exclude_class,
        "image_size": image_size,
        "features_list": list(features_list),
        "sigma": sigma,
        "fusion_type": fusion_type,
        "bank_score_source": bank_score_source,
        "hard_frac": hard_frac,
        "max_good_per_class": max_good_per_class,
        "max_defect_per_class": max_defect_per_class,
        "max_bank_per_layer": max_bank_per_layer,
        "max_fp_per_image": max_fp_per_image,
        "max_defect_per_image": max_defect_per_image,
    }
    cache_path = Path(bank_cache_path) if bank_cache_path else default_bank_cache_path(
        checkpoint_path=checkpoint_path,
        source_root=source_root,
        source_dataset=source_dataset,
        source_exclude_class=source_exclude_class,
        image_size=image_size,
        features_list=features_list,
        hard_frac=hard_frac,
        max_good_per_class=max_good_per_class,
        max_defect_per_class=max_defect_per_class,
        max_bank_per_layer=max_bank_per_layer,
        max_fp_per_image=max_fp_per_image,
        max_defect_per_image=max_defect_per_image,
    )
    if cache_path.exists() and not refresh_bank_cache:
        fp_banks, def_banks, bank_stats, cache_meta = load_bank_cache(cache_path)
        if cache_meta == requested_meta:
            return fp_banks, def_banks, bank_stats, cache_path, cache_meta
        print(json.dumps({"stage": "bank_cache_mismatch", "cache_path": str(cache_path), "cached_meta": cache_meta, "requested_meta": requested_meta}), flush=True)

    fp_banks, def_banks, bank_stats = collect_source_patch_banks(
        model=model,
        textual_learner=textual_learner,
        visual_learner=visual_learner,
        learned_text_features=learned_text_features,
        source_root=source_root,
        source_dataset=source_dataset,
        image_size=image_size,
        features_list=features_list,
        dpam_layer=dpam_layer,
        sigma=sigma,
        fusion_type=fusion_type,
        device=device,
        source_exclude_class=source_exclude_class,
        bank_score_source=bank_score_source,
        hard_frac=hard_frac,
        max_good_per_class=max_good_per_class,
        max_defect_per_class=max_defect_per_class,
        max_bank_per_layer=max_bank_per_layer,
        max_fp_per_image=max_fp_per_image,
        max_defect_per_image=max_defect_per_image,
    )
    save_bank_cache(cache_path, fp_banks, def_banks, bank_stats, requested_meta)
    return fp_banks, def_banks, bank_stats, cache_path, requested_meta


def aggregate_results(results: dict):
    out = {}
    for key in ["sample_ids", "cls_names", "query_paths"]:
        out[key] = np.asarray(results[key], dtype=object)
    for key in ["gt_masks", "pr_masks", "gt_anomalys", "pr_anomalys"]:
        out[key] = torch.cat(results[key], dim=0)
    return out


def prepare_aupro_arrays(masks: torch.Tensor, maps: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    masks_np = masks.detach().cpu().numpy().astype(np.float32)
    maps_np = maps.detach().cpu().numpy().astype(np.float32)
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]
    if maps_np.ndim == 4:
        maps_np = maps_np[:, 0]
    if masks_np.shape[0] != maps_np.shape[0]:
        raise ValueError(f"AUPRO image count mismatch: masks={masks_np.shape}, maps={maps_np.shape}")
    if masks_np.shape[-2:] != maps_np.shape[-2:]:
        masks_t = torch.from_numpy(masks_np[:, None])
        masks_np = F.interpolate(masks_t, size=maps_np.shape[-2:], mode="nearest")[:, 0].numpy()
    return (masks_np > 0.5).astype(np.uint8), maps_np.astype(np.float32)


def _sample_np(arr: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    arr = arr.reshape(-1)
    if arr.size <= max_points:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.size, size=max_points, replace=False)
    return arr[idx]


def _auc_np(pos: np.ndarray, neg: np.ndarray, max_points: int = 100000) -> float:
    pos = _sample_np(pos.astype(np.float32, copy=False), max_points, 0)
    neg = _sample_np(neg.astype(np.float32, copy=False), max_points, 1)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    scores = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([np.ones(pos.size, dtype=np.int64), np.zeros(neg.size, dtype=np.int64)])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    pos_ranks = ranks[labels == 1]
    return float((pos_ranks.sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def _top_fraction_values(arr: np.ndarray, frac: float) -> np.ndarray:
    flat = arr.reshape(-1)
    if flat.size == 0:
        return flat.astype(np.float32, copy=False)
    k = max(1, int(flat.size * frac))
    idx = np.argpartition(flat, -k)[-k:]
    return flat[idx].astype(np.float32, copy=False)


def map_group_separability(
    masks: np.ndarray,
    labels: np.ndarray,
    maps: np.ndarray,
    hard_frac: float = 0.01,
    max_points: int = 100000,
) -> dict[str, float]:
    masks = masks.astype(bool)
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


def precompute_aupro_regions(masks: np.ndarray) -> dict:
    masks = (masks > 0.5).astype(np.uint8)
    inverse_masks = 1 - masks
    regions_by_image: list[list[np.ndarray]] = []
    for mask in masks:
        labeled = measure.label(mask)
        regions_by_image.append([region.coords for region in measure.regionprops(labeled)])
    return {
        "masks_shape": tuple(masks.shape),
        "inverse_masks": inverse_masks,
        "inverse_total": int(inverse_masks.sum()),
        "regions_by_image": regions_by_image,
        "has_foreground": bool(masks.sum() > 0),
    }


def calculate_fast_aupro(
    maps: np.ndarray,
    precomputed: dict,
    max_step: int = 200,
    expect_fpr: float = 0.3,
) -> float:
    if not precomputed["has_foreground"]:
        return float("nan")
    inverse_total = int(precomputed["inverse_total"])
    if inverse_total <= 0:
        return float("nan")
    maps = maps.astype(np.float32, copy=False)
    min_th = float(np.min(maps))
    max_th = float(np.max(maps))
    if not np.isfinite(min_th) or not np.isfinite(max_th) or max_th <= min_th:
        return 0.5
    thresholds = np.linspace(min_th, max_th, num=max(2, int(max_step)), endpoint=False, dtype=np.float32)
    pros: list[float] = []
    fprs: list[float] = []
    inverse_masks = precomputed["inverse_masks"]
    regions_by_image = precomputed["regions_by_image"]
    for th in thresholds:
        binary_maps = maps > float(th)
        pro_vals = []
        for binary_map, regions in zip(binary_maps, regions_by_image):
            for coords in regions:
                if coords.size == 0:
                    continue
                pro_vals.append(float(binary_map[coords[:, 0], coords[:, 1]].sum() / max(coords.shape[0], 1)))
        if not pro_vals:
            continue
        fp_pixels = np.logical_and(inverse_masks, binary_maps).sum()
        pros.append(float(np.mean(pro_vals)))
        fprs.append(float(fp_pixels / inverse_total))
    if len(pros) <= 2:
        return 0.5
    pros_np = np.asarray(pros, dtype=np.float32)
    fprs_np = np.asarray(fprs, dtype=np.float32)
    keep = fprs_np < float(expect_fpr)
    if int(keep.sum()) <= 2:
        return 0.5
    fprs_keep = fprs_np[keep]
    pros_keep = pros_np[keep]
    order = np.argsort(fprs_keep)
    fprs_keep = fprs_keep[order]
    pros_keep = pros_keep[order]
    denom = float(fprs_keep.max() - fprs_keep.min())
    if denom <= 1e-12:
        return 0.5
    fprs_keep = (fprs_keep - fprs_keep.min()) / denom
    return float(np.trapezoid(pros_keep, fprs_keep))


def main():
    parser = argparse.ArgumentParser("Official AdaptCLIP baseline + parallel_margin test runner")
    parser.add_argument("--source_root", type=str, required=True)
    parser.add_argument("--source_dataset", type=str, required=True)
    parser.add_argument("--target_root", type=str, required=True)
    parser.add_argument("--target_dataset", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--pretrained_model", type=str, default="ViT-L/14@336px")
    parser.add_argument(
        "--checkpoint_load_mode",
        type=str,
        default="strict",
        choices=["strict", "compatible"],
        help="Use compatible for backbone-swap stress; only shape-compatible adapter tensors are loaded.",
    )
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--vl_reduction", type=int, default=4)
    parser.add_argument("--fusion_type", type=str, default="average_mean")
    parser.add_argument("--source_exclude_class", type=str, default=None)
    parser.add_argument("--bank_score_source", type=str, choices=["baseline", "tl"], default="baseline")
    parser.add_argument("--hard_frac", type=float, default=0.01)
    parser.add_argument("--max_good_per_class", type=int, default=2)
    parser.add_argument("--max_defect_per_class", type=int, default=4)
    parser.add_argument("--max_bank_per_layer", type=int, default=512)
    parser.add_argument("--max_fp_per_image", type=int, default=64)
    parser.add_argument("--max_defect_per_image", type=int, default=64)
    parser.add_argument("--bank_cache_path", type=str, default=None)
    parser.add_argument("--refresh_bank_cache", action="store_true")
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--fp_weight", type=float, default=1.0)
    parser.add_argument("--include_fullfeat_control", action="store_true")
    parser.add_argument("--fullfeat_tau", type=float, default=0.1)
    parser.add_argument("--fullfeat_chunk", type=int, default=512)
    parser.add_argument("--map_activation", type=str, choices=["identity", "sigmoid"], default="identity")
    parser.add_argument("--map_temperature", type=float, default=1.0)
    parser.add_argument("--hybrid_source", type=str, choices=["none", "tl", "baseline"], default="none")
    parser.add_argument("--hybrid_weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_metrics", type=str, nargs="+", default=["I-AUROC", "I-AP", "P-AUROC", "P-AP", "P-AUPRO"])
    parser.add_argument("--aupro_max_step", type=int, default=200)
    parser.add_argument("--aupro_expect_fpr", type=float, default=0.3)
    parser.add_argument("--report_group_separability", action="store_true")
    parser.add_argument("--separability_hard_frac", type=float, default=0.01)
    args = parser.parse_args()

    if args.source_exclude_class is not None and args.source_exclude_class.lower() in {"", "none", "null", "all"}:
        args.source_exclude_class = None
    if args.class_name is not None and args.class_name.lower() in {"", "none", "null", "all"}:
        args.class_name = None
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, textual_learner, visual_learner, learned_text_features, dpam_layer = build_model(
        args.device,
        args.checkpoint_path,
        args.image_size,
        args.n_ctx,
        args.vl_reduction,
        args.pretrained_model,
        checkpoint_load_mode=args.checkpoint_load_mode,
    )
    axis = decision_axis(learned_text_features).to(args.device)

    fp_banks, def_banks, bank_stats, bank_cache_path, bank_cache_meta = get_or_collect_source_patch_banks(
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
        source_exclude_class=args.source_exclude_class,
        bank_score_source=args.bank_score_source,
        hard_frac=args.hard_frac,
        max_good_per_class=args.max_good_per_class,
        max_defect_per_class=args.max_defect_per_class,
        max_bank_per_layer=args.max_bank_per_layer,
        max_fp_per_image=args.max_fp_per_image,
        max_defect_per_image=args.max_defect_per_image,
        checkpoint_path=args.checkpoint_path,
        bank_cache_path=args.bank_cache_path,
        refresh_bank_cache=args.refresh_bank_cache,
    )

    projected_banks = {}
    for layer_idx in range(len(args.features_list)):
        fp_coeff, _ = project_bank(fp_banks[layer_idx].to(args.device), axis)
        def_coeff, _ = project_bank(def_banks[layer_idx].to(args.device), axis)
        projected_banks[layer_idx] = {"fp": fp_coeff, "defect": def_coeff}

    dataset = build_dataset(
        args.target_root,
        args.image_size,
        args.target_dataset,
        "test",
        str(Path(args.save_dir)),
        class_name=args.class_name,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    results = {
        "baseline_txt": {"sample_ids": [], "gt_masks": [], "pr_masks": [], "cls_names": [], "gt_anomalys": [], "pr_anomalys": [], "query_paths": []},
        "parallel_margin": {"sample_ids": [], "gt_masks": [], "pr_masks": [], "cls_names": [], "gt_anomalys": [], "pr_anomalys": [], "query_paths": []},
    }
    if args.include_fullfeat_control:
        results["fullfeat_margin"] = {"sample_ids": [], "gt_masks": [], "pr_masks": [], "cls_names": [], "gt_anomalys": [], "pr_anomalys": [], "query_paths": []}

    with torch.no_grad():
        for idx, items in enumerate(loader, start=1):
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

            parallel_layer_maps = []
            fullfeat_layer_maps = []
            for layer_idx, patch_feature in enumerate(baseline["query_patch_feats"]):
                patch = F.normalize(patch_feature[:, 1:, :].reshape(-1, patch_feature.shape[-1]).float(), dim=-1)
                coeff = patch @ axis
                q_fp = logmeanexp_negative_sqdist_1d(coeff, projected_banks[layer_idx]["fp"], tau=args.tau)
                q_def = logmeanexp_negative_sqdist_1d(coeff, projected_banks[layer_idx]["defect"], tau=args.tau)
                h = int(round(patch.shape[0] ** 0.5))
                parallel_map = (q_def - args.fp_weight * q_fp).reshape(1, 1, h, h)
                parallel_map = F.interpolate(parallel_map, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)[:, 0]
                parallel_layer_maps.append(parallel_map)
                if args.include_fullfeat_control:
                    q_fp_full = logmeanexp_cosine_support(
                        patch,
                        fp_banks[layer_idx].to(args.device),
                        tau=args.fullfeat_tau,
                        chunk=args.fullfeat_chunk,
                    )
                    q_def_full = logmeanexp_cosine_support(
                        patch,
                        def_banks[layer_idx].to(args.device),
                        tau=args.fullfeat_tau,
                        chunk=args.fullfeat_chunk,
                    )
                    fullfeat_map = (q_def_full - args.fp_weight * q_fp_full).reshape(1, 1, h, h)
                    fullfeat_map = F.interpolate(fullfeat_map, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)[:, 0]
                    fullfeat_layer_maps.append(fullfeat_map)

            parallel_map = torch.stack(parallel_layer_maps, dim=0).mean(dim=0)
            parallel_map = smooth_map(parallel_map, sigma=args.sigma).to(args.device)
            parallel_map = activate_margin_map(parallel_map, activation=args.map_activation, temperature=args.map_temperature)
            if args.hybrid_source != "none" and args.hybrid_weight > 0:
                if args.hybrid_source == "tl":
                    hybrid_ref = smooth_map(baseline["local_tl_map"], sigma=args.sigma).to(args.device)
                    hybrid_ref = resize_map(hybrid_ref, args.image_size)
                elif args.hybrid_source == "baseline":
                    hybrid_ref = baseline["baseline_map"]
                else:
                    raise ValueError(f"Unsupported hybrid_source: {args.hybrid_source}")
                parallel_map = (1.0 - args.hybrid_weight) * parallel_map + args.hybrid_weight * hybrid_ref
            anomaly_map_max_parallel, _ = torch.max(parallel_map.view(image.shape[0], -1), dim=1)
            parallel_img_score = fusion_fun(
                [baseline["global_vl_score"], baseline["global_tl_score"], anomaly_map_max_parallel],
                fusion_type=args.fusion_type,
            )
            mode_triplets = [
                ("baseline_txt", baseline["baseline_map"], baseline["baseline_img_score"]),
                ("parallel_margin", parallel_map, parallel_img_score),
            ]
            if args.include_fullfeat_control:
                fullfeat_map = torch.stack(fullfeat_layer_maps, dim=0).mean(dim=0)
                fullfeat_map = smooth_map(fullfeat_map, sigma=args.sigma).to(args.device)
                fullfeat_map = activate_margin_map(fullfeat_map, activation=args.map_activation, temperature=args.map_temperature)
                anomaly_map_max_fullfeat, _ = torch.max(fullfeat_map.view(image.shape[0], -1), dim=1)
                fullfeat_img_score = fusion_fun(
                    [baseline["global_vl_score"], baseline["global_tl_score"], anomaly_map_max_fullfeat],
                    fusion_type=args.fusion_type,
                )
                mode_triplets.append(("fullfeat_margin", fullfeat_map, fullfeat_img_score))

            for mode, map_tensor, image_score in mode_triplets:
                results[mode]["sample_ids"].append(sample_id)
                results[mode]["cls_names"].append(cls_name)
                results[mode]["query_paths"].append(query_path)
                results[mode]["gt_masks"].append(gt_mask.detach().cpu().int())
                results[mode]["pr_masks"].append(map_tensor.detach().cpu())
                results[mode]["gt_anomalys"].append(gt_anomaly.detach().cpu())
                results[mode]["pr_anomalys"].append(image_score.detach().cpu())

            if idx % 8 == 0:
                print(json.dumps({"stage": "eval_progress", "images_done": idx, "current_class": cls_name}), flush=True)

    rows = []
    separability_mean = {}
    class_order = dataset.obj_list if args.class_name is None else [args.class_name]
    eval_modes = ["baseline_txt", "parallel_margin"]
    if args.include_fullfeat_control:
        eval_modes.append("fullfeat_margin")
    for mode in eval_modes:
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
        "source_exclude_class": args.source_exclude_class,
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
        "bank_score_source": args.bank_score_source,
        "tau": args.tau,
        "fp_weight": args.fp_weight,
        "map_activation": args.map_activation,
        "map_temperature": args.map_temperature,
        "hybrid_source": args.hybrid_source,
        "hybrid_weight": args.hybrid_weight,
        "eval_metrics": args.eval_metrics,
        "aupro_max_step": args.aupro_max_step,
        "aupro_expect_fpr": args.aupro_expect_fpr,
        "report_group_separability": args.report_group_separability,
        "separability_hard_frac": args.separability_hard_frac,
        "aupro_backend": "fast_threshold_cpu",
        "bank_stats": bank_stats,
        "bank_cache_path": str(bank_cache_path),
        "bank_cache_meta": bank_cache_meta,
        "score_definition": {
            "baseline_txt": "official AdaptCLIP zero-shot baseline",
            "parallel_margin": "official global branches preserved; local anomaly map replaced by multilayer text-conditioned competitive margin",
            "fullfeat_margin": "visual-feature memory control; local anomaly map replaced by full-feature defect-vs-hard-FP cosine support",
        },
    }
    if args.report_group_separability:
        summary["separability_mean"] = separability_mean
    save_results(rows, summary, Path(args.save_dir))
    print(json.dumps(jsonable({"summary": summary, "rows": rows}), indent=2))


if __name__ == "__main__":
    main()
