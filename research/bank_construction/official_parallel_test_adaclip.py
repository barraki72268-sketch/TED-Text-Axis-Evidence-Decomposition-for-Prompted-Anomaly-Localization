from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import measure

ROOT = Path("/mnt/data/pilab-kingjinyoung/ted-reproduce-20261008/snapshot")
ADA_ROOT = ROOT / "neurips2026" / "AdaCLIP"
ADAPT_ROOT = ROOT / "neurips2026" / "AdaptCLIP"

if str(ADA_ROOT) not in sys.path:
    sys.path.insert(0, str(ADA_ROOT))
if str(ADAPT_ROOT) not in sys.path:
    sys.path.append(str(ADAPT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-adaclip")

from dataset import dataset_dict  # noqa: E402
from method.trainer import AdaCLIP_Trainer  # noqa: E402


def load_adaptclip_evaluator():
    metric_path = ADAPT_ROOT / "tools" / "effecient_metric.py"
    spec = importlib.util.spec_from_file_location("adaptclip_effecient_metric", metric_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Evaluator


Evaluator = load_adaptclip_evaluator()
_GAUSSIAN_KERNEL_CACHE: dict[tuple[float, str, str], torch.Tensor] = {}


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


def decision_axis(text_feature: torch.Tensor) -> torch.Tensor:
    if text_feature.ndim == 3:
        text_feature = text_feature[0]
    axis = text_feature[:, 1] - text_feature[:, 0]
    return F.normalize(axis.float(), dim=-1)


def resolve_layer_recipe(features_list: list[int], recipe: str) -> list[int]:
    num_layers = len(features_list)
    if num_layers == 0:
        raise ValueError("features_list must not be empty")
    if recipe == "all4":
        return list(range(num_layers))
    if recipe == "last":
        return [num_layers - 1]
    if recipe == "last2":
        return list(range(max(0, num_layers - 2), num_layers))
    if recipe == "earlylate":
        return sorted({0, num_layers - 1})
    if recipe == "mid2":
        if num_layers <= 2:
            return list(range(num_layers))
        mid_left = max(0, (num_layers // 2) - 1)
        mid_right = min(num_layers - 1, mid_left + 1)
        return sorted({mid_left, mid_right})
    raise ValueError(f"Unsupported layer_recipe: {recipe}")


def select_by_indices(items: list, indices: list[int]) -> list:
    return [items[i] for i in indices]


def logmeanexp_negative_sqdist_1d(coeff: torch.Tensor, bank_coeff: torch.Tensor, tau: float) -> torch.Tensor:
    if bank_coeff.numel() == 0:
        return coeff.new_zeros((coeff.shape[0],))
    dist2 = (coeff.unsqueeze(1) - bank_coeff.unsqueeze(0)) ** 2
    logits = -dist2 / tau
    return torch.logsumexp(logits, dim=-1) - math.log(bank_coeff.shape[0])


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if sigma <= 0:
        return torch.ones((1,), device=device, dtype=dtype)
    key = (float(sigma), str(device), str(dtype))
    kernel = _GAUSSIAN_KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    _GAUSSIAN_KERNEL_CACHE[key] = kernel
    return kernel


def smooth_map(batch_map: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return batch_map
    if batch_map.ndim == 3:
        work = batch_map[:, None]
        squeeze_channel = True
    elif batch_map.ndim == 4:
        work = batch_map
        squeeze_channel = False
    else:
        raise ValueError(f"Expected 3D or 4D tensor for smoothing, got shape {tuple(batch_map.shape)}")

    work = work.float()
    kernel_1d = _gaussian_kernel1d(float(sigma), work.device, work.dtype)
    channels = work.shape[1]
    kernel_x = kernel_1d.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    pad = kernel_1d.numel() // 2

    work = F.pad(work, (pad, pad, 0, 0), mode="reflect")
    work = F.conv2d(work, kernel_x, groups=channels)
    work = F.pad(work, (0, 0, pad, pad), mode="reflect")
    work = F.conv2d(work, kernel_y, groups=channels)
    return work[:, 0] if squeeze_channel else work


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


def build_model(
    device: str,
    checkpoint_path: str,
    skip_checkpoint_load: bool,
    backbone: str,
    checkpoint_load_mode: str,
    image_size: int,
    model_layer_recipe: str,
    output_layers: list[int] | None,
    prompting_depth: int,
    prompting_length: int,
    prompting_branch: str,
    prompting_type: str,
    use_hsf: bool,
    k_clusters: int,
):
    config_path = ADA_ROOT / "model_configs" / f"{backbone}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"AdaCLIP model config not found for backbone={backbone}: {config_path}")
    model_configs = json.loads(config_path.read_text())
    n_layers = int(model_configs["vision_cfg"]["layers"])
    substage = n_layers // 4
    full_feat_list = [substage, substage * 2, substage * 3, substage * 4]
    if output_layers is not None:
        feat_list = [int(x) for x in output_layers]
    elif model_layer_recipe == "all4":
        feat_list = full_feat_list
    elif model_layer_recipe == "last":
        feat_list = [full_feat_list[-1]]
    elif model_layer_recipe == "last2":
        feat_list = full_feat_list[-2:]
    elif model_layer_recipe == "earlylate":
        feat_list = [full_feat_list[0], full_feat_list[-1]]
    elif model_layer_recipe == "mid2":
        feat_list = full_feat_list[1:3]
    else:
        raise ValueError(f"Unsupported model_layer_recipe: {model_layer_recipe}")
    model = AdaCLIP_Trainer(
        backbone=backbone,
        feat_list=feat_list,
        input_dim=int(model_configs["vision_cfg"]["width"]),
        output_dim=int(model_configs["embed_dim"]),
        learning_rate=0.0,
        device=device,
        image_size=image_size,
        prompting_depth=prompting_depth,
        prompting_length=prompting_length,
        prompting_branch=prompting_branch,
        prompting_type=prompting_type,
        use_hsf=use_hsf,
        k_clusters=k_clusters,
        use_ir_scorer=False,
        text_leakage_reg=False,
    ).to(device)
    load_info = {
        "backbone": backbone,
        "feat_list": list(feat_list),
        "model_layer_recipe": model_layer_recipe,
        "checkpoint_load_mode": "none" if skip_checkpoint_load else checkpoint_load_mode,
        "loaded_keys": 0,
        "skipped_keys": 0,
        "missing_keys": 0,
        "unexpected_keys": 0,
    }
    if not skip_checkpoint_load:
        if checkpoint_load_mode == "strict":
            model.load(checkpoint_path)
            load_info["checkpoint_load_mode"] = "strict"
        elif checkpoint_load_mode == "compatible":
            source_state = torch.load(checkpoint_path, map_location="cpu")
            target_state = model.state_dict()
            compatible_state = {}
            skipped = []
            for key, value in source_state.items():
                if key in target_state and tuple(value.shape) == tuple(target_state[key].shape):
                    compatible_state[key] = value
                else:
                    skipped.append(key)
            incompatible = model.load_state_dict(compatible_state, strict=False)
            load_info.update(
                {
                    "loaded_keys": len(compatible_state),
                    "skipped_keys": len(skipped),
                    "missing_keys": len(incompatible.missing_keys),
                    "unexpected_keys": len(incompatible.unexpected_keys),
                }
            )
        else:
            raise ValueError(f"Unsupported checkpoint_load_mode: {checkpoint_load_mode}")
    model.eval()
    model.clip_model.eval()
    model._ted_checkpoint_load_info = load_info
    return model


def build_target_dataset(root: str, dataset_name: str, model, training: bool, class_name: str | None = None):
    cls_names, dataset_cls, _ = dataset_dict[dataset_name]
    if class_name is not None:
        cls_names = [class_name]
    return dataset_cls(
        clsnames=cls_names,
        transform=model.preprocess,
        target_transform=model.transform,
        root=root,
        training=training,
    )


class GenericMetaDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transform, target_transform, split: str, class_name: str | None = None):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        meta = json.loads((self.root / "meta.json").read_text())
        split_meta = meta[split]
        if class_name is not None:
            split_meta = {class_name: split_meta[class_name]}
        self.cls_names = sorted(split_meta.keys())
        self.items = []
        for cls_name in self.cls_names:
            self.items.extend(split_meta[cls_name])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        data = self.items[index]
        img_path = self.root / data["img_path"]
        mask_path = data.get("mask_path")
        if str(img_path).endswith(".tif"):
            import cv2

            img = cv2.imread(str(img_path))
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            img = Image.open(img_path).convert("RGB")

        anomaly = int(data["anomaly"])
        if anomaly == 0 or not mask_path:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
        else:
            raw_mask = np.array(Image.open(self.root / mask_path).convert("L")) > 0
            img_mask = Image.fromarray(raw_mask.astype(np.uint8) * 255, mode="L")

        img_out = self.transform(img) if self.transform is not None else img
        mask_out = self.target_transform(img_mask) if self.target_transform is not None else img_mask
        return {
            "img": img_out,
            "img_mask": mask_out,
            "cls_name": data["cls_name"],
            "anomaly": anomaly,
            "img_path": str(img_path),
        }


def compute_baseline_outputs(model, image: torch.Tensor, cls_name: str, sigma: float, selected_layer_indices: list[int] | None = None):
    image_features, patch_tokens, text_features = model.clip_model.extract_feat(image, [cls_name])
    norm_patch_tokens = [F.normalize(x.float(), dim=-1) for x in patch_tokens]
    if selected_layer_indices is None:
        selected_layer_indices = list(range(len(norm_patch_tokens)))
    selected_patch_tokens = select_by_indices(norm_patch_tokens, selected_layer_indices)
    anomaly_map, anomaly_score = model.clip_model.visual_text_similarity(
        image_features,
        selected_patch_tokens,
        text_features,
        aggregation=True,
    )
    anomaly_map = anomaly_map.squeeze(1)
    anomaly_map = smooth_map(anomaly_map, sigma=sigma).to(image.device)
    axis = decision_axis(text_features).to(image.device)
    layer_logits_all = []
    layer_prob_maps_all = []
    layer_scalar_maps_all = []
    layer_token_scores_all = []
    layer_token_logits_all = []
    for patch_feature in norm_patch_tokens:
        layer_token_logit = 100.0 * patch_feature @ text_features
        token_prob = torch.softmax(layer_token_logit, dim=-1)
        layer_token_scores_all.append((token_prob[:, :, 1] + 1.0 - token_prob[:, :, 0]) / 2.0)
        layer_token_logits_all.append(layer_token_logit)
        layer_logit = token_logits_to_layer_logits(layer_token_logit, model.clip_model.image_size)
        layer_prob = torch.softmax(layer_logit, dim=1)
        layer_logits_all.append(layer_logit)
        layer_prob_maps_all.append(layer_prob)
        layer_scalar_maps_all.append(layer_map_to_scalar_map(layer_prob)[:, 0])
    return {
        "image_features": image_features,
        "selected_layer_indices": selected_layer_indices,
        "patch_tokens": selected_patch_tokens,
        "text_features": text_features,
        "axis": axis,
        "baseline_map": anomaly_map,
        "baseline_img_score": anomaly_score.detach(),
        "layer_logits": select_by_indices(layer_logits_all, selected_layer_indices),
        "layer_token_logits": select_by_indices(layer_token_logits_all, selected_layer_indices),
        "layer_prob_maps": select_by_indices(layer_prob_maps_all, selected_layer_indices),
        "layer_scalar_maps": select_by_indices(layer_scalar_maps_all, selected_layer_indices),
        "layer_token_scores": select_by_indices(layer_token_scores_all, selected_layer_indices),
    }


def layer_map_to_scalar_map(layer_map: torch.Tensor) -> torch.Tensor:
    return (layer_map[:, 1:2, :, :] + 1 - layer_map[:, 0:1, :, :]) / 2.0


def token_logits_to_layer_logits(token_logits: torch.Tensor, image_size: int) -> torch.Tensor:
    bsz, num_tokens, _ = token_logits.shape
    side = int(round(num_tokens ** 0.5))
    layer_logits = token_logits.permute(0, 2, 1).view(bsz, 2, side, side)
    return F.interpolate(layer_logits, size=(image_size, image_size), mode="bilinear", align_corners=True)


def logits_to_scalar_map(layer_logits: torch.Tensor) -> torch.Tensor:
    layer_prob = torch.softmax(layer_logits, dim=1)
    return layer_map_to_scalar_map(layer_prob)[:, 0]


def topk_image_score_from_map(map_tensor: torch.Tensor, topk_frac: float) -> torch.Tensor:
    flat = map_tensor.flatten(1)
    k = max(1, int(round(flat.shape[1] * float(topk_frac))))
    k = min(k, flat.shape[1])
    return torch.topk(flat, k=k, dim=1, largest=True).values.mean(dim=1)


def select_image_score(
    official_score: torch.Tensor,
    map_tensor: torch.Tensor,
    mode: str,
    topk_frac: float,
) -> torch.Tensor:
    official_score = official_score.float().view(-1)
    map_score = topk_image_score_from_map(map_tensor.float(), topk_frac).view(-1)
    if mode == "official":
        return official_score
    if mode == "map_topk":
        return map_score
    if mode == "official_plus_map_topk":
        return 0.5 * official_score + 0.5 * map_score
    if mode == "max_official_map_topk":
        return torch.maximum(official_score, map_score)
    raise ValueError(f"Unsupported image score mode: {mode}")


def add_to_logit_difference(layer_logits: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    if delta.ndim == 3:
        delta = delta[:, None]
    logit_center = 0.5 * (layer_logits[:, 1:2] + layer_logits[:, 0:1])
    logit_diff = layer_logits[:, 1:2] - layer_logits[:, 0:1]
    refined_diff = logit_diff + delta
    refined_norm = logit_center - 0.5 * refined_diff
    refined_anom = logit_center + 0.5 * refined_diff
    return torch.cat([refined_norm, refined_anom], dim=1)


def normalize_margin_map(margin_map: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return margin_map
    flat = margin_map.flatten(1)
    if mode == "spatial_zscore":
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
        return ((flat - mean) / std).view_as(margin_map)
    if mode == "spatial_tanh_zscore":
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.tanh(((flat - mean) / std)).view_as(margin_map)
    raise ValueError(f"Unsupported margin normalization mode: {mode}")


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
    model_type = plugin.get("model_type", "host_linear")
    if model_type == "host_mlp":
        hidden = F.relu(x @ plugin["w1"].to(host_score.device) + plugin["b1"].to(host_score.device))
        return hidden @ plugin["w2"].to(host_score.device) + plugin["b2"].to(host_score.device)
    return (x.squeeze(-1) * plugin["weight"].to(host_score.device)) + plugin["bias"].to(host_score.device)


def train_hostscore_plugin_for_layer(
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
    if fp_all.numel() == 0 or def_all.numel() == 0:
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

        for _ in range(epochs):
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
    for _ in range(epochs):
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


def train_scalar_plugin_for_layer(
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
    fp = F.normalize(fp_bank.float(), dim=-1)
    defect = F.normalize(def_bank.float(), dim=-1)
    n = min(fp.shape[0], defect.shape[0], max_train_points // 2 if max_train_points > 0 else 10**9)
    if n <= 0:
        return None

    axis = F.normalize(axis.to(device).float(), dim=0)
    fp_all = fp.to(device)
    defect_all = defect.to(device)
    fp_proj_all = fp_all @ axis
    def_proj_all = defect_all @ axis
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


def host_gate_map(host_map: torch.Tensor, quantile: float, temp: float) -> torch.Tensor:
    flat = host_map.flatten(1)
    threshold = torch.quantile(flat, float(quantile), dim=1).view(-1, 1, 1)
    scale = flat.std(dim=1).view(-1, 1, 1).clamp_min(1e-6)
    return torch.sigmoid((host_map - threshold) / (scale * max(float(temp), 1e-6)))


def support_mass_gate_map(
    q_def: torch.Tensor,
    q_fp: torch.Tensor,
    grid_size: int,
    image_size: int,
    quantile: float,
    temp: float,
) -> torch.Tensor:
    bsz = int(q_def.shape[0]) if q_def.ndim > 1 else 1
    support = (q_def + q_fp).reshape(bsz, 1, grid_size, grid_size)
    support = F.interpolate(
        support,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=True,
    )[:, 0]
    flat = support.flatten(1)
    threshold = torch.quantile(flat, float(quantile), dim=1).view(-1, 1, 1)
    scale = flat.std(dim=1).view(-1, 1, 1).clamp_min(1e-6)
    return torch.sigmoid((support - threshold) / (scale * max(float(temp), 1e-6)))


def apply_residual_activation(residual_map: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "signed":
        return residual_map
    if mode == "positive":
        return F.relu(residual_map)
    if mode == "negative":
        return -F.relu(-residual_map)
    raise ValueError(f"Unsupported residual activation mode: {mode}")


def refine_layer_logits(
    layer_logits: torch.Tensor,
    margin_map: torch.Tensor,
    scoring_mode: str,
    refine_alpha: float,
    refine_beta: float,
    margin_norm_mode: str,
) -> torch.Tensor:
    if scoring_mode == "replace":
        raise ValueError("replace mode does not refine layer logits")

    margin_norm = normalize_margin_map(margin_map, margin_norm_mode)
    logit_center = 0.5 * (layer_logits[:, 1:2] + layer_logits[:, 0:1])
    logit_diff = layer_logits[:, 1:2] - layer_logits[:, 0:1]

    if scoring_mode == "logit_add":
        refined_diff = logit_diff + refine_alpha * margin_norm
    elif scoring_mode == "logit_gate":
        refined_diff = logit_diff + refine_alpha * F.relu(logit_diff) * margin_norm
    elif scoring_mode == "logit_gate_posneg":
        refined_diff = (
            logit_diff
            + refine_alpha * F.relu(logit_diff) * F.relu(margin_norm)
            - refine_beta * F.relu(logit_diff) * F.relu(-margin_norm)
        )
    else:
        raise ValueError(f"Unsupported scoring_mode: {scoring_mode}")

    refined_norm = logit_center - 0.5 * refined_diff
    refined_anom = logit_center + 0.5 * refined_diff
    return torch.cat([refined_norm, refined_anom], dim=1)


def default_bank_cache_path(
    checkpoint_path: str,
    skip_checkpoint_load: bool,
    seed: int,
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
        "skip_checkpoint_load": bool(skip_checkpoint_load),
        "seed": int(seed),
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
    return ROOT / "neurips2026" / "results" / "bank_cache" / f"adaclip_{source_tag}_exclude-{exclude_tag}_{digest}.pt"


def load_bank_cache(cache_path: Path):
    payload = torch.load(cache_path, map_location="cpu")
    fp_banks = {int(k): v.float() for k, v in payload["fp_banks"].items()}
    def_banks = {int(k): v.float() for k, v in payload["def_banks"].items()}
    fp_scores = {int(k): v.float() for k, v in payload.get("fp_scores", {}).items()}
    def_scores = {int(k): v.float() for k, v in payload.get("def_scores", {}).items()}
    bank_stats = {int(k): v for k, v in payload["bank_stats"].items()}
    cache_meta = payload.get("cache_meta", {})
    return fp_banks, def_banks, fp_scores, def_scores, bank_stats, cache_meta


def save_bank_cache(
    cache_path: Path,
    fp_banks: dict[int, torch.Tensor],
    def_banks: dict[int, torch.Tensor],
    fp_scores: dict[int, torch.Tensor],
    def_scores: dict[int, torch.Tensor],
    bank_stats: dict[int, dict],
    cache_meta: dict,
):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fp_banks": {int(k): v.detach().cpu() for k, v in fp_banks.items()},
        "def_banks": {int(k): v.detach().cpu() for k, v in def_banks.items()},
        "fp_scores": {int(k): v.detach().cpu() for k, v in fp_scores.items()},
        "def_scores": {int(k): v.detach().cpu() for k, v in def_scores.items()},
        "bank_stats": bank_stats,
        "cache_meta": cache_meta,
    }
    torch.save(payload, cache_path)


def normalize_bank_cache_meta(cache_meta: dict | None) -> dict:
    meta = dict(cache_meta or {})
    meta.setdefault("skip_checkpoint_load", False)
    meta.setdefault("seed", 0)
    if "features_list" in meta:
        meta["features_list"] = list(meta["features_list"])
    return meta


def collect_source_patch_banks(
    model,
    source_root: str,
    source_dataset: str,
    image_size: int,
    features_list: list[int],
    device: str,
    source_exclude_class: str | None,
    hard_frac: float,
    max_good_per_class: int,
    max_defect_per_class: int,
    max_bank_per_layer: int,
    max_fp_per_image: int,
    max_defect_per_image: int,
    sigma: float,
    source_fp_mode: str = "hardfp",
):
    train_ds = GenericMetaDataset(source_root, model.preprocess, model.transform, split="train")
    test_ds = GenericMetaDataset(source_root, model.preprocess, model.transform, split="test")

    fp_chunks = {i: [] for i in range(len(features_list))}
    def_chunks = {i: [] for i in range(len(features_list))}
    fp_score_chunks = {i: [] for i in range(len(features_list))}
    def_score_chunks = {i: [] for i in range(len(features_list))}
    per_good = {cls_name: 0 for cls_name in train_ds.cls_names}
    per_defect = {cls_name: 0 for cls_name in test_ds.cls_names}

    with torch.no_grad():
        for idx in range(len(train_ds)):
            item = train_ds[idx]
            cls_name = item["cls_name"]
            if source_exclude_class is not None and cls_name == source_exclude_class:
                continue
            if int(item["anomaly"]) != 0:
                continue
            if per_good[cls_name] >= max_good_per_class:
                continue
            per_good[cls_name] += 1

            image = item["img"].unsqueeze(0).to(device)
            image_features, patch_tokens, text_features = model.clip_model.extract_feat(image, [cls_name])
            layer_maps, _ = model.clip_model.visual_text_similarity(
                image_features,
                patch_tokens,
                text_features,
                aggregation=False,
            )
            for layer_idx, patch_feature in enumerate(patch_tokens):
                patch = F.normalize(patch_feature[0].float(), dim=-1)
                h = int(round(patch.shape[0] ** 0.5))
                layer_map = layer_map_to_scalar_map(layer_maps[layer_idx])[0, 0]
                score_small = F.interpolate(
                    layer_map[None, None],
                    size=(h, h),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].reshape(-1)
                k = max(1, int(score_small.numel() * hard_frac))
                if source_fp_mode == "hardfp":
                    selected_idx = torch.topk(score_small, k=k, largest=True).indices
                elif source_fp_mode == "easy_normal":
                    selected_idx = torch.topk(score_small, k=k, largest=False).indices
                elif source_fp_mode == "random_normal":
                    selected_idx = torch.randperm(score_small.numel(), device=score_small.device)[:k]
                else:
                    raise ValueError(f"Unsupported source_fp_mode: {source_fp_mode}")
                selected = patch[selected_idx]
                selected_scores = score_small[selected_idx]
                if selected.shape[0] > max_fp_per_image:
                    keep = farthest_point_subsample_indices(selected, max_fp_per_image)
                    selected = selected[keep]
                    selected_scores = selected_scores[keep]
                fp_chunks[layer_idx].append(selected.cpu())
                fp_score_chunks[layer_idx].append(selected_scores.cpu())
            if (idx + 1) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "bank_progress",
                            "split": "train_good",
                            "images_seen": idx + 1,
                            "source_dataset": source_dataset,
                            "source_exclude_class": source_exclude_class,
                        }
                    ),
                    flush=True,
                )

        for idx in range(len(test_ds)):
            item = test_ds[idx]
            cls_name = item["cls_name"]
            if source_exclude_class is not None and cls_name == source_exclude_class:
                continue
            if int(item["anomaly"]) == 0:
                continue
            if per_defect[cls_name] >= max_defect_per_class:
                continue
            per_defect[cls_name] += 1

            image = item["img"].unsqueeze(0).to(device)
            gt_mask = item["img_mask"].unsqueeze(0).to(device).float()
            image_features, patch_tokens, text_features = model.clip_model.extract_feat(image, [cls_name])
            layer_maps, _ = model.clip_model.visual_text_similarity(
                image_features,
                patch_tokens,
                text_features,
                aggregation=False,
            )
            for layer_idx, patch_feature in enumerate(patch_tokens):
                patch = F.normalize(patch_feature[0].float(), dim=-1)
                h = int(round(patch.shape[0] ** 0.5))
                gt_small = F.interpolate(gt_mask, size=(h, h), mode="nearest")[0, 0].reshape(-1) > 0.5
                if int(gt_small.sum().item()) == 0:
                    continue
                layer_map = layer_map_to_scalar_map(layer_maps[layer_idx])[0, 0]
                score_small = F.interpolate(
                    layer_map[None, None],
                    size=(h, h),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].reshape(-1)
                selected_idx = gt_small
                selected = patch[selected_idx]
                selected_scores = score_small[selected_idx]
                if selected.shape[0] > max_defect_per_image:
                    mask_scores = score_small[selected_idx]
                    keep = torch.topk(mask_scores, k=max_defect_per_image, largest=True).indices
                    selected = selected[keep]
                    selected_scores = selected_scores[keep]
                def_chunks[layer_idx].append(selected.cpu())
                def_score_chunks[layer_idx].append(selected_scores.cpu())
            if (idx + 1) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "bank_progress",
                            "split": "test_anomaly",
                            "images_seen": idx + 1,
                            "source_dataset": source_dataset,
                            "source_exclude_class": source_exclude_class,
                        }
                    ),
                    flush=True,
                )

    feat_dim = 768
    fp_banks = {}
    def_banks = {}
    fp_scores = {}
    def_scores = {}
    bank_stats = {}
    for layer_idx in range(len(features_list)):
        fp_bank = torch.cat(fp_chunks[layer_idx], dim=0) if fp_chunks[layer_idx] else torch.empty((0, feat_dim))
        def_bank = torch.cat(def_chunks[layer_idx], dim=0) if def_chunks[layer_idx] else torch.empty((0, feat_dim))
        fp_score = torch.cat(fp_score_chunks[layer_idx], dim=0) if fp_score_chunks[layer_idx] else torch.empty((0,))
        def_score = torch.cat(def_score_chunks[layer_idx], dim=0) if def_score_chunks[layer_idx] else torch.empty((0,))
        if fp_bank.shape[0] > max_bank_per_layer:
            keep = farthest_point_subsample_indices(fp_bank, max_bank_per_layer)
            fp_bank = fp_bank[keep]
            fp_score = fp_score[keep]
        if def_bank.shape[0] > max_bank_per_layer:
            keep = farthest_point_subsample_indices(def_bank, max_bank_per_layer)
            def_bank = def_bank[keep]
            def_score = def_score[keep]
        fp_banks[layer_idx] = F.normalize(fp_bank.float(), dim=-1) if fp_bank.numel() else fp_bank.float()
        def_banks[layer_idx] = F.normalize(def_bank.float(), dim=-1) if def_bank.numel() else def_bank.float()
        fp_scores[layer_idx] = fp_score.float()
        def_scores[layer_idx] = def_score.float()
        bank_stats[layer_idx] = {
            "num_fp": int(fp_banks[layer_idx].shape[0]),
            "num_defect": int(def_banks[layer_idx].shape[0]),
            "has_host_scores": bool(fp_scores[layer_idx].numel() and def_scores[layer_idx].numel()),
        }
    return fp_banks, def_banks, fp_scores, def_scores, bank_stats


def get_or_collect_source_patch_banks(
    model,
    skip_checkpoint_load: bool,
    seed: int,
    source_root: str,
    source_dataset: str,
    image_size: int,
    features_list: list[int],
    device: str,
    source_exclude_class: str | None,
    hard_frac: float,
    max_good_per_class: int,
    max_defect_per_class: int,
    max_bank_per_layer: int,
    max_fp_per_image: int,
    max_defect_per_image: int,
    sigma: float,
    checkpoint_path: str,
    bank_cache_path: str | None,
    refresh_bank_cache: bool,
    require_host_scores: bool = False,
    source_fp_mode: str = "hardfp",
):
    requested_meta = {
        "checkpoint_path": checkpoint_path,
        "skip_checkpoint_load": bool(skip_checkpoint_load),
        "seed": int(seed),
        "source_root": source_root,
        "source_dataset": source_dataset,
        "source_exclude_class": source_exclude_class,
        "image_size": image_size,
        "features_list": list(features_list),
        "sigma": sigma,
        "hard_frac": hard_frac,
        "max_good_per_class": max_good_per_class,
        "max_defect_per_class": max_defect_per_class,
        "max_bank_per_layer": max_bank_per_layer,
        "max_fp_per_image": max_fp_per_image,
        "max_defect_per_image": max_defect_per_image,
        "source_fp_mode": source_fp_mode,
    }
    cache_path = Path(bank_cache_path) if bank_cache_path else default_bank_cache_path(
        checkpoint_path=checkpoint_path,
        skip_checkpoint_load=skip_checkpoint_load,
        seed=seed,
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
        fp_banks, def_banks, fp_scores, def_scores, bank_stats, cache_meta = load_bank_cache(cache_path)
        normalized_cache_meta = normalize_bank_cache_meta(cache_meta)
        normalized_requested_meta = normalize_bank_cache_meta(requested_meta)
        has_scores = all(
            fp_scores.get(i, torch.empty(0)).numel() and def_scores.get(i, torch.empty(0)).numel()
            for i in range(len(features_list))
        )
        if normalized_cache_meta == normalized_requested_meta and (has_scores or not require_host_scores):
            return fp_banks, def_banks, fp_scores, def_scores, bank_stats, cache_path, cache_meta
        print(
            json.dumps(
                {
                    "stage": "bank_cache_mismatch",
                    "cache_path": str(cache_path),
                    "cached_meta": normalized_cache_meta,
                    "requested_meta": normalized_requested_meta,
                    "require_host_scores": bool(require_host_scores),
                    "cache_has_host_scores": bool(has_scores),
                }
            ),
            flush=True,
        )

    fp_banks, def_banks, fp_scores, def_scores, bank_stats = collect_source_patch_banks(
        model=model,
        source_root=source_root,
        source_dataset=source_dataset,
        image_size=image_size,
        features_list=features_list,
        device=device,
        source_exclude_class=source_exclude_class,
        hard_frac=hard_frac,
        max_good_per_class=max_good_per_class,
        max_defect_per_class=max_defect_per_class,
        max_bank_per_layer=max_bank_per_layer,
        max_fp_per_image=max_fp_per_image,
        max_defect_per_image=max_defect_per_image,
            sigma=sigma,
            source_fp_mode=source_fp_mode,
        )
    save_bank_cache(cache_path, fp_banks, def_banks, fp_scores, def_scores, bank_stats, requested_meta)
    return fp_banks, def_banks, fp_scores, def_scores, bank_stats, cache_path, requested_meta


def compute_source_global_axis(
    model,
    source_root: str,
    source_exclude_class: str | None,
    device: str,
) -> torch.Tensor:
    train_ds = GenericMetaDataset(source_root, model.preprocess, model.transform, split="train")
    axes = []
    seen = set()
    with torch.no_grad():
        for idx in range(len(train_ds)):
            item = train_ds[idx]
            cls_name = item["cls_name"]
            if source_exclude_class is not None and cls_name == source_exclude_class:
                continue
            if cls_name in seen:
                continue
            image = item["img"].unsqueeze(0).to(device)
            _, _, text_features = model.clip_model.extract_feat(image, [cls_name])
            axes.append(decision_axis(text_features).detach().float().cpu())
            seen.add(cls_name)
            if len(seen) >= len(train_ds.cls_names) - (1 if source_exclude_class else 0):
                break
    if not axes:
        raise RuntimeError("Could not compute source global anomaly axis; no source class axes were collected.")
    axis = torch.stack(axes, dim=0).mean(dim=0)
    return F.normalize(axis, dim=0)


def train_layer_calibrators(
    fp_banks: dict[int, torch.Tensor],
    def_banks: dict[int, torch.Tensor],
    fp_scores: dict[int, torch.Tensor],
    def_scores: dict[int, torch.Tensor],
    selected_layer_indices: list[int],
    source_axis: torch.Tensor,
    args,
) -> dict[int, dict]:
    calibrators = {}
    for layer_idx in selected_layer_indices:
        if args.calib_train_score_space == "source_axis_logit":
            fp_score = 100.0 * (F.normalize(fp_banks[layer_idx].float(), dim=-1) @ source_axis.cpu().float())
            defect_score = 100.0 * (F.normalize(def_banks[layer_idx].float(), dim=-1) @ source_axis.cpu().float())
        else:
            fp_score = fp_scores.get(layer_idx, torch.empty(0))
            defect_score = def_scores.get(layer_idx, torch.empty(0))
        banks = {
            "fp": fp_banks[layer_idx],
            "defect": def_banks[layer_idx],
            "fp_score": fp_score,
            "defect_score": defect_score,
        }
        calibrator = train_subspace_host_residual_calibrator(
            banks=banks,
            axis=source_axis,
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
            subspace_basis_control=args.calib_subspace_basis_control,
            pair_label_control=args.calib_pair_label_control,
            preserve_host_margin_weight=args.calib_preserve_host_margin_weight,
            seed=args.seed + int(layer_idx),
            device=torch.device(args.device),
        )
        if calibrator is not None:
            calibrators[layer_idx] = calibrator
    return calibrators


def aggregate_results(results: dict):
    out = {}
    for key in ["sample_ids", "cls_names", "query_paths"]:
        out[key] = np.asarray(results[key], dtype=object)
    for key in ["gt_masks", "pr_masks", "gt_anomalys", "pr_anomalys"]:
        out[key] = torch.cat(results[key], dim=0)
    return out


def metadata_list(value) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return [value]


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
    parser = argparse.ArgumentParser("Official AdaCLIP baseline + parallel_margin test runner")
    parser.add_argument("--source_root", type=str, required=True)
    parser.add_argument("--source_dataset", type=str, required=True)
    parser.add_argument("--target_root", type=str, required=True)
    parser.add_argument("--target_dataset", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--skip_checkpoint_load", action="store_true")
    parser.add_argument("--backbone", type=str, default="ViT-L-14-336")
    parser.add_argument(
        "--checkpoint_load_mode",
        type=str,
        default="strict",
        choices=["strict", "compatible"],
        help="Use compatible for backbone-swap stress; only shape-compatible checkpoint tensors are loaded.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--prompting_depth", type=int, default=4)
    parser.add_argument("--prompting_length", type=int, default=5)
    parser.add_argument("--prompting_branch", type=str, default="VL")
    parser.add_argument("--prompting_type", type=str, default="SD")
    parser.add_argument("--use_hsf", type=lambda x: x.lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--k_clusters", type=int, default=20)
    parser.add_argument("--source_exclude_class", type=str, default=None)
    parser.add_argument("--hard_frac", type=float, default=0.01)
    parser.add_argument("--max_good_per_class", type=int, default=2)
    parser.add_argument("--max_defect_per_class", type=int, default=4)
    parser.add_argument("--max_bank_per_layer", type=int, default=512)
    parser.add_argument("--max_fp_per_image", type=int, default=64)
    parser.add_argument("--max_defect_per_image", type=int, default=64)
    parser.add_argument("--source_fp_mode", type=str, default="hardfp", choices=["hardfp", "random_normal", "easy_normal"])
    parser.add_argument("--bank_cache_path", type=str, default=None)
    parser.add_argument("--refresh_bank_cache", action="store_true")
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--fp_weight", type=float, default=1.0)
    parser.add_argument("--scoring_mode", type=str, default="replace", choices=["replace", "logit_add", "logit_gate", "logit_gate_posneg"])
    parser.add_argument("--refine_alpha", type=float, default=0.3)
    parser.add_argument("--refine_beta", type=float, default=0.15)
    parser.add_argument("--margin_norm_mode", type=str, default="spatial_tanh_zscore", choices=["none", "spatial_zscore", "spatial_tanh_zscore"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_metrics", type=str, nargs="+", default=["I-AUROC", "I-AP", "P-AUROC", "P-AP", "P-AUPRO"])
    parser.add_argument("--aupro_max_step", type=int, default=200)
    parser.add_argument("--aupro_expect_fpr", type=float, default=0.3)
    parser.add_argument("--layer_recipe", type=str, default="all4", choices=["all4", "last", "last2", "earlylate", "mid2"])
    parser.add_argument("--model_layer_recipe", type=str, default="all4", choices=["all4", "last", "last2", "earlylate", "mid2"])
    parser.add_argument("--output_layers", type=int, nargs="+", default=None)
    parser.add_argument("--enable_calibrator", action="store_true")
    parser.add_argument("--calib_epochs", type=int, default=0)
    parser.add_argument("--calib_lr", type=float, default=0.02)
    parser.add_argument("--calib_max_train_points", type=int, default=2048)
    parser.add_argument("--calib_rank_margin", type=float, default=0.05)
    parser.add_argument("--calib_rank_temp", type=float, default=0.05)
    parser.add_argument("--calib_eta_init", type=float, default=0.2)
    parser.add_argument("--calib_eta_max", type=float, default=1.0)
    parser.add_argument("--calib_eta_reg", type=float, default=0.001)
    parser.add_argument("--calib_fp_weight_init", type=float, default=1.0)
    parser.add_argument("--calib_learn_fp_weight", action="store_true")
    parser.add_argument("--calib_hardpair_frac", type=float, default=0.75)
    parser.add_argument("--calib_subspace_rank", type=int, default=8)
    parser.add_argument("--calib_subspace_basis_control", type=str, default="source", choices=["source", "random"])
    parser.add_argument("--calib_pair_label_control", type=str, default="correct", choices=["correct", "shuffle", "swap"])
    parser.add_argument("--calib_preserve_host_margin_weight", type=float, default=1.0)
    parser.add_argument("--calib_lambda", type=float, default=0.3)
    parser.add_argument("--calib_gate_quantile", type=float, default=0.9)
    parser.add_argument("--calib_gate_temp", type=float, default=0.25)
    parser.add_argument("--calib_gate_mode", type=str, default="host", choices=["host", "support_mass", "host_support"])
    parser.add_argument("--calib_residual_activation", type=str, default="signed", choices=["signed", "positive", "negative"])
    parser.add_argument("--calib_insertion_mode", type=str, default="map_residual", choices=["map_residual", "logit_residual", "prescore_rectify"])
    parser.add_argument("--calib_train_score_space", type=str, default="cache_prob", choices=["cache_prob", "source_axis_logit"])
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
    parser.add_argument("--ours_image_score_mode", type=str, default="official", choices=["official", "map_topk", "official_plus_map_topk", "max_official_map_topk"])
    parser.add_argument("--image_topk_frac", type=float, default=0.01)
    parser.add_argument("--report_group_separability", action="store_true")
    parser.add_argument("--separability_hard_frac", type=float, default=0.01)
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.cpu_threads)))
    try:
        torch.set_num_interop_threads(max(1, int(args.cpu_threads)))
    except RuntimeError:
        pass

    if args.source_exclude_class is not None and args.source_exclude_class.lower() in {"", "none", "null", "all"}:
        args.source_exclude_class = None
    if args.class_name is not None and args.class_name.lower() in {"", "none", "null", "all"}:
        args.class_name = None

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    model = build_model(
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        skip_checkpoint_load=args.skip_checkpoint_load,
        backbone=args.backbone,
        checkpoint_load_mode=args.checkpoint_load_mode,
        image_size=args.image_size,
        model_layer_recipe=args.model_layer_recipe,
        output_layers=args.output_layers,
        prompting_depth=args.prompting_depth,
        prompting_length=args.prompting_length,
        prompting_branch=args.prompting_branch,
        prompting_type=args.prompting_type,
        use_hsf=args.use_hsf,
        k_clusters=args.k_clusters,
    )

    features_list = list(model.clip_model.output_layers)
    selected_layer_indices = resolve_layer_recipe(features_list, args.layer_recipe)
    selected_layers = [features_list[i] for i in selected_layer_indices]

    fp_banks, def_banks, fp_scores, def_scores, bank_stats, bank_cache_path, bank_cache_meta = get_or_collect_source_patch_banks(
        model=model,
        skip_checkpoint_load=args.skip_checkpoint_load,
        seed=args.seed,
        source_root=args.source_root,
        source_dataset=args.source_dataset,
        image_size=args.image_size,
        features_list=features_list,
        device=args.device,
        source_exclude_class=args.source_exclude_class,
        hard_frac=args.hard_frac,
        max_good_per_class=args.max_good_per_class,
        max_defect_per_class=args.max_defect_per_class,
        max_bank_per_layer=args.max_bank_per_layer,
        max_fp_per_image=args.max_fp_per_image,
        max_defect_per_image=args.max_defect_per_image,
        sigma=args.sigma,
        checkpoint_path=args.checkpoint_path,
        bank_cache_path=args.bank_cache_path,
        refresh_bank_cache=args.refresh_bank_cache,
        require_host_scores=bool(args.enable_calibrator or (int(args.plugin_epochs) > 0 and args.plugin_feature_space == "hostscore")),
        source_fp_mode=args.source_fp_mode,
    )

    source_axis = None
    source_calibrators: dict[int, dict] = {}
    if args.enable_calibrator or (int(args.plugin_epochs) > 0 and args.plugin_feature_space == "axis"):
        source_axis = compute_source_global_axis(
            model=model,
            source_root=args.source_root,
            source_exclude_class=args.source_exclude_class,
            device=args.device,
        )
    source_plugins: dict[int, dict] = {}
    if int(args.plugin_epochs) > 0:
        for layer_idx in selected_layer_indices:
            if args.plugin_feature_space == "hostscore":
                plugin = train_hostscore_plugin_for_layer(
                    fp_scores=fp_scores[layer_idx],
                    def_scores=def_scores[layer_idx],
                    max_train_points=int(args.plugin_max_train_points),
                    epochs=int(args.plugin_epochs),
                    lr=float(args.plugin_lr),
                    hardpair_frac=float(args.plugin_hardpair_frac),
                    loss_type=str(args.plugin_loss),
                    hidden_dim=int(args.plugin_hidden_dim),
                    rank_margin=float(args.plugin_rank_margin),
                    rank_temp=float(args.plugin_rank_temp),
                    seed=int(args.seed) + 101 * int(layer_idx),
                    device=torch.device(args.device),
                )
            else:
                assert source_axis is not None
                plugin = train_scalar_plugin_for_layer(
                    fp_bank=fp_banks[layer_idx],
                    def_bank=def_banks[layer_idx],
                    axis=source_axis,
                    tau=float(args.tau),
                    max_train_points=int(args.plugin_max_train_points),
                    epochs=int(args.plugin_epochs),
                    lr=float(args.plugin_lr),
                    hardpair_frac=float(args.plugin_hardpair_frac),
                    loss_type=str(args.plugin_loss),
                    hidden_dim=int(args.plugin_hidden_dim),
                    rank_margin=float(args.plugin_rank_margin),
                    rank_temp=float(args.plugin_rank_temp),
                    seed=int(args.seed) + 101 * int(layer_idx),
                    device=torch.device(args.device),
                )
            if plugin is not None:
                source_plugins[layer_idx] = plugin
        print(
            json.dumps(
                {
                    "stage": "source_scalar_plugins_trained",
                    "num_layers": len(source_plugins),
                    "layers": sorted(source_plugins.keys()),
                    "source_pair_rank_accuracy": {
                        str(k): v.get("source_pair_rank_accuracy") for k, v in source_plugins.items()
                    },
                    "plugin_loss": args.plugin_loss,
                    "plugin_feature_space": args.plugin_feature_space,
                }
            ),
            flush=True,
        )
    if args.enable_calibrator:
        assert source_axis is not None
        source_calibrators = train_layer_calibrators(
            fp_banks=fp_banks,
            def_banks=def_banks,
            fp_scores=fp_scores,
            def_scores=def_scores,
            selected_layer_indices=selected_layer_indices,
            source_axis=source_axis,
            args=args,
        )
        print(
            json.dumps(
                {
                    "stage": "source_calibrators_trained",
                    "num_layers": len(source_calibrators),
                    "layers": sorted(source_calibrators.keys()),
                    "source_pair_rank_accuracy": {
                        str(k): v.get("source_pair_rank_accuracy") for k, v in source_calibrators.items()
                    },
                    "source_base_pair_rank_accuracy": {
                        str(k): v.get("source_base_pair_rank_accuracy") for k, v in source_calibrators.items()
                    },
                }
            ),
            flush=True,
        )

    dataset = build_target_dataset(args.target_root, args.target_dataset, model, training=False, class_name=args.class_name)
    loader_kwargs = {
        "batch_size": max(1, int(args.batch_size)),
        "shuffle": False,
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["pin_memory"] = args.device.startswith("cuda")
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    evaluator_metrics = [m for m in args.eval_metrics if not m.startswith("P-AUPRO")]
    evaluator = Evaluator("cpu", metrics=evaluator_metrics, sample_level=False) if evaluator_metrics else None
    fp_banks_gpu = {layer_idx: bank.to(args.device) for layer_idx, bank in fp_banks.items()}
    def_banks_gpu = {layer_idx: bank.to(args.device) for layer_idx, bank in def_banks.items()}

    results = {
        "baseline_txt": {"sample_ids": [], "gt_masks": [], "pr_masks": [], "cls_names": [], "gt_anomalys": [], "pr_anomalys": [], "query_paths": []},
        "parallel_margin": {"sample_ids": [], "gt_masks": [], "pr_masks": [], "cls_names": [], "gt_anomalys": [], "pr_anomalys": [], "query_paths": []},
    }
    if source_plugins:
        results["scalar_replace"] = {"sample_ids": [], "gt_masks": [], "pr_masks": [], "cls_names": [], "gt_anomalys": [], "pr_anomalys": [], "query_paths": []}
        results["scalar_residual"] = {"sample_ids": [], "gt_masks": [], "pr_masks": [], "cls_names": [], "gt_anomalys": [], "pr_anomalys": [], "query_paths": []}
    if args.enable_calibrator:
        results["ted_calibrated"] = {"sample_ids": [], "gt_masks": [], "pr_masks": [], "cls_names": [], "gt_anomalys": [], "pr_anomalys": [], "query_paths": []}

    with torch.no_grad():
        for idx, items in enumerate(loader, start=1):
            image = items["img"].to(args.device)
            cls_name = items["cls_name"][0]
            sample_id = f"{cls_name}_{idx:05d}"
            query_path = items["img_path"][0]
            gt_mask = items["img_mask"][:, 0].clone().to(args.device)
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            gt_mask = gt_mask.int()
            gt_anomaly = items["anomaly"].to(args.device).int()

            baseline = compute_baseline_outputs(
                model,
                image,
                cls_name,
                args.sigma,
                selected_layer_indices=selected_layer_indices,
            )
            axis = baseline["axis"]

            parallel_layer_maps = []
            refined_layer_logits = []
            calibrated_layer_maps = []
            calibrated_layer_logits = []
            scalar_plugin_maps = []
            for local_idx, patch_feature in enumerate(baseline["patch_tokens"]):
                layer_idx = baseline["selected_layer_indices"][local_idx]
                patch = patch_feature[0]
                fp_bank = fp_banks_gpu[layer_idx]
                def_bank = def_banks_gpu[layer_idx]
                fp_coeff = fp_bank @ axis if fp_bank.numel() else torch.empty((0,), device=args.device)
                def_coeff = def_bank @ axis if def_bank.numel() else torch.empty((0,), device=args.device)
                coeff = patch @ axis
                q_fp = logmeanexp_negative_sqdist_1d(coeff, fp_coeff, tau=args.tau)
                q_def = logmeanexp_negative_sqdist_1d(coeff, def_coeff, tau=args.tau)
                h = int(round(patch.shape[0] ** 0.5))
                if layer_idx in source_plugins:
                    plugin = source_plugins[layer_idx]
                    if plugin.get("feature_space") == "hostscore":
                        host_token_score = baseline["layer_token_scores"][local_idx].float()[0]
                        plugin_logit = apply_hostscore_plugin(host_token_score, plugin)
                    else:
                        plugin_logit = apply_scalar_plugin(coeff, q_def, q_fp, plugin)
                    plugin_score = spatial_tanh_zscore_token(plugin_logit.view(1, -1)).view(1, 1, h, h)
                    plugin_map = F.interpolate(
                        plugin_score,
                        size=(args.image_size, args.image_size),
                        mode="bilinear",
                        align_corners=True,
                    )[:, 0]
                    scalar_plugin_maps.append(plugin_map)
                parallel_map = (q_def - args.fp_weight * q_fp).reshape(1, 1, h, h)
                parallel_map = F.interpolate(
                    parallel_map,
                    size=(args.image_size, args.image_size),
                    mode="bilinear",
                    align_corners=True,
                )
                parallel_layer_maps.append(parallel_map)
                if args.scoring_mode != "replace":
                    refined_layer_logits.append(
                        refine_layer_logits(
                            layer_logits=baseline["layer_logits"][local_idx],
                            margin_map=parallel_map,
                            scoring_mode=args.scoring_mode,
                            refine_alpha=args.refine_alpha,
                            refine_beta=args.refine_beta,
                            margin_norm_mode=args.margin_norm_mode,
                        )
                    )
                if args.enable_calibrator and layer_idx in source_calibrators:
                    calibrator = source_calibrators[layer_idx]
                    host_token_score = baseline["layer_token_scores"][local_idx].float()
                    rect_tokens = prescore_subspace_transport_tokens(
                        tokens=patch_feature.float(),
                        q_def=q_def.view(1, -1),
                        q_fp=q_fp.view(1, -1),
                        basis=calibrator["basis"],
                        eta=float(calibrator["eta"]),
                        direction=calibrator["transport_direction"],
                        transport_b=float(calibrator["transport_b"]),
                        fp_weight=float(calibrator["fp_weight"]),
                    )
                    calibrated_token_score = subspace_host_residual_token_score(
                        host_score=host_token_score,
                        rect_tokens=rect_tokens,
                        basis=calibrator["basis"],
                        score_w=calibrator["subspace_score_w"],
                        readout_gamma=float(calibrator["readout_gamma"]),
                    )
                    token_residual = calibrated_token_score - host_token_score
                    if args.calib_insertion_mode == "prescore_rectify":
                        rect_tokens_for_host = prescore_subspace_transport_tokens(
                            tokens=patch_feature.float(),
                            q_def=q_def.view(1, -1),
                            q_fp=q_fp.view(1, -1),
                            basis=calibrator["basis"],
                            eta=float(calibrator["eta"]) * float(args.calib_lambda),
                            direction=calibrator["transport_direction"],
                            transport_b=float(calibrator["transport_b"]),
                            fp_weight=float(calibrator["fp_weight"]),
                        )
                        rect_token_logits = 100.0 * rect_tokens_for_host @ baseline["text_features"]
                        calibrated_layer_logits.append(token_logits_to_layer_logits(rect_token_logits, args.image_size))
                    else:
                        residual_map = token_residual.reshape(1, 1, h, h)
                        residual_map = F.interpolate(
                            residual_map,
                            size=(args.image_size, args.image_size),
                            mode="bilinear",
                            align_corners=True,
                        )[:, 0]
                        residual_map = normalize_margin_map(residual_map, "spatial_tanh_zscore")
                        residual_map = apply_residual_activation(residual_map, args.calib_residual_activation)
                        host_layer_map = baseline["layer_scalar_maps"][local_idx].float()
                        host_gate = host_gate_map(host_layer_map, args.calib_gate_quantile, args.calib_gate_temp)
                        support_gate = support_mass_gate_map(
                            q_def=q_def,
                            q_fp=q_fp,
                            grid_size=h,
                            image_size=args.image_size,
                            quantile=args.calib_gate_quantile,
                            temp=args.calib_gate_temp,
                        )
                        if args.calib_gate_mode == "host":
                            gate = host_gate
                        elif args.calib_gate_mode == "support_mass":
                            gate = support_gate
                        elif args.calib_gate_mode == "host_support":
                            gate = host_gate * support_gate
                        else:
                            raise ValueError(f"Unsupported gate mode: {args.calib_gate_mode}")
                        delta = float(args.calib_lambda) * gate * residual_map
                        if args.calib_insertion_mode == "logit_residual":
                            calibrated_layer_logits.append(add_to_logit_difference(baseline["layer_logits"][local_idx], delta))
                        elif args.calib_insertion_mode == "map_residual":
                            calibrated_layer_maps.append(torch.clamp(host_layer_map + delta, 0.0, 1.0))
                        else:
                            raise ValueError(f"Unsupported calibrated insertion mode: {args.calib_insertion_mode}")

            if args.scoring_mode == "replace":
                parallel_map = torch.stack(parallel_layer_maps, dim=0).mean(dim=0)[:, 0]
            else:
                aggregated_refined_logits = torch.stack(refined_layer_logits, dim=0).mean(dim=0)
                aggregated_refined_probs = torch.softmax(aggregated_refined_logits, dim=1)
                parallel_map = layer_map_to_scalar_map(aggregated_refined_probs)[:, 0]
            parallel_map = smooth_map(parallel_map, sigma=args.sigma).to(args.device)
            parallel_img_score = select_image_score(
                baseline["baseline_img_score"],
                parallel_map,
                args.ours_image_score_mode,
                args.image_topk_frac,
            )
            calibrated_map = None
            calibrated_img_score = None
            if args.enable_calibrator and calibrated_layer_logits:
                aggregated_calibrated_logits = torch.stack(calibrated_layer_logits, dim=0).mean(dim=0)
                calibrated_map = logits_to_scalar_map(aggregated_calibrated_logits)
                calibrated_map = smooth_map(calibrated_map, sigma=args.sigma).to(args.device)
            elif args.enable_calibrator and calibrated_layer_maps:
                calibrated_map = torch.stack(calibrated_layer_maps, dim=0).mean(dim=0)
                calibrated_map = smooth_map(calibrated_map, sigma=args.sigma).to(args.device)
            if calibrated_map is not None:
                calibrated_img_score = select_image_score(
                    baseline["baseline_img_score"],
                    calibrated_map,
                    args.ours_image_score_mode,
                    args.image_topk_frac,
                )
            scalar_replace_map = None
            scalar_residual_map = None
            scalar_replace_img_score = None
            scalar_residual_img_score = None
            if scalar_plugin_maps:
                scalar_plugin_map = torch.stack(scalar_plugin_maps, dim=0).mean(dim=0)
                scalar_plugin_map = smooth_map(scalar_plugin_map, sigma=args.sigma).to(args.device)
                scalar_replace_map = torch.clamp((scalar_plugin_map + 1.0) * 0.5, 0.0, 1.0)
                gate = host_gate_map(baseline["baseline_map"].float(), args.calib_gate_quantile, args.calib_gate_temp)
                scalar_residual_map = torch.clamp(
                    baseline["baseline_map"].float() + float(args.plugin_alpha) * gate * scalar_plugin_map,
                    0.0,
                    1.0,
                )
                scalar_replace_img_score = select_image_score(
                    baseline["baseline_img_score"],
                    scalar_replace_map,
                    args.ours_image_score_mode,
                    args.image_topk_frac,
                )
                scalar_residual_img_score = select_image_score(
                    baseline["baseline_img_score"],
                    scalar_residual_map,
                    args.ours_image_score_mode,
                    args.image_topk_frac,
                )

            outputs_to_append = [
                ("baseline_txt", baseline["baseline_map"], baseline["baseline_img_score"]),
                ("parallel_margin", parallel_map, parallel_img_score),
            ]
            if scalar_replace_map is not None:
                outputs_to_append.append(("scalar_replace", scalar_replace_map, scalar_replace_img_score))
            if scalar_residual_map is not None:
                outputs_to_append.append(("scalar_residual", scalar_residual_map, scalar_residual_img_score))
            if calibrated_map is not None:
                outputs_to_append.append(("ted_calibrated", calibrated_map, calibrated_img_score))

            for mode, map_tensor, image_score in outputs_to_append:
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
    class_order = dataset.cls_names
    mode_order = ["baseline_txt", "parallel_margin"]
    if source_plugins:
        mode_order.extend(["scalar_replace", "scalar_residual"])
    if args.enable_calibrator:
        mode_order.append("ted_calibrated")
    for mode in mode_order:
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
                "layer_recipe": args.layer_recipe,
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
                "layer_recipe": args.layer_recipe,
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
        "skip_checkpoint_load": args.skip_checkpoint_load,
        "backbone": args.backbone,
        "checkpoint_load_mode": args.checkpoint_load_mode,
        "checkpoint_load_info": getattr(model, "_ted_checkpoint_load_info", None),
        "seed": args.seed,
        "image_size": args.image_size,
        "features_list": features_list,
        "model_layer_recipe": args.model_layer_recipe,
        "model_output_layers": args.output_layers,
        "layer_recipe": args.layer_recipe,
        "selected_layer_indices": selected_layer_indices,
        "selected_layers": selected_layers,
        "sigma": args.sigma,
        "tau": args.tau,
        "fp_weight": args.fp_weight,
        "scoring_mode": args.scoring_mode,
        "refine_alpha": args.refine_alpha,
        "refine_beta": args.refine_beta,
        "margin_norm_mode": args.margin_norm_mode,
        "enable_calibrator": args.enable_calibrator,
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
        "calib_subspace_basis_control": args.calib_subspace_basis_control,
        "calib_pair_label_control": args.calib_pair_label_control,
        "calib_preserve_host_margin_weight": args.calib_preserve_host_margin_weight,
        "calib_lambda": args.calib_lambda,
        "calib_gate_quantile": args.calib_gate_quantile,
        "calib_gate_temp": args.calib_gate_temp,
        "calib_gate_mode": args.calib_gate_mode,
        "calib_residual_activation": args.calib_residual_activation,
        "calib_insertion_mode": args.calib_insertion_mode,
        "calib_train_score_space": args.calib_train_score_space,
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
        "ours_image_score_mode": args.ours_image_score_mode,
        "image_topk_frac": args.image_topk_frac,
        "report_group_separability": args.report_group_separability,
        "separability_hard_frac": args.separability_hard_frac,
        "num_workers": args.num_workers,
        "eval_metrics": args.eval_metrics,
        "aupro_max_step": args.aupro_max_step,
        "aupro_expect_fpr": args.aupro_expect_fpr,
        "aupro_backend": "fast_threshold_cpu",
        "bank_stats": bank_stats,
        "bank_cache_path": str(bank_cache_path),
        "bank_cache_meta": bank_cache_meta,
        "source_calibrators": {
            str(k): {
                kk: vv
                for kk, vv in v.items()
                if kk not in {"basis"}
            }
            for k, v in source_calibrators.items()
        },
        "source_scalar_plugins": {
            str(k): {
                kk: vv
                for kk, vv in v.items()
                if kk not in {"weight", "bias", "mean", "std"}
            }
            for k, v in source_plugins.items()
        },
        "score_definition": {
            "baseline_txt": "official AdaCLIP zero-shot baseline",
            "parallel_margin": "official AdaCLIP image branch preserved; local anomaly map replaced by multilayer text-conditioned competitive margin",
            "scalar_replace": "source-only logistic readout on text-axis support features, used directly as the local map",
            "scalar_residual": "source-only logistic readout on text-axis support features, inserted as a host-map residual",
            "ted_calibrated": f"source-only host-residual subspace calibrator on AdaCLIP layer features; insertion={args.calib_insertion_mode}; official image score preserved",
        },
    }
    if args.report_group_separability:
        summary["separability_mean"] = separability_mean
    save_results(rows, summary, Path(args.save_dir))
    print(json.dumps(jsonable({"summary": summary, "rows": rows}), indent=2))


if __name__ == "__main__":
    main()
