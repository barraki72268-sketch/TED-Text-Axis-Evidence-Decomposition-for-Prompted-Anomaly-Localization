#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import measure
from sklearn.metrics import auc, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = ROOT / "neurips2026" / "scripts"
FAP_ROOT = ROOT / "neurips2026" / "FAPrompt"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(FAP_ROOT) not in sys.path:
    sys.path.insert(0, str(FAP_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("FAPROMPT_CACHE_DIR", "/mnt/data/pilab-kingjinyoung/ADPretrain/FAPrompt/.cache/clip")

import AnomalyCLIP_lib  # noqa: E402
from common_failure_metrics import (  # noqa: E402
    BTAD_ROOT,
    MPDD_ROOT,
    MVTEC_ROOT,
    OFFICIAL_VISA_ROOT,
    validate_visa_root,
)
from FAPrompt import FAPrompt  # noqa: E402
from utils import get_transform  # noqa: E402


BANK_PROTOCOL_VERSION = "faprompt_v4_branchscore"

PRESET_CONFIGS = {
    "visa2mvtec": {
        "source_dataset": "visa",
        "target_dataset": "mvtecad",
        "source_root": str(OFFICIAL_VISA_ROOT),
        "target_root": str(MVTEC_ROOT),
        "checkpoint_path": str(FAP_ROOT / "checkpoints" / "trained_on_visa" / "epoch_15.pth"),
    },
    "visa2mpdd": {
        "source_dataset": "visa",
        "target_dataset": "mpdd",
        "source_root": str(OFFICIAL_VISA_ROOT),
        "target_root": str(MPDD_ROOT),
        "checkpoint_path": str(FAP_ROOT / "checkpoints" / "trained_on_visa" / "epoch_15.pth"),
    },
    "visa2btad": {
        "source_dataset": "visa",
        "target_dataset": "btad",
        "source_root": str(OFFICIAL_VISA_ROOT),
        "target_root": str(BTAD_ROOT),
        "checkpoint_path": str(FAP_ROOT / "checkpoints" / "trained_on_visa" / "epoch_15.pth"),
    },
    "mvtec2visa": {
        "source_dataset": "mvtecad",
        "target_dataset": "visa",
        "source_root": str(MVTEC_ROOT),
        "target_root": str(OFFICIAL_VISA_ROOT),
        "checkpoint_path": str(FAP_ROOT / "checkpoints" / "trained_on_mvtecad" / "epoch_15.pth"),
    },
    "mvtec2mpdd": {
        "source_dataset": "mvtecad",
        "target_dataset": "mpdd",
        "source_root": str(MVTEC_ROOT),
        "target_root": str(MPDD_ROOT),
        "checkpoint_path": str(FAP_ROOT / "checkpoints" / "trained_on_mvtecad" / "epoch_15.pth"),
    },
    "mvtec2btad": {
        "source_dataset": "mvtecad",
        "target_dataset": "btad",
        "source_root": str(MVTEC_ROOT),
        "target_root": str(BTAD_ROOT),
        "checkpoint_path": str(FAP_ROOT / "checkpoints" / "trained_on_mvtecad" / "epoch_15.pth"),
    },
}


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resize_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape == target.shape:
        return source.to(dtype=target.dtype)
    out = target.detach().clone()
    if not (torch.is_floating_point(source) and torch.is_floating_point(target)):
        return out
    slices = tuple(slice(0, min(s, t)) for s, t in zip(source.shape, target.shape))
    out[slices] = source[slices].to(dtype=target.dtype)
    return out


def load_prompt_state(
    prompt_learner: FAPrompt,
    checkpoint_path: str,
    mode: str,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint["prompt_learner"]
    if mode == "strict":
        prompt_learner.load_state_dict(source_state, strict=True)
        return {"mode": mode, "adapted": [], "skipped": []}
    if mode != "project_crop":
        raise ValueError(f"Unknown prompt_load_mode: {mode}")

    target_state = prompt_learner.state_dict()
    merged = {}
    adapted = []
    skipped = []
    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            merged[key] = target_value
            skipped.append(key)
            continue
        if source_value.shape == target_value.shape:
            merged[key] = source_value.to(dtype=target_value.dtype) if torch.is_floating_point(target_value) else source_value
            continue
        # Token prefix/suffix buffers are actual backbone token embeddings; keep the new backbone values.
        if key.startswith("token_prefix") or key.startswith("token_suffix"):
            merged[key] = target_value
            skipped.append(key)
            continue
        merged[key] = resize_like(source_value, target_value)
        adapted.append({"key": key, "source_shape": list(source_value.shape), "target_shape": list(target_value.shape)})
    prompt_learner.load_state_dict(merged, strict=True)
    return {"mode": mode, "adapted": adapted, "skipped": skipped}


class ArgsProxy:
    pass


class MetaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        image_size: int,
        split: str,
        class_name: str | None = None,
        normal_only: bool = False,
        anomaly_only: bool = False,
    ):
        self.root = Path(root)
        args = ArgsProxy()
        args.image_size = image_size
        self.image_transform, self.mask_transform = get_transform(args)
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        split_meta = meta[split]
        if class_name is not None:
            split_meta = {class_name: split_meta[class_name]}
        self.class_names = sorted(split_meta.keys())
        self.items: list[dict] = []
        for cls_name in self.class_names:
            for item in split_meta[cls_name]:
                anomaly = int(item["anomaly"])
                if normal_only and anomaly != 0:
                    continue
                if anomaly_only and anomaly == 0:
                    continue
                self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        img = Image.open(self.root / item["img_path"]).convert("RGB")
        orig_w, orig_h = img.size
        image = self.image_transform(img)
        anomaly = int(item["anomaly"])
        mask_path = item.get("mask_path")
        if anomaly == 0 or not mask_path or os.path.isdir(self.root / str(mask_path)):
            mask = Image.fromarray(np.zeros((orig_h, orig_w), dtype=np.uint8), mode="L")
        else:
            raw = np.array(Image.open(self.root / str(mask_path)).convert("L")) > 0
            mask = Image.fromarray(raw.astype(np.uint8) * 255, mode="L")
        mask_t = self.mask_transform(mask)
        return {
            "image": image,
            "mask": mask_t,
            "label": torch.tensor(anomaly, dtype=torch.long),
            "class_name": item["cls_name"],
            "img_path": item["img_path"],
        }


def build_model(
    device: torch.device,
    checkpoint_path: str,
    model_name: str,
    prompt_load_mode: str,
    dpam_layer: int,
    depth: int,
    n_ctx: int,
    t_n_ctx: int,
) -> tuple[torch.nn.Module, FAPrompt, dict]:
    params = {
        "Prompt_length": n_ctx,
        "learnabel_text_embedding_depth": depth,
        "learnabel_text_embedding_length": t_n_ctx,
    }
    model, _ = AnomalyCLIP_lib.load(model_name, device=device, design_details=params)
    model.eval()
    max_visual_layers = int(model.visual.transformer.layers)
    effective_dpam_layer = min(dpam_layer, max_visual_layers)
    model.visual.DAPM_replace(DPAM_layer=effective_dpam_layer)
    prompt_learner = FAPrompt(model.to("cpu"), params)
    prompt_load_info = load_prompt_state(prompt_learner, checkpoint_path, prompt_load_mode)
    prompt_load_info["requested_dpam_layer"] = int(dpam_layer)
    prompt_load_info["effective_dpam_layer"] = int(effective_dpam_layer)
    prompt_load_info["max_visual_layers"] = int(max_visual_layers)
    prompt_learner.to(device)
    prompt_learner.eval()
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in prompt_learner.parameters():
        p.requires_grad = False
    return model, prompt_learner, prompt_load_info


def learned_text_pair(model, prompt_learner) -> torch.Tensor:
    prompts_pos, prompts_neg, tok_pos, tok_neg, compound_prompts_text, _ = prompt_learner.forward()
    text_pos = model.encode_text_learn(prompts_pos, tok_pos, compound_prompts_text).float()
    text_neg = model.encode_text_learn(prompts_neg, tok_neg, compound_prompts_text).float()
    text_neg = torch.mean(text_neg, dim=0, keepdim=True)
    text_features = torch.cat([text_pos, text_neg])
    text_features = torch.stack(torch.chunk(text_features, chunks=2, dim=0), dim=1)
    return F.normalize(text_features, dim=-1)[0]


def compute_faprompt_outputs(
    model,
    prompt_learner,
    image: torch.Tensor,
    features_list: list[int],
    image_size: int,
    sigma: int,
    dap_token_mode: str,
    dpam_layer: int,
    base_text_pair: torch.Tensor | None = None,
    rectifier: dict | None = None,
) -> dict:
    image_features, patch_features = model.encode_image(image, features_list, DPAM_layer=dpam_layer)
    image_features = F.normalize(image_features, dim=-1)
    patch_features = F.normalize(patch_features, dim=-1)
    text_pair = base_text_pair if base_text_pair is not None else learned_text_pair(model, prompt_learner)
    if rectifier is not None:
        rectifier = dict(rectifier)
        rectifier.setdefault("axis", F.normalize(text_pair[1] - text_pair[0], dim=0))
        patch_features = rectify_faprompt_patch_features(patch_features, **rectifier)

    similarity1, _ = AnomalyCLIP_lib.compute_similarity_ori(patch_features, text_pair)
    similarity_map1 = AnomalyCLIP_lib.get_similarity_map(similarity1[1:, :], image_size)
    map_max_score1 = similarity1[1:, :, 1].max(dim=0).values

    num_top = min(10, similarity1[1:, :, 1].shape[0])
    pk_idx = torch.topk(similarity1[1:, :, 1], k=num_top, dim=0, largest=True, sorted=True).indices
    pk_idx = pk_idx.permute(1, 0)
    patch_by_batch = patch_features[1:, :, :].permute(1, 0, 2)
    if dap_token_mode == "topk":
        selected_tokens = []
        for bi in range(pk_idx.shape[0]):
            selected_tokens.append(patch_by_batch[bi, pk_idx[bi], :])
        selected_tokens_t = torch.stack(selected_tokens, dim=0)
    elif dap_token_mode == "official_firstk":
        # Match FAPrompt/test.py exactly: it computes top-k indices but indexes by k.
        selected_tokens_t = patch_by_batch[:, :num_top, :]
    else:
        raise ValueError(f"Unknown dap_token_mode: {dap_token_mode}")

    text_features_list = []
    for bi in range(selected_tokens_t.shape[0]):
        p_pos, p_neg, tok_pos, tok_neg, comp_text, _ = prompt_learner.forward(selected_tokens=selected_tokens_t[bi])
        tf_pos = model.encode_text_learn(p_pos, tok_pos, comp_text).float()
        tf_neg = model.encode_text_learn(p_neg, tok_neg, comp_text).float()
        tf_neg = torch.mean(tf_neg, dim=0, keepdim=True)
        text_features_list.append(torch.cat([tf_pos, tf_neg]))
    text_features = F.normalize(torch.stack(text_features_list, dim=0), dim=-1)

    text_probs = image_features @ text_features.permute(0, 2, 1)
    text_probs = (text_probs / 0.07).softmax(-1)[:, 0, 1]

    similarity2, _ = AnomalyCLIP_lib.compute_similarity(patch_features, text_features)
    similarity_map2 = AnomalyCLIP_lib.get_similarity_map(similarity2[1:, :], image_size)
    map_max_score2 = similarity2[1:, :, 1].max(dim=0).values

    anomaly_map = (
        similarity_map1[:, 1, :] + 1.0 - similarity_map1[:, 0, :] + similarity_map2[:, 1, :] + 1.0 - similarity_map2[:, 0, :]
    ) / 4.0
    if sigma > 0:
        anomaly_map_np = gaussian_filter(anomaly_map.detach().cpu().numpy(), sigma=sigma)
    else:
        anomaly_map_np = anomaly_map.detach().cpu().numpy()

    token_score = 0.5 * (similarity1[1:, :, 1] + similarity2[1:, :, 1])
    token_score = token_score.permute(1, 0).contiguous()
    branch1_token_score = similarity1[1:, :, 1].permute(1, 0).contiguous()
    branch2_token_score = similarity2[1:, :, 1].permute(1, 0).contiguous()
    map_max_score = (2.0 * map_max_score1 + map_max_score2) / 3.0
    official_image_score = 0.5 * (text_probs + map_max_score)
    return {
        "patch_tokens": patch_by_batch.contiguous(),
        "token_score": token_score,
        "branch1_token_score": branch1_token_score,
        "branch2_token_score": branch2_token_score,
        "anomaly_map": anomaly_map_np,
        "official_image_score": official_image_score.detach().cpu().numpy(),
    }


def soft_bank_transport(
    query: torch.Tensor,
    defect_bank: torch.Tensor,
    fp_bank: torch.Tensor,
    tau: float,
    fp_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_shape = query.shape
    query_flat = F.normalize(query.reshape(-1, query.shape[-1]).float(), dim=-1)
    defect_bank = F.normalize(defect_bank.float().to(query.device), dim=-1)
    fp_bank = F.normalize(fp_bank.float().to(query.device), dim=-1)
    sim_def = query_flat @ defect_bank.t()
    sim_fp = query_flat @ fp_bank.t()
    w_def = torch.softmax(sim_def / max(tau, 1e-6), dim=-1)
    w_fp = torch.softmax(sim_fp / max(tau, 1e-6), dim=-1)
    proto_def = w_def @ defect_bank
    proto_fp = w_fp @ fp_bank
    support_def = torch.logsumexp(sim_def / max(tau, 1e-6), dim=-1) - math.log(max(defect_bank.shape[0], 1))
    support_fp = torch.logsumexp(sim_fp / max(tau, 1e-6), dim=-1) - math.log(max(fp_bank.shape[0], 1))
    direction = proto_def - fp_weight * proto_fp
    return (
        direction.reshape(*query_shape),
        support_def.reshape(query_shape[:-1]),
        support_fp.reshape(query_shape[:-1]),
        (proto_def - proto_fp).reshape(*query_shape),
    )


def support_mass_gate(q_def: torch.Tensor, q_fp: torch.Tensor, sharpness: float) -> torch.Tensor:
    mass = q_def + q_fp
    mass_z = spatial_tanh_zscore(mass)
    return torch.sigmoid(float(sharpness) * mass_z)


def rectify_faprompt_patch_features(
    patch_features: torch.Tensor,
    defect_bank: torch.Tensor,
    fp_bank: torch.Tensor,
    strength: float,
    tau: float,
    fp_weight: float,
    gate_sharpness: float,
    gate_mode: str,
    direction_mode: str = "transport",
    axis: torch.Tensor | None = None,
    calibrator: dict | None = None,
    bank_chunk: int = 2048,
) -> torch.Tensor:
    cls_or_global = patch_features[:1]
    tokens = patch_features[1:].permute(1, 0, 2).contiguous()
    if direction_mode == "learned_subspace":
        if axis is None or calibrator is None:
            raise ValueError("learned_subspace rectification requires axis and calibrator")
        axis = F.normalize(axis.to(tokens.device).float(), dim=0)
        fp_proj = F.normalize(fp_bank.float().to(tokens.device), dim=-1) @ axis
        defect_proj = F.normalize(defect_bank.float().to(tokens.device), dim=-1) @ axis
        query_proj = tokens @ axis
        q_def = logmeanexp_negative_sqdist_1d(query_proj, defect_proj, tau=tau, chunk=bank_chunk)
        q_fp = logmeanexp_negative_sqdist_1d(query_proj, fp_proj, tau=tau, chunk=bank_chunk)
        rectified = prescore_subspace_transport_tokens(
            tokens=tokens,
            q_def=q_def,
            q_fp=q_fp,
            basis=calibrator["basis"],
            eta=float(calibrator["eta"]) * float(strength),
            direction=calibrator["transport_direction"],
            transport_b=float(calibrator["transport_b"]),
            fp_weight=float(calibrator["fp_weight"]),
        )
        return torch.cat([cls_or_global, rectified.permute(1, 0, 2).contiguous()], dim=0)

    direction, q_def, q_fp, _ = soft_bank_transport(
        tokens,
        defect_bank=defect_bank,
        fp_bank=fp_bank,
        tau=tau,
        fp_weight=fp_weight,
    )
    if gate_mode == "none":
        gate = torch.ones_like(q_def)
    elif gate_mode == "support_mass":
        gate = support_mass_gate(q_def, q_fp, gate_sharpness)
    elif gate_mode == "hardfp":
        gate = support_mass_gate(q_fp, q_def, gate_sharpness)
    else:
        raise ValueError(f"Unknown rectifier gate_mode: {gate_mode}")
    if direction_mode == "transport":
        step = gate.unsqueeze(-1) * direction
    elif direction_mode == "host_axis":
        if axis is None:
            raise ValueError("host_axis rectification requires axis")
        axis = F.normalize(axis.to(tokens.device).float(), dim=0)
        signed_margin = spatial_tanh_zscore(q_def - fp_weight * q_fp)
        step = gate.unsqueeze(-1) * signed_margin.unsqueeze(-1) * axis.view(1, 1, -1)
    else:
        raise ValueError(f"Unknown rectifier direction_mode: {direction_mode}")
    rectified = F.normalize(tokens + float(strength) * step, dim=-1)
    return torch.cat([cls_or_global, rectified.permute(1, 0, 2).contiguous()], dim=0)


def downsample_mask_to_patch(mask: torch.Tensor, side: int) -> torch.Tensor:
    ds = F.interpolate(mask.float(), size=(side, side), mode="nearest")
    return ds[:, 0].reshape(mask.shape[0], -1) > 0.5


def farthest_point_subsample(feat: torch.Tensor, max_points: int, seed: int, pre_pool_factor: int = 6) -> torch.Tensor:
    if feat.shape[0] <= max_points:
        return F.normalize(feat.float(), dim=-1)
    feat = F.normalize(feat.float(), dim=-1)
    pool_limit = max(max_points, max_points * pre_pool_factor)
    if feat.shape[0] > pool_limit:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        perm = torch.randperm(feat.shape[0], generator=gen)[:pool_limit]
        feat = feat[perm]
    mean_dir = F.normalize(feat.mean(dim=0, keepdim=True), dim=-1)
    min_dist = 1.0 - (feat @ mean_dir.t()).squeeze(1)
    selected = []
    for _ in range(min(max_points, feat.shape[0])):
        idx = int(torch.argmax(min_dist).item())
        selected.append(idx)
        cand = 1.0 - (feat @ feat[idx : idx + 1].t()).squeeze(1)
        min_dist = torch.minimum(min_dist, cand)
        min_dist[idx] = -1.0
    return feat[torch.tensor(selected, dtype=torch.long)]


def farthest_point_subsample_indices(feat: torch.Tensor, max_points: int, seed: int, pre_pool_factor: int = 6) -> torch.Tensor:
    if feat.shape[0] <= max_points:
        return torch.arange(feat.shape[0], dtype=torch.long)
    feat = F.normalize(feat.float(), dim=-1)
    pool_limit = max(max_points, max_points * pre_pool_factor)
    base_idx = torch.arange(feat.shape[0], dtype=torch.long)
    if feat.shape[0] > pool_limit:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        pool_idx = torch.randperm(feat.shape[0], generator=gen)[:pool_limit]
        feat = feat[pool_idx]
        base_idx = base_idx[pool_idx]
    mean_dir = F.normalize(feat.mean(dim=0, keepdim=True), dim=-1)
    min_dist = 1.0 - (feat @ mean_dir.t()).squeeze(1)
    selected = []
    for _ in range(min(max_points, feat.shape[0])):
        idx = int(torch.argmax(min_dist).item())
        selected.append(idx)
        cand = 1.0 - (feat @ feat[idx : idx + 1].t()).squeeze(1)
        min_dist = torch.minimum(min_dist, cand)
        min_dist[idx] = -1.0
    return base_idx[torch.tensor(selected, dtype=torch.long)]


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def get_bank_cache_path(
    preset: str,
    source_root: str,
    checkpoint_path: str,
    model_name: str,
    prompt_load_mode: str,
    cache_dir: Path,
    max_fp_per_class: int,
    max_defect_per_class: int,
    max_fp_total: int,
    max_defect_total: int,
    normal_topk_frac: float,
    features_list: list[int],
    image_size: int,
    dap_token_mode: str,
    dpam_layer: int,
    seed: int,
    source_fp_mode: str = "hardfp",
) -> Path:
    ckpt_tag = Path(checkpoint_path).parent.name + "_" + Path(checkpoint_path).stem
    model_tag = model_name.replace("/", "-").replace("@", "-").replace(" ", "")
    root_tag = Path(source_root).name + "_" + stable_hash(str(Path(source_root).resolve()))
    frac_tag = str(normal_topk_frac).replace(".", "p")
    feat_tag = "f" + "-".join(str(x) for x in features_list)
    short_mode = {"random_normal": "rn", "easy_normal": "en"}.get(source_fp_mode, source_fp_mode)
    mode_tag = "" if source_fp_mode == "hardfp" else f"_fp{short_mode}"
    del preset
    return cache_dir / (
        f"faprompt_sourcebank_{root_tag}_{ckpt_tag}_{BANK_PROTOCOL_VERSION}_"
        f"{model_tag}_{prompt_load_mode}_"
        f"{feat_tag}_img{image_size}_"
        f"dap{dap_token_mode}_dpam{dpam_layer}_"
        f"fpc{max_fp_per_class}_dfc{max_defect_per_class}_fpt{max_fp_total}_dft{max_defect_total}_"
        f"topk{frac_tag}_seed{seed}{mode_tag}.pt"
    )


def get_legacy_bank_cache_path(
    source_root: str,
    checkpoint_path: str,
    cache_dir: Path,
    max_fp_per_class: int,
    max_defect_per_class: int,
    max_fp_total: int,
    max_defect_total: int,
    normal_topk_frac: float,
    seed: int,
) -> Path:
    ckpt_tag = Path(checkpoint_path).parent.name + "_" + Path(checkpoint_path).stem
    root_tag = Path(source_root).name + "_" + stable_hash(str(Path(source_root).resolve()))
    frac_tag = str(normal_topk_frac).replace(".", "p")
    return cache_dir / (
        f"faprompt_sourcebank_{root_tag}_{ckpt_tag}_{BANK_PROTOCOL_VERSION}_"
        f"fpc{max_fp_per_class}_dfc{max_defect_per_class}_fpt{max_fp_total}_dft{max_defect_total}_"
        f"topk{frac_tag}_seed{seed}.pt"
    )


def collect_source_banks(
    model,
    prompt_learner,
    source_root: str,
    model_name: str,
    prompt_load_mode: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    features_list: list[int],
    sigma: int,
    dap_token_mode: str,
    dpam_layer: int,
    max_fp_per_class: int,
    max_defect_per_class: int,
    max_fp_total: int,
    max_defect_total: int,
    normal_topk_frac: float,
    seed: int,
    source_fp_mode: str = "hardfp",
) -> dict:
    meta = json.loads((Path(source_root) / "meta.json").read_text(encoding="utf-8"))
    class_names = sorted(meta["train"].keys())
    base_text_pair = learned_text_pair(model, prompt_learner)
    axis = F.normalize(base_text_pair[1] - base_text_pair[0], dim=0)
    feature_dim = int(axis.numel())
    fp_by_class: list[torch.Tensor] = []
    defect_by_class: list[torch.Tensor] = []
    fp_score_by_class: list[torch.Tensor] = []
    defect_score_by_class: list[torch.Tensor] = []
    fp_branch1_score_by_class: list[torch.Tensor] = []
    defect_branch1_score_by_class: list[torch.Tensor] = []
    fp_branch2_score_by_class: list[torch.Tensor] = []
    defect_branch2_score_by_class: list[torch.Tensor] = []

    for ci, class_name in enumerate(class_names):
        normal_ds = MetaDataset(source_root, image_size, split="train", class_name=class_name, normal_only=True)
        anomaly_ds = MetaDataset(source_root, image_size, split="test", class_name=class_name, anomaly_only=True)
        normal_dl = DataLoader(normal_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        anomaly_dl = DataLoader(anomaly_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        fp_parts: list[torch.Tensor] = []
        defect_parts: list[torch.Tensor] = []
        fp_score_parts: list[torch.Tensor] = []
        defect_score_parts: list[torch.Tensor] = []
        fp_branch1_score_parts: list[torch.Tensor] = []
        defect_branch1_score_parts: list[torch.Tensor] = []
        fp_branch2_score_parts: list[torch.Tensor] = []
        defect_branch2_score_parts: list[torch.Tensor] = []

        for batch in normal_dl:
            image = batch["image"].to(device)
            out = compute_faprompt_outputs(model, prompt_learner, image, features_list, image_size, sigma, dap_token_mode, dpam_layer, base_text_pair)
            tokens = out["patch_tokens"]
            scores = out["token_score"]
            branch1_scores = out["branch1_token_score"]
            branch2_scores = out["branch2_token_score"]
            for bi in range(tokens.shape[0]):
                k = max(1, int(math.ceil(tokens.shape[1] * normal_topk_frac)))
                if source_fp_mode == "hardfp":
                    top_idx = torch.topk(scores[bi], k=k, largest=True).indices
                elif source_fp_mode == "easy_normal":
                    top_idx = torch.topk(scores[bi], k=k, largest=False).indices
                elif source_fp_mode == "random_normal":
                    top_idx = torch.randperm(tokens.shape[1], device=scores.device)[:k]
                else:
                    raise ValueError(f"Unsupported source_fp_mode: {source_fp_mode}")
                fp_parts.append(tokens[bi, top_idx].detach().cpu())
                fp_score_parts.append(scores[bi, top_idx].detach().cpu())
                fp_branch1_score_parts.append(branch1_scores[bi, top_idx].detach().cpu())
                fp_branch2_score_parts.append(branch2_scores[bi, top_idx].detach().cpu())

        for batch in anomaly_dl:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            out = compute_faprompt_outputs(model, prompt_learner, image, features_list, image_size, sigma, dap_token_mode, dpam_layer, base_text_pair)
            tokens = out["patch_tokens"]
            scores = out["token_score"]
            branch1_scores = out["branch1_token_score"]
            branch2_scores = out["branch2_token_score"]
            side = int(round(math.sqrt(tokens.shape[1])))
            patch_mask = downsample_mask_to_patch(mask, side)
            for bi in range(tokens.shape[0]):
                pos_idx = torch.nonzero(patch_mask[bi], as_tuple=False).flatten()
                if pos_idx.numel() == 0:
                    continue
                pos_scores = scores[bi, pos_idx]
                k = min(max_defect_per_class, pos_idx.numel())
                top_pos = torch.topk(pos_scores, k=k, largest=True).indices
                defect_parts.append(tokens[bi, pos_idx[top_pos]].detach().cpu())
                defect_score_parts.append(scores[bi, pos_idx[top_pos]].detach().cpu())
                defect_branch1_score_parts.append(branch1_scores[bi, pos_idx[top_pos]].detach().cpu())
                defect_branch2_score_parts.append(branch2_scores[bi, pos_idx[top_pos]].detach().cpu())

        fp_cat = torch.cat(fp_parts, dim=0) if fp_parts else torch.empty(0, feature_dim)
        defect_cat = torch.cat(defect_parts, dim=0) if defect_parts else torch.empty(0, feature_dim)
        fp_score_cat = torch.cat(fp_score_parts, dim=0) if fp_score_parts else torch.empty(0)
        defect_score_cat = torch.cat(defect_score_parts, dim=0) if defect_score_parts else torch.empty(0)
        fp_branch1_score_cat = torch.cat(fp_branch1_score_parts, dim=0) if fp_branch1_score_parts else torch.empty(0)
        defect_branch1_score_cat = torch.cat(defect_branch1_score_parts, dim=0) if defect_branch1_score_parts else torch.empty(0)
        fp_branch2_score_cat = torch.cat(fp_branch2_score_parts, dim=0) if fp_branch2_score_parts else torch.empty(0)
        defect_branch2_score_cat = torch.cat(defect_branch2_score_parts, dim=0) if defect_branch2_score_parts else torch.empty(0)
        if fp_cat.shape[0] > max_fp_per_class:
            keep_idx = farthest_point_subsample_indices(fp_cat, max_fp_per_class, seed + ci)
            fp_cat = F.normalize(fp_cat.float(), dim=-1)[keep_idx]
            fp_score_cat = fp_score_cat[keep_idx]
            fp_branch1_score_cat = fp_branch1_score_cat[keep_idx]
            fp_branch2_score_cat = fp_branch2_score_cat[keep_idx]
        if defect_cat.shape[0] > max_defect_per_class:
            keep_idx = farthest_point_subsample_indices(defect_cat, max_defect_per_class, seed + 1000 + ci)
            defect_cat = F.normalize(defect_cat.float(), dim=-1)[keep_idx]
            defect_score_cat = defect_score_cat[keep_idx]
            defect_branch1_score_cat = defect_branch1_score_cat[keep_idx]
            defect_branch2_score_cat = defect_branch2_score_cat[keep_idx]
        fp_by_class.append(fp_cat)
        defect_by_class.append(defect_cat)
        fp_score_by_class.append(fp_score_cat.float())
        defect_score_by_class.append(defect_score_cat.float())
        fp_branch1_score_by_class.append(fp_branch1_score_cat.float())
        defect_branch1_score_by_class.append(defect_branch1_score_cat.float())
        fp_branch2_score_by_class.append(fp_branch2_score_cat.float())
        defect_branch2_score_by_class.append(defect_branch2_score_cat.float())
        print(
            json.dumps(
                {
                    "stage": "bank_class_done",
                    "class_name": class_name,
                    "fp_points": int(fp_cat.shape[0]),
                    "defect_points": int(defect_cat.shape[0]),
                }
            ),
            flush=True,
        )

    fp_bank = torch.cat(fp_by_class, dim=0) if fp_by_class else torch.empty(0, feature_dim)
    defect_bank = torch.cat(defect_by_class, dim=0) if defect_by_class else torch.empty(0, feature_dim)
    fp_score = torch.cat(fp_score_by_class, dim=0) if fp_score_by_class else torch.empty(0)
    defect_score = torch.cat(defect_score_by_class, dim=0) if defect_score_by_class else torch.empty(0)
    fp_branch1_score = torch.cat(fp_branch1_score_by_class, dim=0) if fp_branch1_score_by_class else torch.empty(0)
    defect_branch1_score = torch.cat(defect_branch1_score_by_class, dim=0) if defect_branch1_score_by_class else torch.empty(0)
    fp_branch2_score = torch.cat(fp_branch2_score_by_class, dim=0) if fp_branch2_score_by_class else torch.empty(0)
    defect_branch2_score = torch.cat(defect_branch2_score_by_class, dim=0) if defect_branch2_score_by_class else torch.empty(0)
    if fp_bank.shape[0] > max_fp_total:
        keep_idx = farthest_point_subsample_indices(fp_bank, max_fp_total, seed + 2000)
        fp_bank = F.normalize(fp_bank.float(), dim=-1)[keep_idx]
        fp_score = fp_score[keep_idx]
        fp_branch1_score = fp_branch1_score[keep_idx]
        fp_branch2_score = fp_branch2_score[keep_idx]
    if defect_bank.shape[0] > max_defect_total:
        keep_idx = farthest_point_subsample_indices(defect_bank, max_defect_total, seed + 3000)
        defect_bank = F.normalize(defect_bank.float(), dim=-1)[keep_idx]
        defect_score = defect_score[keep_idx]
        defect_branch1_score = defect_branch1_score[keep_idx]
        defect_branch2_score = defect_branch2_score[keep_idx]
    return {
        "fp": F.normalize(fp_bank.float(), dim=-1),
        "defect": F.normalize(defect_bank.float(), dim=-1),
        "fp_score": fp_score.float(),
        "defect_score": defect_score.float(),
        "fp_branch1_score": fp_branch1_score.float(),
        "defect_branch1_score": defect_branch1_score.float(),
        "fp_branch2_score": fp_branch2_score.float(),
        "defect_branch2_score": defect_branch2_score.float(),
        "source_class_names": class_names,
        "fp_by_class": [F.normalize(x.float(), dim=-1) for x in fp_by_class],
        "defect_by_class": [F.normalize(x.float(), dim=-1) for x in defect_by_class],
        "fp_score_by_class": [x.float() for x in fp_score_by_class],
        "defect_score_by_class": [x.float() for x in defect_score_by_class],
        "fp_branch1_score_by_class": [x.float() for x in fp_branch1_score_by_class],
        "defect_branch1_score_by_class": [x.float() for x in defect_branch1_score_by_class],
        "fp_branch2_score_by_class": [x.float() for x in fp_branch2_score_by_class],
        "defect_branch2_score_by_class": [x.float() for x in defect_branch2_score_by_class],
        "axis": axis.detach().cpu(),
        "protocol_version": BANK_PROTOCOL_VERSION,
        "source_root": source_root,
        "max_fp_per_class": max_fp_per_class,
        "max_defect_per_class": max_defect_per_class,
        "max_fp_total": max_fp_total,
        "max_defect_total": max_defect_total,
        "normal_topk_frac": normal_topk_frac,
        "source_fp_mode": source_fp_mode,
        "features_list": list(features_list),
        "image_size": image_size,
        "dap_token_mode": dap_token_mode,
        "dpam_layer": dpam_layer,
        "model_name": model_name,
        "prompt_load_mode": prompt_load_mode,
        "num_fp": int(fp_bank.shape[0]),
        "num_defect": int(defect_bank.shape[0]),
    }


def get_or_collect_source_banks(
    model,
    prompt_learner,
    preset: str,
    source_root: str,
    checkpoint_path: str,
    model_name: str,
    prompt_load_mode: str,
    cache_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    features_list: list[int],
    sigma: int,
    dap_token_mode: str,
    dpam_layer: int,
    max_fp_per_class: int,
    max_defect_per_class: int,
    max_fp_total: int,
    max_defect_total: int,
    normal_topk_frac: float,
    seed: int,
    force_rebuild_bank: bool,
    source_fp_mode: str = "hardfp",
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = get_bank_cache_path(
        preset,
        source_root,
        checkpoint_path,
        model_name,
        prompt_load_mode,
        cache_dir,
        max_fp_per_class,
        max_defect_per_class,
        max_fp_total,
        max_defect_total,
        normal_topk_frac,
        features_list,
        image_size,
        dap_token_mode,
        dpam_layer,
        seed,
        source_fp_mode,
    )
    requested_meta = {
        "protocol_version": BANK_PROTOCOL_VERSION,
        "source_root": source_root,
        "max_fp_per_class": max_fp_per_class,
        "max_defect_per_class": max_defect_per_class,
        "max_fp_total": max_fp_total,
        "max_defect_total": max_defect_total,
        "normal_topk_frac": normal_topk_frac,
        "features_list": list(features_list),
        "image_size": image_size,
        "dap_token_mode": dap_token_mode,
        "dpam_layer": dpam_layer,
        "model_name": model_name,
        "prompt_load_mode": prompt_load_mode,
    }
    if source_fp_mode != "hardfp":
        requested_meta["source_fp_mode"] = source_fp_mode
    if cache_path.exists() and not force_rebuild_bank:
        banks = torch.load(cache_path, map_location="cpu")
        cached_meta = {k: banks.get(k) for k in requested_meta}
        if cached_meta == requested_meta:
            print(json.dumps({"stage": "bank_cache_hit", "path": str(cache_path), "num_fp": banks["num_fp"], "num_defect": banks["num_defect"]}), flush=True)
            return banks
        print(json.dumps({"stage": "bank_cache_mismatch_rebuild", "path": str(cache_path), "cached": cached_meta, "requested": requested_meta}), flush=True)
    if source_fp_mode == "hardfp" and (not force_rebuild_bank) and dap_token_mode == "topk" and list(features_list) == [6, 12, 18, 24] and image_size == 518:
        legacy_path = get_legacy_bank_cache_path(
            source_root,
            checkpoint_path,
            cache_dir,
            max_fp_per_class,
            max_defect_per_class,
            max_fp_total,
            max_defect_total,
            normal_topk_frac,
            seed,
        )
        if legacy_path.exists():
            banks = torch.load(legacy_path, map_location="cpu")
            cached_meta = {k: banks.get(k) for k in requested_meta if k not in {"features_list", "image_size"}}
            requested_legacy_meta = {k: v for k, v in requested_meta.items() if k not in {"features_list", "image_size"}}
            if cached_meta == requested_legacy_meta:
                banks["features_list"] = list(features_list)
                banks["image_size"] = image_size
                torch.save(banks, cache_path)
                print(json.dumps({"stage": "bank_cache_legacy_promoted", "legacy_path": str(legacy_path), "path": str(cache_path), "num_fp": banks["num_fp"], "num_defect": banks["num_defect"]}), flush=True)
                return banks

    banks = collect_source_banks(
        model=model,
        prompt_learner=prompt_learner,
        source_root=source_root,
        model_name=model_name,
        prompt_load_mode=prompt_load_mode,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        features_list=features_list,
        sigma=sigma,
        dap_token_mode=dap_token_mode,
        dpam_layer=dpam_layer,
        max_fp_per_class=max_fp_per_class,
        max_defect_per_class=max_defect_per_class,
        max_fp_total=max_fp_total,
        max_defect_total=max_defect_total,
        normal_topk_frac=normal_topk_frac,
        seed=seed,
        source_fp_mode=source_fp_mode,
    )
    torch.save(banks, cache_path)
    print(json.dumps({"stage": "bank_cache_saved", "path": str(cache_path), "num_fp": banks["num_fp"], "num_defect": banks["num_defect"]}), flush=True)
    return banks


def logmeanexp_negative_sqdist_1d(query_z: torch.Tensor, bank_z: torch.Tensor, tau: float, chunk: int) -> torch.Tensor:
    if bank_z.numel() == 0:
        return torch.full_like(query_z, -1e6)
    running = None
    count = 0
    for start in range(0, bank_z.numel(), chunk):
        part = bank_z[start : start + chunk]
        diff2 = (query_z.unsqueeze(-1) - part.view(1, 1, -1)).pow(2)
        val = torch.logsumexp(-diff2 / tau, dim=-1)
        running = val if running is None else torch.logaddexp(running, val)
        count += part.numel()
    return running - math.log(max(count, 1))


def logmeanexp_cosine_support(query: torch.Tensor, bank: torch.Tensor, tau: float, chunk: int) -> torch.Tensor:
    if bank.numel() == 0:
        return torch.full(query.shape[:2], -1e6, device=query.device, dtype=query.dtype)
    query = F.normalize(query.float(), dim=-1)
    bank = F.normalize(bank.float(), dim=-1)
    running = None
    count = 0
    for start in range(0, bank.shape[0], chunk):
        part = bank[start : start + chunk].to(query.device)
        sim = torch.matmul(query, part.t())
        val = torch.logsumexp(sim / tau, dim=-1)
        running = val if running is None else torch.logaddexp(running, val)
        count += part.shape[0]
    return running - math.log(max(count, 1))


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
    fp_items: Sequence[torch.Tensor],
    defect_items: Sequence[torch.Tensor],
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


def prescore_subspace_transport_tokens(
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


def subspace_host_residual_token_score(
    host_score: torch.Tensor,
    rect_tokens: torch.Tensor,
    basis: torch.Tensor,
    score_w: Sequence[float],
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
    bank_chunk: int,
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
    host_score_key: str = "score",
) -> Dict[str, object] | None:
    if epochs <= 0:
        return None
    fp = F.normalize(banks["fp"].float(), dim=-1)
    defect = F.normalize(banks["defect"].float(), dim=-1)
    if fp.numel() == 0 or defect.numel() == 0:
        return None
    fp_score_key = f"fp_{host_score_key}"
    defect_score_key = f"defect_{host_score_key}"
    has_host_scores = fp_score_key in banks and defect_score_key in banks
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
    fp_host_all = banks[fp_score_key].float().to(device) if has_host_scores else fp_proj_all
    defect_host_all = banks[defect_score_key].float().to(device) if has_host_scores else defect_proj_all
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
    fp_q_def = logmeanexp_negative_sqdist_1d(fp_q_axis.unsqueeze(0), defect_proj_all, tau=tau, chunk=bank_chunk)[0]
    fp_q_fp = logmeanexp_negative_sqdist_1d(fp_q_axis.unsqueeze(0), fp_proj_all, tau=tau, chunk=bank_chunk)[0]
    defect_q_def = logmeanexp_negative_sqdist_1d(defect_q_axis.unsqueeze(0), defect_proj_all, tau=tau, chunk=bank_chunk)[0]
    defect_q_fp = logmeanexp_negative_sqdist_1d(defect_q_axis.unsqueeze(0), fp_proj_all, tau=tau, chunk=bank_chunk)[0]
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

    def current_params() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eta = torch.clamp(F.softplus(eta_raw), max=eta_max)
        readout_gamma = torch.clamp(F.softplus(readout_gamma_raw), max=eta_max)
        fp_weight = F.softplus(fp_weight_raw) + 1e-6
        direction = F.normalize(raw_direction, dim=0)
        score_w = F.normalize(scorer, dim=0)
        return eta, direction, transport_b, score_w, readout_gamma, fp_weight

    def transport_scores(
        features: torch.Tensor,
        host_scores: torch.Tensor,
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
        delta_fp = delta[:n].mean().item()
        delta_def = delta[n:].mean().item()

    return {
        "calibration": "source_calibrated_host_residual_subspace_rank_no_target",
        "transport_mode": "subspace_tanh",
        "eta": float(eta.item()),
        "alpha": float(eta.item()),
        "effective_alpha": float(eta.item()),
        "transport_direction": direction.detach().cpu().tolist(),
        "transport_b": float(b.item()),
        "subspace_score_w": score_w.detach().cpu().tolist(),
        "subspace_readout_mode": "host_residual",
        "readout_gamma": float(readout_gamma.item()),
        "preserve_host_margin_weight": float(preserve_host_margin_weight),
        "basis": basis.detach().cpu(),
        "subspace_rank": int(basis.shape[1]),
        "fp_weight": float(fp_weight.item()),
        "train_points_per_class": int(n),
        "source_pair_rank_accuracy": float(pair_acc),
        "source_base_pair_rank_accuracy": float(base_pair_acc),
        "source_delta_fp_mean": float(delta_fp),
        "source_delta_def_mean": float(delta_def),
        "hardpair_frac": float(hardpair_frac),
        "host_score_source": f"faprompt_{host_score_key}" if has_host_scores else "axis_projection_fallback",
        "subspace_basis_control": str(subspace_basis_control),
        "pair_label_control": str(pair_label_control),
    }


def build_bank_plugin_features(
    tokens: torch.Tensor,
    axis: torch.Tensor,
    defect_proj: torch.Tensor,
    fp_proj: torch.Tensor,
    tau: float,
    chunk: int,
) -> torch.Tensor:
    if tokens.ndim == 2:
        query_proj = tokens @ axis
        q_def = logmeanexp_negative_sqdist_1d(query_proj.unsqueeze(0), defect_proj, tau=tau, chunk=chunk)[0]
        q_fp = logmeanexp_negative_sqdist_1d(query_proj.unsqueeze(0), fp_proj, tau=tau, chunk=chunk)[0]
        return torch.stack([query_proj, q_def, q_fp, q_def - q_fp, q_def + q_fp], dim=-1)
    if tokens.ndim == 3:
        query_proj = tokens @ axis
        q_def = logmeanexp_negative_sqdist_1d(query_proj, defect_proj, tau=tau, chunk=chunk)
        q_fp = logmeanexp_negative_sqdist_1d(query_proj, fp_proj, tau=tau, chunk=chunk)
        return torch.stack([query_proj, q_def, q_fp, q_def - q_fp, q_def + q_fp], dim=-1)
    raise ValueError(f"Expected 2D or 3D tokens, got {tokens.shape}")


def build_fullfeat_bank_plugin_features(
    tokens: torch.Tensor,
    defect_bank: torch.Tensor,
    fp_bank: torch.Tensor,
    tau: float,
    chunk: int,
    fp_weight: float,
) -> torch.Tensor:
    q_def = logmeanexp_cosine_support(tokens, defect_bank, tau=tau, chunk=chunk)
    q_fp = logmeanexp_cosine_support(tokens, fp_bank, tau=tau, chunk=chunk)
    return torch.stack([q_def, q_fp, q_def - fp_weight * q_fp, q_def + q_fp], dim=-1)


def score_source_bank_plugin_features(feats: torch.Tensor, plugin: dict) -> torch.Tensor:
    if plugin.get("plugin_type", "linear") == "support_linear":
        if plugin.get("feature_space", "axis") == "hostscore":
            host_score = feats[..., 0]
            defect_weight = plugin["defect_weight"].to(feats.device)
            bias = plugin["bias"].to(feats.device)
            return defect_weight * host_score + bias
        if plugin.get("feature_space", "axis") == "fullfeat":
            q_def = feats[..., 0]
            q_fp = feats[..., 1]
        else:
            q_def = feats[..., 1]
            q_fp = feats[..., 2]
        defect_weight = plugin["defect_weight"].to(feats.device)
        hardfp_weight = plugin["hardfp_weight"].to(feats.device)
        bias = plugin["bias"].to(feats.device)
        return defect_weight * q_def - hardfp_weight * q_fp + bias
    mean = plugin["mean"].to(feats.device)
    std = plugin["std"].to(feats.device)
    x = ((feats - mean) / std).float()
    if plugin.get("plugin_type", "linear") == "rank_mlp":
        w1 = plugin["w1"].to(feats.device)
        b1 = plugin["b1"].to(feats.device)
        w2 = plugin["w2"].to(feats.device)
        b2 = plugin["b2"].to(feats.device)
        hidden = torch.tanh(x @ w1 + b1)
        return hidden @ w2 + b2
    weight = plugin["weight"].to(feats.device)
    bias = plugin["bias"].to(feats.device)
    return x @ weight + bias


def train_source_bank_plugin(
    banks: dict,
    axis: torch.Tensor,
    tau: float,
    bank_chunk: int,
    max_train_points: int,
    epochs: int,
    lr: float,
    loss_type: str,
    feature_space: str,
    fp_weight: float,
    margin: float,
    rank_temp: float,
    contrast_weight: float,
    contrast_margin: float,
    hidden_dim: int,
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
    fp_idx = torch.randperm(fp.shape[0], generator=rng)[:n]
    def_idx = torch.randperm(defect.shape[0], generator=rng)[:n]
    fp_train = fp[fp_idx].to(device)
    def_train = defect[def_idx].to(device)
    feature_space = feature_space.lower()
    if feature_space == "axis":
        axis_d = axis.to(device)
        fp_proj = fp.to(device) @ axis_d
        defect_proj = defect.to(device) @ axis_d
        x_fp = build_bank_plugin_features(fp_train, axis_d, defect_proj, fp_proj, tau=tau, chunk=bank_chunk)
        x_def = build_bank_plugin_features(def_train, axis_d, defect_proj, fp_proj, tau=tau, chunk=bank_chunk)
        feature_names = ["axis_projection", "defect_support", "hardfp_support", "support_margin", "support_mass"]
    elif feature_space == "fullfeat":
        fp_bank = fp.to(device)
        defect_bank = defect.to(device)
        x_fp = build_fullfeat_bank_plugin_features(fp_train, defect_bank, fp_bank, tau=tau, chunk=bank_chunk, fp_weight=fp_weight)
        x_def = build_fullfeat_bank_plugin_features(def_train, defect_bank, fp_bank, tau=tau, chunk=bank_chunk, fp_weight=fp_weight)
        feature_names = ["full_defect_support", "full_hardfp_support", "full_support_margin", "full_support_mass"]
    elif feature_space == "hostscore":
        if "fp_score" not in banks or "defect_score" not in banks:
            raise ValueError("plugin_feature_space=hostscore requires fp_score and defect_score in source banks")
        fp_scores = banks["fp_score"].float().to(device)
        defect_scores = banks["defect_score"].float().to(device)
        x_fp = fp_scores[fp_idx].unsqueeze(-1)
        x_def = defect_scores[def_idx].unsqueeze(-1)
        feature_names = ["host_score"]
    else:
        raise ValueError(f"Unknown plugin_feature_space: {feature_space}")
    x = torch.cat([x_fp, x_def], dim=0).float()
    y = torch.cat([torch.zeros(x_fp.shape[0], device=device), torch.ones(x_def.shape[0], device=device)], dim=0)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    xz_fp = (x_fp.float() - mean) / std
    xz_def = (x_def.float() - mean) / std
    xz = torch.cat([xz_fp, xz_def], dim=0)

    loss_type = loss_type.lower()
    rank_temp = max(float(rank_temp), 1e-6)
    if loss_type == "bce":
        weight = torch.zeros(xz.shape[1], device=device, requires_grad=True)
        bias = torch.zeros((), device=device, requires_grad=True)
        opt = torch.optim.Adam([weight, bias], lr=lr)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            logit = xz @ weight + bias
            loss = F.binary_cross_entropy_with_logits(logit, y)
            loss.backward()
            opt.step()
        with torch.no_grad():
            logit = xz @ weight + bias
            pred = (logit.sigmoid() >= 0.5).float()
            acc = (pred == y).float().mean().item()
            rank_acc = ((xz_def @ weight + bias) > (xz_fp @ weight + bias)).float().mean().item()
            pos_mean = logit[y > 0.5].mean().item()
            neg_mean = logit[y < 0.5].mean().item()
        plugin = {
            "plugin_type": "linear",
            "weight": weight.detach(),
            "bias": bias.detach(),
        }
    elif loss_type == "support_rank":
        def inv_softplus(value: float) -> float:
            value = max(float(value), 1e-6)
            return math.log(math.expm1(value))

        raw_defect_weight = torch.tensor(inv_softplus(1.0), device=device, requires_grad=True)
        raw_hardfp_weight = torch.tensor(inv_softplus(fp_weight), device=device, requires_grad=True)
        bias = torch.zeros((), device=device, requires_grad=True)

        def forward_support(x_in: torch.Tensor) -> torch.Tensor:
            if feature_space == "hostscore":
                return F.softplus(raw_defect_weight) * x_in[:, 0] + bias
            if feature_space == "fullfeat":
                q_def_col = x_in[:, 0]
                q_fp_col = x_in[:, 1]
            else:
                q_def_col = x_in[:, 1]
                q_fp_col = x_in[:, 2]
            return F.softplus(raw_defect_weight) * q_def_col - F.softplus(raw_hardfp_weight) * q_fp_col + bias

        opt = torch.optim.Adam([raw_defect_weight, raw_hardfp_weight, bias], lr=lr)
        pair_index = torch.arange(n, device=device)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            perm = pair_index[torch.randperm(n, device=device)]
            s_fp = forward_support(x_fp.float()[perm])
            s_def = forward_support(x_def.float())
            loss = F.softplus((s_fp - s_def + margin) / rank_temp).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            s_fp = forward_support(x_fp.float())
            s_def = forward_support(x_def.float())
            logit = torch.cat([s_fp, s_def], dim=0)
            acc = ((logit >= logit.median()).float() == y).float().mean().item()
            rank_acc = (s_def > s_fp).float().mean().item()
            pos_mean = s_def.mean().item()
            neg_mean = s_fp.mean().item()
            defect_weight = F.softplus(raw_defect_weight).detach()
            hardfp_weight = F.softplus(raw_hardfp_weight).detach()
        plugin = {
            "plugin_type": "support_linear",
            "defect_weight": defect_weight,
            "hardfp_weight": hardfp_weight,
            "bias": bias.detach(),
        }
    elif loss_type in {"pair_rank", "rank_contrast"}:
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            hidden_dim = 16
        w1 = torch.empty(xz.shape[1], hidden_dim, device=device, requires_grad=True)
        b1 = torch.zeros(hidden_dim, device=device, requires_grad=True)
        w2 = torch.empty(hidden_dim, device=device, requires_grad=True)
        b2 = torch.zeros((), device=device, requires_grad=True)
        torch.nn.init.xavier_uniform_(w1)
        torch.nn.init.normal_(w2, mean=0.0, std=0.02)

        def forward_rank(x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = torch.tanh(x_in @ w1 + b1)
            score = hidden @ w2 + b2
            z = F.normalize(hidden, dim=-1)
            return score, z

        opt = torch.optim.Adam([w1, b1, w2, b2], lr=lr)
        pair_index = torch.arange(n, device=device)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            perm = pair_index[torch.randperm(n, device=device)]
            s_fp, z_fp = forward_rank(xz_fp[perm])
            s_def, z_def = forward_rank(xz_def)
            loss_rank = F.softplus((s_fp - s_def + margin) / rank_temp).mean()
            loss = loss_rank
            if loss_type == "rank_contrast" and contrast_weight > 0:
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
            loss.backward()
            opt.step()
        with torch.no_grad():
            s_fp, _ = forward_rank(xz_fp)
            s_def, _ = forward_rank(xz_def)
            logit = torch.cat([s_fp, s_def], dim=0)
            acc = ((logit >= logit.median()).float() == y).float().mean().item()
            rank_acc = (s_def > s_fp).float().mean().item()
            pos_mean = s_def.mean().item()
            neg_mean = s_fp.mean().item()
        plugin = {
            "plugin_type": "rank_mlp",
            "w1": w1.detach(),
            "b1": b1.detach(),
            "w2": w2.detach(),
            "b2": b2.detach(),
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
        "feature_names": feature_names,
        "calibration": f"source_bank_{loss_type}_no_target",
        "loss_type": loss_type,
        "feature_space": feature_space,
        "fp_weight": float(fp_weight),
        "rank_margin": float(margin),
        "rank_temp": float(rank_temp),
        "contrast_weight": float(contrast_weight),
        "contrast_margin": float(contrast_margin),
        "hidden_dim": int(hidden_dim) if loss_type in {"pair_rank", "rank_contrast"} else 0,
    })
    if loss_type == "support_rank":
        plugin["learned_defect_weight"] = float(plugin["defect_weight"].item())
        plugin["learned_hardfp_weight"] = float(plugin["hardfp_weight"].item())
    return plugin


def evaluate_source_classheldout_rule(
    banks: dict,
    axis: torch.Tensor,
    tau: float,
    bank_chunk: int,
    max_train_points: int,
    max_val_points: int,
    epochs: int,
    lr: float,
    loss_type: str,
    feature_space: str,
    fp_weight: float,
    margin: float,
    rank_temp: float,
    contrast_weight: float,
    contrast_margin: float,
    hidden_dim: int,
    folds: int,
    seed: int,
    device: torch.device,
) -> dict | None:
    if folds <= 0:
        return None
    fp_by_class = banks.get("fp_by_class")
    defect_by_class = banks.get("defect_by_class")
    class_names = banks.get("source_class_names")
    if not fp_by_class or not defect_by_class or not class_names:
        return {"enabled": False, "reason": "bank_has_no_source_class_metadata"}
    valid = [
        i
        for i, (fp_i, def_i) in enumerate(zip(fp_by_class, defect_by_class))
        if fp_i.numel() > 0 and def_i.numel() > 0
    ]
    if len(valid) < 2:
        return {"enabled": False, "reason": "not_enough_source_classes"}

    rng = random.Random(seed)
    rng.shuffle(valid)
    heldout_indices = valid[: min(folds, len(valid))]
    rows: list[dict] = []
    axis_d = axis.to(device)
    for heldout in heldout_indices:
        train_fp_parts = [fp_by_class[i] for i in valid if i != heldout and fp_by_class[i].numel() > 0]
        train_def_parts = [defect_by_class[i] for i in valid if i != heldout and defect_by_class[i].numel() > 0]
        if not train_fp_parts or not train_def_parts:
            continue
        train_fp = F.normalize(torch.cat(train_fp_parts, dim=0).float(), dim=-1)
        train_def = F.normalize(torch.cat(train_def_parts, dim=0).float(), dim=-1)
        plugin = train_source_bank_plugin(
            banks={"fp": train_fp, "defect": train_def},
            axis=axis,
            tau=tau,
            bank_chunk=bank_chunk,
            max_train_points=max_train_points,
            epochs=epochs,
            lr=lr,
            loss_type=loss_type,
            feature_space=feature_space,
            fp_weight=fp_weight,
            margin=margin,
            rank_temp=rank_temp,
            contrast_weight=contrast_weight,
            contrast_margin=contrast_margin,
            hidden_dim=hidden_dim,
            seed=seed + 10000 + heldout,
            device=device,
        )
        if plugin is None:
            continue
        val_fp = F.normalize(fp_by_class[heldout].float(), dim=-1)
        val_def = F.normalize(defect_by_class[heldout].float(), dim=-1)
        n = min(val_fp.shape[0], val_def.shape[0], max_val_points // 2 if max_val_points > 0 else 10**9)
        if n <= 0:
            continue
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 20000 + heldout)
        val_fp = val_fp[torch.randperm(val_fp.shape[0], generator=gen)[:n]].to(device)
        val_def = val_def[torch.randperm(val_def.shape[0], generator=gen)[:n]].to(device)
        train_fp_d = train_fp.to(device)
        train_def_d = train_def.to(device)
        with torch.no_grad():
            if feature_space == "fullfeat":
                fp_feats = build_fullfeat_bank_plugin_features(val_fp, train_def_d, train_fp_d, tau=tau, chunk=bank_chunk, fp_weight=fp_weight)
                def_feats = build_fullfeat_bank_plugin_features(val_def, train_def_d, train_fp_d, tau=tau, chunk=bank_chunk, fp_weight=fp_weight)
            else:
                train_fp_proj = train_fp_d @ axis_d
                train_def_proj = train_def_d @ axis_d
                fp_q = val_fp @ axis_d
                def_q = val_def @ axis_d
                fp_feats = build_bank_plugin_features(val_fp, axis_d, train_def_proj, train_fp_proj, tau=tau, chunk=bank_chunk)
                def_feats = build_bank_plugin_features(val_def, axis_d, train_def_proj, train_fp_proj, tau=tau, chunk=bank_chunk)
            s_fp = score_source_bank_plugin_features(fp_feats, plugin)
            s_def = score_source_bank_plugin_features(def_feats, plugin)
            rows.append(
                {
                    "heldout_class": class_names[heldout],
                    "num_pairs": int(n),
                    "pair_rank_accuracy": float((s_def > s_fp).float().mean().item()),
                    "mean_margin": float((s_def - s_fp).mean().item()),
                    "train_pair_rank_accuracy": float(plugin["source_pair_rank_accuracy"]),
                    "learned_defect_weight": plugin.get("learned_defect_weight"),
                    "learned_hardfp_weight": plugin.get("learned_hardfp_weight"),
                }
            )
    if not rows:
        return {"enabled": False, "reason": "no_valid_source_cv_rows"}
    return {
        "enabled": True,
        "folds": len(rows),
        "mean_pair_rank_accuracy": float(np.mean([r["pair_rank_accuracy"] for r in rows])),
        "mean_margin": float(np.mean([r["mean_margin"] for r in rows])),
        "rows": rows,
    }


def apply_source_bank_plugin(
    tokens: torch.Tensor,
    axis: torch.Tensor,
    defect_proj: torch.Tensor,
    fp_proj: torch.Tensor,
    tau: float,
    bank_chunk: int,
    plugin: dict,
) -> torch.Tensor:
    feats = build_bank_plugin_features(tokens, axis, defect_proj, fp_proj, tau=tau, chunk=bank_chunk)
    return score_source_bank_plugin_features(feats, plugin)


def apply_source_bank_plugin_from_scores(
    query_proj: torch.Tensor,
    q_def: torch.Tensor,
    q_fp: torch.Tensor,
    plugin: dict,
) -> torch.Tensor:
    feats = torch.stack([query_proj, q_def, q_fp, q_def - q_fp, q_def + q_fp], dim=-1)
    return score_source_bank_plugin_features(feats, plugin)


def apply_source_bank_plugin_from_fullfeat_scores(
    q_def: torch.Tensor,
    q_fp: torch.Tensor,
    fp_weight: float,
    plugin: dict,
) -> torch.Tensor:
    feats = torch.stack([q_def, q_fp, q_def - fp_weight * q_fp, q_def + q_fp], dim=-1)
    return score_source_bank_plugin_features(feats, plugin)


def apply_source_bank_plugin_from_host_score(
    host_score: torch.Tensor,
    plugin: dict,
) -> torch.Tensor:
    feats = host_score.unsqueeze(-1)
    return score_source_bank_plugin_features(feats, plugin)


def high_score_gate_tokens(tokens: torch.Tensor, topk_frac: float, sharpness: float) -> torch.Tensor:
    k = max(1, int(math.ceil(tokens.shape[1] * topk_frac)))
    threshold = torch.topk(tokens, k=k, dim=1, largest=True).values[:, -1:].detach()
    scale = tokens.std(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.sigmoid(sharpness * (tokens - threshold) / scale)


def ambiguity_gate_from_margin(margin: torch.Tensor, temperature: float) -> torch.Tensor:
    margin_z = spatial_tanh_zscore(margin)
    return torch.exp(-margin_z.abs() / max(temperature, 1e-6))


def spatial_tanh_zscore(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=1, keepdim=True)
    sigma = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.tanh((x - mu) / sigma)


def upsample_tokens(tokens: torch.Tensor, image_size: int, sigma: int) -> np.ndarray:
    side = int(round(math.sqrt(tokens.shape[1])))
    out = tokens.view(tokens.shape[0], 1, side, side)
    out = F.interpolate(out, size=(image_size, image_size), mode="bilinear", align_corners=True)[:, 0]
    arr = out.detach().cpu().numpy()
    if sigma > 0:
        arr = np.stack([gaussian_filter(x, sigma=sigma) for x in arr], axis=0)
    return arr


def topk_mean_np(arr: np.ndarray, frac: float) -> np.ndarray:
    flat = arr.reshape(arr.shape[0], -1)
    k = max(1, int(math.ceil(flat.shape[1] * frac)))
    idx = np.argpartition(flat, -k, axis=1)[:, -k:]
    return np.take_along_axis(flat, idx, axis=1).mean(axis=1)


def official_residual_maps(
    baseline_maps: np.ndarray,
    support_maps: np.ndarray,
    alpha: float,
    gate_quantile: float,
    gate_temp: float,
) -> np.ndarray:
    base = baseline_maps.astype(np.float32)
    residual = support_maps.astype(np.float32)
    flat = base.reshape(base.shape[0], -1)
    threshold = np.quantile(flat, gate_quantile, axis=1).reshape(-1, 1, 1)
    scale = flat.std(axis=1).reshape(-1, 1, 1).clip(min=1e-6)
    gate = 1.0 / (1.0 + np.exp(-((base - threshold) / (scale * gate_temp + 1e-6))))
    return (base + alpha * scale * gate * residual).astype(np.float32)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or np.min(y_true) == np.max(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or np.min(y_true) == np.max(y_true):
        return float("nan")
    return float(average_precision_score(y_true, y_score))


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
    max_step: int = 200,
    expect_fpr: float = 0.3,
    precomputed: dict | None = None,
) -> float:
    pre = precomputed if precomputed is not None else precompute_aupro_regions(masks)
    if not pre["has_foreground"]:
        return float("nan")
    maps = maps.astype(np.float32)
    min_th = float(np.min(maps))
    max_th = float(np.max(maps))
    if not np.isfinite(min_th) or not np.isfinite(max_th) or max_th <= min_th:
        return 0.5
    delta = (max_th - min_th) / max_step
    pros: list[float] = []
    fprs: list[float] = []
    inverse_masks = pre["inverse_masks"]
    inverse_total = int(pre["inverse_total"])
    if inverse_total == 0:
        return float("nan")
    for th in np.arange(min_th, max_th, delta):
        binary_maps = maps > th
        pro_vals = []
        for binary_map, regions in zip(binary_maps, pre["regions_by_image"]):
            for coords in regions:
                tp_pixels = binary_map[coords[:, 0], coords[:, 1]].sum()
                pro_vals.append(float(tp_pixels / max(coords.shape[0], 1)))
        if not pro_vals:
            continue
        fp_pixels = np.logical_and(inverse_masks, binary_maps).sum()
        pros.append(float(np.mean(pro_vals)))
        fprs.append(float(fp_pixels / inverse_total))
    if not pros:
        return float("nan")
    pros_np = np.asarray(pros, dtype=np.float32)
    fprs_np = np.asarray(fprs, dtype=np.float32)
    keep = fprs_np < expect_fpr
    if int(keep.sum()) <= 2:
        return 0.5
    fprs_keep = fprs_np[keep]
    denom = float(fprs_keep.max() - fprs_keep.min())
    if denom <= 1e-12:
        return 0.5
    fprs_keep = (fprs_keep - fprs_keep.min()) / denom
    return float(auc(fprs_keep, pros_np[keep]))


def align_masks_to_maps(masks: np.ndarray, maps: np.ndarray) -> np.ndarray:
    if masks.shape[0] != maps.shape[0]:
        raise ValueError(
            "Mask/map image count mismatch before metric computation: "
            f"masks={masks.shape}, maps={maps.shape}. "
            "This usually means the host output still has an uncollapsed batch/channel axis."
        )
    if masks.shape[-2:] == maps.shape[-2:]:
        return masks.astype(np.float32)
    masks_t = torch.from_numpy(masks[:, None].astype(np.float32))
    resized = F.interpolate(masks_t, size=maps.shape[-2:], mode="nearest")[:, 0]
    return resized.numpy().astype(np.float32)


def compute_metrics(
    masks: np.ndarray,
    labels: np.ndarray,
    maps: np.ndarray,
    image_scores: np.ndarray,
    aupro_precomputed: dict | None = None,
    aupro_max_step: int = 200,
) -> dict:
    masks = align_masks_to_maps(masks, maps)
    if aupro_precomputed is not None and tuple(aupro_precomputed.get("masks", np.empty((0,))).shape) != tuple(masks.shape):
        aupro_precomputed = None
    gt_px = masks.reshape(-1).astype(np.uint8)
    pr_px = maps.reshape(-1).astype(np.float32)
    return {
        "pixel_auc": safe_auc(gt_px, pr_px),
        "pixel_ap": safe_ap(gt_px, pr_px),
        "pixel_aupro": calculate_aupro(masks, maps, max_step=aupro_max_step, precomputed=aupro_precomputed),
        "image_auc": safe_auc(labels.astype(np.uint8), image_scores.astype(np.float32)),
        "image_ap": safe_ap(labels.astype(np.uint8), image_scores.astype(np.float32)),
    }


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
            else:
                item = sep.get(mode, {})
            value = item.get(key)
            if value is not None and np.isfinite(value):
                vals.append(float(value))
        out[key] = float(np.nanmean(vals)) if vals else float("nan")
    return out


def evaluate_target(
    model,
    prompt_learner,
    target_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    features_list: list[int],
    sigma: int,
    dap_token_mode: str,
    dpam_layer: int,
    banks: dict,
    tau: float,
    fp_weight: float,
    alphas: list[float],
    insert_modes: list[str],
    image_score_topk: float,
    bank_chunk: int,
    aupro_max_step: int,
    ours_image_score_mode: str,
    residual_gate_quantile: float,
    residual_gate_temp: float,
    source_plugin: dict | None,
    source_prescore_calibrator: dict | None,
    source_branch_calibrators: dict[str, dict] | None,
    plugin_gate_topk_frac: float,
    plugin_gate_sharpness: float,
    plugin_ambiguity_temp: float,
    rectifier_gate_mode: str,
    rectifier_gate_sharpness: float,
    rectifier_direction_mode: str,
    target_limit_per_class: int | None,
    seed: int,
    target_class_name: str | None = None,
    report_group_separability: bool = False,
    separability_hard_frac: float = 0.01,
) -> dict:
    meta = json.loads((Path(target_root) / "meta.json").read_text(encoding="utf-8"))
    class_names = sorted(meta["test"].keys())
    if target_class_name is not None:
        if target_class_name not in meta["test"]:
            raise KeyError(f"{target_class_name} not found in {target_root}/meta.json test split")
        class_names = [target_class_name]
    base_text_pair = learned_text_pair(model, prompt_learner)
    axis = F.normalize(base_text_pair[1] - base_text_pair[0], dim=0)
    fp_proj = banks["fp"].to(device) @ axis
    defect_proj = banks["defect"].to(device) @ axis
    per_class = []
    rng = random.Random(seed)

    for class_name in class_names:
        ds = MetaDataset(target_root, image_size, split="test", class_name=class_name)
        if target_limit_per_class is not None and target_limit_per_class > 0 and target_limit_per_class < len(ds):
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            ds = torch.utils.data.Subset(ds, indices[:target_limit_per_class])
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        masks_all: list[np.ndarray] = []
        labels_all: list[np.ndarray] = []
        baseline_maps_all: list[np.ndarray] = []
        baseline_maptopk_all: list[np.ndarray] = []
        official_img_all: list[np.ndarray] = []
        candidate_keys = [f"{mode}_alpha_{alpha:g}" for mode in insert_modes for alpha in alphas]
        ours_maps_by_key: dict[str, list[np.ndarray]] = {key: [] for key in candidate_keys}
        ours_img_by_key: dict[str, list[np.ndarray]] = {key: [] for key in candidate_keys}

        for batch in dl:
            image = batch["image"].to(device)
            mask = batch["mask"].numpy()
            label = batch["label"].numpy()
            out = compute_faprompt_outputs(model, prompt_learner, image, features_list, image_size, sigma, dap_token_mode, dpam_layer, base_text_pair)
            tokens = out["patch_tokens"]
            baseline_token = out["token_score"]
            branch1_token = out["branch1_token_score"]
            branch2_token = out["branch2_token_score"]
            query_proj = tokens @ axis
            q_def = logmeanexp_negative_sqdist_1d(query_proj, defect_proj, tau=tau, chunk=bank_chunk)
            q_fp = logmeanexp_negative_sqdist_1d(query_proj, fp_proj, tau=tau, chunk=bank_chunk)
            support_margin = spatial_tanh_zscore(q_def - fp_weight * q_fp)
            full_support_margin = None
            q_def_full = None
            q_fp_full = None
            baseline_z = spatial_tanh_zscore(baseline_token)
            branch1_z = spatial_tanh_zscore(branch1_token)
            branch2_z = spatial_tanh_zscore(branch2_token)
            plugin_z = None
            plugin_gate = None
            if source_plugin is not None and any(mode.startswith("plugin_") for mode in insert_modes):
                if source_plugin.get("feature_space", "axis") == "hostscore":
                    plugin_logit = apply_source_bank_plugin_from_host_score(baseline_token, source_plugin)
                    ambiguity_margin = None
                elif source_plugin.get("feature_space", "axis") == "fullfeat":
                    q_def_full = logmeanexp_cosine_support(tokens, banks["defect"].to(device), tau=tau, chunk=bank_chunk)
                    q_fp_full = logmeanexp_cosine_support(tokens, banks["fp"].to(device), tau=tau, chunk=bank_chunk)
                    full_support_margin = spatial_tanh_zscore(q_def_full - fp_weight * q_fp_full)
                    plugin_logit = apply_source_bank_plugin_from_fullfeat_scores(q_def_full, q_fp_full, fp_weight, source_plugin)
                    ambiguity_margin = q_def_full - fp_weight * q_fp_full
                else:
                    plugin_logit = apply_source_bank_plugin_from_scores(query_proj, q_def, q_fp, source_plugin)
                    ambiguity_margin = q_def - fp_weight * q_fp
                plugin_z = spatial_tanh_zscore(plugin_logit)
                plugin_gate = high_score_gate_tokens(baseline_z, plugin_gate_topk_frac, plugin_gate_sharpness)
                if source_plugin.get("feature_space", "axis") == "axis":
                    plugin_gate = plugin_gate * ambiguity_gate_from_margin(ambiguity_margin, plugin_ambiguity_temp)

            baseline_maps = out["anomaly_map"]
            if baseline_maps.ndim == 2:
                baseline_maps = baseline_maps[None, ...]
            support_maps = None
            baseline_maps_all.append(baseline_maps.astype(np.float32))
            baseline_maptopk_all.append(topk_mean_np(baseline_maps.astype(np.float32), image_score_topk))
            official_img_all.append(out["official_image_score"].astype(np.float32))
            masks_all.append(mask.astype(np.float32))
            labels_all.append(label.astype(np.int64))

            for alpha in alphas:
                for mode in insert_modes:
                    key = f"{mode}_alpha_{alpha:g}"
                    mode_img_score = None
                    if mode == "official_residual":
                        if support_maps is None:
                            support_maps = upsample_tokens(support_margin, image_size=image_size, sigma=sigma).astype(np.float32)
                        maps = official_residual_maps(
                            baseline_maps=baseline_maps,
                            support_maps=support_maps,
                            alpha=alpha,
                            gate_quantile=residual_gate_quantile,
                            gate_temp=residual_gate_temp,
                        )
                    elif mode == "prescore_rectify":
                        rect_out = compute_faprompt_outputs(
                            model,
                            prompt_learner,
                            image,
                            features_list,
                            image_size,
                            sigma,
                            dap_token_mode,
                            dpam_layer,
                            base_text_pair,
                            rectifier={
                                "defect_bank": banks["defect"].to(device),
                                "fp_bank": banks["fp"].to(device),
                                "strength": alpha,
                                "tau": tau,
                                "fp_weight": fp_weight,
                                "gate_sharpness": rectifier_gate_sharpness,
                                "gate_mode": rectifier_gate_mode,
                                "direction_mode": rectifier_direction_mode,
                                "axis": axis,
                            },
                        )
                        maps = rect_out["anomaly_map"]
                        if maps.ndim == 2:
                            maps = maps[None, ...]
                        maps = maps.astype(np.float32)
                        mode_img_score = rect_out["official_image_score"].astype(np.float32)
                    elif mode == "prescore_calibrated":
                        if source_prescore_calibrator is None:
                            raise RuntimeError("prescore_calibrated requires --prescore_calib_epochs > 0")
                        rect_tokens = prescore_subspace_transport_tokens(
                            tokens=tokens,
                            q_def=q_def,
                            q_fp=q_fp,
                            basis=source_prescore_calibrator["basis"],
                            eta=float(source_prescore_calibrator["eta"]) * float(alpha),
                            direction=source_prescore_calibrator["transport_direction"],
                            transport_b=float(source_prescore_calibrator["transport_b"]),
                            fp_weight=float(source_prescore_calibrator["fp_weight"]),
                        )
                        corrected = subspace_host_residual_token_score(
                            host_score=baseline_token,
                            rect_tokens=rect_tokens,
                            basis=source_prescore_calibrator["basis"],
                            score_w=source_prescore_calibrator["subspace_score_w"],
                            readout_gamma=float(source_prescore_calibrator.get("readout_gamma", 1.0)),
                        )
                        residual_token = corrected - baseline_token
                        residual_maps = upsample_tokens(residual_token, image_size=image_size, sigma=sigma).astype(np.float32)
                        maps = official_residual_maps(
                            baseline_maps=baseline_maps,
                            support_maps=residual_maps,
                            alpha=alpha,
                            gate_quantile=residual_gate_quantile,
                            gate_temp=residual_gate_temp,
                        )
                        mode_img_score = out["official_image_score"].astype(np.float32)
                    elif mode == "branch_calibrated":
                        if source_branch_calibrators is None:
                            raise RuntimeError("branch_calibrated requires --prescore_calib_epochs > 0")
                        cal1 = source_branch_calibrators.get("branch1")
                        cal2 = source_branch_calibrators.get("branch2")
                        if cal1 is None or cal2 is None:
                            raise RuntimeError("branch_calibrated requires branch1 and branch2 calibrators")
                        rect_tokens1 = prescore_subspace_transport_tokens(
                            tokens=tokens,
                            q_def=q_def,
                            q_fp=q_fp,
                            basis=cal1["basis"],
                            eta=float(cal1["eta"]),
                            direction=cal1["transport_direction"],
                            transport_b=float(cal1["transport_b"]),
                            fp_weight=float(cal1["fp_weight"]),
                        )
                        rect_tokens2 = prescore_subspace_transport_tokens(
                            tokens=tokens,
                            q_def=q_def,
                            q_fp=q_fp,
                            basis=cal2["basis"],
                            eta=float(cal2["eta"]),
                            direction=cal2["transport_direction"],
                            transport_b=float(cal2["transport_b"]),
                            fp_weight=float(cal2["fp_weight"]),
                        )
                        corrected1 = subspace_host_residual_token_score(
                            host_score=branch1_token,
                            rect_tokens=rect_tokens1,
                            basis=cal1["basis"],
                            score_w=cal1["subspace_score_w"],
                            readout_gamma=float(cal1.get("readout_gamma", 1.0)),
                        )
                        corrected2 = subspace_host_residual_token_score(
                            host_score=branch2_token,
                            rect_tokens=rect_tokens2,
                            basis=cal2["basis"],
                            score_w=cal2["subspace_score_w"],
                            readout_gamma=float(cal2.get("readout_gamma", 1.0)),
                        )
                        residual_token = 0.5 * ((corrected1 - branch1_token) + (corrected2 - branch2_token))
                        token_gate = high_score_gate_tokens(
                            baseline_z,
                            topk_frac=max(1e-6, 1.0 - float(residual_gate_quantile)),
                            sharpness=max(1e-6, 1.0 / max(float(residual_gate_temp), 1e-6)),
                        )
                        corrected = baseline_token + float(alpha) * token_gate * residual_token
                        maps = upsample_tokens(corrected, image_size=image_size, sigma=sigma).astype(np.float32)
                        mode_img_score = out["official_image_score"].astype(np.float32)
                    else:
                        if mode == "full_blend":
                            corrected = (1.0 - alpha) * baseline_z + alpha * support_margin
                        elif mode == "dap_rescue":
                            rescued_dap = (1.0 - alpha) * branch2_z + alpha * support_margin
                            corrected = 0.5 * branch1_z + 0.5 * rescued_dap
                        elif mode == "dap_add":
                            corrected = 0.5 * branch1_z + 0.5 * spatial_tanh_zscore(branch2_z + alpha * support_margin)
                        elif mode == "dap_replace":
                            corrected = 0.5 * branch1_z + 0.5 * support_margin
                        elif mode == "support_only":
                            corrected = support_margin
                        elif mode == "fullfeat_residual":
                            if full_support_margin is None:
                                q_def_full = logmeanexp_cosine_support(tokens, banks["defect"].to(device), tau=tau, chunk=bank_chunk)
                                q_fp_full = logmeanexp_cosine_support(tokens, banks["fp"].to(device), tau=tau, chunk=bank_chunk)
                                full_support_margin = spatial_tanh_zscore(q_def_full - fp_weight * q_fp_full)
                            gate = high_score_gate_tokens(baseline_z, plugin_gate_topk_frac, plugin_gate_sharpness)
                            corrected = baseline_z + alpha * gate * full_support_margin
                        elif mode == "fullfeat_replace":
                            if full_support_margin is None:
                                q_def_full = logmeanexp_cosine_support(tokens, banks["defect"].to(device), tau=tau, chunk=bank_chunk)
                                q_fp_full = logmeanexp_cosine_support(tokens, banks["fp"].to(device), tau=tau, chunk=bank_chunk)
                                full_support_margin = spatial_tanh_zscore(q_def_full - fp_weight * q_fp_full)
                            corrected = full_support_margin
                        elif mode == "plugin_replace":
                            if plugin_z is None:
                                raise RuntimeError("plugin_replace requires --plugin_epochs > 0")
                            corrected = plugin_z
                        elif mode == "plugin_residual":
                            if plugin_z is None or plugin_gate is None:
                                raise RuntimeError("plugin_residual requires --plugin_epochs > 0")
                            corrected = baseline_z + alpha * plugin_gate * plugin_z
                        elif mode == "plugin_fullfeat_residual":
                            if plugin_z is None or plugin_gate is None:
                                raise RuntimeError("plugin_fullfeat_residual requires --plugin_epochs > 0")
                            if source_plugin.get("feature_space", "axis") != "fullfeat":
                                raise RuntimeError("plugin_fullfeat_residual requires --plugin_feature_space fullfeat")
                            corrected = baseline_z + alpha * plugin_gate * plugin_z
                        elif mode == "plugin_dap_rescue":
                            if plugin_z is None or plugin_gate is None:
                                raise RuntimeError("plugin_dap_rescue requires --plugin_epochs > 0")
                            rescued_dap = branch2_z + alpha * plugin_gate * plugin_z
                            corrected = 0.5 * branch1_z + 0.5 * spatial_tanh_zscore(rescued_dap)
                        else:
                            raise ValueError(f"Unknown insert mode: {mode}")
                        maps = upsample_tokens(corrected, image_size=image_size, sigma=sigma).astype(np.float32)
                    ours_maps_by_key[key].append(maps)
                    if ours_image_score_mode == "corrected_topk":
                        ours_img = topk_mean_np(maps, image_score_topk)
                    elif ours_image_score_mode == "official":
                        ours_img = mode_img_score if mode_img_score is not None else out["official_image_score"].astype(np.float32)
                    else:
                        raise ValueError(f"Unknown ours_image_score_mode: {ours_image_score_mode}")
                    ours_img_by_key[key].append(ours_img)

        masks_np = np.concatenate(masks_all, axis=0)
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]
        aupro_precomputed = precompute_aupro_regions(masks_np)
        labels_np = np.concatenate(labels_all, axis=0)
        baseline_maps_np = np.concatenate(baseline_maps_all, axis=0)
        official_img_np = np.concatenate(official_img_all, axis=0)
        baseline_maptopk_np = np.concatenate(baseline_maptopk_all, axis=0)
        class_row = {
            "class_name": class_name,
            "num_images": int(labels_np.shape[0]),
            "baseline_official": compute_metrics(
                masks_np, labels_np, baseline_maps_np, official_img_np, aupro_precomputed, aupro_max_step
            ),
            "baseline_maptopk": compute_metrics(
                masks_np, labels_np, baseline_maps_np, baseline_maptopk_np, aupro_precomputed, aupro_max_step
            ),
            "parallel": {},
        }
        if report_group_separability:
            class_row["separability"] = {
                "baseline_official": map_group_separability(
                    masks_np,
                    labels_np,
                    baseline_maps_np,
                    hard_frac=separability_hard_frac,
                ),
                "parallel": {},
            }
        for key in candidate_keys:
            maps_np = np.concatenate(ours_maps_by_key[key], axis=0)
            img_np = np.concatenate(ours_img_by_key[key], axis=0)
            class_row["parallel"][key] = compute_metrics(
                masks_np, labels_np, maps_np, img_np, aupro_precomputed, aupro_max_step
            )
            if report_group_separability:
                class_row["separability"]["parallel"][key] = map_group_separability(
                    masks_np,
                    labels_np,
                    maps_np,
                    hard_frac=separability_hard_frac,
                )
        per_class.append(class_row)
        print(json.dumps({"stage": "target_class_done", "class_name": class_name}), flush=True)

    def mean_metrics(mode: str, alpha_key: str | None = None) -> dict:
        keys = ["pixel_auc", "pixel_ap", "pixel_aupro", "image_auc", "image_ap"]
        out = {}
        for key in keys:
            vals = []
            for row in per_class:
                if mode == "parallel":
                    vals.append(row["parallel"][alpha_key][key])
                else:
                    vals.append(row[mode][key])
            out[key] = float(np.nanmean(vals)) if vals else float("nan")
        return out

    result = {
        "per_class": per_class,
        "mean": {
            "baseline_official": mean_metrics("baseline_official"),
            "baseline_maptopk": mean_metrics("baseline_maptopk"),
            "parallel": {key: mean_metrics("parallel", key) for key in [f"{mode}_alpha_{alpha:g}" for mode in insert_modes for alpha in alphas]},
        },
    }
    if report_group_separability:
        result["separability_mean"] = {
            "baseline_official": mean_separability(per_class, "baseline_official"),
            "parallel": {
                key: mean_separability(per_class, "parallel", key)
                for key in [f"{mode}_alpha_{alpha:g}" for mode in insert_modes for alpha in alphas]
            },
        }
    return result


def parse_alpha_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, required=True, choices=sorted(PRESET_CONFIGS))
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--checkpoint_path", type=str, default="")
    ap.add_argument("--model_name", type=str, default="ViT-L/14@336px")
    ap.add_argument("--prompt_load_mode", type=str, default="strict", choices=["strict", "project_crop"])
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    ap.add_argument("--dpam_layer", type=int, default=20)
    ap.add_argument("--sigma", type=int, default=10)
    ap.add_argument("--dap_token_mode", type=str, default="topk", choices=["topk", "official_firstk"])
    ap.add_argument("--max_fp_per_class", type=int, default=512)
    ap.add_argument("--max_defect_per_class", type=int, default=512)
    ap.add_argument("--max_fp_total", type=int, default=4096)
    ap.add_argument("--max_defect_total", type=int, default=4096)
    ap.add_argument("--normal_topk_frac", type=float, default=0.01)
    ap.add_argument("--source_fp_mode", type=str, default="hardfp", choices=["hardfp", "random_normal", "easy_normal"])
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--fp_weight", type=float, default=1.0)
    ap.add_argument("--alphas", type=str, default="0.25,0.5,0.75,1.0")
    ap.add_argument("--image_score_topk", type=float, default=0.01)
    ap.add_argument("--ours_image_score_mode", type=str, default="corrected_topk", choices=["corrected_topk", "official"])
    ap.add_argument("--residual_gate_quantile", type=float, default=0.9)
    ap.add_argument("--residual_gate_temp", type=float, default=0.25)
    ap.add_argument("--plugin_epochs", type=int, default=0)
    ap.add_argument("--plugin_lr", type=float, default=0.05)
    ap.add_argument("--plugin_max_train_points", type=int, default=4096)
    ap.add_argument("--plugin_source_cv_folds", type=int, default=0)
    ap.add_argument("--plugin_source_cv_max_val_points", type=int, default=512)
    ap.add_argument("--plugin_loss", type=str, default="bce", choices=["bce", "support_rank", "pair_rank", "rank_contrast"])
    ap.add_argument("--plugin_feature_space", type=str, default="axis", choices=["axis", "fullfeat", "hostscore"])
    ap.add_argument("--plugin_rank_margin", type=float, default=0.25)
    ap.add_argument("--plugin_rank_temp", type=float, default=0.1)
    ap.add_argument("--plugin_contrast_weight", type=float, default=0.1)
    ap.add_argument("--plugin_contrast_margin", type=float, default=0.2)
    ap.add_argument("--plugin_hidden_dim", type=int, default=16)
    ap.add_argument("--plugin_gate_topk_frac", type=float, default=0.10)
    ap.add_argument("--plugin_gate_sharpness", type=float, default=6.0)
    ap.add_argument("--plugin_ambiguity_temp", type=float, default=0.5)
    ap.add_argument("--rectifier_gate_mode", type=str, default="support_mass", choices=["support_mass", "hardfp", "none"])
    ap.add_argument("--rectifier_gate_sharpness", type=float, default=3.0)
    ap.add_argument("--rectifier_direction_mode", type=str, default="transport", choices=["transport", "host_axis", "learned_subspace"])
    ap.add_argument("--prescore_calib_epochs", type=int, default=0)
    ap.add_argument("--prescore_calib_lr", type=float, default=0.02)
    ap.add_argument("--prescore_calib_max_train_points", type=int, default=8192)
    ap.add_argument("--prescore_calib_rank_margin", type=float, default=0.05)
    ap.add_argument("--prescore_calib_rank_temp", type=float, default=0.05)
    ap.add_argument("--prescore_calib_alpha_init", type=float, default=0.2)
    ap.add_argument("--prescore_calib_alpha_max", type=float, default=1.0)
    ap.add_argument("--prescore_calib_alpha_reg", type=float, default=0.001)
    ap.add_argument("--prescore_calib_fp_weight_init", type=float, default=1.0)
    ap.add_argument("--prescore_calib_learn_fp_weight", action="store_true")
    ap.add_argument("--prescore_calib_hardpair_frac", type=float, default=0.75)
    ap.add_argument("--prescore_calib_subspace_rank", type=int, default=8)
    ap.add_argument("--prescore_calib_subspace_basis_control", type=str, default="source", choices=["source", "random"])
    ap.add_argument("--prescore_calib_pair_label_control", type=str, default="correct", choices=["correct", "shuffle", "swap"])
    ap.add_argument("--prescore_calib_preserve_host_margin_weight", type=float, default=1.0)
    ap.add_argument("--bank_chunk", type=int, default=2048)
    ap.add_argument("--aupro_max_step", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force_rebuild_bank", action="store_true")
    ap.add_argument("--target_class_name", type=str, default="")
    ap.add_argument("--target_limit_per_class", type=int, default=0)
    ap.add_argument("--report_group_separability", action="store_true")
    ap.add_argument("--separability_hard_frac", type=float, default=0.01)
    ap.add_argument("--insert_modes", type=str, default="full_blend,dap_rescue,dap_add,dap_replace,support_only")
    args = ap.parse_args()

    setup_seed(args.seed)
    cfg = dict(PRESET_CONFIGS[args.preset])
    if args.checkpoint_path:
        cfg["checkpoint_path"] = args.checkpoint_path
    validate_visa_root(cfg["source_root"], cfg["source_dataset"])
    validate_visa_root(cfg["target_root"], cfg["target_dataset"])
    if not Path(cfg["checkpoint_path"]).exists():
        raise FileNotFoundError(cfg["checkpoint_path"])

    device = torch.device(args.device)
    model, prompt_learner, prompt_load_info = build_model(
        device=device,
        checkpoint_path=cfg["checkpoint_path"],
        model_name=args.model_name,
        prompt_load_mode=args.prompt_load_mode,
        dpam_layer=args.dpam_layer,
        depth=args.depth,
        n_ctx=args.n_ctx,
        t_n_ctx=args.t_n_ctx,
    )
    cache_dir = ROOT / "neurips2026" / "results" / "bank_cache"
    banks = get_or_collect_source_banks(
        model=model,
        prompt_learner=prompt_learner,
        preset=args.preset,
        source_root=cfg["source_root"],
        checkpoint_path=cfg["checkpoint_path"],
        model_name=args.model_name,
        prompt_load_mode=args.prompt_load_mode,
        cache_dir=cache_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        features_list=args.features_list,
        sigma=args.sigma,
        dap_token_mode=args.dap_token_mode,
        dpam_layer=prompt_load_info["effective_dpam_layer"],
        max_fp_per_class=args.max_fp_per_class,
        max_defect_per_class=args.max_defect_per_class,
        max_fp_total=args.max_fp_total,
        max_defect_total=args.max_defect_total,
        normal_topk_frac=args.normal_topk_frac,
        seed=args.seed,
        force_rebuild_bank=args.force_rebuild_bank,
        source_fp_mode=args.source_fp_mode,
    )
    alphas = parse_alpha_list(args.alphas)
    insert_modes = parse_str_list(args.insert_modes)
    source_prescore_calibrator = None
    source_branch_calibrators = None
    if any(mode == "prescore_calibrated" for mode in insert_modes):
        base_text_pair = learned_text_pair(model, prompt_learner)
        axis = F.normalize(base_text_pair[1] - base_text_pair[0], dim=0)
        source_prescore_calibrator = train_subspace_host_residual_calibrator(
            banks=banks,
            axis=axis,
            tau=args.tau,
            bank_chunk=args.bank_chunk,
            max_train_points=args.prescore_calib_max_train_points,
            epochs=args.prescore_calib_epochs,
            lr=args.prescore_calib_lr,
            rank_margin=args.prescore_calib_rank_margin,
            rank_temp=args.prescore_calib_rank_temp,
            eta_init=args.prescore_calib_alpha_init,
            eta_max=args.prescore_calib_alpha_max,
            eta_reg=args.prescore_calib_alpha_reg,
            fp_weight_init=args.prescore_calib_fp_weight_init,
            learn_fp_weight=args.prescore_calib_learn_fp_weight,
            hardpair_frac=args.prescore_calib_hardpair_frac,
            subspace_rank=args.prescore_calib_subspace_rank,
            subspace_basis_control=args.prescore_calib_subspace_basis_control,
            pair_label_control=args.prescore_calib_pair_label_control,
            preserve_host_margin_weight=args.prescore_calib_preserve_host_margin_weight,
            seed=args.seed,
            device=device,
        )
        if source_prescore_calibrator is None:
            raise RuntimeError("prescore_calibrated requested, but source calibrator could not be trained")
        print(
            json.dumps(
                {
                    "stage": "source_prescore_calibrator_trained",
                    "calibration": source_prescore_calibrator["calibration"],
                    "subspace_rank": source_prescore_calibrator["subspace_rank"],
                    "eta": source_prescore_calibrator["eta"],
                    "readout_gamma": source_prescore_calibrator["readout_gamma"],
                    "fp_weight": source_prescore_calibrator["fp_weight"],
                    "train_points_per_class": source_prescore_calibrator["train_points_per_class"],
                    "source_pair_rank_accuracy": source_prescore_calibrator["source_pair_rank_accuracy"],
                    "source_base_pair_rank_accuracy": source_prescore_calibrator["source_base_pair_rank_accuracy"],
                    "source_delta_fp_mean": source_prescore_calibrator["source_delta_fp_mean"],
                    "source_delta_def_mean": source_prescore_calibrator["source_delta_def_mean"],
                }
            ),
            flush=True,
        )
    if any(mode == "branch_calibrated" for mode in insert_modes):
        base_text_pair = learned_text_pair(model, prompt_learner)
        axis = F.normalize(base_text_pair[1] - base_text_pair[0], dim=0)
        source_branch_calibrators = {}
        for branch_name, host_score_key in (("branch1", "branch1_score"), ("branch2", "branch2_score")):
            calibrator = train_subspace_host_residual_calibrator(
                banks=banks,
                axis=axis,
                tau=args.tau,
                bank_chunk=args.bank_chunk,
                max_train_points=args.prescore_calib_max_train_points,
                epochs=args.prescore_calib_epochs,
                lr=args.prescore_calib_lr,
                rank_margin=args.prescore_calib_rank_margin,
                rank_temp=args.prescore_calib_rank_temp,
                eta_init=args.prescore_calib_alpha_init,
                eta_max=args.prescore_calib_alpha_max,
                eta_reg=args.prescore_calib_alpha_reg,
                fp_weight_init=args.prescore_calib_fp_weight_init,
                learn_fp_weight=args.prescore_calib_learn_fp_weight,
                hardpair_frac=args.prescore_calib_hardpair_frac,
                subspace_rank=args.prescore_calib_subspace_rank,
                subspace_basis_control=args.prescore_calib_subspace_basis_control,
                pair_label_control=args.prescore_calib_pair_label_control,
                preserve_host_margin_weight=args.prescore_calib_preserve_host_margin_weight,
                seed=args.seed + (101 if branch_name == "branch1" else 202),
                device=device,
                host_score_key=host_score_key,
            )
            if calibrator is None:
                raise RuntimeError(f"branch_calibrated requested, but {branch_name} calibrator could not be trained")
            source_branch_calibrators[branch_name] = calibrator
            print(
                json.dumps(
                    {
                        "stage": "source_branch_calibrator_trained",
                        "branch": branch_name,
                        "calibration": calibrator["calibration"],
                        "host_score_source": calibrator["host_score_source"],
                        "subspace_rank": calibrator["subspace_rank"],
                        "eta": calibrator["eta"],
                        "readout_gamma": calibrator["readout_gamma"],
                        "fp_weight": calibrator["fp_weight"],
                        "train_points_per_class": calibrator["train_points_per_class"],
                        "source_pair_rank_accuracy": calibrator["source_pair_rank_accuracy"],
                        "source_base_pair_rank_accuracy": calibrator["source_base_pair_rank_accuracy"],
                    }
                ),
                flush=True,
            )
    source_plugin = None
    source_plugin_cv = None
    if args.plugin_epochs > 0 or any(mode.startswith("plugin_") for mode in insert_modes):
        base_text_pair = learned_text_pair(model, prompt_learner)
        axis = F.normalize(base_text_pair[1] - base_text_pair[0], dim=0)
        source_plugin = train_source_bank_plugin(
            banks=banks,
            axis=axis,
            tau=args.tau,
            bank_chunk=args.bank_chunk,
            max_train_points=args.plugin_max_train_points,
            epochs=args.plugin_epochs,
            lr=args.plugin_lr,
            loss_type=args.plugin_loss,
            feature_space=args.plugin_feature_space,
            fp_weight=args.fp_weight,
            margin=args.plugin_rank_margin,
            rank_temp=args.plugin_rank_temp,
            contrast_weight=args.plugin_contrast_weight,
            contrast_margin=args.plugin_contrast_margin,
            hidden_dim=args.plugin_hidden_dim,
            seed=args.seed,
            device=device,
        )
        if source_plugin is None:
            raise RuntimeError("Plugin insert mode requested, but source plugin could not be trained")
        print(
            json.dumps(
                {
                    "stage": "source_plugin_trained",
                    "plugin_loss": source_plugin["loss_type"],
                    "plugin_feature_space": source_plugin["feature_space"],
                    "train_points_per_class": source_plugin["train_points_per_class"],
                    "train_accuracy": source_plugin["train_accuracy"],
                    "source_pair_rank_accuracy": source_plugin["source_pair_rank_accuracy"],
                    "source_defect_logit_mean": source_plugin["source_defect_logit_mean"],
                    "source_hardfp_logit_mean": source_plugin["source_hardfp_logit_mean"],
                    "learned_defect_weight": source_plugin.get("learned_defect_weight"),
                    "learned_hardfp_weight": source_plugin.get("learned_hardfp_weight"),
                }
            ),
            flush=True,
        )
        source_plugin_cv = evaluate_source_classheldout_rule(
            banks=banks,
            axis=axis,
            tau=args.tau,
            bank_chunk=args.bank_chunk,
            max_train_points=args.plugin_max_train_points,
            max_val_points=args.plugin_source_cv_max_val_points,
            epochs=args.plugin_epochs,
            lr=args.plugin_lr,
            loss_type=args.plugin_loss,
            feature_space=args.plugin_feature_space,
            fp_weight=args.fp_weight,
            margin=args.plugin_rank_margin,
            rank_temp=args.plugin_rank_temp,
            contrast_weight=args.plugin_contrast_weight,
            contrast_margin=args.plugin_contrast_margin,
            hidden_dim=args.plugin_hidden_dim,
            folds=args.plugin_source_cv_folds,
            seed=args.seed,
            device=device,
        )
        if source_plugin_cv is not None:
            print(json.dumps({"stage": "source_plugin_classheldout_cv", **source_plugin_cv}), flush=True)
    result = evaluate_target(
        model=model,
        prompt_learner=prompt_learner,
        target_root=cfg["target_root"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        features_list=args.features_list,
        sigma=args.sigma,
        dap_token_mode=args.dap_token_mode,
        dpam_layer=prompt_load_info["effective_dpam_layer"],
        banks=banks,
        tau=args.tau,
        fp_weight=args.fp_weight,
        alphas=alphas,
        insert_modes=insert_modes,
        image_score_topk=args.image_score_topk,
        bank_chunk=args.bank_chunk,
        aupro_max_step=args.aupro_max_step,
        ours_image_score_mode=args.ours_image_score_mode,
        residual_gate_quantile=args.residual_gate_quantile,
        residual_gate_temp=args.residual_gate_temp,
        source_plugin=source_plugin,
        source_prescore_calibrator=source_prescore_calibrator,
        source_branch_calibrators=source_branch_calibrators,
        plugin_gate_topk_frac=args.plugin_gate_topk_frac,
        plugin_gate_sharpness=args.plugin_gate_sharpness,
        plugin_ambiguity_temp=args.plugin_ambiguity_temp,
        rectifier_gate_mode=args.rectifier_gate_mode,
        rectifier_gate_sharpness=args.rectifier_gate_sharpness,
        rectifier_direction_mode=args.rectifier_direction_mode,
        target_limit_per_class=args.target_limit_per_class if args.target_limit_per_class > 0 else None,
        seed=args.seed,
        target_class_name=args.target_class_name or None,
        report_group_separability=args.report_group_separability,
        separability_hard_frac=args.separability_hard_frac,
    )
    result.update(
        {
            "preset": args.preset,
            "source_dataset": cfg["source_dataset"],
            "target_dataset": cfg["target_dataset"],
            "source_root": cfg["source_root"],
            "target_root": cfg["target_root"],
            "checkpoint_path": cfg["checkpoint_path"],
            "model_name": args.model_name,
            "prompt_load_mode": args.prompt_load_mode,
            "prompt_load_info": prompt_load_info,
            "bank_protocol": BANK_PROTOCOL_VERSION,
            "bank": {
                "num_fp": int(banks["num_fp"]),
                "num_defect": int(banks["num_defect"]),
                "max_fp_per_class": args.max_fp_per_class,
                "max_defect_per_class": args.max_defect_per_class,
                "max_fp_total": args.max_fp_total,
                "max_defect_total": args.max_defect_total,
                "normal_topk_frac": args.normal_topk_frac,
                "source_fp_mode": args.source_fp_mode,
            },
            "tau": args.tau,
            "fp_weight": args.fp_weight,
            "alphas": alphas,
            "insert_modes": insert_modes,
            "target_class_name": args.target_class_name,
            "target_limit_per_class": args.target_limit_per_class,
            "features_list": list(args.features_list),
            "sigma": args.sigma,
            "dap_token_mode": args.dap_token_mode,
            "aupro_max_step": args.aupro_max_step,
            "ours_image_score_mode": args.ours_image_score_mode,
            "residual_gate_quantile": args.residual_gate_quantile,
            "residual_gate_temp": args.residual_gate_temp,
            "rectifier_gate_mode": args.rectifier_gate_mode,
            "rectifier_gate_sharpness": args.rectifier_gate_sharpness,
            "rectifier_direction_mode": args.rectifier_direction_mode,
            "source_prescore_calibrator": None
            if source_prescore_calibrator is None
            else {
                "calibration": source_prescore_calibrator["calibration"],
                "subspace_rank": source_prescore_calibrator["subspace_rank"],
                "eta": source_prescore_calibrator["eta"],
                "readout_gamma": source_prescore_calibrator["readout_gamma"],
                "fp_weight": source_prescore_calibrator["fp_weight"],
                "train_points_per_class": source_prescore_calibrator["train_points_per_class"],
                "source_pair_rank_accuracy": source_prescore_calibrator["source_pair_rank_accuracy"],
                "source_base_pair_rank_accuracy": source_prescore_calibrator["source_base_pair_rank_accuracy"],
                "source_delta_fp_mean": source_prescore_calibrator["source_delta_fp_mean"],
                "source_delta_def_mean": source_prescore_calibrator["source_delta_def_mean"],
                "prescore_calib_epochs": args.prescore_calib_epochs,
                "prescore_calib_lr": args.prescore_calib_lr,
                "prescore_calib_max_train_points": args.prescore_calib_max_train_points,
                "prescore_calib_rank_margin": args.prescore_calib_rank_margin,
                "prescore_calib_rank_temp": args.prescore_calib_rank_temp,
                "prescore_calib_alpha_init": args.prescore_calib_alpha_init,
                "prescore_calib_alpha_max": args.prescore_calib_alpha_max,
                "prescore_calib_alpha_reg": args.prescore_calib_alpha_reg,
                "prescore_calib_fp_weight_init": args.prescore_calib_fp_weight_init,
                "prescore_calib_learn_fp_weight": bool(args.prescore_calib_learn_fp_weight),
                "prescore_calib_hardpair_frac": args.prescore_calib_hardpair_frac,
                "prescore_calib_subspace_basis_control": args.prescore_calib_subspace_basis_control,
                "prescore_calib_pair_label_control": args.prescore_calib_pair_label_control,
                "prescore_calib_preserve_host_margin_weight": args.prescore_calib_preserve_host_margin_weight,
            },
            "source_branch_calibrators": None
            if source_branch_calibrators is None
            else {
                name: {
                    "calibration": cal["calibration"],
                    "host_score_source": cal["host_score_source"],
                    "subspace_rank": cal["subspace_rank"],
                    "eta": cal["eta"],
                    "readout_gamma": cal["readout_gamma"],
                    "fp_weight": cal["fp_weight"],
                    "train_points_per_class": cal["train_points_per_class"],
                    "source_pair_rank_accuracy": cal["source_pair_rank_accuracy"],
                    "source_base_pair_rank_accuracy": cal["source_base_pair_rank_accuracy"],
                    "source_delta_fp_mean": cal["source_delta_fp_mean"],
                    "source_delta_def_mean": cal["source_delta_def_mean"],
                    "subspace_basis_control": cal.get("subspace_basis_control"),
                    "pair_label_control": cal.get("pair_label_control"),
                }
                for name, cal in source_branch_calibrators.items()
            },
            "source_plugin": None
            if source_plugin is None
            else {
                "calibration": source_plugin["calibration"],
                "feature_names": source_plugin["feature_names"],
                "train_points_per_class": source_plugin["train_points_per_class"],
                "train_accuracy": source_plugin["train_accuracy"],
                "source_defect_logit_mean": source_plugin["source_defect_logit_mean"],
                "source_hardfp_logit_mean": source_plugin["source_hardfp_logit_mean"],
                "plugin_epochs": args.plugin_epochs,
                "plugin_lr": args.plugin_lr,
                "plugin_max_train_points": args.plugin_max_train_points,
                "plugin_source_cv_folds": args.plugin_source_cv_folds,
                "plugin_source_cv_max_val_points": args.plugin_source_cv_max_val_points,
                "source_classheldout_cv": source_plugin_cv,
                "plugin_loss": args.plugin_loss,
                "plugin_feature_space": args.plugin_feature_space,
                "source_pair_rank_accuracy": source_plugin["source_pair_rank_accuracy"],
                "plugin_rank_margin": args.plugin_rank_margin,
                "plugin_rank_temp": args.plugin_rank_temp,
                "plugin_contrast_weight": args.plugin_contrast_weight,
                "plugin_contrast_margin": args.plugin_contrast_margin,
                "plugin_hidden_dim": args.plugin_hidden_dim,
                "learned_defect_weight": source_plugin.get("learned_defect_weight"),
                "learned_hardfp_weight": source_plugin.get("learned_hardfp_weight"),
                "plugin_gate_topk_frac": args.plugin_gate_topk_frac,
                "plugin_gate_sharpness": args.plugin_gate_sharpness,
                "plugin_ambiguity_temp": args.plugin_ambiguity_temp,
            },
        }
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "summary.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2), flush=True)
    print(f"[OK] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
