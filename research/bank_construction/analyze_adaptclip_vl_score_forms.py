from __future__ import annotations

import argparse
import json
import math
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
    compute_baseline_outputs,
    decision_axis,
    farthest_point_subsample,
    fusion_fun,
    logmeanexp_negative_sqdist_1d,
    resize_map,
    save_bank_cache,
    smooth_map,
    load_bank_cache,
)


def append_result(results: dict, mode: str, sample_id: str, cls_name: str, query_path: str, gt_mask: torch.Tensor, gt_anomaly: torch.Tensor, pr_map: torch.Tensor, pr_score: torch.Tensor):
    results[mode]["sample_ids"].append(sample_id)
    results[mode]["cls_names"].append(cls_name)
    results[mode]["query_paths"].append(query_path)
    results[mode]["gt_masks"].append(gt_mask.detach().cpu().int())
    results[mode]["pr_masks"].append(pr_map.detach().cpu())
    results[mode]["gt_anomalys"].append(gt_anomaly.detach().cpu())
    results[mode]["pr_anomalys"].append(pr_score.detach().cpu())


def vl_patch_bank_features(visual_learner, query_patch_feats) -> torch.Tensor:
    patch_feature = visual_learner.local_adater(query_patch_feats[-1])
    patch_feature = F.normalize(patch_feature[:, 1:, :].reshape(-1, patch_feature.shape[-1]).float(), dim=-1)
    return patch_feature


def branch_patch_bank_features(visual_learner, query_patch_feats, bank_feature_source: str) -> torch.Tensor:
    if bank_feature_source == "vl":
        return vl_patch_bank_features(visual_learner, query_patch_feats)
    if bank_feature_source == "tl":
        patch_feature = query_patch_feats[-1]
        return F.normalize(patch_feature[:, 1:, :].reshape(-1, patch_feature.shape[-1]).float(), dim=-1)
    raise ValueError(f"Unsupported bank_feature_source: {bank_feature_source}")


def collect_vl_source_banks(
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
    hard_frac: float,
    max_good_per_class: int,
    max_defect_per_class: int,
    max_bank_points: int,
    max_fp_per_image: int,
    max_defect_per_image: int,
    bank_mining_source: str = "vl",
    bank_feature_source: str = "vl",
):
    train_ds = build_dataset(source_root, image_size, source_dataset, "train", str(ROOT / "neurips2026" / "results" / "_tmp_vl"))
    test_ds = build_dataset(source_root, image_size, source_dataset, "test", str(ROOT / "neurips2026" / "results" / "_tmp_vl"))

    fp_chunks = []
    def_chunks = []
    per_good = {cls_name: 0 for cls_name in train_ds.obj_list}
    per_defect = {cls_name: 0 for cls_name in test_ds.obj_list}

    with torch.no_grad():
        for idx in range(len(train_ds)):
            item = train_ds[idx]
            cls_name = item["cls_name"]
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
            if bank_mining_source == "baseline":
                mining_map = outputs["baseline_map"][0].to(device)
            elif bank_mining_source == "tl":
                mining_map = smooth_map(outputs["local_tl_map"], sigma=sigma)[0].to(device)
            elif bank_mining_source == "vl":
                mining_map = smooth_map(outputs["local_vl_map"], sigma=sigma)[0].to(device)
            else:
                raise ValueError(f"Unsupported bank_mining_source: {bank_mining_source}")
            patch = branch_patch_bank_features(visual_learner, outputs["query_patch_feats"], bank_feature_source)
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
            fp_chunks.append(selected.cpu())

        for idx in range(len(test_ds)):
            item = test_ds[idx]
            cls_name = item["cls_name"]
            anomaly_flag = int(item["anomaly"])
            if anomaly_flag == 0:
                continue
            if per_defect[cls_name] >= max_defect_per_class:
                continue
            per_defect[cls_name] += 1

            image = item["img"].unsqueeze(0).to(device)
            gt_mask = item["img_mask"].unsqueeze(0).to(device).float()
            _, query_patch_feats = model.encode_image(image, features_list, DPAM_layer=dpam_layer)
            patch = branch_patch_bank_features(visual_learner, query_patch_feats, bank_feature_source)
            h = int(round(patch.shape[0] ** 0.5))
            gt_small = F.interpolate(gt_mask, size=(h, h), mode="nearest")[0, 0].reshape(-1) > 0.5
            if int(gt_small.sum().item()) == 0:
                continue
            selected = patch[gt_small]
            if selected.shape[0] > max_defect_per_image:
                selected = farthest_point_subsample(selected, max_defect_per_image)
            def_chunks.append(selected.cpu())

    feat_dim = fp_chunks[0].shape[-1] if fp_chunks else (def_chunks[0].shape[-1] if def_chunks else 768)
    fp_bank = torch.cat(fp_chunks, dim=0) if fp_chunks else torch.empty((0, feat_dim))
    def_bank = torch.cat(def_chunks, dim=0) if def_chunks else torch.empty((0, feat_dim))
    if fp_bank.shape[0] > max_bank_points:
        fp_bank = farthest_point_subsample(fp_bank, max_bank_points)
    if def_bank.shape[0] > max_bank_points:
        def_bank = farthest_point_subsample(def_bank, max_bank_points)
    fp_bank = F.normalize(fp_bank.float(), dim=-1) if fp_bank.numel() else fp_bank.float()
    def_bank = F.normalize(def_bank.float(), dim=-1) if def_bank.numel() else def_bank.float()
    bank_stats = {
        "num_fp": int(fp_bank.shape[0]),
        "num_defect": int(def_bank.shape[0]),
    }
    return fp_bank, def_bank, bank_stats


def get_or_collect_vl_banks(
    model,
    textual_learner,
    visual_learner,
    learned_text_features,
    source_root,
    source_dataset,
    image_size,
    features_list,
    dpam_layer,
    sigma,
    fusion_type,
    device,
    hard_frac,
    max_good_per_class,
    max_defect_per_class,
    max_bank_points,
    max_fp_per_image,
    max_defect_per_image,
    checkpoint_path,
    bank_cache_path,
    bank_mining_source: str = "vl",
    bank_feature_source: str = "vl",
):
    requested_meta = {
        "kind": "adaptclip_vl_branch_bank",
        "checkpoint_path": checkpoint_path,
        "source_root": source_root,
        "source_dataset": source_dataset,
        "image_size": image_size,
        "features_list": list(features_list),
        "sigma": sigma,
        "fusion_type": fusion_type,
        "hard_frac": hard_frac,
        "max_good_per_class": max_good_per_class,
        "max_defect_per_class": max_defect_per_class,
        "max_bank_points": max_bank_points,
        "max_fp_per_image": max_fp_per_image,
        "max_defect_per_image": max_defect_per_image,
        "bank_mining_source": bank_mining_source,
        "bank_feature_source": bank_feature_source,
    }
    cache_path = Path(bank_cache_path)
    if cache_path.exists():
        try:
            fp_banks, def_banks, bank_stats, cache_meta = load_bank_cache(cache_path)
            if cache_meta == requested_meta:
                return fp_banks[0], def_banks[0], bank_stats[0], cache_path, cache_meta
        except Exception:
            pass

    fp_bank, def_bank, bank_stats = collect_vl_source_banks(
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
        hard_frac=hard_frac,
        max_good_per_class=max_good_per_class,
        max_defect_per_class=max_defect_per_class,
        max_bank_points=max_bank_points,
        max_fp_per_image=max_fp_per_image,
        max_defect_per_image=max_defect_per_image,
        bank_mining_source=bank_mining_source,
        bank_feature_source=bank_feature_source,
    )
    save_bank_cache(
        cache_path,
        {0: fp_bank},
        {0: def_bank},
        {0: bank_stats},
        requested_meta,
    )
    return fp_bank, def_bank, bank_stats, cache_path, requested_meta


def main():
    parser = argparse.ArgumentParser("AdaptCLIP VL-branch score-form analysis")
    parser.add_argument("--source_root", type=str, required=True)
    parser.add_argument("--source_dataset", type=str, required=True)
    parser.add_argument("--target_root", type=str, required=True)
    parser.add_argument("--target_dataset", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--bank_cache_path", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, default="ViT-L/14@336px")
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
    parser.add_argument("--bank_feature_source", type=str, choices=["vl", "tl"], default="vl")
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--max_eval_images", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model, textual_learner, visual_learner, learned_text_features, dpam_layer = build_model(
        args.device,
        args.checkpoint_path,
        args.image_size,
        args.n_ctx,
        args.vl_reduction,
        args.pretrained_model,
    )
    vl_axis = decision_axis(textual_learner.static_text_features).to(args.device)

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
        bank_mining_source=args.bank_mining_source,
        bank_feature_source=args.bank_feature_source,
    )
    fp_coeff = fp_bank.to(args.device) @ vl_axis if fp_bank.numel() else torch.empty((0,), device=args.device)
    def_coeff = def_bank.to(args.device) @ vl_axis if def_bank.numel() else torch.empty((0,), device=args.device)

    dataset = build_dataset(
        args.target_root,
        args.image_size,
        args.target_dataset,
        "test",
        str(ROOT / "neurips2026" / "results" / "_tmp_vlscore"),
        class_name=args.class_name,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    evaluator = Evaluator("cpu", metrics=["I-AUROC", "I-AP", "P-AUROC", "P-AP"], sample_level=False)

    modes = [
        "baseline_txt",
        "vl_map",
        "tl_map",
        "vl_margin_fp03_sig",
        "vl_corr_margin_fp03_a03",
        "vl_mul_margin_fp03_b10",
        "refuse_tl_vl_corr_margin_fp03_a03",
        "refuse_tl_vl_mul_margin_fp03_b10",
    ]
    results = {
        mode: {"sample_ids": [], "gt_masks": [], "pr_masks": [], "cls_names": [], "gt_anomalys": [], "pr_anomalys": [], "query_paths": []}
        for mode in modes
    }

    with torch.no_grad():
        for idx, items in enumerate(loader, start=1):
            if args.max_eval_images is not None and idx > args.max_eval_images:
                break

            image = items["img"].to(args.device)
            cls_name = items["cls_name"][0]
            sample_id = f"{cls_name}_{idx}"
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

            vl_patch = vl_patch_bank_features(visual_learner, baseline["query_patch_feats"])
            coeff = vl_patch @ vl_axis
            q_fp = logmeanexp_negative_sqdist_1d(coeff, fp_coeff, tau=args.tau)
            q_def = logmeanexp_negative_sqdist_1d(coeff, def_coeff, tau=args.tau)
            h = int(round(vl_patch.shape[0] ** 0.5))
            margin03_map = (q_def - 0.3 * q_fp).reshape(1, 1, h, h)
            margin03_map = F.interpolate(margin03_map, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)[:, 0]
            margin03_map = smooth_map(margin03_map, sigma=args.sigma).to(args.device)
            margin03_sig = activate_margin_map(margin03_map, "sigmoid", 1.0)

            vl_corr = torch.clamp(vl_map + 0.3 * (margin03_sig - 0.5), 0.0, 1.0)
            vl_mul = torch.clamp(vl_map * (1.0 + (margin03_sig - 0.5)), 0.0, 1.0)
            refuse_corr = smooth_map(fusion_fun([raw_tl_map, vl_corr], fusion_type=args.fusion_type), sigma=args.sigma).to(args.device)
            refuse_corr = resize_map(refuse_corr, args.image_size)
            refuse_mul = smooth_map(fusion_fun([raw_tl_map, vl_mul], fusion_type=args.fusion_type), sigma=args.sigma).to(args.device)
            refuse_mul = resize_map(refuse_mul, args.image_size)

            score_map = {
                "baseline_txt": baseline["baseline_map"],
                "vl_map": vl_map,
                "tl_map": tl_map,
                "vl_margin_fp03_sig": margin03_sig,
                "vl_corr_margin_fp03_a03": vl_corr,
                "vl_mul_margin_fp03_b10": vl_mul,
                "refuse_tl_vl_corr_margin_fp03_a03": refuse_corr,
                "refuse_tl_vl_mul_margin_fp03_b10": refuse_mul,
            }

            global_vl = baseline["global_vl_score"]
            global_tl = baseline["global_tl_score"]
            for mode, map_tensor in score_map.items():
                anomaly_map_max, _ = torch.max(map_tensor.view(image.shape[0], -1), dim=1)
                img_score = fusion_fun([global_vl, global_tl, anomaly_map_max], fusion_type=args.fusion_type)
                append_result(results, mode, sample_id, cls_name, query_path, gt_mask, gt_anomaly, map_tensor, img_score)

    metrics = {}
    for mode in modes:
        result_mode = aggregate_results(results[mode])
        metric_results = evaluator.run(result_mode, args.class_name)
        metrics[mode] = {
            "image_auroc": float(metric_results["I-AUROC"]),
            "image_ap": float(metric_results["I-AP"]),
            "pixel_auroc": float(metric_results["P-AUROC"]),
            "pixel_ap": float(metric_results["P-AP"]),
        }

    output = {
        "class_name": args.class_name,
        "bank_stats": bank_stats,
        "bank_cache_path": str(bank_cache_path),
        "bank_cache_meta": bank_cache_meta,
        "metrics": metrics,
    }
    save_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
