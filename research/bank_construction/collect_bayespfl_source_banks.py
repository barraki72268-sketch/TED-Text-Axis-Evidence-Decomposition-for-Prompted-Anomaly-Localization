from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path("/mnt/data/pilab-kingjinyoung/ted-reproduce-20261008/snapshot")
SCRIPT_ROOT = ROOT / "neurips2026" / "scripts"
BAYES_ROOT = ROOT / "neurips2026" / "Bayes-PFL"
OFFICIAL_VISA_ROOT = ROOT / "neurips2026" / "data" / "VisA_pytorch_official" / "1cls"
os.chdir(BAYES_ROOT)
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(BAYES_ROOT) not in sys.path:
    sys.path.insert(0, str(BAYES_ROOT))

from collect_bayespfl_map_separability import build_models  # noqa: E402
from models.model_CLIP import tokenize  # noqa: E402
from test import _transform_test, setup_seed  # noqa: E402


def resolve_data_root(root: Path) -> Path:
    root = root.expanduser()
    if root.is_absolute():
        return root.resolve()
    repo_relative = (ROOT / root).resolve()
    if repo_relative.exists():
        return repo_relative
    return root.resolve()


def assert_official_visa_root(root: Path, dataset_name: str) -> Path:
    resolved = resolve_data_root(root)
    if "visa" not in dataset_name.lower():
        return resolved
    official = OFFICIAL_VISA_ROOT.resolve()
    if resolved != official:
        raise ValueError(
            "VisA must use the official VisA root. "
            f"Got {resolved}; expected {official}. "
            "Do not use Bayes-PFL/dataset/mvisa/data or MMAD mirrors."
        )
    return resolved


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


class MetaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        dataset_name: str,
        split: str,
        image_transform,
        mask_transform,
        keep_good: bool | None = None,
        class_names: list[str] | None = None,
        exclude_class: str | None = None,
    ) -> None:
        self.root = assert_official_visa_root(Path(root), dataset_name)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        meta_path = self.root / f"meta_{dataset_name}.json"
        if not meta_path.exists():
            meta_path = self.root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Could not find meta file for {dataset_name}: "
                f"{self.root / f'meta_{dataset_name}.json'} or {self.root / 'meta.json'}"
            )
        meta = json.loads(meta_path.read_text())[split]
        if class_names is not None:
            meta = {name: meta[name] for name in class_names}
        self.obj_list = sorted(meta.keys())
        self.items = []
        for cls_name in self.obj_list:
            if exclude_class is not None and cls_name == exclude_class:
                continue
            for item in meta[cls_name]:
                anomaly = int(item["anomaly"])
                if keep_good is True and anomaly != 0:
                    continue
                if keep_good is False and anomaly == 0:
                    continue
                self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = Image.open(self.root / item["img_path"]).convert("RGB")
        anomaly = int(item["anomaly"])
        mask_path = item.get("mask_path", "")
        if anomaly == 0 or not mask_path:
            mask = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8), mode="L")
        else:
            raw_mask = np.array(Image.open(self.root / mask_path).convert("L")) > 0
            mask = Image.fromarray(raw_mask.astype(np.uint8) * 255, mode="L")
        return {
            "img": self.image_transform(image),
            "img_mask": self.mask_transform(mask),
            "cls_name": item["cls_name"],
            "anomaly": anomaly,
            "img_path": item["img_path"],
        }


def build_mask_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def default_bank_cache_path(args) -> Path:
    cfg = {
        "dataset": args.dataset,
        "data_path": str(args.data_path),
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_load_mode": getattr(args, "checkpoint_load_mode", "strict"),
        "config_path": str(args.config_path),
        "pretrained_path": str(args.pretrained_path),
        "image_size": int(args.image_size),
        "features_list": list(args.features_list),
        "hard_frac": float(args.hard_frac),
        "max_good_per_class": int(args.max_good_per_class),
        "max_defect_per_class": int(args.max_defect_per_class),
        "max_bank_per_layer": int(args.max_bank_per_layer),
        "max_fp_per_image": int(args.max_fp_per_image),
        "max_defect_per_image": int(args.max_defect_per_image),
        "source_class": getattr(args, "source_class", None),
        "exclude_class": args.exclude_class,
    }
    digest = hashlib.md5(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    ckpt_tag = Path(args.checkpoint_path).stem
    return ROOT / "neurips2026" / "results" / "bank_cache" / f"bayespfl_{args.dataset}_{ckpt_tag}_{digest}.pt"


def infer_local_branch(
    args,
    device,
    model_clip,
    pfl_text_encoder,
    my_model,
    image: torch.Tensor,
    cls_name: str,
):
    with torch.no_grad():
        image_features, _, patch_tokens = model_clip.encode_image(image, args.features_list)
        text_embeddings, _ = my_model.forward_ensemble(
            pfl_text_encoder,
            image_features,
            patch_tokens,
            [cls_name],
            device,
            tokenize,
            mode="test",
        )
        num_hyp = args.prompt_num * args.sample_num
        outputs = []
        for layer_idx, patch_token in enumerate(patch_tokens):
            text_embeddings_update, dense_feature = my_model.RCA(text_embeddings, patch_token.clone(), layer_idx)
            dense_feature = F.normalize(dense_feature[0].float(), dim=-1)
            logits = my_model.temperature_pixel.exp() * dense_feature @ text_embeddings_update[0].float().t()

            anomaly_scores = []
            for hyp_idx in range(num_hyp):
                pair = torch.stack([logits[:, hyp_idx], logits[:, hyp_idx + num_hyp]], dim=1)
                anomaly_scores.append(pair.softmax(dim=1)[:, 1])
            layer_score = torch.stack(anomaly_scores, dim=0).mean(dim=0)
            outputs.append(
                {
                    "dense_feature": dense_feature,
                    "layer_score": layer_score,
                }
            )
    return outputs


def collect_source_patch_banks(args):
    setup_seed(args.seed)
    device, model_clip, pfl_text_encoder, my_model = build_models(args)

    image_transform = _transform_test(args.image_size)
    mask_transform = build_mask_transform(args.image_size)
    good_ds = MetaDataset(
        root=args.data_path,
        dataset_name=args.dataset,
        split="train",
        image_transform=image_transform,
        mask_transform=mask_transform,
        keep_good=True,
        class_names=[args.source_class] if getattr(args, "source_class", None) else None,
        exclude_class=args.exclude_class,
    )
    defect_ds = MetaDataset(
        root=args.data_path,
        dataset_name=args.dataset,
        split="test",
        image_transform=image_transform,
        mask_transform=mask_transform,
        keep_good=False,
        class_names=[args.source_class] if getattr(args, "source_class", None) else None,
        exclude_class=args.exclude_class,
    )

    fp_chunks = {idx: [] for idx in range(len(args.features_list))}
    def_chunks = {idx: [] for idx in range(len(args.features_list))}
    per_good = {cls_name: 0 for cls_name in good_ds.obj_list}
    per_defect = {cls_name: 0 for cls_name in defect_ds.obj_list}

    for img_idx in range(len(good_ds)):
        item = good_ds[img_idx]
        cls_name = item["cls_name"]
        if per_good[cls_name] >= args.max_good_per_class:
            continue
        per_good[cls_name] += 1
        image = item["img"].unsqueeze(0).to(device)
        outputs = infer_local_branch(args, device, model_clip, pfl_text_encoder, my_model, image, cls_name)
        for layer_idx, out in enumerate(outputs):
            layer_score = out["layer_score"]
            dense_feature = out["dense_feature"]
            k = max(1, int(layer_score.numel() * args.hard_frac))
            top_idx = torch.topk(layer_score, k=k, largest=True).indices
            selected = dense_feature[top_idx]
            if selected.shape[0] > args.max_fp_per_image:
                selected = farthest_point_subsample(selected, args.max_fp_per_image)
            fp_chunks[layer_idx].append(selected.cpu())
        if (img_idx + 1) % 50 == 0:
            print(json.dumps({"stage": "bank_progress", "split": "good", "images_done": img_idx + 1}), flush=True)

    for img_idx in range(len(defect_ds)):
        item = defect_ds[img_idx]
        cls_name = item["cls_name"]
        if per_defect[cls_name] >= args.max_defect_per_class:
            continue
        per_defect[cls_name] += 1
        image = item["img"].unsqueeze(0).to(device)
        gt_mask = item["img_mask"].unsqueeze(0).to(device).float()
        outputs = infer_local_branch(args, device, model_clip, pfl_text_encoder, my_model, image, cls_name)
        for layer_idx, out in enumerate(outputs):
            dense_feature = out["dense_feature"]
            num_patches = dense_feature.shape[0]
            h = int(round(num_patches**0.5))
            w = num_patches // h
            gt_small = F.interpolate(gt_mask, size=(h, w), mode="nearest")[0, 0].reshape(-1) > 0.5
            if int(gt_small.sum().item()) == 0:
                continue
            selected = dense_feature[gt_small]
            if selected.shape[0] > args.max_defect_per_image:
                selected = farthest_point_subsample(selected, args.max_defect_per_image)
            def_chunks[layer_idx].append(selected.cpu())
        if (img_idx + 1) % 50 == 0:
            print(json.dumps({"stage": "bank_progress", "split": "defect", "images_done": img_idx + 1}), flush=True)

    fp_banks = {}
    def_banks = {}
    bank_stats = {}
    token_dim = int(args.text_width)
    for layer_idx in range(len(args.features_list)):
        fp_bank = torch.cat(fp_chunks[layer_idx], dim=0) if fp_chunks[layer_idx] else torch.empty((0, token_dim))
        def_bank = torch.cat(def_chunks[layer_idx], dim=0) if def_chunks[layer_idx] else torch.empty((0, token_dim))
        if fp_bank.shape[0] > args.max_bank_per_layer:
            fp_bank = farthest_point_subsample(fp_bank, args.max_bank_per_layer)
        if def_bank.shape[0] > args.max_bank_per_layer:
            def_bank = farthest_point_subsample(def_bank, args.max_bank_per_layer)
        fp_banks[layer_idx] = F.normalize(fp_bank.float(), dim=-1) if fp_bank.numel() else fp_bank.float()
        def_banks[layer_idx] = F.normalize(def_bank.float(), dim=-1) if def_bank.numel() else def_bank.float()
        bank_stats[layer_idx] = {
            "num_fp": int(fp_banks[layer_idx].shape[0]),
            "num_defect": int(def_banks[layer_idx].shape[0]),
        }
    return fp_banks, def_banks, bank_stats


def main():
    parser = argparse.ArgumentParser("Collect source feature banks for Bayes-PFL local branch")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(BAYES_ROOT / "dataset" / "mvisa" / "data"),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=str(ROOT / "neurips2026" / "results" / "bayespfl_strict_mvtec" / "epoch_post_15_1.pth"),
    )
    parser.add_argument("--checkpoint_load_mode", type=str, choices=["strict", "compatible"], default="strict")
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(BAYES_ROOT / "open_clip_local" / "model_configs" / "ViT-L-14-336.json"),
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="/mnt/data/hf-cache/anomalyclip/ViT-L-14-336px.pt",
    )
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--num_flows", type=int, default=10)
    parser.add_argument("--prompt_context_len", type=int, default=5)
    parser.add_argument("--prompt_num", type=int, default=3)
    parser.add_argument("--prompt_state_len", type=int, default=5)
    parser.add_argument("--sample_num", type=int, default=5)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=333)
    parser.add_argument("--hard_frac", type=float, default=0.01)
    parser.add_argument("--max_good_per_class", type=int, default=200)
    parser.add_argument("--max_defect_per_class", type=int, default=80)
    parser.add_argument("--max_bank_per_layer", type=int, default=512)
    parser.add_argument("--max_fp_per_image", type=int, default=64)
    parser.add_argument("--max_defect_per_image", type=int, default=128)
    parser.add_argument("--source_class", type=str, default=None)
    parser.add_argument("--exclude_class", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    cache_path = Path(args.save_path) if args.save_path else default_bank_cache_path(args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fp_banks, def_banks, bank_stats = collect_source_patch_banks(args)
    payload = {
        "fp_banks": {int(k): v.cpu() for k, v in fp_banks.items()},
        "def_banks": {int(k): v.cpu() for k, v in def_banks.items()},
        "bank_stats": bank_stats,
        "meta": {
            "host": "Bayes-PFL",
            "dataset": args.dataset,
            "data_path": str(args.data_path),
            "checkpoint_path": str(args.checkpoint_path),
            "checkpoint_load_mode": args.checkpoint_load_mode,
            "features_list": list(args.features_list),
            "hard_frac": float(args.hard_frac),
            "max_good_per_class": int(args.max_good_per_class),
            "max_defect_per_class": int(args.max_defect_per_class),
            "max_bank_per_layer": int(args.max_bank_per_layer),
            "max_fp_per_image": int(args.max_fp_per_image),
            "max_defect_per_image": int(args.max_defect_per_image),
            "source_class": args.source_class,
            "exclude_class": args.exclude_class,
        },
    }
    torch.save(payload, cache_path)
    print(
        json.dumps(
            {
                "stage": "bank_done",
                "save_path": str(cache_path),
                "bank_stats": bank_stats,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
