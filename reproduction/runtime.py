"""Prepare an isolated Linux research runtime from verified public inputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys

from .checkpoint_download import digest_file
from .datasets import prepare_dataset
from .execution import verify_execution_recipes
from .source import unpack_source

ORIGINAL = "/mnt/data/pilab-kingjinyoung/ADPretrain-CLIP"
DATA_ROOTS = {
    "mvtec": ["/mnt/data/pilab-kingjinyoung/ijepa/data/mvtec"],
    "visa": [ORIGINAL + "/neurips2026/data/VisA_pytorch_official/1cls"],
    "mpdd": ["/mnt/data/MPDD", "/mnt/data/pilab-kingjinyoung/ADPretrain/datasets/MPDD"],
    "btad": [ORIGINAL + "/neurips2026/data/BTAD_official"],
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_run(root: Path, recipe_id: str, destination: Path, object_roots: list[Path], dataset_config: Path) -> dict:
    if sys.platform != "linux":
        raise RuntimeError("Research runtime preparation currently requires Linux; artifact verification works on Windows too")
    import torch  # CPU-only bank metadata relocation; no model/GPU initialization.

    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError(f"Run workspace must be new: {destination}")
    verify_execution_recipes(root)
    recipes = {r["id"]: r for r in read_json(root / "execution-recipes.json")["recipes"]}
    recipe = recipes[recipe_id]
    source_name, target_name = recipe["transfer"].split("2", 1)
    if source_name not in DATA_ROOTS or target_name not in DATA_ROOTS:
        raise ValueError("This dataset protocol is not yet published; MVTec AD 2 remains required")
    datasets = read_json(dataset_config)
    for name in {source_name, target_name}:
        if name not in datasets:
            raise ValueError(f"Missing dataset roots for {name}")
    kind = "raw" if recipe["host"] in {"RawCLIP", "RawImageBind"} else "host"
    bank_catalog = read_json(root / f"{kind}-source-banks.json")
    bank_binding = next(r for r in bank_catalog["bindings"] if r["recipe"] == recipe_id)
    backbone_catalog = read_json(root / "backbones.json")
    backbone_binding = next(r for r in backbone_catalog["bindings"] if r["recipe"] == recipe_id)
    backbone = next(a for a in backbone_catalog["artifacts"] if a["sha256"] == backbone_binding["sha256"])
    checkpoint_catalog = read_json(root / "host-checkpoints.json")
    checkpoint_binding = next((r for r in checkpoint_catalog["bindings"] if r["recipe"] == recipe_id), None)
    all_assets = {a["sha256"]: a for catalog in (bank_catalog, backbone_catalog, checkpoint_catalog) for a in catalog["artifacts"]}
    needed = {backbone["sha256"]} | {a["sha256"] for a in bank_binding["source_bank_assets"]}
    if checkpoint_binding:
        needed.update(a["sha256"] for a in checkpoint_binding["checkpoint_assets"])
    objects, evidence = {}, []
    for sha in sorted(needed):
        asset = all_assets[sha]
        path = next((p / "objects" / sha for p in object_roots if (p / "objects" / sha).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Required artifact objects/{sha}")
        if path.stat().st_size != asset["bytes"] or digest_file(path) != sha:
            raise ValueError(f"Artifact hash/size mismatch: {sha}")
        objects[sha] = path.resolve()
        evidence.append({"sha256": sha, "bytes": asset["bytes"], "path": str(path.resolve())})
    destination.mkdir(parents=True, exist_ok=False)
    runtime = destination / "source"
    unpack_source(root / "source.zip", root / "source-manifest.json", runtime)
    mapping = {
        "/mnt/data/pilab-kingjinyoung/ted-reproduce-20261008/snapshot": str(runtime),
        ORIGINAL: str(runtime),
        "/mnt/data/pilab-kingjinyoung/ADPretrain": str(runtime / "ADPretrain"),
    }
    prepared_data = {}
    for name in sorted({source_name, target_name}):
        inputs = datasets[name]
        # Keep source basenames used in historical bank filename construction.
        prepared = destination / "data" / name / Path(DATA_ROOTS[name][0]).name
        report = prepare_dataset(root / "datasets" / name, Path(inputs["images"]), Path(inputs["masks"]) if inputs.get("masks") else None, prepared)
        if report["file_errors"]:
            raise ValueError(f"Dataset verification failed: {name}: {report['error_examples']}")
        prepared_data[name] = report
        for original in DATA_ROOTS[name]:
            mapping[original] = str(prepared)
    clip_cache = destination / "clip-cache"
    clip_cache.mkdir()
    backbone_path = clip_cache / backbone["filename"]
    backbone_path.symlink_to(objects[backbone["sha256"]])
    for original in backbone["archived_paths"]:
        mapping[original] = str(backbone_path)
    # URLs are retained; vendored downloaders find the verified cached filename.
    for original in ("~/.cache/clip", "/home/jinyoung/.cache/clip", "/mnt/data/hf-cache/anomalyclip",
                     "/mnt/data/pilab-kingjinyoung/ADPretrain/FAPrompt/.cache/clip",
                     "/mnt/data/pilab-kingjinyoung/.cache/clip"):
        mapping[original] = str(clip_cache)
    if "vit_h14" in backbone["id"]:
        hub = destination / "hf-cache" / "hub" / "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
        revision = "1c2b8495b28150b8a4922ee1c8edee224c284c0c"
        snapshot = hub / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / backbone["filename"]).symlink_to(backbone_path)
        (hub / "refs").mkdir()
        (hub / "refs/main").write_text(revision, encoding="utf-8")
        mapping["/mnt/data/hf-cache/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/blobs/" + backbone["sha256"]] = str(backbone_path)
    if checkpoint_binding:
        checkpoint_dir = destination / "checkpoints"
        checkpoint_dir.mkdir()
        for item in checkpoint_binding["checkpoint_assets"]:
            path = checkpoint_dir / item["checkpoint_filename"]
            path.symlink_to(objects[item["sha256"]])
            mapping[item["archived_path"]] = str(checkpoint_dir if item["role"] == "ckpt_dir" else path)
            for original in all_assets[item["sha256"]]["archived_paths"]:
                mapping[original] = str(path)
    replacements = sorted(mapping.items(), key=lambda pair: -len(pair[0]))
    # One regex pass prevents replacement output from being matched again.
    pattern = re.compile("|".join(re.escape(old) for old, _ in replacements))
    def relocate(value):
        if isinstance(value, str):
            return pattern.sub(lambda match: mapping[match.group()], value)
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, tuple):
            return tuple(relocate(item) for item in value)
        return value
    changes = []
    for path in runtime.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml", ".sh"}:
            before = path.read_text(encoding="utf-8")
            after = relocate(before)
            if before != after:
                prior = digest_file(path)
                path.write_text(after, encoding="utf-8")
                changes.append({"path": str(path.relative_to(runtime)), "before_sha256": prior, "after_sha256": digest_file(path)})
    bank_dir = runtime / "neurips2026/results/bank_cache"
    bank_dir.mkdir(parents=True, exist_ok=True)
    bank_paths, bank_changes = {}, []
    for item in bank_binding["source_bank_assets"]:
        original = objects[item["sha256"]]
        bank = torch.load(original, map_location="cpu", weights_only=True)
        transformed = relocate(bank)
        name = Path(item["archived_path"]).name
        if name.startswith("faprompt_sourcebank_") and bank.get("source_root"):
            new_root = Path(transformed["source_root"])
            tag = hashlib.sha1(str(new_root.resolve()).encode()).hexdigest()[:10]
            name = re.sub(r"^(faprompt_sourcebank_)[^_]+_[0-9a-f]{10}_", lambda m: m.group(1) + new_root.name + "_" + tag + "_", name)
        output = bank_dir / name
        if output.exists():
            raise FileExistsError(f"Bank name collision: {name}")
        torch.save(transformed, output)
        bank_paths[item["sha256"]] = output
        bank_changes.append({"original_sha256": item["sha256"], "derived_sha256": digest_file(output), "path": str(output), "transformation": "path strings only; tensor objects preserved"})
    argv = []
    for arg in recipe["argv"]:
        if arg not in recipe["path_bindings"]:
            argv.append(arg)
            continue
        binding = recipe["path_bindings"][arg]
        kind = binding["kind"]
        if kind == "new_output_directory":
            value = destination / "results"
        elif kind in {"prepared_source_dataset", "prepared_target_dataset"}:
            name = source_name if kind == "prepared_source_dataset" else target_name
            value = Path(prepared_data[name]["prepared_metadata"]).parent
        elif kind == "source_file":
            value = runtime / binding["path"]
        elif binding["sha256"] in bank_paths:
            value = bank_paths[binding["sha256"]]
        elif binding["sha256"] == backbone["sha256"]:
            value = backbone_path
        else:
            value = objects[binding["sha256"]]
        argv.append(str(value))
    environment = {"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HOME": str(destination / "hf-cache"),
                   "HF_HUB_OFFLINE": "1", "FAPROMPT_CACHE_DIR": str(clip_cache), "ANOMALYCLIP_CACHE_DIR": str(clip_cache),
                   "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                   "NO_ALBUMENTATIONS_UPDATE": "1"}
    report = {"recipe": recipe_id, "cwd": str(runtime), "evaluator": str(runtime / recipe["evaluator"]["path"]),
              "argv": argv, "environment": environment, "verified_objects": evidence, "source_path_changes": changes,
              "bank_path_changes": bank_changes, "datasets": prepared_data, "fresh_gpu_benchmark": False,
              "status": "prepared_requires_fresh_execution"}
    (destination / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
