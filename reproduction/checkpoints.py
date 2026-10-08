"""Verify historical host weights as bytes; never deserialize a checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def load_checkpoints(root: Path) -> dict:
    catalog = json.loads((root / "host-checkpoints.json").read_text(encoding="utf-8"))
    recipes = {r["id"]: r for r in json.loads((root / "recipes.json").read_text(encoding="utf-8"))}
    assets = {}
    for asset in catalog["artifacts"]:
        digest = asset["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or digest in assets:
            raise ValueError("Invalid or duplicate checkpoint digest")
        if asset["object_path"] != f"objects/{digest}" or asset["bytes"] <= 0:
            raise ValueError("Invalid checkpoint object specification")
        assets[digest] = asset
    seen = set()
    for binding in catalog["bindings"]:
        recipe = recipes[binding["recipe"]]
        if recipe["id"] in seen or binding["host"] != recipe["host"]:
            raise ValueError("Duplicate or inconsistent checkpoint recipe")
        seen.add(recipe["id"])
        reference = (root / "references" / recipe["reference"]).read_bytes()
        digest = hashlib.sha256(reference).hexdigest()
        if digest != recipe["reference_sha256"] or digest != binding["reference_sha256"]:
            raise ValueError("Checkpoint binding reference hash mismatch")
        def fields(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, str):
                        yield key, item
                    elif isinstance(item, (dict, list)):
                        yield from fields(item)
            elif isinstance(value, list):
                for item in value:
                    yield from fields(item)
        recorded = set(fields(json.loads(reference)))
        if not binding["checkpoint_assets"]:
            raise ValueError("Empty checkpoint recipe")
        for item in binding["checkpoint_assets"]:
            asset = assets[item["sha256"]]
            original = item["archived_path"]
            if (item["role"], original) not in recorded:
                raise ValueError("Checkpoint path is absent from archived reference")
            name = item["checkpoint_filename"]
            if not name or name in {".", ".."} or any(c in name for c in "/\\:"):
                raise ValueError("Unsafe checkpoint filename")
            exact = original if original.rsplit("/", 1)[-1] == name else original.rstrip("/") + "/" + name
            if exact not in asset["archived_paths"]:
                raise ValueError("Checkpoint filename does not match recorded artifact")
    required = {r["id"] for r in recipes.values() if r["host"] in
                {"AA-CLIP", "FAPrompt", "AdaptCLIP", "AdaCLIP", "BayesPFL"}}
    if seen != required or catalog["unresolved"]:
        raise ValueError("Incomplete adapted-host checkpoint bindings")
    return catalog


def verify_checkpoints(root: Path, directory: Path | None = None, recipe: str | None = None) -> dict:
    catalog = load_checkpoints(root)
    bindings = catalog["bindings"]
    if recipe is not None:
        bindings = [b for b in bindings if b["recipe"] == recipe]
        if not bindings:
            raise ValueError(f"No host-checkpoint binding for recipe: {recipe}")
    needed = {a["sha256"] for b in bindings for a in b["checkpoint_assets"]}
    assets = [a for a in catalog["artifacts"] if a["sha256"] in needed]
    errors, verified = [], 0
    if directory is not None:
        directory = directory.resolve()
        for asset in assets:
            path = (directory / asset["object_path"]).resolve()
            try:
                path.relative_to(directory)
                if path.stat().st_size != asset["bytes"]:
                    raise ValueError("size mismatch")
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != asset["sha256"]:
                    raise ValueError("SHA-256 mismatch")
                verified += 1
            except (OSError, ValueError) as exc:
                errors.append({"object": asset["object_path"], "error": str(exc)})
    return {"recipes_bound": len(bindings), "unique_checkpoint_files": len(assets),
            "checkpoint_bytes": sum(a["bytes"] for a in assets),
            "files_checked": directory is not None, "files_verified": verified,
            "file_errors": errors, "fresh_gpu_benchmark": False,
            "scope": catalog["scope"]}
