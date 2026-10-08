"""Validate traced evaluator arguments without loading a model or running code."""
from __future__ import annotations

import json
from pathlib import Path
import re


def verify_execution_recipes(root: Path) -> dict:
    catalog = json.loads((root / "execution-recipes.json").read_text(encoding="utf-8"))
    recipes = {r["id"]: r for r in json.loads((root / "recipes.json").read_text(encoding="utf-8"))}
    sources = {s["path"]: s for s in json.loads((root / "source-manifest.json").read_text(encoding="utf-8"))["files"]}
    seen, paths, evaluators = set(), 0, set()
    for row in catalog["recipes"]:
        reference = recipes[row["id"]]
        if row["id"] in seen:
            raise ValueError("Duplicate execution recipe")
        seen.add(row["id"])
        for key in ("host", "backbone", "transfer", "seed", "reference_sha256"):
            if row[key] != reference[key]:
                raise ValueError(f"Execution recipe {key} differs from reference inventory")
        evaluator = row["evaluator"]
        if evaluator["sha256"] != sources[evaluator["path"]]["sha256"]:
            raise ValueError("Execution evaluator source hash mismatch")
        evaluators.add(evaluator["path"])
        argv, bindings = row["argv"], row["path_bindings"]
        consumed = set()
        for index, arg in enumerate(argv):
            if arg.startswith("/"):
                raise ValueError("Unbound absolute execution argument")
            if arg.startswith("{"):
                binding = bindings[arg]
                if index == 0 or argv[index - 1] != binding["argument"]:
                    raise ValueError("Path binding does not match its argument")
                consumed.add(arg)
                if binding["kind"] == "artifact":
                    if not re.fullmatch(r"[0-9a-f]{64}", binding["sha256"] or "") or binding["bytes"] <= 0:
                        raise ValueError("Unresolved execution artifact")
                elif binding["kind"] == "source_file":
                    if binding["sha256"] != sources[binding["path"]]["sha256"]:
                        raise ValueError("Execution config source hash mismatch")
                elif binding["kind"] not in {"new_output_directory", "prepared_source_dataset", "prepared_target_dataset"}:
                    raise ValueError("Unknown execution path binding")
            flag = arg.split("=", 1)[0]
            if flag in {"--target_limit_per_class", "--target_class_name", "--class_name"}:
                value = arg.split("=", 1)[1] if "=" in arg else argv[index + 1]
                if value != ("0" if flag == "--target_limit_per_class" else ""):
                    raise ValueError("Main/host evaluation unexpectedly restricts target coverage")
        if consumed != set(bindings):
            raise ValueError("Unused execution path binding")
        paths += len(bindings)
    if seen != set(recipes):
        raise ValueError("Execution catalog does not cover the full reference inventory")
    return {"recipes_verified": len(seen), "evaluators": len(evaluators),
            "path_bindings": paths, "fresh_gpu_benchmark": False,
            "portable_execution_ready": False}
