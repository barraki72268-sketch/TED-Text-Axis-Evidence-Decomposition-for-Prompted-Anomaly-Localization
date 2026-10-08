"""Inspect and verify archived-reference recipes without importing PyTorch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .metrics import METRICS, aggregate, compare, extract

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    listing = sub.add_parser("list", help="List the exact archived recipe inventory")
    listing.add_argument("--host")
    sub.add_parser("verify-references", help="Verify every archived reference's byte hash and metric schema")
    sub.add_parser("verify-source", help="Verify the complete archived research source without executing it")
    sub.add_parser("verify-execution-recipes", help="Check traced arguments, source hashes, and explicit path bindings")
    banks = sub.add_parser("verify-source-banks", help="Verify recorded bank bindings and optionally a bank tar archive")
    banks.add_argument("--kind", choices=("host", "raw"), required=True)
    banks.add_argument("--archive", type=Path)
    checkpoints = sub.add_parser("verify-checkpoints", help="Verify host-weight bindings and optionally actual checkpoint bytes")
    checkpoints.add_argument("--directory", type=Path, help="Root containing objects/<sha256> files")
    checkpoints.add_argument("--recipe", help="Limit byte verification to one recipe's host weights")
    prepare_weights = sub.add_parser("prepare-checkpoints", help="Download pinned host weights and verify/extract into a new directory")
    prepare_weights.add_argument("destination", type=Path)
    prepare_weights.add_argument("--archive", type=Path, help="Use an already downloaded pinned tar archive")
    unpack = sub.add_parser("unpack-source", help="Verify and unpack source into a new directory; evaluation paths still need preparation")
    unpack.add_argument("destination", type=Path)
    for action in ("validate-dataset", "prepare-dataset"):
        dataset = sub.add_parser(action, help="Verify recorded image/mask bytes and optionally write portable metadata")
        dataset.add_argument("dataset", choices=("mvtec", "visa", "mpdd", "btad"))
        dataset.add_argument("--images", type=Path, required=True)
        dataset.add_argument("--masks", type=Path)
        if action == "prepare-dataset":
            dataset.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("compare", help="Compare a fresh summary with its hash-pinned historical reference")
    verify.add_argument("recipe")
    verify.add_argument("actual", type=Path)
    verify.add_argument("--output", type=Path)
    table = sub.add_parser("aggregate", help="Aggregate the recorded seed subset and compare with printed host-table cells")
    table.add_argument("--host", required=True)
    table.add_argument("--backbone", required=True)
    table.add_argument("--transfer", required=True)
    table.add_argument("--runs", type=Path, required=True, help="Directory containing <recipe-id>/summary.json")
    args = parser.parse_args()
    if args.action == "verify-source-banks":
        from .banks import verify_banks
        print(json.dumps(verify_banks(ROOT, args.kind, args.archive), indent=2))
        return 0
    if args.action == "verify-execution-recipes":
        from .execution import verify_execution_recipes
        print(json.dumps(verify_execution_recipes(ROOT), indent=2))
        return 0
    if args.action == "prepare-checkpoints":
        from .checkpoint_download import prepare_checkpoints
        result = prepare_checkpoints(ROOT, args.destination, args.archive)
        print(json.dumps(result, indent=2))
        return 1 if result["file_errors"] else 0
    if args.action == "verify-checkpoints":
        from .checkpoints import verify_checkpoints
        result = verify_checkpoints(ROOT, args.directory, args.recipe)
        print(json.dumps(result, indent=2))
        return 1 if result["file_errors"] else 0
    if args.action in {"validate-dataset", "prepare-dataset"}:
        from .datasets import prepare_dataset, validate_dataset
        protocol = ROOT / "datasets" / args.dataset
        if args.action == "prepare-dataset":
            result = prepare_dataset(protocol, args.images, args.masks, args.output)
        else:
            result = validate_dataset(protocol, args.images, args.masks)
        print(json.dumps(result, indent=2))
        return 0 if result["file_errors"] == 0 else 1
    if args.action in {"verify-source", "unpack-source"}:
        from .source import unpack_source, verify_source
        archive, manifest = ROOT / "source.zip", ROOT / "source-manifest.json"
        if args.action == "unpack-source":
            result = unpack_source(archive, manifest, args.destination)
        else:
            metadata = verify_source(archive, manifest)
            result = {"source_files_verified": len(metadata["files"]),
                      "archive_sha256": metadata["archive_sha256"], "evaluation_executed": False}
        print(json.dumps(result, indent=2))
        return 0
    recipes = {row["id"]: row for row in json.loads((ROOT / "recipes.json").read_text(encoding="utf-8"))}
    if args.action == "verify-references":
        count = 0
        for recipe in recipes.values():
            data = (ROOT / "references" / recipe["reference"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != recipe["reference_sha256"]:
                raise ValueError(f"Historical reference hash mismatch: {recipe['id']}")
            count += sum(len(row) for row in extract(json.loads(data), recipe["host"]).values())
        print(json.dumps({"references_verified": len(recipes), "archived_metric_values": count,
                          "fresh_gpu_benchmark": False}))
        return 0
    if args.action == "list":
        for row in recipes.values():
            if args.host is None or row["host"] == args.host:
                print(f"{row['id']}\t{row['host']}\t{row['transfer']}\tseed={row['seed']}\t{row['status']}")
        return 0
    if args.action == "aggregate":
        selected = [row for row in recipes.values() if (row["host"], row["backbone"], row["transfer"]) == (args.host, args.backbone, args.transfer)]
        if not selected or any("printed_paper_values" not in row for row in selected):
            parser.error("No printed adapted-host table group matches these exact names")
        selected.sort(key=lambda row: row["seed"])
        if len({row["seed"] for row in selected}) != len(selected):
            raise ValueError("Duplicate seeds in recipe group")
        missing = [row["id"] for row in selected if not (args.runs / row["id"] / "summary.json").is_file()]
        if missing:
            print(json.dumps({"status": "incomplete", "missing": missing}, indent=2))
            return 2
        convention = selected[0]["std_convention"]
        paper = selected[0]["printed_paper_values"]
        if any(row["std_convention"] != convention or row["printed_paper_values"] != paper for row in selected):
            raise ValueError("Inconsistent paper aggregation metadata")
        runs, evidence = [], []
        for row in selected:
            data = (args.runs / row["id"] / "summary.json").read_bytes()
            runs.append(extract(json.loads(data), row["host"]))
            evidence.append({"recipe": row["id"], "seed": row["seed"], "summary_sha256": hashlib.sha256(data).hexdigest()})
        result = aggregate(runs, convention)
        cells = []
        for method_index, method in enumerate(["Base", "OURS"]):
            for metric_index, metric in enumerate(METRICS):
                actual = result[method][metric]
                expected = paper[method_index][metric_index]
                cells.append({"method": method, "metric": metric, "actual": actual, "printed": expected,
                              "mean_matches_2dp": f'{actual["mean"]:.2f}' == f'{expected["mean"]:.2f}',
                              "std_matches_2dp": None if expected["std"] is None else f'{actual["std"]:.2f}' == f'{expected["std"]:.2f}'})
        matched = all(cell["mean_matches_2dp"] and cell["std_matches_2dp"] is not False for cell in cells)
        print(json.dumps({"status": "matched" if matched else "mismatch", "std_convention": convention,
                          "seed_evidence": evidence, "cells": cells}, indent=2))
        return 0 if matched else 1
    if args.recipe not in recipes:
        parser.error(f"Unknown recipe: {args.recipe}")
    recipe = recipes[args.recipe]
    reference = ROOT / "references" / recipe["reference"]
    reference_bytes = reference.read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != recipe["reference_sha256"]:
        raise ValueError("Historical reference hash mismatch")
    actual_bytes = args.actual.read_bytes()
    cells = compare(extract(json.loads(actual_bytes), recipe["host"]), extract(json.loads(reference_bytes), recipe["host"]))
    report = {"recipe": recipe["id"], "comparison": "fresh run versus archived per-seed summary; not a claim of agreement with every printed paper cell",
              "reference_sha256": recipe["reference_sha256"], "actual_sha256": hashlib.sha256(actual_bytes).hexdigest(),
              "all_match_2dp": all(cell["matches_printed_precision"] for cell in cells), "cells": cells}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Refuse accidental destruction of a previous verification record.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)
    return 0 if report["all_match_2dp"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
