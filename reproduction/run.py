"""Execute one prepared recipe and preserve successful or failed evidence."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from .checkpoint_download import digest_file
from .datasets import validate_dataset
from .metrics import compare, extract
from .coverage import validate_coverage
from .runtime import read_json
from .recipe_lookup import execution_recipe, reference_summary


def run_prepared(root: Path, workspace: Path, require_slurm: bool = False) -> dict:
    workspace = workspace.resolve()
    if require_slurm and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this command inside a Slurm allocation")
    plan_path = workspace / "run.json"
    plan = read_json(plan_path)
    recipe = execution_recipe(root, plan["recipe"])
    if Path(plan["evaluator"]).resolve() != workspace / "source" / recipe["evaluator"]["path"]:
        raise ValueError("Prepared evaluator path mismatch")
    if Path(plan["cwd"]).resolve() != workspace / "source":
        raise ValueError("Prepared working directory mismatch")
    if len(plan["argv"]) != len(recipe["argv"]):
        raise ValueError("Prepared argument count differs from recorded recipe")
    for expected, actual in zip(recipe["argv"], plan["argv"]):
        if expected not in recipe["path_bindings"] and expected != actual:
            raise ValueError("Prepared numerical/evaluation argument differs from recorded recipe")
        if expected in recipe["path_bindings"]:
            binding = recipe["path_bindings"][expected]
            kind = binding["kind"]
            source_name, target_name = recipe["transfer"].split("2", 1)
            if kind == "new_output_directory":
                required = workspace / "results"
            elif kind in {"prepared_source_dataset", "prepared_target_dataset"}:
                name = source_name if kind == "prepared_source_dataset" else target_name
                required = Path(plan["datasets"][name]["prepared_metadata"]).parent
            elif kind == "source_file":
                required = workspace / "source" / binding["path"]
            else:
                bank = next((b for b in plan["bank_path_changes"] if b["original_sha256"] == binding["sha256"]), None)
                artifact = next((a for a in plan["verified_objects"] if a["sha256"] == binding["sha256"]), None)
                if bank is None and artifact is None:
                    raise ValueError("Missing bound artifact evidence")
                required = Path((bank or artifact)["path"])
            if Path(actual).resolve() != required.resolve():
                raise ValueError(f"Prepared path binding differs: {expected}")
    output = Path(plan["argv"][plan["argv"].index("--save_dir") + 1])
    if output.resolve() != workspace / "results" or output.exists() or (workspace / "execution.json").exists():
        raise FileExistsError("Result destination must be new; preserve previous attempts")
    changes = {r["path"]: r["after_sha256"] for r in plan["source_path_changes"]}
    for entry in read_json(root / "source-manifest.json")["files"]:
        if digest_file(workspace / "source" / entry["path"]) != changes.get(entry["path"], entry["sha256"]):
            raise ValueError(f"Prepared source changed: {entry['path']}")
    for item in plan["verified_objects"]:
        if digest_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("Original artifact changed after preparation")
    for item in plan.get("source_asset_links", []):
        path = Path(item["path"])
        if not path.is_symlink() or path.resolve() != Path(item["target"]).resolve() or digest_file(path) != item["sha256"]:
            raise ValueError("Source-relative artifact link changed after preparation")
    for item in plan["bank_path_changes"]:
        if digest_file(Path(item["path"])) != item["derived_sha256"]:
            raise ValueError("Relocated bank changed after preparation")
    for name, item in plan["datasets"].items():
        if digest_file(Path(item["prepared_metadata"])) != item["prepared_metadata_sha256"]:
            raise ValueError("Prepared dataset metadata changed")
        checked = validate_dataset(root / "datasets" / name, Path(item["roots"]["images"]), Path(item["roots"]["masks"]))
        if checked["file_errors"]:
            raise ValueError(f"Dataset bytes changed: {name}")
    execution = {"recipe": plan["recipe"], "status": "running", "started": datetime.now(timezone.utc).isoformat(),
                 "plan_sha256": digest_file(plan_path), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    record = workspace / "execution.json"
    record.write_text(json.dumps(execution, indent=2))
    environment = dict(os.environ, **plan["environment"])
    try:
        with (workspace / "execution.log").open("x") as log:
            result = subprocess.run([sys.executable, str(root / "evaluate.py"), str(plan_path)], cwd=plan["cwd"], env=environment, stdout=log, stderr=subprocess.STDOUT)
        execution.update(returncode=result.returncode, status="failed" if result.returncode else "completed")
        if result.returncode == 0:
            reference, expected_summary = reference_summary(root, plan["recipe"])
            cells = compare(extract(read_json(output / "summary.json"), reference["host"]),
                            extract(expected_summary, reference["host"]))
            comparison = {"recipe": plan["recipe"], "reference_sha256": reference["reference_sha256"],
                          "target_coverage": validate_coverage(root, reference, read_json(output / "summary.json")),
                          "actual_sha256": digest_file(output / "summary.json"),
                          "comparison": "fresh run versus archived per-seed summary; not a claim of agreement with every printed paper cell",
                          "all_match_2dp": all(cell["matches_printed_precision"] for cell in cells), "cells": cells}
            (workspace / "comparison.json").write_text(json.dumps(comparison, indent=2))
            execution["comparison"] = comparison
            execution["status"] = "matched" if comparison["all_match_2dp"] else "mismatch"
    except Exception as error:
        execution.update(status="failed", error=f"{type(error).__name__}: {error}")
    capture_index = output / "artifacts/index.json"
    if capture_index.exists():
        from .captured_export import capture_inventory
        try:
            index_sha, entries = capture_inventory(capture_index.parent)
            execution.update(capture_index_sha256=index_sha, captured_files=entries)
        except Exception as error:
            execution.update(status="failed", capture_error=f"{type(error).__name__}: {error}")
    execution["finished"] = datetime.now(timezone.utc).isoformat()
    record.write_text(json.dumps(execution, indent=2) + "\n")
    return execution
