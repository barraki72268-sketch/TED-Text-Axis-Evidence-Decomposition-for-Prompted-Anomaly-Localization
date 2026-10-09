"""Reassemble the weak-source table with explicit host-native metric units."""
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

from .datasets import load_protocol


def compare_weak_source(root: Path, runs: Path | None = None) -> dict:
    manifest = json.loads((root / "ablations/weak-source.json").read_text(encoding="utf-8"))
    archive = root / "ablations" / manifest["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Weak-source reference archive hash mismatch")
    recipes = manifest["recipes"]
    if len(recipes) != 28 or len({r["id"] for r in recipes}) != 28:
        raise ValueError("Weak-source table requires all 28 distinct configurations")
    report = {"table_label": manifest["table_label"], "input": "archived_references" if runs is None else "provided_summaries",
              "fresh_gpu_execution_verified": False, "n_convention": "transfer settings, not seeds"}
    if runs is not None:
        missing = [r["id"] for r in recipes if not (runs / r["id"] / "summary.json").is_file()]
        if missing:
            return dict(report, status="incomplete", missing=missing)
    groups, evidence, protocols = {}, [], {}
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != 28 or set(bundle.namelist()) != {r["reference"] for r in recipes}:
            raise ValueError("Unexpected reference archive coverage")
        for recipe in recipes:
            reference = bundle.read(recipe["reference"])
            if hashlib.sha256(reference).hexdigest() != recipe["reference_sha256"]:
                raise ValueError("Weak-source reference member hash mismatch")
            payload = reference if runs is None else (runs / recipe["id"] / "summary.json").read_bytes()
            summary = json.loads(payload)
            dataset = recipe["preset"].split("2", 1)[1]
            if dataset not in protocols:
                _, metadata = load_protocol(root / "datasets" / dataset)
                protocols[dataset] = {name: len(items) for name, items in metadata["test"].items()}
            classes = summary["per_class"]
            counts = {row["class_name"]: row["num_images"] for row in classes}
            if len(counts) != len(classes) or counts != protocols[dataset]:
                raise ValueError(f"Target class/image coverage differs from full test split: {recipe['id']}")
            mean = summary["mean"]
            aa = recipe["host"] == "AA-CLIP"
            scale, pro = (1.0, "pixel_pro") if aa else (100.0, "pixel_aupro")
            base = mean["baseline" if aa else "baseline_official"]
            candidate = mean["candidates" if aa else "parallel"][recipe["candidate_key"]]
            dp = (candidate[pro] - base[pro]) * scale
            da = (candidate["pixel_ap"] - base["pixel_ap"]) * scale
            groups.setdefault((recipe["host"], recipe["budget"]), []).append((dp, da, (dp + da) / 2))
            evidence.append({"recipe": recipe["id"], "summary_sha256": hashlib.sha256(payload).hexdigest(),
                             "candidate_key": recipe["candidate_key"], "reported_test_images": sum(counts.values()),
                             "reported_test_classes": len(counts), "reported_coverage_matches_manifest": True})
    cells = []
    for printed in manifest["printed_rows"]:
        values = groups[(printed["host"], printed["budget"])]
        if len(values) != printed["n"]:
            raise ValueError("Weak-source group coverage differs from printed n")
        for index, metric in enumerate(("delta_pro", "delta_ap", "delta_loc")):
            value = statistics.mean(row[index] for row in values)
            cells.append({"host": printed["host"], "budget": printed["budget"], "n": len(values),
                          "metric": metric, "actual": value, "printed": printed[metric],
                          "matches_1dp": f"{value:.1f}" == f"{printed[metric]:.1f}"})
    return dict(report, status="matched" if all(c["matches_1dp"] for c in cells) else "mismatch",
                recipes_checked=len(evidence), evidence=evidence, cells=cells)
