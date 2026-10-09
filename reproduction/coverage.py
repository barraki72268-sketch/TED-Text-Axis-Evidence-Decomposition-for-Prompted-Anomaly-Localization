"""Check declared target coverage without claiming independent image execution."""
from pathlib import Path

from .datasets import load_protocol


def validate_coverage(root: Path, recipe: dict, summary: dict) -> dict:
    dataset = recipe["transfer"].split("2", 1)[1]
    protocol = root / "datasets" / dataset
    if not (protocol / "manifest.json").is_file():
        return {"status": "unverified", "dataset": dataset, "reason": "Dataset protocol is not available",
                "per_image_execution_verified": False}
    _, metadata = load_protocol(protocol)
    expected = {name: len(rows) for name, rows in metadata["test"].items()}
    if recipe["host"] in {"AA-CLIP", "FAPrompt"}:
        rows = summary["per_class"]
        actual = {r["class_name"]: r["num_images"] for r in rows}
        if len(actual) != len(rows) or actual != expected:
            raise ValueError("Reported target classes/image counts differ from the full test split")
        counts_checked = True
    else:
        modes = ["baseline_txt", "ted_calibrated_vl" if recipe["host"] == "AdaptCLIP" else "ted_calibrated"]
        if recipe["host"] in {"RawCLIP", "RawImageBind"}:
            modes.insert(1, "parallel_margin")
        for mode in modes:
            rows = [r for r in summary["rows"] if r["mode"] == mode and r["target_label"] != "mean"]
            classes = [r["target_label"] for r in rows]
            if len(classes) != len(set(classes)) or set(classes) != set(expected):
                raise ValueError(f"Reported target classes differ from the full test split: {mode}")
        counts_checked = False  # These upstream schemas do not record per-class image counts.
    return {"status": "reported_coverage_matches", "dataset": dataset,
            "classes_checked": len(expected), "expected_test_images": sum(expected.values()),
            "reported_image_counts_checked": counts_checked, "per_image_execution_verified": False}
