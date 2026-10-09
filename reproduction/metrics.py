"""Explicit host-native metric contracts and paper rounding conventions."""
from __future__ import annotations

import math
import statistics

METRICS = ("I-AUROC", "P-AUROC", "P-PRO", "P-AP")


def extract_recipe(summary: dict, recipe: dict) -> dict:
    if recipe.get('scope') != 'residual-strength-ablation':
        return extract(summary, recipe['host'])
    alphas = [0, 0.25, 0.5, 1, 2]
    modes = ['prescore_calibrated_residual', 'prescore_calibrated']
    if summary['alphas'] != alphas or summary['blend_modes'] != modes or summary['img_size'] != 224:
        raise ValueError('Residual-strength summary configuration differs')
    candidates = summary['mean']['candidates']
    expected = {mode + '_alpha_' + f'{alpha:g}' for mode in modes for alpha in alphas}
    if set(candidates) != expected:
        raise ValueError('Residual-strength candidate coverage differs')
    rows = dict(Base=summary['mean']['baseline'], **{key: candidates[key] for key in sorted(expected)})
    keys = ('image_auc', 'pixel_auc', 'pixel_pro', 'pixel_ap')
    result = {name: {metric: float(row[key]) for metric, key in zip(METRICS, keys)}
              for name, row in rows.items()}
    if any(not math.isfinite(value) or not 0 <= value <= 100
           for row in result.values() for value in row.values()):
        raise ValueError('Invalid residual-strength metric')
    return result


def extract(summary: dict, host: str) -> dict[str, dict[str, float]]:
    """Return percentage points; do not infer units from observed magnitudes."""
    if host == "AA-CLIP":
        mean = summary["mean"]
        rows = {"Base": mean["baseline"], "OURS": mean["candidates"]["prescore_calibrated_alpha_1"]}
        keys = ("image_auc", "pixel_auc", "pixel_pro", "pixel_ap")
        scale = 1.0
    elif host == "FAPrompt":
        mean = summary["mean"]
        base = mean["baseline_official"]
        ours = dict(mean["parallel"]["branch_calibrated_alpha_0.5"])
        # The archived table assembly retains the official image score.
        ours["image_auc"] = base["image_auc"]
        rows = {"Base": base, "OURS": ours}
        keys = ("image_auc", "pixel_auc", "pixel_aupro", "pixel_ap")
        scale = 100.0
    elif host in {"AdaCLIP", "AdaptCLIP", "BayesPFL", "RawCLIP", "RawImageBind"}:
        mean = {}
        for row in summary["rows"]:
            if row.get("target_label") == "mean":
                if row["mode"] in mean:
                    raise ValueError(f"Duplicate mean mode: {row['mode']}")
                mean[row["mode"]] = row
        if host in {"RawCLIP", "RawImageBind"}:
            rows = {"Base": mean["baseline_txt"], "T-TED": mean["parallel_margin"], "C-TED": mean["ted_calibrated"]}
        else:
            mode = "ted_calibrated_vl" if host == "AdaptCLIP" else "ted_calibrated"
            rows = {"Base": mean["baseline_txt"], "OURS": mean[mode]}
        keys = ("image_auroc", "pixel_auroc", "pixel_aupro", "pixel_ap")
        scale = 100.0
    else:
        raise ValueError(f"Unsupported host: {host}")
    result = {method: {metric: float(row[key]) * scale for metric, key in zip(METRICS, keys)} for method, row in rows.items()}
    for method, metrics in result.items():
        for name, value in metrics.items():
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"Invalid {method} {name}: {value}")
    return result


def compare(actual: dict, expected: dict, decimals: int = 2) -> list[dict]:
    if actual.keys() != expected.keys():
        raise ValueError("Method sets differ")
    return [
        {"method": method, "metric": metric, "actual": actual[method][metric],
         "expected": expected[method][metric],
         "delta_pp": actual[method][metric] - expected[method][metric],
         "matches_printed_precision": format(actual[method][metric], f".{decimals}f") == format(expected[method][metric], f".{decimals}f")}
        for method in expected for metric in METRICS
    ]


def aggregate(runs: list[dict], convention: str) -> dict:
    if not runs:
        raise ValueError("No completed runs; missing runs must not become zero-valued results")
    if convention not in {"sample_std", "population_std"}:
        raise ValueError(f"Unknown standard-deviation convention: {convention}")
    if any(run.keys() != runs[0].keys() for run in runs):
        raise ValueError("Method sets differ across seeds")
    result = {}
    for method in runs[0]:
        result[method] = {}
        for metric in METRICS:
            values = [run[method][metric] for run in runs]
            std = statistics.stdev(values) if convention == "sample_std" and len(values) > 1 else statistics.pstdev(values)
            result[method][metric] = {"mean": statistics.mean(values), "std": std, "n": len(values)}
    return result
