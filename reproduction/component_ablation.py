"""Audit the partial BTAD component/rank trace without hiding unresolved cells."""
import hashlib
import json
from pathlib import Path
import zipfile

from .checkpoint_download import digest_file
from .coverage import validate_coverage


def compare_component_ablation(root: Path, runs: Path | None = None) -> dict:
    manifest = json.loads((root / "ablations/btad-component.json").read_text(encoding="utf-8"))
    archive = root / "ablations" / manifest["archive"]
    if digest_file(archive) != manifest["archive_sha256"]:
        raise ValueError("Component reference archive hash mismatch")
    sources = {s["path"]: s for s in json.loads((root / "source-manifest.json").read_text(encoding="utf-8"))["files"]}
    for launcher in manifest["launchers"]:
        if sources[launcher["path"]]["sha256"] != launcher["sha256"]:
            raise ValueError("Component launcher source hash mismatch")
    references = {r["id"]: r for r in manifest["references"]}
    if len(references) != 6 or len(manifest["rows"]) != 12:
        raise ValueError("Component trace coverage differs from pinned scope")
    summaries, evidence, missing = {}, [], []
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != len(references) or set(bundle.namelist()) != {r["reference"] for r in references.values()}:
            raise ValueError("Unexpected component reference members")
        for ref in references.values():
            original = bundle.read(ref["reference"])
            if len(original) != ref["bytes"] or hashlib.sha256(original).hexdigest() != ref["sha256"]:
                raise ValueError("Component reference member hash/size mismatch")
            if runs is not None and not (runs / ref["id"] / "summary.json").is_file():
                missing.append(ref["id"])
                continue
            payload = original if runs is None else (runs / ref["id"] / "summary.json").read_bytes()
            summary = json.loads(payload)
            coverage = validate_coverage(root, {"transfer": "mvtec2btad", "host": ref["host"]}, summary)
            summaries[ref["id"]] = summary
            evidence.append({"id": ref["id"], "summary_sha256": hashlib.sha256(payload).hexdigest(), "target_coverage": coverage})
    cells, unresolved = [], []
    for row in manifest["rows"]:
        identity = {k: row[k] for k in ("panel", "host", "label")}
        if row["reference_id"] is None:
            unresolved.append(dict(identity, reason="Historical source for this printed row is unresolved", metrics=list(row["printed"])))
            continue
        if row["reference_id"] not in summaries:
            continue
        values = summaries[row["reference_id"]]
        for key in row["selector"]:
            values = values[key]
        for metric, printed in row["printed"].items():
            actual = float(values[metric]) * row["scale"]
            cells.append(dict(identity, metric=metric, actual=actual, printed=printed, matches_2dp=f"{actual:.2f}" == f"{printed:.2f}"))
    matches = sum(c["matches_2dp"] for c in cells)
    return {"table_label": manifest["table_label"], "input": "archived_references" if runs is None else "provided_summaries",
            "status": "mismatch" if matches != len(cells) else "incomplete" if unresolved or missing else "matched",
            "required_printed_cells": sum(len(r["printed"]) for r in manifest["rows"]),
            "compared_cells": len(cells), "matching_cells": matches, "missing_summaries": missing,
            "unresolved_rows": unresolved, "evidence": evidence, "cells": cells,
            "fresh_gpu_execution_verified": False, "scope": manifest["scope"]}
