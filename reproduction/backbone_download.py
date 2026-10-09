"""Fetch exact upstream backbone bytes; never deserialize downloaded weights."""
import json
from pathlib import Path
import re
import urllib.request
import uuid

from .checkpoint_download import digest_file


def required_backbone_assets(catalog: dict, recipe: str) -> list[dict]:
    bindings = [b for b in catalog["bindings"] if b["recipe"] == recipe]
    if len(bindings) != 1:
        raise ValueError(f"Expected exactly one backbone binding for {recipe}")
    assets = {a["sha256"]: a for a in catalog["artifacts"]}
    selected = [assets[bindings[0]["sha256"]]]
    # The archived AA evaluator checks this file before selecting its actual
    # backbone, including B+ and H/14. It is a loader prerequisite, not a
    # replacement for the recipe's selected model.
    if bindings[0].get("host") == "AA-CLIP":
        bootstrap = next(a for a in catalog["artifacts"] if a["id"] == "openai_vit_l14_336")
        if bootstrap["sha256"] != selected[0]["sha256"]:
            selected.append(bootstrap)
    return selected


def prepare_backbones(root: Path, destination: Path, recipe: str | None = None, resume: bool = False) -> dict:
    catalog = json.loads((root / "backbones.json").read_text(encoding="utf-8"))
    artifacts = {a["sha256"]: a for a in catalog["artifacts"]}
    if len(artifacts) != len(catalog["artifacts"]):
        raise ValueError("Duplicate backbone object")
    for sha, asset in artifacts.items():
        if not re.fullmatch(r"[0-9a-f]{64}", sha) or asset["object_path"] != "objects/" + sha:
            raise ValueError("Invalid backbone object path/hash")
        if asset["bytes"] <= 0 or not asset["url"].startswith("https://"):
            raise ValueError("Invalid backbone size/URL")
    if recipe is not None:
        selected = required_backbone_assets(catalog, recipe)
    else:
        selected = list(artifacts.values())
    destination = destination.absolute()
    if destination.exists() and not resume:
        raise FileExistsError(f"Destination must be new: {destination}")
    destination.mkdir(parents=True, exist_ok=resume)
    (destination / "objects").mkdir(exist_ok=resume)
    report = {"recipe": recipe, "expected_objects": len(selected), "files_verified": 0,
              "fresh_gpu_benchmark": False, "status": "incomplete", "objects": []}
    record = destination / "input-verification.json"
    if record.exists():
        history = destination / "verification-history"
        history.mkdir(exist_ok=True)
        (history / (uuid.uuid4().hex + ".json")).write_bytes(record.read_bytes())
    def save():
        record.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        for asset in selected:
            final = destination / asset["object_path"]
            partial = final.with_suffix(".part")
            if final.exists():
                if final.stat().st_size != asset["bytes"] or digest_file(final) != asset["sha256"]:
                    raise ValueError(f"Cached backbone hash/size mismatch: {asset['id']}")
                size = asset["bytes"]
            else:
                size = partial.stat().st_size if partial.exists() else 0
                if size > asset["bytes"]:
                    raise ValueError("Partial backbone exceeds pinned byte size")
                headers = {"User-Agent": "TED-reproducibility/1.0"}
                if size:
                    headers["Range"] = f"bytes={size}-"
                request = urllib.request.Request(asset["url"], headers=headers)
                if size < asset["bytes"]:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        if not response.geturl().startswith("https://"):
                            raise ValueError("Backbone download redirected away from HTTPS")
                        if size and (response.status != 206 or response.headers.get("Content-Range") !=
                                     f"bytes {size}-{asset['bytes'] - 1}/{asset['bytes']}"):
                            raise ValueError("Server did not honor the exact backbone resume range")
                        with partial.open("ab" if size else "wb") as output:
                            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                                size += len(block)
                                if size > asset["bytes"]:
                                    raise ValueError("Backbone exceeds pinned byte size")
                                output.write(block)
                if size != asset["bytes"] or digest_file(partial) != asset["sha256"]:
                    raise ValueError(f"Backbone hash/size mismatch: {asset['id']}")
                partial.rename(final)
            report["objects"].append({"id": asset["id"], "sha256": asset["sha256"],
                                      "bytes": size, "url": asset["url"]})
            report["files_verified"] += 1
            save()
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    report["status"] = "verified"
    save()
    return report
