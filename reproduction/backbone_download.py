"""Fetch exact upstream backbone bytes; never deserialize downloaded weights."""
import json
from pathlib import Path
import re
import urllib.request

from .checkpoint_download import digest_file


def prepare_backbones(root: Path, destination: Path, recipe: str | None = None) -> dict:
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
        bindings = [b for b in catalog["bindings"] if b["recipe"] == recipe]
        if len(bindings) != 1:
            raise ValueError(f"Expected exactly one backbone binding for {recipe}")
        selected = [artifacts[bindings[0]["sha256"]]]
    else:
        selected = list(artifacts.values())
    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError(f"Destination must be new: {destination}")
    destination.mkdir(parents=True)
    (destination / "objects").mkdir()
    report = {"recipe": recipe, "expected_objects": len(selected), "files_verified": 0,
              "fresh_gpu_benchmark": False, "status": "incomplete", "objects": []}
    record = destination / "input-verification.json"
    def save():
        record.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        for asset in selected:
            final = destination / asset["object_path"]
            partial = final.with_suffix(".part")
            size = 0
            with partial.open("xb") as output:
                request = urllib.request.Request(asset["url"], headers={"User-Agent": "TED-reproducibility/1.0"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if not response.geturl().startswith("https://"):
                        raise ValueError("Backbone download redirected away from HTTPS")
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
