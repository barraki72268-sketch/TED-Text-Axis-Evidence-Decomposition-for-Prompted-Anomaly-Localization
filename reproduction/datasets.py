"""Verify recorded dataset bytes and prepare metadata without modifying images."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def relative_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or
            "\\" in name or ":" in name or str(path) != name):
        raise ValueError(f"Invalid dataset-relative path: {name!r}")
    return path


def load_protocol(protocol: Path) -> tuple[dict, dict]:
    manifest = json.loads((protocol / "manifest.json").read_text(encoding="utf-8"))
    metadata_path = protocol / "meta.json"
    if digest(metadata_path) != manifest["metadata_sha256"]:
        raise ValueError("Recorded dataset metadata SHA-256 mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = set()
    for entry in manifest["files"]:
        if entry["role"] not in {"images", "masks"}:
            raise ValueError("Unknown dataset file role")
        relative_path(entry["path"])
        key = (entry["role"], entry["path"])
        if key in expected:
            raise ValueError("Duplicate dataset manifest entry")
        expected.add(key)
    referenced, counts = set(), {}
    for split, classes in metadata.items():
        counts[split] = {}
        for category, rows in classes.items():
            counts[split][category] = len(rows)
            for row in rows:
                if not row.get("img_path"):
                    raise ValueError("Dataset row has no image")
                for field, role in (("img_path", "images"), ("mask_path", "masks")):
                    if row.get(field):
                        relative_path(row[field])
                        referenced.add((role, row[field]))
    if referenced != expected:
        raise ValueError("Dataset metadata references differ from the file manifest")
    if counts != manifest["counts"]:
        raise ValueError("Dataset split/class counts differ from the manifest")
    return manifest, metadata


def validate_dataset(protocol: Path, images: Path, masks: Path | None = None) -> dict:
    manifest, _ = load_protocol(protocol)
    if manifest["layout"] == "datasetninja_images_and_derived_binary_masks" and masks is None:
        raise ValueError("BTAD requires --masks pointing to the recorded binary-mask layout")
    roots = {"images": images.resolve(), "masks": (masks or images).resolve()}
    errors, examples, verified = 0, [], 0
    for entry in manifest["files"]:
        path = roots[entry["role"]].joinpath(*relative_path(entry["path"]).parts)
        problem = None
        try:
            if path.stat().st_size != entry["bytes"]:
                problem = "size_mismatch"
            elif digest(path) != entry["sha256"]:
                problem = "sha256_mismatch"
        except OSError as error:
            problem = type(error).__name__
        if problem:
            errors += 1
            if len(examples) < 20:
                examples.append({"role": entry["role"], "path": entry["path"], "problem": problem})
        else:
            verified += 1
    return {"dataset": manifest["dataset"], "status": "verified" if not errors else "input_mismatch",
            "files_verified": verified, "file_errors": errors, "error_examples": examples,
            "roots": {key: str(value) for key, value in roots.items()},
            "manifest_sha256": digest(protocol / "manifest.json"),
            "metadata_sha256": manifest["metadata_sha256"], "counts": manifest["counts"],
            "evaluation_executed": False}


def prepare_dataset(protocol: Path, images: Path, masks: Path | None, output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Prepared dataset directory already exists: {output}")
    report = validate_dataset(protocol, images, masks)
    if report["file_errors"]:
        return report
    _, metadata = load_protocol(protocol)
    for classes in metadata.values():
        for rows in classes.values():
            for row in rows:
                for field, role in (("img_path", "images"), ("mask_path", "masks")):
                    if row.get(field):
                        row[field] = Path(report["roots"][role]).joinpath(*relative_path(row[field]).parts).as_posix()
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / "meta.json"
    with prepared.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    report["prepared_metadata"] = str(prepared)
    report["prepared_metadata_sha256"] = digest(prepared)
    with (output / "input-verification.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return report
