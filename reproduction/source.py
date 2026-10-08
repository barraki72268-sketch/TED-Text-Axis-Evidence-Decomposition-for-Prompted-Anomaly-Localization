"""Validate and unpack the archived research source without executing it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def verify_source(archive: Path, manifest: Path) -> dict:
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if hashlib.sha256(archive.read_bytes()).hexdigest() != metadata["archive_sha256"]:
        raise ValueError("Research source archive SHA-256 mismatch")
    expected = {}
    portable_names = set()
    for item in metadata["files"]:
        name = item["path"]
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or ".." in path.parts or
                "\\" in name or ":" in name or str(path) != name):
            raise ValueError(f"Unsafe source path: {name!r}")
        if name.casefold() in portable_names:
            raise ValueError(f"Duplicate or case-colliding source path: {name}")
        portable_names.add(name.casefold())
        expected[name] = item
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if len(infos) != len(expected) or {i.filename for i in infos} != set(expected):
            raise ValueError("Source archive entries differ from the manifest")
        for info in infos:
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Unexpected directory or symlink: {info.filename}")
            item = expected[info.filename]
            if info.file_size != item["bytes"]:
                raise ValueError(f"Source size mismatch: {info.filename}")
            data = bundle.read(info)
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"Source SHA-256 mismatch: {info.filename}")
    return metadata


def unpack_source(archive: Path, manifest: Path, destination: Path) -> dict:
    # Check every entry before creating anything. Never replace an existing tree.
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Source destination already exists: {destination}")
    metadata = verify_source(archive, manifest)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        for item in metadata["files"]:
            output = destination.joinpath(*PurePosixPath(item["path"]).parts)
            if not output.resolve().is_relative_to(destination):
                raise ValueError("Source output escaped the destination")
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as stream:
                stream.write(bundle.read(item["path"]))
    return {"source_files": len(metadata["files"]),
            "archive_sha256": metadata["archive_sha256"],
            "destination": str(destination), "evaluation_executed": False}
