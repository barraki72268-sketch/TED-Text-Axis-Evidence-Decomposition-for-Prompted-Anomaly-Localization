"""Acquire and extract the pinned historical host checkpoints without PyTorch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

from .checkpoints import load_checkpoints, verify_checkpoints


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_checkpoints(root: Path, destination: Path, archive: Path | None = None) -> dict:
    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError(f"Destination must be new: {destination}")
    catalog = load_checkpoints(root)
    specification = json.loads((root / "host-checkpoints-download.json").read_text(encoding="utf-8"))
    if archive is None:
        cache = destination.parent / "ted-download-cache"
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / (specification["sha256"] + ".tar")
        if not archive.exists():
            partial = archive.with_suffix(".tar.part")
            # Exclusive creation preserves any interrupted download for diagnosis.
            with partial.open("xb") as output:
                with urllib.request.urlopen(specification["url"], timeout=120) as response:
                    size = 0
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        size += len(block)
                        if size > specification["bytes"]:
                            raise ValueError("Download exceeds pinned archive size")
                        output.write(block)
            if partial.stat().st_size != specification["bytes"] or digest_file(partial) != specification["sha256"]:
                raise ValueError("Downloaded checkpoint archive hash/size mismatch")
            partial.rename(archive)
    if archive.stat().st_size != specification["bytes"] or digest_file(archive) != specification["sha256"]:
        raise ValueError("Checkpoint archive hash/size mismatch")
    expected = {asset["object_path"]: asset for asset in catalog["artifacts"]}
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("Checkpoint archive contains unexpected or duplicate entries")
        for member in members:
            asset = expected[member.name]
            if not member.isfile() or member.size != asset["bytes"]:
                raise ValueError("Checkpoint archive member type/size mismatch")
            digest = hashlib.sha256()
            with bundle.extractfile(member) as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != asset["sha256"]:
                raise ValueError("Checkpoint archive member hash mismatch")
        # No destination files are written until every archive/member check passes.
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "objects").mkdir()
        for member in members:
            with bundle.extractfile(member) as source, (destination / member.name).open("xb") as output:
                shutil.copyfileobj(source, output)
    report = verify_checkpoints(root, destination)
    report.update(archive_sha256=specification["sha256"], revision=specification["revision"])
    (destination / "input-verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
