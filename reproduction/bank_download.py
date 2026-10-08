"""Acquire pinned public source banks without importing PyTorch."""
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

from .banks import verify_banks
from .checkpoint_download import digest_file


def prepare_banks(root: Path, kind: str, destination: Path, archive: Path | None = None) -> dict:
    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError(f"Destination must be new: {destination}")
    specification_path = root / f"{kind}-source-banks-download.json"
    if not specification_path.is_file():
        raise ValueError(f"Pinned {kind} bank download is not published yet")
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    verify_banks(root, kind)
    if archive is None:
        cache = destination.parent / "ted-download-cache"
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / (specification["sha256"] + ".tar")
        if not archive.exists():
            partial = archive.with_suffix(".tar.part")
            with partial.open("xb") as output:
                with urllib.request.urlopen(specification["url"], timeout=120) as response:
                    size = 0
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        size += len(block)
                        if size > specification["bytes"]:
                            raise ValueError("Download exceeds pinned archive size")
                        output.write(block)
            if partial.stat().st_size != specification["bytes"] or digest_file(partial) != specification["sha256"]:
                raise ValueError("Downloaded bank archive hash/size mismatch")
            partial.rename(archive)
    if archive.stat().st_size != specification["bytes"] or digest_file(archive) != specification["sha256"]:
        raise ValueError("Bank archive hash/size mismatch")
    report = verify_banks(root, kind, archive)
    # The catalog and verifier require exactly objects/<sha256>, regular files,
    # and a matching digest for every member before anything is extracted.
    with tarfile.open(archive, "r:") as bundle:
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "objects").mkdir()
        for member in bundle.getmembers():
            with bundle.extractfile(member) as source, (destination / member.name).open("xb") as output:
                shutil.copyfileobj(source, output)
    catalog = json.loads((root / f"{kind}-source-banks.json").read_text(encoding="utf-8"))
    for asset in catalog["artifacts"]:
        if digest_file(destination / asset["object_path"]) != asset["sha256"]:
            raise ValueError("Extracted bank hash mismatch")
    report.update(archive_sha256=specification["sha256"], revision=specification["revision"])
    (destination / "input-verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
