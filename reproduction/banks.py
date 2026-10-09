"""Validate bank catalogs and tar members without deserializing feature tensors."""
import hashlib
import json
from pathlib import Path
import re
import tarfile


def verify_banks(root: Path, kind: str, archive: Path | None = None) -> dict:
    if kind not in {"host", "raw", "weak"}:
        raise ValueError("Unknown bank catalog kind")
    catalog = json.loads((root / f"{kind}-source-banks.json").read_text(encoding="utf-8"))
    if kind == "weak":
        inventory = json.loads((root / "ablations" / "weak-source.json").read_text(encoding="utf-8"))["recipes"]
    else:
        inventory = json.loads((root / "recipes.json").read_text(encoding="utf-8"))
    recipes = {r["id"]: r for r in inventory}
    assets = {}
    for asset in catalog["artifacts"]:
        digest = asset["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or asset["object_path"] != "objects/" + digest:
            raise ValueError("Invalid bank object path/hash")
        if digest in assets or asset["bytes"] <= 0:
            raise ValueError("Duplicate or invalid bank object")
        assets[digest] = asset
    seen, used = set(), set()
    for row in catalog["bindings"]:
        recipe = recipes[row["recipe"]]
        if row["recipe"] in seen or row["reference_sha256"] != recipe["reference_sha256"] or row["host"] != recipe["host"]:
            raise ValueError("Invalid bank recipe/reference binding")
        seen.add(row["recipe"])
        if not row["source_bank_assets"]:
            raise ValueError("Empty bank binding")
        for item in row["source_bank_assets"]:
            asset = assets[item["sha256"]]
            if item["archived_path"] not in asset["archived_paths"]:
                raise ValueError("Bank path/hash binding mismatch")
            used.add(item["sha256"])
    raw_hosts = {"RawCLIP", "RawImageBind"}
    required = {r["id"] for r in recipes.values() if kind == "weak" or (r["host"] in raw_hosts) == (kind == "raw")}
    if seen != required or used != set(assets):
        raise ValueError("Bank catalog coverage mismatch")
    verified = 0
    if archive is not None:
        expected = {a["object_path"]: a for a in assets.values()}
        with tarfile.open(archive, "r:") as bundle:
            members = bundle.getmembers()
            if len(members) != len(expected) or {m.name for m in members} != set(expected):
                raise ValueError("Unexpected or duplicate bank archive member")
            for member in members:
                asset = expected[member.name]
                if not member.isfile() or member.size != asset["bytes"]:
                    raise ValueError("Bank archive member type/size mismatch")
                digest = hashlib.sha256()
                with bundle.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != asset["sha256"]:
                    raise ValueError("Bank archive member hash mismatch")
                verified += 1
    return {"kind": kind, "recipes_bound": len(seen), "bank_objects": len(assets),
            "bytes": sum(a["bytes"] for a in assets.values()), "files_checked": archive is not None,
            "files_verified": verified, "fresh_gpu_benchmark": False,
            "fresh_bank_rebuild": False, "scope": catalog["scope"]}
