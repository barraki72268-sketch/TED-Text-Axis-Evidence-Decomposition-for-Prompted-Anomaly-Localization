import io
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import test_reproduction_checkpoints as fixtures
from reproduction.banks import verify_banks
from reproduction.bank_download import prepare_banks


class BankArchiveTests(unittest.TestCase):
    def test_weak_source_coverage_is_independent_of_main_recipes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog, _ = fixtures.CheckpointTests().fixture(root)
            for row in catalog["bindings"]:
                row["source_bank_assets"] = row.pop("checkpoint_assets")
            (root / "weak-source-banks.json").write_text(json.dumps(catalog))
            (root / "ablations").mkdir()
            recipes = json.loads((root / "recipes.json").read_text())
            (root / "ablations/weak-source.json").write_text(json.dumps({"recipes": recipes}))
            (root / "recipes.json").write_text("[]")
            self.assertEqual(verify_banks(root, "weak")["recipes_bound"], len(recipes))
            (root / "ablations/weak-source.json").write_text(json.dumps({"recipes": recipes + [dict(recipes[0], id="missing-budget")]}))
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                verify_banks(root, "weak")

    def test_modified_or_unexpected_bank_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog, original = fixtures.CheckpointTests().fixture(root)
            for row in catalog["bindings"]:
                row["source_bank_assets"] = row.pop("checkpoint_assets")
            (root / "host-source-banks.json").write_text(json.dumps(catalog))
            archive = root / "banks.tar"
            name = catalog["artifacts"][0]["object_path"]
            data = original.read_bytes()
            for member_name, member_data, valid in [(name, data, True), (name, b"X" * len(data), False), ("../outside", data, False)]:
                with tarfile.open(archive, "w") as bundle:
                    member = tarfile.TarInfo(member_name)
                    member.size = len(member_data)
                    bundle.addfile(member, io.BytesIO(member_data))
                if valid:
                    self.assertEqual(verify_banks(root, "host", archive)["files_verified"], 1)
                else:
                    with self.assertRaises(ValueError):
                        verify_banks(root, "host", archive)
                (root / "host-source-banks-download.json").write_text(json.dumps({
                    "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "revision": "fixture"}))
                destination = root / ("valid" if valid else "invalid")
                if valid:
                    self.assertEqual(prepare_banks(root, "host", destination, archive)["files_verified"], 1)
                    self.assertEqual((destination / name).read_bytes(), data)
                    with self.assertRaises(FileExistsError):
                        prepare_banks(root, "host", destination, archive)
                else:
                    with self.assertRaises(ValueError):
                        prepare_banks(root, "host", destination, archive)
                    self.assertFalse(destination.exists())
