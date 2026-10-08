import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import test_reproduction_checkpoints as fixtures
from reproduction.banks import verify_banks


class BankArchiveTests(unittest.TestCase):
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
