import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import test_reproduction_checkpoints as fixtures
from reproduction.checkpoint_download import prepare_checkpoints


class CheckpointArchiveTests(unittest.TestCase):
    def fixture(self, root, invalid_name=False, invalid_bytes=False):
        catalog, path = fixtures.CheckpointTests().fixture(root)
        archive = root / "weights.tar"
        data = path.read_bytes()
        if invalid_bytes:
            data = b"X" * len(data)
        with tarfile.open(archive, "w") as bundle:
            member = tarfile.TarInfo("../escape" if invalid_name else catalog["artifacts"][0]["object_path"])
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
        (root / "host-checkpoints-download.json").write_text(json.dumps({
            "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "revision": "fixture"}))
        return archive

    def test_exact_archive_extracts_and_existing_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = self.fixture(root)
            destination = root / "prepared"
            result = prepare_checkpoints(root, destination, archive)
            self.assertEqual(result["files_verified"], 1)
            self.assertFalse(result["fresh_gpu_benchmark"])
            with self.assertRaises(FileExistsError):
                prepare_checkpoints(root, destination, archive)
            self.assertEqual(len(list((destination / "objects").iterdir())), 1)

    def test_valid_archive_hash_does_not_allow_traversal_or_wrong_member_bytes(self):
        for invalid_name, invalid_bytes in [(True, False), (False, True)]:
            with self.subTest(invalid_name=invalid_name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive = self.fixture(root, invalid_name, invalid_bytes)
                with self.assertRaises(ValueError):
                    prepare_checkpoints(root, root / "prepared", archive)
                self.assertFalse((root / "prepared").exists())
