import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from reproduction.source import unpack_source, verify_source


class SourceBundleTests(unittest.TestCase):
    def bundle(self, root, name="model/model.py"):
        archive, manifest = root / "source.zip", root / "manifest.json"
        data = b"# archived source\nVALUE = 42\n"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr(name, data)
        metadata = {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "files": [{"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}]}
        manifest.write_text(json.dumps(metadata), encoding="utf-8")
        return archive, manifest, data

    def test_exact_bytes_and_existing_directory_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, manifest, data = self.bundle(root)
            destination = root / "unpacked"
            report = unpack_source(archive, manifest, destination)
            self.assertFalse(report["evaluation_executed"])
            self.assertEqual((destination / "model/model.py").read_bytes(), data)
            with self.assertRaises(FileExistsError):
                unpack_source(archive, manifest, destination)
            self.assertEqual((destination / "model/model.py").read_bytes(), data)

    def test_manifest_file_hash_mismatch_creates_no_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, manifest, _ = self.bundle(root)
            metadata = json.loads(manifest.read_text())
            metadata["files"][0]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Source SHA-256 mismatch"):
                unpack_source(archive, manifest, root / "unpacked")
            self.assertFalse((root / "unpacked").exists())

    def test_rejects_traversal_even_when_all_hashes_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, manifest, _ = self.bundle(root, "../escaped.py")
            with self.assertRaisesRegex(ValueError, "Unsafe source path"):
                unpack_source(archive, manifest, root / "unpacked")
            self.assertFalse((root / "unpacked").exists())
            self.assertFalse((root / "escaped.py").exists())

    def test_archive_integrity_is_checked_before_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, manifest, _ = self.bundle(root)
            with archive.open("ab") as stream:
                stream.write(b"modified")
            with self.assertRaisesRegex(ValueError, "archive SHA-256 mismatch"):
                verify_source(archive, manifest)
