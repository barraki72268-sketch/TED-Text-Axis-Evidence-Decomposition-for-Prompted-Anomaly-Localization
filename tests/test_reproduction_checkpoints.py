import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from reproduction.checkpoints import load_checkpoints, verify_checkpoints


class CheckpointTests(unittest.TestCase):
    def fixture(self, root):
        (root / "references").mkdir()
        (root / "objects").mkdir()
        reference = b'{"checkpoint":"/archive/host.pth"}'
        reference_hash = hashlib.sha256(reference).hexdigest()
        (root / "references/ref.json").write_bytes(reference)
        data = b"historical checkpoint"
        digest = hashlib.sha256(data).hexdigest()
        (root / "objects" / digest).write_bytes(data)
        (root / "recipes.json").write_text(json.dumps([{
            "id": "fixture", "host": "FAPrompt", "reference": "ref.json",
            "reference_sha256": reference_hash}]))
        catalog = {"scope": "fixture", "unresolved": [], "artifacts": [{
            "sha256": digest, "object_path": "objects/" + digest, "bytes": len(data),
            "archived_paths": ["/archive/host.pth"]}], "bindings": [{
            "recipe": "fixture", "host": "FAPrompt", "reference_sha256": reference_hash,
            "checkpoint_assets": [{"role": "checkpoint", "archived_path": "/archive/host.pth",
                                   "checkpoint_filename": "host.pth", "sha256": digest}]}]}
        (root / "host-checkpoints.json").write_text(json.dumps(catalog))
        return catalog, root / "objects" / digest

    def test_missing_or_same_size_changed_bytes_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, path = self.fixture(root)
            self.assertEqual(verify_checkpoints(root, root)["files_verified"], 1)
            path.write_bytes(b"X" * path.stat().st_size)
            result = verify_checkpoints(root, root)
            self.assertEqual(result["files_verified"], 0)
            self.assertEqual(result["file_errors"][0]["error"], "SHA-256 mismatch")
            path.unlink()
            self.assertEqual(len(verify_checkpoints(root, root)["file_errors"]), 1)

    def test_binding_must_be_supported_by_archived_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog, _ = self.fixture(root)
            catalog["bindings"][0]["checkpoint_assets"][0]["role"] = "unrecorded_field"
            (root / "host-checkpoints.json").write_text(json.dumps(catalog))
            with self.assertRaisesRegex(ValueError, "absent from archived reference"):
                load_checkpoints(root)

    def test_real_catalog_covers_all_adapted_host_recipes(self):
        root = Path(__file__).resolve().parents[1] / "reproduction"
        result = verify_checkpoints(root)
        self.assertEqual(result["recipes_bound"], 207)
        self.assertEqual(result["unique_checkpoint_files"], 11)
        self.assertFalse(result["files_checked"])
        self.assertFalse(result["fresh_gpu_benchmark"])
