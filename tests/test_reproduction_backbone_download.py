import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.backbone_download import prepare_backbones


class BackboneDownloadTests(unittest.TestCase):
    def test_resume_checks_ranges_and_revalidates_cached_objects(self):
        for status, content_range in ((206, "bytes 3-6/7"), (200, None), (206, "bytes 2-6/7")):
            with self.subTest(status=status, content_range=content_range), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                sha = hashlib.sha256(b"weights").hexdigest()
                artifact = {"id": "backbone", "sha256": sha, "bytes": 7,
                            "url": "https://example.invalid/pinned", "object_path": "objects/" + sha}
                (root / "backbones.json").write_text(json.dumps({"artifacts": [artifact], "bindings": []}))
                output = root / "prepared"
                (output / "objects").mkdir(parents=True)
                final = output / artifact["object_path"]
                partial = final.with_suffix(".part")
                partial.write_bytes(b"wei")
                previous = b'{"status":"failed","error":"interrupted"}'
                (output / "input-verification.json").write_bytes(previous)
                response = io.BytesIO(b"ghts")
                response.status = status
                response.headers = {"Content-Range": content_range}
                response.geturl = lambda: artifact["url"]
                with patch("reproduction.backbone_download.urllib.request.urlopen", return_value=response) as request:
                    if status == 206 and content_range == "bytes 3-6/7":
                        self.assertEqual(prepare_backbones(root, output, resume=True)["files_verified"], 1)
                        self.assertEqual(final.read_bytes(), b"weights")
                    else:
                        with self.assertRaisesRegex(ValueError, "resume range"):
                            prepare_backbones(root, output, resume=True)
                        self.assertEqual(partial.read_bytes(), b"wei")
                        self.assertFalse(final.exists())
                    self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=3-")
                self.assertIn(previous, [p.read_bytes() for p in (output / "verification-history").iterdir()])
                if final.exists():
                    with patch("reproduction.backbone_download.urllib.request.urlopen") as request:
                        self.assertEqual(prepare_backbones(root, output, resume=True)["files_verified"], 1)
                        final.write_bytes(b"corrupt")
                        with self.assertRaisesRegex(ValueError, "Cached backbone"):
                            prepare_backbones(root, output, resume=True)
                        request.assert_not_called()

    def test_complete_corrupt_partial_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sha = hashlib.sha256(b"weights").hexdigest()
            (root / "backbones.json").write_text(json.dumps({"artifacts": [{"id": "one", "sha256": sha,
                "bytes": 7, "url": "https://example.invalid/pinned", "object_path": "objects/" + sha}], "bindings": []}))
            output = root / "prepared"
            (output / "objects").mkdir(parents=True)
            final = output / "objects" / sha
            final.with_suffix(".part").write_bytes(b"corrupt")
            with patch("reproduction.backbone_download.urllib.request.urlopen") as request:
                with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                    prepare_backbones(root, output, resume=True)
                self.assertFalse(final.exists())
                request.assert_not_called()

    def test_recipe_download_is_pinned_and_corruption_never_becomes_an_object(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                sha = hashlib.sha256(b"weights").hexdigest()
                artifact = {"id": "backbone", "sha256": sha, "bytes": 7,
                            "url": "https://example.invalid/pinned", "object_path": "objects/" + sha}
                (root / "backbones.json").write_text(json.dumps({"artifacts": [artifact],
                    "bindings": [{"recipe": "one", "sha256": sha}]}))
                response = io.BytesIO(b"corrupt" if corrupt else b"weights")
                response.geturl = lambda: artifact["url"]
                output = root / "prepared"
                with patch("reproduction.backbone_download.urllib.request.urlopen", return_value=response) as request:
                    if corrupt:
                        with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                            prepare_backbones(root, output, "one")
                        self.assertFalse((output / artifact["object_path"]).exists())
                        self.assertEqual(json.loads((output / "input-verification.json").read_text())["status"], "failed")
                    else:
                        result = prepare_backbones(root, output, "one")
                        self.assertEqual(result["files_verified"], 1)
                        self.assertEqual((output / artifact["object_path"]).read_bytes(), b"weights")
                        with self.assertRaises(FileExistsError):
                            prepare_backbones(root, output, "one")
                    self.assertEqual(request.call_count, 1)

    def test_unknown_recipe_does_not_download_all_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "backbones.json").write_text('{"artifacts": [], "bindings": []}')
            with patch("reproduction.backbone_download.urllib.request.urlopen") as request:
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    prepare_backbones(root, root / "output", "typo")
                request.assert_not_called()
                self.assertFalse((root / "output").exists())
