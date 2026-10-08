import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.backbone_download import prepare_backbones


class BackboneDownloadTests(unittest.TestCase):
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
