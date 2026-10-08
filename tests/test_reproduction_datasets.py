import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from reproduction.datasets import load_protocol, prepare_dataset, validate_dataset


class DatasetProtocolTests(unittest.TestCase):
    def fixture(self, root):
        protocol, images, masks = root / "protocol", root / "images", root / "masks"
        for directory in (protocol, images, masks):
            directory.mkdir()
        files = []
        for role, path, data in [("images", "normal.bin", b"normal"),
                                 ("images", "defect.bin", b"defect"),
                                 ("masks", "defect.bin", b"mask01")]:
            (root / role / path).write_bytes(data)
            files.append({"role": role, "path": path, "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
        metadata = {"train": {"part": [{"img_path": "normal.bin", "mask_path": "", "anomaly": 0}]},
                    "test": {"part": [{"img_path": "defect.bin", "mask_path": "defect.bin", "anomaly": 1},
                                      {"img_path": "normal.bin", "mask_path": "", "anomaly": 0}]}}
        raw = json.dumps(metadata).encode()
        (protocol / "meta.json").write_bytes(raw)
        manifest = {"dataset": "fixture", "layout": "datasetninja_images_and_derived_binary_masks",
                    "metadata_sha256": hashlib.sha256(raw).hexdigest(), "files": files,
                    "counts": {"train": {"part": 1}, "test": {"part": 2}}}
        (protocol / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return protocol, images, masks

    def test_prepare_preserves_order_labels_and_distinct_image_mask_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol, images, masks = self.fixture(root)
            original = (protocol / "meta.json").read_bytes()
            report = prepare_dataset(protocol, images, masks, root / "prepared")
            self.assertEqual(report["files_verified"], 3)
            metadata = json.loads((root / "prepared/meta.json").read_text())
            rows = metadata["test"]["part"]
            self.assertEqual([row["anomaly"] for row in rows], [1, 0])
            # Windows TEMP may use an 8.3 alias (RUNNER~1); validate the file
            # identity instead of assuming one spelling of its absolute path.
            self.assertTrue(Path(rows[0]["img_path"]).is_absolute())
            self.assertTrue(Path(rows[0]["mask_path"]).is_absolute())
            self.assertTrue(Path(rows[0]["img_path"]).samefile(images / "defect.bin"))
            self.assertTrue(Path(rows[0]["mask_path"]).samefile(masks / "defect.bin"))
            self.assertEqual((protocol / "meta.json").read_bytes(), original)
            with self.assertRaises(FileExistsError):
                prepare_dataset(protocol, images, masks, root / "prepared")

    def test_same_size_changed_image_is_rejected_without_writing_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol, images, masks = self.fixture(root)
            (images / "normal.bin").write_bytes(b"CHANGE")
            report = prepare_dataset(protocol, images, masks, root / "prepared")
            self.assertEqual(report["file_errors"], 1)
            self.assertEqual(report["error_examples"][0]["problem"], "sha256_mismatch")
            self.assertFalse((root / "prepared").exists())

    def test_separate_mask_root_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            protocol, images, _ = self.fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, "requires --masks"):
                validate_dataset(protocol, images)

    def test_published_metadata_and_manifests_cover_identical_inputs(self):
        root = Path(__file__).resolve().parents[1] / "reproduction/datasets"
        for protocol in sorted(root.iterdir()):
            with self.subTest(dataset=protocol.name):
                manifest, _ = load_protocol(protocol)
                self.assertEqual(manifest["dataset"], protocol.name)
