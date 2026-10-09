import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from reproduction.weak_source import compare_weak_source

ROOT = Path(__file__).resolve().parents[1] / "reproduction"


class WeakSourceTests(unittest.TestCase):
    def test_missing_configuration_cannot_pass_table(self):
        with tempfile.TemporaryDirectory() as temp:
            report = compare_weak_source(ROOT, Path(temp))
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(len(report["missing"]), 28)

    def test_modified_metric_changes_table_result(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            manifest = json.loads((ROOT / "ablations/weak-source.json").read_text())
            with zipfile.ZipFile(ROOT / "ablations" / manifest["archive"]) as bundle:
                for recipe in manifest["recipes"]:
                    target = output / recipe["id"] / "summary.json"
                    target.parent.mkdir()
                    target.write_bytes(bundle.read(recipe["reference"]))
            self.assertEqual(compare_weak_source(ROOT, output)["status"], "matched")
            recipe = manifest["recipes"][0]
            target = output / recipe["id"] / "summary.json"
            data = json.loads(target.read_text())
            data["mean"]["candidates"][recipe["candidate_key"]]["pixel_ap"] -= 8
            target.write_text(json.dumps(data))
            report = compare_weak_source(ROOT, output)
            self.assertEqual(report["status"], "mismatch")
            self.assertFalse(report["fresh_gpu_execution_verified"])
            # Aggregate metrics alone cannot certify complete target coverage.
            data["per_class"].pop()
            target.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "full test split"):
                compare_weak_source(ROOT, output)
