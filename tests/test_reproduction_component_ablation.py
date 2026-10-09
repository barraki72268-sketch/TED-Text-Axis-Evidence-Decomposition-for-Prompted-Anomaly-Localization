import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from reproduction.component_ablation import compare_component_ablation


ROOT = Path(__file__).resolve().parents[1] / "reproduction"


class ComponentAblationTests(unittest.TestCase):
    def test_matching_known_cells_does_not_hide_unresolved_row(self):
        report = compare_component_ablation(ROOT)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["matching_cells"], 37)
        self.assertEqual(report["required_printed_cells"], 40)
        self.assertEqual(report["unresolved_rows"][0]["label"], "T-TED")
        self.assertFalse(report["fresh_gpu_execution_verified"])

    def test_missing_or_changed_supplied_summaries_are_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = Path(temp)
            self.assertEqual(len(compare_component_ablation(ROOT, runs)["missing_summaries"]), 6)
            manifest = json.loads((ROOT / "ablations/btad-component.json").read_text())
            with zipfile.ZipFile(ROOT / "ablations" / manifest["archive"]) as bundle:
                for ref in manifest["references"]:
                    path = runs / ref["reference"]
                    path.parent.mkdir()
                    path.write_bytes(bundle.read(ref["reference"]))
            ref = next(r for r in manifest["references"] if r.get("variant") == "rank2")
            path = runs / ref["reference"]
            summary = json.loads(path.read_text())
            summary["mean"]["baseline"]["pixel_ap"] += 1
            path.write_text(json.dumps(summary))
            self.assertEqual(compare_component_ablation(ROOT, runs)["status"], "mismatch")
            summary["per_class"].pop()
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "full test split"):
                compare_component_ablation(ROOT, runs)
