import copy
import json
from pathlib import Path
import unittest

from reproduction.coverage import validate_coverage

ROOT = Path(__file__).resolve().parents[1] / "reproduction"


class CoverageTests(unittest.TestCase):
    def test_removed_or_duplicated_class_rejected_for_every_host(self):
        recipes = json.loads((ROOT / "recipes.json").read_text())
        for host in {r["host"] for r in recipes}:
            recipe = next(r for r in recipes if r["host"] == host and r["transfer"].endswith("2btad"))
            data = json.loads((ROOT / "references" / recipe["reference"]).read_text())
            with self.subTest(host=host):
                evidence = validate_coverage(ROOT, recipe, data)
                self.assertEqual(evidence["classes_checked"], 3)
                self.assertFalse(evidence["per_image_execution_verified"])
                key = "per_class" if "per_class" in data else "rows"
                for duplicate in (False, True):
                    altered = copy.deepcopy(data)
                    if duplicate:
                        altered[key].append(altered[key][0])
                    else:
                        altered[key].pop(0)
                    with self.assertRaisesRegex(ValueError, "full test split"):
                        validate_coverage(ROOT, recipe, altered)

    def test_missing_protocol_never_reports_verified_coverage(self):
        evidence = validate_coverage(ROOT, {"transfer": "mvtec2missing"}, {})
        self.assertEqual(evidence["status"], "unverified")
