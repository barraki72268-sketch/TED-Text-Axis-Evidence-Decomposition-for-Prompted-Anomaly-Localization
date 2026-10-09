import json
from pathlib import Path
import unittest

from reproduction.recipe_lookup import execution_recipe, reference_summary
from reproduction.metrics import extract
from reproduction.coverage import validate_coverage
from reproduction.banks import verify_banks


ROOT = Path(__file__).resolve().parents[1] / "reproduction"


class WeakExecutionTests(unittest.TestCase):
    def test_all_28_keep_launcher_arguments_and_full_dataset_references(self):
        manifest = json.loads((ROOT / "ablations/weak-source.json").read_text())
        for weak in manifest["recipes"]:
            with self.subTest(recipe=weak["id"]):
                execution = execution_recipe(ROOT, weak["id"])
                expected = weak["archived_launcher_command"][3:]
                expected[expected.index("--save_dir") + 1] = "{save_dir}"
                self.assertEqual(execution["argv"], expected)
                self.assertEqual(execution["bank_kind"], "weak")
                self.assertEqual(execution["transfer"], weak["preset"])
                reference, summary = reference_summary(ROOT, weak["id"])
                self.assertEqual(reference["reference_sha256"], weak["reference_sha256"])
                self.assertEqual(validate_coverage(ROOT, execution, summary)["status"], "reported_coverage_matches")
                self.assertEqual(set(extract(summary, weak["host"])), {"Base", "OURS"})
        self.assertEqual(verify_banks(ROOT, "weak")["recipes_bound"], 28)

    def test_unknown_execution_does_not_fall_back_to_a_main_recipe(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            execution_recipe(ROOT, "missing-source-budget")
