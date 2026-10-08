import json
from pathlib import Path
import shutil
import tempfile
import unittest

from reproduction.execution import verify_execution_recipes


class ExecutionScopeTests(unittest.TestCase):
    def fixture(self, root):
        source = Path(__file__).resolve().parents[1] / "reproduction"
        for name in ("execution-recipes.json", "recipes.json", "source-manifest.json"):
            shutil.copyfile(source / name, root / name)
        return json.loads((root / "execution-recipes.json").read_text())

    def test_missing_recipe_cannot_pass_complete_inventory_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = self.fixture(root)
            data["recipes"].pop()
            (root / "execution-recipes.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "full reference inventory"):
                verify_execution_recipes(root)

    def test_target_subset_cannot_masquerade_as_full_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = self.fixture(root)
            data["recipes"][0]["argv"].append("--target_limit_per_class=1")
            (root / "execution-recipes.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "restricts target coverage"):
                verify_execution_recipes(root)
