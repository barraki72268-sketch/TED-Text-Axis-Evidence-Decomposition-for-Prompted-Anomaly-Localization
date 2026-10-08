import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.run import run_prepared


class PreparedRunTests(unittest.TestCase):
    def fixture(self, directory):
        root, work = directory / "catalog", directory / "run"
        (root / "references").mkdir(parents=True)
        (work / "source").mkdir(parents=True)
        reference = {"rows": [{"target_label": "mean", "mode": mode,
                     "image_auroc": .9, "pixel_auroc": .9, "pixel_aupro": .9, "pixel_ap": .9}
                    for mode in ("baseline_txt", "parallel_margin", "ted_calibrated")]}
        payload = json.dumps(reference).encode()
        (root / "references/ref.json").write_bytes(payload)
        recipe = {"id": "example", "host": "RawCLIP", "transfer": "mvtec2btad",
                  "evaluator": {"path": "eval.py"}, "argv": ["--save_dir", "OUTPUT", "--seed", "0"],
                  "path_bindings": {"OUTPUT": {"kind": "new_output_directory"}}}
        (root / "execution-recipes.json").write_text(json.dumps({"recipes": [recipe]}))
        (root / "recipes.json").write_text(json.dumps([{"id": "example", "host": "RawCLIP",
                 "reference": "ref.json", "reference_sha256": hashlib.sha256(payload).hexdigest()}]))
        (root / "source-manifest.json").write_text('{"files": []}')
        plan = {"recipe": "example", "cwd": str(work / "source"), "evaluator": str(work / "source/eval.py"),
                "argv": ["--save_dir", str(work / "results"), "--seed", "0"], "environment": {},
                "source_path_changes": [], "bank_path_changes": [], "verified_objects": [], "datasets": {}}
        (work / "run.json").write_text(json.dumps(plan))
        return root, work, reference, plan

    def test_metrics_mismatch_and_failure_are_distinct_and_preserved(self):
        for status in ("matched", "mismatch", "failed"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                root, work, summary, _ = self.fixture(Path(temp))
                if status == "mismatch":
                    summary["rows"][0]["pixel_ap"] = .8
                def execute(*args, **kwargs):
                    if status != "failed":
                        (work / "results").mkdir()
                        (work / "results/summary.json").write_text(json.dumps(summary))
                    return SimpleNamespace(returncode=1 if status == "failed" else 0)
                with patch("reproduction.run.subprocess.run", side_effect=execute):
                    result = run_prepared(root, work)
                self.assertEqual(result["status"], status)
                saved = (work / "execution.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    run_prepared(root, work)
                self.assertEqual((work / "execution.json").read_bytes(), saved)

    def test_changed_evaluation_seed_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root, work, _, plan = self.fixture(Path(temp))
            plan["argv"][-1] = "1"
            (work / "run.json").write_text(json.dumps(plan))
            with patch("reproduction.run.subprocess.run") as launch:
                with self.assertRaisesRegex(ValueError, "argument differs"):
                    run_prepared(root, work)
                launch.assert_not_called()

    def test_cluster_guard_rejects_login_shell(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Slurm allocation"):
                run_prepared(Path("."), Path("."), require_slurm=True)
