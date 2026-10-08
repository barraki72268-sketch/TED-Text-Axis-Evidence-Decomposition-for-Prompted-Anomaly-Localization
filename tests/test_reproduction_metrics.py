import math
import unittest

from reproduction.metrics import METRICS, aggregate, compare, extract


class ReproductionMetricsTests(unittest.TestCase):
    def test_raw_vlm_checks_all_three_methods(self):
        rows = [{"target_label": "mean", "mode": mode, "image_auroc": .7, "pixel_auroc": .8,
                 "pixel_aupro": .6, "pixel_ap": .2} for mode in ["baseline_txt", "parallel_margin", "ted_calibrated"]]
        results = extract({"rows": rows}, "RawImageBind")
        self.assertEqual(set(results), {"Base", "T-TED", "C-TED"})
        self.assertEqual(results["Base"]["P-AP"], 20)
        changed = {key: dict(value) for key, value in results.items()}
        changed["T-TED"]["P-PRO"] += 1
        cells = compare(changed, results)
        self.assertEqual(len(cells), 12)
        self.assertEqual(sum(not cell["matches_printed_precision"] for cell in cells), 1)

    def test_sample_and_population_std_are_distinct(self):
        runs = [{"Base": {metric: value for metric in METRICS}} for value in [1., 2., 3.]]
        self.assertEqual(aggregate(runs, "sample_std")["Base"]["P-PRO"]["std"], 1.)
        self.assertAlmostEqual(aggregate(runs, "population_std")["Base"]["P-PRO"]["std"], math.sqrt(2 / 3))
        with self.assertRaises(ValueError):
            aggregate([], "sample_std")

    def test_faprompt_uses_official_image_score_and_does_not_mutate_input(self):
        baseline = {"image_auc": .8, "pixel_auc": .7, "pixel_aupro": .6, "pixel_ap": .3}
        branch = dict(baseline, image_auc=.2)
        result = extract({"mean": {"baseline_official": baseline, "parallel": {"branch_calibrated_alpha_0.5": branch}}}, "FAPrompt")
        self.assertEqual(result["OURS"]["I-AUROC"], 80.)
        self.assertEqual(branch["image_auc"], .2)


if __name__ == "__main__":
    unittest.main()
