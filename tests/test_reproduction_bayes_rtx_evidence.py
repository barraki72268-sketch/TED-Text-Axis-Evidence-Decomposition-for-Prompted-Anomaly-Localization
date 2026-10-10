import hashlib
import json
from pathlib import Path
import unittest

from reproduction.coverage import validate_coverage
from reproduction.metrics import compare, extract
from reproduction.recipe_lookup import reference_summary


class BayesRTXEvidenceTests(unittest.TestCase):
    def test_original_checkpoint_binding_and_reported_metric_scope(self):
        root = Path(__file__).resolve().parents[1] / 'reproduction'
        base = root / 'validation/rtx-20261010'
        report = json.loads((base / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual(len(report['results']), 6)
        for result in report['results']:
            for item in result['evidence']:
                self.assertEqual(hashlib.sha256((base / item['file']).read_bytes()).hexdigest(), item['sha256'])
            folder = base / Path(result['evidence'][0]['file']).parent
            execution = json.loads((folder / 'execution.json').read_text(encoding='utf-8'))
            run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
            actual = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
            reference, expected = reference_summary(root, result['recipe'])
            cells = compare(extract(actual, reference['host']), extract(expected, reference['host']))
            self.assertEqual(cells, execution['comparison']['cells'])
            self.assertTrue(all(c['matches_printed_precision'] for c in cells))
            self.assertEqual(len(cells), 8)
            coverage = validate_coverage(root, reference, actual)
            self.assertEqual(coverage, result['target_coverage'])
            self.assertEqual(coverage['classes_checked'], 6 if 'mpdd' in result['recipe'] else 3)
            self.assertFalse(coverage['reported_image_counts_checked'])
            self.assertFalse(coverage['per_image_execution_verified'])
            checkpoint = run['argv'][run['argv'].index('--checkpoint_path') + 1]
            self.assertEqual(Path(checkpoint).name, 'epoch_post_15_1.pth')
            self.assertEqual(hashlib.sha256((folder / 'run.json').read_bytes()).hexdigest(), execution['plan_sha256'])
            self.assertEqual(hashlib.sha256((folder / 'summary.json').read_bytes()).hexdigest(), execution['comparison']['actual_sha256'])


if __name__ == '__main__':
    unittest.main()
