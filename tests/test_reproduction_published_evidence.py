import json
from pathlib import Path
import unittest

from reproduction.checkpoint_download import digest_file
from reproduction.metrics import compare, extract
from reproduction.recipe_lookup import reference_summary


ROOT = Path(__file__).resolve().parents[1] / 'reproduction'


class PublishedEvidenceTests(unittest.TestCase):
    def test_a10_published_bytes_and_claims_match_original_execution_records(self):
        folder = ROOT / 'validation/a10-20261009'
        report = json.loads((folder / 'report.json').read_text())
        self.assertEqual(len(report['results']), 6)
        for row in report['results']:
            with self.subTest(recipe=row['recipe']):
                for item in row['evidence']:
                    self.assertEqual(digest_file(folder / item['file']), item['sha256'])
                run = (folder / row['evidence'][0]['file']).parent
                execution = json.loads((run / 'execution.json').read_text())
                comparison = json.loads((run / 'comparison.json').read_text())
                self.assertEqual(digest_file(run / 'summary.json'), execution['comparison']['actual_sha256'])
                self.assertEqual(comparison, execution['comparison'])
                reference, expected = reference_summary(ROOT, row['recipe'])
                actual = json.loads((run / 'summary.json').read_text())
                cells = compare(extract(actual, reference['host']), extract(expected, reference['host']))
                self.assertEqual(cells, comparison['cells'])
                matched = sum(c['matches_printed_precision'] for c in cells)
                self.assertEqual(matched, row['metrics_matched_2dp'])
                self.assertEqual(len(cells), row['metrics_checked'])
                self.assertEqual(execution['status'], 'matched' if matched == len(cells) else 'mismatch')
