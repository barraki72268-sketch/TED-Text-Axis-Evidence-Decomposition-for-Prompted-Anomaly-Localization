import json
from pathlib import Path
import unittest

from reproduction.checkpoint_download import digest_file
from reproduction.metrics import compare, extract
from reproduction.recipe_lookup import reference_summary


ROOT = Path(__file__).resolve().parents[1] / 'reproduction'


class PublishedEvidenceTests(unittest.TestCase):
    def test_gateway_evidence_covers_each_pinned_release_and_fixture(self):
        folder = ROOT / 'validation/a10-20261009'
        index = json.loads((folder / 'gateway-evidence.json').read_text())
        self.assertEqual(digest_file(folder / index['file']), index['sha256'])
        report = json.loads((folder / index['file']).read_text())
        registry_path = ROOT.parent / 'deployment/pilab-model-registry.json'
        self.assertEqual(digest_file(registry_path), report['registry_sha256'])
        registry = json.loads(registry_path.read_text())
        releases = {m['id']: m['artifact_sha256'] for m in registry['models']}
        self.assertEqual(report['status'], 'matched')
        self.assertEqual(report['cases'], 9)
        self.assertEqual(len(report['rows']), 9)
        self.assertEqual({(r['model'], r['fixture_category']) for r in report['rows']},
                         {(model, category) for model in releases for category in ['01', '02', '03']})
        for row in report['rows']:
            self.assertEqual(row['artifact_sha256'], releases[row['model']])
            for key in ['host_max_abs_error', 'cted_max_abs_error', 'raw_image_score_abs_error']:
                self.assertEqual(row[key], 0)
        self.assertEqual(report['unknown_model_status'], 404)
        self.assertEqual(report['missing_aa_category_status'], 422)

    def test_aa_serving_evidence_retains_cross_host_difference_and_local_parity(self):
        folder = ROOT / 'validation/a10-20261009'
        index = json.loads((folder / 'aa-serving-evidence.json').read_text())
        for entry in index['files']:
            self.assertEqual(digest_file(folder / entry['file']), entry['sha256'])
        for name in ['aa-main-engine-parity.json', 'aa-main-relocation-parity.json',
                     'aa-pilab-original-math.json', 'aa-pilab-container-local-parity.json',
                     'aa-pilab-main-original-math.json', 'aa-pilab-main-container-local-parity.json']:
            report = json.loads((folder / name).read_text())
            self.assertEqual(report['status'], 'matched')
            self.assertEqual({r['category'] for r in report['rows']}, {'01', '02', '03'})
            for row in report['rows']:
                self.assertEqual(row['host_max_abs_error'], 0)
                self.assertEqual(row['cted_max_abs_error'], 0)
                if 'raw_image_score_abs_error' in row:
                    self.assertEqual(row['raw_image_score_abs_error'], 0)
        difference = json.loads((folder / 'aa-pilab-a10-difference.json').read_text())
        self.assertEqual(difference['status'], 'mismatch')
        self.assertGreater(max(r['host_max_abs_error'] for r in difference['rows']), 0)
        runtime = json.loads((folder / 'aa-pilab-container-runtime.json').read_text())
        self.assertTrue(runtime['read_only'])
        self.assertEqual(runtime['ports']['8000/tcp'][0]['HostIp'], '127.0.0.1')

    def test_a10_published_bytes_and_claims_match_original_execution_records(self):
        folder = ROOT / 'validation/a10-20261009'
        report = json.loads((folder / 'report.json').read_text())
        self.assertEqual(len(report['results']), 7)
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
