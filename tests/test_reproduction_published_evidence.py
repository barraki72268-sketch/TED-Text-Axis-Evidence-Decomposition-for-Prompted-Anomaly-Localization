import json
from pathlib import Path
import unittest

from reproduction.checkpoint_download import digest_file
from reproduction.metrics import compare, extract, extract_recipe
from reproduction.recipe_lookup import reference_summary, execution_recipe


ROOT = Path(__file__).resolve().parents[1] / 'reproduction'


class PublishedEvidenceTests(unittest.TestCase):
    def test_residual_btad_fresh_replay_preserves_all_44_comparisons(self):
        self.check_residual_replay('btad', 36, 741)

    def test_residual_mpdd_fresh_replay_preserves_all_44_comparisons(self):
        self.check_residual_replay('mpdd', 31, 458)

    def test_residual_visa_fresh_replay_preserves_all_44_comparisons(self):
        self.check_residual_replay('visa', 35, 2162)

    def check_residual_replay(self, dataset, matched, images):
        folder = ROOT / ('validation/a10-20261009/residual-' + dataset + '-v5')
        execution = json.loads((folder / 'execution.json').read_text())
        comparison = json.loads((folder / 'comparison.json').read_text())
        actual = json.loads((folder / 'summary.json').read_text())
        recipe = execution_recipe(ROOT, execution['recipe'])
        reference, expected = reference_summary(ROOT, execution['recipe'])
        cells = compare(extract_recipe(actual, recipe), extract_recipe(expected, recipe))
        self.assertEqual(comparison, execution['comparison'])
        self.assertEqual(cells, comparison['cells'])
        self.assertEqual(len(cells), 44)
        self.assertEqual(sum(c['matches_printed_precision'] for c in cells), matched)
        self.assertEqual(digest_file(folder / 'summary.json'), comparison['actual_sha256'])
        self.assertEqual(execution['status'], 'mismatch')
        self.assertEqual(execution['returncode'], 0)
        self.assertEqual(execution['slurm_job_id'], '13725')
        self.assertEqual(comparison['target_coverage']['expected_test_images'], images)

    def test_anonymous_aa_release_matches_pinned_archive_and_raw_maps(self):
        folder = ROOT / 'validation/a10-20261009'
        index = json.loads((folder / 'aa-anonymous-evidence.json').read_text())
        releases = json.loads((ROOT / 'aa-serving-releases.json').read_text())['releases']
        for item in index['checks']:
            for entry in item['files']:
                self.assertEqual(digest_file(folder / entry['file']), entry['sha256'])
            report = json.loads((folder / item['files'][0]['file']).read_text())
            pinned = releases[item['release']]
            self.assertEqual(report['status'], 'matched')
            self.assertEqual(report['hf_revision'], pinned['revision'])
            self.assertEqual(report['sha256'], pinned['sha256'])
            self.assertEqual(report['bytes'], pinned['bytes'])
            self.assertTrue(report['authentication'].startswith('none'))
            self.assertEqual(report['unpack']['files'], pinned['files'])
            self.assertEqual(report['unpack']['recipe'], pinned['recipe'])
            parity = json.loads((folder / item['files'][1]['file']).read_text())
            self.assertEqual(report['parity'], parity)
            self.assertEqual(parity['status'], 'matched')
            self.assertEqual(parity['engine']['artifact_sha256'], pinned['export_sha256'])
            self.assertEqual({r['category'] for r in parity['rows']}, {'01', '02', '03'})
            for row in parity['rows']:
                self.assertEqual(row['host_max_abs_error'], 0)
                self.assertEqual(row['cted_max_abs_error'], 0)

    def test_fresh_aa_cpu_environment_and_both_raw_map_checks(self):
        folder = ROOT / 'validation/a10-20261009'
        index = json.loads((folder / 'aa-clean-cpu-evidence.json').read_text())
        for entry in index['files']:
            self.assertEqual(digest_file(folder / entry['file']), entry['sha256'])
        environment = json.loads((folder / 'aa-public-cpu-environment.json').read_text())
        self.assertIsNone(environment['cuda_build'])
        self.assertFalse(environment['cuda_available'])
        self.assertEqual(environment['torch'], '2.9.1+cpu')
        self.assertEqual(environment['freeze_sha256'], digest_file(folder / 'aa-public-cpu-env.freeze.txt'))
        for name, artifact in [('main','58954532ae2ab2097e33b2a46f80011f0412339b098dc95d336bd1d09627fb55'),
                               ('weak','e1576d5236ba7801ef7ef9f6f6d3064e60170b330631106610025399432790d5')]:
            report = json.loads((folder / ('aa-public-cpu-' + name + '-parity.json')).read_text())
            self.assertEqual(report['status'], 'matched')
            self.assertEqual(report['engine']['artifact_sha256'], artifact)
            self.assertEqual(len(report['rows']), 3)
            for row in report['rows']:
                self.assertEqual(row['host_max_abs_error'], 0)
                self.assertEqual(row['cted_max_abs_error'], 0)

    def test_gateway_evidence_covers_each_pinned_release_and_fixture(self):
        folder = ROOT / 'validation/a10-20261009'
        index = json.loads((folder / 'gateway-evidence.json').read_text())
        self.assertEqual(digest_file(folder / index['file']), index['sha256'])
        report = json.loads((folder / index['file']).read_text())
        registry_path = ROOT.parent / 'deployment/pilab-model-registry.json'
        self.assertEqual(digest_file(registry_path), report['registry_sha256'])
        registry = json.loads(registry_path.read_text(encoding='utf-8'))
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
        self.assertEqual(len(report['results']), 12)
        adaptclip = [row for row in report['results'] if row['recipe'] == 'adaptclip-vitl14openai-mvtec2btad-seed0']
        self.assertEqual(len(adaptclip), 1)
        self.assertEqual(adaptclip[0]['metrics_matched_2dp'], 8)
        l336 = [row for row in report['results'] if row['recipe'] == 'adaptclip-vitl336-mvtec2btad-seed0']
        self.assertEqual(len(l336), 1)
        self.assertEqual(l336[0]['metrics_matched_2dp'], 8)
        import hashlib
        import zipfile
        with zipfile.ZipFile(folder / 'adaptclip-l336-btad-v5/preparation-and-results.zip') as archive:
            execution = json.loads(archive.read('execution.json'))
            self.assertEqual(hashlib.sha256(archive.read('run.json')).hexdigest(), execution['plan_sha256'])
            self.assertEqual(archive.read('results/summary.json'), (folder / 'adaptclip-l336-btad-v5/summary.json').read_bytes())
            self.assertTrue(archive.read('execution.log'))
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
