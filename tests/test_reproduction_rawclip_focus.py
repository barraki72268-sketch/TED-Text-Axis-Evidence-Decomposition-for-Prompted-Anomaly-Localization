import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from reproduction.rawclip_focus import read_inputs


ROOT = Path(__file__).resolve().parents[1] / 'reproduction'


class RawclipFocusTests(unittest.TestCase):
    def test_original_array_summary_and_submitted_graphic_are_bound(self):
        manifest, payload = read_inputs(ROOT)
        summary = json.loads(payload['rawclip_aggregated_distribution.json'])
        self.assertEqual(summary['counts']['num_images'], 1725)
        self.assertEqual(summary['counts']['num_normal_images'], 467)
        self.assertEqual(summary['counts']['num_anomaly_images'], 1258)
        self.assertEqual(set(summary['summary']), {'baseline', 'ours'})
        self.assertEqual(manifest['plotter_protocol']['plot_subsample_per_group'], 50000)
        scope = json.loads((ROOT / 'paper-figure-scope.json').read_text())
        figure = next(row for row in scope['figures'] if row['number_in_source_order'] == 4)
        self.assertEqual(manifest['original_paper_graphic'], figure['graphics'][0])
        self.assertEqual(len(manifest['source_scripts']), 2)

    def test_cpu_audit_preserves_six_matching_aucs_and_ten_percentile_differences(self):
        manifest, _ = read_inputs(ROOT)
        bound = manifest['public_cpu_audit']
        raw = (ROOT / bound['path']).read_bytes()
        self.assertEqual(len(raw), bound['bytes'])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), bound['sha256'])
        report = json.loads(raw)
        self.assertEqual(report['public_runtime_commit'], bound['runtime_commit'])
        self.assertEqual(report['status'], 'mismatch')
        self.assertEqual(report['numpy'], '1.25.0')
        self.assertEqual(len(report['statistics']), 30)
        self.assertEqual(sum(row['matches_exact'] for row in report['statistics']), 20)
        mismatches = [row for row in report['statistics'] if not row['matches_exact']]
        self.assertTrue(all(row['metric'] in {'p50', 'p95'} for row in mismatches))
        self.assertEqual(len(report['aucs']), 6)
        self.assertTrue(all(row['matches_exact'] for row in report['aucs']))
        self.assertEqual(report['fresh_gpu_collection'], 'pending')

    def test_archive_corruption_does_not_become_valid_numeric_evidence(self):
        manifest, _ = read_inputs(ROOT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'figures/rawclip-focus/manifest.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(manifest))
            (root / manifest['archive']['path']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'archive hash/size'):
                read_inputs(root)
