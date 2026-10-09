import json
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
