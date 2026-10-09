import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from reproduction.axis_figure import read_inputs


ROOT = Path(__file__).resolve().parents[1] / 'reproduction'


class AxisFigureTests(unittest.TestCase):
    def test_bound_archive_covers_all_panels_and_preserves_different_contracts(self):
        manifest, data, prior = read_inputs(ROOT)
        self.assertEqual(len(manifest['panels']), 3)
        self.assertEqual(len(data['groups']), 9)
        self.assertEqual(len(prior['rows']), 6)
        for panel in manifest['panels']:
            for key, count in [('inside',7973), ('hard_fp',64329), ('outside',200000)]:
                self.assertEqual(len(data['groups'][panel['group_prefix'] + '_' + key]), count)
        self.assertTrue(all(row['matches_3dp'] for row in prior['rows']))
        self.assertTrue(all(row['collector_absolute_error'] == 0 for row in prior['rows']))
        self.assertTrue(any(row['plotter_minus_collector'] != 0 for row in prior['rows']))
        self.assertIn('pending', prior['final_pdf_layout_provenance'])
        scope = json.loads((ROOT / 'paper-figure-scope.json').read_text())
        figure = next(row for row in scope['figures'] if row['number_in_source_order'] == 3)
        self.assertEqual(manifest['original_paper_graphics'], figure['graphics'])

    def test_public_cpu_audit_is_bound_and_preserves_all_six_numeric_results(self):
        manifest, data, prior = read_inputs(ROOT)
        evidence = manifest['public_cpu_audit']
        raw = (ROOT / evidence['path']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), evidence['sha256'])
        report = json.loads(raw)
        self.assertEqual(report['status'], 'matched')
        self.assertEqual(report['public_runtime_commit'], evidence['runtime_commit'])
        self.assertEqual(report['input_sha256'], manifest['archive']['members']['axis_entanglement_mvtec2visa.json']['sha256'])
        self.assertEqual(len(report['rows']), 6)
        for actual, expected in zip(report['rows'], prior['rows']):
            self.assertEqual(actual['panel'], expected['panel'])
            self.assertEqual(actual['comparison'], expected['comparison'])
            self.assertEqual(actual['plotter_auc'], expected['actual'])
            self.assertEqual(actual['collector_auc'], expected['collector_auc'])
            self.assertEqual(actual['stored_panel_auc'], expected['stored_panel_auc'])
            self.assertTrue(actual['matches_printed_3dp'])
            self.assertTrue(actual['matches_stored_exact'])
        self.assertEqual(report['fresh_model_collection'], 'pending')
        self.assertEqual(report['final_pdf_layout_provenance'], 'pending')

    def test_archive_corruption_is_rejected_before_loading_json(self):
        manifest = json.loads((ROOT / 'figures/axis-entanglement.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'figures').mkdir()
            (root / 'figures/axis-entanglement.json').write_text(json.dumps(manifest))
            (root / manifest['archive']['path']).write_bytes(b'corrupted archive')
            with self.assertRaisesRegex(ValueError, 'archive hash/size'):
                read_inputs(root)
