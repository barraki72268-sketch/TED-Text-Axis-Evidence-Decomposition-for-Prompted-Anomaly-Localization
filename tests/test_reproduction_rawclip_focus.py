import json
import hashlib
import ast
import zipfile
from pathlib import Path
import tempfile
import unittest

from reproduction.rawclip_focus import read_inputs


ROOT = Path(__file__).resolve().parents[1] / 'reproduction'


class RawclipFocusTests(unittest.TestCase):
    def test_full_fresh_collection_preserves_counts_aucs_and_numeric_differences(self):
        from reproduction.rawclip_focus_fresh import read_evidence
        manifest, payload = read_evidence(ROOT)
        execution = json.loads(payload['figure4-execution.json'])
        comparison = execution['comparison']
        self.assertEqual(execution['status'], 'mismatch')
        self.assertEqual(execution['allocation']['job_id'], '13735')
        self.assertTrue(comparison['fresh_summary_independently_verified'])
        self.assertEqual(len(comparison['arrays']), 6)
        self.assertTrue(all(row['actual_count'] == 200000 for row in comparison['arrays']))
        self.assertFalse(any(row['exact_match'] for row in comparison['arrays']))
        self.assertEqual(sum(row['matches_archive_exact'] for row in comparison['statistics']), 7)
        self.assertEqual(len(comparison['aucs']), 6)
        self.assertTrue(all(row['matches_archive_3dp'] for row in comparison['aucs']))
        self.assertTrue(all(row['matches_exact'] for row in comparison['counts']))
        self.assertIn('crowded', manifest['pdf_visual_review']['notes'])

    def test_linux_guard_validation_preserves_empty_results_and_bound_logs(self):
        manifest, _ = read_inputs(ROOT)
        binding = manifest['guarded_runner_validation']
        raw = (ROOT / binding['path']).read_bytes()
        self.assertEqual(len(raw), binding['bytes'])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), binding['sha256'])
        report = json.loads(raw)
        self.assertFalse(report['gpu_execution'])
        self.assertFalse(report['execution_record_created'])
        self.assertEqual(report['result_entries'], 0)
        self.assertEqual(report['plan_sha256'], json.loads((ROOT / 'figures/rawclip-focus/linux-preparation.json').read_text())['plan_sha256'])
        self.assertEqual(report['bank_sha256_after'], manifest['public_bank']['sha256'])
        self.assertEqual(report['tests_returncode'], 0)
        self.assertEqual(report['validation_returncode'], 0)
        self.assertEqual(report['outside_allocation_returncode'], 1)
        archive = binding['archive']
        self.assertEqual(hashlib.sha256((ROOT / archive['path']).read_bytes()).hexdigest(), archive['sha256'])
        with zipfile.ZipFile(ROOT / archive['path']) as bundle:
            self.assertEqual(bundle.read('validation.json'), raw)
            for name, expected in report['logs'].items():
                content = bundle.read(name)
                self.assertEqual(len(content), expected['bytes'])
                self.assertEqual(hashlib.sha256(content).hexdigest(), expected['sha256'])

    def test_linux_preparation_binds_inputs_and_keeps_original_numeric_parser_defaults(self):
        manifest, _ = read_inputs(ROOT)
        binding = manifest['linux_preparation']
        raw = (ROOT / binding['path']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), binding['sha256'])
        report = json.loads(raw)
        self.assertFalse(report['gpu_execution'])
        self.assertEqual(report['result_entries'], 0)
        self.assertEqual(report['datasets']['mvtec']['files_verified'], 6612)
        self.assertEqual(sum(report['datasets']['mvtec']['counts']['test'].values()), 1725)
        self.assertEqual(report['assets']['bank']['sha256'], manifest['public_bank']['sha256'])
        self.assertIn('not observed historical argv', report['argument_provenance'])
        spec = report['archive']
        self.assertEqual(hashlib.sha256((ROOT / spec['path']).read_bytes()).hexdigest(), spec['sha256'])
        with zipfile.ZipFile(ROOT / spec['path']) as archive:
            plan_raw = archive.read('figure4-plan.json')
        self.assertEqual(hashlib.sha256(plan_raw).hexdigest(), report['plan_sha256'])
        plan = json.loads(plan_raw)
        self.assertEqual(plan['command'], report['command'])
        collector = next(row for row in manifest['source_scripts'] if row['path'].endswith('generate_rawclip_aggregated_distribution.py'))
        with zipfile.ZipFile(ROOT / 'source.zip') as archive:
            source_raw = archive.read(collector['path'])
        self.assertEqual(hashlib.sha256(source_raw).hexdigest(), collector['sha256'])
        calls = [node for node in ast.walk(ast.parse(source_raw)) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument']
        args = plan['command'][3:]
        relocated = {'--target_root', '--bank_cache_path', '--output_dir', '--device'}
        for call in calls:
            flag = ast.literal_eval(call.args[0])
            default = next((keyword.value for keyword in call.keywords if keyword.arg == 'default'), None)
            if default is None or flag in relocated:
                continue
            try:
                value = ast.literal_eval(default)
            except ValueError:
                continue
            if value is None:
                self.assertNotIn(flag, args)
                continue
            index = args.index(flag) + 1
            expected = value if isinstance(value, list) else [value]
            self.assertEqual(args[index:index + len(expected)], list(map(str, expected)))

    def test_fresh_comparison_recomputes_auc_and_preserves_disagreement(self):
        from reproduction.rawclip_focus_run import compare_results
        try:
            import numpy as np
        except ImportError:
            self.skipTest('Numeric replay test requires the declared NumPy evaluation dependency')
        import io
        _, payload = read_inputs(ROOT)
        summary = json.loads(payload['rawclip_aggregated_distribution.json'])
        with tempfile.TemporaryDirectory() as directory:
            arrays = Path(directory) / 'arrays.npz'
            arrays.write_bytes(payload['rawclip_aggregated_distribution_arrays.npz'])
            # Model-free fixture: use current NumPy statistics so the fresh
            # summary consistency is checked independently of historical NumPy.
            with np.load(io.BytesIO(arrays.read_bytes()), allow_pickle=False) as archive:
                for key in archive.files:
                    values = archive[key]
                    method, kind = key.split('_', 1)
                    summary['summary'][method][kind]['p50'] = float(np.percentile(values, 50))
                    summary['summary'][method][kind]['p95'] = float(np.percentile(values, 95))
            report = compare_results(ROOT, summary, arrays)
            self.assertTrue(report['fresh_summary_independently_verified'])
            self.assertTrue(all(row['exact_match'] for row in report['arrays']))
            self.assertEqual(len(report['statistics']), 30)
            self.assertEqual(len(report['aucs']), 6)
            summary['pairwise_auc']['ours']['abnormal_vs_hard_fp'] = 1.0
            report = compare_results(ROOT, summary, arrays)
            self.assertFalse(report['fresh_summary_independently_verified'])
            self.assertEqual(report['status'], 'mismatch')

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
