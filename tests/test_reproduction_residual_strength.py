import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from reproduction.residual_strength import compare_residual_strength
from reproduction.recipe_lookup import execution_recipe, reference_summary
from reproduction.metrics import compare, extract_recipe

ROOT = Path(__file__).resolve().parents[1] / 'reproduction'


class ResidualStrengthTests(unittest.TestCase):
    def test_execution_mode_binds_terminal_records_without_claiming_independent_gpu_proof(self):
        import hashlib
        spec = json.loads((ROOT / 'ablations/residual-strength.json').read_text())
        folder = ROOT / 'validation/a10-20261009'
        preparation = json.loads((folder / 'residual-preparation-20261009.json').read_text())
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(ROOT / 'ablations' / spec['archive']) as references, zipfile.ZipFile(folder / preparation['plan_archive']['file']) as plans:
            runs = Path(tmp)
            self.assertEqual(len(compare_residual_strength(ROOT, execution_runs=runs)['pending']), 4)
            for recipe in spec['recipes']:
                # Synthetic record fixtures test bindings, not actual GPU execution.
                workspace = runs / recipe['id']
                (workspace / 'results').mkdir(parents=True)
                summary = references.read(recipe['reference'])
                plan = plans.read('residual-preparation-plans/' + recipe['id'] + '.json')
                (workspace / 'run.json').write_bytes(plan)
                (workspace / 'results/summary.json').write_bytes(summary)
                contract = execution_recipe(ROOT, recipe['id'])
                values = extract_recipe(json.loads(summary), contract)
                comparison = {'actual_sha256':hashlib.sha256(summary).hexdigest(),
                              'reference_sha256':recipe['reference_sha256'], 'cells':compare(values,values)}
                execution = {'recipe':recipe['id'], 'status':'matched','finished':'synthetic-test-only',
                             'returncode':0,'slurm_job_id':'123','plan_sha256':hashlib.sha256(plan).hexdigest(),
                             'comparison':comparison}
                (workspace / 'comparison.json').write_text(json.dumps(comparison))
                (workspace / 'execution.json').write_text(json.dumps(execution))
            report = compare_residual_strength(ROOT, execution_runs=runs)
            self.assertEqual(report['status'], 'matched')
            self.assertTrue(report['terminal_execution_records_verified'])
            self.assertFalse(report['fresh_gpu_execution_verified'])
            self.assertTrue(all(row['execution']['metrics_checked']==44 for row in report['evidence']))
            workspace = runs / spec['recipes'][0]['id']
            execution = json.loads((workspace / 'execution.json').read_text())
            execution['finished'] = None
            (workspace / 'execution.json').write_text(json.dumps(execution))
            self.assertEqual(compare_residual_strength(ROOT, execution_runs=runs)['status'], 'incomplete')
            execution['finished'] = 'synthetic-test-only'
            (workspace / 'execution.json').write_text(json.dumps(execution))
            (workspace / 'run.json').write_bytes((workspace / 'run.json').read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'plan hash differs'):
                compare_residual_strength(ROOT, execution_runs=runs)

    def test_linux_preparation_plans_bind_all_four_fresh_source_runs(self):
        import hashlib
        folder = ROOT / 'validation/a10-20261009'
        report = json.loads((folder / 'residual-preparation-20261009.json').read_text())
        archive = folder / report['plan_archive']['file']
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), report['plan_archive']['sha256'])
        self.assertEqual(archive.stat().st_size, report['plan_archive']['bytes'])
        self.assertEqual(report['status'], 'prepared_four_recipes')
        self.assertFalse(report['gpu_execution'])
        spec = json.loads((ROOT / 'ablations/residual-strength.json').read_text())
        self.assertEqual({r['recipe'] for r in report['rows']}, {r['id'] for r in spec['recipes']})
        with zipfile.ZipFile(archive) as bundle:
            self.assertEqual(len(bundle.namelist()), 5)
            for row in report['rows']:
                payload = bundle.read('residual-preparation-plans/' + row['recipe'] + '.json')
                self.assertEqual(hashlib.sha256(payload).hexdigest(), row['plan_sha256'])
                plan = json.loads(payload)
                recipe = execution_recipe(ROOT, row['recipe'])
                for expected, actual in zip(recipe['argv'], plan['argv']):
                    if expected != '{save_dir}':
                        self.assertEqual(actual, expected)
                self.assertEqual(plan['source_bank_policy'], 'fresh_source_only')
                self.assertEqual(plan['bank_path_changes'], [])
                self.assertEqual(row['bank_cache_entries_before_run'], 0)
                self.assertEqual(row['returncode'], 0)
                self.assertEqual(len(plan['verified_objects']), row['verified_weight_objects'])
                self.assertFalse(plan['fresh_gpu_benchmark'])

    def test_four_launchers_keep_all_numerical_arguments_and_compare_every_candidate(self):
        import copy
        spec = json.loads((ROOT / 'ablations/residual-strength.json').read_text())
        for row in spec['recipes']:
            with self.subTest(recipe=row['id']):
                recipe = execution_recipe(ROOT, row['id'])
                expected = row['archived_launcher_command'][3:]
                expected[expected.index('--save_dir') + 1] = '{save_dir}'
                expected[expected.index('--device') + 1] = 'cuda:0'
                self.assertEqual(recipe['argv'], expected)
                self.assertEqual(recipe['source_bank_policy'], 'fresh_source_only')
                self.assertEqual(recipe['resource_relocation']['archived_device'], 'cuda:1')
                self.assertEqual(recipe['backbone'], 'ViT-L/14 OpenAI')
                reference, summary = reference_summary(ROOT, row['id'])
                self.assertEqual(reference['reference_sha256'], row['reference_sha256'])
                metrics = extract_recipe(summary, recipe)
                self.assertEqual(len(compare(metrics, metrics)), 44)
                changed = copy.deepcopy(summary)
                changed['mean']['candidates']['prescore_calibrated_residual_alpha_2']['pixel_ap'] += 0.1
                cells = compare(extract_recipe(changed, recipe), metrics)
                self.assertEqual(sum(not c['matches_printed_precision'] for c in cells), 1)
                changed['alphas'] = [0, 1]
                with self.assertRaisesRegex(ValueError, 'configuration'):
                    extract_recipe(changed, recipe)

    def test_all_four_archived_transfers_reconstruct_all_five_printed_cells(self):
        report = compare_residual_strength(ROOT)
        self.assertEqual(report['status'], 'matched')
        self.assertFalse(report['fresh_gpu_execution_verified'])
        self.assertEqual([r['printed'] for r in report['cells']], ['+0.0','+0.3','+0.3','-0.0','-5.2'])
        self.assertEqual([r['coverage']['expected_test_images'] for r in report['evidence']], [1725,2162,458,741])

    def test_missing_transfer_or_partial_target_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            result = compare_residual_strength(ROOT, runs)
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(len(result['missing']), 4)
            spec = json.loads((ROOT / 'ablations/residual-strength.json').read_text())
            with zipfile.ZipFile(ROOT / 'ablations' / spec['archive']) as archive:
                for recipe in spec['recipes']:
                    path = runs / recipe['id'] / 'summary.json'
                    path.parent.mkdir()
                    path.write_bytes(archive.read(recipe['reference']))
            self.assertEqual(compare_residual_strength(ROOT, runs)['status'], 'matched')
            path = runs / spec['recipes'][0]['id'] / 'summary.json'
            summary = json.loads(path.read_text())
            summary['per_class'][0]['num_images'] -= 1
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'full test split'):
                compare_residual_strength(ROOT, runs)
