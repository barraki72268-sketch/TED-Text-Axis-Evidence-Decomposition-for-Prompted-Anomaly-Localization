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
