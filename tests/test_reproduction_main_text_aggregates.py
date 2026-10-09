from pathlib import Path
import os
import shutil
import tempfile
import unittest

from reproduction.main_text_aggregates import audit


ROOT = Path(__file__).resolve().parents[1] / 'reproduction'


class MainTextAggregateTests(unittest.TestCase):
    def test_full_printed_scope_is_archival_not_fresh_inference(self):
        result = audit(ROOT)
        self.assertEqual(result['status'], 'matched')
        self.assertEqual(result['cells_compared'], 27)
        self.assertEqual(result['cells_matched'], 27)
        self.assertFalse(result['fresh_gpu_execution_verified'])
        self.assertEqual({e['rows'] for e in result['evidence']}, {9, 16})
        self.assertEqual({c['table'] for c in result['cells']},
                         {'tab:failure_conditioned_gains', 'tab:source_memory_controls:b'})

    def test_changed_preserved_values_fail_identity_before_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'ablations/main-text-aggregates'
            shutil.copytree(ROOT / 'ablations/main-text-aggregates', directory)
            os.link(ROOT / 'source.zip', root / 'source.zip')
            path = directory / 'failure_conditioned_gain_20260504.csv'
            path.write_bytes(path.read_bytes().replace(b'Low', b'Mid', 1))
            with self.assertRaisesRegex(ValueError, 'hash/size mismatch'):
                audit(root)
