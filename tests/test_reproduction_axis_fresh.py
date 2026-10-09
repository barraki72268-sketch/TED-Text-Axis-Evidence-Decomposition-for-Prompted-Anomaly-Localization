import json
from pathlib import Path
import tempfile
import unittest

from reproduction.axis_fresh import read_evidence


ROOT = Path(__file__).resolve().parents[1] / 'reproduction'


class AxisFreshTests(unittest.TestCase):
    def test_terminal_real_evidence_keeps_printed_agreement_and_array_differences(self):
        manifest, payload = read_evidence(ROOT)
        record = json.loads(payload['figure3-execution.json'])
        self.assertEqual(record['status'], 'mismatch')
        self.assertEqual(record['allocation']['job_id'], '13728')
        self.assertTrue(record['comparison']['all_annotations_match_3dp'])
        self.assertFalse(record['comparison']['all_arrays_exact'])
        self.assertEqual(len(record['comparison']['arrays']), 9)
        self.assertEqual(len(record['comparison']['annotations']), 6)
        self.assertEqual(manifest['pdf_visual_review']['status'], 'reviewed_with_layout_defect_preserved')

    def test_corrupted_published_archive_is_rejected(self):
        manifest, _ = read_evidence(ROOT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / 'validation/a10-20261009/figure3-fresh-v6/manifest.json'
            record.parent.mkdir(parents=True)
            record.write_text(json.dumps(manifest))
            (root / manifest['archive']['path']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'archive hash/size'):
                read_evidence(root)
