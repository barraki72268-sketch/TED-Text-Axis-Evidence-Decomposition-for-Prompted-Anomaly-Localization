import json
from pathlib import Path
import tempfile
import unittest

from reproduction.captured_export import export_captured_run
from reproduction.checkpoint_download import digest_file
from reproduction.source import unpack_source


ROOT = Path(__file__).resolve().parents[1] / 'reproduction'


class CapturedExportTests(unittest.TestCase):
    def fixture(self, base):
        work = base / 'run'
        (work / 'results/artifacts').mkdir(parents=True)
        # Test the export's byte handling without deserializing tensor data.
        artifact = work / 'results/artifacts/001-train_prescore_calibrators.pt'
        artifact.write_bytes(b'opaque-state-bytes-not-a-pickle')
        entries = [{'function': 'train_prescore_calibrators', 'file': artifact.name, 'sha256': digest_file(artifact)}]
        (artifact.parent / 'index.json').write_text(json.dumps(entries))
        unpack_source(ROOT / 'source.zip', ROOT / 'source-manifest.json', work / 'source')
        public = ROOT / 'validation/a10-20261009/aaclip-weak-source-1'
        for name in ('execution.json', 'comparison.json'):
            (work / name).write_bytes((public / name).read_bytes())
        (work / 'results/summary.json').write_bytes((public / 'summary.json').read_bytes())
        execution = json.loads((work / 'execution.json').read_text())
        plan = dict(recipe=execution['recipe'], source_path_changes=[], verified_objects=[])
        (work / 'run.json').write_text(json.dumps(plan))
        execution.update(plan_sha256=digest_file(work / 'run.json'),
                         capture_index_sha256=digest_file(artifact.parent / 'index.json'), captured_files=entries)
        (work / 'execution.json').write_text(json.dumps(execution))
        return work, artifact

    def test_state_export_retains_exact_bytes_and_requires_further_inference_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            work, artifact = self.fixture(base)
            result = export_captured_run(ROOT, work, base / 'export')
            self.assertEqual(result['capture_binding'], 'terminal_execution_record')
            self.assertIn('requires_inference_parity', result['status'])
            self.assertEqual((base / 'export' / result['captured_state'][0]['object_path']).read_bytes(), artifact.read_bytes())
            self.assertFalse((base / 'export/source').exists())
            with self.assertRaises(FileExistsError):
                export_captured_run(ROOT, work, base / 'export')

    def test_changed_capture_index_or_metrics_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            work, artifact = self.fixture(base)
            original = artifact.read_bytes()
            artifact.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Captured state bytes differ'):
                export_captured_run(ROOT, work, base / 'bad-bytes')
            artifact.write_bytes(original)
            index = artifact.parent / 'index.json'
            index.write_text(index.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'Capture index differs'):
                export_captured_run(ROOT, work, base / 'bad-index')
            execution = json.loads((work / 'execution.json').read_text())
            execution['status'] = 'mismatch'
            (work / 'execution.json').write_text(json.dumps(execution))
            with self.assertRaisesRegex(ValueError, 'Only terminal'):
                export_captured_run(ROOT, work, base / 'mismatch')
            self.assertFalse(any((base / n).exists() for n in ['bad-bytes', 'bad-index', 'mismatch']))
