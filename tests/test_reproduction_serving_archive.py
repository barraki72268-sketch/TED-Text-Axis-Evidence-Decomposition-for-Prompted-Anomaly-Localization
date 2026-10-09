import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from reproduction.serving_archive import pack_aa_bundle, unpack_aa_bundle


class ServingArchiveTests(unittest.TestCase):
    def fixture(self, root):
        bundle = root / 'bundle'
        bundle.mkdir()
        def write(name, value):
            p = bundle / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
            return hashlib.sha256(p.read_bytes()).hexdigest()
        plan = write('run.json', {'recipe': 'fixture'})
        source = write('source/research.py', b'original source\n')
        weights = write('weights.pt', b'weight bytes\x00')
        state = write('export/objects/state', b'fitted state\x00')
        execution = write('export/execution.json', dict(recipe='fixture', status='matched',
                         returncode=0, finished='dated', plan_sha256=plan))
        exported = write('export/manifest.json', dict(host='AA-CLIP', recipe='fixture',
            prepared_source_files=[dict(path='research.py', sha256=source)],
            evidence=[dict(file='execution.json', sha256=execution)],
            captured_state=[dict(object_path='objects/state', sha256=state)]))
        write('serving-bundle.json', dict(schema_version=1, host='AA-CLIP', recipe='fixture',
            export_sha256=exported, original_plan_sha256=plan,
            files=[dict(path=n, sha256=h) for n, h in [('run.json', plan),
                ('source/research.py', source), ('weights.pt', weights)]]))
        return bundle

    def test_deterministic_archive_preserves_every_regular_file_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self.fixture(root)
            first = pack_aa_bundle(bundle, root / 'one.tar.gz')
            second = pack_aa_bundle(bundle, root / 'two.tar.gz')
            self.assertEqual(first['sha256'], second['sha256'])
            loaded = unpack_aa_bundle(root / 'one.tar.gz', root / 'relocated', first)
            self.assertEqual(loaded['status'], 'verified_serving_inputs')
            self.assertEqual(loaded['files'], first['files'])
            with self.assertRaises(FileExistsError):
                unpack_aa_bundle(root / 'one.tar.gz', root / 'relocated', first)
            with tarfile.open(root / 'one.tar.gz') as archive:
                self.assertEqual(len(archive.getmembers()), first['files'])
                for member in archive:
                    self.assertTrue(member.isfile())
                    self.assertEqual(member.uid, 0)
                    self.assertEqual(archive.extractfile(member).read(), (bundle / member.name).read_bytes())
            with self.assertRaises(FileExistsError):
                pack_aa_bundle(bundle, root / 'one.tar.gz')

    def test_unlisted_private_file_and_modified_input_are_rejected_before_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self.fixture(root)
            extra = bundle / 'private-image.png'
            extra.write_bytes(b'image bytes')
            with self.assertRaisesRegex(ValueError, 'Unlisted'):
                pack_aa_bundle(bundle, root / 'rejected.tar.gz')
            extra.unlink()
            (bundle / 'weights.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                pack_aa_bundle(bundle, root / 'rejected.tar.gz')
            self.assertFalse((root / 'rejected.tar.gz').exists())

    def test_hash_correct_archive_with_traversal_is_rejected_before_writes(self):
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'malicious.tar.gz'
            with tarfile.open(archive, 'w:gz') as out:
                info = tarfile.TarInfo('../outside.txt')
                info.size = 1
                out.addfile(info, io.BytesIO(b'x'))
            record = dict(bytes=archive.stat().st_size, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=1)
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                unpack_aa_bundle(archive, root / 'destination', record)
            self.assertFalse((root / 'destination').exists())
            self.assertFalse((root / 'outside.txt').exists())
