import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from reproduction.serving_archive import pack_aa_bundle, unpack_aa_bundle
from reproduction.aa_release import prepare_release


class ServingArchiveTests(unittest.TestCase):
    def test_imagebind_archive_requires_explicit_host_and_restores_empty_import_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self.fixture(root, 'RawImageBind')
            empty = 'source/ADPretrain/models/ImageBind/weights'
            (bundle / empty).mkdir(parents=True)
            with self.assertRaises(ValueError):
                pack_aa_bundle(bundle, root / 'wrong-host.tar.gz', 'RawCLIP')
            record = pack_aa_bundle(bundle, root / 'imagebind.tar.gz', 'RawImageBind')
            unpack_aa_bundle(root / 'imagebind.tar.gz', root / 'relocated', record)
            self.assertTrue((root / 'relocated' / empty).is_dir())
            self.assertEqual(list((root / 'relocated' / empty).iterdir()), [])
            self.assertEqual((root / 'relocated/source/research.py').read_bytes(),
                             (bundle / 'source/research.py').read_bytes())

    def test_rawclip_archive_requires_explicit_host_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self.fixture(root, 'RawCLIP')
            with self.assertRaises(ValueError):
                pack_aa_bundle(bundle, root / 'wrong-host.tar.gz', 'FAPrompt')
            record = pack_aa_bundle(bundle, root / 'rawclip.tar.gz', 'RawCLIP')
            unpack_aa_bundle(root / 'rawclip.tar.gz', root / 'relocated', record)
            self.assertEqual(record['host'], 'RawCLIP')
            self.assertEqual((root / 'relocated/source/research.py').read_bytes(),
                             (bundle / 'source/research.py').read_bytes())
            self.assertEqual((root / 'relocated/export/objects/state').read_bytes(),
                             (bundle / 'export/objects/state').read_bytes())

    def test_faprompt_archive_requires_explicit_matching_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self.fixture(root, 'FAPrompt')
            with self.assertRaises(ValueError):
                pack_aa_bundle(bundle, root / 'wrong-host.tar.gz', 'AdaptCLIP')
            record = pack_aa_bundle(bundle, root / 'faprompt.tar.gz', 'FAPrompt')
            unpack_aa_bundle(root / 'faprompt.tar.gz', root / 'relocated', record)
            self.assertEqual(record['host'], 'FAPrompt')
            self.assertEqual((root / 'relocated/source/research.py').read_bytes(),
                             (bundle / 'source/research.py').read_bytes())

    def test_public_download_roundtrip_and_offline_cache_reuse(self):
        import io
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = pack_aa_bundle(self.fixture(root), root / 'release.tar.gz')
            record['revision'] = 'a' * 40
            catalog = root / 'catalog.json'
            catalog.write_text(json.dumps(dict(repository='fixture/release', releases={'fixture': record})))
            with patch('reproduction.aa_release.urllib.request.urlopen',
                       return_value=io.BytesIO((root / 'release.tar.gz').read_bytes())) as request:
                report = prepare_release('fixture', root / 'first', root / 'cache', catalog)
                self.assertEqual(report['status'], 'verified_serving_inputs')
                self.assertEqual(request.call_args.args[0],
                                 'https://huggingface.co/fixture/release/resolve/' + 'a' * 40 + '/release.tar.gz')
            with patch('reproduction.aa_release.urllib.request.urlopen', side_effect=AssertionError('Network used')):
                prepare_release('fixture', root / 'second', root / 'cache', catalog)
                with self.assertRaises(FileExistsError):
                    prepare_release('fixture', root / 'second', root / 'cache', catalog)
            cached = root / 'cache' / (record['sha256'] + '.tar.gz')
            cached.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'Cached archive'):
                prepare_release('fixture', root / 'third', root / 'cache', catalog)
            self.assertFalse((root / 'third').exists())

    def test_incomplete_public_download_is_preserved_and_never_extracted(self):
        import io
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = pack_aa_bundle(self.fixture(root), root / 'release.tar.gz')
            record['revision'] = 'a' * 40
            catalog = root / 'catalog.json'
            catalog.write_text(json.dumps(dict(repository='fixture/release', releases={'fixture': record})))
            with patch('reproduction.aa_release.urllib.request.urlopen', return_value=io.BytesIO(b'truncated')):
                with self.assertRaisesRegex(ValueError, 'Downloaded archive'):
                    prepare_release('fixture', root / 'destination', root / 'cache', catalog)
            partial = root / 'cache' / (record['sha256'] + '.tar.gz.partial')
            self.assertEqual(partial.read_bytes(), b'truncated')
            self.assertFalse((root / 'destination').exists())
            with self.assertRaises(FileExistsError):
                prepare_release('fixture', root / 'destination', root / 'cache', catalog)

    def test_adaptclip_archive_binds_host_and_preserves_source_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self.fixture(root, 'AdaptCLIP')
            with self.assertRaises(ValueError):
                pack_aa_bundle(bundle, root / 'wrong-host.tar.gz')
            record = pack_aa_bundle(bundle, root / 'adaptclip.tar.gz', 'AdaptCLIP')
            self.assertEqual(record['host'], 'AdaptCLIP')
            unpack_aa_bundle(root / 'adaptclip.tar.gz', root / 'relocated', record)
            self.assertEqual((root / 'relocated/source/research.py').read_bytes(),
                             (bundle / 'source/research.py').read_bytes())
            import io
            from unittest.mock import patch
            from reproduction.adaptclip_release import prepare_release as prepare_adaptclip
            record['revision'] = 'a' * 40
            catalog = root / 'catalog.json'
            catalog.write_text(json.dumps(dict(repository='fixture/release',releases={'adaptclip':record})))
            with patch('reproduction.aa_release.urllib.request.urlopen',
                       return_value=io.BytesIO((root / 'adaptclip.tar.gz').read_bytes())):
                proof = prepare_adaptclip('adaptclip',root / 'downloaded',root / 'cache',catalog)
            self.assertEqual(proof['status'],'verified_serving_inputs')
            self.assertEqual(proof['authentication'],'none; standard-library HTTPS without auth headers')

    def fixture(self, root, host='AA-CLIP'):
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
        exported = write('export/manifest.json', dict(host=host, recipe='fixture',
            prepared_source_files=[dict(path='research.py', sha256=source)],
            evidence=[dict(file='execution.json', sha256=execution)],
            captured_state=[dict(object_path='objects/state', sha256=state)]))
        write('serving-bundle.json', dict(schema_version=1, host=host, recipe='fixture',
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
