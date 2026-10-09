"""Archive byte-verified AA/AdaptCLIP/FAPrompt inputs; never load tensors or images."""
import argparse
import gzip
import json
from pathlib import Path, PurePosixPath
import tarfile

from .checkpoint_download import digest_file


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_aa_bundle(bundle: Path, expected_host='AA-CLIP'):
    bundle = bundle.resolve()
    manifest = read(bundle / 'serving-bundle.json')
    exported = read(bundle / 'export/manifest.json')
    execution = read(bundle / 'export/execution.json')
    if (expected_host not in {'AA-CLIP', 'AdaptCLIP', 'FAPrompt'}
            or manifest.get('schema_version') != 1 or manifest.get('host') != expected_host
            or exported.get('host') != expected_host or execution.get('status') != 'matched'
            or execution.get('returncode') != 0 or not execution.get('finished')
            or manifest['recipe'] != exported['recipe'] or exported['recipe'] != execution['recipe']
            or digest_file(bundle / 'export/manifest.json') != manifest['export_sha256']
            or digest_file(bundle / 'run.json') != execution['plan_sha256']
            or manifest['original_plan_sha256'] != execution['plan_sha256']):
        raise ValueError('Bundle does not bind a passing captured execution')
    expected = {}

    def add(name, sha):
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or '..' in relative.parts
                or '\\' in name or ':' in name or relative.as_posix() != name):
            raise ValueError('Invalid serving file path')
        if name in expected and expected[name] != sha:
            raise ValueError('Conflicting serving file identity')
        expected[name] = sha

    for entry in manifest['files']:
        add(entry['path'], entry['sha256'])
    for entry in exported['prepared_source_files']:
        add('source/' + entry['path'], entry['sha256'])
    for entry in exported['evidence']:
        add('export/' + entry['file'], entry['sha256'])
    for entry in exported['captured_state']:
        add('export/' + entry['object_path'], entry['sha256'])
    for name in ['serving-bundle.json', 'export/manifest.json']:
        add(name, digest_file(bundle / name))
    actual = set()
    for path in bundle.rglob('*'):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('Serving archive forbids links and special files')
        if path.is_file():
            actual.add(path.relative_to(bundle).as_posix())
    if actual != set(expected):
        raise ValueError('Unlisted or missing serving files; dataset and private files must stay outside the bundle')
    for name, sha in expected.items():
        if digest_file(bundle / name) != sha:
            raise ValueError('Serving input bytes changed: ' + name)
    return manifest, expected


def pack_aa_bundle(bundle: Path, destination: Path, expected_host='AA-CLIP') -> dict:
    bundle, destination = bundle.resolve(), destination.absolute()
    if destination.exists() or Path(str(destination) + '.partial').exists():
        raise FileExistsError('Archive and partial destination must be new')
    if bundle in destination.parents:
        raise ValueError('Archive must stay outside the original bundle')
    manifest, expected = verify_aa_bundle(bundle, expected_host)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(destination) + '.partial')
    # Deterministic regular-file members without machine/user ownership metadata.
    with partial.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=1) as gz:
        with tarfile.open(fileobj=gz, mode='w|', format=tarfile.PAX_FORMAT) as archive:
            for name in sorted(expected):
                path = bundle / name
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                with path.open('rb') as source:
                    archive.addfile(info, source)
    # Confirm archive payloads, not only the source inventory, before publishing.
    import hashlib
    checked = set()
    with tarfile.open(partial, 'r:gz') as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in checked:
                raise ValueError('Unexpected archived member')
            sha = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(block)
            if sha.hexdigest() != expected[member.name]:
                raise ValueError('Archived payload bytes differ')
            checked.add(member.name)
    if checked != set(expected):
        raise ValueError('Archive lost serving files')
    partial.rename(destination)
    return dict(host=expected_host, recipe=manifest['recipe'], archive=destination.name,
                sha256=digest_file(destination), bytes=destination.stat().st_size,
                files=len(expected), export_sha256=manifest['export_sha256'],
                scope='Exact serving inputs and fitted state; no datasets. Inference and metric evidence remain separate.')


def unpack_aa_bundle(archive: Path, destination: Path, record: dict) -> dict:
    """Verify a published archive and every extracted input without PyTorch."""
    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError('Extraction destination must be new')
    if archive.stat().st_size != record['bytes'] or digest_file(archive) != record['sha256']:
        raise ValueError('Serving archive size/hash differs from the published record')
    import shutil
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        names = set()
        for member in members:
            name = member.name
            relative = PurePosixPath(name)
            if (not member.isfile() or relative.is_absolute() or '..' in relative.parts
                    or '\\' in name or ':' in name or relative.as_posix() != name
                    or not name or name in names):
                raise ValueError('Serving archive contains an unsafe or duplicate member')
            names.add(name)
        if len(names) != record['files']:
            raise ValueError('Serving archive file count differs from the published record')
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream, target.open('xb') as output:
                shutil.copyfileobj(stream, output)
    manifest, expected = verify_aa_bundle(destination, record.get('host', 'AA-CLIP'))
    if manifest['recipe'] != record['recipe'] or manifest['export_sha256'] != record['export_sha256']:
        raise ValueError('Extracted recipe/artifact identity differs from published record')
    return dict(status='verified_serving_inputs', recipe=manifest['recipe'],
                archive_sha256=record['sha256'], files=len(expected),
                scope='Verified bytes and passing-execution bindings only; inference checks remain separate.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--unpack', action='store_true')
    parser.add_argument('--record', type=Path, help='Published archive JSON record, required for --unpack')
    parser.add_argument('--host', choices=['AA-CLIP','AdaptCLIP','FAPrompt'], default='AA-CLIP')
    args = parser.parse_args()
    if args.unpack and args.record is None:
        parser.error('--unpack requires --record')
    result = (unpack_aa_bundle(args.bundle, args.destination, read(args.record)) if args.unpack
              else pack_aa_bundle(args.bundle, args.destination, args.host))
    print(json.dumps(result, indent=2))
