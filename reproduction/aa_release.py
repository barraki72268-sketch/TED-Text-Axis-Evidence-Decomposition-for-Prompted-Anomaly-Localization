"""Download a pinned public AA serving release and verify all extracted inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from .checkpoint_download import digest_file
from .serving_archive import unpack_aa_bundle


CATALOG = Path(__file__).with_name('aa-serving-releases.json')


def prepare_release(release, destination, cache, catalog=CATALOG):
    specification = json.loads(Path(catalog).read_text(encoding='utf-8'))
    record = specification['releases'][release]
    destination, cache = Path(destination).absolute(), Path(cache).absolute()
    if destination.exists():
        raise FileExistsError('Extraction destination must be new')
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / (record['sha256'] + '.tar.gz')
    url = ('https://huggingface.co/' + specification['repository'] + '/resolve/'
           + record['revision'] + '/' + record['archive'])
    if not archive.exists():
        partial = Path(str(archive) + '.partial')
        digest, size = hashlib.sha256(), 0
        # Exclusive creation preserves interrupted or rejected downloads.
        with partial.open('xb') as output, urllib.request.urlopen(url, timeout=120) as response:
            for block in iter(lambda: response.read(8 * 1024 * 1024), b''):
                size += len(block)
                if size > record['bytes']:
                    raise ValueError('Download exceeds published size')
                digest.update(block)
                output.write(block)
        if size != record['bytes'] or digest.hexdigest() != record['sha256']:
            raise ValueError('Downloaded archive differs from pinned size/hash')
        partial.rename(archive)
    if archive.stat().st_size != record['bytes'] or digest_file(archive) != record['sha256']:
        raise ValueError('Cached archive differs from pinned size/hash')
    result = unpack_aa_bundle(archive, destination, record)
    result.update(release=release, revision=record['revision'], url=url,
                  authentication='none; standard-library HTTPS without auth headers')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', choices=sorted(json.loads(CATALOG.read_text())['releases']))
    parser.add_argument('destination', type=Path)
    parser.add_argument('--cache', type=Path, default=Path('ted-download-cache'))
    args = parser.parse_args()
    print(json.dumps(prepare_release(args.release, args.destination, args.cache), indent=2))
