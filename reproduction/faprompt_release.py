"""Acquire a pinned public FAPrompt archive without authentication."""
import argparse
import json
from pathlib import Path

from .aa_release import prepare_release as prepare_serving_release


CATALOG = Path(__file__).with_name('faprompt-serving-releases.json')


def prepare_release(release, destination, cache, catalog=CATALOG):
    record = json.loads(Path(catalog).read_text(encoding='utf-8'))['releases'][release]
    if record.get('host') != 'FAPrompt':
        raise ValueError('Expected a pinned FAPrompt serving release')
    return prepare_serving_release(release, destination, cache, catalog)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', choices=sorted(json.loads(CATALOG.read_text())['releases']))
    parser.add_argument('destination', type=Path)
    parser.add_argument('--cache', type=Path, default=Path('ted-download-cache'))
    args = parser.parse_args()
    print(json.dumps(prepare_release(args.release,args.destination,args.cache),indent=2))
