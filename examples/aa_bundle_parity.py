"""Check relocated AA raw maps against saved evaluator maps, CPU-only.

Fixtures JSON: [{"category": "01", "image": "/path/image.png",
"image_sha256": "...", "reference_npz": "/path/01-maps.npz"}].
NPZ files must contain expected_host and expected_cted arrays.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from reproduction.checkpoint_download import digest_file
from ted.inference.aaclip_engine import CapturedAAEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('fixtures', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--forbid-read-root', action='append', type=Path, default=[])
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Set CUDA_VISIBLE_DEVICES to an empty string for this CPU-only check')
    if args.output.exists():
        raise FileExistsError('Use a new report path; preserve prior evidence')
    torch.set_num_threads(4)
    fixtures = json.loads(args.fixtures.read_text())
    if not fixtures:
        raise ValueError('No image fixtures')
    loaded = []
    for entry in fixtures:
        if digest_file(Path(entry['image'])) != entry['image_sha256']:
            raise ValueError('Fixture image hash differs from reference')
        with Image.open(entry['image']) as image:
            rgb = image.convert('RGB').copy()
        with np.load(entry['reference_npz'], allow_pickle=False) as reference:
            loaded.append((entry, rgb, reference['expected_host'].copy(), reference['expected_cted'].copy()))
    forbidden = [p.resolve() for p in args.forbid_read_root]
    def audit(event, values):
        if event == 'socket.connect':
            raise RuntimeError('Relocated inference must not access the network')
        if event == 'open' and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(values[0])).resolve()
            if any(path == root or root in path.parents for root in forbidden):
                raise RuntimeError('Relocated inference tried to read an original workspace')
    sys.addaudithook(audit)
    def no_fit(*args, **kwargs):
        raise RuntimeError('Relocated inference must not fit state')
    torch.optim.Optimizer.__init__ = no_fit
    engine = CapturedAAEngine(export_directory=args.bundle / 'export', workspace=args.bundle, device='cpu')
    for name in dir(engine.host):
        if name.startswith('train_'):
            setattr(engine.host, name, no_fit)
    rows = []
    for entry, image, expected_host, expected_cted in loaded:
        actual = engine.predict_for_category(image, entry['category'])
        host, corrected = actual['host_map'], actual['cted_map']
        if host.shape != expected_host.shape or corrected.shape != expected_cted.shape:
            raise ValueError('Relocated raw-map shape differs')
        rows.append(dict(category=entry['category'], image_sha256=entry['image_sha256'],
            host_max_abs_error=float(np.max(np.abs(host - expected_host))),
            cted_max_abs_error=float(np.max(np.abs(corrected - expected_cted)))))
    passed = all(r['host_max_abs_error'] == 0 and r['cted_max_abs_error'] == 0 for r in rows)
    report = dict(status='matched' if passed else 'mismatch', device='cpu', images=len(rows), rows=rows,
        bundle_manifest_sha256=digest_file(args.bundle / 'serving-bundle.json'), engine=engine.info(),
        network='socket.connect prohibited during engine load and inference',
        original_workspace_reads='prohibited for explicitly supplied roots',
        fit_guard='Optimizer construction and evaluator train_* calls prohibited',
        scope='Raw maps for listed images only; no full-dataset metric, raw image-score, or Docker validation claim.')
    args.output.write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
