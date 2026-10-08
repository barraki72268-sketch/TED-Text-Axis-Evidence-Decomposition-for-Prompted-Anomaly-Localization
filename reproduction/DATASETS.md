# Dataset identity and preparation

The release target includes **MVTec AD, VisA, MPDD, BTAD, and MVTec AD 2** in
their reported model/transfer configurations. The four accessible datasets
below now have recorded metadata and complete referenced-file SHA-256 manifests.
MVTec AD 2 remains required: the original image directory is currently unreadable
by the reproduction account, so its image manifest and fresh evaluations are
not yet complete. It must not be treated as a passed or omitted dataset.

| Dataset | Training images | Test images | Referenced image/mask files |
| --- | ---: | ---: | ---: |
| MVTec AD | 3,629 | 1,725 | 6,612 |
| VisA, official 1-class layout | 8,659 | 2,162 | 12,021 |
| MPDD | 888 | 458 | 1,628 |
| BTAD, recorded DatasetNinja layout | 1,799 | 741 | 2,830 |

Files are counted once per role/path. Dataset images are not distributed here.
Acquire datasets under their original terms. The manifests pin the exact bytes
used in the historical experiments; another conversion of the same dataset may
have different file hashes. Such a difference is reported, not silently accepted.

The public commands were tested from a fresh GitHub checkout at commit
`0c7d441afc2640854117b4e475c6af157b9c8a59` against the relocated GPU-server data:
all 23,091 referenced files passed, with zero errors across the four datasets.
The [execution report](validation/2026-10-08/public-dataset-validation.json)
includes input roots, manifest hashes, prepared metadata hashes, counts, and
exit statuses. This validates dataset preparation, not model accuracy.

## Verify local data

These commands use only the Python standard library (Python 3.10 or newer).
For MVTec AD, VisA, and MPDD, use the directory containing category directories:

```bash
python -m reproduction validate-dataset mvtec --images /data/mvtec
python -m reproduction validate-dataset visa --images /data/VisA_pytorch_official/1cls
python -m reproduction validate-dataset mpdd --images /data/MPDD
```

BTAD uses DatasetNinja images and separately generated binary masks:

```bash
python -m reproduction validate-dataset btad --images /data/btad-DatasetNinja --masks /data/btad-converted/masks
```

Its image root contains `train/img` and `test/img`; its mask root contains
`test/01`, `test/02`, and `test/03`. The archived conversion script is included
in the source bundle as `neurips2026/scripts/prepare_btad_meta.py`. Given the
DatasetNinja image/annotation layout, generate masks into a new directory with:

```bash
python ted-source/neurips2026/scripts/prepare_btad_meta.py --input_root /data/btad-DatasetNinja --output_root /data/btad-converted
```

That conversion requires NumPy, OpenCV, and Pillow. The reproduction environment
candidate uses NumPy 1.25.0, OpenCV 4.11.0.86, and Pillow 12.0.0; clean-environment
GPU replay is still pending. Verify the resulting masks with the command above.

Validation checks metadata integrity, class/split counts, the exact set of
referenced files, file lengths, and full SHA-256 hashes. Missing, unreadable,
resized, or modified inputs produce a nonzero exit status and error examples.
Unreferenced extra files are ignored because evaluation follows the pinned
metadata rather than directory enumeration.

## Prepare metadata for relocated data

`prepare-dataset` performs the same full verification and writes a new
`meta.json` with absolute paths plus an `input-verification.json` report:

```bash
python -m reproduction prepare-dataset mvtec --images /data/mvtec --output ./prepared/mvtec
python -m reproduction prepare-dataset btad --images /data/btad-DatasetNinja --masks /data/btad-converted/masks --output ./prepared/btad
```

The destination must not exist. Input images, masks, and original metadata are
never rewritten. Recorded sample order, category assignments, splits, and labels
are preserved. If any input fails verification, no prepared directory is created.

Dataset verification is not model evaluation. A final reproduction must also
verify the model, backbone, source bank, calibration recipe, seed subset, and
fresh prediction metrics. The unified evaluator that consumes these prepared
roots is still being assembled.
