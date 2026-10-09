# Download the saved TED checkpoints and banks

The historical host checkpoints and source banks are already public in
[KIMJINYOUNG/TED-reproducibility](https://huggingface.co/KIMJINYOUNG/TED-reproducibility).
Download and reuse these saved inputs independently of the ongoing fresh GPU
evaluations. The catalogs preserve their original server paths and identify
objects by SHA-256, so repeated references share the same bytes.

| Saved package | Contents | Recipe bindings | Pinned download |
|---|---|---|---|
| Host checkpoints | 11 weight objects, about 594 MB | All 207 AA-CLIP, FAPrompt, AdaptCLIP, AdaCLIP and BayesPFL configurations | [Checkpoint archive](https://huggingface.co/KIMJINYOUNG/TED-reproducibility/resolve/296e20b15e7716c6d9286ca4ebdb8230dbdff140/host-checkpoints-20261008.tar) |
| Adapted-host source banks | 92 objects, about 2.11 GB | All 207 adapted-host configurations | [Host bank archive](https://huggingface.co/KIMJINYOUNG/TED-reproducibility/resolve/cc95c974cf15ee3df6d38d034daf54fca6f3a3bb/host-source-banks-20261008.tar) |
| Frozen-backbone source banks | 10 objects, about 145 MB | All 25 RawCLIP/ImageBind configurations | [Raw bank archive](https://huggingface.co/KIMJINYOUNG/TED-reproducibility/resolve/914f6ab0402a0760dfcb2ec824fd3c46a23cc707/raw-source-banks-20261008.tar) |
| Weak-source experiment banks | 12 objects, about 240 MB | 28 AA-CLIP/FAPrompt configurations | [Weak-source bank archive](https://huggingface.co/KIMJINYOUNG/TED-reproducibility/resolve/404a3215a2a74c96242ec1789fe78fdf5d308796/weak-source-banks-20261009.tar) |

From the repository checkout, use Python 3.10 or newer:

```bash
python -m reproduction prepare-checkpoints ./inputs/host-checkpoints
python -m reproduction prepare-source-banks ./inputs/host-banks --kind host
python -m reproduction prepare-source-banks ./inputs/raw-banks --kind raw
python -m reproduction prepare-source-banks ./inputs/weak-banks --kind weak
```

These commands need neither login nor PyTorch. They verify the pinned archive
and every extracted object before reporting success. Use new destination
directories. Upstream backbone weights are acquired separately with
`python -m reproduction prepare-backbones ./inputs/backbones`.

The original-path and model/seed/transfer bindings are in
[host-checkpoints.json](../reproduction/host-checkpoints.json),
[host-source-banks.json](../reproduction/host-source-banks.json),
[raw-source-banks.json](../reproduction/raw-source-banks.json), and
[weak-source-banks.json](../reproduction/weak-source-banks.json).
Their companion `*-download.json` files specify the immutable public archives.

For evaluation commands and dataset preparation, follow the
[reproduction guide](../reproduction/README.md). For packages containing fitted
serving state and a tested HTTP inference adapter, follow the
[Docker service guide](SERVICE.md).

Saved-input availability, fresh per-seed evaluation, fresh bank/training
rebuilding, and printed mean/standard-deviation verification are separate
checks. This download page describes the saved artifacts and their bindings.
