# Full-paper reproduction package — in preparation

The release target is **every model**, including all four frozen CLIP backbones,
ImageBind, AA-CLIP, FAPrompt, AdaptCLIP, AdaCLIP, and BayesPFL. The first public
FAPrompt artifact is not the complete release.

## Available in this working tree

The inventory contains 207 archived adapted-host seed runs and 25 main-table
frozen-backbone runs. The legacy AA-CLIP launch command has also been recovered;
its default-seed behavior is being checked by replay.
The inventory is not a count of fresh successful GPU runs. Additional ablation
tables remain to be mapped to executable recipes.

These commands need only Python 3.10 or newer, without PyTorch:

```bash
python -m reproduction list
python -m reproduction verify-references
python -m reproduction verify-execution-recipes
python -m reproduction list --host BayesPFL
python -m reproduction compare faprompt-vitl14_336-mvtec2btad-seed0 /path/to/fresh/summary.json
python -m reproduction aggregate --host AA-CLIP --backbone "ViT-L/14-336" --transfer mvtec2btad --runs /path/to/fresh-runs
```

`compare` verifies the historical reference's SHA-256, converts the evaluator's
declared units to percentage points, and checks agreement at two decimal places.
It exits with status 1 on a metric mismatch. It does not run inference. Use
`--output comparison.json` to save a new comparison record.

`aggregate` expects `<recipe-id>/summary.json` directories. It requires every
recorded seed in the group, then compares means and standard deviations directly
with the printed adapted-host table. It exits with status 2 if any required run
is missing. Reference agreement and printed-table agreement are separate checks.

The RawCLIP/ImageBind comparison includes **all three** Base, T-TED, and C-TED
rows. Host comparisons use their native score contracts. FAPrompt retains its
official baseline image score in the archived table aggregation; the corrected
map's top-k image score is not silently substituted.

## Archived research source

The full scope is recorded in [paper-table-scope.json](paper-table-scope.json):
34 active tables across the paper and supplements, including wrap tables.
The current 232-recipe inventory covers Table 1 and Tables 17–21. It does not
stand for all ablations, diagnostic tables, figures, or non-tabular claims;
those remaining traces and evaluations are required before a complete release.
All five reported datasets remain in scope.

The separate [figure scope](paper-figure-scope.json) records all 12 active
figures and their 25 original graphic files, with byte hashes and exact source
filename matches where available. One is a conceptual method diagram; the
other 11 contain empirical diagnostics or qualitative results whose generating
data and execution provenance remain required. Preserved PDF hashes do not
prove these analyses have been reproduced. Non-tabular prose claims still need
a separate audit.

The complete captured source collection is included as a hash-pinned archive.
Validate it or extract it into a **new, short path**:

```bash
python -m reproduction verify-source
python -m reproduction unpack-source ./ted-source
```

All 920 file hashes are checked before extraction begins. An existing
destination is never overwritten. Model checkpoints and dataset images are
separate artifacts. See [source notices](SOURCE-NOTICES.md) for component
licenses, historical evaluator versions, and remaining execution preparation.
This command does not run a reproduction benchmark.

## Fresh validation evidence

### Historical host checkpoints

`host-checkpoints.json` links all 207 adapted-host recipes to 11 unique host
checkpoint files (593,619,560 bytes). Each binding is checked against its
hash-pinned archived result, including the separate AA-CLIP image/text adapters.
These are historical model weights; they are not evidence of a successful fresh
benchmark. Backbone weights, source banks, and TED calibrators are separate
requirements and are not included in this host-checkpoint catalog.

```bash
python -m reproduction verify-checkpoints
python -m reproduction prepare-checkpoints ./host-checkpoints
python -m reproduction verify-checkpoints --directory /path/to/checkpoints
python -m reproduction verify-checkpoints --directory /path/to/checkpoints --recipe faprompt-vitl14_336-mvtec2btad-seed0
```

Without `--directory`, only the catalog and its reference bindings are verified.
With it, files must exist at `objects/<sha256>` beneath that directory. Every
required byte hash and size is checked, and missing or modified files cause a
nonzero exit. No checkpoint is deserialized by this command. The complete
11-file host-checkpoint archive is now [public on Hugging Face](https://huggingface.co/KIMJINYOUNG/TED-reproducibility/blob/296e20b15e7716c6d9286ca4ebdb8230dbdff140/HOST-CHECKPOINTS.md).
`host-checkpoints-download.json` pins its revision, download URL, byte size,
and SHA-256. Extract into a new directory, then run the byte verifier above.
Original component licenses apply; this archive does not include source banks,
backbones, or calibrated TED states for all recipes.

`prepare-checkpoints` downloads the pinned archive without login or PyTorch,
checks its hash and every member, and writes only into a new destination. It
keeps the downloaded archive in `ted-download-cache` next to the destination.
For offline use, pass `--archive /path/to/host-checkpoints-20261008.tar`.
An interrupted download is retained as `.tar.part` for diagnosis and will not
be overwritten automatically; move that partial file aside before retrying.
The command writes `input-verification.json` alongside the prepared objects.
The public command at commit `a96f309b0f60e4256d7ed5f3cbe488719e92180f`
completed an anonymous download and preparation of all 11 objects in 38.63
seconds on the validation server, with zero errors. See the
[download execution report](validation/2026-10-08/public-checkpoint-download-validation.json).
An independent [anonymous byte audit](validation/2026-10-08/huggingface-hosts-public-verification.json)
also checked all 29 manifest files at the pinned Hugging Face revision.

A fresh clone at `a5c846b344c3764ea5dc5d2b5957817cc21f54d8` verified the
actual 11 checkpoint objects with zero errors; the execution report is
[`public-checkpoint-validation.json`](validation/2026-10-08/public-checkpoint-validation.json).

Dataset input verification and path preparation are documented in
[DATASETS.md](DATASETS.md). The four currently readable datasets have manifests
covering 23,091 referenced image/mask files. MVTec AD 2 remains part of the
required scope and is blocked on source-data access.

Fresh BTAD execution evidence is available in
[`validation/2026-10-08`](validation/2026-10-08/report.json).
ImageBind on the older RTX 6000 Ada matched all 12 archived Base/T-TED/C-TED
metrics at two decimal places. The Blackwell run matched 6/12, so that hardware
configuration is not marked as an exact reproduction. This is one dataset and
one recipe, not a claim about the full paper. Verify the successful fresh run:

```bash
python -m reproduction compare rawimagebind_mvtec2btad reproduction/validation/2026-10-08/imagebind-rtx6000ada-summary.json
```

AA-CLIP H/14 still has four calibrated-metric mismatches on the older GPU.
Reconstructing the April 28 evaluator from archived edits leaves the same
deviations; the current and reconstructed evaluators produce identical shared
calibration-summary fields. Both failed comparisons are retained. Historical
numerical-environment and source-bank provenance remain under investigation.

The Windows and Linux reference-contract CI checks pass. These checks validate
the verifier and archived inputs; they do not execute the GPU benchmark.

## What remains before this is a complete runnable release

- A portable, pinned evaluation environment and path-independent host code.
- Dataset preparation commands and split/metadata/image manifests.
- Verified checkpoints, banks, calibrators, and their download manifest.
- Fresh GPU runs for every required model/backbone/dataset/seed combination.
- Seed aggregation against printed means and standard deviations.
- Resolution of the historical provenance differences recorded in
  `unresolved-paper-cells.json`.
- All-model inference adapters and Docker service verification.

No completed benchmark claim follows from comparing a reference file with
itself. The final release must include fresh execution records and pass from a
clean checkout outside the original research workspace.

`execution-recipes.json` preserves the replay arguments for all 232 traced
main/host recipes and identifies eight archived evaluator versions by hash.
Its 1,016 explicit filesystem arguments are separate bindings for datasets,
source config files, output directories, or hash-pinned artifacts. The verifier
rejects missing recipes and target class/image limits that would turn these
full-target evaluations into subset checks. These are input contracts; source
internal paths, implicit backbone loading, and bank-cache behavior still need
portable runtime preparation before a public evaluation `run` command is ready.

## Reproduction contracts

`host-source-banks.json` maps the 207 adapted-host recipes to 92 bank objects
(2,109,842,800 bytes). For 139 recipes the bank paths are explicitly recorded
in archived summaries. For the other 68, entries are clearly labeled as replay
of the archived cache-selection algorithm against the recorded input inventory.
All 20 FAPrompt entries found metadata-compatible nominal paths. AA-CLIP found
eight nominal paths and selected the first sorted compatible cache for 40
entries. Its compatibility fields omit the seed, so a seed-specific filename
does not imply a separately mined source bank. Hash-checked, weights-only CPU
inspection validated the metadata of all 81 candidate bank files involved.
Historical directory membership still requires execution-log corroboration;
these bindings are not proof of fresh bank rebuilding or complete GPU replay.

- MVTec AD → VisA/MPDD/BTAD and VisA → MVTec AD are separate protocols.
  The archived BayesPFL MVTec row uses MVTec → MVTec; it must retain that label.
- Use the paper's recorded seed subset. Missing seeds are not zero-valued runs.
- AA-CLIP and BayesPFL use sample standard deviation in the traced tables;
  FAPrompt, AdaCLIP, and AdaptCLIP use population standard deviation.
- Fix source bank identity explicitly. Bank creation can consume random numbers,
  and cache-hit versus cache-miss execution is a separate reproducibility issue.
  Verify rebuilding banks separately from replaying published frozen banks.
- Preserve source-only calibration. Target labels may evaluate predictions, but
  must not select a correction strength, checkpoint, or bank.
- Preserve failed and mismatching runs as evidence. Do not select whichever run
  is closest to the paper.

Historical JSON files retain original path strings for provenance; those strings
are not instructions to create directories on a new machine. Portable execution
configuration is still being assembled.
