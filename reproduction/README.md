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

## Reproduction contracts

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
