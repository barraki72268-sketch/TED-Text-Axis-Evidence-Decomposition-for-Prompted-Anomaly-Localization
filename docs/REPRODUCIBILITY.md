# Reproducibility scope and release status

[Research overview](../README.md) · [Executable guide](../reproduction/README.md) · [Service](SERVICE.md)

Status: **October 9, 2026**.

## What is available

The public package includes the archived research source, hash-pinned historical
summaries, checkpoint/bank acquisition tools, full-dataset input verification,
Linux runtime preparation, and fresh-run comparison. Artifact acquisition and
reference auditing work without PyTorch. Runtime preparation and evaluation
require Linux and the compatible research dependencies; all-loader clean GPU
validation is still in progress.

| Release component | Current evidence |
|---|---|
| Main/host inventory | 232 configurations: 207 adapted-host seed runs and 25 frozen-backbone runs; eight evaluator source versions pinned |
| Weak-source ablation | 28 additional seed-0 configurations, 12 bank objects; all 24 printed gain cells match archived-summary aggregation |
| Host model weights | 11 public hash-verified checkpoint objects |
| Source evidence banks | 92 adapted-host and 10 frozen-backbone public objects; 12 weak-source banks also published and anonymously verified |
| Backbones | All 8 upstream files acquired and matched to pinned byte counts and SHA-256 |
| Dataset manifests | Four full protocols, 23,091 referenced files; MVTec AD 2 still required |
| Fresh run and service records | Selected full-BTAD GPU replay and FAPrompt CPU/API checks; see the linked records below |

Use [reproduction/README.md](../reproduction/README.md) for runnable commands,
[DATASETS.md](../reproduction/DATASETS.md) for data roots, and
[SOURCE-NOTICES.md](../reproduction/SOURCE-NOTICES.md) for upstream components.
These counts describe recorded or verified inputs, not fully reproduced models.

## What remains

The target is all reported models, backbones, five datasets, seeds, ablations,
and empirical figures. It is not satisfied by one successful model or dataset.

- **Full GPU replay:** execute every required configuration from a clean checkout and preserve logs, environment, artifacts, and comparisons.
- **Data access:** obtain readable MVTec AD 2 inputs. The original directory currently denies access to the execution account; no full five-dataset claim is made.
- **Historical provenance:** resolve the 13 printed-mean differences in [unresolved-paper-cells.json](../reproduction/unresolved-paper-cells.json), plus missing historical seed evidence.
- **Numerical mismatches:** retain and investigate ImageBind Blackwell and AA-CLIP H/14 differences; do not select parameters to force agreement with target metrics.
- **Bank rebuilding:** regenerate source-only banks under the traced settings and compare provenance/content; publishing historical banks does not establish fresh rebuilding.
- **Remaining analyses:** map and rerun the other tables, empirical figures, and prose claims. The [34-table inventory](../reproduction/paper-table-scope.json) and [12-figure inventory](../reproduction/paper-figure-scope.json) define the broader scope. The main 232 recipes cover Table 1 and Tables 17–21; the weak-source table is additional.
- **Publication and deployment:** publish remaining verified artifacts and validate all-model inference adapters, model selection, and portable Docker execution.

Mismatches and failed attempts remain evidence. They are not dropped from the
release simply because another configuration reproduces successfully.

## How to interpret the checks

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| Hash/catalog verification | Published bytes and bindings match the pinned archive | Fresh inference or agreement with the paper |
| Archived-summary aggregation | Recorded summaries reproduce the selected printed cells | A new GPU run |
| Fresh-run comparison | The stated run matches its archived per-seed reference at the stated precision | All printed means/stds, seeds, or hardware |
| Full target-coverage check | Reported classes and, where available, image counts match the full test manifest | Independent per-image execution proof |
| API raw-map parity | HTTP output agrees with the stated inference path/input/batch | Whole-dataset metrics or throughput guarantees |

Selected public evidence:

- [Anonymous checkpoint acquisition](../reproduction/validation/2026-10-08/public-checkpoint-download-validation.json).
- [Full dataset byte verification](../reproduction/validation/2026-10-08/public-dataset-validation.json).
- [Eight upstream backbones](../reproduction/validation/public-backbone-download-validation.json).
- [ImageBind BTAD hardware comparison and AA-CLIP mismatch investigation](../reproduction/validation/2026-10-08/report.json).
- [Weak-source archived table reconstruction](../reproduction/validation/weak-source-archived-table-validation.json).
- [FAPrompt deployment and API records](SERVICE.md#evidence-and-limits).

## Experimental workflow

1. Prepare an official host checkpoint and its original input resolution, prompts, selected layers, and post-processing.
2. Specify the source dataset and disjoint target evaluation protocol.
3. Extract source defect patches using source masks and source-normal Hard-FPs using the frozen host's scores.
4. Construct banks with the host-specific image/category/layer caps; retain full-dimensional features.
5. For T-TED, use the text-axis support margin without fitting a TED calibrator. For C-TED, fit the residual using source data only.
6. Freeze bank-selection and calibration settings before target evaluation.
7. Evaluate with the corresponding host's metric and aggregation protocol and report all configured settings, including decreases.

Target masks are for evaluation and clearly labeled post-hoc diagnostics, not for bank construction, correction fitting, or target-specific model selection. Do not interpret target-free calibration as inference without a target image.

## Source calibration reported in the author response

The host is frozen. C-TED uses source defect–Hard-FP pairs with a softplus ranking objective, host-margin preservation, and quadratic regularization. The reported pair mixture is 75% hard and 25% random, with Hard-FP assignments permuted each epoch.

| Host | Rank | Epochs | Maximum sampled source points per calibration unit |
|---|---:|---:|---:|
| AA-CLIP | 8 | 20 | 8,192 |
| AdaCLIP | 8 | 30 | 2,048 |
| FAPrompt | 4 | 30 | 2,048 |

These source-point budgets are **not retained-bank sizes**. Host-native recipes, shared-recipe controls, and compatibility variants must be distinguished. In particular, optimizer settings from a shared-recipe audit should not be silently applied to all submitted host-native experiments. The released [C-TED cores and notes](CTED.md) expose parameterization, loss, pair selection, and host interfaces. The [232 main/host recipes](../reproduction/README.md) and separate 28-configuration weak-source inventory preserve the current mappings. Additional paper analyses remain to be traced; synthetic example settings are not paper recipes.

## Data and third-party terms

The study uses MVTec AD, VisA, MPDD, BTAD, and MVTec AD 2. Obtain original
images from their providers under the applicable terms. Public model weights,
source-derived banks, and code retain upstream restrictions; this repository
does not grant a blanket MIT/Apache or commercial license to every component.
Private conversation history, credentials, and original dataset archives are
not included in the public release.

## Residual-strength archival audit

The residual-strength boundary table now has a separate archival audit:

```bash
python -m reproduction.residual_strength --references
```

All five printed cells are reconstructed from four AA-CLIP seed-0 transfer
summaries with full reported target coverage. Every alpha is retained,
including the decrease at alpha 2; no alpha is selected for deployment by this
audit. The launcher command is reconstructed from its archived source, not
proof of the historical process argv. Fresh replay and bank provenance for
these four configurations remain required. `--runs DIRECTORY` instead checks
all four supplied summaries and rejects missing transfers or partial targets;
summary files alone do not prove fresh GPU execution.

## Reporting a reproduction issue

Include the Git commit, recipe ID, artifact hashes, source/target datasets,
environment, GPU model, seed, exact command, and execution/comparison records.
State whether the issue concerns artifact acquisition, full-data execution,
per-seed agreement, or printed aggregation. Avoid attaching credentials or
restricted original datasets.

### Residual-strength boundary execution (Table14)

Four AA OpenAI L/14, input224, seed0 recipes are now accepted by
`prepare-run`, `run-prepared`, and `compare`. IDs are listed in
[the traced launcher manifest](../reproduction/ablations/residual-strength.json).
Each run preserves the original two blend modes and five alphas; comparison
checks44 metrics (baseline and every candidate), not only alpha1. These recipes
start with an empty source-bank cache and rebuild using the original source-only
settings. Checkpoint/backbone dependencies bind to the corresponding OpenAI
L/14 main recipe; main-table banks are not substituted. The historical logical
GPU ordinal `cuda:1` is relocated to the single Slurm-visible `cuda:0`, recorded
in the preparation report. No numerical parameter is changed.

With full dataset roots and verified weight objects already prepared:

```bash
python -m reproduction prepare-run aaclip_vitl_openai224_mvtec2btad_strength_seed0 \
  ./runs/residual-btad --objects ./inputs --datasets ./dataset-roots.json
# Inside a valid Slurm GPU allocation:
python -m reproduction run-prepared ./runs/residual-btad --require-slurm
python -m reproduction compare aaclip_vitl_openai224_mvtec2btad_strength_seed0 \
  ./runs/residual-btad/results/summary.json
```

Repeat the other three transfer IDs. To assemble the printed five-cell table,
place each fresh `summary.json` under `<runs>/<recipe-id>/summary.json`, then run
`python -m reproduction.residual_strength --runs <runs>`. That assembly checks
summary content and coverage; inspect matching terminal execution records for
fresh-GPU evidence. All four transfers completed under observed Slurm job13725.
The [bound fresh table audit](../reproduction/validation/a10-20261009/residual-fresh-table-audit.json)
checks terminal records, original numeric settings and full reported target
coverage. All five Table14 aggregate cells match at one decimal. Individual
44-metric comparisons at two decimals remain mismatches: BTAD36/44, MPDD31/44,
VisA35/44, MVTec18/44. These are separate comparison scopes; the aggregate result
does not erase the detailed differences. MPDD covers all six target classes and
458 test images; MVTec covers all15 classes and1725; VisA all12 and2162; BTAD
all3 and741. Independent per-image traces are not supplied by these records.

Audit a complete set of four execution workspaces directly with:

```bash
python -m reproduction.residual_strength --executions ./residual-runs
```

This mode requires all four terminal successful evaluations, checks exact
plan hashes, fresh-source bank policy, original numeric arguments, full target
coverage and each44-cell comparison before aggregating the five printed cells.
An in-progress or missing transfer returns `incomplete`. It retains each
transfer's two-decimal mismatches even when the five aggregate cells match at
one decimal. `terminal_execution_records_verified` describes file bindings;
`fresh_gpu_execution_verified` stays false because an offline audit cannot
independently prove a past Slurm allocation. The actual Slurm/log observations
must be inspected alongside this report.

Figure 3 has an independent CPU audit of its archived VisA capsules diagnostic
arrays. After installing NumPy1.25.0 and scikit-learn1.7.2 in Python3.10, run:

```bash
python -m reproduction.axis_figure --references --output figure3-audit.json
```

The command verifies the archive and individual input hashes, then executes only
the two pure AUC functions from each hash-pinned original source member. It checks
all six PDF annotations at three decimals and all six collector AUCs against the
stored panel summary. The collector subsamples to100000 points and handles ties
with sklearn; the wideslim plotter uses full groups and stable ranks without tie
averaging. Their exact differences are retained, rather than substituted.
This is an archival array audit of a diagnostic subset, not fresh model inference
or a whole-dataset evaluation. Fresh collection now has separate evidence below; final PDF style
provenance remains pending. The archive contains derived arrays only; source
dataset terms still apply. Existing audit output files are never overwritten.
The public command passed in a separate Linux Git worktree at commitc1ef6d5,
using an isolated Python3.10 CPU environment with NumPy1.25.0 and sklearn1.7.2.
The [bound CPU report](../reproduction/validation/a10-20261009/figure3-public-cpu-audit.json)
records every calculation and keeps fresh collection and layout status pending.


Figure 3 fresh collection has a separate Linux preparation command:

```bash
CUDA_VISIBLE_DEVICES='' python -m reproduction.axis_collection ./runs/figure3 \
  --checkpoint ./epoch_15.pth --backbone ./ViT-L-14-336px.pt \
  --datasets ./dataset-roots.json
```

Preparation verifies the current traced checkpoint and backbone bytes, all
MVTec/VisA dataset files and the source archive; records path-only relocation;
and creates an empty cache for a fresh source-only bank. It does not run a GPU,
create a PDF or submit a job. The checkpoint is now publicly available at pinned
Hugging Face revisionb84c0603f3aa66b258169adbe6b6c657faed0c25:

```bash
hf download KIMJINYOUNG/TED-reproducibility anomalyclip-figure3-epoch15.pth \
  --revision b84c0603f3aa66b258169adbe6b6c657faed0c25 --local-dir ./figure3-checkpoint
```

Pass the downloaded file to `--checkpoint`. Its size22631975 and
SHA-256415c5dcb52668b8c33fb9c1a351c686d632b919df5b384d63fa9ce7a2338ced4
are verified by preparation. An anonymous download of checkpoint and its two
provenance documents matched all original bytes; this is acquisition evidence,
not fresh inference or proof of the historical training dataset.
The resulting command is reconstructed from the diagnostic JSON and original
parser/model defaults, not a recovered historical command. The historical bank
is missing at its recorded path. Original RNG state and historical checkpoint
hash were not recorded in that JSON, and training data cannot be inferred from
the checkpoint folder name. These gaps remain explicit in the plan. The guarded runner has now completed fresh collection; see the evidence audit below.

After preparation, CPU validation is available without running a model:

```bash
python -m reproduction.axis_run ./runs/figure3 --validate-only
```

Inside a valid GPU Slurm allocation, omit `--validate-only` to collect. The runner
checks the job is RUNNING, GPU resources are allocated, the current host belongs
to that job, GPUs are visible and Slurm tracks the current process in that job.
It revalidates source, assets, full dataset
bytes, metadata, command arguments and empty output/bank directories before
launch. The worker records initial Python/NumPy/Torch RNG states without choosing
a seed or claiming the historical RNG state. Success/failure, logs, all nine
array comparisons and six AUC annotations are retained. Array equality and AUC
agreement are reported separately; a generated PDF alone proves neither.
The collector also creates its own diagnostic PDF and PNG. Final printed layout
provenance remains pending even if the numeric comparisons pass.

Fresh Figure3 collection completed under observed Slurm job13728 with a newly
mined source-only bank. All six AUC annotations match the printed values at
three decimals. All nine array counts match, but their float32 values differ
(maximum absolute error approximately2.02e-6); exact-array status remains
`mismatch`. The historical RNG/checkpoint/bank gaps remain unresolved, and
this diagnostic subset is not a whole-dataset benchmark.

```bash
python -m reproduction.axis_fresh
```

This CPU audit verifies the complete published evidence archive and individual
members, binds terminal execution to its plan/result hashes, recomputes all nine
array comparisons and all six collector AUCs from the fresh arrays using the
hash-pinned original function. Exit1 preserves the exact-array mismatch even
when all printed annotations match. It does not infer a live GPU allocation
from offline records. The evidence includes the original collector PDF/PNG;
visual review found its overall title overlaps panel headings, so final paper
layout reproduction remains pending. Original outputs are retained without
silently fixing the historical collector.
