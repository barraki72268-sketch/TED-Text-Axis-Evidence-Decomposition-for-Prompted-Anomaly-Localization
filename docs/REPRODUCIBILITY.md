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

## Reporting a reproduction issue

Include the Git commit, recipe ID, artifact hashes, source/target datasets,
environment, GPU model, seed, exact command, and execution/comparison records.
State whether the issue concerns artifact acquisition, full-data execution,
per-seed agreement, or printed aggregation. Avoid attaching credentials or
restricted original datasets.
