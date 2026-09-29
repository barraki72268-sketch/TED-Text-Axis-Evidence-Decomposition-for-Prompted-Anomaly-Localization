# Reproducibility and release notes

[← Research overview](../README.md)

## Current availability

The [feature-level T-TED scoring core](TTED.md), unit tests, and a synthetic example are available. Full image-inference/training pipelines, checkpoints, and benchmark evaluation commands are not included yet. The following describes the experimental workflow and planned release contents, not a full benchmark installation guide.

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

These source-point budgets are **not retained-bank sizes**. Host-native recipes, shared-recipe controls, and compatibility variants must be distinguished. In particular, optimizer settings from a shared-recipe audit should not be silently applied to all submitted host-native experiments. Exact parameterization, loss implementation, hard-pair selection, optimizer configuration, and host insertion points will accompany the code release.

## Planned implementation release

- [x] T-TED feature-level support and margin computation with numerical tests.
- [ ] C-TED calibration modules.
- [ ] Host-specific feature/score adapters and source-bank builders.
- [ ] Explicit source splits, category/image caps, sampling seeds, and bank-size units.
- [x] Tested core environment documented (Python 3.10.19, PyTorch 2.9.1+cu128).
- [ ] Full host/benchmark dependency specification.
- [ ] Checkpoint acquisition instructions and source-calibrated artifacts where redistributable.
- [ ] Exact commands for primary result tables and labeled diagnostic experiments.
- [ ] Per-setting metrics and configuration-to-paper mapping.
- [ ] Visualization scripts with paired color normalization and selection provenance.
- [ ] License and third-party notice review.

No release date is promised here. This list will be updated as artifacts become available.

## Data and third-party assets

The study uses MVTec AD, VisA, MPDD, BTAD, and the separately labeled MVTec AD 2 diagnostic. Dataset images shown in figures remain subject to their original terms. No dataset archive or third-party checkpoint is redistributed in this documentation release. Obtain data and host weights from their official providers and follow their access conditions.

A software license has not yet been selected. Do not infer an MIT or Apache license from the repository being public. Upstream code, models, and datasets retain their own licenses.

## Reporting a reproduction issue

Once the implementation is released, include the commit, host/checkpoint, source and target datasets, environment, seed, full command, and logs when opening an issue. Avoid uploading private data, credentials, or restricted datasets.
