# Release status

Updated October 9, 2026. The active reproducibility branch is
[`codex/all-model-reproduction`](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/tree/codex/all-model-reproduction).
Full-paper reproduction and all-model deployment remain in progress.

## Available components

| Component | What has been verified | Start here |
|---|---|---|
| Research inputs | 920 source files, 11 host checkpoint objects, 92 host banks, 10 frozen-backbone banks, and 12 weak-source banks | [Acquisition guide](../reproduction/README.md) |
| Execution recipes | 232 main/host configurations and 28 weak-source configurations; availability does not imply all reruns have passed | [Recipe guide](../reproduction/README.md) |
| ImageBind replay | Full BTAD: all 12 metrics match the archived reference at 2 decimals on RTX 6000 Ada | [Evidence](../reproduction/validation/2026-10-08/report.json) |
| Fresh A10 replay | AA-CLIP and FAPrompt main L/14-336 configurations, plus AA source-limit-1 and FAP bank-budget-8, full BTAD, seed 0: each matches 8/8 metrics | [Evidence](../reproduction/validation/a10-20261009/report.json) |
| Image inference | FAPrompt Docker/API checks and AA-CLIP captured-state CPU HTTP checks, with three-image raw-map parity for each stated artifact | [Service guide](SERVICE.md) |
| Feature-level code | T-TED and C-TED cores, numerical parity checks, and synthetic examples | [T-TED](TTED.md), [C-TED](CTED.md) |

The [README verification table](../README.md#verified-release) links the detailed
source, download, dataset, and execution records.

## How to interpret verification

- **Paper results** are the author-reported values in the paper and archived references.
- **Archive checks** verify files, recipes, and table aggregation. They do not constitute a fresh GPU run.
- **Fresh reproduction** evaluates the complete named target dataset and compares each metric at the printed precision.
- **Inference parity** compares raw maps for the listed images and artifact. It does not establish full-dataset metric parity.

A matching weak-source ablation does not establish the main configuration.
AA-CLIP main L/14-224 matches 7/8 metrics and B+ matches 6/8 in the published
A10 checks. Differences and failed attempts remain part of the evidence.
The A10 differs from the paper's RTX 6000 Ada setup; the cause of these numerical
differences has not been established.

## Deployment and remaining work

| Area | Current boundary |
|---|---|
| Datasets | MVTec AD, VisA, MPDD, and BTAD inputs verified; MVTec AD 2 still needs readable source data |
| Hugging Face | Original source, checkpoints, and banks are public; all-model serving bundles remain in progress |
| pilab Docker | FAPrompt CPU service path verified; its recorded deployment uses a host-mounted Python runtime |
| AA-CLIP worker | CPU HTTP and relocated-bundle map parity verified for one weak-source configuration; pilab Docker validation remains in progress |
| Complete release | All model/backbone/dataset/seed runs, remaining ablations and figures, fresh bank rebuilding, and all-model deployment remain required |

Continue with the [quick start](../README.md#quick-start),
[full reproduction guide](../reproduction/README.md), or
[scope and provenance notes](REPRODUCIBILITY.md).
