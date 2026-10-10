# Release status

Updated October 10, 2026. The active reproducibility branch is
[`reproduction/all-models`](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/tree/reproduction/all-models).
Full-paper reproduction and all-model deployment remain in progress.

## Current verified release

- **39 BayesPFL per-seed configurations** across four backbones have all eight archived metrics matching at two decimals. This includes eight corrected VisA replays (B+ and H/14, seeds 0/1/2; OpenAI L/14, seeds 0/1). Their summaries report classes but omit per-class image counts, so this evidence is separate from the full-count index and does not establish all paper means/stds or deployment. See the [RTX metric replay index](../reproduction/validation/rtx-20261010/report.json).

- **31 full-dataset configurations** match every archived metric at the recorded printed precision, with the named model, seed, target classes and reported image counts verified. See the [execution index](../reproduction/validation/a10-20261009/verified-results.json).
- **11 selectable worker configurations** cover FAPrompt, AA-CLIP, AdaptCLIP, RawCLIP and ImageBind. Routing evidence consists of 42 cases for the previous ten-worker registry and three additional ImageBind cases; these are separate validations. See the [service guide](SERVICE.md).
- **27 printed cells** in main-text Table 2 and Table 4(b) can be recomputed from pinned archived CSVs with `python -m reproduction.main_text_aggregates`. This is an archival aggregation audit.
- Public checkpoints, source banks and pinned serving bundles are available from [Hugging Face](https://huggingface.co/KIMJINYOUNG/TED-reproducibility). Anonymous download and isolated image-inference checks are linked in the service guide.

Main-text reproduction and all-model deployment continue. The historical notes below preserve earlier verification boundaries and evidence; use the current execution index and service guide for the latest release.

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
AdaCLIP B/16 v4 also completes with 6/8 matches: both image AUROCs differ,
while all six pixel metrics match at two decimals. Its source bank was rebuilt
after strict cache metadata rejected the relocated historical bank. The original
attempts and fresh execution records are preserved.
AdaCLIP H/14 v4 completes with 4/8 matches; its rebuilt-bank differences are
retained in the same dated evidence report. Neither Ada result is promoted as
a numerically matching deployment candidate.
The A10 differs from the paper's RTX 6000 Ada setup; the cause of these numerical
differences has not been established.

## Deployment and remaining work

| Area | Current boundary |
|---|---|
| Datasets | MVTec AD, VisA, MPDD, and BTAD inputs verified; MVTec AD 2 still needs readable source data |
| Hugging Face | Original source, checkpoints, and banks are public; all-model serving bundles remain in progress |
| pilab Docker | FAPrompt CPU service path verified; its recorded deployment uses a host-mounted Python runtime |
| AA-CLIP worker | Weak-source and main L/14-336 CPU Docker HTTP each match same-pilab original equations on three images. The weak-source cross-A10 comparison retains a small difference |
| Model selection | Three artifact-pinned pilab workers are selectable through a Docker gateway; nine direct-worker/gateway cases match exactly. Other families await validation |
| Complete release | All model/backbone/dataset/seed runs, remaining ablations and figures, fresh bank rebuilding, and all-model deployment remain required |

Continue with the [quick start](../README.md#quick-start),
[full reproduction guide](../reproduction/README.md), or
[scope and provenance notes](REPRODUCIBILITY.md).

### Public AA client verification

Both main L/14-336 seed-0 (input518) and source-limit-1 seed-0 (input224)
serving archives passed anonymous download, pinned SHA/size verification,
933-file verification after new-directory extraction, and isolated CPU
three-image raw-map parity with zero error. See the
[bound client evidence](../reproduction/validation/a10-20261009/aa-anonymous-evidence.json)
and [download commands](SERVICE.md#download-a-pinned-public-aa-release).
These checks do not establish full-dataset CPU metrics or all-model deployment.

AdaCLIP B16/H14/L14 OpenAI v4 fresh full-BTAD evaluations match6/8,4/8,3/8
archived per-seed metrics respectively. All differences and exact summaries
are retained in the [fresh A10 report](../reproduction/validation/a10-20261009/report.json).
