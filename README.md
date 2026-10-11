# TED: Text-Axis Evidence Decomposition for Prompted Anomaly Localization

<p align="center">
  <strong>NeurIPS 2026 · Main Track · Poster</strong><br>
  JinYoung Kim · Geonho Kim · GiJeong Park · Geonu Lee · Youngjoon Yoo<br>
  <sub>Corresponding author: Youngjoon Yoo</sub>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.39033">Paper / OpenReview</a> ·
  <a href="https://huggingface.co/KIMJINYOUNG/TED-reproducibility">Models & artifacts (Hugging Face)</a> ·
  <a href="#method">Method</a> ·
  <a href="#results">Results</a> ·
  <a href="#qualitative-results">Visualizations</a> ·
  <a href="#code-and-reproduction">Code & Reproduction</a> ·
  <a href="#citation">Citation</a>
</p>

Official repository for **TED**, a post-hoc local scoring method for prompted anomaly localization.
The paper experiments are complete; public code packaging and reproduction checks are proceeding in stages.

## Public artifacts and current reproduction release

[**Download checkpoints, source banks and verified serving bundles on Hugging Face**](https://huggingface.co/KIMJINYOUNG/TED-reproducibility).

The latest execution records, pinned download clients, Docker worker guides and
model selector are on the [**active reproduction branch**](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/tree/reproduction/all-models).
Start with its [reproduction guide](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/blob/reproduction/all-models/reproduction/README.md)
or [serving and download guide](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/blob/reproduction/all-models/docs/SERVICE.md).

Eleven model/configuration workers have passed their documented image-inference
and routing checks (42 cases for the earlier ten-worker registry, plus three
ImageBind cases checked separately). The active release lists
[31 verified full-dataset configurations](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/blob/reproduction/all-models/reproduction/validation/a10-20261009/verified-results.json)
with metric, seed, class coverage and execution evidence. Full reproduction across all five datasets, backbones,
seeds, ablations and figures is still in progress. New releases list verified
configurations with their exact model, dataset, seed and metric scope; artifact
availability alone does not establish reproduction. The sections below retain the earlier
main-branch implementation scope; use the active release guides for current commands.

## News

- **September 2026:** TED has been accepted to **NeurIPS 2026 Main Track as a Poster**!

## Method

<p align="center">
  <img src="assets/method.jpg" alt="TED overview: frozen host features, defect and hard-FP evidence banks, text-axis support, and T-TED/C-TED readouts" width="100%">
</p>

TED compares source defect support with source hard false-positive (Hard-FP) support along the host's normal-to-anomaly text axis. The backbone and prompts remain frozen.

1. **Keep the host fixed.** Obtain patch features and the host's normal/anomaly text embeddings.
2. **Construct two source-only banks.** Collect source defect patches using source masks and mine high-scoring source-normal patches as Hard-FPs.
3. **Compare evidence on the text axis.** Project both the query and bank features onto the same normal-to-anomaly text direction and contrast their support.
4. **Choose the readout.** Use the support margin directly with T-TED, or add a bounded source-calibrated correction to the host score with C-TED.

| | T-TED | C-TED |
|---|---|---|
| Main use | Frozen VLMs with raw prompt scoring | Adapted CLIP-AD hosts |
| Local score | Defect support minus weighted Hard-FP support | Host score plus bounded source-trained residual |
| TED calibration training | None | Source-only calibration |
| Backbone and prompts | Frozen | Frozen |
| Target-domain training or calibration | None | None |

Banks store full-dimensional features; query and bank support are computed in the same text-axis coordinate. Axis selection and insertion follow the host interface; see the [C-TED implementation notes](docs/CTED.md#inference-and-host-integration). Both variants use source evidence, including source defect annotations; **train-free does not mean source-free**. Target labels, masks, and performance are not used for bank construction or calibration.

Unlike generic Full-D memory scoring, TED contrasts two source evidence roles on the text axis and preserves the host-score anchor in C-TED. [Same-bank retrieval controls](docs/RESULTS.md#same-bank-retrieval-controls) examine this distinction.

## Results

The study covers frozen CLIP backbones and ImageBind, adapted CLIP-AD hosts, and cross-dataset localization on MVTec AD, VisA, MPDD, and BTAD. MVTec AD 2 is additionally used for the frozen-backbone diagnostic.

### C-TED on adapted hosts

The following is the author-reported aggregation of **76 settings from the complete adapted-host result tables**, also summarized in the author response. Values are changes from each corresponding host in **absolute percentage points**, not absolute performance or relative percentages.

| Host | ΔP-AUC | ΔP-PRO | ΔP-AP |
|---|---:|---:|---:|
| AA-CLIP | +9.07 | +6.24 | +1.07 |
| AdaCLIP | +0.38 | +5.47 | −0.14 |
| AdaptCLIP | +0.08 | +0.07 | +0.34 |
| FAPrompt | +3.47 | +8.33 | +10.28 |
| BayesPFL | −0.02 | −0.02 | −0.08 |
| **Overall (76 settings)** | **+2.25** | **+3.90** | **+2.36** |

The overall row averages settings, not the five host means. These are paper-reported results, not a new benchmark run from this repository. Host-native results must not be mixed with separately reported shared-interface controls.

Gains depend on the host and metric: BayesPFL is a near-no-op boundary case, and AdaCLIP's average P-AP slightly decreases. Source mismatch and host-interface compatibility can limit improvements; better pixel localization does not guarantee better image-level screening.

[More: same-bank controls and measured inference overhead →](docs/RESULTS.md)

## Qualitative results

<p align="center">
  <img src="assets/qualitative.jpg" alt="Five selected FAPrompt examples: input with red ground-truth boundaries, host heatmap, and C-TED heatmap" width="760">
</p>

**Selected illustrative examples, all using FAPrompt—not an unbiased sample of the test set.** Rows show MPDD, BTAD, two MVTec AD carpet examples, and VisA fryum. The source is MVTec AD for MPDD/BTAD/VisA and VisA for MVTec AD.

Each Host/C-TED pair shares the same color scale, using the pooled 2nd–99.5th percentile range of the two score maps. Red contours indicate ground-truth defects. These selected improvement examples illustrate local behavior; aggregate results and limitations should be considered alongside them.

## Defect and Hard-FP bank construction

The original collectors for all seven host families are available in the [bank-construction guide](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/blob/reproduction/all-models/docs/BANK_CONSTRUCTION.md), with direct code links, complete-source extraction commands and source-only selection details. A runnable BayesPFL B/16+ source-bank builder has now completed MVTec collection for seed 0 inside a verified GPU allocation; the guide includes commands and tensor-audit evidence. Evaluation using this newly built bank and fresh reconstruction across all paper configurations are being verified separately. Saved banks are also available on [Hugging Face](https://huggingface.co/KIMJINYOUNG/TED-reproducibility).

For this BayesPFL configuration, CPU preparation using only the model checkpoint,
backbone and MVTec source data is also verified. The guide documents
`prepare-run --bank-collection-only`, which requires no saved-bank objects or
BTAD inputs. GPU collection on this new preparation and cross-dataset evaluation
with its newly generated bank remain in progress.

For the verified B/16+ MVTec seed-0 GPU collection, all eight newly collected
Hard-FP/Defect layer tensors exactly match the SHA-verified historical bank
(maximum absolute error 0). Serialized file hashes differ; target evaluation
using the fresh bank is still pending. See the
[comparison evidence](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/blob/reproduction/all-models/reproduction/validation/fresh-bank-20261011/bayes-historical-tensors-v1/index.json).

## Code and reproduction

**Completed experiments and public code availability are separate.** The table below describes release progress, not experiments still to be run.

| Component | Status |
|---|---|
| Research overview and selected figures | Available |
| Reported result summaries | Available |
| T-TED feature-level scoring core | [Available and tested](docs/TTED.md) |
| C-TED calibration and feature-level inference | [AA-CLIP, AdaCLIP, FAPrompt cores available and CPU-tested](docs/CTED.md) |
| Source-bank construction code | [Original collectors and construction guide available](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/blob/reproduction/all-models/docs/BANK_CONSTRUCTION.md); all-configuration fresh reconstruction checks in progress |
| Full benchmark environment and evaluation commands | Packaging and reproduction checks pending |
| Checkpoints / calibrated residuals | Public release in preparation |
| Final camera-ready / arXiv link | To be added when available |

### Try C-TED

The C-TED cores retain the original **host-specific** training and readout functions:

| Host | Code |
|---|---|
| AA-CLIP | [`ted/cted/aaclip.py`](ted/cted/aaclip.py) |
| AdaCLIP | [`ted/cted/adaclip.py`](ted/cted/adaclip.py) |
| FAPrompt | [`ted/cted/faprompt.py`](ted/cted/faprompt.py) |

With PyTorch and NumPy installed, run from the repository root:

```bash
python -m examples.cted_synthetic --host all
python -m unittest discover -s tests -v
```

These examples use synthetic features, not paper benchmark data. Source calibration
and feature-level inference were checked against the original functions. An
experimental FAPrompt image-inference bridge is now available below; standalone
host packaging, bank mining, and full benchmark reproduction remain pending.
See [C-TED training, input contracts, and host integration](docs/CTED.md).

### Try the T-TED core

From the repository root, with PyTorch installed:

```bash
python -m unittest discover -s tests -v
python -m examples.tted_synthetic
```

CPU validation used Python 3.10.19 and PyTorch 2.9.1+cu128; GPU validation is not complete.
The released module scores supplied features. It does not include a full image detector, and the synthetic example does not reproduce a paper result.
See [T-TED usage](docs/TTED.md), [verification](docs/VERIFICATION.md), and the
[reproducibility notes](docs/REPRODUCIBILITY.md).
Experiments described in the paper should not be confused with currently available repository functionality.

## TED Vision Inspection Service — engineering track

**Goal:** extend the research implementation into a tested inference service with
FastAPI, Docker, cloud deployment, model lifecycle management, and an optional
LangChain-assisted inspection report. **This is a work in progress, not a claim
of completed AWS deployment or production readiness.** The research results
above remain separate from service validation and performance measurements.

```text
Offline: source banks → branch calibration → versioned inference artifact
Online:  image → validation → persistent host + TED engine → maps / score
Planned: FastAPI → Docker → AWS deployment → logs / metrics / rollback
Optional: inspection result + retrieved documents → LangChain report with sources
```

### Implemented: offline export and reusable local inference

- [`export_artifact`](ted/inference/faprompt.py) fits two source-only FAPrompt
  branch calibrators and stores their parameters, fixed-axis bank projections,
  interface configuration, training settings, and source-bank/checkpoint hashes.
- [`FAPromptEngine`](ted/inference/engine.py) loads the host and artifact once,
  checks the checkpoint/evaluator hashes and source axis, and reuses the model
  for single-image inference. Overlapping calls are rejected rather than queued
  without a bound. It does not fit models or read target masks during prediction.
- [`CLI`](ted/inference/cli.py) exports artifacts and writes raw Host/C-TED maps,
  paired grayscale previews, artifact identity, and timing metadata. Existing
  artifacts/output directories are not silently overwritten.
- Fixed-axis projections and branch directions are cached. Cached readout
  parity is tested against uncached core composition; this is not yet a measured
  speedup claim. Changing the host or axis requires re-exporting the artifact.

**Current dependency boundary:** image inference still imports the trusted
`neurips2026/scripts/official_parallel_test_faprompt.py` evaluator and its modified
FAPrompt checkout from the original research workspace. It is a local bridge,
not an independently reproducible image detector from this GitHub clone alone.
Dataset iteration is not called by the bridge, but evaluator import dependencies
remain. Upstream extraction, dependency locking, and container portability are
required before deployment. Do not use untrusted Python/checkpoint files.

With the original host environment and cached backbone weights available:

```bash
# Explicit alpha: the research evaluator swept strengths; this is not a
# universal configuration for reproducing every paper result.
python -m ted.inference.cli export \
  --research-root /path/to/research-checkout \
  --bank /path/to/source_branchscore_bank.pt \
  --checkpoint /path/to/faprompt/epoch_15.pth \
  --alpha 1.0 --output artifacts/faprompt-source-v1.pt

python -m ted.inference.cli infer \
  --research-root /path/to/research-checkout \
  --artifact artifacts/faprompt-source-v1.pt \
  --checkpoint /path/to/faprompt/epoch_15.pth \
  --image /path/to/inspection.png \
  --output-dir outputs/inspection-001 --device cpu
```

The exporter uses stored branch scores and rejects absent scores rather than
silently replacing them with axis scores. Current prompt settings are depth 9,
12 context tokens, and 4 compound context tokens; use compatible source banks
and checkpoints. A cache created with other prompt settings must not be reused.
Hash/axis checks reduce accidental mismatches but do not attest the source bank's
historical provenance or hash every upstream dependency. The host loader may
download backbone weights when its cache is missing.

On October 7, 2026, CPU and reserved RTX PRO 5000 GPU paths were exercised with
a real MPDD image, MVTec-source banks, and ViT-L/14@336px FAPrompt. A real
loopback HTTP server returned three predictions whose raw Host/C-TED maps
exactly matched the GPU CLI output. This is single-image smoke validation,
not full benchmark reproduction, a precision study, or a service load test.
The returned image score is the unchanged official FAPrompt score, **not a
calibrated defect probability**. Display-normalized maps are not decision masks;
no default pass/fail decision or binary defect mask is invented.

### Private HTTP service (research dependencies still required)

The [FastAPI adapter](ted/inference/api.py) initializes the model once using
[lifespan](https://fastapi.tiangolo.com/advanced/events/). It provides `/health`,
`/ready`, `/model-info`, and `/predict`, caps image uploads at 20 MB / 25 million
pixels, rejects concurrent predictions with HTTP 503, and logs request IDs and
timing without storing submitted images. This is **not an authenticated public
endpoint**. Keep the binding private; external deployment requires access
control, TLS, ingress limits/timeouts, and deployment approval.

```bash
pip install -r deployment/requirements-api.txt
export TED_ARTIFACT=/path/to/calibrated-artifact.pt
export TED_CHECKPOINT=/path/to/faprompt/epoch_15.pth
export TED_RESEARCH_ROOT=/path/to/research-checkout
export FAPROMPT_CACHE_DIR=/path/to/cached-backbone-directory
export TED_DEVICE=cuda:0  # only inside an authorized GPU allocation
python -m uvicorn ted.inference.api:create_app --factory \
  --host 127.0.0.1 --port 8000 --workers 1
```

In a second shell on the same server, send raw image bytes (not multipart):

```bash
curl --fail http://127.0.0.1:8000/predict \
  -H 'Content-Type: image/png' --data-binary @inspection.png
```

The JSON response includes unchanged host image score, artifact hash, timings,
lossless raw maps in a base64 NPZ (`host`, `cted`, float32; load with
`allow_pickle=False`), and paired grayscale PNG previews using a common display
range. The [HTTP smoke runner](examples/service_smoke.py) starts a loopback
server, checks three requests against saved CLI maps, and stops the server.

### Deployment milestones and acceptance checks

| Stage | Deliverable | Evidence required before marking complete |
|---|---|---|
| 1. Inference foundation | Persistent engine, offline artifact, single-image CLI | Local smoke and readout parity tests; available with research dependency |
| 2. Portable API | `/predict`, `/health`, `/ready`, `/model-info` | Private GPU HTTP smoke and contract tests passed; host dependency isolation remains |
| 3. Docker and CI | Reproducible container and automated tests/build | Clean-environment run; no weights, secrets, or datasets in image; local-only port binding initially |
| 4. Benchmark | Warm-up, p50/p95, throughput, memory, precision comparison | Fixed hardware/input/bank settings; accuracy regression; API vs model timing separated |
| 5. AWS | Controlled endpoint and runtime logs | Account/region/budget approval; access control/TLS; deployment and shutdown evidence |
| 6. MLOps | Tracking, model registry, release manifest, rollback | Version-linked evaluation; explicit promotion; rollback test; no automatic target-based retraining |
| 7. Inspection assistant | LangChain tools, document retrieval, sourced report | Tool/grounding tests; abstention on missing evidence; separate LLM cost and latency |

Stage 2 has a working private research-backed API, not a standalone distribution.
Stages 3–7 remain planned. Model artifacts and local output images
are excluded from Git. Research figures remain illustrative, not deployment
screenshots. Other C-TED hosts and T-TED can join the same service interface after
the first complete path is verified; their existing core availability does not
mean their service adapters are complete.

The intended API separates image-level thresholds from pixel-mask thresholds.
Only versioned, validated thresholds should produce decisions. A LangChain
assistant should call the inspection API and retrieve approved manuals, not
replace the detector or invent defect causes from an anomaly map. It must not
autonomously retrain models, change thresholds, or deploy artifacts. References:
[LangChain learning resources](https://docs.langchain.com/oss/python/learn) and
[MLflow registry workflows](https://mlflow.org/docs/latest/ml/model-registry/workflow/).

## Citation

```bibtex
@article{kim2026ted,
  title={TED: Text-Axis Evidence Decomposition for Prompted Anomaly Localization},
  author={Kim, JinYoung and Kim, Geonho and Park, GiJeong and Lee, Geonu and Yoo, YoungJoon},
  journal={arXiv preprint arXiv:2609.39033},
  year={2026}
}
```

## Acknowledgments

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) [RS-2021-II211341, Artificial Intelligence Graduate School Program (Chung-Ang University), and RS-2022-II220124, Development of Artificial Intelligence Technology for Self-Improving Competency-Aware Learning Capabilities]. This research was also supported by the AI Seoul Tech Research Support Program of the Seoul Future Foundation.

We acknowledge the authors and maintainers of CLIP, ImageBind, the evaluated CLIP-AD hosts, and the datasets used in this study. Useful upstream projects include [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP), [FAPrompt](https://github.com/mala-lab/FAPrompt), [AdaCLIP](https://github.com/caoyunkang/AdaCLIP), and [AA-CLIP](https://github.com/Mwxinnn/AA-CLIP). Third-party assets retain their original terms.

## Contact

For research or release questions, please [open an issue](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/issues) or contact JinYoung Kim at **barraki7226@cau.ac.kr**.
