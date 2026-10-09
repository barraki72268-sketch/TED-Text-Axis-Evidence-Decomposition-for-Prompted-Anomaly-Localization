# FAPrompt inference and Docker service

[Research overview](../README.md) · [Reproduction scope](REPRODUCIBILITY.md)

## Verified path

The released service runs **FAPrompt ViT-L/14@336px + C-TED**, input size 518,
with MVTec-source calibration. The paper-aligned artifact uses **alpha 0.5**.
It is available from [KIMJINYOUNG/TED-reproducibility](https://huggingface.co/KIMJINYOUNG/TED-reproducibility).
Other models are being reproduced; their service adapters and model selector
are not yet released as verified functionality.

The service returns raw Host/C-TED maps and the unchanged official host image
score. That score is not a calibrated defect probability. Display-normalized
heatmaps are not binary defect decisions, and no operational threshold has
been fitted or invented.

## Evidence and limits

| Check | Result | Record |
|---|---|---|
| Paper-aligned API parity | Three BTAD images, CPU batch size 1; Host and C-TED raw-map maximum absolute errors both 0 | [API parity](../deployment/validation/paper-api-parity-20261008.json) |
| pilab release request | Container HTTP response returned the expected FAPrompt model and artifact identity | [Deployment response](../deployment/validation/paper-deployment-20261008.json) |
| Standalone CPU image, earlier engineering artifact | Offline model loading, 20 tests passed / 1 GPU test skipped, three repeated MPDD-image requests matched reference maps | [Standalone validation](../deployment/validation/standalone-cpu-20261008.json) |

The paper-aligned artifact SHA-256 is
`519e0354a87b2508034d4a85ee9a00c698b0714c52e0e467c57dae9614d62379`.
The earlier standalone test used the alpha 1 engineering artifact
`968db05a3cac035ad0e647a23d8468460a915ad87712da0d4bc8028ae588415d`.
These are separate validation records; the earlier image test does not certify
a standalone container for every paper model.

The pilab paper release mounts an existing Python runtime read-only. It is
therefore a working local container deployment with a host dependency.
The [recorded Compose file](../deployment/pilab-release.compose.yaml) applies to
the prepared release directory containing `app/`, `models/`, and its existing
container image. Running that file from this repository alone will not create
those prerequisites. The recorded default port is 18082; the deployed release
was configured to use 18080.

The API check uses batch size 1; the historical full-BTAD evaluation used batch size 4.
The original evaluator has batch-dependent smoothing behavior, so API parity
does not establish paper-batch benchmark equality. Single-image timings and
the recorded short container benchmark are not a general throughput SLA.

## Access the existing pilab deployment

From your local computer, keep this SSH tunnel open:

```bash
ssh -N -L 18080:127.0.0.1:18080 pilab
```

Then open `http://127.0.0.1:18080/` for the upload demo or `/docs` for API docs.
The SSH alias and running remote service must already be configured. Check
readiness before sending a prediction:

```bash
curl --fail http://127.0.0.1:18080/ready
curl --fail http://127.0.0.1:18080/model-info
curl --fail -X POST http://127.0.0.1:18080/predict \
  -H 'Content-Type: image/png' --data-binary @inspection.png
```

A connection refusal can mean the SSH tunnel or remote service is stopped;
the dated records above are not a live health guarantee.

## Repository inference modules

- [`ted/inference/faprompt.py`](../ted/inference/faprompt.py): offline source-only artifact export.
- [`ted/inference/engine.py`](../ted/inference/engine.py): persistent host/artifact loading and single-image inference.
- [`ted/inference/cli.py`](../ted/inference/cli.py): export and inference CLI.
- [`ted/inference/api.py`](../ted/inference/api.py): health/readiness, model information, prediction, upload limits, and bounded concurrency.

Feature-level C-TED usage and host-specific contracts are documented in
[CTED.md](CTED.md). Original model/evaluator licenses and source-dataset terms
continue to apply; see [source notices](../reproduction/SOURCE-NOTICES.md).

## Remaining service work

All-model adapters and selection, standalone installation of the paper-aligned
release, versioned promotion/rollback verification, and broader performance
validation remain required. AWS deployment has not been completed.

## AA-CLIP captured-state worker

An experimental AA-CLIP worker now reads fitted state from
`reproduction export-captured-run`. The validated setting is the
`aaclip_vitl_openai224_mvtec2btad_sourcelimit1_seed0` ablation, using a 224-pixel
input with the recorded OpenAI L/14-336 weights. This is not the default AA
paper setting or a replacement for completing the full model matrix.

The [CPU image check](../reproduction/validation/a10-20261009/aa-engine-parity.json)
compares two forwards through the same initialized trusted host, composing the
original evaluator functions independently of the bridge. The
[real HTTP check](../reproduction/validation/a10-20261009/aa-http-parity.json)
checks three images, one per BTAD category, with zero raw-map differences.
Neither check establishes full-dataset CPU/GPU metric equivalence.

Use a Linux prepared workspace whose full evaluation matched its archived
reference, and export it with the current CLI. The export must include
`prepared_source_files`; earlier exports without those bindings need a new
export directory. Keep each upstream host in its own worker process, because
the archived projects use overlapping module names.

With the evaluation dependencies and
[`deployment/requirements-api.txt`](../deployment/requirements-api.txt)
installed in a separate service environment:

```bash
python -m reproduction export-captured-run ./runs/AA_RECIPE ./exports/AA_RECIPE
export TED_ENGINE_FAMILY=aaclip
export TED_CAPTURED_EXPORT="$PWD/exports/AA_RECIPE"
export TED_RUN_WORKSPACE="$PWD/runs/AA_RECIPE"
export TED_DEVICE=cpu
python -m uvicorn ted.inference.api:create_app --factory --workers 1 \
  --host 127.0.0.1 --port 18083
```

```bash
curl --fail http://127.0.0.1:18083/model-info
curl --fail -X POST 'http://127.0.0.1:18083/predict?category=01' \
  -H 'Content-Type: image/png' --data-binary @inspection.png
```

The category must come from `/model-info` (`01`, `02`, `03` in this check).
Missing/unknown categories return 422. `image_score` is the raw AA detection
token score. The paper's image metrics additionally use dataset-wide
normalization and map fusion, so this score is not a per-request reproduction
of that metric or an anomaly probability. Prediction performs no source-bank
mining or calibration fitting.

The initial HTTP check used a temporary CPU worker on the A10 node and was
stopped afterwards. The later pilab Docker checks are recorded below; all-model
selection remains required. The initial service dependency overlay is recorded
[here](../reproduction/validation/a10-20261009/aa-api-extra.freeze.txt).

### Preparing an AA serving bundle

The new builder copies the exact prepared source, original run plan, selected
backbone, adapter checkpoints, and captured calibration state from a passing
AA execution. It rechecks the original metrics, coverage, and byte hashes.
It includes no dataset images and does not fit new state.

```bash
python -m reproduction build-aa-serving-bundle ./runs/AA_RECIPE ./bundles/AA_RECIPE
```

The engine accepts the bundle as `TED_RUN_WORKSPACE` and its `export/` directory
as `TED_CAPTURED_EXPORT`. The source and original execution evidence remain
unchanged; the engine binds the selected verified weight to its new local path.
This packaging step alone does not certify portable inference or Docker deployment.

After relocation and serving checks, a byte-verified bundle can be packaged:

```bash
python -m reproduction.serving_archive ./bundles/AA_RECIPE ./AA_RECIPE.tar.gz
```

Use a clean staging copy with generated `__pycache__` directories omitted;
preserve the original bundle. The archive command rejects unlisted files,
links, changed weights/source/state, and existing output paths. It verifies
every decompressed archive member against the bound inventory before returning
the archive hash. This contains the original weights, exact prepared source,
run plan, captured bank/calibrator state, and execution evidence; dataset
images remain outside the archive. Upstream licenses and dataset-derived
artifact terms continue to apply.

Consumers can verify the published archive record and extract only regular,
relative-path members into a new directory, then recheck the full serving
inventory without importing PyTorch:

```bash
python -m reproduction.serving_archive --unpack --record AA_RECIPE.json \
  ./AA_RECIPE.tar.gz ./downloaded-AA_RECIPE
```

Use the archive JSON supplied by the release. A passing extraction report
proves input integrity and execution binding; run inference checks separately.

### Download a pinned public AA release

From a fresh clone, download and verify a complete serving bundle without a
Hugging Face login or PyTorch installation:

```bash
python -m reproduction.aa_release main-l336-seed0 ./aa-main-bundle
# Separate weak-source release:
python -m reproduction.aa_release source1-seed0 ./aa-source1-bundle
```

The checked-in release catalog pins the public commit, archive size/SHA-256,
933-file inventory, recipe and captured-artifact identity. Downloads are cached
by hash and rechecked before extraction. Interrupted `.partial` files and
existing destinations are preserved; use a new cache or destination after
inspecting a failed attempt. The command does not fit or run a model.
Follow the CPU installation and API instructions below for inference.

### Fresh AA CPU environment

A new Linux Python 3.10.22 environment with official CPU wheels passed
dependency checking and offline three-image raw-map parity for both AA main
and source-limit-1 bundles. It has no CUDA build. The
[environment record](../reproduction/validation/a10-20261009/aa-public-cpu-environment.json)
and [main](../reproduction/validation/a10-20261009/aa-public-cpu-main-parity.json)/
[weak-source](../reproduction/validation/a10-20261009/aa-public-cpu-weak-parity.json)
checks retain the exact scope. This does not establish full-dataset CPU metrics
or a clean environment for every host family.

Inside a new Python 3.10 virtual environment:

```bash
python -m pip install --upgrade pip==25.2 setuptools==79.0.1
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r deployment/requirements-aa-cpu.txt
python -m pip check
```

The recorded initial install failed with the older bundled pip's package-name
metadata handling; the isolated pip upgrade resolved it. The original failed
install log was retained. All installed AA service packages are pinned in the
CPU requirements file. Public archive acquisition and inference checks are
separate from this already passing relocated-bundle check.

Before promotion, move the bundle to another directory and compare its maps
against saved original-evaluator maps using
[`examples.aa_bundle_parity`](../examples/aa_bundle_parity.py). Supply a fixtures
JSON as documented in that module and forbid reads from the original workspace:

```bash
CUDA_VISIBLE_DEVICES='' python -m examples.aa_bundle_parity \
  ./relocated/AA_RECIPE ./fixtures.json ./relocation-report.json \
  --forbid-read-root /absolute/path/to/original/workspace
```

The checker prohibits network connections and calibration fitting during
engine loading and prediction. A matching report proves only the named image
maps. A real container run and all-model service selection remain separate gates.

The [October 9 relocation check](../reproduction/validation/a10-20261009/aa-relocation-parity.json)
passed on CPU for three BTAD images using the source-limit-1 bundle: both raw
maps had zero maximum absolute error for every image. Original-workspace reads,
network connections, and calibration fitting were prohibited during loading and
inference. This does not establish parity for the newly replayed main configuration.

### pilab AA Docker checks (October 9)

An isolated AA source-limit-1 worker was started on pilab loopback port 18083
using the [host-runtime Compose profile](../deployment/aa-captured.pilab.compose.yaml).
The dated [container record](../reproduction/validation/a10-20261009/aa-pilab-container-runtime.json)
identifies the image, read-only mounts, resource limits, and observed healthy state.
Readiness at that time does not guarantee future availability.

Two comparisons are retained:

- [A10 CPU versus pilab Docker](../reproduction/validation/a10-20261009/aa-pilab-a10-difference.json): strict zero-error comparison failed, with maximum host-map difference approximately 0.000103. The cause has not been isolated.
- [Same-pilab original equations versus Docker HTTP](../reproduction/validation/a10-20261009/aa-pilab-container-local-parity.json): all three images have zero raw-map and raw image-score error. The [independent original-equation check](../reproduction/validation/a10-20261009/aa-pilab-original-math.json) is also retained. Missing and unknown categories return 422.

The main L/14-336 seed-0 configuration separately passed
[CPU original-equation parity](../reproduction/validation/a10-20261009/aa-main-engine-parity.json)
and [offline relocation parity](../reproduction/validation/a10-20261009/aa-main-relocation-parity.json)
on three images at its recorded input size of 518. Its separate pilab worker on
loopback port 18084 then passed [Docker HTTP parity](../reproduction/validation/a10-20261009/aa-pilab-main-container-local-parity.json)
against the [same-pilab original equations](../reproduction/validation/a10-20261009/aa-pilab-main-original-math.json):
both raw maps and the raw detection-token score match exactly on all three images.
The [main worker runtime record](../reproduction/validation/a10-20261009/aa-pilab-main-container-runtime.json)
identifies its image and mounts. Neither worker check establishes full-dataset
CPU metric parity. Both workers use the same host-runtime profile with separate
Compose project names; the main project sets `TED_AA_PORT=18084`.
All-model publication and a standalone container remain required.

### Selecting the three verified pilab workers

The [gateway Compose profile](../deployment/gateway.pilab.compose.yaml) serves
the model selector on pilab loopback port 18085. Its
[registry](../deployment/pilab-model-registry.json) pins the artifact identity of
FAPrompt paper alpha 0.5, AA main L/14-336 seed 0, and AA source-limit-1.
These releases have different evaluation scopes, shown beside the selector.
Other model families will be added after their serving checks pass.

From your own computer, keep this SSH tunnel open:

```bash
ssh -N -L 18085:127.0.0.1:18085 pilab
```

Open `http://127.0.0.1:18085/`, select a model and (for AA) a target category,
then upload an actual PNG or JPEG. The interface displays the input, host map,
C-TED map, raw image score, artifact hash, and downloadable raw maps.
The image score is not an anomaly probability or the dataset-wide paper metric.

The [nine-case gateway check](../reproduction/validation/a10-20261009/model-gateway-parity.json)
compares direct-worker and gateway HTTP predictions on three BTAD images for
each of the three registered releases. Both maps and raw image scores have
zero error in all nine cases. Missing AA categories return 422; unknown models
return 404. Contract tests also check artifact mismatches, upload limits,
unavailable workers, and independent per-worker admission.
The actual browser upload and AA main result display were checked separately.
This verifies service routing; it does not extend any full-dataset reproduction claim.

The gateway uses Linux host networking to reach loopback-only workers, binds
only to `127.0.0.1:18085`, and uses the existing host-mounted Python runtime.
The recorded API parity check ran on commit `c674fa4`; the later `0b5a3b2`
deployment only fixes hiding inactive category controls. No standalone image
or public Internet endpoint is claimed.

### Public archive client verification

The [source-limit-1 anonymous client record](../reproduction/validation/a10-20261009/aa-anonymous-weak-verification.json)
confirms a download without authentication from the pinned Hugging Face commit,
archive SHA-256/size verification, extraction into a new directory, all 933 input
files verified, and CPU raw-map agreement on three BTAD images in the isolated
CPU-only environment. The check prohibits network access, fitting and reads
from both original/previously relocated bundles during inference. The [main archive client record](../reproduction/validation/a10-20261009/aa-anonymous-main-verification.json)
passes the same checks for the input-518 main release. These checks do not establish full
CPU dataset metric parity or reproduction of all paper configurations.

### AdaptCLIP captured-state worker

The OpenAI L/14 and L/14-336 BTAD seed-0 recipes each match all eight archived
per-seed metrics at two decimals. Their prepared-workspace engines also match
the original CPU evaluator's host/C-TED maps and both raw image scores on the
first recorded test image of each BTAD class. See the
[export and image evidence index](../reproduction/adaptclip-serving-exports.json).
This does not establish all seeds, five datasets, or the paper's printed means
and standard deviations.

From a passing prepared workspace, create a new bundle:

```bash
python -m reproduction.adaptclip_serving_bundle RUN_WORKSPACE NEW_BUNDLE
python -m reproduction.serving_archive NEW_BUNDLE NEW_ARCHIVE.tar.gz --host AdaptCLIP
```

The bundle preserves the 920 research source files, checkpoint, backbone,
terminal evaluation evidence, and captured VL/TL calibrators. It includes no
dataset images. The engine changes only source ROOT, checkpoint and backbone
paths in memory. It retains the named L/14-336 loader's architecture choice,
image preprocessing, numerical arguments and scoring equations.

To check a fresh extraction against the original output block:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m reproduction.adaptclip_image_parity NEW_BUNDLE/export NEW_BUNDLE \
  --fixture-workspace RUN_WORKSPACE --deny-original-inputs \
  --output NEW_PARITY_REPORT.json
```

The check denies network access and original workspace/model input reads during
inference. Original fixture metadata is an explicit exception; test images stay
outside the bundle and are checked against the full dataset input manifest.
Run each model in its own process because upstream imports use shared names.

For an experimental loopback HTTP worker with the pinned research dependencies:

```bash
TED_ENGINE_FAMILY=adaptclip TED_DEVICE=cpu \
TED_CAPTURED_EXPORT=NEW_BUNDLE/export TED_RUN_WORKSPACE=NEW_BUNDLE \
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m uvicorn ted.inference.api:app --host 127.0.0.1 --port 18086 --workers 1
```

The response preserves `image_score` and separately returns `cted_image_score`.
Both are raw evaluator scores, not calibrated probabilities. Anonymous public
archive acquisition, pilab Docker HTTP parity and registry integration are
separate gates; this worker is not yet in the deployed selector.
