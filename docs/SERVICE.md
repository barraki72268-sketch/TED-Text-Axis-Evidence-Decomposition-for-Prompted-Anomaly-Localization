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

The dated HTTP check used a temporary CPU worker on the A10 node and was
stopped afterwards. AA deployment on pilab Docker and model selection remain
required; the existing pilab release is still the FAPrompt service described
above. The checked service dependency overlay is recorded
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
