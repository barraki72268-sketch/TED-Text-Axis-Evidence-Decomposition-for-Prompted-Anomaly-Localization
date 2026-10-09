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
