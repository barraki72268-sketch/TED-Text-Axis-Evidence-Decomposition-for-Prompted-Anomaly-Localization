---

language:

- en

- ko

tags:

- anomaly-detection

- anomaly-localization

- reproducibility

- faprompt

- ted

- pytorch

---

# TED reproducibility: research artifacts and verified inference

Research artifact release for **TED: Text-Axis Evidence Decomposition for Prompted Anomaly Localization**, [arXiv:2609.39033](https://arxiv.org/abs/2609.39033).

Seven verified pilab workers are selectable: FAPrompt L/14-336, captured FAPrompt B/16+ seed1 and H/14 seed0, AA-CLIP main/source-limit-1, and AdaptCLIP OpenAI/L/14-336. Each has a separate validation scope. The gateway preserves model identity and recorded strength across all33 fixed-image routing cases; the H/14 alpha1.5 browser upload also completes. All-model, all-dataset, all-seed reproduction and deployment remain incomplete; no universally best model is declared.

## Captured FAPrompt releases: verified public acquisition

The H/14 seed0 and B/16+ seed1 BTAD executions each match eight archived
per-seed metrics at two decimals. Their archives retain actual source banks,
fitted branch calibrators, original prepared source, checkpoint and backbone.
No calibrator is refitted for serving and no dataset images are included.

Both archives are public at immutable revision
`f49bcc6c857606fd7edb1148856157532f6b2271`. Each was downloaded anonymously
with standard-library HTTPS, extracted into a new directory, and verified
against size/SHA plus all932 input files and terminal state bindings.
CPU image inference matches the original evaluator output block on three
canonical BTAD images at all three recorded strengths (0.5,1.0,1.5): nine
cases per variant with zero map/raw-score error. Original workspace/model
reads and network access are denied during these inference checks.

Use the GitHub package's pinned client with an empty cache and new destination:

```bash
python -m reproduction.faprompt_release bplus-seed1 NEW_BPLUS_BUNDLE --cache NEW_CACHE
python -m reproduction.faprompt_release h14-seed0 NEW_H14_BUNDLE --cache NEW_CACHE
```

See [FAPROMPT-CAPTURED-SERVING.md](FAPROMPT-CAPTURED-SERVING.md) and
[the immutable GitHub checks](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/tree/e19809f/reproduction/validation/a10-20261009).
This validates these specific packages and image cases; all-dataset/seed
means, other models and standalone dependency images remain incomplete.

## Additional historical model checkpoints (2026-10-08)

The eleven exact host checkpoints for 207 traced AA-CLIP, FAPrompt, AdaptCLIP,

AdaCLIP, and BayesPFL recipes are now preserved in `host-checkpoints-20261008.tar`.

See [HOST-CHECKPOINTS.md](HOST-CHECKPOINTS.md) for extraction, byte verification,

provenance, and component licenses. `host-checkpoints-archive.json` pins the

archive, and `host-checkpoints.json` maps each recipe to its objects.

`source.zip` and `source-manifest.json` preserve 920 research source files.

This addition is a historical artifact release, **not completion of all-dataset

reproduction or all-model API deployment**. Banks, backbone assets, calibrators,

full clean runs, and portable execution remain in progress. The FAPrompt inference

release below is retained with its original validation scope.

## Additional adapted-host source banks

`host-source-banks-20261008.tar` preserves 92 hash-verified bank objects mapped
to the 207 adapted-host recipes. See [SOURCE-BANKS.md](SOURCE-BANKS.md) for byte
verification and the distinction between explicit historical paths and replayed
cache-selection evidence. This preserves source features; it does not claim
fresh bank rebuilding or complete all-dataset reproduction. Raw-backbone and ablation banks are released in the separate archives described below; fresh rebuilding and complete reruns remain pending.

## Legacy FAPrompt L/14-336 files and provenance

- `host.pth`: preserved FAPrompt checkpoint `trained_on_mvtecad/epoch_15.pth`; not a newly trained checkpoint.

- `source_bank_mvtec_seed0.pt`: preserved source bank with 4,096 false-positive and 4,096 defect features, scores and interface metadata. These are features, not dataset images.

- `calibrator.pt`: two branch calibrators captured directly during the historical GPU recipe, projected source banks and inference settings. No target labels enter calibration.

- `training_recipe.json`: historical arguments and deployment settings.

- `capture_historical_calibrators.py`: the capture script as executed in the research workspace; contains workspace paths and is an audit record, not a portable full training entry point.

- `runtime-source.zip`: FastAPI inference engine, vendored evaluator, tests, Dockerfile and Compose recipe.

- `backbone.json`: upstream OpenAI CLIP download URL and exact digest. The 934 MB backbone is fetched separately.

- `MANIFEST.json`: SHA-256 and byte lengths for release files.

The calibration source is **MVTec AD**, and the evaluation target is **BTAD**. Source anomaly labels are used; this is not training from normal images only. Neither host pretraining nor fresh mining of the archived source bank has been reproduced in this release.

## Validation scope

The preserved FAPrompt paper recipe was rerun on all 741 BTAD test images with historical batch size 4. Base and C-TED I-AUROC, P-AUROC, PRO and pixel AP matched their archived results at two decimal places. C-TED alpha 0.5 values (%): **89.24 / 94.99 / 70.95 / 45.01**.

The broader provenance audit linked 300/300 main-table means to archived summaries, and 627/640 host-table means. Thirteen host-table means still require provenance resolution. This is not a claim that every paper experiment has been rerun.

The HTTP API processes one image at a time. Historical batch behavior and host postprocessing are retained in the evaluator; paper batch-4 benchmark scores must not be presented as measurements of the batch-1 CPU API. See included release validation reports for actual API checks.

## Run the legacy FAPrompt L/14-336 release with Docker

Download this repository, then use Python 3.10+:

```bash

python prepare_models.py

TED_MODEL_DIR="$PWD" docker compose -f runtime/compose.yaml up -d --build

curl http://127.0.0.1:18080/ready

curl -X POST http://127.0.0.1:18080/predict \

  -H 'Content-Type: image/png' --data-binary @your-image.png

```

Open `http://127.0.0.1:18080/` for the upload demo or `/docs` for the API. CPU inference is intended for a small research demo; no throughput SLA is claimed. The API returns raw host/C-TED maps and an unchanged host image score, which is not a defect probability. No operational pass/fail threshold has been calibrated.

The generic standalone CPU image was previously tested with offline model loading. The pilab deployment uses the host-runtime Compose variant because its shared root Docker disk has insufficient space for the standalone image. The production machine's Python environment is mounted read-only; this deployment is not yet fully portable. GPU Compose is a recipe only and must obey the host's scheduler policy.

## Attribution and usage

FAPrompt: Jiawen Zhu, Yew-Soon Ong, Chunhua Shen and Guansong Pang, *Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection*, ICCV 2025. [Upstream code](https://github.com/mala-lab/FAPrompt), MIT notice in `LICENSE-FAPrompt.txt` and inside the runtime archive.

MVTec AD: Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger, *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019. [Dataset and license](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). Source-derived artifacts retain the dataset's **CC BY-NC-SA 4.0** restrictions; see `LICENSE-SOURCE-DATA.txt`. Use for non-commercial research and portfolio demonstration. This repository does not grant a blanket commercial license to upstream artifacts.

OpenAI CLIP: [upstream repository](https://github.com/openai/CLIP). Third-party notices remain applicable. Original datasets, credentials, and private conversation logs are not included. The AA serving bundles include their exact pinned CLIP backbone weights; upstream notices remain applicable.

## Frozen-backbone bank archive

`raw-source-banks-20261008.tar` preserves 10 additional source banks for the 25 traced RawCLIP/ImageBind main-table configurations. See [RAW-SOURCE-BANKS.md](RAW-SOURCE-BANKS.md) for verification and limitations. Full GPU reproduction, MVTec AD 2 access, fresh bank rebuilding, and all-model deployment remain unfinished.

## Weak-source ablation bank archive

`weak-source-banks-20261009.tar` preserves the 12 source banks for all 28 archived weak-source configurations. See [WEAK-SOURCE-BANKS.md](WEAK-SOURCE-BANKS.md) for evidence levels and verification. Fresh GPU ablation replay and full-paper deployment remain unfinished.

## AA-CLIP serving bundles (October 9)

See [AA-SERVING.md](AA-SERVING.md) and [AA-SERVING-ARCHIVES.json](AA-SERVING-ARCHIVES.json) for two complete serving-input archives, including original weights, prepared research source, captured calibration/bank state, run settings and execution evidence. Neither contains dataset images. Both recipes matched 8/8 archived full-BTAD seed-0 metrics on A10; main and weak-source are distinct configurations. Three-image same-pilab original-equation/Docker parity passed for both. The small cross-A10 CPU difference for the weak-source worker remains preserved.

The pilab model-selection gateway routes these two AA workers and FAPrompt. All nine direct-worker/gateway image predictions matched exactly; this is routing verification, not nine additional benchmarks. Other model families and all-model publication remain in progress.

Both AA serving archives have now passed anonymous download, new-directory full-input verification and isolated CPU three-image raw-map parity. See [AA-SERVING.md](AA-SERVING.md) for pinned acquisition commands and precise evidence scope.
