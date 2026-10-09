# FAPrompt captured-state serving bundles

These packages preserve the actual source bank and fitted branch calibrators
from matching full-BTAD executions: ViT-H/14 seed 0 and ViT-B/16+ seed 1.
Each execution matches all eight archived per-seed metrics at two decimals.
This is not a claim that every paper mean, standard deviation, dataset,
backbone, seed or ablation has been reproduced.

Each archive contains 932 regular files: the prepared research source,
checkpoint, backbone, captured state, provenance and input manifests.
No dataset images are included. No calibrator is refitted for serving.
The JSON record pins archive byte size, SHA-256 and captured-state identity.
Source, pretrained weights and source-dataset terms remain applicable; this
publication does not grant new rights. Consult SOURCE-NOTICES.md and the
original notices inside the source tree.

Both archives passed fresh extraction and all-file hash verification.
CPU inference matches the original evaluator output block on the first
canonical BTAD image from each of three classes at all recorded strengths
(0.5, 1.0 and 1.5): nine cases per model, zero map and raw-score error.
The original initialization seed is retained. Original workspace/model
input reads and network access were denied during these image checks.
This is a fixed-image check, not a full-dataset CPU metric replay.
Raw image scores are evaluator outputs, not anomaly probabilities.

Implementation, evidence and usage:
https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/tree/e19809f

Explicitly choose a recorded strength when starting a CPU worker:

```bash
TED_ENGINE_FAMILY=faprompt_captured TED_DEVICE=cpu TED_ALPHA=0.5 \
TED_RUN_WORKSPACE=NEW_BUNDLE CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m uvicorn ted.inference.api:app --host 127.0.0.1 --port 18088
```

The API accepts `POST /predict?alpha=0.5`, `1.0` or `1.5` with image bytes.
It rejects unrecorded strengths and echoes the selected strength and artifact
identity. Docker HTTP, gateway integration, anonymous public acquisition and
standalone dependency-image validation have separate evidence gates.
Both pilab CPU Docker workers also pass all nine same-machine HTTP comparisons against the original evaluator output block. The seven-model gateway passes all33 recorded model/image/strength comparisons with zero map and raw-score error. The H/14 alpha1.5 browser upload also completes and retains the selected identity/strength. These profiles mount an existing host runtime. Both public archives have now been acquired anonymously into new directories: archive size/SHA, all932inputs, and all9image/strength comparisons are verified per variant. Original workspace/model reads and unrelated network are denied during CPU image inference; no fitting is performed. See the accompanying faprompt-*-anonymous-acquisition.json and faprompt-*-anonymous-image-parity.json. Standalone dependency-image validation remains pending.

The complete paper reproduction remains in progress.
