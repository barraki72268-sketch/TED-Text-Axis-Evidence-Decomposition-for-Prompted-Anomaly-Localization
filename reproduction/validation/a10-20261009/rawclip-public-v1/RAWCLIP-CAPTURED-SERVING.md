# Captured RawCLIP serving release

This release contains the captured RawCLIP ViT-L/14 OpenAI MVTec-to-BTAD configuration from the TED evaluation, with its original source bank and calibrated state. It provides Base, T-TED and C-TED maps and fused image scores without refitting.

- Archive: `rawclip-openai-btad-20261009-v1.tar.gz`
- SHA-256: `aca0e26915b42f12806266fc15e8f80352f5272c6a01e6d6730b4f6c706a95e8`
- Size: 898354287 bytes; 930 verified regular files.
- Export identity: `d1b619823f1ca7e55cf10e7f3536f4c2108a73b74e67369bf079912d69efc452`
- No dataset images are included. Supply a separately acquired BTAD test image and its recorded category (`01`, `02`, or `03`).

## Verified scope

The full BTAD evaluator matched all 12 archived per-seed metrics at two decimal places. This is one configuration; it does not establish every paper mean, standard deviation, dataset, seed or ablation.

CPU relocation checks compared three protocol-fixed images, one per BTAD category, with the original evaluation output block. Base, T-TED and C-TED maps and all three image scores were identical. The pilab CPU Docker worker passed the same three original-code HTTP comparisons. These CPU image checks are separate from full-dataset GPU metric replay.

## Code and Docker

Use the release branch of [TED on GitHub](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/tree/codex/all-model-reproduction). The serving implementation is `ted/inference/rawclip_engine.py`, the archive verifier is `reproduction/serving_archive.py`, and the pilab CPU profile is `deployment/rawclip-captured.pilab.compose.yaml`. Detailed evidence is under `reproduction/validation/a10-20261009/rawclip-pilab-v1` and `rawclip-relocated-v1`.

The pilab profile mounts a pinned host Python runtime. It is a verified local deployment profile; a standalone dependency image is a separate deliverable. GPU use requires a valid Slurm allocation. Image scores are fused raw scores, not probabilities.

Original source and upstream backbone licensing remain applicable. The archive preserves the supplied source files and license notices. It does not grant rights to redistribute evaluation datasets.
