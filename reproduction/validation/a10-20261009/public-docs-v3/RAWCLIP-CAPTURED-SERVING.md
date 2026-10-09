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


## Additional verified RawCLIP backbones

The H/14 and L/14-336 MVTec-to-BTAD configurations also match all12 archived Base/T-TED/C-TED per-seed metrics at two decimal places on the full BTAD evaluation. Each portable package passes three protocol-fixed CPU image comparisons with all three raw maps and fused scores exactly matching the original output block. These checks use the terminal captured bank/calibrators without refitting. Each archive contains930 regular files and no dataset images.

| Backbone | Archive | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| ViT-H/14 | `rawclip-h14-btad-20261009-v1.tar.gz` | 3708782197 | `7f4de490b783d483ecd327c686653dac5e39870df0abb3757e04e4284799bdb5` |
| ViT-L/14-336 | `rawclip-l336-btad-20261009-v1.tar.gz` | 905080940 | `9f9f74eee72ea4a0357ca2822a0eda45ad76db78528e5ce6f77af8bb0909068e` |

The JSON archive records bind each variant to its own recipe and captured export identity. Use the corresponding verified record when extracting; do not interchange variants. Public acquisition and Docker checks are documented separately for each variant as they are completed.

## Fresh public acquisition checks

H/14 and L/14-336 each passed anonymous HTTPS download into a new cache, new-directory extraction with all930 input hashes verified, and three original-code CPU image comparisons with all Base/T-TED/C-TED maps and raw scores exactly equal. [Immutable evidence](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/tree/0b70ba5/reproduction/validation/a10-20261009/rawclip-public-client-v2).

```bash
python -m reproduction.rawclip_release h14 ./rawclip-h14 --cache ./rawclip-cache
python -m reproduction.rawclip_release l336 ./rawclip-l336 --cache ./rawclip-cache
```

Each profile also passes three original-code Docker HTTP cases; [container evidence](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/tree/eb9227a/reproduction/validation/a10-20261009/rawclip-pilab-v2).
