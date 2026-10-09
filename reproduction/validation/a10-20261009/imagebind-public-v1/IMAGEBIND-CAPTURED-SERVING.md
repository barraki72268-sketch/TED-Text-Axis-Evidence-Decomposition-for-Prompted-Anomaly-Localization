# Captured ImageBind serving package

The MVTec-source to BTAD execution matches all 12 archived Base/T-TED/C-TED metrics at two decimals. This package contains the original ImageBind backbone, exact prepared research source and captured source bank/calibrators from that passing execution. It contains no dataset images and does not refit or mine banks for serving.

Download `imagebind-btad-20261009-v1.tar.gz` and `imagebind-archive-20261009-v1.json`. From the GitHub reproduction branch, unpack to a new directory:

```bash
python -m reproduction.serving_archive imagebind-btad-20261009-v1.tar.gz NEW_IMAGEBIND_BUNDLE --unpack --record imagebind-archive-20261009-v1.json
```

The unpacker verifies the full archive and all 930 extracted inputs. It also restores the empty upstream weights directory required for read-only imports. Fresh extraction and three original-code CPU image comparisons have passed. On pilab, three original-code Docker HTTP comparisons also have zero error for all Base/T-TED/C-TED maps and fused raw image scores. The production worker has no dataset mount and uses CPU.

The Compose profile is `deployment/imagebind-captured.pilab.compose.yaml`. It mounts the package and pinned host runtime read-only. Existing identical backbone bytes may be supplied with `TED_IMAGEBIND_WEIGHT` as a read-only file mount. This profile uses the host runtime; it is not a standalone dependency image.

Supply BTAD category `01`, `02`, or `03` explicitly. Raw scores are not probabilities. Image inference checks are separate from whole-dataset metric verification and all-paper reproduction.

[Code, profile and evidence](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/tree/68fb576/reproduction/validation/a10-20261009/imagebind-pilab-v1)
