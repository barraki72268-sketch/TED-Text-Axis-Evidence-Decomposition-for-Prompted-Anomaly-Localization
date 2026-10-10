# Constructing the Defect Bank and Hard-FP Bank

The original bank-construction code is publicly available for **AA-CLIP, FAPrompt, AdaCLIP, AdaptCLIP, BayesPFL, RawCLIP and ImageBind**. The links below expose the original collector implementations as ordinary GitHub files, without needing to search an archive.

These are byte-identical copies of the captured research source. They depend on the full host code, datasets and checkpoints; the copied files are not standalone programs. The complete source and its dependencies can be extracted with the commands below. A uniform, independently verified fresh-bank command for every paper configuration is still being prepared. Downloading saved banks is a separate operation.

## Find the construction code

| Host | Original collector | Bank-construction function |
|---|---|---|
| AA-CLIP | [official_parallel_test_aaclip.py](../research/bank_construction/official_parallel_test_aaclip.py) | `collect_source_banks` |
| FAPrompt | [official_parallel_test_faprompt.py](../research/bank_construction/official_parallel_test_faprompt.py) | `collect_source_banks` |
| AdaCLIP | [official_parallel_test_adaclip.py](../research/bank_construction/official_parallel_test_adaclip.py) | `collect_source_patch_banks` |
| AdaptCLIP | [official_parallel_test_adaptclip.py](../research/bank_construction/official_parallel_test_adaptclip.py) | `collect_source_patch_banks` |
| BayesPFL | [collect_bayespfl_source_banks.py](../research/bank_construction/collect_bayespfl_source_banks.py) | `collect_source_patch_banks` |
| RawCLIP | [official_parallel_test_rawclip.py](../research/bank_construction/official_parallel_test_rawclip.py) | `collect_source_patch_banks` |
| ImageBind | [official_parallel_test_rawimagebind.py](../research/bank_construction/official_parallel_test_rawimagebind.py) | `collect_source_patch_banks` |

The [manifest](../research/bank_construction/manifest.json) records each original archive path, file size and SHA-256. Original component licenses are listed in [source notices](../reproduction/SOURCE-NOTICES.md). AdaptCLIP's VL-refine evaluation imports the collector from `official_parallel_test_adaptclip.py`; its readout is not interchangeable with the plain evaluator.

## Obtain the complete source

From a new checkout of the active release:

```bash
git clone --branch reproduction/all-models https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization.git
cd TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization
python -m reproduction verify-source
python -m reproduction unpack-source ./ted-source
```

Python 3.10 or newer suffices for these extraction commands; they do not run GPU inference. The destination must be new. All 920 source-file hashes are checked before extraction. Collector scripts are then under `ted-source/neurips2026/scripts/`, alongside their host dependencies. Historical absolute paths require the relocation described in the [runtime preparation guide](../reproduction/README.md). Running an archived script directly with unchanged lab paths is not a portable reproduction command.

## How the banks are constructed

1. **Fix the source dataset, host, checkpoint, seed and preprocessing.** Use the archived recipe's settings; host feature layers, prompt interfaces, image size and budgets differ. See [execution recipes](../reproduction/execution-recipes.json) and [bank bindings](../reproduction/host-source-banks.json).
2. **Mine Hard-FPs from source normal images.** Rank local patches by the host's anomaly response and retain the high-scoring candidates under the recorded top-fraction and per-image/per-class limits. These are normal-source patches that the host treats as suspicious.
3. **Collect defects from source anomalous images using source masks.** Select local features at annotated defect locations, following the original collector's mask resizing, ranking and retention rule. The original metadata generally places source normal images in `train` and source anomalous images in `test`; here `test` belongs to the *source dataset*, not the held-out target dataset.
4. **Apply the host-specific bank budget.** Keep the original feature normalization, subsampling, layer/branch grouping and RNG behavior. For example, BayesPFL stores dictionaries of banks by layer, while FAPrompt also records branch scores and its text-axis binding. There is no single bank format or budget shared by all hosts.
5. **Save bank tensors and provenance.** Record source selection, model/checkpoint identity, seed and numerical settings. C-TED's source-fitted calibrators are separate from collecting the Defect and Hard-FP banks.

Target images, masks and evaluation scores must not select bank entries or tune bank construction. Same-dataset held-out-class controls additionally require the recorded source exclusion. Fresh construction needs an empty, separate cache; using the published historical bank files is saved-bank evaluation, not rebuilding. Do not delete existing banks or evidence to clear a cache.

## Saved banks and validation scope

For immediate evaluation using the historical banks, see [artifact downloads](ARTIFACTS.md) and [Hugging Face](https://huggingface.co/KIMJINYOUNG/TED-reproducibility). `prepare-source-banks` downloads and verifies saved banks; it does **not** mine new banks.

The [verified execution index](../reproduction/validation/a10-20261009/verified-results.json) identifies successful per-configuration evaluation checks. It does not establish fresh reconstruction of every bank, every reported seed aggregate, or the entire paper. We are completing the fresh-construction entry points and will document their verified model/dataset/seed scope separately.
