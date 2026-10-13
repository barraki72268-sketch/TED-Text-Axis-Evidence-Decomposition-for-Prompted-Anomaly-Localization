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
| AdaptCLIP VL/TL branches | [analyze_adaptclip_vl_score_forms.py](../research/bank_construction/analyze_adaptclip_vl_score_forms.py) | `collect_vl_source_banks` |
| BayesPFL | [collect_bayespfl_source_banks.py](../research/bank_construction/collect_bayespfl_source_banks.py) | `collect_source_patch_banks` |
| RawCLIP | [official_parallel_test_rawclip.py](../research/bank_construction/official_parallel_test_rawclip.py) | `collect_source_patch_banks` |
| ImageBind | [official_parallel_test_rawimagebind.py](../research/bank_construction/official_parallel_test_rawimagebind.py) | `collect_source_patch_banks` |

The [manifest](../research/bank_construction/manifest.json) records each original archive path, file size and SHA-256. Original component licenses are listed in [source notices](../reproduction/SOURCE-NOTICES.md).

For recorded AdaptCLIP VL-refine recipes, the
[original evaluator](../research/bank_construction/official_parallel_test_adaptclip_vlrefine.py)
imports the model builder from `official_parallel_test_adaptclip.py`, and imports
`get_or_collect_vl_banks` from `analyze_adaptclip_vl_score_forms.py`. That cache
wrapper calls `collect_vl_source_banks`. For `ted_insertion=dual_branch`, the
original evaluator supplies VL mining/VL features for one bank and TL mining/TL
features for the other. Use those recorded branch settings and budgets; the
plain AdaptCLIP collector is a different collection path. These additional
scripts are unchanged archive bytes and require the full source dependencies.

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

### Verified BayesPFL source collection

The release now includes a runnable source-collection entry point for the recorded
BayesPFL B/16+ MVTec-to-BTAD recipe, seed 0. It calls the original collector
directly and creates new bank tensors from MVTec normal images and defect masks.
During collection, an audit hook rejects reads of the historical bank and BTAD
target inputs, and rejects network access. The recorded source limits remain
8 normal and 8 defect images per class, with a 1,024-entry cap per layer.

Use Linux, Python 3.10, the recorded model dependencies and a GPU Slurm allocation.
The [observed package inventory](../reproduction/validation/fresh-bank-20261010/bayes-bplus-mvtec-s0/packages-observed.txt)
records the environment used for this check; it is not a claim that a clean
installation of every listed package has been independently verified.
Download the recipe's checkpoint, backbone and saved-bank inputs using the
[artifact guide](ARTIFACTS.md), and supply the original MVTec and BTAD data roots
using the [runtime preparation format](../reproduction/README.md).

```bash
python -m reproduction prepare-run bayespfl-vitb_plus-mvtec2btad-seed0 ./runs/bayes-source \
  --objects ./inputs --datasets ./dataset-roots.json
# Inside your allocated GPU Slurm shell:
python -m reproduction.bayes_bank_build ./runs/bayes-source ./fresh-banks/bayes-source
```

Both destinations must be new. This produces `source-bank.pt` and
`construction.json`; keep stdout/stderr as the collection log. Preparation and
preflight still verify saved-bank bytes and both dataset manifests. The collector
does not load that saved bank or evaluate the target. Preparation from only model weights and source data has since been checked on
CPU, as documented below; GPU collection on that preparation remains pending.

[Execution and tensor audit](../reproduction/validation/fresh-bank-20261010/bayes-bplus-mvtec-s0/index.json)
record Slurm job 13805, a successful source collection, four layers with
240 Hard-FP and 1,024 defect entries each, and finite normalized 640-dimensional
tensors. This verifies this source-collection configuration. Evaluation using
this newly built bank, other model bank builders and paper-wide reproduction
remain separate checks. The historical-bank tensor comparison is documented below.

A second collection from a clean clone of the public GitHub commit
`db9f5fa1098e6c0a8b30e8c81ebca58d7cb9b604` also completed in Slurm job 13808.
It used a new output directory and the same verified prepared inputs, seed and
GPU runtime. CPU comparison found all eight layer-bank tensors exactly equal
to the first fresh collection, and the serialized bank SHA also matched.
[Public-checkout collection and comparison evidence](../reproduction/validation/fresh-bank-20261010/bayes-bplus-public-checkout-v3/index.json)
records this check. It does not compare against the historical bank or verify
target evaluation or preparation without saved-bank inputs.

A subsequent CPU comparison against the recipe's SHA-verified historical bank
found **all eight Hard-FP/Defect layer tensors exactly equal**, with maximum
absolute error 0, for this B/16+ MVTec seed-0 configuration. The serialized files
have different hashes: the historical `meta` value is null in the comparison, while the fresh
output includes construction metadata. This is tensor identity, not file identity.
[Historical-bank comparison evidence](../reproduction/validation/fresh-bank-20261011/bayes-historical-tensors-v1/index.json)
binds both bank hashes to the public-checkout construction proof. This comparison
does not establish target evaluation, other configurations or paper-wide means.

### Preparation using only source data and model weights

For BayesPFL B/16+ MVTec, seed 0, CPU preparation has also been checked without
saved-bank objects or BTAD inputs. Use an object directory containing the
recipe's checkpoint and backbone, and a dataset-roots JSON containing only the
MVTec entry:

```bash
python -m reproduction prepare-run bayespfl-vitb_plus-mvtec2btad-seed0 ./runs/bayes-cold \
  --objects ./model-inputs --datasets ./mvtec-roots.json --bank-collection-only
```

This verifies the model weights, original source and MVTec bytes, preserves the
checkpoint basename and recorded numerical arguments, and writes a workspace
restricted to source collection. `run-prepared` rejects it because it has no
target data. The [CPU preparation proof](../reproduction/validation/fresh-bank-20261011/bayes-cold-preparation-v1/index.json)
records actual denial checks for historical-bank and target reads and network
blocking throughout preparation and validation. GPU collection using this new
workspace has not yet been run. Once checked in a valid allocation, the intended
collection command is `python -m reproduction.bayes_bank_build ./runs/bayes-cold ./fresh-banks/bayes-cold`.
The earlier two successful GPU collections used the prepared workspace described
above; they are separate evidence from this CPU preparation check. Cross-dataset
evaluation using the new bank remains a separate step.

The same CPU preparation CLI also passed from a clean GitHub checkout pinned to
`86dfdb4f5ca72707494853289a4276a3fa1996ac`, using new input and workspace
directories containing only the two model objects and MVTec data roots.
[Public-checkout CLI evidence](../reproduction/validation/fresh-bank-20261011/bayes-cold-public-checkout-v2/index.json)
preserves the plan, actual command wrapper, denial checks and log. This check
does not add GPU bank-generation or target-evaluation evidence.

### AdaptCLIP source-only preparation

Source-only preparation is also supported for cross-dataset AdaptCLIP recipes.
For `adaptclip-vitl14openai-mvtec2btad-seed0`, the CPU preparation CLI passed
with only the model checkpoint/backbone and MVTec dataset roots. Both the VL and
TL historical-bank paths remain absent; an actual denial hook rejected bank and
BTAD read self-checks and saw no further attempted blocked reads. Ordinary
target evaluation rejects this source-only workspace.

```bash
python -m reproduction prepare-run adaptclip-vitl14openai-mvtec2btad-seed0 ./runs/adapt-cold \
  --objects ./adapt-model-inputs --datasets ./mvtec-roots.json --bank-collection-only
```

[AdaptCLIP CPU preparation evidence](../reproduction/validation/fresh-bank-20261012/adapt-cold-preparation-v2/index.json)
records a prototype overlay run. The same CPU preparation also passed from a
clean published checkout at `61de3cbef1732d03ae69f25296ecad3fbf326ed3`, using new
model-only input roots and a new workspace.
[Public-checkout evidence](../reproduction/validation/fresh-bank-20261013/adapt-cold-public-v3/index.json)
binds the actual CLI wrapper, plan, denial checks and rejection of target
evaluation. Fresh VL/TL collection and verification of the GPU construction
path remain pending.
The BayesPFL bank-generation command above does not support AdaptCLIP.

The experimental AdaptCLIP entry point can inspect this source-only workspace:

```bash
python -m reproduction.adapt_bank_build ./runs/adapt-cold --inspect
```

Inspection executes only the original argument-parser prefix and reports the
original model call and direct VL/TL collection arguments. It initializes no
model and constructs no bank. [Prototype inspection evidence](../reproduction/validation/fresh-bank-20261013/adapt-bank-build-inspection-v1/index.json)
records an actual CPU invocation, checks of all 46 recorded AdaptCLIP recipe
arguments, and rejection of collection without a GPU allocation before creating
an output directory. That initial prototype used a code overlay. Inspection
also passed from a clean published checkout at
`4cc102cfab7be89980621ad8776771490ab0ef0c`.
[Public-checkout inspection and cache-format evidence](../reproduction/validation/fresh-bank-20261013/adapt-bank-public-inspection-v3/index.json)
records the actual CPU CLI and unallocated collection rejection. Separately,
the original save/load/getter functions accepted two synthetic VL/TL tensor
fixtures with exact tensor and metadata round trips and no source mining.
This compatibility test is not fresh bank reconstruction or image inference.

The collection implementation calls the original `collect_vl_source_banks`
directly for both recorded branches, retaining original seed, model-build
arguments, metadata and bank-save function. It requires a new destination and
live Slurm GPU membership. Its GPU construction path has **not been executed or
verified**; the published inspection checks do not establish new-bank tensor or
metric agreement. No AdaptCLIP bank from this entry point is published yet.

### FAPrompt source-only preparation

For `faprompt-vitb16plus-mvtec2btad-seed1`, CPU source-only preparation passed
with only the original checkpoint/backbone and MVTec dataset roots. An actual
read-denial hook rejected historical-bank and BTAD read self-checks and observed
no further attempted blocked reads. The prepared source cache must remain empty
before collection; ordinary target evaluation rejects this workspace.

```bash
python -m reproduction prepare-run faprompt-vitb16plus-mvtec2btad-seed1 ./runs/fap-cold \
  --objects ./fap-model-inputs --datasets ./mvtec-roots.json --bank-collection-only
```

[FAPrompt CPU preparation evidence](../reproduction/validation/fresh-bank-20261013/fap-cold-preparation-v1/index.json)
records a prototype overlay run. The same CPU preparation also passed from a
clean published checkout at `238ff4aefcf67e2be2068e0b5ab1045fb9290694`, with new
model-only input roots and a new workspace.
[Public-checkout preparation and static call evidence](../reproduction/validation/fresh-bank-20261013/fap-cold-public-v2/index.json)
binds the command wrapper, plan, actual denial checks and original collector
argument mapping for 20 execution-catalog recipes. The direct collection path
must retain the original model loader's computed `effective_dpam_layer`, seed
and class/total budgets. This static mapping does not verify bank construction
or cover all paper ablations. Actual GPU collection remains pending. The existing
BayesPFL and AdaptCLIP collection entry points do not support FAPrompt.

The experimental FAPrompt entry point supports CPU inspection:

```bash
python -m reproduction.fap_bank_build ./runs/fap-cold --inspect
```

[Prototype inspection evidence](../reproduction/validation/fresh-bank-20261013/fap-bank-build-inspection-v1/index.json)
records an actual CPU CLI invocation on the prepared MVTec source workspace,
read-denial self-checks for historical banks, target data and source bank caches,
and rejection of collection without an allocation before creating a destination.
Inspection executes the original parser and preset configuration assignment;
it initializes no model and constructs no bank. The code and evidence use a
prototype overlay. Parser and direct-call checks cover 20 recorded recipes,
not all paper ablations.

Inspection and the unallocated collection rejection also passed from a clean
published checkout at `13316b8317661735504b504f759fa7fd27a23b67`, reusing the
verified source-only workspace read-only.
[Public-checkout inspection evidence](../reproduction/validation/fresh-bank-20261013/fap-bank-public-inspection-v2/index.json)
binds the actual CLI wrapper, plan, entry-point hash, three denial self-checks
and zero subsequent attempted blocked reads. This adds CPU inspection evidence.

The collection implementation calls the original `collect_source_banks`
directly, retaining `setup_seed`, model-build arguments, the computed
`effective_dpam_layer`, collection budgets and the complete original bank
dictionary and protocol metadata. It requires a new destination and live Slurm
GPU membership. Its GPU construction path has **not been executed or verified**;
no FAPrompt bank from this entry point is published yet. Source calibration and
cross-dataset target evaluation remain separate pending work.

### BayesPFL target preparation

The following preparation CLI has passed a CPU check for BayesPFL B/16+
MVTec-to-BTAD, seed 0. Supply the collection output directory and the exact
`run.json` used to collect it. Preparation binds the new bank hash, collection
proof and plan, and preserves the recorded settings and checkpoint basename.
The current default needs only the checkpoint/backbone objects, both dataset
roots and the new bank with its construction proof and collection plan. It does
not require the historical saved-bank object. The optional `--historical-preflight`
retains the earlier preparation mode and its full saved-bank byte verification.

```bash
python -m reproduction.bayes_fresh_evaluation prepare \
  bayespfl-vitb_plus-mvtec2btad-seed0 ./runs/bayes-fresh-target \
  --objects ./inputs --datasets ./dataset-roots.json \
  --collection ./fresh-banks/bayes-source --collection-plan ./runs/bayes-source/run.json
# Only inside a new, valid GPU Slurm allocation:
python -m reproduction.bayes_fresh_evaluation run ./runs/bayes-fresh-target
```

The destination must be new. The run command checks live Slurm GPU allocation
and actual process membership; an unallocated invocation was rejected in the
[CPU preparation evidence](../reproduction/validation/fresh-bank-20261011/bayes-fresh-evaluation-preparation-v2/index.json).
That initial evidence used a prototype code overlay on a GitHub checkout.
The same CPU CLI subsequently passed from a clean published checkout at
`ca4d6f944413f51607674a9b39febfd4932a1d44`, using a separate new workspace.
[Public-checkout preparation evidence](../reproduction/validation/fresh-bank-20261011/bayes-fresh-evaluation-public-v3/index.json)
binds the plan, collection provenance, command wrapper and unallocated-run
rejection. GPU target evaluation remains pending; these preparation checks
do not add cross-dataset metric results.

The newer bank-independent preparation was checked on CPU using an object root
containing only the two model objects. An actual read-denial hook rejected a
historical-bank self-check and observed no further attempted historical-bank
reads during preparation or validation. Both source and target dataset bytes
were verified, the fresh bank was bound, and GPU execution without an allocation
was rejected. [Bank-independent preparation evidence](../reproduction/validation/fresh-bank-20261011/bayes-fresh-evaluation-bankfree-v4/index.json)
preserves that prototype overlay run and exact code bytes. The same preparation
subsequently passed from a clean published checkout at
`c37e03278fef87f6168576b36a3f54f9089d58ca`, with new model-only inputs and a new
workspace. [Clean public-checkout evidence](../reproduction/validation/fresh-bank-20261012/bayes-fresh-evaluation-bankfree-public-v5/index.json)
binds the actual CLI wrapper, plan, read-denial self-check and unallocated-run
rejection. Actual GPU target evaluation remains pending. The collection proof supplied here comes from the
previous successful GPU construction; collection on the newer source-only
preparation remains a separate pending check.

For immediate evaluation using the historical banks, see [artifact downloads](ARTIFACTS.md) and [Hugging Face](https://huggingface.co/KIMJINYOUNG/TED-reproducibility). `prepare-source-banks` downloads and verifies saved banks; it does **not** mine new banks.

The [verified execution index](../reproduction/validation/a10-20261009/verified-results.json) identifies successful per-configuration evaluation checks. It does not establish fresh reconstruction of every bank, every reported seed aggregate, or the entire paper. We are completing the fresh-construction entry points and will document their verified model/dataset/seed scope separately.
