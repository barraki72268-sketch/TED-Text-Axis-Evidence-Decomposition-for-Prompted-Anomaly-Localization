# C-TED: host-specific source calibration

This release provides the **AA-CLIP, AdaCLIP, and FAPrompt calibration and
feature-level inference cores**, extracted from the corresponding research
evaluators without modifying their function bodies. It is not yet a complete
image-to-metrics benchmark package. No host checkpoints, datasets, or upstream
detector implementations are bundled.

## Quick start

Install PyTorch and NumPy in your environment, then run from the repository root:

```bash
python -m examples.cted_synthetic --host all
python -m unittest discover -s tests -v
```

The example trains tiny calibrators on synthetic source features and applies
them to synthetic query tokens on CPU. Its three epochs, rank, bank size, and
other settings are **smoke-test settings, not paper reproduction defaults**.
Use `--host aaclip`, `--host adaclip`, or `--host faprompt` separately.

## Code map

| Host | Implementation | Calibration anchor | Output from the released readout |
|---|---|---|---|
| AA-CLIP | [`aaclip.py`](../ted/cted/aaclip.py) | Source text-axis score | One interpolated layer map |
| AdaCLIP | [`adaclip.py`](../ted/cted/adaclip.py) | Stored source host token scores | Corrected token scores |
| FAPrompt | [`faprompt.py`](../ted/cted/faprompt.py) | Stored source fused or branch scores | Corrected token scores for the selected calibrator |

Import with `from ted.cted import aaclip, adaclip, faprompt`.
The common mathematical structure does not imply interchangeable host interfaces.
AdaptCLIP and BayesPFL compatibility implementations are not part of this release.

## Inputs and source-only protocol

- `fp`: source-normal Hard-FP features, shape `[K_fp, D]`.
- `defect`: source defect features, shape `[K_defect, D]`.
- `axis`: normal-to-anomaly text direction, shape `[D]`.
- AdaCLIP banks additionally contain aligned `fp_score` and `defect_score`.
- FAPrompt uses `fp_{host_score_key}` and `defect_{host_score_key}`;
  `host_score_key="branch1_score"` selects `fp_branch1_score` and
  `defect_branch1_score`, for example.

Supply finite, detached floating-point source tensors on CPU; the trainers move
selected points to `device`. Feature and score rows must remain aligned during
sampling. Source and query features must use the same host feature interface.
The core normalizes feature rows internally. Both banks must be nonempty and
have sufficient independent source variation for the requested residual rank.
Use positive `tau` and `rank_temp`, and a positive point budget large enough for
pair construction. These are low-level research functions, not a validated input
schema. Disabled training or empty banks return `None`; callers must handle it.

The AdaCLIP/FAPrompt functions retain the research fallback to axis scores when
stored host scores are absent. **Supply the correct host scores for host-native
calibration; this fallback is not an equivalent reproduction protocol.**

Bank construction uses source annotations for defects and high-scoring source
normal patches for Hard-FPs. Target masks or target outcomes must not select
source banks, hyperparameters, or calibrators. Full image loading, host feature
extraction, category caps, candidate mining, and exact benchmark configurations
are not included yet; the included FPS helpers alone do not construct the banks.

## Source calibration

The source basis is built from the text axis, the difference between normalized
source defect/Hard-FP mean directions, and leading PCs of centered source feature
differences, followed by orthonormalization. Its realized rank can be smaller than
requested. `subspace_basis_control="source"` and (where applicable)
`pair_label_control="correct"` select the intended evidence roles; random/shuffled
controls retained in the source are diagnostic variants.

The sampler takes up to `max_train_points // 2` points from each bank. With
`hardpair_frac=0.75`, 75% are hard examples (highest-scoring Hard-FPs and
lowest-scoring defects), and the remainder are randomly sampled. Hard-FP pairing
is permuted each epoch. Set both `torch.manual_seed(seed)` and the trainer's
`seed` argument: the latter alone does not control all epoch permutations.

For corrected pair margin `m = s_defect - s_fp` and original margin `m0`, the
loss implemented by the residual trainers is:

```text
rank_loss = mean(softplus((rank_margin - m) / rank_temp))
w = sigmoid((m0 - rank_margin) / rank_temp)
preservation = sum(w * relu(m0 - m)) / max(sum(w), 1e-6)
loss = rank_loss + preserve_host_margin_weight * preservation
       + eta_reg * (eta**2 + readout_gamma**2)
```

Adam optimizes the low-rank correction, not the host. The transport magnitude
`eta` and residual gain `readout_gamma` are softplus-parameterized and capped by
`eta_max`; the transport and residual readout directions are normalized.
`fp_weight` can be fixed or learned. AA-CLIP should use
`subspace_readout_mode="host_residual"` for this residual path; its original
non-residual diagnostic path remains in the extracted function.

Returned dictionaries contain the source basis, transport direction and bias,
Hard-FP weight, residual direction (`subspace_score_w`), and gain
(`readout_gamma`), plus source diagnostics. Save the dictionary together with the
host checkpoint identity, source bank identity, axis, and training settings;
the dictionary alone is not a full detector checkpoint.

## Inference and host integration

Query tokens have shape `[B, N, D]`; scalar support tensors have shape `[B, N]`.
Compute defect/Hard-FP support on the same axis for query and bank features,
then call the host's `prescore_subspace_transport_*` function and residual readout.
The synthetic example demonstrates these calls explicitly. Spatial
standardization requires **more than one token**; AA-CLIP's map helper additionally
requires a square token grid. The inference helpers assume appropriate normalized
host tokens and do not extract them from images.

### AA-CLIP

`train_one_layer_subspace_transport_calibrator` trains a layer-specific module.
`prescore_subspace_transport_seg_tokens` performs bounded transport, and
`subspace_host_residual_map_from_tokens` anchors its output to the supplied target
text axis before interpolation. In the inspected native evaluator, support uses
the source axis while this map readout uses the target axis. Do not silently
replace this with a universal target-axis support recipe. The returned map is
not the final multi-layer/host-fused image prediction.

### AdaCLIP

`train_subspace_host_residual_calibrator` uses stored layer-level source host
scores. `subspace_host_residual_token_score` returns corrected token scores.
The native evaluator subsequently forms the token residual, interpolates and
normalizes it, applies the configured activation/gate and strength, then inserts
it into the host interface. The included `add_to_logit_difference` helper adjusts
the anomaly-versus-normal logit difference while preserving their center.
Full gating, layer aggregation, and final softmax/smoothing are not packaged here.
Consequently, the token readout alone is not the complete AdaCLIP prediction.

### FAPrompt

The `branch_calibrated` evaluator trains separate calibrators using
`host_score_key="branch1_score"` and `"branch2_score"`. Each branch is corrected
against its own stored host scores. Its two token residuals are averaged, gated,
and added with the configured strength to the fused baseline token score before
upsampling/smoothing. The evaluator retains the official image-level score.
The included example exercises one explicit branch only. Training a single
calibrator with the default `"score"` key is a distinct fused-score variant,
not a substitute for the two-branch pipeline.

## Provenance and validation

[`cted_source_manifest.json`](cted_source_manifest.json) records the original
research script and SHA-256 hashes of each extracted function. Unit tests check
those function hashes. The functions are intentionally separate across hosts
to retain their source behavior, including support chunking and score interfaces.

See [verification](VERIFICATION.md) for exact checks and limitations. Synthetic
CPU parity does not validate paper numbers, GPU execution, or complete host
integration. Upstream detector code and checkpoints remain subject to their own
licenses; obtain them from the original projects linked in the main README.
