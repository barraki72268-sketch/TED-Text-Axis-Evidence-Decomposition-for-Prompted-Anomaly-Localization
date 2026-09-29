# T-TED feature-level core

This first code release implements the support and signed-margin calculation only.
It is not the full CLIP/ImageBind benchmark pipeline, and does not reproduce the
paper's tables on its own. C-TED is not included.

## Run

From the repository root, with PyTorch already installed:

```bash
python -m unittest discover -s tests -v
python -m examples.tted_synthetic
```

Validated locally with Python 3.10.19 and PyTorch 2.9.1+cu128.
This is a tested environment, not a claim that all host adapters support this version.
The synthetic example uses random features, not industrial anomaly data.

## Interface

```python
from ted import text_axis, tted_score

# Host-projected normal/anomaly text embeddings: [D].
axis = text_axis(normal_embedding, anomaly_embedding)
# Query patches [N,D], source defect bank [B_def,D], source Hard-FP bank [B_fp,D].
margin = tted_score(patches, defect_bank, hard_fp_bank, axis,
                    tau=source_config_tau, fp_weight=1.0)
```

All inputs must be floating-point tensors in the same host feature coordinate
system. This snippet assumes the named feature tensors and source configuration
have already been constructed; it is not a standalone image-inference command.
Text-axis construction normalizes each text embedding before taking the normalized
anomaly-minus-normal direction. Raw text-space features cannot be mixed with
unprojected visual-token features.

The scorer normalizes patches and bank rows in float32, projects both banks onto
the current axis, computes log-mean-exp negative squared-distance support, and
returns defect support minus weighted Hard-FP support. Query chunking limits
temporary pairwise allocations. An empty bank returns zero support, preserving
the research code's ablation fallback; ordinary two-bank evaluation requires both
banks to contain source evidence.

No source bank is inferred from target labels. Bank mining, FPS/caps, prompts,
backbone loading, interpolation, smoothing, layer fusion, image-level scoring,
and benchmark metrics are intentionally not part of this release.

## Verification and provenance

The mathematical core was extracted and refactored from the project's
`official_parallel_test_rawclip.py` functions
`logmeanexp_negative_sqdist_1d` and `compute_layer_host_and_ted`.
The ImageBind evaluator uses the same support/margin calculation in joint space.
The release adds input validation and query chunking; empty-bank behavior is preserved.

Unit tests cover dense-reference agreement, chunking, role swapping, identical
banks, empty inputs, bank ordering, changing query axes, invalid inputs, and
an opt-in CPU/CUDA comparison (set `TED_TEST_CUDA=1`). GPU validation is not
claimed complete; see the verification note. Local parity checks against the
original research functions are documented in the release verification note.

Feature-level numerical parity is not a substitute for end-to-end dataset
reproduction. Do not infer paper-level performance from the synthetic smoke test.
