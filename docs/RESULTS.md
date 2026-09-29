# Experimental summaries

[← Research overview](../README.md)

These are author-reported results from the TED submission and additional author-response experiments, not new measurements obtained from released code. The [OpenReview record](https://openreview.net/forum?id=2lzL7Y6lmU) is the paper reference; access to discussion may depend on its publication status. Code and per-setting reproduction artifacts are still being prepared.

## Reading the metrics

- **P-AUC:** pixel-level AUROC.
- **P-PRO:** the paper's per-region-overlap evaluation metric.
- **P-AP:** pixel-level average precision.
- **I-AUC:** image-level AUROC, a secondary metric for this work.
- **Δ:** absolute percentage-point change relative to the corresponding host.

Role-discrimination AUC and per-image pixel AUC diagnostics are not interchangeable with dataset-level P-AUC. Evaluation grids and aggregation units also differ across studies; do not average the following tables together.

## Same-bank retrieval controls

**Scope:** AA-CLIP, AdaCLIP, and FAPrompt × six transfers, covering all 18 settings of this comparison. Source banks, hosts, targets, and aggregation are held fixed; the evidence readout changes. Replacement substitutes the host local map; residual mode adds a correction.

| Readout | Mode | ΔP-AUC | ΔP-PRO | ΔP-AP |
|---|---|---:|---:|---:|
| Full-D 1-NN | Replacement | −45.65 | −39.39 | −20.68 |
| Full-D 1-NN | Residual | −10.99 | −0.22 | −7.66 |
| Full-D prototype | Replacement | −11.85 | −3.13 | −12.63 |
| Full-D prototype | Residual | −3.27 | +11.04 | −5.64 |
| Full-D LogMeanExp | Replacement | −24.70 | −18.56 | −16.67 |
| Full-D LogMeanExp | Residual | −8.68 | +3.69 | −7.86 |
| C-TED | Residual | +0.87 | +4.38 | +5.21 |

None of the tested Full-D readouts improves all three metrics on average; C-TED does. The prototype residual improves P-PRO more than C-TED but decreases P-AUC and P-AP. This supports the importance of the readout beyond generic bank access; it is not a claim that C-TED wins every metric or that every possible retrieval method has been tested.

This 18-setting control is separate from the 76-setting host-native aggregation in the README and from broader shared-interface audits.

## Inference overhead

**Protocol:** paired synchronized-GPU profiling on one RTX 6000 Ada, over all 733 BTAD images after eight warm-up images per host, as reported in the author response.

| Host | Host (ms/image) | Host + TED (ms/image) | Added (ms/image) | Added runtime | Added GPU memory (MB) |
|---|---:|---:|---:|---:|---:|
| AA-CLIP | 15.93 | 21.90 | 5.97 | 37.47% | 19.46 |
| FAPrompt | 136.50 | 155.91 | 19.41 | 14.22% | 28.07 |
| AdaCLIP | 183.26 | 190.10 | 6.84 | 3.73% | 8.25 |

The AA-CLIP percentage is relatively high because the host is fast. These measurements are specific to the evaluated implementation and hardware; they are not deployment guarantees and do not include a claim that source calibration is free.

## Evidence and applicability

The paper and author response additionally study source coverage, Hard-FP mining fractions, bank diversity, shared training recipes, role/axis controls, small defects, and image-level behavior. These are distinct audits with distinct settings. The release should preserve their configuration and source/target split rather than present them as a single benchmark.

The qualitative gallery contains selected FAPrompt improvement examples. It does not establish average performance, multi-host coverage, or robustness to every failure mode. Negative and near-zero aggregate changes remain visible in the main summary.
