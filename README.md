# TED: Text-Axis Evidence Decomposition for Prompted Anomaly Localization

<p align="center">
  <strong>NeurIPS 2026 · Main Track · Poster</strong><br>
  JinYoung Kim · Geonho Kim · GiJeong Park · Geonu Lee · Youngjoon Yoo<br>
  <sub>Corresponding author: Youngjoon Yoo</sub>
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=2lzL7Y6lmU">Paper / OpenReview</a> ·
  <a href="#method-overview">Method</a> ·
  <a href="#results-at-a-glance">Results</a> ·
  <a href="#qualitative-results">Visualizations</a> ·
  <a href="#release-status">Release status</a> ·
  <a href="#citation">Citation</a>
</p>

> **TL;DR:** A high anomaly response is not necessarily a defect. TED compares support from source defects and source hard false positives along the host's text axis to improve local anomaly ranking—without updating the host backbone or prompts.

**Official research repository. The feature-level T-TED scoring core is available.** Full benchmark pipelines, C-TED, and checkpoints are still being prepared. The core is executable on supplied features; it is not an end-to-end image detector.

## News

- **September 29, 2026:** Released the feature-level T-TED core, tests, and a synthetic example. See [usage and scope](docs/TTED.md).

- **September 2026:** TED was accepted to NeurIPS 2026 as a poster.
- **September 2026:** Research documentation and selected qualitative results are available. Code release is in preparation.

## Why TED?

CLIP-based anomaly detectors use prompt learning or lightweight adaptation to increase defect sensitivity. Yet stronger sensitivity does not always produce better localization: textures, edges, reflections, and other complex normal regions can receive scores comparable to true defects. We call these competing normal responses **hard false positives (Hard-FPs)**.

TED treats this as an **evidence-decoding problem**. Instead of trusting the magnitude of a local anomaly response alone, it asks:

> Is this response better supported by source defects, or by source normal patches that the host misranks as anomalous?

The method changes the local readout rather than retraining the host. Its primary objective is **pixel-level anomaly localization**, not a universal improvement in image-level screening.

## Method overview

<p align="center">
  <img src="assets/method.jpg" alt="TED overview: frozen host features, defect and hard-FP evidence banks, text-axis support, and T-TED/C-TED readouts" width="100%">
</p>

1. **Keep the host fixed.** Obtain patch features and the host's normal/anomaly text embeddings.
2. **Construct two source-only banks.** Collect source defect patches using source masks and mine high-scoring source-normal patches as Hard-FPs.
3. **Compare evidence on the text axis.** Project both the query and bank features onto the same normal-to-anomaly text direction and contrast their support.
4. **Choose the readout.** Use the support margin directly with T-TED, or add a bounded source-calibrated correction to the host score with C-TED.

| | T-TED | C-TED |
|---|---|---|
| Main use | Frozen VLMs with raw prompt scoring | Adapted CLIP-AD hosts |
| Local score | Defect support minus Hard-FP support | Host score plus bounded source-trained residual |
| TED calibration training | None | Source-only calibration |
| Backbone and prompts | Frozen | Frozen |
| Target-domain training or calibration | None | None |

**Train-free does not mean source-free.** T-TED still uses source evidence, including source defect annotations. Neither variant uses target labels, masks, or target performance to construct banks or tune the correction. Target images are, of course, processed at inference; class names may instantiate the fixed prompt protocol.

The banks retain **full-dimensional features** and are projected onto the query-conditioned text direction. TED does not transfer a fixed scalar histogram or require identical source/target score distributions. It relies on transferable evidence roles, which can fail under mismatch.

### How is this different from memory-bank scoring?

Bank lookup itself is not the claimed novelty. Canonical normal-memory methods such as PatchCore score distance to normal visual memory. TED instead contrasts **source defects against host-specific source Hard-FPs**, uses the **text axis** as the decision coordinate, and—in C-TED—**preserves the host-score anchor**. Hard-FP mining does not update the host.

See [experimental summaries](docs/RESULTS.md) for same-bank retrieval controls that keep the banks fixed while changing the readout.

## Results at a glance

The study covers frozen CLIP backbones and ImageBind, adapted CLIP-AD hosts, and cross-dataset localization on MVTec AD, VisA, MPDD, and BTAD. MVTec AD 2 is additionally used for the frozen-backbone diagnostic.

### Adapted-host localization

The following is the author-reported aggregation of **76 settings from the complete adapted-host result tables**, also summarized in the author response. Values are changes from each corresponding host in **absolute percentage points**, not absolute performance or relative percentages.

| Host | ΔP-AUC | ΔP-PRO | ΔP-AP |
|---|---:|---:|---:|
| AA-CLIP | +9.07 | +6.24 | +1.07 |
| AdaCLIP | +0.38 | +5.47 | −0.14 |
| AdaptCLIP | +0.08 | +0.07 | +0.34 |
| FAPrompt | +3.47 | +8.33 | +10.28 |
| BayesPFL | −0.02 | −0.02 | −0.08 |
| **Overall (76 settings)** | **+2.25** | **+3.90** | **+2.36** |

The overall row averages settings, not the five host means. These are paper-reported results, not a new benchmark run from this repository. Host-native results must not be mixed with separately reported shared-interface controls.

The variation matters: TED does not improve every host or metric. BayesPFL is a near-no-op boundary case; AdaCLIP's average P-AP slightly decreases. Residual Hard-FP competition and transferable source evidence are important applicability conditions.

[More: same-bank controls and measured inference overhead →](docs/RESULTS.md)

## Qualitative results

<p align="center">
  <img src="assets/qualitative.jpg" alt="Five selected FAPrompt examples: input with red ground-truth boundaries, host heatmap, and C-TED heatmap" width="760">
</p>

**Selected illustrative examples, all using FAPrompt—not an unbiased sample of the test set.** Rows show MPDD, BTAD, two MVTec AD carpet examples, and VisA fryum. The source is MVTec AD for MPDD/BTAD/VisA and VisA for MVTec AD.

Each Host/C-TED pair shares the same color scale, using the pooled 2nd–99.5th percentile range of the two score maps. Red contours indicate ground-truth defects. These selected improvement examples illustrate local behavior; aggregate results and limitations should be considered alongside them.

## Release status

| Component | Status |
|---|---|
| Research overview and selected figures | Available |
| Reported result summaries | Available |
| T-TED feature-level scoring core | [Available and tested](docs/TTED.md) |
| C-TED implementation | In preparation |
| Host integrations and source-bank construction | In preparation |
| Verified environment and evaluation commands | In preparation |
| Checkpoints / calibrated residuals | In preparation |
| Final camera-ready / arXiv link | To be added when available |

### Try the T-TED core

From the repository root, with PyTorch installed:

```bash
python -m unittest discover -s tests -v
python -m examples.tted_synthetic
```

Validated with Python 3.10.19 and PyTorch 2.9.1+cu128.
The synthetic example uses random features and does not reproduce a paper result.
See [T-TED usage](docs/TTED.md), [verification](docs/VERIFICATION.md), and the
[reproducibility notes](docs/REPRODUCIBILITY.md).
Experiments described in the paper should not be confused with currently available repository functionality.

## Scope and limitations

- **Source evidence matters.** Missing source defect patterns or mismatched Hard-FPs can weaken or reverse the correction.
- **Host interfaces matter.** Native score insertion and shared-interface diagnostic variants are different protocols.
- **Localization is primary.** Better pixel ranking does not guarantee better image-level detection.
- **Small defects remain difficult.** Sub-token defects can be resolution-limited; small area and semantic rarity are not interchangeable.
- **No universal gain is claimed.** Already well-separated host maps may leave little useful correction.
- **Runtime is not free.** Overhead depends on the host, bank size, patch count, and selected layers; see [profiling results](docs/RESULTS.md#inference-overhead).

## Citation

```bibtex
@inproceedings{kim2026ted,
  title     = {{TED}: Text-Axis Evidence Decomposition for Prompted Anomaly Localization},
  author    = {Kim, JinYoung and Kim, Geonho and Park, GiJeong and Lee, Geonu and Yoo, Youngjoon},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=2lzL7Y6lmU}
}
```

## Acknowledgments

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) [RS-2021-II211341, Artificial Intelligence Graduate School Program (Chung-Ang University), and RS-2022-II220124, Development of Artificial Intelligence Technology for Self-Improving Competency-Aware Learning Capabilities]. This research was also supported by the AI Seoul Tech Research Support Program of the Seoul Future Foundation.

We acknowledge the authors and maintainers of CLIP, ImageBind, the evaluated CLIP-AD hosts, and the datasets used in this study. Useful upstream projects include [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP), [FAPrompt](https://github.com/mala-lab/FAPrompt), [AdaCLIP](https://github.com/caoyunkang/AdaCLIP), and [AA-CLIP](https://github.com/Mwxinnn/AA-CLIP). Third-party assets retain their original terms.

## Contact

For research or release questions, please [open an issue](https://github.com/barraki72268-sketch/TED-Text-Axis-Evidence-Decomposition-for-Prompted-Anomaly-Localization/issues) or contact JinYoung Kim at **barraki7226@cau.ac.kr**.
