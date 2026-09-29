# TED: Text-Axis Evidence Decomposition for Prompted Anomaly Localization

<p align="center">
  <strong>NeurIPS 2026 · Main Track · Poster</strong><br>
  JinYoung Kim · Geonho Kim · GiJeong Park · Geonu Lee · Youngjoon Yoo<br>
  <sub>Corresponding author: Youngjoon Yoo</sub>
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=2lzL7Y6lmU">Paper / OpenReview</a> ·
  <a href="#method">Method</a> ·
  <a href="#results">Results</a> ·
  <a href="#qualitative-results">Visualizations</a> ·
  <a href="#code-and-reproduction">Code & Reproduction</a> ·
  <a href="#citation">Citation</a>
</p>

Official repository for **TED**, a post-hoc local scoring method for prompted anomaly localization.
The paper experiments are complete; public code packaging and reproduction checks are proceeding in stages.

## News

- **September 2026:** TED has been accepted to **NeurIPS 2026 Main Track as a Poster**!

## Method

<p align="center">
  <img src="assets/method.jpg" alt="TED overview: frozen host features, defect and hard-FP evidence banks, text-axis support, and T-TED/C-TED readouts" width="100%">
</p>

TED compares source defect support with source hard false-positive (Hard-FP) support along the host's normal-to-anomaly text axis. The backbone and prompts remain frozen.

1. **Keep the host fixed.** Obtain patch features and the host's normal/anomaly text embeddings.
2. **Construct two source-only banks.** Collect source defect patches using source masks and mine high-scoring source-normal patches as Hard-FPs.
3. **Compare evidence on the text axis.** Project both the query and bank features onto the same normal-to-anomaly text direction and contrast their support.
4. **Choose the readout.** Use the support margin directly with T-TED, or add a bounded source-calibrated correction to the host score with C-TED.

| | T-TED | C-TED |
|---|---|---|
| Main use | Frozen VLMs with raw prompt scoring | Adapted CLIP-AD hosts |
| Local score | Defect support minus weighted Hard-FP support | Host score plus bounded source-trained residual |
| TED calibration training | None | Source-only calibration |
| Backbone and prompts | Frozen | Frozen |
| Target-domain training or calibration | None | None |

Banks store full-dimensional features; query and bank support are computed in the same text-axis coordinate. Axis selection and insertion follow the host interface; see the [C-TED implementation notes](docs/CTED.md#inference-and-host-integration). Both variants use source evidence, including source defect annotations; **train-free does not mean source-free**. Target labels, masks, and performance are not used for bank construction or calibration.

Unlike generic Full-D memory scoring, TED contrasts two source evidence roles on the text axis and preserves the host-score anchor in C-TED. [Same-bank retrieval controls](docs/RESULTS.md#same-bank-retrieval-controls) examine this distinction.

## Results

The study covers frozen CLIP backbones and ImageBind, adapted CLIP-AD hosts, and cross-dataset localization on MVTec AD, VisA, MPDD, and BTAD. MVTec AD 2 is additionally used for the frozen-backbone diagnostic.

### C-TED on adapted hosts

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

Gains depend on the host and metric: BayesPFL is a near-no-op boundary case, and AdaCLIP's average P-AP slightly decreases. Source mismatch and host-interface compatibility can limit improvements; better pixel localization does not guarantee better image-level screening.

[More: same-bank controls and measured inference overhead →](docs/RESULTS.md)

## Qualitative results

<p align="center">
  <img src="assets/qualitative.jpg" alt="Five selected FAPrompt examples: input with red ground-truth boundaries, host heatmap, and C-TED heatmap" width="760">
</p>

**Selected illustrative examples, all using FAPrompt—not an unbiased sample of the test set.** Rows show MPDD, BTAD, two MVTec AD carpet examples, and VisA fryum. The source is MVTec AD for MPDD/BTAD/VisA and VisA for MVTec AD.

Each Host/C-TED pair shares the same color scale, using the pooled 2nd–99.5th percentile range of the two score maps. Red contours indicate ground-truth defects. These selected improvement examples illustrate local behavior; aggregate results and limitations should be considered alongside them.

## Code and reproduction

**Completed experiments and public code availability are separate.** The table below describes release progress, not experiments still to be run.

| Component | Status |
|---|---|
| Research overview and selected figures | Available |
| Reported result summaries | Available |
| T-TED feature-level scoring core | [Available and tested](docs/TTED.md) |
| C-TED calibration and feature-level inference | [AA-CLIP, AdaCLIP, FAPrompt cores available and CPU-tested](docs/CTED.md) |
| Host integrations and source-bank construction | Public release in preparation |
| Full benchmark environment and evaluation commands | Packaging and reproduction checks pending |
| Checkpoints / calibrated residuals | Public release in preparation |
| Final camera-ready / arXiv link | To be added when available |

### Try C-TED

The C-TED cores retain the original **host-specific** training and readout functions:

| Host | Code |
|---|---|
| AA-CLIP | [`ted/cted/aaclip.py`](ted/cted/aaclip.py) |
| AdaCLIP | [`ted/cted/adaclip.py`](ted/cted/adaclip.py) |
| FAPrompt | [`ted/cted/faprompt.py`](ted/cted/faprompt.py) |

With PyTorch and NumPy installed, run from the repository root:

```bash
python -m examples.cted_synthetic --host all
python -m unittest discover -s tests -v
```

These examples use synthetic features, not paper benchmark data. Source calibration
and feature-level inference were checked against the original functions; complete
bank mining, detector integration, and benchmark reproduction are not yet packaged.
See [C-TED training, input contracts, and host integration](docs/CTED.md).

### Try the T-TED core

From the repository root, with PyTorch installed:

```bash
python -m unittest discover -s tests -v
python -m examples.tted_synthetic
```

CPU validation used Python 3.10.19 and PyTorch 2.9.1+cu128; GPU validation is not complete.
The released module scores supplied features. It does not include a full image detector, and the synthetic example does not reproduce a paper result.
See [T-TED usage](docs/TTED.md), [verification](docs/VERIFICATION.md), and the
[reproducibility notes](docs/REPRODUCIBILITY.md).
Experiments described in the paper should not be confused with currently available repository functionality.

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
