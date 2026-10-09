# Historical core and engineering verification

These are dated validation snapshots. See [the current release overview](../README.md#verified-release), [the reproduction guide](../reproduction/README.md), and [the service evidence](SERVICE.md) for later source/artifact publication and container checks. Statements about missing code or deployment below describe their original dates, not current release availability.

## Private GPU HTTP smoke (October 7, 2026)

- RTX PRO 5000 in an authorized Slurm allocation; Python 3.12.3,
  PyTorch 2.9.1+cu128, FastAPI 0.142.2, Uvicorn 0.54.0, HTTPX 0.28.1.
- FAPrompt ViT-L/14@336px, input 518, FP32, MVTec-source engineering artifact
  SHA-256 `968db05a3cac035ad0e647a23d8468460a915ad87712da0d4bc8028ae588415d`.
  Alpha remains 1; this is **not** the paper Table 18 alpha-0.5 artifact.
- A real Uvicorn server on loopback answered three sequential HTTP predictions
  for the same MPDD bracket-black scratches image. Host and C-TED raw maps
  exactly matched the prior GPU CLI maps in all three requests (max error 0).
- Corrupt image input returned HTTP 400. The server was stopped after testing;
  there is no continuously running endpoint from this smoke test.
- The combined suite passed 20 tests with one opt-in GPU test skipped. A separate
  T-TED suite with `TED_TEST_CUDA=1` passed all eight tests, including CUDA parity.
  API contract tests cover startup reuse, shutdown, image limits, malformed
  inputs, busy rejection, health responsiveness, and sanitized error responses.
- GPU CLI model load was 5,246 ms; HTTP-process load was 5,083 ms. Three observed
  HTTP round trips were 341, 256, and 256 ms, including raw-map/preview encoding.
  These are **smoke timings**, not warmed p50/p95/throughput benchmark claims.

The original host checkout remains required. Docker build, standalone install,
AWS, full-dataset reproduction, and mixed-precision quality remain unverified.

## Experimental inference bridge (October 7, 2026)

- Five inference test methods cover cached-vs-uncached branch readout parity at
  three chunk sizes, artifact serialization, zero-strength host preservation,
  invalid inputs, and rejection of missing source branch scores.
- The combined suite passes 16 tests; one opt-in CUDA test is skipped.
- Offline export was exercised on the existing 4,096-entry-per-role MVTec source
  bank, followed by CPU inference on one MPDD bracket-black scratches image with
  the original ViT-L/14@336px FAPrompt checkpoint at input size 518.
- The exported engineering artifact uses alpha 1, rank 4, 30 epochs, and at most
  2,048 source points. It is not identified with a particular paper-table cell.
- Raw map outputs and paired previews were generated locally. Dataset images,
  checkpoints, and calibrated artifacts are not redistributed with this commit.

The bridge still imports the original research evaluator and local modified
FAPrompt dependencies. The later private HTTP test above does not establish a
standalone container, public API, AWS deployment, mixed-precision quality
comparison, or load benchmark.

## C-TED

Validation on September 29, 2026, using Python 3.10.19, PyTorch 2.9.1+cu128,
NumPy, and CPU execution:

- Four C-TED test methods passed: extracted function integrity; training,
  inference, bounds, and determinism; zero-gain host preservation; AdaCLIP logit
  center preservation.
- Training/inference checks cover AA-CLIP, AdaCLIP, and FAPrompt at three seeds.
- Nine comparisons against AST-loaded original research functions produced
  exactly matching calibration dictionaries and inference outputs on synthetic
  inputs (zero absolute/relative tolerance).
- The three-host synthetic example passed. FAPrompt tests one explicit branch,
  not complete two-branch fusion.
- The combined suite passed 11 tests, with one opt-in CUDA test skipped.

This is CPU core validation, not benchmark reproduction. Complete host feature
extraction, bank mining, gates/fusion, GPU execution, and dataset metrics have
not been verified in this public package. Function provenance is recorded in
`cted_source_manifest.json`; this does not identify a single canonical recipe
for every paper result.

```bash
python -m examples.cted_synthetic --host all
python -m unittest discover -s tests -v
```

## T-TED

Validation performed on September 29, 2026:

- Python 3.10.19; PyTorch 2.9.1+cu128.
- Seven CPU unit tests passed; one GPU test is opt-in and skipped by default.
- The synthetic 64-patch example passed.
- Ninety CPU comparisons against the original RawCLIP and ImageBind research
  scoring functions passed: five seeds, three temperatures, three Hard-FP weights,
  and two original evaluators.
- Maximum absolute difference in those comparisons: 7.153e-7
  (assertion tolerances: absolute/relative 2e-5).

The original functions were loaded directly from their Python AST definitions
without importing or running their dataset evaluation pipelines. Source features
were synthetic; this verifies mathematical parity, not dataset results.
GPU validation was not reliable in this environment: an initial run printed passing
results, but subsequent GPU runs were terminated. GPU support is therefore not
claimed as validated in this release. The optional test can be run on a suitable
machine with `TED_TEST_CUDA=1 python -m unittest discover -s tests -v`.

Run the public checks:

```bash
python -m unittest discover -s tests -v
python -m examples.tted_synthetic
```

The original full research evaluators are not included in this staged release.
End-to-end feature extraction, source-bank construction, layer aggregation,
post-processing, and benchmark metrics remain unverified as a public package.
