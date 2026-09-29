# T-TED core verification

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
