# Main-text aggregate tables

Run from the repository root, using Python's standard library:

```sh
python -m reproduction.main_text_aggregates
```

The audit verifies the preserved CSV bytes and original aggregation-script
hashes, checks group/setting coverage, and recomputes all 27 printed cells in
Table 2 (failure-conditioned gains) and Table 4(b) (scalar source readouts).
The inputs contain nine host-transfer bins and sixteen distinct settings.

The original CSVs were recovered from the research workspace without changing
their values. Their paths, sizes and SHA-256 hashes are recorded in
[manifest.json](manifest.json). The original scripts are preserved in
`reproduction/source.zip`.

This command verifies archival table assembly. Fresh inference and per-image
provenance are separate checks. The result explicitly records
`fresh_gpu_execution_verified=false`.

To save a new report without overwriting previous evidence:

```sh
python -m reproduction.main_text_aggregates --output main-text-aggregate-audit.json
```

The checked report is [available here](../../validation/a10-20261009/main-text-aggregate-archival-v1.json).
