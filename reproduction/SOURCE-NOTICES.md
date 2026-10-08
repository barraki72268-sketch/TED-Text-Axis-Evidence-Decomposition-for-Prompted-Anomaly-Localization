# Research source bundle

`source.zip` contains 920 hash-pinned research files. `source-manifest.json`
records each path, byte size, and SHA-256. This includes model implementations,
evaluation scripts, tokenizer vocabularies, and the historical configuration
files preserved with those implementations. It contains no model checkpoints
or dataset images. The archive format preserves historical names without
making a normal Windows Git checkout exceed its default path-length limit.

The bundle is a collection of separately licensed components. The repository's
top-level license does not replace the included upstream licenses or notices.
After extraction, consult these original files:

| Component | Included license or notice | Terms identified in the source |
| --- | --- | --- |
| AA-CLIP | `neurips2026/AA-CLIP/LICENSE` | Apache License 2.0 |
| FAPrompt | `neurips2026/FAPrompt/LICENSE` | MIT |
| AdaCLIP | `neurips2026/AdaCLIP/LICENSE` | MIT |
| AdaptCLIP | `neurips2026/AdaptCLIP/LICENSE` | GNU GPL version 2 |
| Bayes-PFL | `neurips2026/Bayes-PFL/README.md`, License section | Upstream declares MIT; the captured tree has no separate license file |
| AnomalyCLIP | `neurips2026/AnomalyCLIP/LICENSE` | MIT |
| WinClip | `neurips2026/WinClip/License` | MIT |
| ImageBind | `ADPretrain/models/ImageBind/LICENSE` | CC BY-NC-SA 4.0 |

File-level notices and dependencies' own licenses also remain in effect.
Checkpoints and datasets require their own provenance and license records;
these source notices do not grant rights to omitted artifacts.

The evaluators named `probe_bayespfl_ted_smoke_20260430.py`,
`probe_bayespfl_ted_smoke_20260501.py`, and
`official_parallel_test_aaclip_20260428.py` are preserved historical versions
reconstructed from archived source edits. The current AA-CLIP evaluator is
also retained. AA-CLIP H/14's remaining metric discrepancy persists with the
April 28 version; its presence in the bundle is not a claim that all historical
scores have been reproduced.

The extracted code still contains historical filesystem defaults. Extraction
verifies source integrity; it does not install dependencies, acquire datasets,
rewrite dataset metadata, or execute a benchmark. Portable path preparation and
full clean-environment validation remain release requirements.
