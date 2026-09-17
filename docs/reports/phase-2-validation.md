# Phase 2 validation report

The canonical deterministic sampler schema is version 3 and includes the independent
FOLK2 origin-country marginal. Both runs below were regenerated offline with seed
`20260914`, without LLM calls and without changing any validation threshold.

## Artefacts

- Source bundle: `fda86665792f7734`
- Sampling-config SHA-256:
  `65619c66c67bfac30fe3c97f94fc590de1022612f95d2354d2fd52e2c1d45868`
- Smoke run: `0122b894dec6829e` (2,000 rows; **PASS**)
- Smoke Parquet SHA-256:
  `d841e37ab0077905badad1aae203eccb6dd709e42dd8dd12b472dad24480dfa2`
- Smoke logical-content SHA-256:
  `dd5f0df38a7f6d9ebbf02fe9c35cf9e3998be8f7dc2d5e733ef00ddcac0ae2ce`
- Statistical run: `ea321089a79d3650` (100,000 rows; **PASS**)
- Statistical Parquet SHA-256:
  `b12d9e719b968afe1dd27e43dfe68fdd242791c7bb713b8e8e676901b1f130b4`
- Statistical logical-content SHA-256:
  `e19122c398e59dcf235cfe45c5586232f637451b68260002bfd16e0fe2425301`
- Frozen text-development input: 1,000 stratified rows (ignored local artefact).
  This deliberate Phase-3 development size is separate from the 2,000-row Phase-2
  validation smoke size.

The compressed raw snapshots are committed with attribution. Restored and generated
datasets remain ignored by Git and are reproducible from the source archive, lock,
mappings, configurations, and code.

## Source result

- Six official tables and one official classification snapshot were checksummed.
- Adjusted RAS209 adult total: 4,845,759.
- RAS202 adult total: 4,845,960.
- Sparse-cell sampling table: 531 cells.
- Retained RAS209 coverage: 99.6439%.
- Source validation: **PASS**.

## Deterministic run results

| Metric | 2,000-row smoke | 100,000-row statistical | Threshold |
| --- | ---: | ---: | ---: |
| Rows | 2,000 | 100,000 | exact |
| Schema errors | 0 | 0 | 0 |
| Status mapping errors | 0 | 0 | 0 |
| Education proxy errors | 0 | 0 | 0 |
| Maximum fitted marginal TV | 1.4587% | 0.1739% | 5% / 2% |
| FOLK2 origin marginal TV | 0.8430% | 0.0256% | 5% / 2% |
| Full 531-cell fitted joint TV | 3.5609% | 0.0676% | 5% / 2% |
| Held-out population joint TV | 0.8651% | 0.1094% | 10% / 5% |
| Held-out status-by-sex TV | 0.4283% | 0.0309% | 10% / 5% |
| OCEAN score-bound errors | 0 | 0 | 0 |
| Maximum pairwise OCEAN correlation | 0.040965 | 0.004863 | info / 0.02 |
| Origin mapping errors | 0 | 0 | 0 |
| Unexpected fitted categories | 0 | 0 | 0 |
| Zero-weight origin categories emitted | 0 | 0 | 0 |
| LLM calls | 0 | 0 | 0 |

Every mandatory cell-wise confidence check passed in both runs. Overall Phase-2 result:
**PASS**.

The machine-readable reports remain beside the local source bundle and generated runs.
