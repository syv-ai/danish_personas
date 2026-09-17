# Phase 2 validation report

Run regenerated offline without any LLM calls.

The deterministic sampler schema is version 3 and includes the independent FOLK2
origin-country marginal. This report was regenerated offline without LLM calls.

## Artefacts

- Source bundle: `fda86665792f7734`
- Smoke run: `ec9ada7369d882eb` (1,000 rows; generated, but failed the existing
  small-sample fitted-joint and one labour-status cell gate)
- Statistical run: `f65747f0e1fb35d5` (100,000 rows)
- Statistical Parquet SHA-256:
  `b12d9e719b968afe1dd27e43dfe68fdd242791c7bb713b8e8e676901b1f130b4`
- Logical-content SHA-256:
  `e19122c398e59dcf235cfe45c5586232f637451b68260002bfd16e0fe2425301`
- Frozen text-development input: 1,000 stratified rows (ignored local artefact)

The compressed raw snapshots are committed with attribution. Restored and generated
datasets remain ignored by Git and are reproducible from the source archive, lock,
mappings, configurations, and code.

## Source result

- Six official tables fetched and checksummed.
- Adjusted RAS209 adult total: 4,845,759.
- RAS202 adult total: 4,845,960.
- Sparse-cell sampling table: 531 cells.
- Retained RAS209 coverage: 99.6439%.
- Source validation: **PASS**.

## Statistical run result

| Metric | Result | Threshold |
| --- | ---: | ---: |
| Rows | 100,000 | 100,000 |
| Schema errors | 0 | 0 |
| Status mapping errors | 0 | 0 |
| Education proxy errors | 0 | 0 |
| Maximum fitted marginal TV | 0.1739% | 2% |
| FOLK2 origin marginal TV | 0.0256% | 2% |
| Full 531-cell fitted joint TV | 0.0676% | 2% |
| Held-out population joint TV | 0.1094% | 5% |
| Held-out status-by-sex TV | 0.0309% | 5% |
| OCEAN score-bound errors | 0 | 0 |
| Maximum pairwise OCEAN correlation | 0.004863 | 0.02 |
| Origin mapping errors | 0 | 0 |
| Zero-weight origin categories emitted | 0 | 0 |
| LLM calls | 0 | 0 |

Every mandatory cell-wise confidence check also passed. Overall Phase-2 result:
**PASS**.

The machine-readable reports remain beside the local source bundle and generated run.
