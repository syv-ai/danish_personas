# Phase 2 validation report

Run completed on 14 September 2026 without any LLM calls.

## Artefacts

- Source bundle: `e7757f736ef5652f`
- Smoke run: `78f2ec8279025a5f` (1,000 rows)
- Statistical run: `f5f37949670df476` (100,000 rows)
- Statistical Parquet SHA-256:
  `5d182c8fa4d35090ff284e095ed7da38ddf04fe996cfda6b6aa249b00c8dd38c`
- Logical-content SHA-256:
  `fa2c6e5d3541298c4723200cbd51bda3647960da087cc86bbf4fb4f99d74b1c5`
- Frozen text-development input: 1,000 stratified rows

The compressed raw snapshots are committed with attribution. Restored and generated
datasets remain ignored by Git and are reproducible from the source archive, lock,
mappings, configurations, and code.

## Source result

- Five official tables fetched and checksummed.
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
| Full 531-cell fitted joint TV | 0.0676% | 2% |
| Held-out population joint TV | 0.1094% | 5% |
| Held-out status-by-sex TV | 0.0309% | 5% |
| OCEAN score-bound errors | 0 | 0 |
| Maximum pairwise OCEAN correlation | 0.004863 | 0.02 |
| LLM calls | 0 | 0 |

Every mandatory cell-wise confidence check also passed. Overall Phase-2 result:
**PASS**.

The machine-readable reports remain beside the local source bundle and generated run.
