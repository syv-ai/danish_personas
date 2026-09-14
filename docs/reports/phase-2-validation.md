# Phase 2 validation report

Run completed on 14 September 2026 without any LLM calls.

## Artefacts

- Source bundle: `e7757f736ef5652f`
- Smoke run: `78f2ec8279025a5f` (1,000 rows)
- Statistical run: `dd191c5db9e27630` (100,000 rows)
- Statistical Parquet SHA-256:
  `447f62c6f5ec95e1496b51909529586e62f151b3b60ccd130acddc4b5410ed81`
- Logical-content SHA-256:
  `172fa742917b06ded4d9532345bacc6c4626b6e95bf74d8936eb64af12a5fcc7`
- Frozen text-development input: 1,000 stratified rows

Generated datasets and raw snapshots are ignored by Git but remain reproducible from the
committed source lock, mappings, configurations, and code.

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
| Maximum fitted marginal TV | 0.2352% | 2% |
| Held-out population joint TV | 0.8868% | 5% |
| Held-out status-by-sex TV | 0.0835% | 5% |
| OCEAN score-bound errors | 0 | 0 |
| Maximum pairwise OCEAN correlation | 0.004863 | 0.02 |
| LLM calls | 0 | 0 |

Every mandatory cell-wise confidence check also passed. Overall Phase-2 result:
**PASS**.

The machine-readable reports remain beside the local source bundle and generated run.
