# Non-LLM acceptance criteria

These gates apply before any LLM integration may be enabled.

## Source bundle

- Every query is explicit and remains below StatBank's one-million-cell limit.
- Raw metadata, query, CSV, headers, and SHA-256 manifest exist for every source.
- Existing raw snapshots are never overwritten.
- Every selected table has a positive population total and zero unhandled suppressed
  cells.
- Municipality codes map to one of the five regions.
- Prepared-file checksums match the bundle manifest.
- Sparse-cell pooling retains at least 99% of the relevant source universe.

## Generated records

- Exactly the requested number of records exists.
- Every identifier is unique and deterministic.
- Every record passes the strict Phase-2 Pydantic schema.
- No generated person is younger than 18.
- Detailed labour status remains within its sampled broad status.
- The RAS209 `67+` education proxy is labelled for every person aged 70+ and nobody
  younger than 70.
- Every fitted marginal cell with expected count of at least five lies within the larger
  of 0.5 percentage points or three binomial standard errors.
- Total variation is at most 5% for smoke marginals and 2% for 100,000-row fitted
  marginals.
- Smoke holdout total variation is at most 10%; 100,000-row holdout total variation is
  at most 5%.
- OCEAN scores lie in `[20, 80]`.
- Maximum absolute pairwise OCEAN correlation is at most 0.02 at 100,000 rows.
- The run manifest records exactly zero LLM calls.

A failed mandatory gate returns a non-zero command exit code. Thresholds may not be
changed retrospectively to make a completed statistical run pass.
