# Non-LLM acceptance criteria

These gates apply before any LLM integration may be enabled.

## Source bundle

- Every non-BULK query is explicit and remains below StatBank's one-million-cell
  limit, calculated as selected observations multiplied by returned columns (all
  selected dimensions, including time, plus the observation value). BULK is the
  explicitly documented streaming-format exemption.
- Raw metadata, query, CSV, headers, and SHA-256 manifest exist for every source.
- Existing raw snapshots are never overwritten.
- Every selected table has a positive population total and zero unhandled suppressed
  cells.
- FOLK2's prepared national origin marginal has a positive total, unique official
  IELAND codes and labels, an exact selected code-to-label mapping from official
  metadata, zero suppression, no unhandled values, and the expected official raw
  partition. The reviewed expected-zero-code set is bound into the source config and
  lock; only those omitted all-zero codes may be materialised, and unexpected missing
  selected codes fail preparation and source validation.
- FOLK2 retains official categories such as Stateless and Not stated explicitly; it
  does not create continents, regions, or inferred correlations.
- Municipality codes map to one of the five regions, and that mapping agrees with
  the official Statistics Denmark geography classification.
- Prepared-file checksums match the bundle manifest.
- Sparse-cell pooling retains at least 99% of the relevant source universe.

## Generated records

- Exactly the requested number of records exists.
- Every identifier is unique and deterministic.
- Every record passes the strict Phase-2 Pydantic schema.
- No generated person is younger than 18.
- Detailed labour status remains within its sampled broad status, including when a
  draw backs off to a coarser cell.
- Every record records the back-off level that produced its age, marital status,
  and detailed status, and each of those ladders independently keeps at most 1% of
  records on a coarser cell. The validation configuration is schema version 2 and
  explicitly requires `maximum_backoff_rate`.
- A combination no ladder can serve is a hard failure, not a reported rate: generation
  aborts rather than emitting a record from an unsupported cell.
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
- The run manifest records exactly zero LLM calls and the sampler schema version.
- The FOLK2 marginal is not emitted in Phase 2 records and is not sent to LLMs;
  adding origin sampling requires a later sampling and privacy review.

`SAMPLER_SCHEMA_VERSION` must be incremented whenever deterministic sampling
semantics or generated record columns change incompatibly. It is part of the
content-addressed run identity, so a legacy run cannot be silently reused.

## Persona smoke runs

- The committed configuration remains disabled and every live invocation requires
  `--live` explicitly.
- The input checksum and successful Phase-2 validation report match the upstream run.
- No invocation can request more than five rows.
- Generated attributes and all seven persona descriptions satisfy strict schemas,
  including required 2-4 sentence `visual_persona` guidance.
- Upstream demographic and OCEAN columns remain byte-for-byte equivalent in logical
  values and order.
- Generated text is Danish, contains no detected contact details or identifying-number
  patterns, and does not contain the configured sensitive terms. Visual guidance uses
  only the closed Danish grammar and vocabulary in
  [the visual-persona format](visual-persona-format.md): mutable clothing,
  accessories, colours, and a generic background. No other visual claims are
  accepted.
- Exact duplicate persona descriptions, including visual guidance, are rejected. The
  visual vocabulary provides many clothing, accessory, colour, background, and
  lighting combinations, so diversity does not depend on accepting free-form prose.
- Per-record checkpoints support resume without repeating completed model calls.
- The manifest records model, selected inference provider, endpoint, prompt and input
  hashes, HTTP attempts, retries, token use, provider-estimated cost when available, and
  output checksum.
- Automated validation is necessary but not sufficient: a blinded human review remains
  mandatory before any development-sample or release-scale generation.

A failed mandatory gate returns a non-zero command exit code. Thresholds may not be
changed retrospectively to make a completed statistical run pass.
