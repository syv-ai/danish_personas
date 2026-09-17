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

- Exactly the requested number of records exists. The canonical deterministic workflow
  passes first at the configured 2,000-row smoke size and then at 100,000 rows, without
  changing thresholds or choosing a different seed.
- Every identifier is unique and deterministic.
- Every record passes the strict Phase-2 Pydantic schema.
- No generated person is younger than 18.
- Detailed labour status remains within its sampled broad status, including when a
  draw backs off to a coarser cell.
- Every record records the back-off level that produced its age, marital status,
  and detailed status, and each of those ladders independently keeps at most 1% of
  records on a coarser cell. The validation configuration is schema version 3 and
  explicitly requires `maximum_backoff_rate` and the origin marginal gate.
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
- Every Phase 2 record has the official FOLK2 `origin_country_code` and
  `origin_country` values, sampled independently from the national marginal with
  deterministic largest-remainder quotas and a separate RNG child. Equal largest
  remainders are resolved by sorted official code regardless of input order. Unequal
  official weights are retained; zero-weight categories are never emitted; malformed
  code-label-count distributions fail loudly. Every observed code-label pair is checked,
  including an alternate label beside valid rows; unexpected pairs and all unexpected
  fitted categories fail explicitly and remain included in distribution accounting.
  The mapping and origin marginal meet the same statistical gates as other mandatory
  marginals.
- `country` remains the residence value `Danmark`, and `education_level` is retained.
  Origin is not ethnicity, citizenship, or residence and cannot drive language,
  culture, religion, occupation, personality, or visual appearance.
- Both origin fields remain in upstream/generated outputs and input/checkpoint hashes,
  but are withheld from both LLM payloads; all resolution fields are also withheld.

`SAMPLER_SCHEMA_VERSION` must be incremented whenever deterministic sampling
semantics or generated record columns change incompatibly. It is part of the
content-addressed run identity, so a legacy run cannot be silently reused.

## Persona smoke runs

- The committed configuration remains disabled and every live invocation requires
  `--live` explicitly.
- The input checksum and successful Phase-2 validation report match the upstream run.
  The upstream sampler schema version must equal the current version, and every frozen
  row and column must validate against the current `DemographicRecord`; legacy
  origin-less samples require migration and cannot cross the Phase-3 boundary.
- No invocation can request more than five rows. The deliberately stratified 1,000-row
  text-development input is separate from the 2,000-row Phase-2 smoke run.
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

## Release eligibility contract

- The committed release policy is disabled by default. An enabled policy must name an
  exact model allowlist, a nonblank SPDX-like dataset licence identifier, and the exact
  64-hex SHA-256 of its licence file.
- Exactly 10,000 rows requires at least 300 unique IDs in an explicit, blinded human
  review attestation. Any population of at least 100,000 requires at least 500.
  Other positive population sizes fail closed; policy minima may only be stricter.
- The attestation binds the pilot ID, output checksum, population, protocol, timestamp,
  and reviewed IDs. Eligibility requires exactly one unique output ID per attested row
  and rejects reviewed IDs not present in the output.
- Policy, attestation, and approval-result contracts are frozen and use immutable tuple
  collections. Security-relevant values are strict and are never silently coerced or
  stripped.
- These checks are pure in-memory contracts. Packaging, publication, uploads, and
  release manifests are not implemented by this release.

A failed mandatory gate returns a non-zero command exit code. Thresholds may not be
changed retrospectively to make a completed statistical run pass.
