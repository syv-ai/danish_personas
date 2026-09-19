# Non-LLM acceptance criteria

These gates apply before any LLM integration may be enabled.

## Source bundle

- Every non-BULK query is explicit and remains below StatBank's one-million-cell limit,
  calculated as selected observations multiplied by returned columns (all selected
  dimensions, including time, plus the observation value). BULK is the explicitly
  documented streaming-format exemption.
- Raw metadata, query, CSV, headers, and SHA-256 manifest exist for every source.
- Existing raw snapshots are never overwritten.
- Every selected table has a positive population total and zero unhandled suppressed
  cells.
- LONS20 is fixed to 2024 `ANTAL`, all sectors, all forms of pay, the employee-group
  total, M/K, and exactly the 42 two-digit DISCO-08 groups. Its separately reviewed,
  versioned `config/lons20-contract.yaml` freezes the title, every required dimension
  label, fixed selector code-label pairs, and every selected ARBF code-label pair. Lock
  metadata and raw snapshot metadata must independently match that contract; coordinated
  drift, missing or extra codes, and contract checksum changes fail or produce a
  different bundle identity. Its required prepared marginal preserves official code,
  label, sex, and positive count. Wrong hierarchy levels, totals, duplicates, blank or
  changed labels, malformed or nonpositive counts, suppression, and missing sex
  distributions fail.
- FOLK2's prepared national origin marginal has a positive total, unique official IELAND
  codes and labels, an exact selected code-to-label mapping from official metadata, zero
  suppression, no unhandled values, and the expected official raw partition. The
  reviewed expected-zero-code set is bound into the source config and lock; only those
  omitted all-zero codes may be materialised, and unexpected missing selected codes fail
  preparation and source validation.
- FOLK2 retains official categories such as Stateless and Not stated explicitly; it does
  not create continents, regions, or inferred correlations.
- RAS209 selects all official level-3 municipality areas, including Christiansø when
  exposed, and preserves the municipality x education x status x age-band x sex joint.
- Prepared person-sampling artefacts retain municipality keys and never aggregate those
  rows by region. Municipality names and official region parents come only from the
  validated hierarchy lookup; missing, duplicate, or mismatched mappings fail.
- Municipality codes map to one of the five regions, and that mapping agrees with the
  official Statistics Denmark geography classification.
- A shared boundary verifier requires prepared-bundle schema 5, all mandatory Parquet
  schemas, successful source preparation, and every manifest checksum before either
  sampling or demographic validation. Legacy, malformed, and tampered bundles fail.
- The locked RAS209 selection, official hierarchy, and prepared RAS209 joint have
  exactly equal municipality-code sets. Blank hierarchy codes, titles, or parents fail.
- Sparse-cell pooling retains at least 99% of the relevant source universe.

## Generated records

- Exactly the requested number of records exists. The canonical deterministic workflow
  passes first at the configured 2,000-row smoke size and then at 100,000 rows, without
  changing thresholds or choosing a different seed.
- Every identifier is unique and deterministic.
- Every record passes the strict Phase-2 Pydantic schema.
- No generated person is younger than 18.
- Detailed labour status remains within its sampled broad status, including when a draw
  backs off to a coarser cell.
- Every record records the back-off level that produced its age, marital status, and
  detailed status, and each of those ladders independently keeps at most 1% of records
  on a coarser cell. Age and marital back-off can relax age or sex only while retaining
  the same municipality; no region or national fallback exists. RAS202's national
  detailed-status refinement remains a separate ladder. The validation configuration is
  schema version 5.
- A combination no ladder can serve is a hard failure, not a reported rate: generation
  aborts rather than emitting a record from an unsupported cell.
- The RAS209 `67+` education proxy is labelled for every person aged 70+ and nobody
  younger than 70.
- Every fitted marginal cell with expected count of at least five lies within the larger
  of 0.5 percentage points or three binomial standard errors.
- Municipality replaces region in fitted and held-out geography gates. Region has only
  an exact official-parent consistency gate.
- Total variation is at most 5% for smoke marginals and 2% for 100,000-row fitted
  marginals. The high-dimensional municipality RAS209 joint is informational at 2,000
  rows and capped at 10% at 100,000 rows.
- Municipality held-out population total variation is at most 25% for the 2,000-row
  smoke and 5% at 100,000 rows.
- OCEAN scores lie in `[20, 80]`.
- Maximum absolute pairwise OCEAN correlation is at most 0.02 at 100,000 rows.
- The run manifest records exactly zero LLM calls and the sampler schema version.
- Every Phase 2 record has the official FOLK2 `origin_country_code` and `origin_country`
  values, sampled independently from the national marginal with deterministic
  largest-remainder quotas and a separate RNG child. Equal largest remainders are
  resolved by sorted official code regardless of input order. Unequal official weights
  are retained; zero-weight categories are never emitted; malformed code-label-count
  distributions fail loudly. Every observed code-label pair is checked, including an
  alternate label beside valid rows; unexpected pairs and all unexpected fitted
  categories fail explicitly and remain included in distribution accounting. The mapping
  and origin marginal meet the same statistical gates as other mandatory marginals.
- Eligible RAS202 employee status codes 15, 20, 25, 30, 35, and 40 receive paired
  `job_function_code` and `job_function` values with `lons20_sex_marginal` resolution.
  Every other detailed status receives paired nulls and `not_applicable`. Within each
  sex, deterministic largest-remainder quotas use sorted-code ties and a fourth isolated
  RNG stream. Changing LONS20 weights cannot perturb any existing field or RNG stream.
  Exact code-label, eligibility, and sex-conditional distribution gates are mandatory.
- Job function is a synthetic allocation calibrated to LONS20's incomplete
  earnings-statistics universe, not observed occupation or all-worker representation. It
  is never conditioned on municipality, origin, age, education, OCEAN, or an unsupported
  joint.
- `country` remains the residence value `Danmark`; mandatory municipality code and name,
  official region parent, and `education_level` are retained. Origin is not ethnicity,
  citizenship, residence, or appearance and cannot drive language, culture, religion,
  occupation, or personality.
- Origin and job-function fields remain in upstream/generated outputs and
  input/checkpoint hashes. Human-readable municipality, origin, and job-function labels
  may reach the provider for grounding; their source codes and resolution fields do not.
  Job titles are synthetic and must not imply unsupported work history.

`SAMPLER_SCHEMA_VERSION` must be incremented whenever deterministic sampling semantics
or generated record columns change incompatibly. It is part of the content-addressed run
identity, so a legacy run cannot be silently reused.

## Persona smoke runs

- The committed configuration remains disabled and every live invocation requires
  `--live` explicitly.
- The input checksum and successful Phase-2 validation report match the upstream run.
  The upstream sampler schema version must equal the current version, and every frozen
  row and column must validate against the current `DemographicRecord`; legacy
  origin-less samples require migration and cannot cross the Phase-3 boundary.
- No invocation can request more than five rows. The deliberately stratified 1,000-row
  text-development input is separate from the 2,000-row Phase-2 smoke run.
- Generated attributes and all six v2 persona descriptions satisfy strict schemas: five
  specialised texts plus one short, grounded `persona`. The persona contains age,
  statistical sex, municipality, education, origin, an official-job-function-grounded
  synthetic job title or current nonemployee status, two or three interests in prose,
  and cautious OCEAN tendencies. `visual_persona` is not a v2 field.
- The provider receives human-readable municipality, origin, and job-function labels,
  plus the reviewed allowlist of Danish titles for that label, but never source codes or
  resolution fields. A generated title must equal an allowlist entry exactly.
- The versioned 42-code title mapping is checksum-bound into generation context,
  checkpoints, shards, pilots, and the offline release package.
- Upstream demographic and OCEAN columns remain byte-for-byte equivalent in logical
  values and order.
- Generated text is Danish, contains no detected contact details or identifying-number
  patterns, and does not contain inflectional sensitive terms. It must make no
  unsupported family claims or physical-appearance claims. Downstream image models may
  stereotype, so this contract does not make image generation safe.
- Exact duplicate persona descriptions are rejected. The specialised texts remain
  separate domains, and all six texts must be distinct.
- Current v2 per-record checkpoints support resume without repeating completed model
  calls. Historical v1 outputs and old pilots are not resumable under v2.
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
  review attestation. Any population of at least 100,000 requires at least 500. Other
  positive population sizes fail closed; policy minima may only be stricter.
- The attestation binds the pilot ID, output checksum, population, protocol, timestamp,
  and reviewed IDs. Eligibility requires exactly one unique output ID per attested row
  and rejects reviewed IDs not present in the output.
- Policy, attestation, and approval-result contracts are frozen and use immutable tuple
  collections. Security-relevant values are strict and are never silently coerced or
  stripped.
- A release must identify generation contract v2 and retain five specialised texts plus
  one short grounded persona. It must record that provider payloads contain approved
  human-readable municipality, origin, and job-function labels only, not source codes or
  resolutions, and must disclose synthetic job titles and image-model stereotyping risk.
- These checks are pure in-memory contracts. Packaging, publication, uploads, and
  release manifests are not implemented by this release.

A failed mandatory gate returns a non-zero command exit code. Thresholds may not be
changed retrospectively to make a completed statistical run pass.
