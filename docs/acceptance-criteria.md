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
- FOLK2 retains all 241 official rows, including categories such as Stateless and Not
  stated explicitly; each row has a non-null Boolean `eligible_for_sampling` equal
  exactly to `count >= minimum_source_count` (currently 50). Preparation reports the
  eligible and excluded row metrics, while later sampling and dashboard filtering are
  separate steps. It does not create continents, regions, or inferred correlations.
- RAS209 selects all official level-3 municipality areas, including Christiansø when
  exposed, and preserves the municipality x education x status x age-band x sex joint.
- Prepared person-sampling artefacts retain municipality keys and never aggregate those
  rows by region. Municipality names and official region parents come only from the
  validated hierarchy lookup; missing, duplicate, or mismatched mappings fail.
- Municipality codes map to one of the five regions, and that mapping agrees with the
  official Statistics Denmark geography classification.
- A shared boundary verifier requires prepared-bundle schema 8, all mandatory Parquet
  schemas, successful source preparation, and every manifest checksum before either
  sampling or demographic validation. Legacy, malformed, and tampered bundles fail.
- The locked RAS209 selection, official hierarchy, and prepared RAS209 joint have
  exactly equal municipality-code sets. Blank hierarchy codes, titles, or parents fail.
- Sparse-cell pooling excludes RAS209 H90/`not_stated` from the eligible pooled
  sampling universe while retaining it in the unpooled audit joint, and retains at
  least 99% of the remaining eligible source universe.

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
  detailed-status refinement remains a separate ladder. The sampler schema is version
  8, the frozen-sample schema is version 5, and the validation configuration remains
  version 5.
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
- Every Phase 2 record has the official FOLK2 `origin_country_code`, English
  `origin_country`, and mandatory Danish `origin_country_da` values, sampled
  independently from the national marginal with deterministic largest-remainder quotas
  and a separate RNG child. The English label is retained as official source/audit
  provenance. The Danish label is resolved from the archived official FOLK2 metadata-da
  241-code contract (`config/folk2-ieland-labels-da.yaml`) with source metadata SHA-256
  `f5c1f0a20f29372d6b222ce7a23cdc4ef0481d9e23fa6bd9b66b116e7adcb213`. Exact code-label
  pairs, including Danish display labels, are checked. Equal largest remainders are
  resolved by sorted official code regardless of input order. Unequal official weights
  are retained; only rows marked `eligible_for_sampling` are sampled (with the target
  renormalised over those rows); malformed code-label-count-eligibility distributions
  fail loudly. Every observed code-label pair is checked against the full official
  mapping, including an alternate label beside valid rows; unexpected pairs,
  ineligible codes, and all unexpected fitted categories fail explicitly and remain
  included in distribution accounting. The eligibility threshold is the bundle-bound
  `minimum_source_count`; the mapping and eligible origin marginal meet the same
  statistical gates as other mandatory marginals.
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
  official region parent, and `education_level` are retained. Neither origin label is
  ethnicity, citizenship, residence, or appearance. Origin cannot drive culture,
  religion, job, interests, personality, or visual traits.
- Origin and job-function fields remain in upstream/generated outputs and
  input/checkpoint hashes. Human-readable municipality, Danish `origin_country_da`,
  sampled legal `marital_status`, and job-function labels may reach the provider request
  for grounding; origin code, English `origin_country`, contract metadata, and all
  resolution fields do not. The exact Danish label is the origin fact in the grounded
  persona. Job titles are synthetic and must not imply unsupported work history.

`SAMPLER_SCHEMA_VERSION` is 8 and must be incremented whenever deterministic
sampling semantics or generated record columns change incompatibly. The frozen-sample
schema is 5. Both freeze modes first create the global population-proportional or
stratified round-robin STRATA selection, then impose the observed eligible origin
marginal with sorted-code largest-remainder quotas. Deterministic same-STRATA swaps
preserve the original global STRATA counts whenever the margins are feasible. If they
are not, the documented deterministic maximum-overlap fallback enforces the origin
quotas while minimising the number of changed rows from the global selection. These
versions are part of content-addressed identities, so legacy bundles, runs, or samples
cannot be silently reused. Previous canonical IDs are historical and non-resumable;
regenerate and record new IDs rather than inventing
 them.

## Persona runs
- The Hydra configuration names the endpoint and model explicitly. Each invocation
  remains bounded by row and HTTP-request limits.
- Library and dataset/release validation require the input checksum and successful
  Phase-2 validation report to match the upstream run. The single-row
  `generate_persona.py` command uses an explicit checksum-tolerant policy for stored
  checksum mismatches only; all semantic, schema, provenance-content, and safety gates
  remain active. The upstream sampler schema must be version 8 and the frozen-sample
  schema must be version 5. Every frozen row and column must validate against the current
  `DemographicRecord`; legacy, origin-less samples require migration and cannot cross
  the Phase-3 boundary.
- No invocation can request more than five rows. The default population-proportional
  1,000-row text-development input is separate from the 2,000-row Phase-2 smoke run;
  stratified round-robin remains an explicit alternative mode.
- One provider response contains both generated attributes and the generation-5
  `persona` field. A valid first response costs one request per row; only validation or
  transport retries add requests.
- The persona is natural Danish prose of 300-900 characters and at least four
  sentences. It uses at least two supplied interests, one supplied skill, one
  compatible personality tendency, and any generated ambition.
- The persona preserves age, the supplied `han` or `hun`, municipality,
  `origin_country_da`, broad education, and the synthetic allowlisted title or current
  status without requiring verbatim clauses. Natural paraphrase and benign consistent
  elaboration are allowed. Pronouns must remain consistent. Token-bounded
  `vedkommende`, `personen`, `kan være`, and `ungdoms- eller
  erhvervsuddannelse` are rejected, along with redundant sex nouns and data-model
  jargon.
- The provider receives human-readable municipality, the official Danish
  `origin_country_da`, sampled legal `marital_status`, and job-function labels, plus the
  reviewed allowlist of Danish titles for that label, but never origin code, English
  label, contract metadata, or resolution fields. A generated title must equal an
  allowlist entry exactly.
- The versioned 42-code title mapping is checksum-bound into generation context,
  checkpoints, shards, pilots, and the offline release package.
- Upstream demographic and OCEAN columns remain byte-for-byte equivalent in logical
  values and order.
- Generated text is Danish, contains no detected contact details or identifying-number
  patterns, and does not contain inflectional sensitive terms or physical-appearance
  claims. The persona is written with `han` or `hun`; partners and family members are
  referred to by relationship terms rather than names. Surnames, real employer or
  institution names, and exact addresses are prohibited.
  Downstream image models may stereotype, so this contract does not make
  image generation safe.
- Exact duplicate persona descriptions are rejected across each run and pilot.
- Current generation-5 per-record checkpoints support resume without repeating
  completed model calls. Historical v1-v4 outputs and old pilots are not resumable
  under generation 5.
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
- A release must identify generation contract 5 and validator `persona-safety-v20`,
  and retain only the detailed grounded `persona`. Detailed education is explicitly
  fictional rather than source-backed. The release manifest remains schema 2, while
  release evidence is schema 3 because its prompt provenance contract changed. The
  package must record that the provider request contains approved human-readable
  municipality, Danish origin, sampled legal marital status, and job-function labels
  only, not origin code, English label, contract metadata, or resolution fields, and
  must disclose synthetic job titles and image-model stereotyping risk.
- Release packaging and verification must bind the schema-2 release manifest and
  evidence checksums, provenance, row counts, and human-review evidence. Hugging Face
  upload may include only that verified package and must create a dataset pull request;
  raw pilot files must never be uploaded. Until a current package is regenerated,
  release IDs, checksums, and canonical output IDs are placeholders rather than claims
  about historical artefacts.

A failed mandatory gate returns a non-zero command exit code. Thresholds may not be
changed retrospectively to make a completed statistical run pass.
