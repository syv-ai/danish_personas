# Changelog

## Unreleased

### Fixed

- UTF-8 source verification and deterministic text output now behave consistently on
  Windows and Unix.

### Added

- Reproducible acquisition and immutable snapshots for five Statistics Denmark tables
  and the official `NUTS_V1_2007_DK` geography classification, which is published as a
  `dst.dk` attachment rather than through the StatBank data API and so uses its own
  source adapter.
- Prepared demographic distributions with age and education pooling for sparse cells,
  plus a `geography_hierarchy` table holding the official region, landsdel, and
  municipality hierarchy.
- A compressed, attributed Statistics Denmark source archive, an offline restoration
  command for deterministic demographic regeneration, and a byte-stable packing command
  that reproduces identical archive bytes from unchanged snapshots.
- Deterministic demographic and independent OCEAN generation without LLM calls.
- Source, structural, statistical, held-out, and personality validation gates.
- Guarded, resumable two-stage persona generation through an OpenAI-compatible API.
- Provider-qualified Hugging Face routing, optional model thinking/reasoning control,
  and resumable five-row pilot sharding with validated merging.
- Strict generated-attribute and six-description schemas, Danish and safety checks,
  provenance manifests, token accounting, mocked HTTP tests, and a five-row hard limit.
- Source, privacy, acceptance, Phase-2 validation, Phase-3 smoke, Hugging Face Gemma
  cost experiment, blinded four-model comparison, and DeepSeek finalist comparison
  documentation, plus the incomplete DeepSeek V4.1 pilot status.

### Changed

- The municipality-to-region map now comes from the official classification rather than
  from positional inference over FOLK1A's StatBank metadata. Preparation cross-checks the
  two and fails the bundle on any disagreement, missing municipality, or null value; on
  the committed snapshots they agree exactly.
- `config/sources.yaml` and `config/sources.lock.yaml` carry a top-level
  `classifications:` list and are `version: 2`. A lock predating that schema is warned
  about and rewritten.
- RAS209's H10-H90 education codes are no longer documented as mapped to DISCED-15. They
  are a StatBank presentation grouping of HFUDD, DISCED-15 does not contain them, and
  Statistics Denmark publishes no crosswalk. `config/categories.yaml` now records the
  official Danish and English labels verbatim, with any ISCED level flagged as this
  repository's own editorial assertion.
- Adding the classification changed the prepared bundle identifier, so the Phase 2
  validation report describes a superseded bundle and run. Regeneration is deferred to a
  separate branch; the report's measured numbers stand as a historical record.
