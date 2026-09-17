# Changelog

## Unreleased

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
- Sparse-cell back-off for the demographic sampler. Each drawn field walks an
  ordered ladder and stops at the most specific populated cell, recording the level
  that produced it in `age_resolution`, `marital_resolution`, and
  `detailed_status_resolution`. A ladder ends at the most general cell that is still
  structurally valid rather than a national one, so an age cannot leave its band and
  a detailed status cannot leave its broad RAS209 status; a cell missing at the final
  level is a structural zero and still fails loudly. Validation reports the back-off
  rate against a configured ceiling, and the generation stage withholds the levels
  from the prompts.
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
- `config/sampling.yaml`'s `smoothing` setting is now applied. It reweights the cells
  that survived the structural filter, so it cannot resurrect an absent category, and
  remains a no-op at its configured `0.0`.
- Adding the classification changed the prepared bundle identifier, so the Phase 2
  validation report describes a superseded bundle and run. Regeneration is deferred to a
  separate branch; the report's measured numbers stand as a historical record.
