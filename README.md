# Danish Personas

A reproducible pipeline for creating statistically grounded Danish synthetic persona
records from public aggregate data. The default workflow is deterministic and does not
call an LLM or use personal microdata. A separate, explicitly guarded workflow can add
Danish attributes and persona text to a small frozen sample.

The design rationale and deferred work are documented in
[`docs/danish-personas-plan.md`](docs/danish-personas-plan.md).

## Status and scope

The source-acquisition, preparation, deterministic sampling, and validation stages are
implemented. The committed configuration keeps LLM generation disabled. The LLM code is
smoke-test infrastructure only: each direct generation invocation is capped at five
rows, while a pilot can span multiple shards. Release-scale generation and human
approval are not implemented release gates.

The municipality-native canonical Phase 2 workflow passes at both 2,000-row smoke and
100,000-row statistical sizes using prepared bundle `cfc1b56f5586a2d7`, prepared-bundle
schema 5, sampler schema 5, and validation schema 5. Read the reports before making
claims:

- [`docs/reports/phase-2-validation.md`](docs/reports/phase-2-validation.md)
- [`docs/reports/phase-3-smoke.md`](docs/reports/phase-3-smoke.md)
- [`docs/privacy-risk-register.md`](docs/privacy-risk-register.md)

## Developer setup guide

This guide takes a developer from a fresh clone to validated demographic data and the
frozen input used for persona generation. No network access or LLM credentials are
required for deterministic data regeneration.

### Prerequisites

- Python 3.14 or later (and below Python 4.0), as required by `pyproject.toml`.
- [`uv`](https://docs.astral.sh/uv/) for Python and dependency management.
- GNU Make and Bash for the convenience targets; `git` for manifests and checks.
- Network access only when fetching Statistics Denmark data or using an LLM endpoint.
- Enough local disk for raw CSV snapshots, Parquet files, and generated reports.

The pipeline uses `polars`, `numpy`, `scipy`, Pydantic, Click, PyYAML, and HTTPX. The
full dependency list and locked versions are in `pyproject.toml` and `uv.lock`.

### Install dependencies

From the repository root, install the locked development environment:

```bash
uv sync --locked --all-extras --dev
```

If `uv` is not installed, follow its installation instructions. The repository's
`make install` can install or update `uv`, create `.env`, initialise Git settings, and
add a GitHub remote; use it only when those side effects are wanted:

```bash
make install
```

Install the pre-commit hook separately when needed. This target also runs
`pre-commit autoupdate`, which may modify the tracked hook configuration; it is not part
of `make install`:

```bash
make install-pre-commit
```

### Configure the local environment

The application does not require environment variables for the non-LLM pipeline. For
local Git metadata or optional LLM credentials, create the ignored environment file:

```bash
cp .env.example .env
```

Set `GIT_NAME` and `GIT_EMAIL` only if using the Makefile's Git setup. `OPENAI_API_KEY`
and `HF_TOKEN` are examples of optional bearer-token variables; a generation config
selects the variable through `api_key_env`. Direct LLM commands do not load `.env`; use
a short-lived shell export or command-scoped assignment for the configured token. The
Makefile includes `.env` and exports all of its variables to subprocesses and hooks, so
do not use it as credential loading for direct LLM commands. Never commit `.env`,
tokens, or generated data artefacts.

### Unified CLI

The installed CLI is the recommended interface for new workflows. It keeps every service
boundary in one process, passes returned artefact paths directly, and stops before the
next validation boundary when a report fails:

```bash
uv run danish-personas workflow deterministic --target smoke
uv run danish-personas workflow deterministic --target statistical \
  --sample-rows 1000 \
  --raw-parent data
```

The default statistical workflow runs the configured smoke and statistical row counts,
then freezes a 1,000-row sample. The sample and its adjacent manifest are written inside
the exact content-addressed statistical run directory so its final stdout path can be
passed directly to offline persona shard planning. Change that development sample size
with `--sample-rows`. `--raw-parent` defaults to `data`; the workflow always restores
and prepares `data/raw-hardened-20260919` (or the same fixed child below a custom
parent). Configuration and run-root paths are also configurable. It is offline and
makes no LLM calls. Source `resolve` and `fetch` need explicit `--network`, while persona
shards are dry runs unless `--live` is supplied and pilots always require `--live`. See
[`docs/cli.md`](docs/cli.md) for the command tree.

The legacy script commands below remain supported and compatible.

### Regenerate development data

The following commands restore the seven exact Statistics Denmark aggregate snapshots
and the official geography classification snapshot, prepare a local source bundle,
generate 2,000 deterministic smoke records, and validate every stage without network
access. The archive and attribution are documented in
[`data/README.md`](data/README.md).

```bash
set -o pipefail

uv run src/scripts/restore_raw_sources.py

BUNDLE=$( \
  uv run src/scripts/build_distributions.py \
    --lock config/sources.lock.yaml \
    --categories config/categories.yaml \
    --raw-dir data/raw-hardened-20260919 \
    --output-dir data/processed \
    2>&1 | tee /dev/stderr | sed -n 's/^INFO Prepared bundle: //p' \
)
test -n "$BUNDLE" || exit 1
uv run src/scripts/validate_dataset.py sources --bundle "$BUNDLE"

RUN=$( \
  uv run src/scripts/generate_demographics.py \
    --bundle "$BUNDLE" \
    --config config/sampling.yaml \
    --rows 2000 \
    --seed 20260914 \
    --output-dir data/runs/smoke \
    2>&1 | tee /dev/stderr | sed -n 's/^INFO Generated run: //p' \
)
test -n "$RUN" || exit 1
uv run src/scripts/validate_dataset.py demographics \
  --run "$RUN" \
  --bundle "$BUNDLE"
```

The assignments capture the exact paths printed by the CLI commands. They do not select
an arbitrary newest directory, and fail if a command emits no path.

All generated run identifiers are derived from input checksums, row count, and seed.
Repeating a valid command reuses the existing run; a checksum mismatch fails instead of
overwriting data.

### Regenerate canonical data and persona seeds

Use the same ordering for a 100,000-row local run. Resolve selectors or fetch from the
StatBank API only when intentionally updating the source lock; refreshed responses form
a new provenance chain rather than reproducing this release.

```bash
set -o pipefail

uv run src/scripts/restore_raw_sources.py

BUNDLE=$( \
  uv run src/scripts/build_distributions.py \
    --lock config/sources.lock.yaml \
    --categories config/categories.yaml \
    --raw-dir data/raw-hardened-20260919 \
    --output-dir data/processed \
    2>&1 | tee /dev/stderr | sed -n 's/^INFO Prepared bundle: //p' \
)
test -n "$BUNDLE" || exit 1
uv run src/scripts/validate_dataset.py sources --bundle "$BUNDLE"

RUN=$( \
  uv run src/scripts/generate_demographics.py \
    --bundle "$BUNDLE" \
    --config config/sampling.yaml \
    --rows 100000 \
    --seed 20260914 \
    --output-dir data/runs/statistical \
    2>&1 | tee /dev/stderr | sed -n 's/^INFO Generated run: //p' \
)
test -n "$RUN" || exit 1
uv run src/scripts/validate_dataset.py demographics \
  --run "$RUN" \
  --bundle "$BUNDLE"
```

The assignments capture the exact paths printed by the CLI commands rather than
selecting an arbitrary newest directory.

Freeze a deterministic 1,000-row input for optional LLM development work only after the
statistical validation passes. This is a separate, deliberate Phase-3 development size;
it does not replace the canonical 2,000-row Phase-2 smoke validation:

```bash
uv run src/scripts/freeze_demographic_sample.py \
  --run "$RUN" \
  --rows 1000 \
  --output "$RUN/text-development-seeds.parquet"
```

The source preparation stage uses LONS20, FOLK2, FOLK1A, RAS209, RAS202, BEFOLK3, and
RAS210. LONS20's title, dimension semantics, fixed selector labels, and all selected
two-digit ARBF English labels are independently frozen in the reviewed
`config/lons20-contract.yaml`; lock and raw snapshot metadata must each match it. The
contract checksum and version are part of source provenance and bundle identity. LONS20
provides an optional sex-conditional synthetic broad job-function marginal for eligible
employees only. Its incomplete earnings-statistics universe is not an all-worker
representation. The human-readable job-function label reaches the provider so the model
can produce a grounded synthetic job title; the job-function code and resolution do not.
The allocation is conditioned on sex alone, not municipality, origin, education, age,
OCEAN, or any unsupported joint. FOLK2 is an independent national marginal of official
IELAND country-of-origin categories for adults. It preserves categories such as
Stateless and Not stated, but is not ethnicity, citizenship, residence, or appearance.
Each Phase 2 record receives an independently quota-sampled `origin_country_code` and
`origin_country`; the label reaches the provider, while the code and sampling resolution
do not. Origin cannot drive language, culture, religion, occupation, personality, or
visual appearance. FOLK1A, RAS209, and RAS202 ground the distributions; BEFOLK3 and
RAS210 are held-out aggregate diagnostics. Municipality aggregates remain
municipality-keyed throughout source preparation and sampling; they are never grouped
into regional person-sampling artefacts. Generated records retain mandatory
`municipality_code` and `municipality` fields. Region is attached only as the official
parent from the Statistics Denmark classification snapshot, and preparation fails unless
the locked, hierarchy, and prepared RAS209 municipality sets match exactly. The
municipality label reaches the provider for grounded prose; its code and resolution do
not.

## Optional LLM workflow

This workflow is separate from the non-LLM pipeline and may incur provider charges. It
sends frozen aggregate-derived records to the configured OpenAI-compatible endpoint.
Automated checks are necessary but do not replace blinded human review.

First create a local configuration. Keep the committed file disabled; for a dry run, the
local copy can retain `false`, null endpoint/model values, and no token:

```bash
cp config/generation.yaml config/generation.local.yaml
```

Run the safe plan first. It validates the upstream report, sample checksum, prompts,
schemas, and row/request limits without making network requests or creating output
files:

```bash
uv run src/scripts/generate_personas.py \
  --input "$RUN/text-development-seeds.parquet" \
  --sample-manifest "$RUN/text-development-seeds.manifest.json" \
  --config config/generation.local.yaml \
  --output-dir data/persona-smoke \
  --rows 3
```

For an approved smoke test, edit only the ignored local config: set
`llm_generation_enabled: true`, `base_url`, and `model`. Set `api_key_env` to the name
of a bearer-token variable if the endpoint requires authentication. Use a short-lived
command-scoped token assignment and add `--live` explicitly. Replace `OPENAI_API_KEY`
below with the configured `api_key_env` name when needed:

```bash
set -o pipefail
PERSONA_RUN=$( \
  OPENAI_API_KEY='replace-with-a-token' \
  uv run src/scripts/generate_personas.py \
    --input "$RUN/text-development-seeds.parquet" \
    --sample-manifest "$RUN/text-development-seeds.manifest.json" \
    --config config/generation.local.yaml \
    --output-dir data/persona-smoke \
    --rows 3 \
    --live \
    2>&1 | tee /dev/stderr | sed -n 's/^INFO Persona generation run: //p' \
)
test -n "$PERSONA_RUN" || exit 1
uv run src/scripts/validate_dataset.py personas --run "$PERSONA_RUN"
```

Each record uses two model stages: structured attributes, then six Danish texts: five
specialised descriptions (`professional_persona`, `sports_persona`, `arts_persona`,
`travel_persona`, and `culinary_persona`) plus one short, grounded `persona`. The v2
persona includes age, statistical sex, municipality, education, origin, a synthetic job
title grounded in the official job-function label (or the current nonemployee status),
two or three interests in prose, and cautious OCEAN tendencies. It is not a visual
description; `visual_persona` is removed. See
[`docs/persona-prompt-format.md`](docs/persona-prompt-format.md) for the contract. A
repeated live command resumes valid v2 per-record checkpoints and does not repeat
completed calls. v1 checkpoints and old pilots are not resumable under v2. Each
`generate_personas.py` invocation is one shard capped at five rows, while a pilot can
span multiple such shards. The default HTTP-attempt budget is 15 per shard.

For a multi-shard pilot, use `generate_persona_pilot.py`. It requires `--live`, limits
each shard to five rows, validates each shard, merges them, records token/cost
accounting, and validates the merged pilot. A pilot can therefore contain more than five
rows. Enter the provider's current list prices before running the live pilot. Use zero
only when the configured endpoint is genuinely free:

```bash
read -r -p "Current input price (USD per million tokens): " \
  INPUT_PRICE_PER_MILLION
read -r -p "Current output price (USD per million tokens): " \
  OUTPUT_PRICE_PER_MILLION
```

Then run the pilot with the configured bearer-token variable as a short-lived command
assignment. Replace `OPENAI_API_KEY` below with the configured `api_key_env` name when
needed:

```bash
OPENAI_API_KEY='replace-with-a-token' \
uv run src/scripts/generate_persona_pilot.py \
  --input "$RUN/text-development-seeds.parquet" \
  --sample-manifest "$RUN/text-development-seeds.manifest.json" \
  --config config/generation.local.yaml \
  --output-dir data/persona-pilot \
  --rows 10 \
  --batch-size 5 \
  --concurrency 1 \
  --maximum-total-requests 30 \
  --input-price-per-million "$INPUT_PRICE_PER_MILLION" \
  --output-price-per-million "$OUTPUT_PRICE_PER_MILLION" \
  --live
```

The dry-run commands above do not contact a provider. Only commands with `--live` need
provider reachability, and live commands can consume paid requests.

## Outputs and data handling

The repository includes the compressed raw Statistics Denmark snapshot archive and its
attribution. The source archive, lock, category mappings, sampling parameters,
validation thresholds, and code are version controlled. Restored and derived artefacts
remain ignored and reproducible.

All generated artefacts belong under the ignored `data/` directory. Important outputs
are:

- `data/raw-hardened-20260919/<table>/<query-hash>/`: restored immutable `data.csv`,
  metadata, query, response headers, and `snapshot-manifest.json` files;
- `data/raw-hardened-20260919/classifications/<classification-id>/<url-hash>/`: restored
  immutable `data.csv`, `response-headers.json`, and `snapshot-manifest.json` files;
- `data/processed/<bundle-id>/`: normalised Parquet distributions,
  `bundle-manifest.json`, and source preparation reports;
- `data/runs/<name>/<run-id>/`: `structured-records.parquet`, `run-manifest.json`, and
  JSON/Markdown validation reports;
- the frozen sample Parquet file and adjacent `.manifest.json` file;
- `data/persona-smoke/<run-id>/`: generated Parquet, `generation-manifest.json`, request
  ledger, checkpoints, and validation report;
- `data/persona-pilot/<pilot-id>/`: merged Parquet, pilot manifest, shard directories,
  and pilot validation report.

Manifests contain SHA-256 checksums, source/config/prompt provenance, row counts, seeds,
model metadata, request/retry/token accounting, and (when available) cost estimates.
Accepted LLM response metadata and response hashes are checkpointed; rejected completion
text is not stored. Generated outputs remain local until privacy and human review
approve any proposed release.

## Safety and privacy boundary

Statistics Denmark inputs are public aggregate tables, not individual-level records. The
pipeline must not be used to reconstruct or link people. Phase 2 emits synthetic adults
aged 18-125 with the fixed residence value `Danmark`, independently sampled official
FOLK2 origin fields, sex, age, marital status, municipality, its official region parent,
broad education, labour status, detailed status, and independent OCEAN scores. Origin is
not ethnicity, citizenship, or residence, and cannot drive language, culture, religion,
occupation, personality, or visual appearance. It does not emit names, exact addresses,
coordinates, CPR or other administrative identifiers, employers, observed occupations,
income, household details, ancestry, citizenship, health, religion, sexuality, politics,
criminal history, or free text. The sole occupation-related exception is the optional
synthetic broad job function allocated from the incomplete LONS20 earnings-statistics
universe; it is not conditioned on municipality, origin, education, age, OCEAN, or any
unsupported joint. LLM prompts prohibit identifying and sensitive details, stereotypes,
and deterministic claims about demographics or personality. Validators check strict
schemas, Danish text, contact and identifying-number patterns, configured sensitive
terms, duplicate descriptions, grounded persona facts, upstream preservation, checksums,
and checkpoint provenance. Origin labels are not ethnicity or appearance; job titles are
synthetic and must not imply unsupported work history or family claims. Downstream image
models may still stereotype, so text validation is not a guarantee of safe image
generation. These are finite automated checks, not a guarantee of anonymity or safe use.
Treat municipality-level combinations, accepted text, checkpoints, tokens, and provider
telemetry as restricted. Review [`SECURITY.md`](SECURITY.md) for vulnerability reporting
and [`docs/privacy-risk-register.md`](docs/privacy-risk-register.md) before sharing
outputs.

## Validation and development checks

Run the test suite without network access or LLM calls:

```bash
uv run pytest
```

Pytest also runs doctests, enables coverage for `src/danish_personas`, and treats most
warnings as errors. Run the repository's full pre-commit checks directly before
submitting changes:

```bash
uv run pre-commit run --all-files
```

The direct command runs the configured annotation, hygiene, vulture, function-ordering,
Ruff, type-checking, notebook, and Markdown checks. It may fix files in place. Avoid
`make check` unless the index is disposable: it runs `git add .`, invokes the same
hooks, and unconditionally runs `git reset`; it requires nothing staged and discards
staged state. The `make test` target also runs `readme-cov` and commits a README
coverage-badge update, so prefer `uv run pytest` when a side-effect-free test run is
required.

## Further documentation

- [`docs/acceptance-criteria.md`](docs/acceptance-criteria.md): mandatory gates and
  non-zero failure behaviour;
- [`docs/source-register.md`](docs/source-register.md): source tables, periods, and
  harmonisation decisions;
- [`docs/danish-personas-plan.md`](docs/danish-personas-plan.md): design and deferred
  delivery phases;
- [`docs/persona-prompt-format.md`](docs/persona-prompt-format.md): generation contract
  v2 for the six Danish persona texts and provider input boundary;
- [`CONTRIBUTING.md`](CONTRIBUTING.md): project contribution process;
- [`LICENSE`](LICENSE): project licence.
