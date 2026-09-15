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

The validated local Phase 2 run has 100,000 records. The Phase 3 smoke report
documents a passed three-record run, but that generated data is not committed. Read
the reports before making statistical or quality claims:

- [`docs/reports/phase-2-validation.md`](docs/reports/phase-2-validation.md)
- [`docs/reports/phase-3-smoke.md`](docs/reports/phase-3-smoke.md)
- [`docs/privacy-risk-register.md`](docs/privacy-risk-register.md)

## How a persona is generated

Two stages produce a persona, and they have different guarantees. The demographic stage
is deterministic and grounded in Statistics Denmark aggregates; the text stage is a
language model writing prose from that record.

### Stage 1: the statistical record

`personas demographics` restores the committed snapshots, prepares the source bundle,
and samples each record in a fixed order:

1. Quota-sample the RAS209 joint of region, age band, sex, broad education, and
   labour-market status.
2. Draw an exact age from FOLK1A, conditioned on age band and sex.
3. Draw marital status from FOLK1A, conditioned on region, age band, and sex.
4. Draw origin from FOLK1E, conditioned on region, age band, and sex.
5. Draw an origin region from the FOLK1C country mix, conditioned on sex and the
   origin category, so the western and non-western groups stay consistent.
6. Draw detailed labour-market status from RAS202, conditioned on age band, sex, and
   labour-market status.
7. Draw five independent OCEAN scores from the normal distribution configured in
   `config/sampling.yaml`.

Rare cells are dropped before sampling: every source frame is filtered by a release
count floor, so a combination too small to publish cannot be drawn. The run identifier
is derived from the bundle, the sampling configuration, the row count, and the seed, so
the same inputs always reproduce the same records.

### Stage 2: the generated text

`personas generate` freezes a stratified sample of the run, then makes two model calls
per persona. The first sends the whole record as JSON and receives four attributes;
the second sends the record plus those attributes and receives seven Danish
descriptions. Both are validated against strict schemas before they are checkpointed.

A record whose generated text fails a schema, language or safety gate is retried, and
dropped from the run if it fails again. The run continues, and the manifest records the
dropped identifiers in `skipped_persona_ids` next to `generated_rows`, so a dataset is
never silently short. Provenance failures, such as a tampered checkpoint, still abort
the run, as does a shard in which every record fails, since that signals a broken prompt
or configuration rather than one awkward record. Keep `--batch-size` above one in a
pilot, so a single rejected record does not empty its shard.

Dropping records is not neutral: the gates correlate with topics, so a dataset with
many skips is no longer a clean sample of its inputs.

Six of those descriptions are prose about one facet of the person. The seventh,
`visual_persona`, is written to seed a portrait image instead: it opens with sex and
age, states the origin region in one fixed sentence, and then gives two or three
invented appearance details such as hair, glasses, or clothing. It never guesses a
country, skin colour, ethnicity, or religion, and it describes nothing below the
shoulders. Those appearance details are invented in the same sense the hobbies are;
they are not derived from the origin category, and only the fixed sentence carries the
statistical fact.

### What comes from where

The released record has 38 columns, and only ten of them are sampled from official
statistics. [`docs/sampling-shares.md`](docs/sampling-shares.md) lists every field
with its actual share.

| Source | Columns | Fields |
| --- | ---: | --- |
| Statistics Denmark | 10 | `sex`, `age_band`, `region`, `education_level`, `labour_market_status`, `age`, `marital_status`, `origin`, `origin_region`, `detailed_status` |
| Derived or constant | 7 | `persona_id`, `country`, `region_code`, `education_resolution`, and the three StatBank source codes |
| Configured distribution | 10 | The five OCEAN scores and their labels |
| Language model | 11 | `cultural_context`, `skills_and_expertise`, `hobbies_and_interests`, `career_goals_and_ambitions`, and the seven persona descriptions |

The OCEAN scores are not Danish statistics. They are five independent draws from a
configured normal distribution, and nothing in the source tables constrains them. The
attributes and descriptions are invented by the model from the record alone; no source
table says anything about hobbies, skills, or ambitions.

### Relationship to Nemotron Personas

The schema and the two-stage structure follow NVIDIA's
[Nemotron-Personas][nemotron-dataset], and the prompts adopt several of its documented
instructions: definite present-tense writing, specific detail over generic phrasing,
attributes that stay internally consistent with the profile, age used actively, and
cultural context infused implicitly rather than named.

This project diverges where Danish sources and the privacy boundary require it. It emits
no names, addresses, employers, or occupations, so the implicit signal NVIDIA gets from
census-sampled names is absent here. `cultural_context` is deliberately renamed from
`cultural_background` to discourage invented identity claims, there are seven persona
descriptions rather than nine, and `origin` carries Statistics Denmark's official
ancestry categories without a country of origin.
[`docs/danish-personas-plan.md`](docs/danish-personas-plan.md) records the full
comparison and the decisions behind it.

[nemotron-dataset]: https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA

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
`pre-commit autoupdate`, which may modify the tracked hook configuration; it is not
part of `make install`:

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
selects the variable through `api_key_env`.

Generation reads `.env` when it resolves the provider token, so the configured token
does not have to be repeated on every command. Variables already in the
environment win, so a command-scoped assignment still overrides the file. Note that the
Makefile separately includes `.env` and exports all of its variables to subprocesses and
hooks, including the third-party hooks that `make check` runs; keep that in mind when
deciding what to store there. Never commit `.env`, tokens, or generated data
artefacts.

### The `personas` command

`uv sync` installs a `personas` command that runs the whole pipeline. It has three main
subcommands and takes ordinary flags:

```bash
uv run personas demographics --rows 100000 --seed 20260914
uv run personas brief --rows 5
uv run personas generate --rows 2 --live
```

`demographics` restores the raw snapshots if needed, prepares the source bundle, and
generates and validates the dataset. `brief` prints plain records from it: age, region,
education, labour-market status, and job type. `generate` adds the LLM-written
attributes and descriptions, keeping every statistical field on the record. Add
`--export` to `brief` or `generate` to write each record to `data/exports` as JSON and
Markdown, with the statistical inputs and the generated text in separate sections.

`uv run personas --help` lists every command, and `summary`, `runs`, `clean`, and
`setup-llm` cover distributions, local state, cleanup, and provider configuration. The
current dataset is recorded under `data/.state/`, so the reading commands need no
arguments; pass `--run <path>` to read a different one.

`--rows` defaults to `statistical_rows`, because the distribution thresholds in
`config/validation.yaml` are calibrated for that size; a smaller dataset fails its
validation gates. `--skip-validation` keeps such a dataset for a quick look, but it
cannot seed persona generation and is not release data.

`generate` plans a dry run and makes no network calls until `--live`. Configure a
provider first with `uv run personas setup-llm --base-url <url> --model <id>
[--api-key-env <name>]`, which enables the ignored `config/generation.local.yaml`. The
token is read from the environment variable that `api_key_env` names, and `.env` fills
it in when the variable is not already set. Remember that the Makefile separately
exports every `.env` variable to its subprocesses and hooks.

`make demographics`, `make brief`, and `make personas` are zero-argument shortcuts for
the three commands; anything with arguments goes to the command directly.

### Regenerate development data

The following commands restore the six exact Statistics Denmark aggregate snapshots,
prepare a local source bundle, generate 1,000 deterministic records, and validate every
stage without network access. The archive and attribution are documented in
[`data/README.md`](data/README.md).

```bash
set -o pipefail

uv run src/scripts/restore_raw_sources.py

BUNDLE=$( \
  uv run src/scripts/build_distributions.py \
    --lock config/sources.lock.yaml \
    --categories config/categories.yaml \
    --raw-dir data/raw-hardened-20260914 \
    --output-dir data/processed \
    2>&1 | tee /dev/stderr | sed -n 's/^INFO Prepared bundle: //p' \
)
test -n "$BUNDLE" || exit 1
uv run src/scripts/validate_dataset.py sources --bundle "$BUNDLE"

RUN=$( \
  uv run src/scripts/generate_demographics.py \
    --bundle "$BUNDLE" \
    --rows 1000 \
    --seed 20260914 \
    --output-dir data/runs/smoke \
    2>&1 | tee /dev/stderr | sed -n 's/^INFO Generated run: //p' \
)
test -n "$RUN" || exit 1
uv run src/scripts/validate_dataset.py demographics \
  --run "$RUN" \
  --bundle "$BUNDLE"
```

The assignments capture the exact paths printed by the CLI commands. They do not
select an arbitrary newest directory, and fail if a command emits no path.

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
    --raw-dir data/raw-hardened-20260914 \
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
statistical validation passes:

```bash
uv run src/scripts/freeze_demographic_sample.py \
  --run "$RUN" \
  --rows 1000 \
  --output "$RUN/text-development-seeds.parquet"
```

The source preparation stage uses FOLK1A, FOLK1E, RAS209, RAS202, BEFOLK3, and RAS210.
The first four ground the distributions; BEFOLK3 and RAS210 are held-out aggregate diagnostics.
Municipality aggregates are used to construct regional counts, but municipality fields
are not emitted in generated records.

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
command-scoped token assignment and add `--live` explicitly. Replace
`OPENAI_API_KEY` below with the configured `api_key_env` name when needed:

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

Each record uses two model stages: structured attributes, then seven Danish
descriptions.
A repeated live command resumes valid per-record checkpoints and does not repeat
completed calls. Each `generate_personas.py` invocation is one shard capped at five
rows, while a pilot can span multiple such shards. The default HTTP-attempt budget is 15
per shard.

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

The repository includes the 609 KB compressed raw StatBank snapshot archive and its
attribution. The source archive, lock, category mappings, sampling parameters,
validation thresholds, and code are version controlled. Restored and derived artefacts
remain ignored and reproducible.

All generated artefacts belong under the ignored `data/` directory. Important outputs
are:

- `data/raw-hardened-20260914/<table>/<query-hash>/`: restored immutable `data.csv`,
  metadata, query, response headers, and `snapshot-manifest.json` files;
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
aged 18-125 with country, sex, age, marital status, region, official origin category,
broad education, labour status, detailed status, and independent OCEAN scores. Origin
uses FOLK1E's five administrative ancestry categories, which are not ethnicity, and no
country of origin is emitted. It does not emit names, exact
addresses, coordinates, CPR or other administrative identifiers, employers, occupations,
income, household details, ancestry, citizenship, health, religion, sexuality, politics,
criminal history, or free text.

LLM prompts prohibit identifying and sensitive details, and traits that follow
categorically from sex, age, region, education, or labour-market status. They ask for
definite, present-tense descriptions of an invented individual rather than hedged ones,
so the text asserts what that fictional person does; it never asserts that a demographic
group behaves that way, and it never reproduces personality scores or labels. Validators
check strict schemas, Danish text, contact and identifying-number patterns, configured
sensitive terms, duplicate descriptions, upstream preservation, checksums, and
checkpoint provenance. These are finite automated checks, not a guarantee of anonymity
or safe use. Treat regional combinations, accepted text, checkpoints, tokens, and
provider telemetry as restricted.
Review [`SECURITY.md`](SECURITY.md) for vulnerability reporting and
[`docs/privacy-risk-register.md`](docs/privacy-risk-register.md) before sharing outputs.

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
- [`docs/sampling-shares.md`](docs/sampling-shares.md): every sampled field, its
  source, and its share of the dataset;
- [`docs/source-register.md`](docs/source-register.md): source tables, periods, and
  harmonisation decisions;
- [`docs/danish-personas-plan.md`](docs/danish-personas-plan.md): design and deferred
  delivery phases;
- [`CONTRIBUTING.md`](CONTRIBUTING.md): project contribution process;
- [`LICENSE`](LICENSE): project licence.
