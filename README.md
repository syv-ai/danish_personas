# Danish Personas

A reproducible pipeline for creating statistically grounded Danish synthetic persona
records from public aggregate data. The default workflow is deterministic and does not
call an LLM or use personal microdata. A separate, explicitly guarded workflow can add
Danish attributes and persona text to a small frozen sample.

The design rationale and deferred work are documented in
[`docs/danish-personas-plan.md`](docs/danish-personas-plan.md).

## Status and scope

The source-acquisition, preparation, deterministic sampling, and validation stages are
implemented. The Hydra config defaults to a local OpenAI-compatible endpoint. Each
direct generation shard is capped at five rows, while a pilot can span multiple shards.
Release-scale generation and human approval remain pending; schema-2 package verification
and evidence are required before
any release claim.

Future runs use prepared-bundle schema 7, sampler schema 7, frozen-sample schema 4,
and generation contract 5 with validator `persona-safety-v19`. The default frozen
sample mode is population-proportional; stratified round-robin remains an explicit
alternative. No new bundle, deterministic run, frozen sample, persona output, or
release has been generated for this contract change. Existing IDs and checksums are
historical evidence only and are not resumable under these contracts:

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

### Script interface

Run the four public scripts directly from the repository root with `uv run`:

```bash
uv run src/scripts/fix_dot_env_file.py --help
uv run src/scripts/generate_persona.py --help
uv run src/scripts/build_dataset.py --help
uv run src/scripts/build_persona_dashboard.py --help
```

The three Hydra-based scripts use `config/config.yaml`. Override values
with expressions such as `llm.model=MODEL`, `build_dataset.rows=10`, and
`persona_dashboard.input=PATH`. `generate_persona.py` emits one validated Danish
persona to stdout. `build_dataset.py` saves the merged Parquet dataset below `data/`
and displays row progress on stderr. The dataset builder can optionally package,
verify, and upload an approved release to a Hugging Face dataset pull request. The
dashboard builder writes a self-contained offline HTML file.

When `generate_persona.input` or `build_dataset.input` is null, the script automatically
restores the committed archive when necessary, prepares and validates the deterministic
source bundle, runs and validates the smoke and statistical demographic stages, and
freezes the standard 1,000-row sample. Existing content-addressed artefacts are reused.
When an input is provided, its adjacent `.manifest.json` is used.

Source acquisition, archive packing and restoration, deterministic generation, sample
freezing, and validation remain importable maintenance services rather than public
scripts. See [`docs/cli.md`](docs/cli.md) for the four-script interface.

### Regenerate deterministic prerequisites

The persona scripts prepare these prerequisites automatically. To invoke the underlying
maintenance services from Python, call
`danish_personas.workflows.prepare_standard_sample()`. It returns the frozen sample and
its adjacent manifest path, and fails if any source or demographic validation gate
fails. The committed archive and its attribution are documented in
[`data/README.md`](data/README.md).

The workflow restores the seven exact Statistics Denmark aggregate snapshots and the
official geography classification snapshot, prepares a local source bundle, generates
and validates 2,000 deterministic smoke records, then generates and validates 100,000
statistical records before freezing the 1,000-row development sample. Resolve selectors
or fetch from StatBank only when intentionally updating the source lock; refreshed
responses form a new provenance chain rather than reproducing this release.

All generated run identifiers are derived from input checksums, row count, and seed.
Repeating a valid workflow reuses existing runs; a checksum mismatch fails instead of
overwriting data.

## Optional LLM workflow

This workflow is separate from the non-LLM pipeline and may incur provider charges. It
sends frozen aggregate-derived records to the configured OpenAI-compatible endpoint.
Automated checks are necessary but do not replace blinded human review.

All model settings live in Hydra
[`config/config.yaml`](config/config.yaml). Its defaults are:

```yaml
llm:
  base_url: http://127.0.0.1:18080/v1
  model: gpt-5.6-sol
  api_key_env: null
```

Edit the `llm` section or use a Hydra override when changing providers or models. If
authentication is required, set `llm.api_key_env` to the environment-variable name
containing the bearer token; never put the token itself in `config/config.yaml`.
Generation commands execute immediately and can consume paid requests. Each command
persists the resolved, secret-free flat generation configuration below its output area
and uses that immutable snapshot for generation provenance.

The single-persona command validates the upstream report, sample checksum, prompts,
schemas, and request limits before emitting the final Danish text. The standard sample
is prepared automatically when needed:

```bash
uv run src/scripts/generate_persona.py
```

Use `generate_persona.input=PATH` to select another current frozen sample; its adjacent
`.manifest.json` is used automatically. With a null input, the deterministic
prerequisites are prepared first. Each invocation samples one demographic locally, then
starts a fresh model request to return both structured attributes and a detailed Danish
`persona`. Direct invocations
do not reuse earlier persona checkpoints. Only approved human-readable fields reach the
provider. Source codes, resolution fields, the English origin label, and origin-contract
metadata remain withheld.

For a larger dataset, enter the provider's current list prices. Use zero only when the
configured endpoint is genuinely free:

```bash
read -r -p "Current input price (USD per million tokens): " \
  INPUT_PRICE_PER_MILLION
read -r -p "Current output price (USD per million tokens): " \
  OUTPUT_PRICE_PER_MILLION

uv run src/scripts/build_dataset.py \
  build_dataset.rows=10 \
  build_dataset.concurrency=1 \
  build_dataset.request_limit=30 \
  build_dataset.input_price_per_million="$INPUT_PRICE_PER_MILLION" \
  build_dataset.output_price_per_million="$OUTPUT_PRICE_PER_MILLION"
```

The builder limits each shard to five rows, validates and merges all shards, records
request/token/cost accounting, shows `tqdm` progress on stderr, and writes only the
completed Parquet path to stdout.

Setting `build_dataset.hf_repo=OWNER/DATASET` additionally requires the
`build_dataset.attestation`, `build_dataset.policy`, `build_dataset.dataset_card`, and
`build_dataset.licence` paths. The script checks them before generation and packages the
release below
`<output-dir>/releases`, uses the current working directory as repository root, and
independently verifies it before uploading to a Hugging Face dataset pull request.
Authentication comes from the standard `HF_TOKEN` or cached Hugging Face credentials;
tokens are never CLI arguments. Re-running after review resumes already validated
generation shards.
Build a self-contained offline dashboard with explicit Hydra path overrides:

```bash
uv run src/scripts/build_persona_dashboard.py \
  persona_dashboard.input=data/personas/generated-personas.parquet \
  persona_dashboard.bundle=data/processed/BUNDLE_ID \
  persona_dashboard.output=data/personas/dashboard.html
```

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
The current release documentation targets release manifest schema 2 and evidence
schema 2; release identifiers and checksums are placeholders until regeneration and
packaging produce them. Accepted LLM response metadata and response hashes are checkpointed;
rejected completion text is not stored. Generated outputs remain local until privacy and
human review approve any proposed release.

## Safety and privacy boundary

Statistics Denmark inputs are public aggregate tables, not individual-level records. The
pipeline must not be used to reconstruct or link people. Phase 2 emits synthetic adults
aged 18-125 with the fixed residence value `Danmark`, independently sampled official
FOLK2 origin fields, sex, age, marital status, municipality, its official region parent,
broad education, labour status, detailed status, and independent OCEAN scores. The
English `origin_country` is retained only as official source/audit provenance; the
mandatory `origin_country_da` is the archived official Danish display label. Neither
label is ethnicity, citizenship, residence, or appearance. Origin cannot drive culture,
religion, job, interests, personality, or visual traits. It does not emit names, exact
addresses, coordinates, CPR or other administrative identifiers, employers, observed
occupations, income, household details, ancestry, citizenship, health, religion,
sexuality, politics, criminal history, or free text. The sole occupation-related
exception is the optional synthetic broad job function allocated from the incomplete
LONS20 earnings-statistics universe; it is not conditioned on municipality, origin,
education, age, OCEAN, or any unsupported joint. LLM prompts prohibit identifying and
sensitive details, stereotypes, and deterministic claims about demographics or
personality. Validators check strict schemas, Danish text, contact and
identifying-number patterns, configured sensitive terms, duplicate descriptions,
grounded persona facts, upstream preservation, checksums, and checkpoint provenance.
Neither origin label is ethnicity, citizenship, residence, or appearance; origin cannot
drive culture, religion, job, interests, personality, or visual traits. Job titles are
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
  v4, Danish origin-label contract, one persona text, and provider boundary;
- [`CONTRIBUTING.md`](CONTRIBUTING.md): project contribution process;
- [`LICENSE`](LICENSE): project licence.
