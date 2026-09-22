# Danish Personas

A reproducible pipeline for creating statistically grounded Danish synthetic persona
records from public aggregate data. The default workflow is deterministic and does not
call an LLM or use personal microdata. A separate OpenAI-compatible workflow can add
schema-constrained attributes and persona text to a frozen sample.

## Status and scope

The source-acquisition, preparation, deterministic sampling, and demographic validation
stages are implemented. Each LLM generation shard is capped at five rows, while a
dataset can span multiple shards. Generated responses are constrained by the Pydantic
JSON schema and parsed locally once; no semantic content or completed-run validation is
performed.

Future runs use prepared-bundle schema 8, sampler schema 8, frozen-sample schema 5,
and generation contract 6. The default frozen sample mode is population-proportional;
stratified round-robin remains an explicit alternative. Existing generation IDs and
checkpoints are historical and are not resumable under generation contract 6.

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
selects the variable through `api_key_env`. Every executable script loads all variables
from the repository-root `.env` before starting its CLI, while existing process
variables take precedence. Values are never logged. The Makefile also includes `.env`
and exports its variables to subprocesses and hooks. Never commit `.env`, tokens, or
generated data artefacts.

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
`persona_dashboard.input=PATH`. `generate_persona.py` emits one schema-valid persona to
stdout. `build_dataset.py` saves the merged Parquet dataset below `data/` and displays
row progress on stderr. Setting `build_dataset.hf_repo=OWNER/DATASET` uploads that
Parquet file directly to a Hugging Face dataset pull request using standard Hugging Face
authentication. No content review or release-policy gate is applied. The dashboard
builder writes a self-contained offline HTML file.

When `generate_persona.input` or `build_dataset.input` is null, the script automatically
restores the committed archive when necessary, prepares and validates the deterministic
source bundle, runs and validates the smoke and statistical demographic stages, and
freezes the standard 1,000-row sample. Existing content-addressed artefacts are reused.
When an input is provided, its adjacent `.manifest.json` is used.

Source acquisition, archive packing and restoration, deterministic generation, sample
freezing, and validation remain importable maintenance services rather than public
scripts.

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
The request includes a strict JSON schema and the response is parsed against the same
Pydantic model. Generated content is not otherwise checked.

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

Both persona commands validate deterministic upstream inputs and request limits, send
the Pydantic JSON schema to the provider, and parse each response once against that
schema. They do not check language, safety, names, grounding, age, pronouns,
relationships, job titles, personality wording, or duplicates, and they do not produce
completed-run validation reports. The standard sample is prepared automatically when
needed:

```bash
uv run src/scripts/generate_persona.py
```

Use `generate_persona.input=PATH` to select another current frozen sample; its adjacent
`.manifest.json` is used automatically. With a null input, the deterministic
prerequisites are prepared first. Each invocation samples one demographic locally, then
starts a fresh model request to return both structured attributes and a detailed Danish
`persona`. Direct invocations do not reuse earlier persona checkpoints. Approved
human-readable fields and an internal deterministic same-sex-partner target reach the
provider. The target is synthetic, non-public, not observed individual data, and does
not describe sexual orientation. Source codes, resolution fields, the English origin
label, and origin-contract metadata remain withheld.

For a larger dataset, configure the row count, concurrency, and request limit as
needed:

```bash
uv run src/scripts/build_dataset.py \
  build_dataset.rows=10 \
  build_dataset.concurrency=1 \
  build_dataset.request_limit=30
```

Pricing is not a public Hydra setting. The builder uses zero list-price values for
internal accounting while retaining provider-reported costs when available. It limits
each shard to five rows, merges schema-parsed shards, records request/token/cost
accounting, shows `tqdm` progress on stderr, and writes only the completed Parquet path
to stdout. Setting `build_dataset.hf_repo=OWNER/DATASET` uploads the merged Parquet
directly. Authentication comes from the standard `HF_TOKEN` or cached Hugging Face
credentials; tokens are never CLI arguments.
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
- persona run directories: generated Parquet, `generation-manifest.json`, request ledger,
  and per-person checkpoints;
- persona dataset directories: merged Parquet, pilot manifest, and shard directories.

Manifests contain SHA-256 checksums, source/config/prompt/schema provenance, row counts,
model metadata, request/retry/token accounting, and available cost estimates. Provider
response metadata and response hashes are checkpointed. Setting `build_dataset.hf_repo`
can upload generated output without additional review or packaging.

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
education, age, OCEAN, or any unsupported joint. LLM prompts ask the provider to avoid
identifying and sensitive details, stereotypes, and deterministic claims about
demographics or personality, but those instructions are
not enforced after generation. The local parser checks only the response schema. Output
may therefore be non-Danish, unsafe, contradictory, duplicated, or ungrounded and must
not be treated as reviewed. Neither origin label is ethnicity, citizenship, residence,
or appearance; origin should not drive culture, religion, job, interests, personality,
or visual traits. Treat municipality-level combinations, generated text, checkpoints,
tokens, and provider telemetry as restricted. Review [`SECURITY.md`](SECURITY.md) for vulnerability
reporting.

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

- [`docs/source-register.md`](docs/source-register.md): source tables, periods, and
  harmonisation decisions;
- [`CONTRIBUTING.md`](CONTRIBUTING.md): project contribution process;
- [`LICENSE`](LICENSE): project licence.
