# Danish Personas

This repository builds a synthetic Danish persona dataset from public Statistics Denmark
aggregate tables. It has a deterministic demographic/OCEAN pipeline and a separate,
small, guarded OpenAI-compatible LLM pipeline for attributes and persona prose.

## Stack and layout

- Python 3.14+ (less than 4.0), managed with `uv`; dependencies are locked in
  `pyproject.toml` and `uv.lock`.
- Polars handles tabular data; NumPy and SciPy support sampling and diagnostics;
  Pydantic defines strict contracts; Click provides CLI scripts; HTTPX handles StatBank
  and LLM API calls.
- `src/danish_personas/` contains importable package code. `src/scripts/` contains
  executable Click entry points. `tests/` contains unit and fixture-based integration
  tests. `config/` contains versioned inputs and prompts. `data/` is local output only.
- `docs/` contains the plan, source register, privacy register, acceptance criteria, and
  experiment/validation reports. `.github/workflows/ci.yaml` runs pre-commit and pytest
  on Python 3.14 across Ubuntu, macOS, and Windows.

## Modules and packages

| Path | Responsibility |
| --- | --- |
| `danish_personas/__init__.py` | Package metadata and module docstring. |
| `danish_personas/io.py` | YAML, canonical JSON, atomic writes, SHA-256. |
| `danish_personas/models.py` | Strict Pydantic source, sampling, record contracts. |
| `danish_personas/generation/__init__.py` | LLM-generation package marker. |
| `danish_personas/generation/client.py` | Retrying OpenAI client and accounting. |
| `danish_personas/generation/models.py` | LLM, checkpoint, ledger, pilot contracts. |
| `danish_personas/generation/pipeline.py` | Guarded generation, resume, provenance. |
| `danish_personas/generation/report.py` | Persona-run and pilot integrity gates. |
| `danish_personas/generation/validation.py` | JSON, Danish, safety, duplicate gates. |
| `danish_personas/sampling/__init__.py` | Sampling package marker. |
| `danish_personas/sampling/generator.py` | Deterministic demographics and OCEAN. |
| `danish_personas/sources/__init__.py` | Source-acquisition package marker. |
| `danish_personas/sources/statbank.py` | StatBank selectors and immutable snapshots. |
| `danish_personas/sources/prepare.py` | Aggregate normalisation and calibration. |
| `danish_personas/validation/__init__.py` | Validation package marker. |
| `danish_personas/validation/checks.py` | Source, structure, distribution, OCEAN. |

## Scripts

Run every script from the repository root with `uv run`. Paths in config files and
prompts are interpreted relative to that working directory.

| Script | Responsibility and invocation |
| --- | --- |
| `download_sources.py` | `resolve` locks selectors; `fetch` gets immutable snapshots. |
| `build_distributions.py` | Builds a checksummed offline bundle from raw snapshots. |
| `generate_demographics.py` | Creates deterministic Phase 2 and OCEAN records. |
| `validate_dataset.py` | Validates `sources`, `demographics`, or `personas`. |
| `freeze_demographic_sample.py` | Makes a deterministic stratified Phase 3 sample. |
| `generate_personas.py` | Guarded two-stage LLM run, max five rows. |
| `generate_persona_pilot.py` | Merges validated shards; requires `--live`. |
| `fix_dot_env_file.py` | Creates `.env`; non-interactive leaves Git identity blank. |

Use `uv run src/scripts/<script>.py --help` to inspect Click options. There is no
`generate_attributes.py`; structured attributes are the first stage of
`generate_personas.py`.

## Tests

| Path | Coverage |
| --- | --- |
| `tests/test_models.py` | Source-selection contract validation. |
| `tests/test_llm_guard.py` | Default LLM-disabled guard. |
| `tests/test_non_llm_pipeline.py` | Deterministic fixture pipeline. |
| `tests/test_source_validation.py` | Bundle and raw-snapshot checksum/query gates. |
| `tests/generation/test_client.py` | Request budgets, retries, rate limits, schemas. |
| `tests/generation/test_pipeline.py` | Resume, provenance, tamper, pilot merging. |
| `tests/generation/test_validation.py` | Danish, safety, duplicate-text gates. |

Tests must remain offline and must not call a provider. Mock HTTPX or the generation
client when testing LLM paths.

## Configuration

| Path | Responsibility |
| --- | --- |
| `config/sources.yaml` | Dynamic StatBank selectors and source-count thresholds. |
| `config/sources.lock.yaml` | Resolved codes, queries, URLs, periods, and timestamps. |
| `config/categories.yaml` | Canonical demographic and labour-status mappings. |
| `config/sampling.yaml` | Seed, rows, adult age range, region, OCEAN settings. |
| `config/validation.yaml` | Distribution, expected-count, and OCEAN thresholds. |
| `config/generation.yaml` | Disabled endpoint, guards, response mode, prompt paths. |
| `config/generation.local.yaml` | Ignored local LLM override and provider settings. |
| `config/prompts/attributes-da.md` | Danish attributes schema and safety rules. |
| `config/prompts/personas-da.md` | Danish six-description schema and safety rules. |

Changing a lock, category map, sampling setting, validation threshold, prompt, schema,
or validator changes provenance and can change content-addressed run IDs. Do not adjust
thresholds retrospectively to make an existing run pass.

## Setup and checks

Install the exact development environment:

```bash
uv sync --locked --all-extras --dev
```

For a local environment file, use `cp .env.example .env`. The non-LLM tests and pipeline
need no secrets. `make install` is a convenience bootstrap that can install/update `uv`,
initialise Git, configure identity, and add a remote; do not use it merely to install
Python dependencies in an existing clone.

Run tests and quality checks with these exact commands:

```bash
uv run pytest
make check
uv run pre-commit run --all-files
```

`pytest` runs `tests/` plus package doctests, with coverage enabled for
`src/danish_personas`. `make check` stages files temporarily, runs all configured
pre-commit hooks, resets the index, and can rewrite files. The CI equivalent is
`uvx pre-commit run --show-diff-on-failure --color=always --all-files` followed by
`uv run pytest` in a locked environment.

Avoid `make test` for routine verification: after pytest it runs `readme-cov`, stages
`README.md`, and creates a coverage-badge commit. This side effect is part of the
target. Use `make tree` only when the `tree` utility is installed.

Make target notes:

- `help` lists documented targets; `install` is the interactive full bootstrap.
- `install-non-interactive` performs the same bootstrap with blank Git identity;
  `install-uv`, `install-dependencies`, and `install-pre-commit` are its component
  steps.
- `setup-environment-variables` prompts for Git identity; its non-interactive variant
  leaves values blank. `setup-git` changes local Git settings.
- `add-repo-to-git` may create an initial commit and add the GitHub origin remote.
- `docker` runs the bootstrap, builds the image, and starts an interactive container.
  It requires Docker and has the bootstrap side effects.

## Pipeline ordering

Do not skip a boundary or call an LLM before the demographic gate passes:

1. Resolve selectors when refreshing sources, then review the resulting lock.
2. Fetch locked StatBank snapshots into `data/raw`.
3. Build and validate the offline bundle in `data/processed`.
4. Generate deterministic demographic/OCEAN records.
5. Validate the smoke run, then generate and validate the statistical run.
6. Freeze the stratified text-development sample.
7. Dry-run LLM planning; only an approved operator may enable `--live`.
8. Validate each persona run; pilot shards must pass before merge and pilot validation.

The normal non-LLM commands are:

```bash
uv run src/scripts/download_sources.py resolve \
  --config config/sources.yaml --lock config/sources.lock.yaml
uv run src/scripts/download_sources.py fetch \
  --lock config/sources.lock.yaml --raw-dir data/raw
uv run src/scripts/build_distributions.py \
  --lock config/sources.lock.yaml --categories config/categories.yaml \
  --raw-dir data/raw --output-dir data/processed
uv run src/scripts/validate_dataset.py sources --bundle "$BUNDLE"
uv run src/scripts/generate_demographics.py \
  --bundle "$BUNDLE" --rows 1000 --seed 20260914 \
  --output-dir data/runs/smoke
uv run src/scripts/validate_dataset.py demographics \
  --run "$RUN" --bundle "$BUNDLE"
```

Use the full copy-pasteable workflows in `README.md` for bundle/run discovery, the
100,000-row run, and sample freezing. The `resolve` step is not needed to reproduce the
committed lock and requires StatBank network access. Fetching sources also downloads
external data and is deliberately not part of ordinary tests.

## Outputs and provenance

`data/` is ignored and starts with only `.gitkeep`. Source snapshots are
content-addressed by table and query checksum. They contain `data.csv`,
English/Danish metadata, `query.json`, response headers, and
`snapshot-manifest.json`. Existing valid snapshots are immutable.

Prepared bundles contain normalised Parquet files, `bundle-manifest.json`, and source
preparation reports. Deterministic runs contain `structured-records.parquet`,
`run-manifest.json`, and JSON/Markdown validation reports. Frozen samples have an
adjacent `.manifest.json`. Persona runs contain `generated-personas.parquet`,
`generation-manifest.json`, `request-ledger.json`, per-person attribute/final
checkpoints, and `validation-report.json`. Pilots additionally contain merged output,
a pilot manifest, shard references, and `pilot-validation-report.json`.

Manifests bind outputs to input/config/prompt/schema/validator checksums, row order, and
request accounting. Deterministic run IDs derive from bundle/config/row/seed inputs;
LLM run IDs include the frozen input and generation context. Existing checksum failures
must fail loudly, not be repaired by overwriting files.

## Repository-specific conventions

- Keep Python and Markdown lines at 88 characters or fewer. Use Ruff formatting and
  linting, Google-style docstrings, full type annotations, and Python 3.14 syntax.
- Use relative imports inside `danish_personas`; scripts and tests import the package
  absolutely. Keep imports at the top, use `pathlib.Path`, and call functions with
  keyword arguments where practical.
- Use British English in documentation and code prose. Avoid `print`; use logging in
  scripts. Keep modules focused and order high-level functions before helpers.
- Add tests under `tests/` for behavioural changes. Prefer fixture-based, no-network
  tests.
- Use Conventional Commits (`docs: ...`, `fix: ...`, `feat: ...`, and so on). Do not
  commit `.env`, `.name_and_email`, `config/*.local.yaml`, `data/`, tokens, or raw
  provider data.

## Non-obvious gotchas and safety

- The committed generation config is disabled. `generate_personas.py` dry-run performs
  validation and planning only; `--live` additionally requires local enablement,
  endpoint, model, and may spend money. The pilot has its own global request limit and
  prices, and `--concurrency` can issue requests in parallel.
- The LLM input must be a frozen sample with a matching manifest and successful upstream
  demographic report. Checkpoints reject changed inputs, prompts, config, model, or
  validator context. Re-running a valid live run resumes completed records.
- The generation client records HTTP attempts before network I/O, retries only bounded
  transport/rate/server failures, and persists a request ledger. Accepted response
  metadata and hashes are retained; rejected completion text is not.
- LLM output must remain strict JSON, Danish, non-identifying, free of configured
  sensitive terms, and free of exact duplicate descriptions. Automated validation is not
  a substitute for blinded human review.
- Statistics Denmark tables are aggregates. Do not link them to people or infer
  individual records. Municipality data is used for regional calibration and is absent
  from Phase 2 output. Do not add names, addresses, occupations, employers, income,
  households, citizenship, ancestry, health, religion, sexuality, politics, criminal
  history, or other sensitive fields without a separate privacy review.
- `makefile` and `Makefile` currently have identical tracked contents. Make targets can
  mutate Git state; inspect `git status` before and after using them.
- `fix_dot_env_file.py` deletes `.name_and_email` after copying any values and does not
  populate optional token variables. Direct Python execution should still use `uv run`.

Read `docs/acceptance-criteria.md` and `docs/privacy-risk-register.md` before changing
validation or data fields. Report security issues privately as described in
`SECURITY.md`.
