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
- `docs/` contains the plan, source register, sampling shares, privacy register,
  acceptance criteria, and
  experiment/validation reports. `.github/workflows/ci.yaml` runs pre-commit and pytest
  on Python 3.14 across Ubuntu, macOS, and Windows.

## Modules and packages

| Path | Responsibility |
| --- | --- |
| `danish_personas/__init__.py` | Package metadata and module docstring. |
| `danish_personas/io.py` | YAML, canonical JSON, atomic writes, SHA-256, `.env`. |
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
| `danish_personas/cli.py` | The `personas` command: dataset, brief, generate. |
| `danish_personas/workflow.py` | Stage chaining, run pointers, run inspection. |

## Scripts

Run every script from the repository root with `uv run`. Paths in config files and
prompts are interpreted relative to that working directory.

| Script | Responsibility and invocation |
| --- | --- |
| `restore_raw_sources.py` | Safely restores the archive after validating its members. |
| `build_raw_archive.py` | Repacks restored snapshots into the committed archive. |
| `download_sources.py` | `resolve` locks selectors; `fetch` refreshes snapshots. |
| `build_distributions.py` | Builds a checksummed offline bundle from raw snapshots. |
| `generate_demographics.py` | Creates deterministic Phase 2 and OCEAN records. |
| `validate_dataset.py` | Validates `sources`, `demographics`, or `personas`. |
| `freeze_demographic_sample.py` | Makes a deterministic stratified Phase 3 sample. |
| `generate_personas.py` | Guarded two-stage LLM run, max five rows per shard. |
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
| `tests/test_raw_archive.py` | Committed archive integrity and safe restoration. |
| `tests/test_workflow.py` | Run pointers, record selection, summary, export. |
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
| `config/origin-regions.yaml` | Country-of-origin groupings; the project's own. |
| `config/sampling.yaml` | Seed, rows, adult age range, region, OCEAN settings. |
| `config/validation.yaml` | Distribution, expected-count, and OCEAN thresholds. |
| `config/generation.yaml` | Disabled endpoint, guards, response mode, prompt paths. |
| `config/generation.local.yaml` | Ignored local LLM override and provider settings. |
| `config/prompts/attributes-da.md` | Danish attributes schema and safety rules. |
| `config/prompts/personas-da.md` | Danish seven-description schema and safety rules. |

Changing a lock, category map, sampling setting, validation threshold, prompt, schema,
or validator changes provenance and can change content-addressed run IDs. Do not adjust
thresholds retrospectively to make an existing run pass.

## Setup and checks

Use the [`README.md` developer setup guide](README.md#developer-setup-guide) for the
complete clean-clone workflow, including offline raw-source restoration, prepared bundle
generation, canonical demographic generation, validation, and persona-seed freezing.

Install the exact development environment:

```bash
uv sync --locked --all-extras --dev
```

For a local environment file, use `cp .env.example .env`. The non-LLM tests and pipeline
need no secrets. Generation resolves the provider token in
`generation.pipeline._resolve_api_key`, which loads `.env` through `io.load_env_file`;
variables already in the environment take precedence, so a command-scoped assignment
still wins. The Makefile separately includes `.env` and
exports all of its variables to subprocesses and hooks, including the third-party hooks
that `make check` runs. `make install` is a convenience bootstrap
that can install/update `uv`, initialise Git, configure identity, and add a remote; do
not use it merely to install Python dependencies in an existing clone.

Run tests and quality checks with these exact commands:

```bash
uv run pytest
uv run pre-commit run --all-files
```

`pytest` runs `tests/` plus package doctests, with coverage enabled for
`src/danish_personas`. `make check` runs `git add .`, runs all configured pre-commit
hooks, then unconditionally runs `git reset`; it requires nothing staged and discards
any staged state. Prefer `uv run pre-commit run --all-files` directly. The CI equivalent
is the `uvx pre-commit run --show-diff-on-failure --color=always --all-files` command
followed by `uv run pytest` in a locked environment.

Avoid `make test` for routine verification: after pytest it runs `readme-cov`, stages
`README.md`, and creates a coverage-badge commit. This side effect is part of the
target. Use `make tree` only when the `tree` utility is installed.

Make target notes:

- `help` lists documented targets; `install` is the interactive full bootstrap.
- `install-non-interactive` performs the same bootstrap with blank Git identity;
  `install-uv` and `install-dependencies` are its component steps. `install-pre-commit`
  is standalone, installs the hook, and runs `pre-commit autoupdate`, which may modify
  tracked hook configuration; it is not part of `install`.
- `setup-environment-variables` prompts for Git identity; its non-interactive variant
  leaves values blank. `setup-git` changes local Git settings.
- `add-repo-to-git` may create an initial commit and add the GitHub origin remote.
- `docker` runs the bootstrap, builds the image, and starts an interactive container.
  It requires Docker and has the bootstrap side effects.

## Pipeline ordering

Do not skip a boundary or call an LLM before the demographic gate passes:

1. Restore the committed raw-source archive with safe member validation for offline
   reproduction.
2. Build and validate the offline bundle in `data/processed`.
3. Generate deterministic demographic/OCEAN records.
4. Validate the smoke run, then generate and validate the statistical run.
5. Freeze the stratified text-development sample.
6. Dry-run LLM planning; it needs no provider, and only an approved operator may
   enable `--live`.
7. Validate each persona run; each `generate_personas.py` shard is capped at five rows,
   while a pilot may span multiple validated shards before merge and pilot validation.

The normal non-LLM stages are:

```bash
uv run src/scripts/restore_raw_sources.py
uv run src/scripts/build_distributions.py \
  --lock config/sources.lock.yaml --categories config/categories.yaml \
  --raw-dir data/raw-hardened-20260914 --output-dir data/processed
uv run src/scripts/validate_dataset.py sources --bundle "$BUNDLE"
uv run src/scripts/generate_demographics.py \
  --bundle "$BUNDLE" --rows 1000 --seed 20260914 \
  --output-dir data/runs/smoke
uv run src/scripts/validate_dataset.py demographics \
  --run "$RUN" --bundle "$BUNDLE"
```

Use the full copy-pasteable workflows in `README.md` to capture exact bundle/run paths,
produce the 100,000-row run, and freeze a sample. Run `download_sources.py resolve` and
`fetch` only for an intentional source refresh: both require StatBank network access,
and refreshed responses create a new provenance chain. Review changes to the lock and
source register before accepting refreshed snapshots.

## Outputs and provenance

`data/raw-hardened-20260914.tar.zst` and `data/README.md` are tracked. The restore
script rejects empty archives and unsafe members, permits only regular files under the
expected root, and stages extraction before installing the ignored, immutable snapshots
into `data/raw-hardened-20260914/`. It does not compare the archive's top-level SHA-256
or validate snapshot manifests and checksums; bundle preparation validates each locked
snapshot's manifest, provenance, query, and file checksums.

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
  validation and planning only; it does not need provider reachability. `--live`
  additionally requires local enablement, endpoint, model, provider reachability, and
  may spend money. The pilot has its own global request limit and requires current input
  and output prices; use zero only for a genuinely free endpoint. `--concurrency` can
  issue requests in parallel.
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
  households, citizenship, country of origin, health, religion, sexuality, politics,
  criminal history, or other sensitive fields without a separate privacy review.
- `origin` holds FOLK1E's official ancestry categories and `origin_region` groups
  FOLK1C's country mix. Treat both as administrative categories, never as ethnicity,
  and never let generated text name a country of origin.
- Only lowercase `makefile` is tracked; case-insensitive systems may display it as
  `Makefile`. Make targets can mutate Git state; inspect `git status` before and after
  using them.
- `fix_dot_env_file.py` deletes `.name_and_email` after copying any values and does not
  populate optional token variables. Direct Python execution should still use `uv run`.

Read `docs/acceptance-criteria.md` and `docs/privacy-risk-register.md` before changing
validation or data fields. Report security issues privately as described in
`SECURITY.md`.
