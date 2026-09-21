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

| Path                                        | Responsibility                                      |
| ------------------------------------------- | --------------------------------------------------- |
| `danish_personas/__init__.py`               | Package metadata and module docstring.              |
| `danish_personas/io.py`                     | YAML, canonical JSON, atomic writes, SHA-256.       |
| `danish_personas/models.py`                 | Strict Pydantic source, sampling, record contracts. |
| `danish_personas/generation/__init__.py`    | LLM-generation package marker.                      |
| `danish_personas/generation/client.py`      | Retrying OpenAI client and accounting.              |
| `danish_personas/generation/models.py`      | LLM, checkpoint, ledger, pilot contracts.           |
| `danish_personas/generation/pipeline.py`    | Guarded generation, resume, provenance.             |
| `danish_personas/generation/report.py`      | Persona-run and pilot integrity gates.              |
| `danish_personas/generation/validation.py`  | JSON, Danish, safety, duplicate gates.              |
| `danish_personas/sampling/__init__.py`      | Sampling package marker.                            |
| `danish_personas/sampling/generator.py`     | Deterministic demographics, back-off, OCEAN.        |
| `danish_personas/sources/__init__.py`       | Source-acquisition package marker.                  |
| `danish_personas/sources/http.py`           | Retrying requests and header serialisation.         |
| `danish_personas/sources/statbank.py`       | StatBank selectors and immutable snapshots.         |
| `danish_personas/sources/classification.py` | dst.dk classification attachment snapshots.         |
| `danish_personas/sources/prepare.py`        | Aggregate normalisation and calibration.            |
| `danish_personas/validation/__init__.py`    | Validation package marker.                          |
| `danish_personas/validation/checks.py`      | Source, structure, distribution, OCEAN.             |

## Scripts

Run every script from the repository root with `uv run`. Paths in config files and
prompts are interpreted relative to that working directory.

| Script                         | Responsibility and invocation                              |
| ------------------------------ | ---------------------------------------------------------- |
| `restore_raw_sources.py`       | Safely restores the archive after validating its members.  |
| `build_raw_archive.py`         | Packs restored raw snapshots into a byte-stable archive.   |
| `download_sources.py`          | `resolve` locks selectors; `fetch` refreshes snapshots.    |
| `build_distributions.py`       | Builds a checksummed offline bundle from raw snapshots.    |
| `generate_demographics.py`     | Creates deterministic Phase 2 and OCEAN records.           |
| `validate_dataset.py`          | Validates `sources`, `demographics`, or `personas`.        |
| `freeze_demographic_sample.py` | Makes a deterministic stratified Phase 3 sample.           |
| `generate_persona.py`          | Guarded two-stage LLM run for one persona.                 |
| `build_dataset.py`             | Merges validated shards; requires `--live`.                |
| `fix_dot_env_file.py`          | Creates `.env`; non-interactive leaves Git identity blank. |

Use `uv run src/scripts/<script>.py --help` to inspect Click options. There is no `generate_attributes.py`; structured attributes are the first stage of
`generate_persona.py` and `build_dataset.py`.
## Tests

| Path                 | Coverage                                                    |
| -------------------- | ----------------------------------------------------------- |
| `tests/cli/`         | Command contracts, guards, delegation, and workflows.       |
| `tests/generation/`  | Client, pipeline, pilot, provenance, and persona gates.     |
| `tests/integration/` | Deterministic generation and cross-boundary invariants.     |
| `tests/models/`      | Strict record and source-selection contracts.               |
| `tests/origin/`      | Origin labels, source preparation, contracts, and sampling. |
| `tests/release/`     | Packaging, policy, security, and verifier contracts.        |
| `tests/sampling/`    | Sparse-cell back-off and job-function allocation.           |
| `tests/sources/`     | Acquisition, archives, bundles, geography, and StatBank.    |
| `tests/validation/`  | Dataset validation contracts.                               |
| `tests/support/`     | Shared factories for synthetic bundles and source rows.     |

Tests must remain offline and must not call a provider. Mock HTTPX or the generation
client when testing LLM paths.

## Configuration

| Path                                 | Responsibility                                            |
| ------------------------------------ | --------------------------------------------------------- |
| `config/sources.yaml`                | Dynamic StatBank selectors, classifications, thresholds.  |
| `config/sources.lock.yaml`           | Resolved codes, queries, URLs, periods, and timestamps.   |
| `config/categories.yaml`             | Canonical demographic and labour-status mappings.         |
| `config/sampling.yaml`               | Seed, rows, adult age range, region, OCEAN settings.      |
| `config/validation.yaml`             | Distribution, expected-count, back-off, OCEAN thresholds. |
| `config/generation.yaml`             | Disabled endpoint, guards, response mode, prompt paths.   |
| `config/generation.local.yaml`       | Ignored local LLM override and provider settings.         |
| `config/prompts/attributes-da.md`    | Danish attributes schema and safety rules.                |
| `config/folk2-ieland-labels-da.yaml` | Archived official FOLK2 Danish 241-code label contract.   |
| `config/prompts/personas-da.md`      | Danish v4 persona writing brief and safety rules.         |

`config/sources.yaml` and `config/sources.lock.yaml` carry a top-level
`classifications:` list beside `sources:`, and their `version` is `2` to signal that
lock schema. Statistics Denmark publishes classifications as attachments on dst.dk
rather than through the StatBank data API, so they use `classification.py` instead of a
StatBank selector. `download_sources.py resolve` warns and rewrites a lock that predates
the current schema, and `download_sources.py fetch` fetches classifications as well as
tables.

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
need no secrets. Direct LLM commands do not load `.env`; use a short-lived shell export
or command-scoped assignment for the configured provider token. The Makefile includes
`.env` and exports all of its variables to subprocesses and hooks, so do not use it as
credential loading for direct LLM commands. `make install` is a convenience bootstrap
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
- `docker` runs the bootstrap, builds the image, and starts an interactive container. It
  requires Docker and has the bootstrap side effects.

## Pipeline ordering

Do not skip a boundary or call an LLM before the demographic gate passes:

1. Restore the committed raw-source archive with safe member validation for offline
   reproduction.
2. Build and validate the offline bundle in `data/processed`.
3. Generate deterministic demographic/OCEAN records.
4. Validate the smoke run, then generate and validate the statistical run.
5. Freeze the stratified text-development sample.
6. Dry-run LLM planning; it needs no provider, and only an approved operator may enable
   `--live`.
7. Validate each persona run; `generate_persona.py` emits one row, while each
   `build_dataset.py` shard is capped at five rows before merge and pilot validation.

The normal non-LLM stages are:

```bash
uv run src/scripts/restore_raw_sources.py
uv run src/scripts/build_distributions.py \
  --lock config/sources.lock.yaml --categories config/categories.yaml \
  --raw-dir data/raw-hardened-20260919 --output-dir data/processed
uv run src/scripts/validate_dataset.py sources --bundle "$BUNDLE"
uv run src/scripts/generate_demographics.py \
  --bundle "$BUNDLE" --rows 1000 --seed 20260914 \
  --output-dir data/runs/smoke
uv run src/scripts/validate_dataset.py demographics \
  --run "$RUN" --bundle "$BUNDLE"
```

Use the full copy-pasteable workflows in `README.md` to capture exact bundle/run paths,
produce the 100,000-row run, and freeze a sample. Run `download_sources.py resolve` and
`fetch` only for an intentional source refresh: both require network access to
Statistics Denmark, and refreshed responses create a new provenance chain. Review
changes to the lock and source register before accepting refreshed snapshots.

## Outputs and provenance

`data/raw-hardened-20260919.tar.zst` and `data/README.md` are tracked. The restore
script rejects empty archives and unsafe members, permits only regular files under the
expected root, and stages extraction before installing the ignored, immutable snapshots
into `data/raw-hardened-20260919/`. It does not compare the archive's top-level SHA-256
or validate snapshot manifests and checksums; bundle preparation validates each locked
snapshot's manifest, provenance, query, and file checksums.

Table snapshots sit under `<table>/<query-hash>/`. Classification snapshots sit under
`classifications/<classification-id>/<url-hash>/` and hold three files: `data.csv`,
`response-headers.json`, and `snapshot-manifest.json`. `build_raw_archive.py` repacks
the restored snapshots byte-stably, sorting members by archive path and fixing mode,
owner, and timestamp, so an unchanged snapshot tree always produces identical archive
bytes.

Prepared bundles contain normalised Parquet files, `bundle-manifest.json`, and source
preparation reports. `normalized/geography_hierarchy.parquet` holds the official region,
landsdel, and municipality hierarchy read from the `geography_hierarchy` classification.
Preparation cross-checks it against FOLK1A and RAS209 metadata and requires exact
equality of the locked, hierarchy, and prepared RAS209 municipality sets. Blank,
missing, duplicate, or mismatched hierarchy values fail. Municipality code and name
remain in generated records; landsdel stays inside the prepared bundle. Deterministic
runs contain `structured-records.parquet`, `run-manifest.json`, and JSON/Markdown
validation reports. Frozen samples have an adjacent `.manifest.json`. Persona runs
contain `generated-personas.parquet`, `generation-manifest.json`, `request-ledger.json`,
per-person attribute/final checkpoints, and `validation-report.json`. Current v4 outputs
retain only one short grounded `persona`
and do not contain the removed specialised fields or `visual_persona`. Pilots
additionally contain merged output, a pilot manifest, shard references, and
`pilot-validation-report.json`. Release packages use release manifest
schema 2 and evidence schema 2.

Manifests bind outputs to input/config/prompt/schema/validator checksums, row order,
and request accounting. The current versions are prepared bundle 6, sampler 6, frozen
sample 3, generation 4, validator `persona-safety-v16`, and release manifest/evidence 2.
Deterministic run IDs derive from bundle/config/row/seed inputs; LLM run IDs include the
frozen input and generation context. Existing checksum failures must fail loudly, not be
repaired by overwriting files. Do not document a canonical bundle or run ID until the
current contracts have been regenerated; use placeholders in instructions.

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

- The committed generation config is disabled. `generate_persona.py` performs
  validation and planning only; it does not need provider reachability. `--live`
  additionally requires local enablement, endpoint, model, provider reachability, and
  may spend money. The pilot has its own global request limit and requires current input
  and output prices; use zero only for a genuinely free endpoint. `--concurrency` can
  issue requests in parallel.
- The LLM input must be a frozen sample with a matching manifest and successful upstream
  demographic report. Checkpoints reject changed inputs, prompts, config, model, or
  validator context. Human-readable municipality, the official Danish
  `origin_country_da`, and job-function labels reach both provider stages; the origin
  code, English `origin_country`, resolution fields, and origin-contract metadata do
  not. The Danish label is also the exact origin fact in the grounded persona.
  Re-running a valid v4 live run resumes completed records, but v1-v3 checkpoints, the
  previous v2/v13 ten-person smoke, and old pilots are historical and not resumable
  under v4.
- The generation client records HTTP attempts before network I/O, retries only bounded
  transport/rate/server failures, and persists a request ledger. Accepted response
  metadata and hashes are retained; rejected completion text is not.
- LLM output must remain strict JSON, Danish, non-identifying, free of configured
  sensitive terms, and free of exact duplicate descriptions. The short persona must be
  grounded in its supplied facts, use a synthetic job title or current nonemployee
  status, avoid unsupported family claims, and treat OCEAN as cautious tendencies.
  Automated validation is not a substitute for blinded human review; downstream image
  models may still stereotype.
- The sampler backs off through ordered ladders when a conditional cell is missing, and
  each record records the level that produced it. A ladder stops at the most general
  cell that is still structurally valid, never a national one, so an age cannot leave
  its band and a detailed status cannot leave its broad RAS209 status. A cell missing at
  a ladder's final level is a structural zero and must keep failing loudly. Back-off
  consumes one random draw at any level, so reordering the draws or adding a ladder step
  changes every record for a given seed.
- Age and marital ladders may relax age or sex only within the same municipality; they
  never fall back to region or national geography. Education and broad status arrive
  together from the municipality-native RAS209 joint instead of from a ladder. RAS202
  detailed-status refinement is explicitly separate and national, but may never leave
  the sampled broad status. A missing terminal cell must fail loudly.
- Statistics Denmark tables are aggregates. Do not link them to people or infer
  individual records. Phase 2 retains official municipality fields; the municipality
  label may reach the LLM provider, but its code and resolution do not. The English
  `origin_country` remains official source/audit provenance, while the mandatory Danish
  `origin_country_da` comes from the archived official FOLK2 metadata-da 241-code
  contract (SHA-256 `f5c1f0a20f29372d6b222ce7a23cdc4ef0481d9e23fa6bd9b66b116e7adcb213`).
  Only the Danish label reaches both provider stages and exact persona grounding; the
  code, English label, resolutions, and contract metadata are withheld. Neither label is
  ethnicity, citizenship, residence, or appearance, and origin cannot drive culture,
  religion, job, interests, personality, or visual traits. Treat municipality-level
  combinations and provider payloads as restricted. Do not add names, addresses,
  occupations, employers, income, households, citizenship, ancestry, health, religion,
  sexuality, politics, criminal history, or other sensitive fields without a separate
  privacy review.
- Only lowercase `makefile` is tracked; case-insensitive systems may display it as
  `Makefile`. Make targets can mutate Git state; inspect `git status` before and after
  using them.
- `fix_dot_env_file.py` deletes `.name_and_email` after copying any values and does not
  populate optional token variables. Direct Python execution should still use `uv run`.

Read `docs/acceptance-criteria.md` and `docs/privacy-risk-register.md` before changing
validation or data fields. Report security issues privately as described in
`SECURITY.md`.
