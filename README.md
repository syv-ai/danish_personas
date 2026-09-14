# Danish Personas

A reproducible pipeline for creating statistically grounded Danish synthetic persona
seeds from public aggregate data. The implemented scope contains no LLM calls and no
personal microdata.

The research and delivery design is documented in
[`docs/danish-personas-plan.md`](docs/danish-personas-plan.md).

## Current status

Phases 0-2 are implemented and validated:

1. typed codebase, configuration, commands, tests, and an LLM execution guard;
2. immutable acquisition and preparation of five Statistics Denmark tables;
3. deterministic generation and validation of demographic and OCEAN records.

The validated local statistical run contains 100,000 records. Phase-3 infrastructure now
supports guarded, resumable two-stage generation of structured attributes and six Danish
persona descriptions. A three-record smoke test passed; development-sample generation,
human evaluation, model selection, and any release-scale generation remain deferred. A
separate two-record Hugging Face Gemma experiment measured a low token-cost floor but
failed manual language and grounding review.

## Setup

```bash
uv sync
make check
uv run pytest
```

Python 3.14 or later is required.

## Reproduce the non-LLM pipeline

Resolve source selectors into an explicit, idempotent lock:

```bash
uv run src/scripts/download_sources.py resolve \
  --config config/sources.yaml \
  --lock config/sources.lock.yaml
```

Fetch immutable raw snapshots:

```bash
uv run src/scripts/download_sources.py fetch \
  --lock config/sources.lock.yaml \
  --raw-dir data/raw
```

Prepare and validate the offline source bundle:

```bash
uv run src/scripts/build_distributions.py \
  --lock config/sources.lock.yaml \
  --categories config/categories.yaml \
  --raw-dir data/raw \
  --output-dir data/processed

BUNDLE=$(ls -td data/processed/* | head -1)
uv run src/scripts/validate_dataset.py sources --bundle "$BUNDLE"
```

Generate and validate the smoke run before the statistical run:

```bash
uv run src/scripts/generate_demographics.py \
  --bundle "$BUNDLE" \
  --rows 1000 \
  --seed 20260914 \
  --output-dir data/runs/smoke

SMOKE=$(ls -td data/runs/smoke/* | head -1)
uv run src/scripts/validate_dataset.py demographics \
  --run "$SMOKE" \
  --bundle "$BUNDLE"

uv run src/scripts/generate_demographics.py \
  --bundle "$BUNDLE" \
  --rows 100000 \
  --seed 20260914 \
  --output-dir data/runs/statistical

RUN=$(ls -td data/runs/statistical/* | head -1)
uv run src/scripts/validate_dataset.py demographics \
  --run "$RUN" \
  --bundle "$BUNDLE"
```

Freeze the deterministic Phase-3 development input without calling an LLM:

```bash
uv run src/scripts/freeze_demographic_sample.py \
  --run "$RUN" \
  --rows 1000 \
  --output "$RUN/text-development-seeds.parquet"
```

## Run a guarded LLM smoke test

Copy `config/generation.yaml` to the Git-ignored
`config/generation.local.yaml`. Set `llm_generation_enabled: true`, `base_url`, and
`model` in the local copy. If the endpoint needs a bearer token, set `api_key_env` to
the environment variable containing it.

Always run without `--live` first. This verifies the upstream validation report,
sample checksum, configuration, prompt files, and hard five-row limit without making
network requests:

```bash
uv run src/scripts/generate_personas.py \
  --input "$RUN/text-development-seeds.parquet" \
  --sample-manifest "$RUN/text-development-seeds.manifest.json" \
  --config config/generation.local.yaml \
  --output-dir data/persona-smoke \
  --rows 3
```

Add `--live` to authorise the planned requests explicitly. Validate the resulting run:

```bash
PERSONA_RUN=$(ls -td data/persona-smoke/* | head -1)
uv run src/scripts/validate_dataset.py personas --run "$PERSONA_RUN"
```

A repeated command resumes from per-record checkpoints and does not call the model for
completed records.

## Outputs

The ignored `data/` directory contains:

- raw StatBank CSV and metadata snapshots with SHA-256 manifests;
- normalized and pooled Parquet sampling tables;
- source and demographic validation reports in JSON and Markdown;
- deterministic Parquet records and run manifests;
- a 1,000-row stratified seed set for LLM development;
- local persona smoke outputs, checkpoints, provenance manifests, and validation reports.

The source lock, category mappings, sampling parameters, validation thresholds, and code
are version controlled. Large generated artefacts remain local and reproducible.

## Safety boundary

`config/generation.yaml` keeps live LLM generation disabled. Enabling it requires both a
local configuration and the explicit `--live` flag. The pipeline checks the frozen-input
checksum and successful Phase-2 report before contacting a provider, and never permits
more than five rows per invocation. Exact addresses, CPR numbers, names, occupation,
ancestry, citizenship, income, household information, and sensitive traits are not
generated in the implemented scope. Deterministic validators reject contact details,
sensitive terms, non-Danish output, duplicate descriptions, schema violations, checksum
changes, and modifications to upstream fields.

## Validation

See:

- [`docs/acceptance-criteria.md`](docs/acceptance-criteria.md);
- [`docs/source-register.md`](docs/source-register.md);
- [`docs/privacy-risk-register.md`](docs/privacy-risk-register.md);
- [`docs/reports/phase-2-validation.md`](docs/reports/phase-2-validation.md);
- [`docs/reports/phase-3-smoke.md`](docs/reports/phase-3-smoke.md);
- [`docs/reports/hf-gemma-cost-smoke.md`](docs/reports/hf-gemma-cost-smoke.md).
