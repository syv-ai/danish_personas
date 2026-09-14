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

The validated local statistical run contains 100,000 records. LLM-generated attributes
and persona descriptions are deliberately deferred.

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

## Outputs

The ignored `data/` directory contains:

- raw StatBank CSV and metadata snapshots with SHA-256 manifests;
- normalized and pooled Parquet sampling tables;
- source and demographic validation reports in JSON and Markdown;
- deterministic Parquet records and run manifests;
- a 1,000-row stratified seed set for the future LLM phase.

The source lock, category mappings, sampling parameters, validation thresholds, and code
are version controlled. Large generated artefacts remain local and reproducible.

## Safety boundary

`config/generation.yaml` keeps LLM generation disabled. Running
`src/scripts/generate_personas.py` fails before contacting any provider. Exact addresses,
CPR numbers, names, occupation, ancestry, citizenship, income, household information,
and sensitive traits are not generated in the implemented scope.

## Validation

See:

- [`docs/acceptance-criteria.md`](docs/acceptance-criteria.md);
- [`docs/source-register.md`](docs/source-register.md);
- [`docs/privacy-risk-register.md`](docs/privacy-risk-register.md);
- [`docs/reports/phase-2-validation.md`](docs/reports/phase-2-validation.md).
