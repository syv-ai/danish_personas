# Script interface

Run the four public scripts directly from the repository root with `uv run`:

```text
src/scripts/
├── generate_persona.py         # one validated persona on stdout
├── build_dataset.py            # validated dataset with progress on stderr
├── build_persona_dashboard.py  # self-contained offline HTML dashboard
└── fix_dot_env_file.py         # bootstrap local environment and Git identity
```

The first three scripts use Hydra and the shared entry configuration
`config/config.yaml`. Pass settings with Hydra override syntax; for example,
`llm.model=MODEL`, `build_dataset.rows=10`, or
`persona_dashboard.input=PATH`. Use `--help` to inspect the composed configuration.
Hydra does not change the working directory or create run, log, or `.hydra` artefacts.
`fix_dot_env_file.py` retains its Click options.

The persona scripts prepare and validate the standard deterministic source bundle,
smoke run, statistical run, and 1,000-row frozen sample automatically when their
`input` setting is null. Existing content-addressed artefacts are reused. Supplying an
input uses its adjacent `.manifest.json` file.

Source acquisition, archive handling, deterministic generation, sample freezing, and
validation remain importable maintenance services. They are not public scripts.

## Persona commands

`generate_persona.py` validates the upstream frozen sample and generated run, samples
one frozen demographic locally, starts a fresh model request, and writes only the final
Danish persona plus a newline to stdout. It loads Hydra `config/config.yaml` by default
and runs immediately. Direct invocations do not reuse earlier persona checkpoints.
Generated evidence is stored below `data/personas` by default. Diagnostics use stderr.
This command deliberately uses an explicit checksum-tolerant validation policy: stored
checksum mismatches alone do not block the command, while schema, content, provenance,
safety, grounding, and accounting checks remain active. Library validation,
`build_dataset.py`, and release verification remain checksum-strict by default; this
exception increases the risk that altered artefacts are used and requires review of the
stored evidence before reuse. An alternative invocation is:

```bash
uv run src/scripts/generate_persona.py \
  generate_persona.input=data/sample.parquet \
  generate_persona.output_dir=data/personas \
  llm.model=MODEL
```

`build_dataset.py` requires rows, a global request limit, and the two current token
prices. It generates validated shards of five rows, resumes valid checkpoints, merges
and validates the complete dataset, displays a `tqdm` row progress bar on stderr, and
writes only the merged Parquet path to stdout:

```bash
uv run src/scripts/build_dataset.py \
  build_dataset.rows=10 \
  build_dataset.concurrency=1 \
  build_dataset.request_limit=30 \
  build_dataset.input_price_per_million=0.25 \
  build_dataset.output_price_per_million=2.00
```

Both commands load all variables from the repository-root `.env` before Hydra starts;
existing process variables take precedence and values are never logged. Shared model
settings live under `llm` in `config/config.yaml`. Set `llm.api_key_env` to an
environment-variable name if the provider requires authentication. The
`llm.same_sex_partner_probability` setting is bounded to `[0, 1]` and defaults to
`0.00701`; it is the 2026 FAM100N share among legally formalised couples only:
(PARS 10,180 + RP 4,376) / (PARS + RP + PARF 2,060,670). This formalised-couple
proxy is applied to all generated partnered personas. DST cannot identify most
unmarried same-sex couples, so it is not an estimate of the all-partnership rate or
sexual orientation. Never store token values in configuration. Commands can consume
paid provider requests.

Each persona command writes a deterministic, flat effective `GenerationConfig` YAML
snapshot below its output area's `generation-configs` directory. The snapshot captures
resolved Hydra overrides for generation and provenance, contains no token value, is
reused when identical, and is never overwritten when its content differs. Its stable
format header deliberately versions the snapshot bytes: pilots created before the
Hydra migration from the old flat `config/config.yaml` format remain separate
historical artefacts and are not resumable under the current snapshot contract.
Adding the partner-target configuration field and policy binding also intentionally
invalidates earlier generation-v5 contexts and checkpoints; they are historical and
cannot be resumed under the current contract.

## Persona dashboard

Build an offline, self-contained dashboard from a generated Parquet file and its
prepared source bundle:

```bash
uv run src/scripts/build_persona_dashboard.py \
  persona_dashboard.input=data/personas/generated-personas.parquet \
  persona_dashboard.bundle=data/processed/BUNDLE_ID \
  persona_dashboard.output=data/personas/dashboard.html
```

The selected input and bundle must exist. The command validates the selected dashboard
configuration, keeps Plotly inline, and writes only the output HTML path to stdout.

## Hugging Face upload

Setting `build_dataset.hf_repo=OWNER/DATASET` additionally requires
`build_dataset.attestation`, `build_dataset.policy`, `build_dataset.dataset_card`, and
`build_dataset.licence`. These are validated before sample preparation or provider
requests. The script packages the pilot below `<output-dir>/releases`, independently
verifies the package against its manifest digest, and only then uploads it to a Hugging
Face dataset pull request.

Authentication uses `HF_TOKEN` or standard cached Hugging Face credentials. The script
has no token setting and does not log credentials. A normal workflow therefore
generates and reviews the dataset first, then repeats the resumable command with the
upload and release settings.
