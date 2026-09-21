# Script interface

Run the three public scripts directly from the repository root with `uv run`:

```text
src/scripts/
├── generate_persona.py  # one validated persona on stdout
├── build_dataset.py     # validated dataset with progress on stderr
└── fix_dot_env_file.py  # bootstrap local environment and Git identity
```

Use `uv run src/scripts/<name>.py --help` for exact options. The persona scripts
prepare and validate the standard deterministic source bundle, smoke run, statistical
run, and 1,000-row frozen sample automatically when `--input` is omitted. Existing
content-addressed artefacts are reused. Supplying `--input` uses its adjacent
`.manifest.json` file.

Source acquisition, archive handling, deterministic generation, sample freezing, and
validation remain importable maintenance services. They are not public scripts.

## Persona commands

`generate_persona.py` validates the upstream frozen sample and generated run, samples
one frozen demographic locally, starts a fresh model request, and writes only the final
Danish persona plus a newline to stdout. It loads Hydra `config/config.yaml` by default
and runs immediately. Direct invocations do not reuse earlier persona checkpoints.
Generated evidence is stored below `data/personas` by default. Diagnostics use stderr.

`build_dataset.py` requires `--rows`, `--request-limit`, and the two current token-price
values. It generates validated shards of five rows, resumes valid checkpoints, merges
and validates the complete dataset, stores it below `data/persona-datasets` by default,
displays a `tqdm` row progress bar on stderr, and writes only the merged Parquet path to
stdout. Concurrency is configurable; shard size is fixed at five and batch delay is
zero.

Neither command loads `.env`. All model settings live in `config/config.yaml`. Set
`api_key_env` to an environment-variable name if the provider requires authentication,
and provide that variable only for the command invocation. Never store token values in
the configuration. Commands can consume paid provider requests.

## Hugging Face upload

`build_dataset.py --hf-repo OWNER/DATASET` additionally requires `--attestation`,
`--policy`, `--dataset-card`, and `--licence`. The script packages the pilot below
`<output-dir>/releases`, uses the current working directory as repository root,
independently verifies the package against its manifest digest, and only then uploads it
to a Hugging Face dataset pull request.

Authentication uses `HF_TOKEN` or standard cached Hugging Face credentials. The script
has no token option and does not log credentials. A normal workflow therefore generates
and reviews the dataset first, then repeats the resumable command with the upload and
release options.
