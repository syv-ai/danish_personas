# Script interface

Run scripts directly from the repository root with `uv run`:

```text
src/scripts/
├── generate_persona.py          # one validated persona on stdout
├── build_dataset.py             # validated dataset with tqdm progress
├── restore_raw_sources.py       # restore the pinned offline source archive
├── build_raw_archive.py         # reproducibly repack source snapshots
├── download_sources.py          # intentionally refresh locked sources
├── build_distributions.py       # prepare the offline distribution bundle
├── generate_demographics.py     # generate deterministic demographic records
├── freeze_demographic_sample.py # freeze the LLM input sample
├── validate_dataset.py          # run source, demographic, or persona gates
└── fix_dot_env_file.py          # bootstrap local environment and Git identity
```

Use `uv run src/scripts/<name>.py --help` for exact options. The source,
demographic, freeze, and validation commands are retained because they provide the
clean-clone reproducibility path; they are not one-off migrations.

## Persona commands

`generate_persona.py` validates the upstream frozen sample and generated run, stores
resumable evidence below `data/personas` by default, and writes only the final Danish
persona plus a newline to stdout. It loads Hydra `config/config.yaml` by default and
runs immediately. When `--offset` is omitted, it samples one frozen demographic locally
and logs the offset on stderr; pass `--offset N` to reproduce or resume that selection.
Diagnostics use stderr.

`build_dataset.py` requires `--rows`. It generates validated shards of at most five
rows, resumes valid checkpoints, merges and validates
the complete dataset, stores it below `data/persona-datasets` by default, displays a
`tqdm` row progress bar on stderr, and writes only the merged Parquet path to stdout.
The request budget, prices, concurrency, input sample, and output paths are explicit CLI
options.

Neither command loads `.env`. All model settings live in `config/config.yaml`. Set
`api_key_env` to an environment-variable name if the provider requires authentication,
and provide that variable only for the command invocation. Never store token values in
the configuration. Commands can consume paid provider requests.

## Hugging Face upload

`build_dataset.py --hf-repo OWNER/DATASET` does not upload raw pilot files. It requires
an enabled release policy, blinded-review attestation, dataset card, licence, repository
root, and release output directory. The script packages the pilot with the existing
release gates, independently verifies that package against its manifest digest, and only
then uploads the verified directory to a Hugging Face dataset pull request.

Authentication uses `HF_TOKEN` or standard cached Hugging Face credentials. The script
has no token option and does not log credentials. A normal workflow therefore generates
and reviews the dataset first, then repeats the resumable command with the upload and
release options.
