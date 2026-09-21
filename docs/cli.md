# Unified CLI

The installed `danish-personas` command exposes the package services without shelling
out to legacy scripts:

```text
sources restore | pack | prepare | resolve | fetch
validate sources | demographics | personas | pilot
demographics run
sample freeze
personas shard | pilot
workflow deterministic --target smoke|statistical [--raw-parent PATH]
release package | verify
```

The deterministic workflow defaults to the committed raw archive and versioned
configuration. It restores, prepares, validates, generates, and validates in order,
passing each service's returned path to the next boundary. `--raw-parent` defaults to
`data`; the restored and prepared path is always its fixed `raw-hardened-20260919`
child. `statistical` runs the smoke stage first, then generates the configured
statistical row count and freezes 1,000 rows by default. Use `--sample-rows` to change
that development sample size. The frozen Parquet file and its manifest remain inside the
returned content-addressed statistical run directory, beside `run-manifest.json` and
`validation-report.json`; the workflow's final stdout line is the exact frozen-sample
path accepted by `personas shard`.

Source `resolve` and `fetch` refuse to run without explicit `--network`. Persona shards
are dry runs unless `--live` is supplied; pilots always require `--live`. The command
never loads `.env`, and deterministic workflows make no network or LLM requests.

Release packaging is offline and has no upload, authentication, or token options. A
current package must bind release manifest schema 2 and evidence schema 2, including the
current generation-4, validator-v16, origin-label, and artefact checksums. Do not reuse
historical release IDs or invent new canonical IDs: package first, then copy the printed
IDs and digests into release evidence. Retain `release-manifest.sha256` (or the printed
digest) externally and pass it to `release verify` after relocating the package. After
verification and an external digest check, an operator may upload manually. Before
current regeneration, use `<new-release-id>`, `<new-pilot-id>`, and `<new-digest>` as
documentation placeholders rather than historical values:

```bash
hf upload OWNER/DATASET RELEASE_DIR --type dataset --create-pr
```

Uploading is deliberately not implemented by Python; never upload before verification.
The existing `uv run src/scripts/*.py` commands remain supported for compatibility. Run
`uv run danish-personas --help` or append `--help` to any group for all options.
