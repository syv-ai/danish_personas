# Data

`raw-hardened-20260914.tar.zst` contains the immutable aggregate source snapshots used
to build the Danish persona demographic distributions. The archive contains only public
aggregate data; it contains no personal microdata.

Rebuild it after adding or refreshing a locked source with
`uv run src/scripts/build_raw_archive.py`, which packs the snapshots byte-stably.

Restore the snapshots with Python 3.14:

```bash
uv run src/scripts/restore_raw_sources.py
```

The command creates `data/raw-hardened-20260914/`. Derived source bundles, demographic
records, and LLM outputs remain ignored by Git and can be rebuilt using the main
[developer setup guide](../README.md#developer-setup-guide).

Archive SHA-256:

```text
047e33646e94a476fc4e17d55114526efb0135de4ce814947113ce79e852aadd
```

Source: Statistics Denmark. The included StatBank data may be reused under
[CC BY 4.0][licence]. This project further processes the source data; see the
[source register](../docs/source-register.md) for tables, reference periods,
transformations, and individual source checksums.

[licence]: https://www.dst.dk/en/presse/kildeangivelse

## Releases

Generated personas are ignored by Git, with one exception: `data/releases/` is tracked,
so a reviewed dataset can be shared exactly as generated. Language-model output is not
reproducible from its inputs, so committing the file is the only way for everyone to
work from the same personas.

Publish a completed pilot with:

```bash
uv run src/scripts/publish_release.py \
  --pilot data/persona-pilot/<pilot-id> \
  --name <release-name> \
  --reviewed-by "<who reviewed it>"
```

The command refuses a pilot whose own validation did not pass. It copies the merged
Parquet, the pilot manifest and the validation report, and writes a `release-manifest.json`
with the row count, the skipped identifiers, the reviewer and a SHA-256 for each file.

Publishing is a deliberate act. The privacy register requires human review before a
generated dataset leaves the machine, and the automated gates do not replace it.
