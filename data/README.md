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
