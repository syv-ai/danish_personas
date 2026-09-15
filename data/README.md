# Data

`raw-hardened-20260914.tar.zst` contains the immutable aggregate source snapshots used
to build the Danish persona demographic distributions. The archive contains only public
aggregate data; it contains no personal microdata.

Restore the snapshots with Python 3.14:

```bash
uv run src/scripts/restore_raw_sources.py
```

The command creates `data/raw-hardened-20260914/`. Derived source bundles, demographic
records, and LLM outputs remain ignored by Git and can be rebuilt using the main
[reproduction instructions](../README.md#reproduce-the-non-llm-pipeline).

Archive SHA-256:

```text
0d00761c284c616d66a5871d3418cc565b1bfe9879f24665119fdf7dcc4c014f
```

Source: Statistics Denmark. The included StatBank data may be reused under
[CC BY 4.0][licence]. This project further processes the source data; see the
[source register](../docs/source-register.md) for tables, reference periods,
transformations, and individual source checksums.

[licence]: https://www.dst.dk/en/presse/kildeangivelse
