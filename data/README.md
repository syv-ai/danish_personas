# Data

`raw-hardened-20260917.tar.zst` contains the immutable aggregate source snapshots used
to build the Danish persona demographic distributions. It holds the five Statistics
Denmark StatBank table snapshots and the official Statistics Denmark classification
snapshot that supplies the region, landsdel, and municipality hierarchy. The archive
contains only public aggregate data; it contains no personal microdata.

Restore the snapshots with Python 3.14:

```bash
uv run src/scripts/restore_raw_sources.py
```

The command creates `data/raw-hardened-20260917/`. Derived source bundles, demographic
records, and LLM outputs remain ignored by Git and can be rebuilt using the main
[developer setup guide](../README.md#developer-setup-guide).

Archive SHA-256:

```text
48ae0befbcfe78474162c343b5926b882e86ce3cc109db658675e7da42199a00
```

Source: Statistics Denmark. The included StatBank data and classification may be reused
under [CC BY 4.0][licence]. This project further processes the source data; see the
[source register](../docs/source-register.md) for tables, classifications, reference
periods, transformations, and individual source checksums.

[licence]: https://www.dst.dk/en/presse/kildeangivelse
