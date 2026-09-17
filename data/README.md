# Data

`raw-hardened-20260917.tar.zst` contains the immutable aggregate source snapshots used
to build the Danish persona demographic distributions. It holds the six Statistics
Denmark StatBank table snapshots and the official Statistics Denmark classification
snapshot that supplies the region, landsdel, and municipality hierarchy. FOLK2 supplies
an official national marginal of country-of-origin categories for adults. It is sampled
independently into `origin_country_code` and `origin_country` in generated records; the
fields are withheld from LLM payloads. Origin is not ethnicity, citizenship, or residence
and cannot drive language, culture, religion, occupation, personality, or appearance.
The archive contains only public aggregate data; it
contains no personal microdata. It contains 39 files: six table snapshots and one
classification snapshot. Restore the snapshots with Python 3.14:

```bash
uv run src/scripts/restore_raw_sources.py
```

The command creates `data/raw-hardened-20260917/`. Derived source bundles, demographic
records, and LLM outputs remain ignored by Git and can be rebuilt using the main
[developer setup guide](../README.md#developer-setup-guide).

Archive SHA-256 (reproducible from the restored snapshot tree):

```text
480d1aedec2f583849b1d5a9eec801180c9fe1140c0ec4c1333d23dd66192e8a
```

Source: Statistics Denmark. The included StatBank data and classification may be reused
under [CC BY 4.0][licence]. This project further processes the source data; see the
[source register](../docs/source-register.md) for tables, classifications, reference
periods, transformations, and individual source checksums.

[licence]: https://www.dst.dk/en/presse/kildeangivelse
