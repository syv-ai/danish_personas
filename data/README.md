# Data

`raw-hardened-20260918.tar.zst` contains the immutable aggregate source snapshots used
to build the Danish persona demographic distributions. It holds the six Statistics
Denmark StatBank table snapshots and the official Statistics Denmark classification
snapshot that supplies the region, landsdel, and municipality hierarchy. FOLK2 supplies
an official national marginal of country-of-origin categories for adults. It is sampled
independently into `origin_country_code` and `origin_country` in generated records; the
fields are withheld from LLM payloads. Origin is not ethnicity, citizenship, or
residence and cannot drive language, culture, religion, occupation, personality, or
appearance. The archive contains only public aggregate data; it contains no personal
microdata. It contains 39 files: six table snapshots and one classification snapshot.
RAS209 is locked to all 99 official level-3 areas (including Christiansø) and retains
its municipality x education x status x age-band x sex joint. Restore the snapshots
with Python 3.14:

```bash
uv run src/scripts/restore_raw_sources.py
```

The command creates `data/raw-hardened-20260918/`. Derived source bundles, demographic
records, and LLM outputs remain ignored by Git and can be rebuilt using the main
[developer setup guide](../README.md#developer-setup-guide).

The canonical offline source bundle is `a276e45eb987fb73` (prepared-bundle schema 3).
It passed source validation. The prior bundle is not reusable; demographic runs must
be rebuilt from this new source chain. The separate frozen text-development sample
remains 1,000 rows by design.

Archive SHA-256 (reproducible from the restored snapshot tree):

```text
7361a6e199d9977c63478cc167f89383c3288e4f247219c10056882a7f31ac99
```

The new archive contains 39 files and is 1,490,022 bytes on disk; its RAS209 CSV is
23,962,725 bytes. The archive is a new immutable chain; the 20260917 archive is not
overwritten.

Source: Statistics Denmark. The included StatBank data and classification may be reused
under [CC BY 4.0][licence]. This project further processes the source data; see the
[source register](../docs/source-register.md) for tables, classifications, reference
periods, transformations, and individual source checksums.

[licence]: https://www.dst.dk/en/presse/kildeangivelse
