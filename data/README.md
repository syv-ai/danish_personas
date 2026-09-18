# Data

`raw-hardened-20260919.tar.zst` contains the immutable aggregate source snapshots used
to build the Danish persona demographic distributions. It holds seven Statistics
Denmark StatBank table snapshots and the official Statistics Denmark classification
snapshot that supplies the region, landsdel, and municipality hierarchy. The archive
contains only public aggregate data and no personal microdata. Restore its 45 files with
Python 3.14:

```bash
uv run src/scripts/restore_raw_sources.py
```

The command creates `data/raw-hardened-20260919/`. The merged 20260918 source chain is
retained unchanged. Derived source bundles, records, and LLM outputs remain ignored by
Git and can be rebuilt using the main
[developer setup guide](../README.md#developer-setup-guide).

LONS20 supplies 84 cells: exactly 42 two-digit DISCO-08 job-function groups for each of
women and men. Its official 2024 counts total 881,774 women and 919,819 men. The selected
universe is `LØNMÅL=ANTAL`, `SEKTOR=1000`, `AFLOEN=TIFA`, `LONGRP=LTOT`, and M/K.
LONS20 covers all public employees and private organisations with at least 10
full-time-equivalent employees; smaller private organisations and other documented
earnings-statistics exclusions are absent. Generated job function is therefore a
synthetic sex-conditional allocation for eligible employees, not observed occupation or
all-worker representation. The fields are withheld from LLM payloads.

FOLK2 remains an independent national marginal of official country-of-origin categories
for adults. Origin is not ethnicity, citizenship, or residence and cannot drive
language, culture, religion, job function, personality, or appearance. RAS209 remains
locked to all 99 official level-3 areas, including Christiansø.

The canonical offline source bundle is `333a0a166ca0e030` (prepared-bundle schema 4).
It passed source validation. Sampler schema 5 and validation schema 5 produced passing
runs `c7387bc90e8896d5` (2,000 rows) and `6c288bd32f96e1f1` (100,000 rows).

Archive SHA-256 (reproducible from the restored snapshot tree):

```text
4b3715e193a5efb22001390d98bde53ab54b6049d710e06498e54afb8e94cda9
```

The archive is 1,510,378 bytes. Source: Statistics Denmark. The included StatBank data
and classification may be reused under [CC BY 4.0][licence]. This project further
processes the source data; see the [source register](../docs/source-register.md) for
selections, reference periods, transformations, and checksums.

[licence]: https://www.dst.dk/en/presse/kildeangivelse
