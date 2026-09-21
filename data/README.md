# Data

`raw-hardened-20260919.tar.zst` contains the immutable aggregate source snapshots used
to build the Danish persona demographic distributions. It holds seven Statistics Denmark
StatBank table snapshots and the official Statistics Denmark classification snapshot
that supplies the region, landsdel, and municipality hierarchy. The archive contains
only public aggregate data and no personal microdata. The three public persona scripts restore its 45 files automatically when the standard
input workflow needs them. The same operation remains available as the importable
`danish_personas.sources.archive.restore_raw_sources` maintenance service. It creates
`data/raw-hardened-20260919/`. The merged 20260918 source chain is retained unchanged.
Derived source bundles, records, and LLM outputs remain ignored by Git and can be rebuilt
using the main [developer setup guide](../README.md#developer-setup-guide).

LONS20 supplies 84 cells: exactly 42 two-digit DISCO-08 job-function groups for each of
women and men. Its official 2024 counts total 881,774 women and 919,819 men. The
selected universe is `LØNMÅL=ANTAL`, `SEKTOR=1000`, `AFLOEN=TIFA`, `LONGRP=LTOT`, and
M/K. LONS20 covers all public employees and private organisations with at least 10
full-time-equivalent employees; smaller private organisations and other documented
earnings-statistics exclusions are absent. Generated job function is therefore a
synthetic sex-conditional allocation for eligible employees, not observed occupation or
all-worker representation. The fields are withheld from LLM payloads.

FOLK2 remains an independent national marginal of official IELAND country-of-origin
categories for adults. The English `origin_country` is retained as official source and
audit provenance. Mandatory `origin_country_da` is the Danish display label from the
archived official FOLK2 `metadata-da` 241-code contract at
`config/folk2-ieland-labels-da.yaml`, whose source metadata SHA-256 is
`f5c1f0a20f29372d6b222ce7a23cdc4ef0481d9e23fa6bd9b66b116e7adcb213`. Only the Danish
label reaches the provider request and exact persona grounding; code, English label,
resolutions, and contract metadata are withheld. Neither label is ethnicity,
citizenship, residence, or appearance. Origin cannot drive culture, religion, job,
interests, personality, or visual traits. RAS209 remains locked to all 99 official
level-3 areas, including Christiansø.

The current offline source bundle is `8a4133e5a0a52050` (prepared-bundle schema 6).
Its passing runs are `f4dffe214a2faf0b` (2,000 rows) and `55fb89fb303a67f0`
(100,000 rows), both using sampler schema 6. The statistical run's adjacent 1,000-row
sample uses frozen-sample schema 3 and SHA-256
`354616ddcf601e5a02f03a0a3b09b8711f97f314bf4c300c9172ceb5b9ca8629`.
Generation contract 4, validator `persona-safety-v18`, and release manifest/evidence
schema 2 are current. The former schema-5 bundle `cfc1b56f5586a2d7` and runs
`f449f1de01d18c08` / `3ebc00282c621ef2` are historical and non-resumable.

Archive SHA-256 (reproducible from the restored snapshot tree):

```text
4b3715e193a5efb22001390d98bde53ab54b6049d710e06498e54afb8e94cda9
```

The archive is 1,510,378 bytes. Source: Statistics Denmark. The included StatBank data
and classification may be reused under [CC BY 4.0][licence]. This project further
processes the source data; see the [source register](../docs/source-register.md) for
selections, reference periods, transformations, and checksums.

[licence]: https://www.dst.dk/en/presse/kildeangivelse
