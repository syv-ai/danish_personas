# Source register

Retrieved through the official Statistics Denmark StatBank API on 18 September 2026.
The exact dimension selections are frozen in `config/sources.lock.yaml`. The immutable
snapshots are committed as `data/raw-hardened-20260918.tar.zst` under CC BY 4.0; its
SHA-256 is `7361a6e199d9977c63478cc167f89383c3288e4f247219c10056882a7f31ac99`.
Restoration creates content-addressed query subdirectories under
`data/raw-hardened-20260918/`; the previous 20260917 chain is retained unchanged.

| Table | Period | Pipeline role | Selected observations | API cells | Raw bytes | CSV SHA-256 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| [FOLK2][folk2] | 2025 | Adult national origin marginal (official IELAND) | 312,336 | 2,186,352 | 2,980,413 | `5842fa32fc4b333e2955697db87f3d595940c8a015688119d8f0eea7c01cc7cf` |
| [FOLK1A][folk1a] | 2025Q1 | Exact age, sex, municipality, marital status | 87,120 | 522,720 | 5,637,970 | `fc70f900e628c169660c354f487723d4556e707e49ef95a96fad3d63b01c4741` |
| [RAS209][ras209] | 2024 | Municipality joint education and labour status | 807,840 | 5,654,880 | 23,962,725 | `8db475275c36062a20307673565b57ebae6738d46f8e7cbcf9dc878cff2d0c67` |
| [RAS202][ras202] | 2024 | Detailed status by exact age and sex | 3,672 | 18,360 | 231,540 | `9d5293ed0979a245c990a4177b35f3c5877ac34011d7f978138ceded0226ca86` |
| [BEFOLK3][befolk3] | 2025 | Held-out population validation | 21,384 | 106,920 | 1,008,139 | `b6e72015815cae98a484059c0261c00ffc0597a600ecc4ee5d74b9819cd7acd9` |
| [RAS210][ras210] | 2024 | Held-out status-by-municipality validation | 32,076 | 192,456 | 2,072,366 | `d1eca1c04b11fba402a3cc18d0c44503dbadc2bbcb289170aa09e0e0dd66cefa` |

For each table, the snapshot also contains English and Danish metadata, the exact POST
query, response headers, and a machine-readable checksum manifest. Selected observations
are the product of selected values; API cells multiply that count by every selected
dimension plus the observation-value column. FOLK2 and RAS209 use StatBank's BULK
streaming response because their 2,186,352 and 5,654,880 actual cells exceed the
one-million limit for non-streaming formats. BULK is explicitly exempt from that limit
and is streamed to disk in bounded chunks. StatBank omits zero-count BULK rows;
the 31 omitted IELAND partitions in this snapshot are recorded explicitly as reviewed
`expected_zero_codes` in `config/sources.yaml` and its resolved lock. Preparation
materialises only those approved omissions as explicit unsuppressed zeroes; any other
missing selected code fails the source gate. The stored CSV is canonical UTF-8 with LF
line endings.

## Classifications

Statistics Denmark publishes its classifications as attachments on `dst.dk` rather than
through the StatBank data API, so they have no reference period or POST query and are
acquired by a separate adapter. The attachment below is a semicolon-delimited CSV served
behind a redirect, retrieved on 17 September 2026 and frozen in
`config/sources.lock.yaml` alongside the tables.

| Classification | Valid from | Pipeline role | Raw bytes | CSV SHA-256 |
| --- | --- | --- | ---: | --- |
| [Regioner, landsdele og kommuner][nuts] (`NUTS_V1_2007_DK`) | 2007-01-01 | Official region, landsdel, and municipality hierarchy | 5,748 | `67a193164777e61552daae0d52cb59bc4b589d16cc696365fc6c51a31f92f55f` |

Statistics Denmark marks this classification as still valid. The snapshot contains the
attachment CSV, the response headers, and a machine-readable checksum manifest.

## Harmonisation decisions

- FOLK2 selects ages 18-125, both sexes, all three HERKOMST values, both STATSB
  values, all 241 official IELAND values, and 2025. Its 312,336 selected observations
  produce 2,186,352 API cells and are aggregated across the selected age, sex,
  HERKOMST, and STATSB dimensions into a national marginal; the resulting official
  IELAND category is not interpreted as ethnicity or citizenship.
- FOLK2 preserves unequal official weights and the complete selected official
  code-to-label mapping, including Stateless and Not stated. Preparation requires unique
  codes and labels and rejects mapping changes. The raw BULK partition remains available
  to provenance checks before approved zero omissions are materialised. This marginal is
  not ethnicity or citizenship, and no country groups, correlations, or joint
  associations are inferred.
- FOLK2 is sampled independently as a national marginal into the Phase 2
  `origin_country_code` and `origin_country` fields. Official unequal weights and labels,
  including Stateless and Not stated, are retained; zero-weight categories are excluded.
  Origin is withheld from both LLM payloads and cannot drive language, culture, religion,
  occupation, personality, or visual appearance. It is not ethnicity, citizenship, or
  residence.
- FOLK1A 2025Q1 is the closest demographic snapshot to the November 2024 RAS data.
- FOLK1A ages 16-19 estimate the age-18-and-over share of RAS209's 16-19 band. Ages 16
  and 17 are excluded from generated records.
- RAS209 is acquired for all 99 official level-3 areas, including Christiansø (`411`),
  and the raw and unpooled prepared joint retains municipality, education, status,
  age-band, and sex. The pooled person-sampling artefact also retains municipality
  keys; it never aggregates those rows by region.
- RAS209 age is pooled to 18-29, 30-49, 50-66, and 67+.
- RAS209 age and education pooling is performed within each municipality. The
  municipality-level sampling artefact retains the complete adjusted source universe;
  sparse cells are not silently replaced by a regional aggregate.
- The RAS209 `67+` education category is explicitly labelled as a proxy for generated
  people aged 70 and over.
- RAS202 may refine a detailed status only within the broad RAS209 status already
  sampled.
- BEFOLK3 and RAS210 remain held-out aggregate diagnostics. They are never treated as
  linked observations or personal microdata.
- The municipality-to-region map comes from the `NUTS_V1_2007_DK` classification rather
  than from positional inference over FOLK1A's StatBank metadata value list. The
  classification's codes are byte-identical to StatBank's `OMRÅDE` dimension ids, and it
  contributes 5 regions, 11 landsdele, and 99 level-3 areas: Denmark's 98 municipalities
  plus Christiansø (`411`). FOLK1A also includes code `411`, and municipality code and
  name are retained in generated records.
- Source preparation cross-checks the classification against the maps derived from both
  FOLK1A and RAS209 metadata and requires exact equality of the locked, hierarchy, and
  prepared RAS209 municipality sets. It fails on any disagreement, missing or extra
  municipality, duplicate code, null, blank code, blank title, or blank parent. On this
  chain the check reports 99 level-3 areas, 11 landsdele, 5 regions, and zero
  disagreements. All prepared names
  and region parents come from the validated classification lookup.
- The hierarchy is written to `normalized/geography_hierarchy.parquet` in the prepared
  bundle with `municipality_code`, `municipality`, `landsdel_code`, `landsdel`,
  `region_code`, and `region`. Landsdel is carried in the prepared bundle only; it is
  not added to generated records.

## Terms

Statistics Denmark's public StatBank API is free to access. Its open data may be freely
reused commercially and non-commercially under CC BY 4.0 with source attribution. The
same terms apply to the published classification attachment. This project further
processes the data. See [Statistics Denmark's source-attribution guidance][terms].

[folk2]: https://www.statbank.dk/FOLK2
[folk1a]: https://www.statbank.dk/FOLK1A
[ras209]: https://www.statbank.dk/RAS209
[ras202]: https://www.statbank.dk/RAS202
[befolk3]: https://www.statbank.dk/BEFOLK3
[ras210]: https://www.statbank.dk/RAS210
[nuts]: https://www.dst.dk/da/Statistik/dokumentation/nomenklaturer/nuts
[terms]: https://www.dst.dk/en/presse/kildeangivelse
