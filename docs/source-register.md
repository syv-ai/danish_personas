# Source register

Retrieved through the official Statistics Denmark StatBank API on 14 September 2026.
The exact dimension selections are frozen in `config/sources.lock.yaml`. The immutable
snapshots are committed as `data/raw-hardened-20260914.tar.zst` under CC BY 4.0; its
SHA-256 is `0d00761c284c616d66a5871d3418cc565b1bfe9879f24665119fdf7dcc4c014f`.
Restoration creates content-addressed query subdirectories under
`data/raw-hardened-20260914/`.

| Table | Period | Pipeline role | Raw bytes | CSV SHA-256 |
| --- | --- | --- | ---: | --- |
| [FOLK1A][folk1a] | 2025Q1 | Exact age, sex, municipality, marital status | 5,637,970 | `fc70f900e628c169660c354f487723d4556e707e49ef95a96fad3d63b01c4741` |
| [RAS209][ras209] | 2024 | Joint broad education and labour status | 4,995,164 | `f0422d7ae0d2c33bd19f62a4647db036530f38b9e4ba945f408fbc75aeb418ca` |
| [RAS202][ras202] | 2024 | Detailed status by exact age and sex | 231,540 | `9d5293ed0979a245c990a4177b35f3c5877ac34011d7f978138ceded0226ca86` |
| [BEFOLK3][befolk3] | 2025 | Held-out population validation | 1,008,139 | `b6e72015815cae98a484059c0261c00ffc0597a600ecc4ee5d74b9819cd7acd9` |
| [RAS210][ras210] | 2024 | Held-out status-by-municipality validation | 2,072,366 | `d1eca1c04b11fba402a3cc18d0c44503dbadc2bbcb289170aa09e0e0dd66cefa` |

For each table, the snapshot also contains English and Danish metadata, the exact POST
query, response headers, and a machine-readable checksum manifest.

## Harmonisation decisions

- FOLK1A 2025Q1 is the closest demographic snapshot to the November 2024 RAS data.
- FOLK1A ages 16-19 estimate the age-18-and-over share of RAS209's 16-19 band. Ages 16
  and 17 are excluded from generated records.
- RAS209 education is pooled to primary, secondary or vocational, higher education, and
  not stated.
- RAS209 age is pooled to 18-29, 30-49, 50-66, and 67+.
- Calibration cells must contain at least 50 source persons and have an expected count
  of at least five in a 100,000-row release. The resulting 531-cell table retains
  99.6439% of the adjusted RAS209 population.
- The RAS209 `67+` education category is explicitly labelled as a proxy for generated
  people aged 70 and over.
- RAS202 may refine a detailed status only within the broad RAS209 status already
  sampled.
- BEFOLK3 and RAS210 remain held-out aggregate diagnostics. They are never treated as
  linked observations or personal microdata.

## Terms

Statistics Denmark's public StatBank API is free to access. Its open data may be freely
reused commercially and non-commercially under CC BY 4.0 with source attribution. This
project further processes the data. See [Statistics Denmark's source-attribution
guidance][terms].

[folk1a]: https://www.statbank.dk/FOLK1A
[ras209]: https://www.statbank.dk/RAS209
[ras202]: https://www.statbank.dk/RAS202
[befolk3]: https://www.statbank.dk/BEFOLK3
[ras210]: https://www.statbank.dk/RAS210
[terms]: https://www.dst.dk/en/presse/kildeangivelse
