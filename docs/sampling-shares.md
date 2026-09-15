# Sampling shares

Every value in a generated record, where it comes from, and how often it occurs. Figures
are the 100,000-row run `89d6c3caaf8cb629` from bundle `ce70ae87f78d564a`, seed
`20260914`. Regenerate them with `uv run personas summary`.

## Where each field comes from

| Field | Source | Conditioned on |
| --- | --- | --- |
| `sex`, `age_band`, `region`, `education_level`, `labour_market_status` | RAS209 | Drawn together as one joint quota sample |
| `age` | FOLK1A | `age_band`, `sex` |
| `marital_status` | FOLK1A | `region`, `age_band`, `sex` |
| `origin` | FOLK1E | `region`, `age_band`, `sex` |
| `origin_region` | FOLK1C + `config/origin-regions.yaml` | `sex`, `origin`; Danish origin maps to `danmark` |
| `detailed_status` | RAS202 | `age_band`, `sex`, `labour_market_status` |
| OCEAN scores and labels | `config/sampling.yaml` | Nothing; five independent normal draws |
| `persona_id`, `country`, `education_resolution`, source codes | Derived | Computed from the drawn values |
| Attributes and the seven persona texts | Language model | The whole record as JSON |

Rare cells are dropped before sampling: each source frame is filtered by a release-count
floor, so a combination too small to publish cannot be drawn.

## Shares

### Sex

| Value | Share |
| --- | ---: |
| female | 50.7 % |
| male | 49.3 % |

### Age

Mean 49.8, median 50. Adults only, 18 and over.

| Band | Share |
| --- | ---: |
| 30-49 | 30.7 % |
| 50-66 | 27.2 % |
| 67+ | 23.0 % |
| 18-29 | 19.1 % |

### Marital status

| Value | Share |
| --- | ---: |
| married_or_separated | 44.8 % |
| never_married | 37.7 % |
| divorced | 11.6 % |
| widowed | 5.8 % |

### Region

| Value | Share |
| --- | ---: |
| Region Hovedstaden | 32.3 % |
| Region Midtjylland | 22.8 % |
| Region Syddanmark | 20.7 % |
| Region Sjælland | 14.3 % |
| Region Nordjylland | 9.9 % |

### Origin

Statistics Denmark's official ancestry categories. They are administrative categories,
not ethnicity.

| Value | Share |
| --- | ---: |
| danish_origin | 83.4 % |
| immigrant_non_western | 8.4 % |
| immigrant_western | 6.0 % |
| descendant_non_western | 2.0 % |
| descendant_western | 0.2 % |

### Origin region

FOLK1C's country mix, grouped by `config/origin-regions.yaml`. The country is sampled
internally and never emitted.

| Value | Share of all records | Share of non-Danish records |
| --- | ---: | ---: |
| danmark | 83.4 % | - |
| vesteuropa_og_eu | 5.9 % | 35.7 % |
| mellemosten_og_nordafrika | 4.5 % | 27.2 % |
| asien | 2.3 % | 14.0 % |
| europa_uden_for_eu | 2.1 % | 12.5 % |
| afrika_syd_for_sahara | 0.9 % | 5.7 % |
| latinamerika_og_caribien | 0.5 % | 2.8 % |
| nordamerika_og_oceanien | 0.3 % | 2.1 % |
| ovrige | 0.0 % | 0.1 % |

### Education

Broad RAS209 attainment. The `67+` band is a documented proxy for ages 70 and over.

| Value | Share |
| --- | ---: |
| secondary_or_vocational | 40.2 % |
| higher_education | 37.1 % |
| primary | 21.4 % |
| not_stated | 1.3 % |

### Labour-market status

| Value | Share |
| --- | ---: |
| employed | 62.2 % |
| retired | 26.3 % |
| other | 6.0 % |
| student | 3.5 % |
| unemployed | 2.0 % |

### Detailed status

The eight most frequent of RAS202's detailed categories; the remaining categories cover
15.6 %.

| Value | Share |
| --- | ---: |
| Employees - basic level | 23.0 % |
| Old-age pension | 20.2 % |
| Employees - upper level | 16.8 % |
| Employees - medium level | 6.5 % |
| Other employees | 5.1 % |
| Disability pension | 4.6 % |
| Employees, not specified | 4.6 % |
| Self-employed | 3.6 % |

### OCEAN

Not Danish statistics. Each trait is an independent draw from a normal distribution with
mean 50 and standard deviation 10, clipped to [20, 80], and labelled at the boundaries
35, 45, 55 and 65. All five means land on 50.0 by construction.

## Validation

`uv run personas demographics` compares the generated marginals against the prepared
source frames and fails the run if any exceeds the total-variation thresholds in
`config/validation.yaml`. The thresholds are calibrated for the statistical row count;
smaller runs fail them. See [`source-register.md`](source-register.md) for the tables,
periods, and checksums behind each source.
