# Phase 2 validation report

The canonical deterministic sampler schema is version 4 and the prepared-bundle schema
is version 3. Both municipality-native runs were regenerated offline with seed
`20260914`, with no LLM calls.

## Artefacts

- Source bundle: `a276e45eb987fb73`
- Sampling-config SHA-256:
  `aaba1c0ec2d338fe2020b2946eaaeb77e1c89d4bd85e1b63b46b280845b8a033`
- Validation-config SHA-256:
  `c744ecbfe9d68516c8c1feaf2fff7b9482543bc9f713191f2713b42e1a5493df`
- Smoke run: `11a191f044aa245a` (2,000 rows; **PASS**)
- Smoke Parquet SHA-256:
  `3b102cb5980e690fa4486921abcafb7fea65e9a75c34f0d6e2c3086fbfc37fa2`
- Smoke logical-content SHA-256:
  `761c610ce854eb6839303c43ad402a093fc501ee1512054b6e4a27a431c6f8cb`
- Statistical run: `cd12e81f3de71f0d` (100,000 rows; **PASS**)
- Statistical Parquet SHA-256:
  `e49dea3332fbbf0edf8728f9ab5d280e3251108f43aeabb7b6c62cb2734c1654`
- Statistical logical-content SHA-256:
  `c847bb9d8d68c46c33c881e1693b75a485b6ebd620220c33cc39e49a15644cb2`
- Frozen text-development input remains a separate 1,000-row local artefact. Sample
  provenance schema 2 uses municipality, education, and labour status as strata.

The compressed raw snapshots are committed with attribution. Restored and generated
datasets remain ignored by Git and are reproducible from the source archive, lock,
mappings, configurations, and code.

## Source result

- Six official tables and one official classification snapshot were checksummed.
- The lock, hierarchy, and prepared RAS209 joint contain the same 99 municipalities.
- FOLK1A and RAS209 remain municipality-keyed; no regional reaggregation is used by the
  sampler.
- RAS202 remains a separate national detailed-status refinement.
- Shared integrity, schema, and successful-preparation verification passed.
- Source validation: **PASS**.

## Deterministic run results

| Metric | 2,000-row smoke | 100,000-row statistical | Threshold |
| --- | ---: | ---: | ---: |
| Rows | 2,000 | 100,000 | exact |
| Schema errors | 0 | 0 | 0 |
| Municipality hierarchy errors | 0 | 0 | 0 |
| Maximum fitted marginal TV | 1.5148% | 0.2903% | 5% / 2% |
| Municipality marginal TV | 0.6321% | 0.0137% | 5% / 2% |
| FOLK2 origin marginal TV | 0.8430% | 0.0256% | 5% / 2% |
| Municipality RAS209 joint TV | 49.2167% | 8.9276% | info / 10% |
| Held-out municipality population TV | 21.8453% | 2.9804% | 25% / 5% |
| Held-out status-by-sex TV | 1.8680% | 0.3806% | 25% / 5% |
| Age back-off | 0% | 0% | 1% |
| Marital back-off | 0% | 0% | 1% |
| Detailed-status back-off | 0% | 0.003% | 1% |
| Maximum OCEAN correlation | 0.040965 | 0.004863 | info / 0.02 |
| LLM calls | 0 | 0 | 0 |

The 2,000-row municipality joint has more populated source cells than observations, so
its joint TV is explicitly informational at smoke size. Its marginal, hierarchy, and
held-out municipality gates are mandatory. Every mandatory gate passed in both runs.
Overall Phase 2 result: **PASS**.

The machine-readable reports remain beside the local source bundle and generated runs.
