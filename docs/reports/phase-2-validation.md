# Phase 2 validation report

The canonical deterministic sampler schema is version 5 and the prepared-bundle schema
is version 5. Both municipality-native runs were regenerated offline from the
20260919 archive with seed `20260914`, with no LLM calls and no threshold changes.

## Regeneration commands

```bash
rm -rf data/regeneration-20260919
mkdir -p data/regeneration-20260919
uv run src/scripts/restore_raw_sources.py \
  --archive data/raw-hardened-20260919.tar.zst \
  --output-dir data/regeneration-20260919
uv run src/scripts/build_distributions.py \
  --lock config/sources.lock.yaml \
  --categories config/categories.yaml \
  --raw-dir data/regeneration-20260919/raw-hardened-20260919 \
  --output-dir data/regeneration-20260919/processed
uv run src/scripts/validate_dataset.py sources \
  --bundle data/regeneration-20260919/processed/cfc1b56f5586a2d7
uv run src/scripts/generate_demographics.py \
  --bundle data/regeneration-20260919/processed/cfc1b56f5586a2d7 \
  --config config/sampling.yaml --rows 2000 --seed 20260914 \
  --output-dir data/regeneration-20260919/runs/smoke
uv run src/scripts/validate_dataset.py demographics \
  --run data/regeneration-20260919/runs/smoke/f449f1de01d18c08 \
  --bundle data/regeneration-20260919/processed/cfc1b56f5586a2d7
uv run src/scripts/generate_demographics.py \
  --bundle data/regeneration-20260919/processed/cfc1b56f5586a2d7 \
  --config config/sampling.yaml --rows 100000 --seed 20260914 \
  --output-dir data/regeneration-20260919/runs/statistical
uv run src/scripts/validate_dataset.py demographics \
  --run data/regeneration-20260919/runs/statistical/3ebc00282c621ef2 \
  --bundle data/regeneration-20260919/processed/cfc1b56f5586a2d7
```

## Artefacts

- Source archive: `data/raw-hardened-20260919.tar.zst` (45 files; seven official
  tables and one official classification snapshot).
- Archive SHA-256:
  `4b3715e193a5efb22001390d98bde53ab54b6049d710e06498e54afb8e94cda9`
- Archive size: 1,510,378 bytes.
- Source bundle: `cfc1b56f5586a2d7` (prepared-bundle schema 5; **PASS**).
- Bundle manifest SHA-256:
  `1fa7fba08b17a11ff03616b9953b3d4fb1d6acbced6bf284ae656e8ba20df7e8`
- Sampling-config SHA-256:
  `696fec4e5369fcf91b010f5a111819a638f2e6f32e503941a1e4e976b379e825`
- Validation-config SHA-256:
  `1a1a8847ef8efa0c26f31258d5fbd87fe184ba1835cef5b39567e9564569991e`
- LONS20 contract SHA-256:
  `d37f75fccaedbeb1b970a72c09722bf9fb195e2b41e143b926a659ede2443733`
- Smoke run: `f449f1de01d18c08` (2,000 rows; **PASS**).
- Smoke Parquet SHA-256:
  `64059a786346f797742b3a4bd4f077569edc6335c6f84aea4d6380254eb51c05`
- Smoke logical-content SHA-256:
  `d53d7d6000e18efee14259036c23df1f24da14762e6e123fa943b6226bd90ad1`
- Statistical run: `3ebc00282c621ef2` (100,000 rows; **PASS**).
- Statistical Parquet SHA-256:
  `691d1a90b9a704f13af68cab7ef804f5e0fd6d75949e0e412630f8d2f00c8320`
- Statistical logical-content SHA-256:
  `28201b12b916b1fffc8b7bcfcb98bdb4dbcce1744385eb7f63efea3b880add35`
- Both run manifests record sampler schema 5, seed `20260914`, and exactly zero LLM
  calls.
- The frozen text-development input remains a separate 1,000-row local artefact. Sample
  provenance schema 2 uses municipality, education, and labour status as strata.

The compressed raw snapshots are committed with attribution. Restored and generated
artefacts remain ignored by Git and are reproducible from the source archive, lock,
mappings, configurations, and code.

## Source result

- Forty-five archive members were restored and checksummed: seven official tables and
  one official classification snapshot.
- LONS20 contains exactly 42 two-digit DISCO-08 groups per sex: 881,774 women and
  919,819 men in its incomplete earnings-statistics universe.
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
| LONS20 job function TV, women | 0.9350% | 0.0186% | 5% / 2% |
| LONS20 job function TV, men | 0.9535% | 0.0181% | 5% / 2% |
| Job-function mapping/eligibility errors | 0 | 0 | 0 |
| Municipality RAS209 joint TV | 49.2167% | 8.9276% | info / 10% |
| Held-out municipality population TV | 21.8453% | 2.9804% | 25% / 5% |
| Held-out status-by-sex TV | 1.8680% | 0.3806% | 25% / 5% |
| Age back-off | 0% | 0% | 1% |
| Marital back-off | 0% | 0% | 1% |
| Detailed-status back-off | 0% | 0.003% | 1% |
| Maximum OCEAN correlation | 0.040965 | 0.004863 | info / 0.02 |
| LLM calls | 0 | 0 | 0 |

The 2,000-row municipality joint has more populated source cells than observations,
so its joint TV is explicitly informational at smoke size. Its marginal, hierarchy,
and held-out municipality gates are mandatory. Every mandatory gate passed in both
runs. Overall Phase 2 result: **PASS**.

The machine-readable reports remain beside the local source bundle and generated runs
while the regeneration root is present locally; all generated artefacts are ignored by
Git and are removed after verification.
