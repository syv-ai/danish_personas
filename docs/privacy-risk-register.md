# Privacy risk register

<!-- markdownlint-disable MD060 -->

The current pipeline uses public aggregate tables only. It does not access or
reconstruct individual-level Statistics Denmark records.

| Risk                                        | Current control                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Residual risk                                                                                                                                                 |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Synthetic record mistaken for a real person | Dataset card and manifests identify every row as synthetic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Users may ignore documentation                                                                                                                                |
| Rare-cell reconstruction                    | Minimum source count, release-size pooling, and 99% coverage gate                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Aggregate combinations may still appear distinctive                                                                                                           |
| Geographic identification                   | Phase-2 records contain official municipality and region fields but exclude exact addresses, coordinates, and CPR numbers; the municipality label reaches the provider, while its code and resolution do not                                                                                                                                                                                                                                                                                                                                 | Municipality-level synthetic combinations can still be distinctive and require review before release                                                          |
| Sensitive-attribute inference               | Prompts prohibit inference and names; validator `persona-safety-v19` rejects configured health, religion, sexuality, ethnicity, and political terms. The English `origin_country` remains source/audit provenance, while mandatory Danish `origin_country_da` comes from the archived official FOLK2 metadata-da 241-code contract. Only the Danish label reaches the provider request and exact persona grounding; code, English label, resolutions, and contract metadata do not. Neither label is ethnicity, citizenship, residence, or appearance. | General term lists have false negatives; aggregate origin categories can still be misread, provider models can stereotype, and human review remains mandatory |
| Personality stereotyping                    | OCEAN is independent of demographics; prompts require cautious, probabilistic, non-deficit framing in the persona                                                                                                                                                                                                                                                                                                                                                                                                                      | Generated prose can still reintroduce associations or overstate traits; downstream image models may stereotype                                                |
| Source-response leakage                     | Only approved aggregate-derived fields enter prompts: municipality, the official Danish `origin_country_da`, sampled legal `marital_status`, and job-function labels are allowed. The origin code, English label, resolutions, and contract metadata are withheld. Raw response envelopes and rejected completion text are not stored, while accepted content, usage metadata, and response hashes are checkpointed                                                                                                                                                          | Accepted generated text, provider payloads, and checkpoints still require restricted handling before release                                                  |
| False statistical claims                    | Source universes, dates, proxies, pooling, and holdouts are explicit; job function is labelled as a sex-calibrated synthetic allocation from the incomplete LONS20 earnings-statistics universe, and any generated job title is synthetic                                                                                                                                                                                                                                                                                                    | The output is still a fitted synthetic model, not official microdata, observed occupation, or all-worker representation                                       |
| Synthetic biography mistaken for source data | The generation-4 persona preserves source-backed broad facts while prompts explicitly define names, detailed education, workplace setting, relationships, family, interests, and ambitions as synthetic elaboration. Surnames, real employers or institutions, exact addresses, former work, and appearance remain prohibited. | Readers may still mistake plausible fictional details for observed Statistics Denmark data, so dataset documentation and blinded review remain mandatory |
| Visual or image-model stereotyping          | `visual_persona` is not part of the active schema. The generation-4 persona is not a visual description, but downstream image models may infer or stereotype from ordinary text or origin labels                                                                                                                                                                                                                                                                                                                                                                   | Image generation is outside this contract and requires separate human review                                                                                  |
| Unauthorised LLM execution                  | The Hydra config defaults to a local endpoint; invocations have row and HTTP-request caps, and credentials must come from an environment variable                                                                                                                                                                                                                                                                                                                                                                                                       | A user with repository and provider access can deliberately enable bounded calls                                                                              |
| Premature publication                       | Hugging Face upload accepts only a packaged and independently verified release after an enabled policy and blinded human-review attestation; upload creates a pull request rather than publishing directly                                                                                                                                                                                                                                                                                                                                   | Reviewer authenticity is not cryptographically established, and a repository maintainer can still merge an unsafe pull request                                |
| Checksum-tolerant single-persona command    | `generate_persona.py` alone may continue after stored checksum mismatches, but retains schema, content, provenance, safety, grounding, and accounting gates; dataset, release, and default library paths remain strict                                                                                                                                                                                                                                                                                                                              | Altered artefacts can reach a provider request or local output, so command evidence requires review and must not be treated as release verification     |

<!-- markdownlint-enable MD060 -->

## Prohibited Phase-2 fields

- surnames and contact information;
- exact addresses and coordinates;
- CPR or other administrative identifiers;
- real employer or institution names, income, exact housing details, and exact
  occupations outside the approved synthetic LONS20 title allowlist;
- inferred ancestry, citizenship, and all special-category personal data (the official
  FOLK2 origin category is not any of these);
- generated free text that encodes sensitive or identifying traits, exact addresses,
  real organisations, surnames, or visual appearance.

FOLK2's official IELAND categories are sampled as a national independent marginal and
retained verbatim, including Stateless and Not stated. The English `origin_country` is
retained as official source/audit provenance. Mandatory `origin_country_da` is resolved
from the archived official FOLK2 metadata-da 241-code contract, source metadata SHA-256
`f5c1f0a20f29372d6b222ce7a23cdc4ef0481d9e23fa6bd9b66b116e7adcb213`. Only the Danish
label reaches the provider request and exact persona grounding; code, English label,
resolutions, and contract metadata are withheld. Neither label is ethnicity,
citizenship, residence, or appearance. Origin cannot drive culture, religion, job,
interests, personality, or visual traits. Both labels remain in upstream outputs and
input/checkpoint hashes.

The pooled secondary/vocational education category is a temporary source-backed broad
level. Persona prose may choose one fictional branch and add a fictional direction and
town for narrative specificity, but this must not be interpreted as observed source
data. Real institution names remain prohibited.

LONS20 covers all public employees but only private organisations with at least 10
full-time-equivalent employees; smaller private organisations and other
earnings-statistics exclusions are absent. The broad job function is allocated only to
RAS202 employee codes 15, 20, 25, 30, 35, and 40, conditioned on sex alone. It is not an
observed occupation and must not drive other fields. Self-employed people, assisting
spouses, and all non-employee statuses receive null job-function fields. The official
job-function label reaches the provider so it can ground a short synthetic job title;
the job-function code and resolution do not. The function remains bound into sample and
checkpoint hashes.

Municipality aggregates remain municipality-keyed throughout Phase 2, and generated
records retain the official municipality code and name. Region is only its hierarchy
parent. The municipality name reaches the provider as a human-readable grounding label;
the municipality code and resolution do not. Any release of the structured fields still
requires the geographic-identification review recorded above.
