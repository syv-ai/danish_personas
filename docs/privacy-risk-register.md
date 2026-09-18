# Privacy risk register

The current pipeline uses public aggregate tables only. It does not access or reconstruct
individual-level Statistics Denmark records.

| Risk | Current control | Residual risk |
| --- | --- | --- |
| Synthetic record mistaken for a real person | Dataset card and manifests identify every row as synthetic | Users may ignore documentation |
| Rare-cell reconstruction | Minimum source count, release-size pooling, and 99% coverage gate | Aggregate combinations may still appear distinctive |
| Geographic identification | Phase-2 records contain official municipality and region fields but exclude exact addresses, coordinates, and CPR numbers; municipality is withheld from LLM payloads | Municipality-level synthetic combinations can still be distinctive and require review before release |
| Sensitive-attribute inference | Prompts prohibit inference; versioned validators reject configured health, religion, sexuality, ethnicity, and political terms; visual guidance uses a controlled non-sensitive format; official FOLK2 origin is sampled independently, retained in outputs and hashes, and withheld from LLM payloads | General term lists have false negatives; aggregate origin categories can still be misread as ethnicity or citizenship, and human review remains mandatory |
| Personality stereotyping | OCEAN is independent of demographics; prompts require probabilistic, non-deficit framing | Generated prose can still reintroduce associations or overstate traits |
| Source-response leakage | Only aggregate-derived records enter prompts; raw response envelopes and rejected completion text are not stored, while accepted content, usage metadata, and response hashes are checkpointed | Accepted generated text and checkpoints still require restricted handling before release |
| False statistical claims | Source universes, dates, proxies, pooling, and holdouts are explicit; job function is labelled as a sex-calibrated synthetic allocation from the incomplete LONS20 earnings-statistics universe | The output is still a fitted synthetic model, not official microdata, observed occupation, or all-worker representation |
| Visual presentation becoming sensitive or identifying | `visual_persona` is required to use the closed Danish format and vocabulary for mutable clothing, accessories, colours, and a generic background; the validator rejects every other token, including sensitive, location, institution, and physical claims | Human review remains mandatory for implications outside the controlled format |
| Unauthorised LLM execution | Committed configuration is disabled; live calls require a local configuration and `--live`; invocations have row and HTTP-request caps | A user with repository and provider access can deliberately enable bounded calls |
| Premature publication | Release eligibility requires a disabled-capable policy and a blinded human-review attestation bound to the output | Packaging, publication, and reviewer authenticity are not implemented or cryptographically established |

## Prohibited Phase-2 fields

- names and contact information;
- exact addresses and coordinates;
- CPR or other administrative identifiers;
- exact occupation, employer, income, household, and housing details (the approved broad
  synthetic LONS20 job-function allocation is the sole occupation-related exception);
- inferred ancestry, citizenship, and all special-category personal data (the official
  FOLK2 origin category is not any of these);
- generated free text, including visual presentation that encodes sensitive or
  identifying traits.

FOLK2's official IELAND categories are sampled as a national independent marginal and
retained verbatim, including Stateless and Not stated. They are not treated as
ethnicity, citizenship, or residence. Origin cannot drive language, culture, religion,
occupation, personality, or visual appearance. The two origin fields are withheld from
both LLM stages, while remaining in upstream outputs and input/checkpoint hashes.

LONS20 covers all public employees but only private organisations with at least 10
full-time-equivalent employees; smaller private organisations and other
earnings-statistics exclusions are absent. The broad job function is allocated only to
RAS202 employee codes 15, 20, 25, 30, 35, and 40, conditioned on sex alone. It is not an
observed occupation and must not drive other fields. Self-employed people, assisting
spouses, and all non-employee statuses receive null job-function fields. Job function is
withheld from both LLM stages but remains bound into sample and checkpoint hashes.

Municipality aggregates remain municipality-keyed throughout Phase 2, and generated
records retain the official municipality code and name. Region is only its hierarchy
parent. Municipality is withheld from both LLM stages, and any release of the structured
fields still requires the geographic-identification review recorded above.
