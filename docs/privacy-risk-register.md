# Privacy risk register

The current pipeline uses public aggregate tables only. It does not access or reconstruct
individual-level Statistics Denmark records.

| Risk | Current control | Residual risk |
| --- | --- | --- |
| Synthetic record mistaken for a real person | Dataset card and manifests identify every row as synthetic | Users may ignore documentation |
| Rare-cell reconstruction | Minimum source count, release-size pooling, and 99% coverage gate | Aggregate combinations may still appear distinctive |
| Geographic identification | Phase-2 records contain region only; exact addresses, CPR numbers, and municipality are excluded | Regional combinations can still be distinctive |
| Sensitive-attribute inference | Prompts prohibit inference; versioned validators reject configured health, religion, sexuality, ethnicity, political, and unsafe visual terms | A finite term list has false negatives; human review remains mandatory |
| Personality stereotyping | OCEAN is independent of demographics; prompts require probabilistic, non-deficit framing | Generated prose can still reintroduce associations or overstate traits |
| Source-response leakage | Only aggregate-derived records enter prompts; raw response envelopes and rejected completion text are not stored, while accepted content, usage metadata, and response hashes are checkpointed | Accepted generated text and checkpoints still require restricted handling before release |
| False statistical claims | Source universes, dates, proxies, pooling, and holdouts are explicit | The output is still a fitted synthetic model, not official microdata |
| Visual presentation becoming sensitive or identifying | `visual_persona` is required to use the closed Danish format and vocabulary for mutable clothing, accessories, colours, and a generic background; the validator rejects every other token, including sensitive, location, institution, and physical claims | Human review remains mandatory for implications outside the controlled format |
| Unauthorised LLM execution | Committed configuration is disabled; live calls require a local configuration and `--live`; invocations have row and HTTP-request caps | A user with repository and provider access can deliberately enable bounded calls |

## Prohibited Phase-2 fields

- names and contact information;
- exact addresses and coordinates;
- CPR or other administrative identifiers;
- occupation, employer, income, household, and housing details;
- ancestry, citizenship, and all special-category personal data;
- generated free text, including visual presentation that encodes sensitive or
  identifying traits.

Municipality aggregates are used only to construct official regional source counts.
Municipality fields are absent from generated Phase-2 records. A later release must pass
a separate privacy review before adding more detailed geography.
