# Privacy risk register

The current pipeline uses public aggregate tables only. It does not access or reconstruct
individual-level Statistics Denmark records.

| Risk | Current control | Residual risk |
| --- | --- | --- |
| Synthetic record mistaken for a real person | Dataset card and manifests identify every row as synthetic | Users may ignore documentation |
| Rare-cell reconstruction | Minimum source count, release-size pooling, and 99% coverage gate | Aggregate combinations may still appear distinctive |
| Geographic identification | Exact addresses and CPR numbers are prohibited; municipality is internal-only by default | Small municipalities remain more distinctive than regions |
| Sensitive-attribute inference | No ancestry, citizenship, ethnicity, religion, health, sexuality, political belief, or criminal-history field | Future LLM prose could imply these traits and requires a separate gate |
| Personality stereotyping | OCEAN is sampled independently of demographics | Downstream text generation could reintroduce associations |
| Source-response leakage | Only aggregate counts enter sampling; raw model responses do not exist in Phases 0-2 | Future LLM operations require restricted response handling |
| False statistical claims | Source universes, dates, proxies, pooling, and holdouts are explicit | The output is still a fitted synthetic model, not official microdata |
| Unauthorised LLM execution | Configuration disables generation and the placeholder command exits before provider access | A future code change could enable it and must receive review |

## Prohibited Phase-2 fields

- names and contact information;
- exact addresses and coordinates;
- CPR or other administrative identifiers;
- occupation, employer, income, household, and housing details;
- ancestry, citizenship, and all special-category personal data;
- generated free text.

Municipality fields are retained only in local structured seeds for calibration and
review. The intended public projection is region-level unless a later privacy review
explicitly approves more detailed geography.
