# Hugging Face model comparison

## Scope

This is a historical generation-contract v1 comparison. Its retained outputs and metrics
are not resumable under generation contract v2 and do not establish the current v2
quality gate.

Four candidate models generated the same five frozen demographic records using the same
revised prompts, strict schemas, and two-stage process. The records cover all five
Danish regions and labour statuses, ages 23-73, both sexes, all four education
categories, and varied OCEAN labels. Each model ran in batches of 2, 2, and 1 records,
with no batch permitted more than five HTTP attempts.

All Qwen models used `enable_thinking: false`. Providers were pinned so
structured-output support and price were known before generation.

## Candidate selection

| Model                                               | Provider  | Rationale                                                       |
| --------------------------------------------------- | --------- | --------------------------------------------------------------- |
| `Qwen/Qwen3.5-397B-A17B`                            | DeepInfra | Highest expected multilingual and instruction-following quality |
| `Qwen/Qwen3-235B-A22B-Instruct-2507`                | Novita    | Strong explicit Danish support at low cost                      |
| `Qwen/Qwen3.6-35B-A3B`                              | DeepInfra | Compact, efficient comparison                                   |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | Novita    | Non-Qwen baseline                                               |

`CohereLabs/command-a-plus-05-2026-w4a4` was shortlisted for its explicit Danish
training, but the account had exhausted its monthly model-specific request allowance.
`openai/gpt-oss-120b` returned no content within a 64-token schema probe and was
excluded rather than introducing a separate reasoning configuration.

## Operational results

Costs include all successful accepted and rejected responses retained in checkpoints and
use the provider prices listed by the HF router at experiment time.

| Rank label | Model          | Requests | Retries | Tokens |      Cost | Cost/persona | Mean latency |
| ---------- | -------------- | -------: | ------: | -----: | --------: | -----------: | -----------: |
| C          | Qwen 397B      |       11 |       1 | 14,814 | $0.018845 |    $0.003769 |      12.64 s |
| A          | Qwen 235B      |       10 |       0 | 14,794 | $0.003865 |    $0.000773 |      26.37 s |
| D          | Qwen 35B       |       11 |       1 | 13,920 | $0.005075 |    $0.001015 |       5.35 s |
| B          | Llama Maverick |       12 |       2 | 13,517 | $0.006029 |    $0.001206 |       5.51 s |

Linear cost estimates from this five-record sample are:

| Model          | 1,000 rows | 10,000 rows | 100,000 rows |
| -------------- | ---------: | ----------: | -----------: |
| Qwen 397B      |      $3.77 |      $37.69 |      $376.90 |
| Qwen 235B      |      $0.77 |       $7.73 |       $77.31 |
| Qwen 35B       |      $1.02 |      $10.15 |      $101.50 |
| Llama Maverick |      $1.21 |      $12.06 |      $120.58 |

These are planning estimates, not quotes. A larger run must use p95 accepted-record cost
and include regeneration rates.

## Blinded review

Two independent reviewers examined labels A-D without model identities or cost data.
Both produced the same ranking: **C > A > B > D**.

The language reviewer scored fluency, domain relevance, consistency, and usefulness. The
safety reviewer scored grounding, stereotype avoidance, diversity, and personality
calibration.

| Model          | Language average | Safety average | Reviewer gate                                |
| -------------- | ---------------: | -------------: | -------------------------------------------- |
| Qwen 397B      |           4.35/5 |         2.75/5 | Best; language pass, strict safety near-fail |
| Qwen 235B      |           3.65/5 |         2.75/5 | Conditional language pass; safety fail       |
| Llama Maverick |           3.35/5 |         2.50/5 | Fail                                         |
| Qwen 35B       |           2.35/5 |         1.25/5 | Fail; regenerate rather than edit            |

### Qwen 397B

This was the clear winner for fluent, specific, coherent Danish. It still repeated
quiet, local, traditional activities and sometimes inferred detailed habits from weak
evidence. The strict reviewer found demographic coupling around age, employment,
education, and low extraversion. It is the only candidate close to a pilot-quality gate.

### Qwen 235B

This model produced the strongest cross-record variety at about one fifth of Qwen 397B's
cost. It also made more grammatical errors, contradicted retired status in one record,
introduced unsupported activities, and inserted a Cyrillic `е` into a Danish word. It is
the strongest economical challenger but not release-ready.

### Llama Maverick

Llama was fast and generally fluent, but too generic and weakly grounded. It repeated
cycling, gardening, local activities, and practical skills, misread average openness in
multiple records, and invented culinary or professional backgrounds. It does not justify
further testing with the current prompts.

### Qwen 35B

The compact model produced pervasive malformed language, unsupported claims,
deterministic personality statements, and Chinese/Cyrillic fragments. It performed worse
than the larger models while costing more than Qwen 235B. It should be rejected.

## Automated-validator correction

All outputs initially passed automated checks. Blinded review found Cyrillic and Chinese
fragments in Qwen 235B and Qwen 35B output. The validator now rejects Cyrillic and Han
characters. Revalidation results are:

- Qwen 397B: three of three runs pass;
- Qwen 235B: two pass, one fails;
- Llama Maverick: three pass;
- Qwen 35B: one passes, two fail.

This demonstrates that automated language classification is necessary but not
sufficient. Human review remains mandatory.

## Recommendation

Advance **Qwen 397B** as the quality leader and **Qwen 235B** as the cost challenger.
Reject Llama Maverick and Qwen 35B for this task. Before a 10,000-row pilot, revise the
prompt and validators around cross-record repetition, unsupported domain claims,
combined marital-status interpretation, and demographic/OCEAN stereotyping, then conduct
a blinded 20-50-record head-to-head between the two finalists.
