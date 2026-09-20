# Hugging Face DeepSeek comparison

## Scope

This is a historical generation-contract v1 comparison. Its retained outputs and metrics
are not resumable under generation contract 3; the experiment is evidence about the
listed models and must not be treated as current generation-3 validation. It predates
the mandatory Danish `origin_country_da` provider boundary and validator
`persona-safety-v14`.

`deepseek-ai/DeepSeek-V4.1-Flash` and `deepseek-ai/DeepSeek-V4-Flash-0731` generated the
same five frozen records used in the previous four-model comparison. Each model ran
through Baseten in batches of 2, 2, and 1 records, with five HTTP attempts permitted per
batch.

Baseten supports strict structured output for both models. Initial V4.1 generation
returned null content after consuming the output budget with reasoning. Setting
`reasoning_effort: none` made both models operationally comparable to the no-thinking
Qwen runs.

## Operational results

| Model                  | Requests | Retries | Tokens |      Cost | Cost/persona | Mean latency |
| ---------------------- | -------: | ------: | -----: | --------: | -----------: | -----------: |
| DeepSeek V4.1 Flash    |       11 |       1 | 13,716 | $0.008022 |    $0.001604 |       4.89 s |
| DeepSeek V4 Flash 0731 |       11 |       1 | 15,176 | $0.002634 |    $0.000527 |       4.88 s |

The costs use the HF router's Baseten rates at experiment time:

- V4.1 Flash: `$0.30` per million input and `$1.20` per million output tokens;
- V4 Flash 0731: `$0.13` per million input and `$0.26` per million output tokens.

Linear planning estimates are:

| Model                  | 1,000 rows | 10,000 rows | 100,000 rows |
| ---------------------- | ---------: | ----------: | -----------: |
| DeepSeek V4.1 Flash    |      $1.60 |      $16.04 |      $160.43 |
| DeepSeek V4 Flash 0731 |      $0.53 |       $5.27 |       $52.69 |

## Automated validation

- V4.1 Flash: three of three runs passed.
- V4 Flash 0731: zero of three runs passed the strengthened text-quality validator.

The older model produced malformed or mixed-script text despite returning schema-valid
JSON. Its low price therefore does not represent accepted-record cost.

## Blinded finalist review

A fresh blinded file compared both DeepSeek variants with the earlier Qwen 397B and Qwen
235B outputs. Two independent reviewers inspected all 20 records without model
identities or cost data. Both ranked DeepSeek V4.1 first.

| Blinded label | Model                  | Language reviewer | Grounding/bias reviewer | Pilot result      |
| ------------- | ---------------------- | ----------------: | ----------------------: | ----------------- |
| Y             | DeepSeek V4.1 Flash    |    3.85/5, rank 1 |          3.50/5, rank 1 | Pass with caveats |
| W             | DeepSeek V4 Flash 0731 |    3.55/5, rank 3 |          2.75/5, rank 2 | Fail              |
| X             | Qwen 397B              |    3.65/5, rank 2 |          2.25/5, rank 3 | Fail              |
| Z             | Qwen 235B              |    2.60/5, rank 4 |          1.75/5, rank 4 | Fail              |

Review scores are subjective and based on only five records. The previous Qwen-only
review also scored the same Qwen outputs differently, reinforcing the need for more
reviewers and a larger blinded sample before model freeze.

### V4.1 strengths

- best restraint when a specialised domain lacked evidence;
- strongest internal consistency and labour-status preservation;
- natural Danish with no foreign-script contamination;
- fast inference at approximately five seconds per accepted response;
- lower observed cost than Qwen 397B.

### V4.1 remaining issues

- repeated birds, reuse, gardening, podcasts, and board games;
- older women still received crafts, preserves, gardening, and local-community themes;
- one record placed hiking and card games under skills rather than interests;
- occasional generic or circular hedging;
- some personality-derived local or calm travel assumptions remained.

### V4 Flash 0731 issues

The older model was generally fluent but made unsupported negative claims, confused
unemployment with being outside the labour market, inferred nearby travel from low
openness, and repeated domestic/traditional bundles for older women. Its universal
automated-validation failure rules it out despite being the cheapest tested model.

## Recommendation

Promote **DeepSeek V4.1 Flash through Baseten with `reasoning_effort: none`** to the
leading candidate. Keep Qwen 397B only as a quality reference and Qwen 235B as a cost
reference. Reject DeepSeek V4 Flash 0731.

Before the 10,000-row pilot, revise interest generation to break age/gender/status
associations and reduce cross-record repetition, then run a 20-50-record blinded V4.1
evaluation with multiple Danish reviewers. Five records are enough to select a leader,
not enough to freeze the production model.
