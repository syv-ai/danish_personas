# Hugging Face Gemma cost smoke

## Scope

This is a historical generation-contract v1 experiment. Its retained outputs and metrics
are not resumable under generation contract v2 and do not establish the current v2
quality gate.

This was a deliberately small cost and quality experiment against Hugging Face's routed
Inference Providers API. It used two frozen demographic records and four retained
schema-constrained completions: one attributes and one descriptions call per record. It
does not approve the model or prompts for larger generation.

## Configuration

- API base: `https://router.huggingface.co/v1`.
- Model: `google/gemma-4-31B-it`.
- Routed provider selected by Hugging Face: DeepInfra.
- Authentication: `HF_TOKEN`, loaded locally and never written to artefacts.
- Response mode: strict JSON Schema.
- Per-batch maximum: two personas and five total HTTP attempts, including retries.
- Completion ceiling: 1,400 tokens.

A separate 32-token capability probe succeeded and cost `$0.00000666` according to the
router response.

## Observed run

The completed run had ID `e89f8c8caaf57156`. Its four retained completions used:

| Metric                            |              Observed |
| --------------------------------- | --------------------: |
| Input tokens                      |                 3,152 |
| Output tokens                     |                   952 |
| Total tokens                      |                 4,104 |
| Retained-response estimated cost  |           $0.00077152 |
| Cost per retained persona         |           $0.00038576 |
| Successful-attempt latency, mean  |        42.962 seconds |
| Successful-attempt latency, range | 22.754-66.302 seconds |

The response costs imply the DeepInfra rates applied by the router were `$0.13` per
million input tokens and `$0.38` per million output tokens.

The first five-request batch stopped at its configured limit. Across both bounded
batches, ten persona HTTP attempts produced four 504 responses and six successful
responses. Two successful first-batch responses were discarded by the then record-level
checkpoint implementation and repeated during resume. The `$0.00077152` above therefore
covers retained responses only and is a lower bound on billed experimental usage. The
pipeline now checkpoints the completed attributes stage immediately and carries request
counts across resumed batches.

## Linear planning estimate

Holding the observed token profile constant and excluding failed or discarded work
gives:

| Persona rows | Lower-bound generation cost |
| -----------: | --------------------------: |
|        1,000 |                     $0.3858 |
|       10,000 |                     $3.8576 |
|      100,000 |                    $38.5760 |
|    1,000,000 |                   $385.7600 |

These are not quotes. Prompt revisions, output-length changes, provider routing,
retries, pricing changes, and safety-regeneration rates will change the total. A larger
estimate should use the p95 cost per accepted record and include rejected generations.

Hugging Face currently documents monthly routed-inference credits of `$0.10` for a free
account, `$2.00` for PRO, and `$2.00` per Team or Enterprise seat. At the lower-bound
observed rate, `$0.10` covers about 259 personas and `$2.00` covers about 5,184
personas. A 10,000-row pilot would therefore require roughly `$1.86` beyond one month's
PRO credit, while a 100,000-row run would require roughly `$36.58` beyond it.

## Quality result

Both records passed the deterministic schema, provenance, language, and privacy gates,
but manual review failed the model/prompt combination. Problems included:

- an invented medium-sized provincial town and local cultural amenities;
- asserted professional experience and travel habits not supplied by the inputs;
- non-sport hobbies placed in the sports description;
- repeated interest patterns across both records;
- Danish errors and awkward Anglicisms such as `medtætte` and `cykel touring`.

The prompts were tightened after the experiment to prohibit those patterns.

## Revised five-record experiment

A follow-up selected five frozen records covering all regions and labour statuses, ages
23-73, both sexes, all four education categories, and varied OCEAN labels. Batches were
limited to 2, 2, and 1 records with five HTTP attempts each.

HF's fastest routing initially selected Together. With its default behaviour, many short
JSON responses consumed the 1,400-token ceiling and were truncated; other calls
returned 504. An explicit `chat_template_kwargs.enable_thinking: false` probe reduced a
trivial schema response to six completion tokens. All five records then completed in
exactly ten requests with no retries or transport failures and passed automated
validation.

The accepted runs used 8,920 input tokens and 2,313 output tokens. Together did not
return cost metadata, so cost was calculated from the router's listed rates of `$0.39`
per million input tokens and `$0.97` per million output tokens:

| Persona rows | No-thinking Together estimate |
| -----------: | ----------------------------: |
|   5 observed |                     $0.005722 |
|        1,000 |                       $1.1445 |
|       10,000 |                      $11.4448 |
|      100,000 |                     $114.4482 |
|    1,000,000 |                   $1,144.4820 |

Manual review still failed the model/prompt combination. The five records showed strong
demographic stereotypes, including knitting, cooking, and gardening for older women and
cars, fishing, and repairs for a middle-aged man. Interests repeated across records;
sports and travel fields leaked unrelated hobbies; a combined married-or-separated
category became definitely married; and phrases such as `digitalt indholdskabende`,
`Kryds- og Sudoku`, and `podcasting om livsstil` were unnatural or semantically wrong.
Some average OCEAN values were also overstated as definite traits.

The experiment therefore resolves the operational question but not the quality gate:
Gemma 4 requires no-thinking mode for this structured workload, and Together is reliable
but roughly three times the observed DeepInfra token cost. Do not fund a larger Gemma
run with these prompts. Compare another model and add cross-record stereotype,
repetition, and semantic-domain review before seeking a dataset-scale budget.
