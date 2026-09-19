# Phase-3 LLM smoke report

## Scope

This report covers a three-record integration smoke test only. It does not approve the
frozen 1,000-row development sample, the 10,000-row pilot, or any release-scale persona
generation. It is a historical generation-contract v1 report from before the v2 change:
its six-description results are not evidence for the current six-text grounded-persona
schema or validation gates. The old run is retained for historical facts only and cannot
be resumed under v2.

The run used the first three persona identifiers from the checksum-locked, stratified
Phase-3 seed file. Generation preserved the existing demographic and independently
sampled OCEAN fields.

## Provider

- Endpoint: local `pi-openai-api` OpenAI-compatible proxy.
- Model: `gpt-5.6-sol` through provider `openai-codex`.
- Response mode: strict JSON Schema.
- Stages per record in this historical v1 run: structured attributes, then six persona
  descriptions under the old prompt contract; current v2 runs return five specialised
  texts plus one short grounded persona, with no `visual_persona`.
- Sampling parameters: none; the proxy rejects temperature, seed, and token-limit
  parameters.

The proxy advertised `gpt-5.4-mini`, but the upstream ChatGPT account returned
`model_not_found` for that model. A 56-token schema probe against `gpt-5.6-sol`
succeeded before the persona run.

## Final smoke run

- Run ID: `73aa018ab9db2132`.
- Upstream run ID: `f5f37949670df476`.
- Rows: 3.
- HTTP requests: 6.
- Retries: 0.
- Prompt tokens: 4,795.
- Completion tokens: 2,545.
- Total tokens: 7,340.
- Output SHA-256: `34a9d898dd7df8793c17ec2d4647c84d92e71bff06d2fe51ece757c0c993bac3`.
- Report result: passed all five mandatory automated metrics.

The proxy did not report monetary cost, so actual cost remains unmeasured. Token counts
are recorded for later model comparison.

A repeated live command loaded all three per-record checkpoints and issued zero new
model requests.

## Quality review

An initial three-record run was mechanically valid but translated extreme OCEAN values
into overly definite limitations and reused generic combinations of reading, walking,
and cooking. The prompts were tightened before the final run to:

- treat OCEAN values as weak tendencies rather than facts about abilities or problems;
- prohibit score labels and deficit framing;
- use the persona identifier only as a hidden variation key;
- discourage repeated default interest combinations.

Manual review of all 18 final historical descriptions found natural Danish, no
identifying details, no sensitive-attribute inference, no demographic changes, and more
varied interests. The text remains deliberately conservative when a specialised domain
is unsupported. Some phrases still sound statistical, particularly translations of
detailed labour status, and the sample is too small to assess systematic stereotyping,
diversity, cross-record repetition, or broad language quality.

## Gate status

The provider integration and smoke path work. The Phase-3 exit gate is not met. Before
larger generation, the project still requires candidate-model comparison, broader
automatic distribution-level text checks, blinded human review, latency measurement,
actual cost estimation, and explicit approval of the selected prompts and model.
