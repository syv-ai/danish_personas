# DeepSeek V4.1 persona pilot

## Status

**Stopped incomplete at the user's request.** No merged pilot dataset was produced, and
this run must not be described as a 1,000-person dataset.

The intended input was the complete 1,000-row frozen Phase-3 sample:

`data/canonical/f5f37949670df476/text-development-seeds.parquet`

## Guarded execution design

The pilot runner keeps the existing five-row invocation ceiling. It partitions the
ordered frozen input by offset, gives every shard its own HTTP-attempt budget and durable
stage checkpoints, validates each completed shard, stops scheduling after a failure, and
merges only after all expected persona IDs are present exactly once.

The attempted pilot used:

- `deepseek-ai/DeepSeek-V4.1-Flash`;
- `reasoning_effort: none`;
- strict JSON Schema responses;
- five records per shard;
- an absolute 3,000-request pilot ceiling;
- ignored local outputs and configuration.

## Partial progress

### Baseten through Hugging Face

- 28 complete persona checkpoints;
- one additional attribute-only checkpoint;
- five fully completed and validated shards, covering 25 rows;
- 59 successful stage responses carrying 68 response-attached attempts;
- additional failed 429 attempts exist outside successful-response telemetry;
- 51,353 input and 24,777 output tokens in retained checkpoints;
- approximately `$0.04514` for successful retained tokens at the experiment-time rates.

Four-way concurrency immediately triggered HTTP 429 responses. Serial execution also hit
a persistent provider limit after approximately one shard. Shard pacing and a 60-second
429 fallback were insufficient for sustained operation.

Partial directory:

`data/persona-pilot-deepseek41/c457c9adb17bc227/`

### DeepInfra through Hugging Face

- 12 complete persona checkpoints;
- one additional attribute-only checkpoint;
- two fully completed and validated shards, covering 10 rows;
- 26 successful stage responses and 28 recorded attempts;
- 22,316 input and 10,441 output tokens in retained checkpoints;
- approximately `$0.01073` for successful retained tokens at the experiment-time rates.

DeepInfra was stopped after four hours because only ten complete rows had reached
validated shard manifests and two 504 responses had occurred. This route was too slow for
the pilot.

Partial directory:

`data/persona-pilot-deepseek41-deepinfra/8315bcef5faa407f/`

The two provider attempts overlap the beginning of the same frozen sample and therefore
must not be added together as distinct personas. Provider billing records, rather than
these token estimates, remain authoritative for actual charges.

## Resume requirements

No generation process remains active. All retained checkpoints are Git-ignored and can be
resumed only with the matching input, provider-qualified model, local configuration,
prompts, schemas, and validator version. Baseten and DeepInfra checkpoints cannot be
mixed because provider routing is part of the generation context.

Sustained capacity requires one of:

1. enabling Hugging Face pay-as-you-go for a structured-output provider such as Fireworks;
2. configuring a direct provider credential;
3. accepting an impractically slow or intermittently rate-limited run.

The user chose to stop and retain the partial run rather than enable another provider.
