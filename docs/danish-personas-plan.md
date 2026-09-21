# Plan for a Danish synthetic persona dataset

**Status:** Proposed

**Research date:** 2026-09-14

## Executive summary

Build a Danish-language dataset using the same broad architecture as
[Nemotron-Personas-USA][nemotron-dataset], but do not attempt to reproduce NVIDIA's
undisclosed production system exactly.

The recommended first public release is **100,000 adult personas**, preceded by a
**10,000-row pilot**. A million rows should only be generated if downstream experiments
show that the additional scale is useful. The structured demographic sampler should be
built from official, published aggregate statistics. A language model should only add
synthetic attributes and Danish persona text after the demographic records pass
statistical and consistency checks.

The initial release should:

- cover people aged 18 and over;
- use region or municipality, but not exact addresses or CPR numbers;
- model age, sex, marital status, broad education, labour-market status, and geography
  jointly where official cross-tabulations support it;
- include only a broad synthetic job-function allocation calibrated by sex to LONS20's
  incomplete earnings-statistics universe, never a representative observed occupation;
- sample personality independently of demographic and protected attributes;
- produce one detailed, grounded Danish `persona` under generation contract 4, with
  only the official Danish FOLK2 origin label reaching the provider request and exact
  persona grounding;
- exclude health, religion, politics, sexuality, criminal history, exact income, and
  other sensitive or high-risk fields;
- include reproducible source snapshots, prompts, model versions, validation reports,
  and a detailed dataset card.

This is a synthetic-data project, not an attempt to reconstruct Danish residents. It
must not ingest or link person-level registers for the first public release.

## What NVIDIA documents

### Dataset artifact

The public v1.1 dataset has 1,000,000 records and 0.94 billion tokens. NVIDIA changed
its generation model from `mistralai/Mixtral-8x22B-v0.1` to `openai/gpt-oss-120b` for
v1.1. The repository contains 11 Parquet shards and occupies approximately 2.83 GB on
Hugging Face.

The NVIDIA dataset card describes 22 historical content fields: six persona fields and
16 contextual fields. Its physical Parquet schema also contains a UUID, giving 23
physical columns. That historical count is not the current Danish contract. Generation
contract 4 retains one detailed, grounded
`persona`; `visual_persona` is removed. Historical v1/v2 outputs, the previous v2/v13
ten-person smoke, and old pilots are not resumable under generation contract 4. The
historical NVIDIA comparison must not be read as current Danish validation:

- `persona`
- `cultural_background`
- `skills_and_expertise`
- `skills_and_expertise_list`
- `hobbies_and_interests`
- `hobbies_and_interests_list`
- `career_goals_and_ambitions`
- `sex`
- `age`
- `marital_status`
- `education_level`
- `bachelors_field`
- `occupation`
- `city`
- `state`
- `zipcode`
- `country`
- `uuid`

Despite list-like names, the two list columns are stored as strings in the public
schema. A Danish release should use native Parquet list columns instead.

The public artifact excludes dedicated name and address fields, as well as finance and
healthcare personas available in NVIDIA's commercial tooling. Names can still occur in
the generated prose.

### Four-stage generation pipeline

NVIDIA documents the following common pipeline for its locale-specific persona data:

1. **OCEAN personality sampling.** Each of the five Big Five traits is sampled as a
   T-score from a normal distribution with mean 50 and standard deviation 10, clipped to
   20-80. Scores are mapped to five labels and curated prose descriptions.
2. **Demographically grounded sampling.** A probabilistic graphical model (PGM) uses
   aggregate census, administrative, and survey distributions. It is intended to retain
   correlations between age, education, occupation, marital status, and geography that
   independent sampling would lose.
3. **Structured persona attributes.** A reasoning language model receives demographics
   and OCEAN descriptions, then produces cultural background, skills, hobbies, career
   goals, and list variants under a Pydantic schema.
4. **Persona descriptions.** A second structured language-model call synthesizes the
   attributes into domain-specific and general persona narratives.

For the U.S. data, NVIDIA identifies the American Community Survey and aggregate name
statistics from Rosenman et al. as seed sources. The public card says the production PGM
is proprietary. NVIDIA has since released [SDG-PGMs][sdg-pgms], which describes a
similar cascaded PGM architecture and how to port it to another country.

### What is not reproducible from public information

NVIDIA does not disclose all of the following:

- the exact American Community Survey release and source tables;
- the complete production PGM graph and conditional probability tables;
- all production prompts and system prompts;
- model decoding parameters and generation infrastructure;
- validator and evaluator definitions or thresholds;
- rejection, retry, and filtering rates;
- semantic deduplication procedures;
- quantitative pass criteria or the complete validation report.

The count tables shipped with the open-source U.S. SDG-PGMs example are explicitly
random placeholders. They must not be treated as real demographic data. The Danish
implementation can reproduce the disclosed architecture, but it must build and validate
its own distributions.

## Proposed Danish dataset

### Intended use

The dataset should support:

- diverse seed personas for synthetic instruction and dialogue generation;
- Danish-language model evaluation and data augmentation;
- controlled sampling by broad demographic or geographic properties;
- research into representation and coverage in synthetic data.

It should not be represented as a population microdata product or used to make decisions
about real people. It should not be used for profiling, eligibility, credit, employment,
healthcare, policing, or political targeting.

### Release sizes

| Release                |      Rows | Purpose                                             |
| ---------------------- | --------: | --------------------------------------------------- |
| Development sample     |     1,000 | End-to-end plumbing and prompt iteration            |
| Pilot                  |    10,000 | Statistical, language, safety, and human evaluation |
| v1                     |   100,000 | Recommended public release                          |
| Optional scale release | 1,000,000 | Only after measured downstream benefit              |

NVIDIA's 0.94 billion tokens for one million rows implies roughly 940 released tokens
per row. A comparable 10,000-row pilot would therefore contain about 9.4 million output
tokens before accounting for prompt tokens, retries, and rejected generations. Actual
cost should be measured during the 1,000-row development run rather than estimated from
provider list prices in advance.

### Proposed public schema

Use stable English column names so the data is easy to process, but require all
free-text values to be Danish.

#### Identifiers and provenance

- `persona_id`: random UUID with no source-system meaning
- `dataset_version`: release version
- `source_reference_year`: demographic reference year
- `generation_model`: exact model identifier and revision
- `generation_seed`: reproducibility seed where the backend supports it

#### Demographic context

- `age`: integer, 18 years or older
- `sex`: official statistical category from the selected source table
- `marital_status`: normalized Statistics Denmark category
- `education_level`: broad RAS209 education category mapped to DISCED-15
- `education_resolution`: source category or disclosed proxy level
- `labour_market_status`: employed, unemployed, student, retired, or another documented
  RAS category
- `job_function_code` and `job_function`: optional broad two-digit DISCO-08 synthetic
  allocation for eligible employees only, calibrated by sex to LONS20
- `job_function_resolution`: `lons20_sex_marginal` or `not_applicable`
- `municipality_code`: official municipality code, subject to privacy review
- `municipality`: official municipality name, subject to privacy review
- `region`: one of the five Danish regions
- `country`: fixed to `Danmark`
- `origin_country_code`: official FOLK2 code retained for source/audit provenance
- `origin_country`: official English FOLK2 label retained for source/audit provenance
- `origin_country_da`: mandatory official Danish FOLK2 IELAND display label

Do not include a street address, exact workplace, CPR number, exact birth date, phone
number, email, or a synthetic identifier that resembles an official identifier.

#### Personality

Keep all five traits as structured values rather than silently embedding them into text:

- `openness_label`
- `conscientiousness_label`
- `extraversion_label`
- `agreeableness_label`
- `neuroticism_label`

Raw T-scores can remain internal unless a downstream use requires them. Personality must
not be inferred from sex, ancestry, geography, education, or labour-market status.

#### Generated attributes

- `cultural_context`
- `skills_and_expertise`
- `skills_and_expertise_list`: native `list[string]`
- `hobbies_and_interests`
- `hobbies_and_interests_list`: native `list[string]`
- `career_goals_and_ambitions`

`cultural_context` should describe plausible everyday context without claiming a
religion, ethnicity, political view, diagnosis, sexuality, or other sensitive trait. The
field should be renamed from NVIDIA's `cultural_background` to discourage invented
identity claims.

#### Generated persona text

- `persona`: one detailed, grounded Danish portrait preserving pronoun and age,
  municipality, origin, education, and an allowlisted synthetic job title or canonical
  current status, while incorporating concrete interests, skills, ambitions, fictional
  biography, and cautious OCEAN tendencies.

The provider receives the human-readable municipality, official Danish
`origin_country_da`, and job-function labels for this grounding. It does not receive
municipality or job-function codes, the origin code, English `origin_country`, contract
metadata, or resolution fields. The English label remains source/audit provenance only.
Neither origin label is ethnicity, citizenship, residence, or appearance. Origin cannot
drive culture, religion, job, interests, personality, or visual traits. A job title is
synthetic and must not imply unsupported work history. Fictional given names, education
detail, workplace settings, relationships, and family details are permitted, while
surnames, real organisations, exact addresses, appearance, and sensitive traits remain
prohibited. The persona is not a visual description; downstream image models may still
stereotype.

Fields that are irrelevant to a record should contain a natural, age- and status-aware
statement or be null according to a documented rule. They must not be filled with
implausible expertise merely to avoid null values.

### Optional name handling

A name is not required for the statistical model. For the pilot, use either no names or
only a synthetic given name in prose. If names are included later:

- sample from published name frequencies or the approved-name list;
- avoid rare names and rare name-demographic-geography combinations;
- do not generate complete addresses or employers alongside names;
- do not claim that name distributions encode ethnicity;
- disclose that coincidental similarity to a real person is possible;
- test generated full names and prose for exact web-search matches on a review sample.

## Danish source plan

Use the [StatBank API][statbank-api] for machine-readable table metadata and aggregate
counts. Freeze every source response used for a release and record its checksum,
retrieval date, reference period, publisher, licence, and attribution requirement.

| Variable                       | Candidate official sources           | Notes                                                             |
| ------------------------------ | ------------------------------------ | ----------------------------------------------------------------- |
| Age, sex, geography            | FOLK1A, BEFOLK3                      | Select compatible reference dates                                 |
| Adult origin marginal          | FOLK2                                | National official IELAND categories; independent Phase 2 marginal |
| Marital status                 | FOLK1A                               | Preserve official definitions                                     |
| Citizenship validation         | FOLK1B                               | Uses broad age bands                                              |
| Ancestry validation            | FOLK1E                               | Do not interpret as ethnicity                                     |
| Broad education and status     | RAS209                               | Municipality, age band, sex, education, status                    |
| Detailed status and retirement | RAS202                               | Exact age through 70, then `71+`; no region                       |
| Municipality status validation | RAS210                               | Three status groups; title says ages 13-70                        |
| Detailed education under 70    | HFUDD11, HFUDD16                     | Both cover ages 15-69 only                                        |
| Synthetic job function         | LONS20 `ANTAL`, DISCO-08             | Sex marginal in the incomplete earnings-statistics universe only  |
| Household extensions           | FAM55N, FAM122N, FAM44N              | Defer to a later release                                          |
| Names                          | Statistics Denmark name statistics   | First first-name/final surname limits                             |
| Geography codes                | DST classification `NUTS_V1_2007_DK` | Region, landsdel, and municipality hierarchy                      |
| Geography boundaries           | DAGI                                 | Needs Datafordeler credentials; not used                          |
| Housing extensions             | BOL103, BOL104, BBR aggregates       | Do not link addresses to people                                   |

Use RAS209 as the primary municipality-level joint calibration table for broad
education, socioeconomic status, age band, and sex; attach the official region parent
only through hierarchy lookup. Use RAS202 to refine detailed retirement and other status
categories by age and sex. RAS is measured on the last working day of November.

No active public table found during this research combines DISCO-08 occupation counts
with age, education, and geography. The limited extension therefore uses only LONS20
2024 `ANTAL`, all sectors, all forms of pay, the employee-group total, M/K, and exactly
the 42 two-digit DISCO-08 groups. LONS20 covers all public employees and private
organisations with at least 10 full-time-equivalent employees; smaller private
organisations and other earnings-statistics exclusions are absent. The result is a
synthetic sex-conditional allocation for eligible RAS202 employee statuses, not observed
occupation or all-worker representation. It is not conditioned on geography, origin,
age, education, OCEAN, or any unsupported joint.

The exact source extracts must be selected after inspecting their dimensions and
reference periods. Aggregate tables cannot be joined as if they were observations about
the same individuals. They are constraints from different statistical universes and
measurement dates, requiring a documented synthesis and reconciliation method.

Important source limitations to carry into the dataset card include:

- Danish population statistics are register-based and cover registered residents;
- delayed or missing emigration registrations can cause a small population overcount;
- household and family categories are administrative definitions;
- education levels for some immigrants are imputed at broad levels and should not be
  interpreted as valid individual-level observations;
- HFUDD11 and HFUDD16 cover only ages 15-69, so they cannot ground detailed attainment
  for older personas;
- RAS209 provides broad education for an oldest band of `67 years and over`; use that
  band as a disclosed proxy for personas aged 70 and over rather than claiming observed
  detailed attainment at those ages;
- name statistics use the first first-name and final surname, affecting compound names;
- historical family definitions changed, limiting comparisons across time;
- RAS register employment status is not interchangeable with survey employment;
- FOLK1B measures citizenship, FOLK1E measures ancestry, and BEFOLK3 contains neither.
- FOLK2 combines age, sex, HERKOMST, STATSB, and official IELAND counts only as an
  independent national marginal for adults in 2025. Its 312,336 selected observations
  produce 2,186,352 API cells, so the locked query uses the BULK exemption. It is not
  ethnicity, citizenship, or residence; official labels such as Stateless and Not stated
  are retained without custom country groups or inferred correlations. It is sampled
  independently into Phase 2 origin fields. The English label remains official
  source/audit provenance. The mandatory Danish label comes from the archived
  metadata-da 241-code contract (source metadata SHA-256
  `f5c1f0a20f29372d6b222ce7a23cdc4ef0481d9e23fa6bd9b66b116e7adcb213`). Only the Danish
  label reaches the provider request and exact persona grounding; code, English label,
  resolutions, and contract metadata do not.

## Required execution order

The work must proceed through three strict boundaries:

1. **Set up the complete codebase and contracts.** Create the package structure,
   configuration formats, typed schemas, command-line entry points, tests, logging, and
   fixture-based pipeline before downloading production data.
2. **Fetch and prepare every non-LLM input.** Download and freeze all official aggregate
   tables, metadata, codebooks, classifications, and geographic mappings. Build the
   normalized distributions and validate the source bundle without generating records.
3. **Start generation.** Generate and validate demographic records first. Only after the
   demographic gate passes may an LLM generate attributes and persona text.

No production demographic or persona rows should be generated during steps 1 or 2. No
LLM should be called until the demographic-only generation and validation stage passes.
This keeps setup, source acquisition, deterministic generation, and probabilistic text
generation independently testable and restartable.

### Implementation status

Phases 0-2 are implemented and validated. The source bundle prepares the official FOLK2
adult origin marginal. Phase 2 samples this marginal independently with deterministic
quotas and retains the English code-label pair for source/audit provenance plus the
mandatory Danish display label. Only the Danish label may reach the provider request and
exact persona grounding; code, English label, resolutions, and contract metadata do not.
Neither label is ethnicity, citizenship, residence, or appearance. Origin cannot drive
culture, religion, job, interests, personality, or visual traits.

The current contracts are prepared-bundle schema 6, sampler schema 6, frozen-sample
schema 3, generation contract 4, validator `persona-safety-v17`, and release
release manifest schema 2 and evidence schema 3. The current offline bundle is
`8a4133e5a0a52050`; its
passing smoke and statistical runs are `f4dffe214a2faf0b` and `55fb89fb303a67f0`.
The previous schema-5 bundle `cfc1b56f5586a2d7` and runs `f449f1de01d18c08` /
`3ebc00282c621ef2` remain historical, non-resumable evidence. Release IDs remain
placeholders until approved live generation, human review, packaging, and
verification. The
LONS20 extension assigns broad job functions within sex only to RAS202 employee codes
15, 20, 25, 30, 35, and 40, using deterministic largest-remainder quotas and an isolated
fourth RNG stream. The human-readable job-function label may reach the provider for a
synthetic title, while its code and resolution do not. See the [Phase 2 validation
report][phase-2-report] for exact checksums and metrics. LLM generation uses the root
Hydra configuration and remains bounded by row and HTTP-request limits.

The frozen text-development input remains a separate, deliberately stratified 1,000-row
Phase-3 sample taken only after statistical validation; it is not the Phase-2 smoke run.
The current release contract is disabled by default and binds schema-2 release
manifest/evidence, exact artefact checksums, provenance, and blinded human review.
Exactly 10,000 rows requires 300 unique blinded-human-reviewed IDs, and populations of
at least 100,000 require 500 (with stricter policy minima permitted). Other sizes fail
closed. Release IDs and checksums remain placeholders until a current package is
regenerated.

## Generation architecture

### 1. Source registry and immutable snapshots

Create a manifest for every source with:

- source name and StatBank table ID;
- API query and selected dimensions;
- publisher and canonical URL;
- reference date and retrieval timestamp;
- licence and required attribution;
- SHA-256 checksum of the raw response;
- category definitions and known limitations.

Store raw responses unchanged, using explicit UTF-8 and LF for canonical CSV snapshots.
Derived tables should be reproducible from those snapshots and transformation code.

### 2. Canonical categories

Normalize official source categories into versioned enums. Retain the original code and
label beside every mapping. Explicitly encode structural zeros, including:

- education and labour-market states that are impossible for an age group;
- detailed education where the selected table does not cover the age;
- municipality-to-region mismatches;
- incompatible marital-status and age combinations.

Avoid silently collapsing `unknown`, `not applicable`, and missing values.

### 3. Demographic sampler

Start with transparent cascaded conditional distributions rather than a black-box model.
A candidate v1 dependency graph is:

```text
municipality -> official region parent
age + sex + municipality -> marital_status
age_band + sex + municipality -> broad_education + broad_status  # RAS209
age + sex + broad_status -> detailed_status                 # RAS202
```

The RAS209 joint block should be sampled or calibrated together rather than decomposed
into unsupported conditionals. RAS202 can refine a broad status, but reconciliation must
preserve the RAS209 totals. For ages 70 and over, retain only RAS209's broad education
proxy and mark `education_resolution` accordingly.

Only retain edges for which an official cross-tabulation or defensible fitted model
exists. For sparse cells, use a documented hierarchy:

1. use the exact municipality, age-band, and sex conditional cell;
2. relax sex or age-band only while retaining the same municipality;
3. fail if that municipality has no terminal cell.

RAS202 detailed-status refinement is explicitly separate: it is a national table and may
relax age and sex only while retaining the RAS209 broad status. No FOLK1A or RAS209 draw
may back off to region or national geography.

Apply smoothing only after structural zeros. Record which back-off level produced each
row for audit and validation, even if that field is omitted from the public release.

SDG-PGMs can be evaluated as the implementation framework, but the statistical model,
not framework parity with NVIDIA, is the deliverable. A simpler auditable sampler is
preferable if it matches the available Danish tables more faithfully.

### 4. Personality sampler

Implement NVIDIA's disclosed OCEAN procedure as an optional compatibility mode:

- draw five independent normal T-scores with mean 50 and standard deviation 10;
- clip each score to 20-80;
- map scores to `very_low`, `low`, `average`, `high`, or `very_high`;
- attach curated Danish trait descriptions for prompting.

The normal distribution is a design assumption, not a claim about the Danish population.
Document it as such. Do not condition traits on demographic variables.

### 5. Structured attribute generation

Generate attributes from the demographic and personality blocks using a strict Pydantic
schema. The prompt should require:

- Danish output;
- internal consistency with age, education, labour-market status, and location;
- concrete but non-identifying details;
- no unsupported sensitive attributes;
- no stereotypes or deterministic demographic-personality associations;
- no exact employers, addresses, institutions, or public figures;
- null or conservative output where evidence is insufficient.

Keep prompts and schemas in version control. Store raw model responses and validation
errors in a restricted intermediate area, not in the release artifact.

### 6. Persona description generation

Use one structured generation call returning attributes and one detailed, grounded
`persona`. The provider request may receive the municipality, official Danish
`origin_country_da`, and job-function labels as appropriate, but never
their codes, English origin label, contract metadata, or resolution fields. The exact
Danish label must ground the persona. It must use a synthetic job title or current
nonemployee status, allow natural use of interests in prose, and express OCEAN only as
cautious tendencies. `visual_persona` is removed; the persona is not a visual
description, and downstream image models may stereotype.

Evaluate at least two Danish-capable models on the same stratified development set. Pick
the model using blinded human ratings for fluency, consistency, specificity, stereotype
risk, and unsupported claims. Do not select solely on cost or generic benchmark scores.

### 7. Deterministic post-processing

Normalize whitespace and Unicode, enforce Danish language checks, validate list fields,
and attach provenance. Do not use post-processing to conceal invalid model output.
Reject and regenerate records that fail semantic or safety rules.

## Validation plan

### Statistical fidelity

Compare the synthetic data with every compatible source marginal and cross-tabulation.
Report, at minimum:

- observed and target counts and proportions;
- absolute percentage-point error by category;
- total variation and Jensen-Shannon distance for categorical distributions;
- Wasserstein distance for age;
- errors by municipality and for smaller demographic groups;
- back-off rates and unsupported combinations.

Acceptance bands should account for sample size. A practical starting rule is that each
target proportion must lie within the larger of 0.5 percentage points or three binomial
standard errors. This error rule must be paired with release-size-aware pooling:

- never use a calibration cell containing fewer than 50 source persons;
- pool each cell into a documented parent category until its expected count in the
  planned release is at least five;
- reconcile pooled categories to at least 99% of each matching table universe;
- do not force every rare source cell to occur in a representative release sample;
- evaluate excluded rare slices in a separate stratified, weighted QA sample that is not
  included in distribution-fidelity metrics or the public release;
- preserve missing or suppressed source values rather than treating them as zero.

The threshold of five is an initial engineering rule, not a privacy guarantee. Register
final thresholds before evaluating the 10,000-row pilot. Matching one-dimensional
marginals is insufficient; validate every available joint table within its own
statistical universe.

### Structural and semantic validity

Automated tests should require:

- 100% schema-valid released rows;
- zero structural-zero violations;
- valid municipality-region, RAS, and DISCED mappings;
- no impossible age-education-status combinations;
- correct use and labelling of the `67+` education proxy;
- consistency between structured attributes and all persona texts;
- natural Danish grounding for pronoun and age, municipality, origin, broad education,
  and job title or current nonemployee status, while allowing natural use of supplied
  interests in running prose and cautious OCEAN tendencies in `persona`;
- no exact addresses, CPR-like values, phone numbers, or email addresses;
- no appearance claims, surnames, real organisations, exact addresses, or disallowed
  sensitive-attribute claims;
- Danish language above a pre-registered classifier threshold.

### Diversity and duplication

Measure:

- exact duplicate rows and text fields;
- normalized and near-duplicate texts using MinHash or embeddings;
- repeated templates and common n-grams;
- education, status, hobby, skill, and geographic coverage;
- embedding-cluster concentration overall and by demographic slice.

Set retry limits. If repeated regeneration still fails, retain a structured demographic
row without prose or drop it while reporting the drop rate. Never quietly resample until
only easy demographic combinations remain.

### Bias and stereotype evaluation

For matched demographic records, vary one attribute at a time and compare generated
skills, hobbies, ambitions, sentiment, and socioeconomic language. Flag large
differences that are not grounded in source variables. Review rare and intersectional
groups separately rather than relying on average scores.

Conduct a stratified Danish-language human review of at least:

- 300 records from the 10,000-row pilot;
- 500 records from the proposed v1 release;
- oversamples from rare categories and every municipality or geographic grouping.

Reviewers should rate fluency, coherence, realism, unsupported specificity, stereotype
risk, and whether the text could be mistaken for a real identifiable person.

### Privacy and re-identification review

Synthetic data does not automatically have zero privacy risk. Danish Data Protection
Agency guidance states that pseudonymised data remains personal data and that aggregate
or anonymised information is outside GDPR only when identification is not reasonably
possible using other available information.

For v1:

- use only public aggregate inputs;
- never ingest CPR data or person-level register extracts;
- exclude exact addresses, dates of birth, workplaces, and contact details;
- suppress or coarsen source cells and generated combinations below an agreed threshold;
- scan for rare combinations and accidental identifiers;
- manually search a risk-weighted sample for apparent real-person matches;
- obtain a documented privacy and legal review before public release.

If restricted microdata is ever considered, stop and create a separate governance plan,
legal basis, access agreement, disclosure-control method, and data-protection impact
assessment. It is outside the scope of this plan.

## Delivery phases and gates

### Phase 0: Complete codebase setup

This phase prepares the entire pipeline shape before production data is fetched.

#### Work

- Confirm intended downstream tasks, prohibited uses, age range, reference-year policy,
  geography level, and v1 schema.
- Create the proposed package, script, configuration, data, documentation, and test
  directories.
- Add dependencies through `uv` and establish the locked development environment.
- Define typed schemas for source manifests, normalized counts, demographic records,
  OCEAN traits, generated attributes, persona text, and run manifests.
- Define configuration formats for sources, category mappings, sampling, generation,
  validation, and release settings.
- Implement command-line entry points for downloading, preparing, generating,
  validating, and exporting, initially backed by small local fixtures.
- Establish structured logging, checkpoints, resumability, deterministic seeds, and
  restricted handling of raw model responses.
- Add unit tests, integration tests over fixture data, linting, type checks, and CI.
- Create the source/licence register, privacy risk register, human-review rubric, and
  initial acceptance criteria.

**Restrictions:** do not download production source tables, generate production records,
or call an LLM in this phase.

**Exit gate:** every planned pipeline command exists and passes checks end to end on
fixture data; schemas, configuration contracts, privacy boundaries, and validation
interfaces are approved.

### Phase 1: Fetch and prepare all non-LLM inputs

This phase creates the complete immutable input bundle needed by deterministic
generation.

#### Work

- Inspect candidate StatBank metadata and select compatible tables and reference
  periods, including FOLK2's exact ages 18-125, both sexes, all HERKOMST and STATSB
  values, all official IELAND values, and 2025.
- Download every selected aggregate table, metadata response, codebook, classification,
  and geographic mapping through the implemented source adapters. Calculate StatBank's
  query size as selected observations (the product of selected values) multiplied by all
  returned columns, including time dimensions and the observation value; reject
  over-limit non-streaming queries and use BULK only as its documented exemption.
- Record canonical URLs, queries, retrieval times, licences, attribution, and SHA-256
  checksums in the source manifest.
- Normalize categories and create versioned mappings while retaining original codes and
  labels.
- Derive the calibrated count tables, structural-zero rules, sparse-cell pooling,
  reconciliation parameters, and sampling distributions.
- Prepare the curated Danish OCEAN label descriptions used later by the generator.
- Run source-integrity, coverage, sparsity, mapping, licence, and reference-period
  checks.
- Produce a source-preparation report describing unresolved conflicts and all modelling
  assumptions.

**Restrictions:** do not generate demographic or persona rows and do not call an LLM in
this phase.

**Exit gate:** the pipeline can run offline from a complete, checksummed source bundle;
every modeled field has a licensed source, definition, reference period, transformation
test, and prepared sampling distribution.

### Phase 2: Generate and validate structured demographics

Generation begins here, but this phase remains entirely non-LLM.

#### Work

- Instantiate the prepared conditional sampler, structural zeros, smoothing, and
  back-off rules.
- Generate demographic-only records and independently sample OCEAN traits.
- Start with a 2,000-row smoke run, then generate 100,000 structured records for
  inexpensive statistical validation. Keep the separate stratified text-development
  sample at 1,000 rows.
- Compare records with held-out aggregate tables rather than only fitted marginals.
- Run structural, statistical, sparse-cell, geographic, and proxy-resolution checks.
- Freeze a stratified set of validated demographic records for text-generation
  development.

**Restrictions:** do not call an LLM until this phase's exit gate passes.

**Exit gate:** demographic records pass the pre-registered statistical and structural
thresholds, and the demographic sampler configuration is frozen.

### Phase 3: Develop LLM generation

This is the first phase that may call an LLM. The guarded provider adapter,
single-response schema, checkpointing, deterministic validators, and a three-record
smoke test are complete. Full development-sample generation and evaluation remain pending.

#### Work

- Finalize Danish prompts, validators, and safety rules against the already defined
  typed schemas.
- Generate attributes and the generation-4 persona text field for the frozen,
  stratified 1,000-row development sample using candidate models.
- Measure token use, latency, retries, failures, and actual cost.
- Conduct blinded human evaluation and select the model and configuration.

**Exit gate:** the selected model passes language, coherence, safety, consistency, and
cost criteria without changing the frozen demographic distribution.

### Phase 4: Generate the 10,000-row pilot

#### Work

- Freeze source, sampler, prompt, model, and validator versions.
- Sample 10,000 validated demographic records from the frozen sampler.
- Generate structured attributes and one generation-4 persona text field in a single
  provider response per record.
- Run statistical, structural, duplication, bias, privacy, and human evaluation.
- Publish an internal report including all token, retry, rejection, and drop rates.

**Exit gate:** independent review approves scaling. Failed slices must be fixed and the
pilot rerun rather than waived without explanation.

### Phase 5: Generate the 100,000-row v1

#### Work

- Generate deterministic demographic records and versioned persona attributes and text.
- Re-run all validations and the 500-row stratified human review.
- Produce Parquet shards, checksums, dataset card, validation report, and licences.
- Publish generation code and prompts unless a model licence or security issue prevents
  it; document any omission.

**Exit gate:** statistical, safety, privacy, reproducibility, and licence review
complete.

### Phase 6: Optional scaling

Scale toward one million rows only if a controlled downstream evaluation shows a useful
improvement over the 100,000-row release. First test whether more text variants per
structured persona provide more value than more demographic records. Re-estimate cost,
duplication, and quality at each scale step.

## Proposed repository layout

```text
docs/
  danish-personas-plan.md
  data-card.md
  validation-report.md
config/
  sources.yaml
  schema.yaml
  generation.yaml
  validation.yaml
data/
  raw/
  interim/
  processed/
src/danish_personas/
  sources/
  schema/
  sampling/
  generation/
  validation/
src/scripts/
  fix_dot_env_file.py
  generate_persona.py
  build_dataset.py
tests/
  sources/
  sampling/
  generation/
  validation/
```

Raw model responses and any restricted review material should not be committed. Large
public artifacts should be stored on Hugging Face or object storage rather than Git.

## Reproducibility and release checklist

A release is complete only when it includes:

- immutable source manifest and checksums;
- source licences and attribution;
- exact source reference periods;
- category mappings and structural-zero rules;
- code revision and environment lock file;
- random seeds and sampler configuration;
- prompt, schema, model, model revision, and decoding configuration;
- prepared-bundle 6, sampler 6, frozen-sample 3, generation 4, validator v16, and
  release manifest schema 2 and evidence schema 3 version bindings;
- official FOLK2 English provenance label and Danish 241-code display-label contract,
  including source metadata SHA-256
  `f5c1f0a20f29372d6b222ce7a23cdc4ef0481d9e23fa6bd9b66b116e7adcb213`;
- token, retry, rejection, and drop counts;
- statistical and qualitative validation reports;
- known limitations and prohibited uses;
- Parquet schema, shard checksums, and row count;
- contact and process for reporting harmful or identifying output.

## Recommended initial decisions

Unless downstream requirements indicate otherwise, begin with these defaults:

1. 10,000-row pilot and 100,000-row v1.
2. Adults aged 18 and over.
3. Most recent common reference year supported by the selected source tables.
4. Municipality-level sampling internally, with municipality published only if the
   privacy review passes; otherwise publish region.
5. Broad RAS209 education for all ages, with its `67+` band disclosed as a proxy for
   personas aged 70 and over; no unsupported detailed attainment for those ages.
6. No observed occupation, household, income, ancestry, citizenship, full name, or
   sensitive fields in v1; broad job function is the documented synthetic exception.
7. One generation-4 persona text field: a short, grounded
   persona; `visual_persona` is removed. Historical v1/v2 outputs, the previous v2/v13
   ten-person smoke, and old pilots cannot be resumed under this contract.
8. Native list columns and explicit provenance fields, even where this differs from the
   NVIDIA schema.
9. Open generation code, prompts, source manifests, and validation results.

## Sources

All sources were accessed on 2026-09-14.

### NVIDIA and Hugging Face

- [Nemotron-Personas-USA dataset card][nemotron-dataset]
- [Nemotron-Personas-USA dataset repository metadata][nemotron-api]
- [Nemotron-Personas first-rows schema][nemotron-schema]
- [Designing Nemotron-Personas: four-stage pipeline][nemotron-pipeline]
- [SDG-PGMs repository][sdg-pgms]
- [SDG-PGMs U.S. person example and porting notes][sdg-us-example]
- [NVIDIA's original Nemotron-Personas blog][nemotron-blog]

### Danish statistics and classifications

- [StatBank API documentation][statbank-api]
- [Statistics Denmark population tables][folk1a]
- [Statistics Denmark household and family tables][fam55n]
- [Statistics Denmark detailed education table HFUDD11][hfudd11]
- [Statistics Denmark detailed education/status table HFUDD16][hfudd16]
- [Statistics Denmark joint education/status table RAS209][ras209]
- [Statistics Denmark detailed status table RAS202][ras202]
- [Statistics Denmark municipality status table RAS210][ras210]
- [Statistics Denmark occupation earnings table LONS20][lons20]
- [FOLK1B citizenship table][folk1b]
- [FOLK1E ancestry table][folk1e]
- [BEFOLK3 population table][befolk3]
- [DISCO-08 occupation classification][disco]
- [DISCED-15 education classification][disced]
- [Statistics Denmark name statistics][names]
- [DAGI administrative geography][dagi]
- [Statistics Denmark source attribution terms][dst-licence]

### Privacy and law

- [Danish Data Protection Agency: what is personal data][personal-data]
- [Danish Data Protection Agency: anonymisation and pseudonymisation][anonymisation]
- [EU General Data Protection Regulation][gdpr]
- [Danish Data Protection Act][data-protection-act]
- [Statistics Denmark microdata security rules][microdata-rules]

[nemotron-dataset]: https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA
[nemotron-api]: https://huggingface.co/api/datasets/nvidia/Nemotron-Personas-USA
[nemotron-schema]:
 https://datasets-server.huggingface.co/first-rows?dataset=nvidia%2FNemotron-Personas-USA&config=default&split=train
[nemotron-pipeline]:
 https://docs.nvidia.com/nemo/datadesigner/dev-notes/designing-nemotron-personas
[sdg-pgms]: https://github.com/NVIDIA-NeMo/SDG-PGMs
[sdg-us-example]: https://github.com/NVIDIA-NeMo/SDG-PGMs/tree/main/examples/us_person
[nemotron-blog]: https://huggingface.co/blog/nvidia/nemotron-personas
[phase-2-report]: reports/phase-2-validation.md
[statbank-api]: https://www.dst.dk/en/Statistik/hjaelp-til-statistikbanken/api
[folk1a]: https://www.statbank.dk/FOLK1A
[fam55n]: https://www.statbank.dk/FAM55N
[hfudd11]: https://www.statbank.dk/HFUDD11
[hfudd16]: https://www.statbank.dk/HFUDD16
[ras209]: https://www.statbank.dk/RAS209
[ras202]: https://www.statbank.dk/RAS202
[ras210]: https://www.statbank.dk/RAS210
[lons20]: https://www.statbank.dk/LONS20
[folk1b]: https://www.statbank.dk/FOLK1B
[folk1e]: https://www.statbank.dk/FOLK1E
[befolk3]: https://www.statbank.dk/BEFOLK3
[disco]: https://www.dst.dk/en/Statistik/dokumentation/nomenklaturer/disco
[disced]: https://www.dst.dk/en/Statistik/dokumentation/nomenklaturer/disced15-audd
[names]: https://www.dst.dk/en/Statistik/emner/borgere/navne
[dagi]:
 https://datafordeler.dk/dataoversigt/danmarks-administrative-geografiske-inddeling-dagi/
[dst-licence]: https://www.dst.dk/en/presse/kildeangivelse
[personal-data]:
 https://www.datatilsynet.dk/english/fundamental-concepts/what-is-personal-data
[anonymisation]:
 https://www.datatilsynet.dk/regler-og-vejledning/behandlingssikkerhed/katalog-over-foranstaltninger/pseudonymisering-og-anonymisering
[gdpr]: https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng
[data-protection-act]: https://www.retsinformation.dk/eli/lta/2024/289
[microdata-rules]:
 https://www.dst.dk/en/TilSalg/data-til-forskning/regler-og-datasikkerhed/regler-for-arbejdet-med-mikrodata
