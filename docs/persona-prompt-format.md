# Persona prompt format

## Generation contract v2

The second generation stage returns six Danish texts per record:

1. `professional_persona`
2. `sports_persona`
3. `arts_persona`
4. `travel_persona`
5. `culinary_persona`
6. `persona`

The first five are specialised texts. They stay in their respective domains and are
written as short, distinct prose. The sixth is one short, grounded persona. There is no
`visual_persona` field in v2.

## Provider input boundary

The provider receives the human-readable labels needed to ground the text:

- age and statistical sex;
- municipality name;
- education label;
- official origin label;
- current labour-market status;
- official job-function label when the record is an eligible employee; and
- the generated attributes and OCEAN values or labels needed for the two stages.

Source and sampler identifiers do not reach the provider. In particular, the payload
does not include municipality, origin, or job-function codes, or education,
job-function, and sampling-resolution fields. The output and provenance may retain those
fields locally.

## Grounded `persona`

The short `persona` must state the following in natural Danish prose when applicable:

- age;
- statistical sex;
- municipality;
- education;
- official origin label;
- a short synthetic job title grounded only in the official job-function label, or the
  current canonical nonemployee status;
- two or three different interests copied from the generated attributes; and
- one or two complete, cautious OCEAN phrases copied literally from the supplied
  `allowed_personality_tendencies` list. The phrases must not be composed from a
  tendency term and a separate hedge.

The job title is synthetic. It is not an observed occupation and must not invent an
employer, institution, duties, seniority, qualifications, or previous work. A
nonemployee must not be given an earlier job or employer. The text must not invent
names, addresses, family or household claims, diagnoses, physical appearance, or other
unsupported facts.

The official origin label is a statistical category. It is not ethnicity, citizenship,
residence, appearance, language, culture, religion, or ancestry. It must not be used to
infer any of those traits.

## Image use and review

The v2 persona is ordinary grounded prose, not an image prompt or visual description.
`visual_persona` was removed rather than replaced with a visual grammar. A downstream
image model may still turn ordinary demographic or origin language into stereotypes.
Image generation is outside this contract and requires its own human review.

Historical v1 outputs and old pilots used earlier schemas or prompt context. They are
retained as historical evidence only and are not resumable under generation contract v2.
