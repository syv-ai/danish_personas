# Persona prompt format

## Generation contract 4

The second generation stage returns one short Danish text in `persona`. The former
specialised persona fields are not part of the active schema or Parquet output. The
attributes stage remains structured and separate. Generation contract 4 uses validator
`persona-safety-v16`; release manifest and evidence schemas remain 2 because their
structures did not change.

## Provider input boundary

Both provider stages receive only approved human-readable fields: municipality,
the official Danish `origin_country_da`, and reviewed job-function labels and title
allowlist entries. They never receive the origin code, English origin label, contract
metadata, or resolution fields. The English label remains source/audit provenance but
is withheld from provider payloads. The Danish label is not ethnicity, citizenship,
residence, or appearance, and origin cannot drive culture, religion, job, interests,
personality, or visual traits.

The stage-two payload supplies demographic context, broad grounding facts, compatible
personality wording, and generated attributes. These are context, not clauses that the
model must copy verbatim. Personality wording does not require a fixed count, and the
persona may naturally use none, some, or more than one supplied interest or tendency.

## Grounded `persona`

The validator requires the prose to preserve age, the supplied `han` or `hun`,
municipality, official Danish origin label, broad education level, and current title or
status. It allows natural paraphrase and benign consistent elaboration, but rejects
unsupported family, appearance, sensitive-attribute, identifying, former-work, and
technical claims. The pronoun must remain consistent. Token-bounded occurrences of
`vedkommende`, `personen`, `kan være`, and `ungdoms- eller erhvervsuddannelse` are
rejected, as are redundant sex nouns.

Education remains deliberately broad: no qualification or institution may be inferred.
The approved job-title mapping is synthetic and allowlisted. OCEAN inputs are cautious
tendencies rather than diagnoses, abilities, or fixed traits. Exact duplicate persona
text is rejected across an output population, and checkpoints retain the existing
checksum, accounting, tamper, and resume gates. Historical v1-v3 artefacts are not
resumable or releasable under v4.

## Image use and review

`persona` is ordinary grounded prose, not an image prompt or visual description.
Downstream image generation is outside this contract and requires separate blinded
human review.
