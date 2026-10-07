"""Conservative offline proposals for explicit Danish partner-gender clauses."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARTNER_CLAUSE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)Personen har en "
    r"(?P<gender>mandlig|kvindelig) partner\.(?P<trailing>[ \t]*)$"
)
_PARTNER_MENTION = re.compile(r"\bpartner(?:en|ens|e|es)?\b", re.IGNORECASE)
_GENDER_MENTION = re.compile(
    r"\b(?:mand|mænd|kvinde|kvinder|mandlig|kvindelig|nonbinær|nonbinære)\b",
    re.IGNORECASE,
)
_PRONOUN = re.compile(
    r"\b(?:han|hun|hen|ham|hende|dem|de|sin|sit|sine|hans|hendes|deres|"
    r"he|she|they|him|her|them|his|their)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PartnerProseProposal:
    """A narrow source-grounded prose proposal, or a fail-closed abstention."""

    proposed_text: str | None
    evidence: str | None
    abstention_reason: str | None

    @property
    def abstained(self) -> bool:
        """Whether no prose edit was proposed."""
        return self.proposed_text is None


def propose_partner_gender_prose_repair(
    *,
    old_partner_gender: str,
    new_partner_gender: str,
    old_gender: str,
    new_gender: str,
    old_persona_text: str,
) -> PartnerProseProposal:
    """Neutralise one explicit outdated partner-gender clause, otherwise abstain.

    Only the complete standalone clause ``Personen har en mandlig/kvindelig
    partner.`` is eligible. Self-gender changes and all pronoun-bearing prose
    abstain; sex is deliberately not an input and cannot determine gender.
    """
    old_partner = _normalise_partner_gender(old_partner_gender)
    new_partner = _normalise_partner_gender(new_partner_gender)
    if old_partner not in {"man", "woman"}:
        return _abstain("old partner gender is not a known binary value")
    if new_partner not in {"man", "woman", "nonbinary"}:
        return _abstain("new partner gender is invalid")
    if old_gender not in {"man", "woman", "nonbinary"} or new_gender not in {
        "man",
        "woman",
        "nonbinary",
    }:
        return _abstain("self gender is invalid")
    if old_gender != new_gender:
        return _abstain("self-gender change requires prose review")
    if old_partner == new_partner:
        return _abstain("partner gender did not change")
    if not isinstance(old_persona_text, str):
        return _abstain("persona text is not a string")

    matches = list(_PARTNER_CLAUSE.finditer(old_persona_text))
    mentions = list(_PARTNER_MENTION.finditer(old_persona_text))
    if len(matches) != 1 or len(mentions) != 1:
        return _abstain("expected exactly one standalone partner clause")
    match = matches[0]
    expected_partner_span = (match.end("gender") + 1, match.end("gender") + 8)
    if mentions[0].span() != expected_partner_span:
        return _abstain("partner mention is not the eligible clause")
    if _PRONOUN.search(old_persona_text):
        return _abstain("persona text contains pronoun references")
    gender_mentions = list(_GENDER_MENTION.finditer(old_persona_text))
    if len(gender_mentions) != 1 or gender_mentions[0].span() != match.span("gender"):
        return _abstain("persona text contains additional gender mentions")

    expected = "mandlig" if old_partner == "man" else "kvindelig"
    if match.group("gender") != expected:
        return _abstain("standalone partner clause does not match old source")

    proposed = (
        old_persona_text[: match.start()]
        + match.group("indent")
        + "Personen har en partner."
        + match.group("trailing")
        + old_persona_text[match.end() :]
    )
    if len(proposed) < 300 or len(proposed) > 900:
        return _abstain("proposed prose is outside the 300–900 character limit")
    return PartnerProseProposal(
        proposed,
        f"Partner gender changed from {old_partner!r} to {new_partner!r}; "
        "one explicit standalone clause was neutralised.",
        None,
    )


def _normalise_partner_gender(value: str) -> str | None:
    """Map supported source spellings to Danish gender labels."""
    return {"male": "man", "female": "woman"}.get(value, value)


def _abstain(reason: str) -> PartnerProseProposal:
    """Return a proposal that makes no text change."""
    return PartnerProseProposal(None, None, reason)
