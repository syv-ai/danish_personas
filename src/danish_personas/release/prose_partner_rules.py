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

    Returns:
        A prose proposal when the single eligible clause can be safely
        neutralised; otherwise a fail-closed abstention.
    """
    old_partner = _normalise_partner_gender(old_partner_gender)
    new_partner = _normalise_partner_gender(new_partner_gender)
    reason = _input_abstention_reason(
        old_partner=old_partner,
        new_partner=new_partner,
        old_gender=old_gender,
        new_gender=new_gender,
        old_persona_text=old_persona_text,
    )
    if reason is not None:
        return _abstain(reason)

    match, reason = _eligible_partner_clause(
        old_persona_text=old_persona_text, old_partner=old_partner
    )
    if reason is not None:
        return _abstain(reason)
    assert match is not None

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


def _input_abstention_reason(
    *,
    old_partner: str | None,
    new_partner: str | None,
    old_gender: str,
    new_gender: str,
    old_persona_text: str,
) -> str | None:
    """Return why supplied source values or text cannot support a repair.

    Returns:
        An abstention reason, or ``None`` when the inputs are eligible for
        prose inspection.
    """
    if old_partner not in {"man", "woman"}:
        return "old partner gender is not a known binary value"
    if new_partner not in {"man", "woman", "nonbinary"}:
        return "new partner gender is invalid"
    valid_genders = {"man", "woman", "nonbinary"}
    if old_gender not in valid_genders or new_gender not in valid_genders:
        return "self gender is invalid"
    if old_gender != new_gender:
        return "self-gender change requires prose review"
    if old_partner == new_partner:
        return "partner gender did not change"
    if not isinstance(old_persona_text, str):
        return "persona text is not a string"
    return None


def _eligible_partner_clause(
    *, old_persona_text: str, old_partner: str | None
) -> tuple[re.Match[str] | None, str | None]:
    """Find the sole source-matching standalone clause, or explain abstention.

    Returns:
        The matching clause and no reason when eligible, or ``None`` and an
        abstention reason when any fail-closed prose check fails.
    """
    matches = list(_PARTNER_CLAUSE.finditer(old_persona_text))
    mentions = list(_PARTNER_MENTION.finditer(old_persona_text))
    if len(matches) != 1 or len(mentions) != 1:
        return None, "expected exactly one standalone partner clause"
    match = matches[0]
    expected_partner_span = (match.end("gender") + 1, match.end("gender") + 8)
    if mentions[0].span() != expected_partner_span:
        return None, "partner mention is not the eligible clause"
    if _PRONOUN.search(old_persona_text):
        return None, "persona text contains pronoun references"
    gender_mentions = list(_GENDER_MENTION.finditer(old_persona_text))
    if len(gender_mentions) != 1 or gender_mentions[0].span() != match.span("gender"):
        return None, "persona text contains additional gender mentions"

    expected = "mandlig" if old_partner == "man" else "kvindelig"
    if match.group("gender") != expected:
        return None, "standalone partner clause does not match old source"
    return match, None


def _normalise_partner_gender(value: str) -> str | None:
    """Map supported source spellings to Danish gender labels.

    Returns:
        The canonical label when the value is supported; otherwise the input.
    """
    return {"male": "man", "female": "woman"}.get(value, value)


def _abstain(reason: str) -> PartnerProseProposal:
    """Create an abstention with its fail-closed reason.

    Returns:
        A proposal containing no text or evidence and the supplied reason.
    """
    return PartnerProseProposal(None, None, reason)
