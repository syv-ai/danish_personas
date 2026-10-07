"""Conservative offline proposals for explicit Danish legal-status clauses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

LegalDetail = Literal["married", "separated"]

# These are deliberately complete, standalone clauses. They are not intended to
# detect ordinary narrative references to marriage or separation.
_CLAUSE = re.compile(
    r"(?m)^[ \t]*(?:(?P<label>Civilstand: )|(?P<state>Personen er ))"
    r"(?P<legal>gift|separeret)\.[ \t]*$"
)
_ALTERNATE_PRONOUN = re.compile(
    r"\b(?:han|hun|de|ham|hende|dem|he|she|they|him|her|them)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class LegalProseProposal:
    """A source-grounded edit proposal, or a fail-closed abstention."""

    proposed_text: str | None
    evidence: str | None
    abstention_reason: str | None

    @property
    def abstained(self) -> bool:
        """Whether no prose edit was proposed.

        Returns:
            True when this proposal contains no edited prose.
        """
        return self.proposed_text is None


def propose_legal_prose_repair(
    *,
    old_marital_status: str,
    new_marital_status: str,
    old_legal_status_detail: str | None,
    new_legal_status_detail: str | None,
    current_relationship_status: str,
    old_persona_text: str,
) -> LegalProseProposal:
    """Propose a narrowly scoped legal-clause correction, otherwise abstain.

    The function edits only one complete standalone ``Civilstand`` or
    ``Personen er`` sentence. It never infers an individual legal state from a
    merged source category whose detail is absent.

    Returns:
        A validated prose edit or an explicit fail-closed abstention.
    """
    reason = _validate_inputs(
        old_marital_status=old_marital_status,
        new_marital_status=new_marital_status,
        old_detail=old_legal_status_detail,
        new_detail=new_legal_status_detail,
        relationship=current_relationship_status,
        text=old_persona_text,
    )
    if reason:
        return _abstain(reason)

    if _ALTERNATE_PRONOUN.search(old_persona_text):
        return _abstain("persona text contains alternate pronouns")
    clause = _replacement_for_clause(
        text=old_persona_text,
        old_detail=old_legal_status_detail,
        new_detail=new_legal_status_detail,
    )
    if isinstance(clause, str):
        return _abstain(clause)
    match, replacement = clause
    proposed = (
        old_persona_text[: match.start()]
        + replacement
        + old_persona_text[match.end() :]
    )
    if proposed == old_persona_text:
        return _abstain("prose is already current")
    if len(proposed) < 300 or len(proposed) > 900:
        return _abstain("proposed prose is outside the 300–900 character limit")
    evidence = (
        f"Source legal detail changed from {old_legal_status_detail!r} to "
        f"{new_legal_status_detail!r}; one standalone clause was replaced."
    )
    return LegalProseProposal(proposed, evidence, None)


def _abstain(reason: str) -> LegalProseProposal:
    """Create an explicit no-edit outcome.

    Returns:
        A proposal that records the supplied abstention reason.
    """
    return LegalProseProposal(None, None, reason)


def _replacement_for_clause(
    *, text: str, old_detail: str, new_detail: str | None
) -> tuple[re.Match[str], str] | str:
    """Validate the sole legal clause and return its replacement.

    Returns:
        A matched clause and replacement text, or an abstention reason.
    """
    matches = list(_CLAUSE.finditer(text))
    if len(matches) != 1:
        return "expected exactly one standalone legal clause"
    match = matches[0]
    if match.group("legal") != _danish(old_detail):
        return "standalone legal clause does not match old source"

    if new_detail is None:
        # Missing detail is not evidence of either state; retain only a neutral
        # description of the two possible legal states.
        return match, "Civilstand: gift eller separeret."
    if old_detail == new_detail:
        return "legal detail did not change"
    return match, _render_clause(match, _danish(new_detail))


def _danish(detail: str) -> str:
    """Map a canonical detail to its exact Danish clause word.

    Returns:
        The Danish word corresponding to the canonical legal detail.
    """
    return {"married": "gift", "separated": "separeret"}[detail]


def _render_clause(match: re.Match[str], word: str) -> str:
    """Keep the matched clause form while changing only its legal word.

    Returns:
        The replacement sentence with the original clause style.
    """
    if match.group("label"):
        return f"Civilstand: {word}."
    return f"Personen er {word}."


def _validate_inputs(
    *,
    old_marital_status: str,
    new_marital_status: str,
    old_detail: str | None,
    new_detail: str | None,
    relationship: str,
    text: str,
) -> str | None:
    """Return a reason for inconsistent or unsafe source and prose inputs.

    Returns:
        An explanation when inputs are unsafe, otherwise None.
    """
    if not isinstance(text, str) or not 300 <= len(text) <= 900:
        return "persona prose must contain 300–900 characters"
    if (
        old_marital_status != "married_or_separated"
        or new_marital_status != "married_or_separated"
    ):
        return "both source marital statuses must be married_or_separated"
    if old_detail not in {"married", "separated"}:
        return "old legal detail must be married or separated"
    if new_detail not in {None, "married", "separated"}:
        return "new legal detail must be married, separated, or null"
    if relationship not in {"partnered", "not_partnered"}:
        return "relationship status is invalid"
    if new_detail == "married" and relationship != "partnered":
        return "married legal detail conflicts with relationship status"
    return None
