"""Offline contracts for conservative legal-clause prose proposals."""

from danish_personas.release.prose_legal_rules import (
    LegalProseProposal,
    propose_legal_prose_repair,
)


def _text(clause: str = "Personen er gift.") -> str:
    """Build valid-length fixture prose with a configurable legal clause.

    Returns:
        Persona prose containing the requested opening clause.
    """
    return (
        f"{clause}\n"
        + "Personaen har et roligt hverdagsliv og sætter pris på gode samtaler, "
        "kulturelle oplevelser og tid til egne interesser. Arbejdet giver plads "
        "til både fordybelse og samarbejde, og fritiden bruges gerne på læsning, "
        "madlavning og ture i nærområdet. Personen foretrækker en praktisk tilgang "
        "til nye opgaver og møder andre med venlighed. Værdierne er stabilitet, "
        "nysgerrighed og omtanke, og små planer for fremtiden giver motivation. "
        "En god uge rummer variation, pauser og meningsfulde aktiviteter."
    )


def _propose(
    text: str,
    *,
    old: str = "married",
    new: str | None = "separated",
    relationship: str = "partnered",
) -> LegalProseProposal:
    """Propose a legal repair using consistent default source statuses.

    Returns:
        The legal prose proposal for the supplied fixture.
    """
    return propose_legal_prose_repair(
        old_marital_status="married_or_separated",
        new_marital_status="married_or_separated",
        old_legal_status_detail=old,
        new_legal_status_detail=new,
        current_relationship_status=relationship,
        old_persona_text=text,
    )


def test_changes_only_exact_standalone_clause_and_returns_evidence() -> None:
    """Replace one matching standalone clause and include source evidence."""
    text = _text("Civilstand: gift.")
    result = _propose(text)

    assert not result.abstained
    assert result.proposed_text == text.replace(
        "Civilstand: gift.", "Civilstand: separeret."
    )
    assert result.evidence and "married" in result.evidence
    assert 300 <= len(result.proposed_text) <= 900


def test_merged_detail_replaces_known_clause_with_neutral_source_description() -> None:
    """Avoid inferring either legal state when the new detail is merged."""
    result = _propose(_text(), new=None)

    assert result.proposed_text is not None
    assert "Civilstand: gift eller separeret." in result.proposed_text
    assert "Personen er gift." not in result.proposed_text


def test_abstains_for_narrative_mentions_and_embedded_or_multiple_clauses() -> None:
    """Reject clauses that are not one standalone sentence."""
    for text in (
        _text("Personen er gift, men foretrækker ro."),
        _text("I familien er Personen er gift."),
        _text() + "\nCivilstand: gift.",
        _text("Gift."),
    ):
        assert _propose(text).abstained


def test_abstains_on_clause_source_mismatch_or_alternate_pronouns() -> None:
    """Reject mismatched legal clauses and alternate-pronoun prose."""
    assert _propose(_text("Personen er separeret.")).abstained
    assert _propose(_text() + " Hun holder af naturen.").abstained


def test_abstains_for_inconsistent_source_inputs_and_invalid_length() -> None:
    """Fail closed for inconsistent statuses and out-of-range prose."""
    result = propose_legal_prose_repair(
        old_marital_status="never_married",
        new_marital_status="married_or_separated",
        old_legal_status_detail="married",
        new_legal_status_detail="separated",
        current_relationship_status="partnered",
        old_persona_text=_text(),
    )
    assert result.abstained

    assert _propose("Personen er gift.").abstained


def test_safe_change_is_idempotent_as_a_second_old_to_new_proposal() -> None:
    """Abstain when the proposed legal detail is already current."""
    first = _propose(_text())
    assert first.proposed_text is not None

    second = propose_legal_prose_repair(
        old_marital_status="married_or_separated",
        new_marital_status="married_or_separated",
        old_legal_status_detail="separated",
        new_legal_status_detail="separated",
        current_relationship_status="partnered",
        old_persona_text=first.proposed_text,
    )
    assert second.abstained
