"""Offline contracts for conservative partner-gender prose proposals."""

from danish_personas.release.prose_partner_rules import (
    PartnerProseProposal,
    propose_partner_gender_prose_repair,
)


def test_abstains_for_narrative_or_multiple_partner_mentions() -> None:
    """Abstain when prose is not exactly one standalone partner clause."""
    for text in (
        _text("Personen har en mandlig partner, som hun holder af."),
        _text() + " Partneren støtter Personen.",
        _text("Personen har en mandlig partner. Personen har en partner."),
        _text("I familien har Personen en mandlig partner."),
        _text("Personen har en mandlig ægtefælle."),
    ):
        assert _propose(text).abstained


def _propose(
    text: str,
    *,
    old: str = "male",
    new: str = "nonbinary",
    old_gender: str = "woman",
    new_gender: str = "woman",
) -> PartnerProseProposal:
    """Propose a repair using the test's default unchanged self gender.

    Returns:
        The repair proposal or fail-closed abstention.
    """
    return propose_partner_gender_prose_repair(
        old_partner_gender=old,
        new_partner_gender=new,
        old_gender=old_gender,
        new_gender=new_gender,
        old_persona_text=text,
    )


def _text(clause: str = "Personen har en mandlig partner.") -> str:
    """Build a sufficiently long persona around a configurable clause.

    Returns:
        Persona text containing the supplied clause.
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


def test_abstains_for_pronouns_and_self_gender_changes() -> None:
    """Abstain for pronouns or a change to the persona's own gender."""
    assert _propose(_text() + " Hun læser ofte.").abstained
    assert _propose(_text(), old_gender="woman", new_gender="nonbinary").abstained


def test_abstains_for_source_mismatch_invalid_or_unchanged_inputs() -> None:
    """Abstain when source values are invalid, unchanged, or mismatched."""
    assert _propose(_text("Personen har en kvindelig partner.")).abstained
    assert _propose(_text(), old="nonbinary").abstained
    assert _propose(_text(), old="male", new="man").abstained
    assert _propose(_text(), old="female", new="female").abstained
    assert _propose("Personen har en mandlig partner.").abstained


def test_accepts_female_to_man_source_change() -> None:
    """Allow the other binary source spelling to change to man."""
    result = _propose(
        _text("Personen har en kvindelig partner."), old="female", new="man"
    )

    assert result.proposed_text is not None
    assert "Personen har en partner." in result.proposed_text


def test_neutralises_only_exact_standalone_clause_and_returns_evidence() -> None:
    """Neutralise the eligible clause and retain source-change evidence."""
    text = _text()
    result = _propose(text)

    assert not result.abstained
    assert result.proposed_text == text.replace(
        "Personen har en mandlig partner.", "Personen har en partner."
    )
    assert result.evidence and "nonbinary" in result.evidence
    assert len(result.proposed_text) == len(text) - len("mandlig ")


def test_safe_neutralisation_is_idempotent() -> None:
    """A previously neutralised clause does not produce another edit."""
    first = _propose(_text())
    assert first.proposed_text is not None
    assert _propose(first.proposed_text).abstained
