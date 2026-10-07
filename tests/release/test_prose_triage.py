"""Offline contracts for conservative persona-prose change triage."""

import polars as pl
import pytest

from danish_personas.release.prose_triage import triage_prose_changes


def test_actual_field_reasons_and_marker_only_persona_are_supported() -> None:
    """Accept actual changed-field reasons and recognised marker-only IDs."""
    original = _frame()
    repaired = _frame(
        origin_country_code=["NO", "SE"],
        hobbies_and_interests=[["MUSIK.", "LÆSNING", "musik"], ["have", "løb"]],
        marital_status=["married", "married"],
    )

    result = triage_prose_changes(
        original,
        repaired,
        {
            "p1": ["origin_country_code", "hobbies_and_interests", "marital_status"],
            "p2": ["gender_prose_review"],
        },
    )

    assert _persona_result(result, "p1")["classification"] == (
        "needs_prose_review_or_regeneration"
    )
    assert _persona_result(result, "p1")["changed_fields"] == [
        "origin_country_code",
        "hobbies_and_interests",
        "marital_status",
    ]
    assert _persona_result(result, "p2")["classification"] == (
        "needs_prose_review_or_regeneration"
    )
    assert _persona_result(result, "p2")["changed_fields"] == []
    assert result["counts"] == {
        "semantic_equivalence_reviewed": 0,
        "needs_prose_review_or_regeneration": 2,
    }


def _persona_result(result: object, persona_id: str) -> dict[str, object]:
    """Narrow the triage result to one typed persona record."""
    assert isinstance(result, dict)
    personas = result.get("personas")
    assert isinstance(personas, dict)
    persona = personas.get(persona_id)
    assert isinstance(persona, dict)
    assert all(isinstance(key, str) for key in persona)
    return {key: value for key, value in persona.items() if isinstance(key, str)}


def _frame(**updates: list[object]) -> pl.DataFrame:
    values: dict[str, list[object]] = {
        "persona_id": ["p1", "p2"],
        "origin_country_code": ["DK", "SE"],
        "job_title": ["sygeplejerske", "lærer"],
        "hobbies_and_interests": [["læsning", "musik"], ["have", "løb"]],
        "marital_status": ["single", "married"],
        "persona_text": ["Tekst 1", "Tekst 2"],
    }
    values.update(updates)
    return pl.DataFrame(values)


def test_arbitrary_extra_ledger_id_fails_closed() -> None:
    """Reject extra ledger IDs without a recognised unsafe marker."""
    with pytest.raises(ValueError, match="does not match frame diff"):
        triage_prose_changes(_frame(), _frame(), {"p1": ["unrecognised_marker"]})


def test_changed_persona_text_fails_closed() -> None:
    """Reject any attempt to modify the original persona prose."""
    with pytest.raises(ValueError, match="Persona prose must not change"):
        triage_prose_changes(
            _frame(),
            _frame(persona_text=["Revideret tekst", "Tekst 2"]),
            {"p1": ["persona_text"]},
        )


def test_lexical_identity_preserves_internal_punctuation_and_words() -> None:
    """Preserve internal punctuation and distinct lexical identities."""
    original = _frame()
    repaired = _frame(job_title=["Sygeplejerske!", "lærer"])

    result = triage_prose_changes(original, repaired, {"p1": ["job_title"]})
    assert _persona_result(result, "p1")["classification"] == (
        "semantic_equivalence_reviewed"
    )

    changed = _frame(job_title=["sygeplejerske-assistent", "lærer"])
    result = triage_prose_changes(original, changed, {"p1": ["job_title"]})
    assert _persona_result(result, "p1")["classification"] == (
        "needs_prose_review_or_regeneration"
    )


def test_reason_id_mismatch_fails_closed() -> None:
    """Reject missing and mismatched persona IDs in the reason ledger."""
    with pytest.raises(ValueError, match="does not match frame diff"):
        triage_prose_changes(
            _frame(),
            _frame(job_title=["Sygeplejerske", "lærer"]),
            {"p2": ["job_title"]},
        )
