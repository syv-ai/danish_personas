"""Offline contracts for conservative persona-prose change triage."""

import polars as pl
import pytest

from danish_personas.release.prose_triage import triage_prose_changes


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


def test_safe_mixed_changes_are_only_marked_reviewed_if_all_safe() -> None:
    original = _frame()
    repaired = _frame(
        origin_country_code=["NO", "SE"],
        job_title=["Sygeplejerske!", "lærer"],
        hobbies_and_interests=[
            ["MUSIK.", "LÆSNING", "musik"],
            ["have", "løb"],
        ],
    )

    result = triage_prose_changes(
        original,
        repaired,
        {
            "p1": ["metadata_only", "lexical_formatting_only", "list_deduplication_or_formatting"]
        },
    )

    assert result["personas"]["p1"]["classification"] == "semantic_equivalence_reviewed"
    assert result["personas"]["p1"]["changed_fields"] == [
        "origin_country_code",
        "job_title",
        "hobbies_and_interests",
    ]
    assert result["counts"] == {
        "semantic_equivalence_reviewed": 1,
        "needs_prose_review_or_regeneration": 0,
    }


def test_any_unsafe_change_in_mixed_row_requires_prose_review() -> None:
    original = _frame()
    repaired = _frame(
        job_title=["Sygeplejerske", "lærer"],
        marital_status=["married", "married"],
    )

    result = triage_prose_changes(
        original,
        repaired,
        {"p1": ["lexical_formatting_only", "unresolved_attribute_review"]},
    )

    assert result["personas"]["p1"]["classification"] == "needs_prose_review_or_regeneration"


def test_reason_id_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="does not match frame diff"):
        triage_prose_changes(
            _frame(),
            _frame(job_title=["Sygeplejerske", "lærer"]),
            {"p2": ["lexical_formatting_only"]},
        )


def test_changed_persona_text_fails_closed() -> None:
    with pytest.raises(ValueError, match="does not match frame diff"):
        triage_prose_changes(
            _frame(),
            _frame(persona_text=["Revideret tekst", "Tekst 2"]),
            {"p1": ["lexical_formatting_only"]},
        )


def test_matching_persona_text_change_is_never_safe() -> None:
    result = triage_prose_changes(
        _frame(),
        _frame(persona_text=["Revideret tekst", "Tekst 2"]),
        {"p1": ["unresolved_attribute_review"]},
    )

    assert result["personas"]["p1"]["classification"] == "needs_prose_review_or_regeneration"
