"""Offline contracts for deterministic generated-list completion."""

from __future__ import annotations

import typing as t

import polars as pl
import pytest

from danish_personas.release.list_completion import complete_generated_lists


def test_completes_real_like_duplicate_fields_preserving_rows_and_prose() -> None:
    """Complete only the three-persona, five-field uniqueness failures."""
    frame = _three_case_frame()

    completed, report = complete_generated_lists(frame)

    assert completed["persona_id"].to_list() == frame["persona_id"].to_list()
    assert completed["status"].to_list() == frame["status"].to_list()
    assert completed["persona"].to_list() == frame["persona"].to_list()
    assert completed["sex"].to_list() == frame["sex"].to_list()
    assert (
        completed["origin_country_da"].to_list() == frame["origin_country_da"].to_list()
    )
    assert (
        completed["skills_and_expertise"][1].to_list()
        == frame["skills_and_expertise"][1].to_list()
    )

    changed_pairs = {
        (change["persona_id"], change["field"]) for change in report["changes"]
    }
    assert changed_pairs == {
        ("p-1", "skills_and_expertise"),
        ("p-1", "hobbies_and_interests"),
        ("p-2", "hobbies_and_interests"),
        ("p-3", "skills_and_expertise"),
        ("p-3", "hobbies_and_interests"),
    }
    assert len(report["changes"]) == 5
    assert {item["persona_id"] for item in report["prose_review"]} == {
        "p-1",
        "p-2",
        "p-3",
    }
    report_text = repr(report)
    assert "female" not in report_text
    assert "Danmark" not in report_text
    assert "Uændret prose" not in report_text
    for change in report["changes"]:
        assert change["original"] != change["new"]
        assert change["new"][0] == change["original"][0]
        assert len(change["new"]) == len(change["original"])
        assert change["seed"] > 0
        assert change["method"] == (
            "synthetic_list_completion_v1_seeded_by_persona_id_and_field"
        )
        assert change["bank_version"] == report["bank_version"]


def test_completion_is_idempotent_and_independent_of_row_order() -> None:
    """Seed from persona ID and field, not from row position."""
    frame = _three_case_frame()

    completed, report = complete_generated_lists(frame)
    repeated, repeated_report = complete_generated_lists(completed)
    reordered_completed, _ = complete_generated_lists(frame.reverse())

    assert repeated.to_dicts() == completed.to_dicts()
    assert repeated_report["changes"] == []
    assert repeated_report["prose_review"] == []
    assert len(report["changes"]) == 5
    assert _lists_by_persona(reordered_completed) == _lists_by_persona(completed)


def test_repairs_hobby_format_without_treating_eller_as_defect() -> None:
    """Repair objective hobby-format failures without marital-disjunction rules."""
    frame = pl.DataFrame(
        [
            _row(
                "p-format",
                skills_and_expertise=[
                    "analyse eller tal",
                    "analyse",
                    "formidling",
                ],
                hobbies_and_interests=["Løb", "musik.", "gåture "],
            )
        ]
    )

    completed, report = complete_generated_lists(frame)

    assert completed["skills_and_expertise"][0].to_list() == [
        "analyse eller tal",
        "analyse",
        "formidling",
    ]
    assert completed["hobbies_and_interests"][0].to_list() == [
        "løb",
        "musik",
        "gåture ",
    ]
    assert [(item["persona_id"], item["field"]) for item in report["changes"]] == [
        ("p-format", "hobbies_and_interests")
    ]


def test_unique_punctuated_whitespace_disjunctive_lists_stay_unchanged() -> None:
    """Leave unique list text untouched when objective list checks pass."""
    frame = pl.DataFrame(
        [
            _row(
                "p-untouched",
                skills_and_expertise=[
                    "analyse.",
                    "tal eller data",
                    " samarbejde ",
                ],
                hobbies_and_interests=[
                    "musik eller radio",
                    "gåture ",
                    "brætspil…",
                ],
            )
        ]
    )

    completed, report = complete_generated_lists(frame)

    assert completed.to_dicts() == frame.to_dicts()
    assert report["changes"] == []
    assert report["prose_review"] == []


def test_rejects_invalid_ids_list_types_and_sizes() -> None:
    """Fail closed for structures that are not safe synthetic-list repairs."""
    duplicate_ids = pl.DataFrame([_row("p-1"), _row("p-1")])
    non_string_id = pl.DataFrame([_row("")])
    non_list = pl.DataFrame([_row("p-1", skills_and_expertise="analyse")], strict=False)
    undersized = pl.DataFrame(
        [_row("p-1", skills_and_expertise=["analyse", "formidling"])]
    )

    for invalid_frame in (
        duplicate_ids,
        non_string_id,
        non_list,
        undersized,
    ):
        with pytest.raises(ValueError):
            complete_generated_lists(invalid_frame)


def _row(
    persona_id: str,
    *,
    skills_and_expertise: list[str] | str | None = None,
    hobbies_and_interests: list[str] | None = None,
) -> dict[str, object]:
    return {
        "persona_id": persona_id,
        "status": "candidate",
        "sex": "female",
        "origin_country_da": "Danmark",
        "persona": f"Uændret prose for {persona_id}.",
        "skills_and_expertise": skills_and_expertise
        if skills_and_expertise is not None
        else ["analyse", "formidling", "samarbejde"],
        "hobbies_and_interests": hobbies_and_interests
        if hobbies_and_interests is not None
        else ["musik", "læsning", "gåture"],
    }


def _three_case_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            _row(
                "p-1",
                skills_and_expertise=["analyse", "analyse", "formidling"],
                hobbies_and_interests=["musik", "musik", "læsning"],
            ),
            _row(
                "p-2",
                skills_and_expertise=["formidling", "samarbejde", "dokumentation"],
                hobbies_and_interests=["læsning", "læsning", "gåture"],
            ),
            _row(
                "p-3",
                skills_and_expertise=[
                    "planlægning",
                    "planlægning",
                    "samarbejde",
                    "samarbejde",
                ],
                hobbies_and_interests=[
                    "brætspil",
                    "brætspil",
                    "madlavning",
                    "madlavning",
                ],
            ),
        ]
    )


def _lists_by_persona(frame: pl.DataFrame) -> dict[str, tuple[list[str], list[str]]]:
    return {
        str(row["persona_id"]): (
            t.cast(list[str], row["skills_and_expertise"]),
            t.cast(list[str], row["hobbies_and_interests"]),
        )
        for row in frame.to_dicts()
    }
