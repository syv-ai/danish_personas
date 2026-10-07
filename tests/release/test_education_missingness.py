"""Offline contracts for source-backed education missingness repair."""

import hashlib

import polars as pl
import pytest

from danish_personas.release.education_missingness import (
    EDUCATION_MISSINGNESS_METHOD,
    EducationMissingnessError,
    allocate_education_missingness,
)


def test_impossible_quota_and_lower_target_fail() -> None:
    """Infeasible source-backed targets fail instead of fabricating categories."""
    frame = _frame()
    with pytest.raises(EducationMissingnessError, match="Insufficient eligible"):
        allocate_education_missingness(
            frame=frame,
            ras209_joint_unpooled=_source(),
            official_age20plus_h90_share=0.9,
        )

    with pytest.raises(EducationMissingnessError, match="fewer than existing"):
        allocate_education_missingness(
            frame=frame,
            ras209_joint_unpooled=_source(),
            official_age20plus_h90_share=0.0,
        )


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "persona_id": [
                "existing-h90",
                "supported-a-1",
                "supported-a-2",
                "supported-a-3",
                "supported-b-1",
                "supported-b-2",
                "unsupported-zero",
                "unsupported-suppressed",
                "under-20",
            ],
            "age": [25, 26, 27, 28, 31, 32, 36, 41, 19],
            "municipality_code": [101, 101, 101, 101, 102, 102, 103, 104, 101],
            "sex": [
                "female",
                "female",
                "female",
                "female",
                "male",
                "male",
                "female",
                "male",
                "female",
            ],
            "labour_market_status": [
                "employed",
                "employed",
                "employed",
                "employed",
                "unemployed",
                "unemployed",
                "employed",
                "employed",
                "employed",
            ],
            "education_source_code": [
                "H90",
                "H10",
                "H20-H35",
                "H40-H80",
                "H10",
                "H40-H80",
                "H10",
                "H10",
                "H10",
            ],
            "education_level": [
                "not_stated",
                "primary",
                "secondary_or_vocational",
                "higher_education",
                "primary",
                "higher_education",
                "primary",
                "primary",
                "primary",
            ],
            "education_resolution": ["ras209_age_band"] * 9,
            "persona": [f"Prose {index}" for index in range(9)],
            "origin_country": ["Danmark"] * 9,
        }
    )


def _source() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "age_band": ["25-29", "25-29", "30-34", "30-34", "35-39", "40-44", "16-19"],
            "municipality_code": [101, 101, 102, 102, 103, 104, 101],
            "sex": ["female", "female", "male", "male", "female", "male", "female"],
            "education_source_code": ["H90", "H10", "H90", "H10", "H90", "H90", "H90"],
            "labour_market_status": [
                "employed",
                "employed",
                "unemployed",
                "unemployed",
                "employed",
                "employed",
                "employed",
            ],
            "count": [3, 97, 1, 99, 0, 4, 10],
            "suppressed": [False, False, False, False, False, True, False],
        }
    )


def test_preserves_valid_rows_and_all_non_education_fields() -> None:
    """The candidate changes only source code and public education label."""
    frame = _frame()
    repaired, _ = allocate_education_missingness(
        frame=frame,
        ras209_joint_unpooled=_source(),
        official_age20plus_h90_share=0.5,
        seed=7,
    )

    assert frame["persona_id"].to_list() == repaired["persona_id"].to_list()
    assert frame.drop("education_source_code", "education_level").equals(
        repaired.drop("education_source_code", "education_level")
    )
    for persona_id in _changed_ids(before=frame, after=repaired):
        row = repaired.filter(pl.col("persona_id") == persona_id)
        assert row["education_source_code"].item() == "H90"
        assert row["education_level"].item() == "not_stated"
    unchanged = set(frame["persona_id"]) - _changed_ids(before=frame, after=repaired)
    for persona_id in unchanged:
        before = frame.filter(pl.col("persona_id") == persona_id)
        after = repaired.filter(pl.col("persona_id") == persona_id)
        assert before.equals(after)


def _changed_ids(*, before: pl.DataFrame, after: pl.DataFrame) -> set[str]:
    changed: set[str] = set()
    for before_row, after_row in zip(before.to_dicts(), after.to_dicts(), strict=True):
        if before_row["education_source_code"] != after_row["education_source_code"]:
            changed.add(str(before_row["persona_id"]))
    return changed


def test_selection_is_deterministic_and_order_invariant() -> None:
    """Persona-ID ranking makes selected rows independent of input order."""
    frame = _frame()
    shuffled = frame.gather([3, 8, 1, 0, 6, 2, 5, 4, 7])

    first, first_report = allocate_education_missingness(
        frame=frame,
        ras209_joint_unpooled=_source(),
        official_age20plus_h90_share=0.5,
        seed=19,
    )
    second, second_report = allocate_education_missingness(
        frame=shuffled,
        ras209_joint_unpooled=_source(),
        official_age20plus_h90_share=0.5,
        seed=19,
    )

    assert _changed_ids(before=frame, after=first) == _changed_ids(
        before=shuffled, after=second
    )
    assert (
        first_report["changed_persona_id_sha256"]
        == second_report["changed_persona_id_sha256"]
    )
    assert second["persona_id"].to_list() == shuffled["persona_id"].to_list()


def test_source_cell_structural_zero_and_supported_selection() -> None:
    """Only known rows with positive unsuppressed H90 source cells are changed."""
    frame = _frame()
    repaired, report = allocate_education_missingness(
        frame=frame,
        ras209_joint_unpooled=_source(),
        official_age20plus_h90_share=0.5,
        seed=7,
    )

    changed = _changed_ids(before=frame, after=repaired)
    assert changed == _expected_changed_ids(seed=7)
    assert "unsupported-zero" not in changed
    assert "unsupported-suppressed" not in changed
    assert (
        repaired.filter(pl.col("persona_id") == "under-20")[
            "education_source_code"
        ].item()
        == "H10"
    )
    assert report["source_target_h90"] == 4
    assert report["current_h90"] == 1
    assert report["after_h90"] == 4
    assert report["changed_rows"] == 3
    assert report["source_support"]["after"] == {
        "h90_rows": 4,
        "supported_h90_rows": 4,
        "unsupported_h90_rows": 0,
    }
    assert "not observed person-level" in report["synthetic_allocation_note"]


def _expected_changed_ids(*, seed: int) -> set[str]:
    supported_a = ["supported-a-1", "supported-a-2", "supported-a-3"]
    supported_b = ["supported-b-1", "supported-b-2"]
    return {*_ranked(supported_a, seed=seed)[:2], *_ranked(supported_b, seed=seed)[:1]}


def _ranked(persona_ids: list[str], *, seed: int) -> list[str]:
    return sorted(
        persona_ids,
        key=lambda persona_id: hashlib.sha256(
            f"{EDUCATION_MISSINGNESS_METHOD}:{seed}:{persona_id}".encode()
        ).hexdigest(),
    )
