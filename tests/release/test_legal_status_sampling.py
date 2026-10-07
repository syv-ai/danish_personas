"""Contracts for synthetic legal-status detail sampling."""

import polars as pl
import pytest

from danish_personas.release.legal_status_sampling import apply_legal_status_sampling


def test_fills_3790_source_g_rows_with_concrete_fine_detail() -> None:
    """Fill every ambiguous G row while preserving absolute counts."""
    frame = _frame(
        [
            _row(
                f"g-{index:04d}",
                current_relationship_status=(
                    "not_partnered" if index % 3 == 0 else "partnered"
                ),
            )
            for index in range(3790)
        ]
    )

    repaired, report = apply_legal_status_sampling(frame, seed=17)

    assert repaired.height == 3790
    assert repaired.columns == frame.columns
    assert repaired["persona_id"].to_list() == frame["persona_id"].to_list()
    assert repaired["marital_status"].to_list() == ["married_or_separated"] * 3790
    details = repaired["legal_status_detail"].to_list()
    assert set(details) <= {"married", "separated"}
    assert "married_or_separated" not in details
    assert all("/" not in detail for detail in details)
    not_partnered = repaired.filter(
        pl.col("current_relationship_status") == "not_partnered"
    )
    assert (
        not_partnered["legal_status_detail"].to_list()
        == ["separated"] * not_partnered.height
    )
    assert report["row_count"] == 3790
    counts = report["counts"]
    assert isinstance(counts, dict)
    absolute = counts["absolute"]
    assert isinstance(absolute, dict)
    assert absolute["married_or_separated_rows"] == 3790
    assert absolute["changed_rows"] == 3790
    assert len(report["changed"]) == 3790
    assert report["prose_review_ids"] == frame["persona_id"].to_list()
    assert "50/50" in report["synthetic_method"]
    assert "not an official Statistics Denmark proportion" in report["synthetic_method"]


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema_overrides={
            "persona_id": pl.String,
            "marital_status": pl.String,
            "legal_status_detail": pl.String,
            "current_relationship_status": pl.String,
            "persona": pl.String,
            "other_value": pl.Int64,
        },
    )


def _row(
    persona_id: str | None,
    *,
    marital_status: str = "married_or_separated",
    legal_status_detail: str | None = None,
    current_relationship_status: str = "partnered",
    persona: str = "Original persona prose.",
    other_value: int = 1,
) -> dict[str, object]:
    return {
        "persona_id": persona_id,
        "marital_status": marital_status,
        "legal_status_detail": legal_status_detail,
        "current_relationship_status": current_relationship_status,
        "persona": persona,
        "other_value": other_value,
    }


def test_invalid_rows_fail_closed() -> None:
    """Reject ambiguous, contradictory, or unsupported inputs."""
    cases = [
        ([_row("dup"), _row("dup")], "unique"),
        ([_row(None)], "non-empty"),
        ([_row("bad-rel", current_relationship_status="complicated")], "Invalid"),
        ([_row("bad-marital", marital_status="cohabiting")], "Invalid"),
        (
            [
                _row(
                    "contradictory",
                    current_relationship_status="not_partnered",
                    legal_status_detail="married",
                )
            ],
            "contradicts",
        ),
        (
            [
                _row(
                    "outside-g",
                    marital_status="never_married",
                    legal_status_detail="separated",
                )
            ],
            "only allowed",
        ),
    ]
    for rows, match in cases:
        with pytest.raises(ValueError, match=match):
            apply_legal_status_sampling(_frame(rows), seed=1)


def test_missing_required_column_and_bad_seed_fail_closed() -> None:
    """Reject malformed inputs before returning a repaired frame."""
    frame = _frame([_row("ok")])
    malformed_detail = pl.DataFrame(
        {
            "persona_id": ["malformed-detail"],
            "marital_status": ["married_or_separated"],
            "legal_status_detail": [["married"]],
            "current_relationship_status": ["partnered"],
        }
    )

    with pytest.raises(ValueError, match="Missing"):
        apply_legal_status_sampling(frame.drop("current_relationship_status"), seed=1)
    with pytest.raises(ValueError, match="seed"):
        apply_legal_status_sampling(frame, seed=-1)
    with pytest.raises(ValueError, match="Malformed"):
        apply_legal_status_sampling(malformed_detail, seed=1)


def test_partnered_sampling_is_reproducible_and_order_independent() -> None:
    """Use only seed and persona ID for partnered ambiguous rows."""
    frame = _frame(
        [
            _row(f"p-{index:02d}", current_relationship_status="partnered")
            for index in range(32)
        ]
        + [
            _row("not-partnered", current_relationship_status="not_partnered"),
            _row(
                "already-married",
                current_relationship_status="partnered",
                legal_status_detail="married",
            ),
        ]
    )
    reordered = frame.reverse()

    first, first_report = apply_legal_status_sampling(frame, seed=3)
    repeated, repeated_report = apply_legal_status_sampling(frame, seed=3)
    idempotent, idempotent_report = apply_legal_status_sampling(first, seed=3)
    reversed_result, _ = apply_legal_status_sampling(reordered, seed=3)
    other_seed, _ = apply_legal_status_sampling(frame, seed=4)

    assert repeated.equals(first)
    assert repeated_report == first_report
    assert idempotent.equals(first)
    assert idempotent_report["changed"] == {}
    assert idempotent_report["prose_review_ids"] == []
    assert _detail_by_id(reversed_result) == _detail_by_id(first)
    assert (
        first.filter(pl.col("persona_id") == "not-partnered")[
            "legal_status_detail"
        ].item()
        == "separated"
    )
    assert (
        first.filter(pl.col("persona_id") == "already-married")[
            "legal_status_detail"
        ].item()
        == "married"
    )
    ambiguous_partnered_ids = {f"p-{index:02d}" for index in range(32)}
    changed_by_seed = {
        persona_id
        for persona_id, detail in _detail_by_id(first).items()
        if _detail_by_id(other_seed)[persona_id] != detail
    }
    assert changed_by_seed <= ambiguous_partnered_ids


def _detail_by_id(frame: pl.DataFrame) -> dict[str, str | None]:
    return dict(
        zip(
            frame["persona_id"].to_list(),
            frame["legal_status_detail"].to_list(),
            strict=True,
        )
    )


def test_preserves_valid_rows_and_never_rewrites_other_columns() -> None:
    """Preserve reviewed legal detail and all non-detail fields."""
    frame = _frame(
        [
            _row(
                "valid-married",
                current_relationship_status="partnered",
                legal_status_detail="married",
                persona="Gift persona prose.",
                other_value=10,
            ),
            _row(
                "valid-separated",
                current_relationship_status="not_partnered",
                legal_status_detail="separated",
                persona="Separeret persona prose.",
                other_value=11,
            ),
            _row(
                "never-married",
                marital_status="never_married",
                current_relationship_status="not_partnered",
                persona="Aldrig gift persona prose.",
                other_value=12,
            ),
            _row(
                "invalid-g-detail",
                current_relationship_status="partnered",
                legal_status_detail="unknown",
                persona="Ambiguous persona prose.",
                other_value=13,
            ),
        ]
    )

    repaired, report = apply_legal_status_sampling(frame, seed=9)

    assert repaired.schema == frame.schema
    assert repaired["persona_id"].to_list() == frame["persona_id"].to_list()
    assert repaired.drop("legal_status_detail").equals(
        frame.drop("legal_status_detail")
    )
    assert repaired["legal_status_detail"].to_list()[:3] == [
        "married",
        "separated",
        None,
    ]
    assert set(report["changed"]) == {"invalid-g-detail"}
    assert report["prose_review_ids"] == ["invalid-g-detail"]
    counts = report["counts"]
    assert isinstance(counts, dict)
    absolute = counts["absolute"]
    assert isinstance(absolute, dict)
    assert absolute["preserved_valid_detail_rows"] == 2
    assert absolute["filled_invalid_detail_rows"] == 1
