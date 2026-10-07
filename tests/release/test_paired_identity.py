"""Contracts for the restricted synthetic paired identity overlay."""

import polars as pl
import pytest

from danish_personas.release.lgbt_overlay import (
    LgbtOverlayConfig,
    generate_lgbt_overlay,
)
from danish_personas.release.paired_identity import generate_paired_identity


def test_age_support_null_partners_and_source_are_preserved() -> None:
    """Ages outside support are unknown and unpartnered fields are null."""
    frame = pl.DataFrame(
        {
            "persona_id": ["young", "adult", "senior", "single"],
            "age": [17, 18, 65, 40],
            "sex": ["male", "female", "male", "female"],
            "current_relationship_status": [
                "partnered",
                "partnered",
                "partnered",
                "not_partnered",
            ],
            "partner_gender": ["female", "male", "male", None],
        }
    )
    orientation = pl.DataFrame(
        {
            "persona_id": ["young", "adult", "senior", "single"],
            "sexual_orientation_identity": ["homo", "hetero", "bi", "ace"],
        }
    )
    before = frame.clone()

    result, _ = generate_paired_identity(frame, orientation)

    assert frame.equals(before)
    assert result.columns == [
        "persona_id",
        "gender",
        "partner_gender",
        "sexual_orientation",
        "partner_sexual_orientation",
        "transgender",
        "partner_transgender",
    ]
    assert result.row(0, named=True)["gender"] == "unknown"
    assert result.row(0, named=True)["sexual_orientation"] == "unknown"
    assert result.row(0, named=True)["partner_gender"] == "woman"
    assert result.row(2, named=True)["gender"] == "unknown"
    assert result.row(2, named=True)["partner_gender"] == "man"
    assert result.row(2, named=True)["partner_transgender"] == "unknown"
    single = result.row(3, named=True)
    assert all(
        single[column] is None
        for column in (
            "partner_gender",
            "partner_sexual_orientation",
            "partner_transgender",
        )
    )


def test_chart_overlay_feeds_paired_identity_without_label_mismatch() -> None:
    """The categorical chart labels and the paired-identity input interoperate."""
    frame, _ = _frames(count=200, sex="female", partner_gender="male")
    chart_overlay, _ = generate_lgbt_overlay(frame, config=LgbtOverlayConfig(seed=11))
    paired, provenance = generate_paired_identity(frame, chart_overlay, seed=11)
    assert paired.height == frame.height
    assert paired["sexual_orientation"].null_count() == 0
    assert provenance["row_count"] == 200
    assert "sexual_orientation_minority_identity" not in paired.columns


def _frames(
    *,
    count: int = 1,
    age: int = 30,
    sex: str = "male",
    relationship: str = "partnered",
    partner_gender: str | None = "female",
    orientation: str = "hetero",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ids = [f"p-{index}" for index in range(count)]
    release = pl.DataFrame(
        {
            "persona_id": ids,
            "age": [age] * count,
            "sex": [sex] * count,
            "current_relationship_status": [relationship] * count,
            "partner_gender": [partner_gender] * count,
            "protected_text": ["do not copy"] * count,
        }
    )
    orientations = pl.DataFrame(
        {"persona_id": ids, "sexual_orientation_identity": [orientation] * count}
    )
    return release, orientations


def test_duplicate_ids_and_invalid_orientation_fail_closed() -> None:
    """Ambiguous identifiers and unrecognised categories are rejected."""
    frame, orientation = _frames()
    with pytest.raises(ValueError, match="unique"):
        generate_paired_identity(
            pl.concat([frame, frame]), pl.concat([orientation, orientation])
        )
    invalid = orientation.with_columns(
        pl.lit("unclassified").alias("sexual_orientation_identity")
    )
    with pytest.raises(ValueError, match="Unsupported orientation"):
        generate_paired_identity(frame, invalid)


def test_nonbinary_partner_rules_follow_scenario() -> None:
    """Same/different-gender assumptions also handle nonbinary people."""
    frame, orientation = _frames(count=5_000, orientation="homo")
    homo, _ = generate_paired_identity(frame, orientation, seed=47)
    nonbinary_homo = homo.filter(pl.col("gender") == "nonbinary")
    assert nonbinary_homo.height > 0
    assert nonbinary_homo["partner_gender"].unique().to_list() == ["nonbinary"]
    assert nonbinary_homo["partner_sexual_orientation"].unique().to_list() == [
        "unknown"
    ]
    hetero_orientation = orientation.with_columns(
        pl.lit("hetero").alias("sexual_orientation_identity")
    )
    hetero, _ = generate_paired_identity(frame, hetero_orientation, seed=47)
    nonbinary_hetero = hetero.filter(pl.col("gender") == "nonbinary")
    assert nonbinary_hetero.height > 0
    assert set(nonbinary_hetero["partner_gender"].to_list()) <= {"man", "woman"}


def test_partner_fields_compatibility_and_weighted_gender_draws() -> None:
    """Partner orientations exclude impossible scenario pairings; gender is weighted."""
    frame, orientation = _frames(
        count=5_000, orientation="bi", sex="female", partner_gender=None
    )
    frame = frame.with_columns(
        pl.when(pl.col("persona_id").str.ends_with("0"))
        .then(pl.lit("male"))
        .otherwise(pl.lit("female"))
        .alias("sex")
    )
    result, provenance = generate_paired_identity(frame, orientation, seed=19)
    genders = result.get_column("partner_gender").value_counts()
    gender_counts = dict(zip(genders["partner_gender"], genders["count"], strict=True))
    assert 350 < gender_counts["man"] < 650
    assert 4_300 < gender_counts["woman"] < 4_650
    assert 0 < gender_counts["nonbinary"] < 60
    assert provenance["changed_partner_gender_ids"]
    assert "do not copy" not in str(result)

    partnered = result.filter(pl.col("partner_gender").is_not_null())
    assert partnered.get_column("partner_sexual_orientation").null_count() == 0
    assert partnered.filter(pl.col("partner_gender") == "nonbinary").select(
        "partner_sexual_orientation", "partner_transgender"
    ).unique().to_dicts() == [
        {"partner_sexual_orientation": "unknown", "partner_transgender": "unknown"}
    ]


def test_stable_draws_and_orientation_pairing_scenario() -> None:
    """Stable domain draws obey homosexual and heterosexual gender rules."""
    homo_frame, homo_orientation = _frames(
        count=300, sex="male", orientation="homo", partner_gender="male"
    )
    first, provenance = generate_paired_identity(homo_frame, homo_orientation, seed=12)
    repeated, repeated_provenance = generate_paired_identity(
        homo_frame, homo_orientation, seed=12
    )
    assert first.equals(repeated)
    assert provenance == repeated_provenance
    for row in first.iter_rows(named=True):
        if row["gender"] in ("man", "woman"):
            assert row["partner_gender"] == row["gender"]

    hetero_frame, hetero_orientation = _frames(
        count=300, orientation="hetero", sex="male"
    )
    hetero, _ = generate_paired_identity(hetero_frame, hetero_orientation, seed=3)
    for row in hetero.iter_rows(named=True):
        if row["gender"] == "man":
            assert row["partner_gender"] == "woman"
        elif row["gender"] == "woman":
            assert row["partner_gender"] == "man"
        else:
            assert row["partner_gender"] in ("man", "woman")


def test_trans_gender_mapping_nonbinary_and_marginals() -> None:
    """Large deterministic draws retain SEXUS rates and trans gender mappings."""
    frame, orientation = _frames(
        count=100_000, sex="male", relationship="not_partnered", partner_gender=None
    )
    result, _ = generate_paired_identity(frame, orientation, seed=87)
    counts = result.group_by("gender").len().to_dict(as_series=False)
    observed = dict(zip(counts["gender"], counts["len"], strict=True))
    assert 99_400 <= observed.get("man", 0) <= 99_600
    assert 25 <= observed.get("woman", 0) <= 75
    assert 390 <= observed.get("nonbinary", 0) <= 490
    assert observed.get("unknown", 0) == 0
    rows = result.filter(pl.col("transgender") == "true")
    assert set(rows.get_column("gender").unique().to_list()) == {"man", "woman"}
    nonbinary = result.filter(pl.col("gender") == "nonbinary")
    assert nonbinary.get_column("transgender").unique().to_list() == ["unknown"]
