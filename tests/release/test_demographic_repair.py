"""Offline orchestration tests for demographic repair."""

from pathlib import Path

import polars as pl

from danish_personas.release.demographic_repair import repair_demographics


def test_repair_preserves_identity_and_protected_columns_on_missing_sources(
    tmp_path: Path,
) -> None:
    """Missing offline inputs block repair without mutating the release frame."""
    frame = pl.DataFrame(
        {
            "persona_id": ["p1", "p2"],
            "age": [25, 31],
            "municipality_code": [101, 101],
            "region": ["Capital", "Capital"],
            "sex": ["female", "male"],
            "origin_country": ["DK", "DK"],
            "ocean_openness": [3, 4],
            "marital_status": ["never_married", "married_or_separated"],
            "education_source_code": ["H10", "H40-H80"],
            "education_level": ["primary", "higher_education"],
            "labour_market_status": ["employed", "student"],
            "detailed_status_code": ["05", "130"],
        }
    )
    repaired, report = repair_demographics(frame, tmp_path)
    assert repaired.equals(frame)
    assert repaired.get_column("persona_id").to_list() == ["p1", "p2"]
    assert repaired.select(
        "age", "municipality_code", "region", "sex", "origin_country",
        "ocean_openness",
    ).equals(
        frame.select(
            "age", "municipality_code", "region", "sex", "origin_country",
            "ocean_openness",
        )
    )
    assert len(report["stages"]) == 3
    assert all(stage["changed_count"] == 0 for stage in report["stages"])
    assert all(stage["blocker"] for stage in report["stages"])
    assert report["blocking_infeasibility"]
