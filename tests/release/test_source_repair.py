"""Offline tests for source-backed release repair primitives."""

from pathlib import Path

import polars as pl
import pytest

from danish_personas.release.source_repair import (
    InfeasibleRepairError,
    assess_release_support,
    repair_from_source,
)


def test_repair_preserves_strata_order_and_non_categories() -> None:
    """Reassignment changes only categories and meets source-proportional quotas."""
    frame = pl.DataFrame(
        {
            "persona_id": ["a", "b", "c", "d"],
            "age_band": ["30-34"] * 4,
            "sex": ["female"] * 4,
            "labour_market_status": ["employed"] * 4,
            "detailed_status_code": ["A", "A", "A", "B"],
            "origin_country_code": ["DK", "SE", "DK", "NO"],
        }
    )
    source = pl.DataFrame(
        {
            "age_band": ["30-34", "30-34"],
            "sex": ["female", "female"],
            "labour_market_status": ["employed", "employed"],
            "detailed_status_code": ["A", "B"],
            "count": [1, 3],
            "suppressed": [False, False],
        }
    )
    repaired, diagnostics = repair_from_source(
        frame=frame,
        source=source,
        strata=["age_band", "sex", "labour_market_status"],
        categories=["detailed_status_code"],
    )
    assert repaired.get_column("persona_id").to_list() == ["a", "b", "c", "d"]
    assert repaired.get_column("origin_country_code").to_list() == [
        "DK",
        "SE",
        "DK",
        "NO",
    ]
    assert repaired.get_column("detailed_status_code").to_list().count("B") == 3
    assert diagnostics["changed_rows"] == 2
    assert diagnostics["category_columns"] == ["detailed_status_code"]


def test_repair_fails_for_unsupported_stratum() -> None:
    """A source-empty fixed stratum is infeasible."""
    frame = pl.DataFrame({"stratum": ["unknown"], "category": ["x"]})
    source = pl.DataFrame(
        {"stratum": ["known"], "category": ["x"], "count": [1], "suppressed": [False]}
    )
    with pytest.raises(InfeasibleRepairError, match="No disclosed positive"):
        repair_from_source(
            frame=frame, source=source, strata=["stratum"], categories=["category"]
        )


def test_assessment_reports_exact_joint_support(tmp_path: Path) -> None:
    """Support assessment counts exact source joint matches offline."""
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    folk = pl.DataFrame(
        {
            "age": [30],
            "sex": ["female"],
            "municipality_code": ["101"],
            "marital_status": ["single"],
            "suppressed": [False],
        }
    )
    ras209 = pl.DataFrame(
        {
            "age_band": ["30-34"],
            "municipality_code": ["101"],
            "sex": ["female"],
            "education_source_code": ["E"],
            "labour_market_status": ["employed"],
            "suppressed": [False],
        }
    )
    ras202 = pl.DataFrame(
        {
            "age_band": ["30-34"],
            "sex": ["female"],
            "labour_market_status": ["employed"],
            "detailed_status_code": ["A"],
            "suppressed": [False],
        }
    )
    folk.write_parquet(normalized / "folk1a_base_unpooled.parquet")
    ras209.write_parquet(normalized / "ras209_joint_unpooled.parquet")
    ras202.write_parquet(normalized / "ras202_detail_unpooled.parquet")
    frame = pl.DataFrame(
        {
            "age": [30, 31],
            "sex": ["female", "female"],
            "municipality_code": ["101", "101"],
            "marital_status": ["single", "married"],
            "age_band": ["30-34", "30-34"],
            "education_source_code": ["E", "E"],
            "labour_market_status": ["employed", "employed"],
            "detailed_status_code": ["A", "missing"],
        }
    )
    result = assess_release_support(frame=frame, bundle_dir=tmp_path)
    assert result["folk1a"]["supported_rows"] == 1
    assert result["folk1a"]["unsupported_rows"] == 1
    assert result["ras209"]["supported_rows"] == 2
    assert result["ras202"]["unsupported_rows"] == 1
