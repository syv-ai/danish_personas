"""LONS20 source preparation and contract tests."""

from pathlib import Path

import polars as pl
import pytest

from danish_personas.io import load_yaml_model
from danish_personas.models import DISCO_TWO_DIGIT_CODES, SourceLock
from danish_personas.sources.prepare import (
    LONS20_DIMENSIONS,
    _job_function_sex_marginal,
    _read_source,
)
from tests.support.job_function import job_function_labels, raw_job_function_marginal


def test_lons20_lock_freezes_exact_two_digit_partition() -> None:
    """The lock contains only the authorised LONS20 query."""
    lock = load_yaml_model(path=Path("config/sources.lock.yaml"), model=SourceLock)
    source = next(source for source in lock.sources if source.table_id == "LONS20")

    assert source.role == "job_function_sex_marginal"
    assert source.period == "2024"
    assert source.dimensions == LONS20_DIMENSIONS
    assert len(source.dimensions["ARBF"]) == 42


def test_lons20_preparation_preserves_official_cells() -> None:
    """Preparation retains each official code-label-sex count cell."""
    raw = raw_job_function_marginal()
    prepared = _job_function_sex_marginal(
        raw_frame=raw,
        official_labels=job_function_labels(),
        selected_codes=list(DISCO_TWO_DIGIT_CODES),
        sex_mapping={"M": "male", "K": "female"},
    )

    assert prepared.shape == (84, 4)
    assert prepared.get_column("count").sum() == raw.get_column("count").sum()
    assert prepared.filter(pl.col("job_function_code") == "01").to_dicts() == [
        {
            "job_function_code": "01",
            "job_function": "01 Official label",
            "sex": "female",
            "count": 2,
        },
        {
            "job_function_code": "01",
            "job_function": "01 Official label",
            "sex": "male",
            "count": 1,
        },
    ]


@pytest.mark.parametrize(
    "failure",
    ["level", "duplicate", "label", "suppressed", "zero", "sex"],
    ids=[
        "wrong-level",
        "duplicate-sex-cell",
        "drifted-label",
        "suppressed-cell",
        "zero-count",
        "missing-sex-cell",
    ],
)
def test_lons20_preparation_rejects_malformed_cells(failure: str) -> None:
    """Wrong levels, duplicates, suppression, counts, and sex gaps fail."""
    raw = raw_job_function_marginal()
    if failure == "level":
        raw = (
            raw.with_row_index()
            .with_columns(
                pl.when(pl.col("index") == 0)
                .then(pl.lit("1"))
                .otherwise(pl.col("ARBF"))
                .alias("ARBF")
            )
            .drop("index")
        )
    elif failure == "duplicate":
        raw = (
            raw.with_row_index()
            .with_columns(
                pl.when(pl.col("index") == 1)
                .then(pl.lit("M"))
                .otherwise(pl.col("KØN"))
                .alias("KØN")
            )
            .drop("index")
        )
    elif failure == "label":
        raw = (
            raw.with_row_index()
            .with_columns(
                pl.when(pl.col("index") == 0)
                .then(pl.lit("01 Altered label"))
                .otherwise(pl.col("ARBF__label"))
                .alias("ARBF__label")
            )
            .drop("index")
        )
    elif failure == "suppressed":
        raw = (
            raw.with_row_index()
            .with_columns(
                (pl.col("suppressed") | (pl.col("index") == 0)).alias("suppressed")
            )
            .drop("index")
        )
    elif failure == "zero":
        raw = (
            raw.with_row_index()
            .with_columns(
                pl.when(pl.col("index") == 0)
                .then(pl.lit(0))
                .otherwise(pl.col("count"))
                .alias("count")
            )
            .drop("index")
        )
    else:
        raw = raw.filter(~((pl.col("ARBF") == "01") & (pl.col("KØN") == "K")))

    with pytest.raises(ValueError):
        _job_function_sex_marginal(
            raw_frame=raw,
            official_labels=job_function_labels(),
            selected_codes=list(DISCO_TWO_DIGIT_CODES),
            sex_mapping={"M": "male", "K": "female"},
        )


def test_source_reader_rejects_negative_counts_before_transforms(
    tmp_path: Path,
) -> None:
    """Negative counts cannot be hidden by later source filtering or aggregation."""
    path = tmp_path / "source.csv"
    path.write_text("A;value\ncode;-1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Negative count"):
        _read_source(csv_path=path, dimension_codes=["A"])
