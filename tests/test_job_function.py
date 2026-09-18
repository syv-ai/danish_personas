"""Offline source, sampler, and validation tests for LONS20 job function."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from danish_personas.io import load_yaml_model
from danish_personas.models import DISCO_TWO_DIGIT_CODES, DemographicRecord, SourceLock
from danish_personas.sampling.generator import _attach_job_functions, generate_records
from danish_personas.sources.prepare import (
    LONS20_DIMENSIONS,
    _job_function_sex_marginal,
)
from danish_personas.validation.checks import validate_demographics
from tests.test_non_llm_pipeline import _write_bundle


def test_demographic_schema_requires_job_function_resolution() -> None:
    """The release record schema cannot omit job-function provenance."""
    assert DemographicRecord.model_fields["job_function_resolution"].is_required()
    assert not DemographicRecord.model_fields["job_function_code"].is_required()
    assert not DemographicRecord.model_fields["job_function"].is_required()


def test_job_function_allocation_is_eligible_and_deterministic() -> None:
    """Only employee codes receive deterministic sex-conditional allocations."""
    records: list[dict[str, object]] = [
        {"sex": "female", "detailed_status_code": "15"},
        {"sex": "female", "detailed_status_code": "05"},
        {"sex": "male", "detailed_status_code": "40"},
        {"sex": "male", "detailed_status_code": "10"},
    ]
    first = [dict(record) for record in records]
    second = [dict(record) for record in records]

    _attach_job_functions(
        records=first, marginal=_prepared_marginal(), rng=np.random.default_rng(42)
    )
    _attach_job_functions(
        records=second, marginal=_prepared_marginal(), rng=np.random.default_rng(42)
    )

    assert first == second
    assert first[0]["job_function_resolution"] == "lons20_sex_marginal"
    assert first[2]["job_function_resolution"] == "lons20_sex_marginal"
    for index in (1, 3):
        assert first[index]["job_function_code"] is None
        assert first[index]["job_function"] is None
        assert first[index]["job_function_resolution"] == "not_applicable"


def _prepared_marginal() -> pl.DataFrame:
    return _job_function_sex_marginal(
        raw_frame=_raw_marginal(),
        official_labels=_labels(),
        selected_codes=list(DISCO_TWO_DIGIT_CODES),
        sex_mapping={"M": "male", "K": "female"},
    )


def _labels() -> dict[str, str]:
    return {code: f"{code} Official label" for code in DISCO_TWO_DIGIT_CODES}


def _raw_marginal() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "ARBF": code,
                "ARBF__label": f"{code} Official label",
                "SEKTOR": "1000",
                "AFLOEN": "TIFA",
                "LONGRP": "LTOT",
                "LØNMÅL": "ANTAL",
                "KØN": sex,
                "Tid": "2024",
                "count": index + (1 if sex == "M" else 2),
                "suppressed": False,
            }
            for index, code in enumerate(DISCO_TWO_DIGIT_CODES)
            for sex in ("M", "K")
        ]
    )


def test_job_function_validation_rejects_mapping_and_eligibility_tampering(
    tmp_path: Path,
) -> None:
    """Validation reports altered labels and assignments to ineligible rows."""
    bundle, sampling, validation, categories = _write_bundle(root=tmp_path)
    run = generate_records(
        bundle_dir=bundle,
        sampling_config_path=sampling,
        output_dir=tmp_path / "runs",
        rows=200,
        seed=42,
    )
    path = run / "structured-records.parquet"
    frame = (
        pl.read_parquet(path)
        .with_row_index()
        .with_columns(
            pl.when(pl.col("index") == 0)
            .then(pl.lit("Altered official label"))
            .otherwise(pl.col("job_function"))
            .alias("job_function"),
            pl.when(pl.col("index") == 1)
            .then(pl.lit("05"))
            .otherwise(pl.col("detailed_status_code"))
            .alias("detailed_status_code"),
        )
        .drop("index")
    )
    frame.write_parquet(path)

    report = validate_demographics(
        run_dir=run,
        bundle_dir=bundle,
        validation_config_path=validation,
        categories_path=categories,
    )
    metrics = {metric.name: metric for metric in report.metrics}
    assert not metrics["job_function_mapping_errors"].passed
    assert not metrics["job_function_eligibility_errors"].passed


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
    raw = _raw_marginal()
    prepared = _job_function_sex_marginal(
        raw_frame=raw,
        official_labels=_labels(),
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
    "failure", ["level", "duplicate", "label", "suppressed", "zero", "sex"]
)
def test_lons20_preparation_rejects_malformed_cells(failure: str) -> None:
    """Wrong levels, duplicates, suppression, counts, and sex gaps fail."""
    raw = _raw_marginal()
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
        raw = raw.with_row_index().with_columns(
            pl.when(pl.col("index") == 0)
            .then(pl.lit("01 Altered label"))
            .otherwise(pl.col("ARBF__label"))
            .alias("ARBF__label")
        ).drop("index")
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
            official_labels=_labels(),
            selected_codes=list(DISCO_TWO_DIGIT_CODES),
            sex_mapping={"M": "male", "K": "female"},
        )
