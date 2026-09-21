"""Validation tests for generated job-function fields."""

from pathlib import Path

import polars as pl

from danish_personas.sampling.generator import generate_records
from danish_personas.validation.checks import validate_demographics
from tests.support.bundles import _write_bundle


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


def test_parquet_validation_rejects_blank_eligible_job_function(tmp_path: Path) -> None:
    """The Parquet validation gate rejects blank eligible job-function labels."""
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
            .then(pl.lit(""))
            .otherwise(pl.col("job_function"))
            .alias("job_function")
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
    assert not metrics["job_function_eligibility_errors"].passed
