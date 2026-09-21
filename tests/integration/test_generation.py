"""Deterministic demographic generation integration tests."""

from pathlib import Path

import polars as pl

from danish_personas.models import DemographicRecord, RunManifest
from danish_personas.sampling.generator import generate_records
from danish_personas.validation.checks import validate_demographics
from tests.support.bundles import _write_bundle


def test_generation_is_deterministic_and_valid(tmp_path: Path) -> None:
    """Identical inputs produce identical valid logical records without an LLM."""
    bundle_dir, sampling_path, validation_path, categories_path = _write_bundle(
        root=tmp_path
    )
    first = generate_records(
        bundle_dir=bundle_dir,
        sampling_config_path=sampling_path,
        output_dir=tmp_path / "runs-a",
        rows=200,
        seed=42,
    )
    second = generate_records(
        bundle_dir=bundle_dir,
        sampling_config_path=sampling_path,
        output_dir=tmp_path / "runs-b",
        rows=200,
        seed=42,
    )
    first_manifest = RunManifest.model_validate_json(
        (first / "run-manifest.json").read_text(encoding="utf-8")
    )
    second_manifest = RunManifest.model_validate_json(
        (second / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert first_manifest.logical_content_sha256 == (
        second_manifest.logical_content_sha256
    )
    assert first_manifest.llm_calls == 0
    frame = pl.read_parquet(first / first_manifest.data_file)
    assert {"origin_country_code", "origin_country"} <= set(frame.columns)
    assert "Not stated" not in frame.get_column("origin_country").unique().to_list()
    DemographicRecord.model_validate(frame.row(0, named=True))
    report = validate_demographics(
        run_dir=first,
        bundle_dir=bundle_dir,
        validation_config_path=validation_path,
        categories_path=categories_path,
    )
    assert report.passed

    corrupted_origin = (
        frame.with_row_index()
        .with_columns(
            pl.when(pl.col("index") == 0)
            .then(pl.lit("Corrupted label"))
            .otherwise(pl.col("origin_country"))
            .alias("origin_country")
        )
        .drop("index")
    )
    corrupted_origin.write_parquet(first / first_manifest.data_file)
    corrupted_report = validate_demographics(
        run_dir=first,
        bundle_dir=bundle_dir,
        validation_config_path=validation_path,
        categories_path=categories_path,
    )
    corrupted_metrics = {metric.name: metric for metric in corrupted_report.metrics}
    assert corrupted_metrics["origin_country_mapping"].value == 1
    assert not corrupted_metrics["origin_country_mapping"].passed
    assert corrupted_metrics["origin_country_unexpected_categories"].value == 1
    assert not corrupted_metrics["origin_country_unexpected_categories"].passed

    frame.with_columns((pl.col("age") + 1).alias("age")).write_parquet(
        first / first_manifest.data_file
    )
    tampered_report = validate_demographics(
        run_dir=first,
        bundle_dir=bundle_dir,
        validation_config_path=validation_path,
        categories_path=categories_path,
    )
    assert not tampered_report.passed
