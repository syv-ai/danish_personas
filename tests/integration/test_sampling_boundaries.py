"""Integration tests for sampler identity and back-off boundaries."""

from pathlib import Path

import polars as pl
import yaml

from danish_personas.io import sha256_file, sha256_text
from danish_personas.models import (
    SAMPLER_SCHEMA_VERSION,
    DemographicRecord,
    RunManifest,
)
from danish_personas.sampling.generator import generate_records
from danish_personas.validation.checks import validate_demographics
from tests.support.bundles import _write_bundle, refresh_bundle_manifest


def test_new_sampler_schema_does_not_reuse_legacy_run(tmp_path: Path) -> None:
    """A legacy run directory is not reused after the sampler schema changes."""
    bundle_dir, sampling_path, _, _ = _write_bundle(root=tmp_path)
    legacy_id = sha256_text(f"fixture-bundle:{sha256_file(sampling_path)}:200:42")[:16]
    legacy_dir = tmp_path / "runs" / legacy_id
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "run-manifest.json").write_text("{}", encoding="utf-8")

    run_dir = generate_records(
        bundle_dir=bundle_dir,
        sampling_config_path=sampling_path,
        output_dir=tmp_path / "runs",
        rows=200,
        seed=42,
    )

    assert run_dir != legacy_dir
    manifest = RunManifest.model_validate_json(
        (run_dir / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.sampler_schema_version == SAMPLER_SCHEMA_VERSION
    frame = pl.read_parquet(run_dir / manifest.data_file)
    assert {
        "age_resolution",
        "marital_resolution",
        "detailed_status_resolution",
    } <= set(frame.columns)


def test_terminal_backoff_reaches_records_and_each_ceiling_is_enforced(
    tmp_path: Path,
) -> None:
    """Terminal ladder fallbacks construct valid records and fail strict ceilings."""
    bundle_dir, sampling_path, validation_path, categories_path = _write_bundle(
        root=tmp_path
    )
    normalized = bundle_dir / "normalized"
    for filename in (
        "folk_age_sampling.parquet",
        "folk_marital_sampling.parquet",
        "ras202_sampling.parquet",
    ):
        path = normalized / filename
        frame = pl.read_parquet(path).with_columns(pl.lit("other").alias("sex"))
        if filename == "folk_marital_sampling.parquet":
            frame = frame.with_columns(pl.lit("18-29").alias("age_band"))
        frame.write_parquet(path)
    refresh_bundle_manifest(bundle_dir=bundle_dir)

    run_dir = generate_records(
        bundle_dir=bundle_dir,
        sampling_config_path=sampling_path,
        output_dir=tmp_path / "runs",
        rows=20,
        seed=42,
    )
    manifest = RunManifest.model_validate_json(
        (run_dir / "run-manifest.json").read_text(encoding="utf-8")
    )
    frame = pl.read_parquet(run_dir / manifest.data_file)
    for row in frame.iter_rows(named=True):
        DemographicRecord.model_validate(row)
    assert set(frame.get_column("age_resolution").unique()) == {"municipality_age_band"}
    assert set(frame.get_column("marital_resolution").unique()) == {"municipality"}
    assert set(frame.get_column("detailed_status_resolution").unique()) == {"status"}

    validation = yaml.safe_load(validation_path.read_text(encoding="utf-8"))
    validation["maximum_backoff_rate"] = 0.0
    validation_path.write_text(yaml.safe_dump(validation), encoding="utf-8")
    report = validate_demographics(
        run_dir=run_dir,
        bundle_dir=bundle_dir,
        validation_config_path=validation_path,
        categories_path=categories_path,
    )
    backoff_metrics = {
        metric.name: metric
        for metric in report.metrics
        if metric.name.endswith("_backoff_rate")
    }
    assert set(backoff_metrics) == {
        "age_resolution_backoff_rate",
        "marital_resolution_backoff_rate",
        "detailed_status_resolution_backoff_rate",
    }
    assert all(not metric.passed for metric in backoff_metrics.values())
