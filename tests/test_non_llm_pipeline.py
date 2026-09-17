"""Offline integration tests for deterministic Phase 2 generation."""

from pathlib import Path

import polars as pl
import yaml

from danish_personas.io import sha256_file, sha256_text, write_json
from danish_personas.models import (
    SAMPLER_SCHEMA_VERSION,
    BundleManifest,
    DemographicRecord,
    RunManifest,
)
from danish_personas.sampling.generator import generate_records
from danish_personas.validation.checks import validate_demographics


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


def _write_bundle(root: Path) -> tuple[Path, Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    bundle_dir = root / "bundle"
    normalized = bundle_dir / "normalized"
    normalized.mkdir(parents=True)
    folk = pl.DataFrame(
        {
            "municipality_code": ["101", "101"],
            "sex": ["male", "female"],
            "age": [30, 30],
            "marital_status": ["never_married", "married_or_separated"],
            "count": [100, 100],
            "suppressed": [False, False],
            "municipality": ["Copenhagen", "Copenhagen"],
            "region_code": ["084", "084"],
            "region": ["Region Hovedstaden", "Region Hovedstaden"],
        }
    )
    ras209 = pl.DataFrame(
        {
            "region_code": ["084", "084"],
            "region": ["Region Hovedstaden", "Region Hovedstaden"],
            "age_band": ["30-49", "30-49"],
            "sex": ["male", "female"],
            "education_source_code": ["H70", "H70"],
            "education_level": ["masters", "masters"],
            "labour_market_status": ["employed", "employed"],
            "count": [100, 100],
            "suppressed": [False, False],
        }
    )
    ras202 = pl.DataFrame(
        {
            "detailed_status_code": ["30", "30"],
            "detailed_status": ["Employees - basic level", "Employees - basic level"],
            "labour_market_status": ["employed", "employed"],
            "age_band": ["30-49", "30-49"],
            "sex": ["male", "female"],
            "count": [100, 100],
            "suppressed": [False, False],
        }
    )
    befolk = folk.drop("marital_status")
    ras210 = pl.DataFrame(
        {
            "municipality_code": ["101"],
            "status_group_code": ["00"],
            "age_key": ["30"],
            "sex": ["male"],
            "count": [100],
            "suppressed": [False],
            "municipality": ["Copenhagen"],
            "region_code": ["084"],
            "region": ["Region Hovedstaden"],
        }
    )
    origin = pl.DataFrame(
        {
            "origin_country_code": ["5100", "5103", "5999"],
            "origin_country": ["Denmark", "Stateless", "Not stated"],
            "count": [107, 2, 0],
        }
    )
    age_sampling = folk.select("sex", "age", "count", "suppressed").with_columns(
        pl.lit("30-49").alias("age_band")
    )
    marital_sampling = folk.select(
        "region_code", "region", "sex", "marital_status", "count", "suppressed"
    ).with_columns(pl.lit("30-49").alias("age_band"))
    frames = {
        "folk1a_base_unpooled": folk,
        "folk2_origin_country_marginal": origin,
        "folk_age_sampling": age_sampling,
        "folk_marital_sampling": marital_sampling,
        "ras209_sampling": ras209,
        "ras202_sampling": ras202,
        "befolk3_holdout": befolk,
        "ras210_holdout": ras210,
    }
    files: dict[str, str] = {}
    for name, frame in frames.items():
        path = normalized / f"{name}.parquet"
        frame.write_parquet(path)
        files[str(path.relative_to(bundle_dir))] = sha256_file(path)
    manifest = BundleManifest(
        bundle_id="fixture-bundle",
        created_at="2026-09-14T00:00:00+00:00",
        source_lock_sha256="0" * 64,
        categories_sha256="1" * 64,
        source_snapshots=[],
        classification_snapshots=[],
        files=files,
        reference_periods={},
        assumptions=[],
    )
    write_json(path=bundle_dir / "bundle-manifest.json", payload=manifest)
    sampling_path = root / "sampling.yaml"
    validation_path = root / "validation.yaml"
    categories_path = root / "categories.yaml"
    _write_yaml(
        path=sampling_path,
        payload={
            "version": 1,
            "seed": 42,
            "smoke_rows": 100,
            "statistical_rows": 200,
            "country": "Danmark",
            "minimum_age": 18,
            "maximum_age": 125,
            "publication_geography": "region",
            "smoothing": 0.0,
            "ocean": {
                "mean": 50.0,
                "standard_deviation": 10.0,
                "minimum": 20.0,
                "maximum": 80.0,
                "label_boundaries": [35.0, 45.0, 55.0, 65.0],
                "labels": ["very_low", "low", "average", "high", "very_high"],
            },
        },
    )
    _write_yaml(
        path=validation_path,
        payload={
            "version": 3,
            "absolute_proportion_tolerance": 0.2,
            "standard_error_multiplier": 5.0,
            "minimum_expected_count": 1.0,
            "maximum_ocean_pairwise_correlation": 1.0,
            "maximum_total_variation": {
                "fitted_marginal": 0.2,
                "heldout_marginal": 0.5,
            },
            "maximum_backoff_rate": 0.01,
            "smoke_maximum_total_variation": 0.2,
            "smoke_holdout_maximum_total_variation": 0.2,
            "mandatory_marginals": [
                "sex",
                "region_code",
                "marital_status",
                "age_band",
                "education_level",
                "labour_market_status",
                "origin_country",
            ],
        },
    )
    _write_yaml(
        path=categories_path,
        payload={
            "version": 1,
            "sex": {"1": "male", "2": "female", "M": "male", "K": "female"},
            "marital_status": {
                "U": "never_married",
                "G": "married_or_separated",
                "E": "widowed",
                "F": "divorced",
            },
            "education": {"H70": "masters"},
            "education_pooling": {"masters": "higher_education"},
            "labour_market_status": {
                "employed": ["30"],
                "unemployed": [],
                "student": [],
                "retired": [],
                "other": [],
            },
        },
    )
    return bundle_dir, sampling_path, validation_path, categories_path


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


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


def test_origin_stream_does_not_change_existing_fields(tmp_path: Path) -> None:
    """Changing only FOLK2 weights leaves all pre-existing fields unchanged."""
    first_paths = _write_bundle(root=tmp_path / "first")
    second_paths = _write_bundle(root=tmp_path / "second")
    origin_path = (
        second_paths[0] / "normalized" / "folk2_origin_country_marginal.parquet"
    )
    pl.DataFrame(
        {
            "origin_country_code": ["5100", "5103", "5999"],
            "origin_country": ["Denmark", "Stateless", "Not stated"],
            "count": [2, 107, 0],
        }
    ).write_parquet(origin_path)
    first = generate_records(
        bundle_dir=first_paths[0],
        sampling_config_path=first_paths[1],
        output_dir=tmp_path / "runs-first",
        rows=200,
        seed=42,
    )
    second = generate_records(
        bundle_dir=second_paths[0],
        sampling_config_path=second_paths[1],
        output_dir=tmp_path / "runs-second",
        rows=200,
        seed=42,
    )
    first_frame = pl.read_parquet(first / "structured-records.parquet")
    second_frame = pl.read_parquet(second / "structured-records.parquet")
    old_fields = [
        field
        for field in first_frame.columns
        if field not in {"origin_country_code", "origin_country"}
    ]
    assert first_frame.select(old_fields).equals(second_frame.select(old_fields))


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
            frame = frame.with_columns(pl.lit("999").alias("region_code"))
        frame.write_parquet(path)

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
    assert set(frame.get_column("age_resolution").unique()) == {"age_band"}
    assert set(frame.get_column("marital_resolution").unique()) == {"age_band"}
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
