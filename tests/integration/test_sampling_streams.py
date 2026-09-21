"""Integration tests for independent demographic sampling streams."""

from pathlib import Path

import polars as pl

from danish_personas.origin_labels import load_origin_label_contract
from danish_personas.sampling.generator import generate_records
from tests.support.bundles import _write_bundle, refresh_bundle_manifest


def test_job_function_stream_does_not_change_existing_fields(tmp_path: Path) -> None:
    """Changing only LONS20 weights leaves every pre-existing field unchanged."""
    first_paths = _write_bundle(root=tmp_path / "first-job")
    second_paths = _write_bundle(root=tmp_path / "second-job")
    marginal_path = second_paths[0] / "normalized" / "job_function_sex_marginal.parquet"
    marginal = pl.read_parquet(marginal_path).with_columns(
        pl.when(pl.col("job_function_code") == "01")
        .then(pl.lit(1_000_000))
        .otherwise(pl.col("count"))
        .alias("count")
    )
    marginal.write_parquet(marginal_path)
    refresh_bundle_manifest(bundle_dir=second_paths[0])

    first = generate_records(
        bundle_dir=first_paths[0],
        sampling_config_path=first_paths[1],
        output_dir=tmp_path / "job-runs-first",
        rows=200,
        seed=42,
    )
    second = generate_records(
        bundle_dir=second_paths[0],
        sampling_config_path=second_paths[1],
        output_dir=tmp_path / "job-runs-second",
        rows=200,
        seed=42,
    )
    first_frame = pl.read_parquet(first / "structured-records.parquet")
    second_frame = pl.read_parquet(second / "structured-records.parquet")
    existing_fields = [
        field
        for field in first_frame.columns
        if field not in {"job_function_code", "job_function", "job_function_resolution"}
    ]

    assert first_frame.select(existing_fields).equals(
        second_frame.select(existing_fields)
    )
    assert not first_frame.select("job_function_code").equals(
        second_frame.select("job_function_code")
    )


def test_origin_stream_does_not_change_existing_fields(tmp_path: Path) -> None:
    """Changing only FOLK2 weights leaves all pre-existing fields unchanged."""
    first_paths = _write_bundle(root=tmp_path / "first")
    second_paths = _write_bundle(root=tmp_path / "second")
    origin_path = (
        second_paths[0] / "normalized" / "folk2_origin_country_marginal.parquet"
    )
    contract = load_origin_label_contract()
    pl.DataFrame(
        {
            "origin_country_code": list(contract.labels_en),
            "origin_country": list(contract.labels_en.values()),
            "origin_country_da": list(contract.labels_da.values()),
            "count": [
                2 if code == "5100" else 107 if code == "5103" else 0
                for code in contract.labels_en
            ],
        }
    ).write_parquet(origin_path)
    refresh_bundle_manifest(bundle_dir=second_paths[0])
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
        if field not in {"origin_country_code", "origin_country", "origin_country_da"}
    ]
    assert first_frame.select(old_fields).equals(second_frame.select(old_fields))
