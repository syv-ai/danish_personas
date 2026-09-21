"""Integration tests for prepared-bundle boundary checks."""

from pathlib import Path

import polars as pl
import pytest

from danish_personas.sampling.generator import generate_records
from danish_personas.validation.checks import validate_demographics
from tests.support.bundles import _write_bundle, refresh_bundle_manifest


def test_sampler_and_validator_reject_tampered_bundle(tmp_path: Path) -> None:
    """Both Phase-2 boundaries reject prepared files changed after manifesting."""
    bundle_dir, sampling_path, validation_path, categories_path = _write_bundle(
        root=tmp_path
    )
    run_dir = generate_records(
        bundle_dir=bundle_dir,
        sampling_config_path=sampling_path,
        output_dir=tmp_path / "runs",
        rows=20,
        seed=42,
    )
    target = bundle_dir / "normalized" / "ras209_sampling.parquet"
    target.write_bytes(target.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="Prepared bundle verification failed"):
        generate_records(
            bundle_dir=bundle_dir,
            sampling_config_path=sampling_path,
            output_dir=tmp_path / "other-runs",
            rows=20,
            seed=42,
        )
    with pytest.raises(ValueError, match="Prepared bundle verification failed"):
        validate_demographics(
            run_dir=run_dir,
            bundle_dir=bundle_dir,
            validation_config_path=validation_path,
            categories_path=categories_path,
        )


def test_sampler_rejects_manifested_malformed_bundle_schema(tmp_path: Path) -> None:
    """A matching checksum cannot bless a prepared table with missing columns."""
    bundle_dir, sampling_path, _, _ = _write_bundle(root=tmp_path)
    target = bundle_dir / "normalized" / "ras209_sampling.parquet"
    pl.read_parquet(target).drop("municipality_code").write_parquet(target)
    refresh_bundle_manifest(bundle_dir=bundle_dir)

    with pytest.raises(ValueError, match="Prepared bundle schema mismatch"):
        generate_records(
            bundle_dir=bundle_dir,
            sampling_config_path=sampling_path,
            output_dir=tmp_path / "runs",
            rows=20,
            seed=42,
        )
