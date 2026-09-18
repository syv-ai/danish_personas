"""Integrity and schema verification for prepared source bundles."""

import json
from pathlib import Path

import polars as pl

from ..io import verify_checksums
from ..models import PREPARED_BUNDLE_SCHEMA_VERSION, BundleManifest

REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "normalized/folk1a_base_unpooled.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "region_code",
            "region",
            "sex",
            "age",
            "age_band",
            "marital_status",
            "count",
            "suppressed",
        }
    ),
    "normalized/folk_age_sampling.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "region_code",
            "region",
            "age_band",
            "sex",
            "age",
            "count",
            "suppressed",
        }
    ),
    "normalized/folk_marital_sampling.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "region_code",
            "region",
            "age_band",
            "sex",
            "marital_status",
            "count",
            "suppressed",
        }
    ),
    "normalized/ras209_joint_unpooled.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "region_code",
            "region",
            "age_band",
            "sex",
            "education_source_code",
            "education_level",
            "labour_market_status",
            "count",
            "suppressed",
        }
    ),
    "normalized/ras209_sampling.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "region_code",
            "region",
            "age_band",
            "sex",
            "education_source_code",
            "education_level",
            "labour_market_status",
            "count",
            "suppressed",
        }
    ),
    "normalized/ras202_detail_unpooled.parquet": frozenset(
        {
            "age_band",
            "age_key",
            "sex",
            "labour_market_status",
            "detailed_status_code",
            "detailed_status",
            "count",
            "suppressed",
        }
    ),
    "normalized/ras202_sampling.parquet": frozenset(
        {
            "age_band",
            "sex",
            "labour_market_status",
            "detailed_status_code",
            "detailed_status",
            "count",
            "suppressed",
        }
    ),
    "normalized/befolk3_holdout.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "region_code",
            "region",
            "sex",
            "age",
            "count",
            "suppressed",
        }
    ),
    "normalized/ras210_holdout.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "region_code",
            "region",
            "status_group_code",
            "age_key",
            "sex",
            "count",
            "suppressed",
        }
    ),
    "normalized/folk2_origin_country_marginal.parquet": frozenset(
        {"origin_country_code", "origin_country", "count"}
    ),
    "normalized/geography_hierarchy.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "landsdel_code",
            "landsdel",
            "region_code",
            "region",
        }
    ),
}
SOURCE_REPORT = "source-preparation-report.json"


def verify_prepared_bundle(*, bundle_dir: Path) -> BundleManifest:
    """Verify a prepared bundle before sampling or validation.

    Args:
        bundle_dir:
            Prepared bundle directory.

    Returns:
        The verified bundle manifest.

    Raises:
        ValueError:
            If the bundle is legacy, incomplete, malformed, tampered, or did not pass
            source preparation.
    """
    manifest_path = bundle_dir / "bundle-manifest.json"
    try:
        manifest = BundleManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Invalid prepared bundle manifest: {manifest_path}"
        ) from error
    if manifest.prepared_bundle_schema_version != PREPARED_BUNDLE_SCHEMA_VERSION:
        message = (
            "Prepared bundle uses unsupported schema version "
            f"{manifest.prepared_bundle_schema_version}"
        )
        raise ValueError(message)
    required_files = {*REQUIRED_COLUMNS, SOURCE_REPORT}
    missing_files = sorted(required_files - set(manifest.files))
    if missing_files:
        raise ValueError(f"Prepared bundle manifest is missing files: {missing_files}")
    verify_checksums(
        base_dir=bundle_dir,
        expected=manifest.files,
        message="Prepared bundle verification failed",
    )
    _verify_schemas(bundle_dir=bundle_dir)
    try:
        report = json.loads((bundle_dir / SOURCE_REPORT).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Prepared bundle source report is malformed") from error
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise ValueError("Prepared bundle source preparation did not pass")
    return manifest


def _verify_schemas(*, bundle_dir: Path) -> None:
    for relative_path, expected_columns in REQUIRED_COLUMNS.items():
        try:
            columns = frozenset(
                pl.scan_parquet(bundle_dir / relative_path).collect_schema().names()
            )
        except (OSError, pl.exceptions.PolarsError) as error:
            raise ValueError(
                f"Prepared bundle schema cannot be read: {relative_path}"
            ) from error
        if columns != expected_columns:
            missing = sorted(expected_columns - columns)
            unexpected = sorted(columns - expected_columns)
            message = (
                f"Prepared bundle schema mismatch for {relative_path}; "
                f"missing={missing}, unexpected={unexpected}"
            )
            raise ValueError(message)
