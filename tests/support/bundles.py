"""Factories for small, valid prepared-bundle fixtures."""

from pathlib import Path

import polars as pl
import yaml

from danish_personas.io import sha256_file, write_json
from danish_personas.models import (
    PREPARED_BUNDLE_SCHEMA_VERSION,
    BundleManifest,
    SnapshotManifest,
)
from danish_personas.origin_labels import load_origin_label_contract
from tests.support.origin import origin_contract_fields


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
            "age_band": ["30-49", "30-49"],
            "municipality": ["Copenhagen", "Copenhagen"],
            "region_code": ["084", "084"],
            "region": ["Region Hovedstaden", "Region Hovedstaden"],
        }
    )
    ras209 = pl.DataFrame(
        {
            "municipality_code": ["101", "101"],
            "municipality": ["Copenhagen", "Copenhagen"],
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
    befolk = folk.drop("marital_status", "age_band")
    ras210 = pl.DataFrame(
        {
            "municipality_code": ["101", "101"],
            "status_group_code": ["00", "00"],
            "age_key": ["30", "30"],
            "sex": ["male", "female"],
            "count": [100, 100],
            "suppressed": [False, False],
            "municipality": ["Copenhagen", "Copenhagen"],
            "region_code": ["084", "084"],
            "region": ["Region Hovedstaden", "Region Hovedstaden"],
        }
    )
    contract = load_origin_label_contract()
    origin = pl.DataFrame(
        {
            "origin_country_code": list(contract.labels_en),
            "origin_country": list(contract.labels_en.values()),
            "origin_country_da": list(contract.labels_da.values()),
            "count": [
                107 if code == "5100" else 2 if code == "5103" else 0
                for code in contract.labels_en
            ],
        }
    )
    job_function = pl.DataFrame(
        [
            {
                "job_function_code": code,
                "job_function": f"{code} Fixture job function",
                "sex": sex,
                "count": index + 1,
            }
            for sex in ("female", "male")
            for index, code in enumerate(
                (
                    "01",
                    "02",
                    "03",
                    "11",
                    "12",
                    "13",
                    "14",
                    "21",
                    "22",
                    "23",
                    "24",
                    "25",
                    "26",
                    "31",
                    "32",
                    "33",
                    "34",
                    "35",
                    "41",
                    "42",
                    "43",
                    "44",
                    "51",
                    "52",
                    "53",
                    "54",
                    "61",
                    "62",
                    "71",
                    "72",
                    "73",
                    "74",
                    "75",
                    "81",
                    "82",
                    "83",
                    "91",
                    "92",
                    "93",
                    "94",
                    "95",
                    "96",
                )
            )
        ]
    )
    age_sampling = folk.select(
        "municipality_code",
        "municipality",
        "region_code",
        "region",
        "sex",
        "age",
        "count",
        "suppressed",
    ).with_columns(pl.lit("30-49").alias("age_band"))
    marital_sampling = folk.select(
        "municipality_code",
        "municipality",
        "region_code",
        "region",
        "sex",
        "marital_status",
        "count",
        "suppressed",
    ).with_columns(pl.lit("30-49").alias("age_band"))
    geography = pl.DataFrame(
        {
            "municipality_code": ["101"],
            "municipality": ["Copenhagen"],
            "landsdel_code": ["01"],
            "landsdel": ["Landsdel Byen København"],
            "region_code": ["084"],
            "region": ["Region Hovedstaden"],
        }
    )
    frames = {
        "folk1a_base_unpooled": folk,
        "folk2_origin_country_marginal": origin,
        "job_function_sex_marginal": job_function,
        "folk_age_sampling": age_sampling,
        "folk_marital_sampling": marital_sampling,
        "ras209_joint_unpooled": ras209,
        "ras209_sampling": ras209,
        "ras202_detail_unpooled": ras202.with_columns(pl.lit("30").alias("age_key")),
        "ras202_sampling": ras202,
        "befolk3_holdout": befolk,
        "ras210_holdout": ras210,
        "geography_hierarchy": geography,
    }
    files: dict[str, str] = {}
    for name, frame in frames.items():
        path = normalized / f"{name}.parquet"
        frame.write_parquet(path)
        files[str(path.relative_to(bundle_dir))] = sha256_file(path)
    source_report = bundle_dir / "source-preparation-report.json"
    write_json(path=source_report, payload={"passed": True})
    files[source_report.name] = sha256_file(source_report)
    manifest = BundleManifest(
        bundle_id="fixture-bundle",
        prepared_bundle_schema_version=PREPARED_BUNDLE_SCHEMA_VERSION,
        created_at="2026-09-14T00:00:00+00:00",
        source_lock_sha256="0" * 64,
        categories_sha256="1" * 64,
        source_snapshots=[
            SnapshotManifest(
                table_id="FOLK2",
                role="origin",
                period="2025",
                metadata_sha256=(
                    "cb2558d35bee7b3eed451984f457cc4dc7b683f71e67022d6767f3415d04884a"
                ),
                metadata_da_sha256=(
                    "f5c1f0a20f29372d6b222ce7a23cdc4ef0481d9e23fa6bd9b66b116e7adcb213"
                ),
                query_sha256="3" * 64,
                data_sha256="4" * 64,
                response_headers_sha256="5" * 64,
                retrieved_at="2026-09-14T00:00:00+00:00",
                data_bytes=1,
            )
        ],
        classification_snapshots=[],
        files=files,
        reference_periods={},
        assumptions=[],
        lons20_contract_version=1,
        lons20_contract_sha256="2" * 64,
        **origin_contract_fields(),
    )
    write_json(path=bundle_dir / "bundle-manifest.json", payload=manifest)
    sampling_path = root / "sampling.yaml"
    validation_path = root / "validation.yaml"
    categories_path = root / "categories.yaml"
    _write_yaml(
        path=sampling_path,
        payload={
            "version": 3,
            "seed": 42,
            "smoke_rows": 100,
            "statistical_rows": 200,
            "country": "Danmark",
            "minimum_age": 18,
            "maximum_age": 125,
            "publication_geography": "municipality",
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
            "version": 5,
            "absolute_proportion_tolerance": 0.2,
            "standard_error_multiplier": 5.0,
            "minimum_expected_count": 1.0,
            "maximum_ocean_pairwise_correlation": 1.0,
            "maximum_total_variation": {
                "fitted_marginal": 0.2,
                "heldout_marginal": 0.5,
            },
            "maximum_municipality_joint_total_variation": 0.2,
            "maximum_backoff_rate": 0.01,
            "smoke_maximum_total_variation": 0.2,
            "smoke_holdout_maximum_total_variation": 0.2,
            "mandatory_marginals": [
                "sex",
                "municipality_code",
                "marital_status",
                "age_band",
                "education_level",
                "labour_market_status",
                "origin_country",
                "job_function",
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


def refresh_bundle_manifest(*, bundle_dir: Path) -> None:
    """Recompute fixture manifest file checksums after a table mutation."""
    manifest_path = bundle_dir / "bundle-manifest.json"
    manifest = BundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    files = {
        relative_path: sha256_file(bundle_dir / relative_path)
        for relative_path in manifest.files
    }
    write_json(path=manifest_path, payload=manifest.model_copy(update={"files": files}))
