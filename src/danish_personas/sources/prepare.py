"""Normalise raw official aggregates into an offline sampling bundle."""

import csv
import logging
import math
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import polars as pl

from ..io import load_yaml_model, sha256_file, sha256_text, verify_checksums, write_json
from ..models import (
    DISCO_TWO_DIGIT_CODES,
    PREPARED_BUNDLE_SCHEMA_VERSION,
    BundleManifest,
    CategoryConfig,
    ClassificationManifest,
    LockedSource,
    SnapshotManifest,
    SourceLock,
    SourceMetadataExpectations,
    StatBankMetadata,
)
from ..origin_labels import (
    DEFAULT_ORIGIN_LABEL_CONTRACT_PATH,
    ORIGIN_LABEL_CONTRACT_SHA256,
    ORIGIN_LABEL_COUNT,
    OriginLabelContract,
    bind_origin_triples,
    load_origin_label_contract,
)
from .bundle import verify_prepared_bundle
from .classification import classification_snapshot_dir, verify_classification_snapshot
from .lons20 import (
    DEFAULT_LONS20_CONTRACT_PATH,
    load_lons20_contract,
    lons20_expectations,
)
from .statbank import (
    _validate_metadata_expectations,
    source_query_content,
    source_snapshot_dir,
)

LOGGER = logging.getLogger(__name__)
REGION_PREFIX = "Region "
REGION_LEVEL = "1"
LANDSDEL_LEVEL = "2"
MUNICIPALITY_LEVEL = "3"
LONS20_DIMENSIONS = {
    "ARBF": list(DISCO_TWO_DIGIT_CODES),
    "SEKTOR": ["1000"],
    "AFLOEN": ["TIFA"],
    "LONGRP": ["LTOT"],
    "LØNMÅL": ["ANTAL"],
    "KØN": ["M", "K"],
    "Tid": ["2024"],
}


def _region_map_from_geography(geography: pl.DataFrame) -> dict[str, tuple[str, str]]:
    """Map each municipality to its region code and label.

    Args:
        geography:
            Hierarchy read from the official classification.

    Returns:
        Municipality code mapped to its region code and region name.
    """
    return {
        row["municipality_code"]: (row["region_code"], row["region"])
        for row in geography.iter_rows(named=True)
    }


def prepare_bundle(
    lock_path: Path,
    categories_path: Path,
    raw_dir: Path,
    output_dir: Path,
    contract_path: Path = DEFAULT_LONS20_CONTRACT_PATH,
    origin_labels_contract_path: Path = DEFAULT_ORIGIN_LABEL_CONTRACT_PATH,
) -> Path:
    """Build a validated, immutable source bundle.

    Args:
        lock_path:
            Resolved source lock.
        categories_path:
            Canonical category mappings.
        raw_dir:
            Root directory containing raw snapshots.
        output_dir:
            Root destination for prepared bundles.
        contract_path:
            Separately reviewed canonical LONS20 contract.
        origin_labels_contract_path:
            Separately reviewed Danish FOLK2 label contract.

    Returns:
        Prepared bundle directory.

    Raises:
        ValueError:
            If the lock has no geography classification.
    """
    if origin_labels_contract_path != DEFAULT_ORIGIN_LABEL_CONTRACT_PATH:
        raise ValueError("Only the canonical origin-label contract is permitted")
    lock = load_yaml_model(path=lock_path, model=SourceLock)
    contract = load_lons20_contract(path=contract_path)
    origin_labels_contract = load_origin_label_contract(
        path=origin_labels_contract_path
    )
    contract_expectations = lons20_expectations(contract=contract)
    origin_labels_contract_sha256 = sha256_file(origin_labels_contract_path)
    if origin_labels_contract_sha256 != ORIGIN_LABEL_CONTRACT_SHA256:
        raise ValueError(
            "Origin-label contract bytes do not match the reviewed contract"
        )
    origin_labels_contract_content = origin_labels_contract_path.read_text(
        encoding="utf-8"
    )
    origin_labels_contract_relative_path = _repository_relative_path(
        origin_labels_contract_path
    )
    job_source = _validate_lons20_source(
        lock=lock, contract_expectations=contract_expectations
    )
    categories = load_yaml_model(path=categories_path, model=CategoryConfig)
    contract_sha256 = sha256_file(contract_path)
    bundle_id = _bundle_id(
        lock_path=lock_path,
        categories_path=categories_path,
        contract_path=contract_path,
        contract_version=contract.version,
        origin_labels_contract_path=origin_labels_contract_relative_path,
        origin_labels_contract_version=origin_labels_contract.version,
        origin_labels_contract_sha256=origin_labels_contract_sha256,
    )
    bundle_dir = output_dir / bundle_id
    manifest_path = bundle_dir / "bundle-manifest.json"
    if manifest_path.exists():
        _verify_existing_bundle(bundle_dir=bundle_dir, manifest_path=manifest_path)
        LOGGER.info("Reusing prepared bundle %s", bundle_id)
        return bundle_dir

    normalized_dir = bundle_dir / "normalized"
    snapshots: list[SnapshotManifest] = []
    source_frames: dict[str, pl.DataFrame] = {}
    metadata_by_table: dict[str, StatBankMetadata] = {}
    for source in lock.sources:
        snapshot_dir = source_snapshot_dir(source=source, raw_dir=raw_dir)
        snapshot = SnapshotManifest.model_validate_json(
            (snapshot_dir / "snapshot-manifest.json").read_text(encoding="utf-8")
        )
        verify_raw_snapshot(
            snapshot_dir=snapshot_dir,
            snapshot=snapshot,
            table_id=source.table_id,
            role=source.role,
            period=source.period,
            expected_query=source_query_content(source=source),
        )
        snapshots.append(snapshot)
        metadata = StatBankMetadata.model_validate_json(
            (snapshot_dir / "metadata-en.json").read_text(encoding="utf-8")
        )
        expectations = (
            contract_expectations
            if source.table_id == "LONS20"
            else source.metadata_expectations
        )
        _validate_metadata_expectations(
            table_id=source.table_id,
            metadata=metadata,
            expectations=expectations,
            dimensions=source.dimensions,
        )
        metadata_by_table[source.table_id] = metadata
        source_frames[source.table_id] = _read_source(
            csv_path=snapshot_dir / "data.csv", dimension_codes=list(source.dimensions)
        )

    origin_source = next(
        source for source in lock.sources if source.table_id == "FOLK2"
    )
    ras209_source = next(
        source for source in lock.sources if source.table_id == "RAS209"
    )
    origin_metadata_da_path = (
        source_snapshot_dir(source=origin_source, raw_dir=raw_dir) / "metadata-da.json"
    )
    origin_metadata_da_sha256 = sha256_file(origin_metadata_da_path)
    origin_metadata_en_path = (
        source_snapshot_dir(source=origin_source, raw_dir=raw_dir) / "metadata-en.json"
    )
    origin_metadata_en_sha256 = sha256_file(origin_metadata_en_path)
    origin_metadata_da = StatBankMetadata.model_validate_json(
        origin_metadata_da_path.read_bytes()
    )
    english_origin_labels, danish_origin_labels = _validate_origin_metadata(
        lock_codes=origin_source.dimensions["IELAND"],
        english_metadata=metadata_by_table["FOLK2"],
        danish_metadata=origin_metadata_da,
        contract=origin_labels_contract,
        metadata_en_sha256=origin_metadata_en_sha256,
        metadata_da_sha256=origin_metadata_da_sha256,
    )
    prepared_source_frames = dict(source_frames)
    prepared_source_frames["FOLK2"] = _materialise_origin_zero_codes(
        raw_frame=source_frames["FOLK2"],
        selected_codes=origin_source.dimensions["IELAND"],
        expected_zero_codes=origin_source.expected_zero_codes,
    )

    classification_snapshots: list[ClassificationManifest] = []
    geography_csv_path: Path | None = None
    for classification in lock.classifications:
        classification_dir = classification_snapshot_dir(
            classification=classification, raw_dir=raw_dir
        )
        classification_snapshot = ClassificationManifest.model_validate_json(
            (classification_dir / "snapshot-manifest.json").read_text(encoding="utf-8")
        )
        verify_classification_snapshot(
            snapshot_dir=classification_dir,
            snapshot=classification_snapshot,
            classification=classification,
        )
        classification_snapshots.append(classification_snapshot)
        if classification.role == "geography_hierarchy":
            geography_csv_path = classification_dir / "data.csv"
    if geography_csv_path is None:
        message = "The source lock has no geography_hierarchy classification"
        raise ValueError(message)

    geography = read_geography_classification(csv_path=geography_csv_path)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    frames = _normalise_frames(
        raw_frames=prepared_source_frames,
        metadata_by_table=metadata_by_table,
        categories=categories,
        geography=geography,
        release_rows=lock.release_rows,
        minimum_source_count=lock.minimum_source_count,
        minimum_expected_release_count=lock.minimum_expected_release_count,
        job_function_codes=job_source.dimensions["ARBF"],
        origin_labels=english_origin_labels,
        origin_labels_da=danish_origin_labels,
    )
    _verify_ras209_municipality_sets(
        locked_codes=set(ras209_source.dimensions["OMRÅDE"]),
        geography=geography,
        prepared=frames["ras209_sampling"],
    )
    files: dict[str, str] = {}
    for name, frame in frames.items():
        path = normalized_dir / f"{name}.parquet"
        frame = frame.sort(sorted(frame.columns))
        frame.write_parquet(path)
        files[str(path.relative_to(bundle_dir))] = sha256_file(path)

    geography_path = normalized_dir / "geography_hierarchy.parquet"
    geography.sort(sorted(geography.columns)).write_parquet(geography_path)
    files[str(geography_path.relative_to(bundle_dir))] = sha256_file(geography_path)

    geography_metrics = _geography_metrics(
        geography=geography,
        statbank_region_map=_build_region_map(metadata=metadata_by_table["FOLK1A"]),
    )
    ras209_geography_metrics = _geography_metrics(
        geography=geography,
        statbank_region_map=_build_region_map(metadata=metadata_by_table["RAS209"]),
    )
    geography_metrics["ras209_mapping"] = ras209_geography_metrics
    geography_metrics["passed"] = bool(geography_metrics["passed"]) and bool(
        ras209_geography_metrics["passed"]
    )
    origin_metrics = _origin_country_metrics(
        raw_frame=source_frames["FOLK2"],
        prepared_frame=frames["folk2_origin_country_marginal"],
        official_labels=_metadata_labels(metadata_by_table["FOLK2"])["IELAND"],
        selected_codes=origin_source.dimensions["IELAND"],
        expected_zero_codes=origin_source.expected_zero_codes,
        official_labels_da=danish_origin_labels,
    )
    source_metrics = _source_metrics(
        frames=frames,
        geography_metrics=geography_metrics,
        origin_metrics=origin_metrics,
        lons20_contract_version=contract.version,
        lons20_contract_sha256=contract_sha256,
        origin_labels_contract=origin_labels_contract,
        origin_labels_contract_path=origin_labels_contract_relative_path,
        origin_labels_contract_sha256=origin_labels_contract_sha256,
        origin_metadata_en_sha256=origin_metadata_en_sha256,
        origin_metadata_da_sha256=origin_metadata_da_sha256,
    )
    source_report_path = bundle_dir / "source-preparation-report.json"
    write_json(path=source_report_path, payload=source_metrics)
    files[str(source_report_path.relative_to(bundle_dir))] = sha256_file(
        source_report_path
    )
    markdown_path = bundle_dir / "source-preparation-report.md"
    markdown_path.write_text(
        _source_report_markdown(bundle_id=bundle_id, metrics=source_metrics),
        encoding="utf-8",
    )
    files[str(markdown_path.relative_to(bundle_dir))] = sha256_file(markdown_path)

    manifest = BundleManifest(
        bundle_id=bundle_id,
        prepared_bundle_schema_version=PREPARED_BUNDLE_SCHEMA_VERSION,
        created_at=_now(),
        source_lock_sha256=sha256_file(lock_path),
        categories_sha256=sha256_file(categories_path),
        source_snapshots=snapshots,
        classification_snapshots=classification_snapshots,
        files=files,
        reference_periods={source.role: source.period for source in lock.sources},
        lons20_contract_version=contract.version,
        lons20_contract_sha256=contract_sha256,
        origin_labels_contract_path=origin_labels_contract_relative_path,
        origin_labels_contract_version=origin_labels_contract.version,
        origin_labels_contract_sha256=origin_labels_contract_sha256,
        origin_labels_contract_content=origin_labels_contract_content,
        assumptions=[
            "FOLK1A 2025Q1 is the demographic base nearest RAS November 2024.",
            "FOLK1A ages 16-19 estimate the adult share of RAS209's 16-19 band.",
            "RAS209 jointly supplies broad education and labour-market status.",
            "RAS202 refines detailed status only within the RAS209 broad status.",
            "The RAS209 67+ education band is a proxy for ages 70 and over.",
            "BEFOLK3 and RAS210 are held-out diagnostics, not fitted microdata.",
            "The municipality-region hierarchy comes from the official DST "
            "classification, not from StatBank metadata ordering.",
            "FOLK2 is an independent national adult marginal by official IELAND "
            "code and label; it is neither ethnicity nor citizenship.",
            "FOLK2 origin categories are retained verbatim, including Stateless "
            "and Not stated; no continents, regions, or correlations are inferred.",
            "FOLK2 is sampled independently into Phase 2 origin fields and "
            "withheld from both LLM stages.",
            "LONS20 supplies only a 2024 sex-conditional marginal over exactly "
            "the 42 two-digit DISCO-08 job-function groups.",
            "LONS20 covers all public employees and private organisations with "
            "at least 10 full-time-equivalent employees; smaller private "
            "organisations and other earnings-statistics exclusions are absent.",
            "Job function is a synthetic allocation for eligible RAS202 employee "
            "statuses, not an observed occupation or an all-worker distribution.",
            "Job function is not conditioned on municipality, origin, age, "
            "education, OCEAN, or any unsupported joint and is withheld from both "
            "LLM stages.",
            "OCEAN traits are a documented design distribution, not official "
            "statistics.",
        ],
    )
    write_json(path=manifest_path, payload=manifest)
    LOGGER.info("Prepared source bundle %s", bundle_id)
    return bundle_dir


def _build_region_map(metadata: StatBankMetadata) -> dict[str, tuple[str, str]]:
    area = next(variable for variable in metadata.variables if variable.id == "OMRÅDE")
    region_code = ""
    region_name = ""
    mapping: dict[str, tuple[str, str]] = {}
    for value in area.values:
        if value.text.startswith(REGION_PREFIX):
            region_code = value.id
            region_name = value.text
        elif len(value.id) == 3 and value.id != "000":
            if not region_code:
                message = f"Municipality {value.id} appears before a region marker"
                raise ValueError(message)
            mapping[value.id] = (region_code, region_name)
    return mapping


def _bundle_id(
    *,
    lock_path: Path,
    categories_path: Path,
    contract_path: Path,
    contract_version: int,
    origin_labels_contract_path: str = "",
    origin_labels_contract_version: int = 0,
    origin_labels_contract_sha256: str = "",
) -> str:
    """Return the content identity for a prepared source bundle."""
    return sha256_text(
        f"{PREPARED_BUNDLE_SCHEMA_VERSION}:{sha256_file(lock_path)}:"
        f"{sha256_file(categories_path)}:{contract_version}:{sha256_file(contract_path)}:"
        f"{origin_labels_contract_path}:{origin_labels_contract_version}:"
        f"{origin_labels_contract_sha256}"
    )[:16]


def _geography_metrics(
    geography: pl.DataFrame, statbank_region_map: dict[str, tuple[str, str]]
) -> dict[str, object]:
    """Cross-check the official hierarchy against StatBank table metadata.

    Args:
        geography:
            Hierarchy read from the official classification.
        statbank_region_map:
            Municipality-to-region map inferred from FOLK1A's metadata.

    Returns:
        Counts, disagreements, and the overall pass flag for the report.
    """
    municipality_codes = geography.get_column("municipality_code")
    duplicate_municipalities = sorted(
        geography.filter(pl.col("municipality_code").is_duplicated())
        .get_column("municipality_code")
        .unique()
        .to_list()
    )
    classification_codes = dict(
        zip(municipality_codes, geography.get_column("region_code"), strict=True)
    )
    statbank_codes = {code: region[0] for code, region in statbank_region_map.items()}
    disagreements = sorted(
        code
        for code in classification_codes.keys() & statbank_codes.keys()
        if classification_codes[code] != statbank_codes[code]
    )
    missing = sorted(statbank_codes.keys() - classification_codes.keys())
    extra = sorted(classification_codes.keys() - statbank_codes.keys())
    complete = geography.null_count().sum_horizontal().item() == 0
    passed = (
        not disagreements
        and not missing
        and not extra
        and not duplicate_municipalities
        and complete
    )
    return {
        "municipalities": geography.height,
        "regions": geography.get_column("region_code").n_unique(),
        "landsdele": geography.get_column("landsdel_code").n_unique(),
        "statbank_disagreements": disagreements,
        "missing_from_classification": missing,
        "extra_in_classification": extra,
        "duplicate_municipalities": duplicate_municipalities,
        "complete": complete,
        "passed": passed,
    }


def _materialise_origin_zero_codes(
    raw_frame: pl.DataFrame,
    selected_codes: list[str],
    expected_zero_codes: list[str] | None = None,
) -> pl.DataFrame:
    """Materialise only reviewed zero-count omissions from a BULK response.

    StatBank's BULK response omits combinations whose observation is zero. The source
    lock therefore records the selected IELAND codes that are approved omissions. Any
    other missing selected code is an unexpected source change and fails preparation.

    Args:
        raw_frame:
            Parsed FOLK2 response rows.
        selected_codes:
            IELAND values frozen in the source lock.
        expected_zero_codes (optional):
            Reviewed selected IELAND values omitted as all-zero BULK partitions.
            Defaults to an empty set.

    Returns:
        Parsed rows with explicit zero rows for approved absent IELAND codes.

    Raises:
        ValueError:
            If the expected-zero set is not a subset of the selection, contains an
            observed code, or does not account for every missing selected code.
    """
    expected_codes = set(selected_codes)
    approved_codes = set(expected_zero_codes or [])
    observed_codes = set(raw_frame.get_column("IELAND").to_list())
    missing_codes = expected_codes - observed_codes
    invalid_approved = sorted(approved_codes - expected_codes)
    if invalid_approved:
        message = f"Expected-zero IELAND codes are not selected: {invalid_approved}"
        raise ValueError(message)
    observed_approved = sorted(approved_codes & observed_codes)
    if observed_approved:
        message = (
            "Expected-zero IELAND codes are present in raw BULK data: "
            f"{observed_approved}"
        )
        raise ValueError(message)
    unexpected_missing = sorted(missing_codes - approved_codes)
    if unexpected_missing:
        message = f"Unexpected missing FOLK2 IELAND codes: {unexpected_missing}"
        raise ValueError(message)
    if not missing_codes:
        return raw_frame
    zero_rows = pl.DataFrame(
        {
            "IELAND": sorted(missing_codes),
            "count": [0] * len(missing_codes),
            "suppressed": [False] * len(missing_codes),
        }
    )
    return pl.concat([raw_frame, zero_rows], how="diagonal_relaxed")


def _metadata_labels(metadata: StatBankMetadata) -> dict[str, dict[str, str]]:
    return {
        variable.id: {value.id: value.text for value in variable.values}
        for variable in metadata.variables
    }


def _normalise_frames(
    raw_frames: dict[str, pl.DataFrame],
    metadata_by_table: dict[str, StatBankMetadata],
    categories: CategoryConfig,
    geography: pl.DataFrame,
    release_rows: int,
    minimum_source_count: int,
    minimum_expected_release_count: int,
    job_function_codes: list[str],
    origin_labels: dict[str, str] | None = None,
    origin_labels_da: dict[str, str] | None = None,
) -> dict[str, pl.DataFrame]:
    _reject_negative_counts(raw_frames=raw_frames)
    labels = {
        table: _metadata_labels(metadata=metadata)
        for table, metadata in metadata_by_table.items()
    }
    status_map = _invert_status_mapping(categories=categories)

    folk2 = _origin_country_marginal(
        raw_frame=raw_frames["FOLK2"],
        official_labels=origin_labels or labels["FOLK2"]["IELAND"],
        official_labels_da=origin_labels_da,
    )
    job_function = _job_function_sex_marginal(
        raw_frame=raw_frames["LONS20"],
        official_labels=labels["LONS20"]["ARBF"],
        selected_codes=job_function_codes,
        sex_mapping=categories.sex,
    )

    folk_calibration = raw_frames["FOLK1A"].select(
        pl.col("OMRÅDE").alias("municipality_code"),
        pl.col("KØN").replace_strict(categories.sex).alias("sex"),
        pl.col("ALDER").cast(pl.Int16).alias("age"),
        pl.col("CIVILSTAND")
        .replace_strict(categories.marital_status)
        .alias("marital_status"),
        pl.col("count"),
        pl.col("suppressed"),
    )
    folk_calibration = _add_geography(frame=folk_calibration, geography=geography)
    folk = folk_calibration.filter(pl.col("age") >= 18).with_columns(
        _age_band_expression().alias("age_band")
    )
    # Municipality is an invariant, so source-cell filtering must not force a
    # regional or national fallback. Structural zero handling remains in the sampler.
    folk_threshold = 0
    folk_age_sampling = (
        folk.group_by(
            [
                "municipality_code",
                "municipality",
                "region_code",
                "region",
                "age_band",
                "sex",
                "age",
            ]
        )
        .agg(pl.col("count").sum(), pl.col("suppressed").any())
        .filter(pl.col("count") >= folk_threshold)
    )
    folk_marital_sampling = (
        folk.group_by(
            [
                "municipality_code",
                "municipality",
                "region_code",
                "region",
                "age_band",
                "sex",
                "marital_status",
            ]
        )
        .agg(pl.col("count").sum(), pl.col("suppressed").any())
        .filter(pl.col("count") >= folk_threshold)
    )

    ras209 = raw_frames["RAS209"].select(
        pl.col("OMRÅDE").alias("municipality_code"),
        pl.col("UDDANNELSE").alias("education_source_code"),
        pl.col("UDDANNELSE")
        .replace_strict(categories.education)
        .alias("education_level"),
        pl.col("SOCIO").replace_strict(status_map).alias("labour_market_status"),
        pl.col("ALDER").alias("age_band"),
        pl.col("KØN").replace_strict(categories.sex).alias("sex"),
        pl.col("count"),
        pl.col("suppressed"),
    )
    ras209 = _add_geography(frame=ras209, geography=geography)
    ras209 = _adjust_young_adult_counts(
        ras209=ras209, folk_calibration=folk_calibration
    )
    ras209_joint = ras209.group_by(
        [
            "municipality_code",
            "municipality",
            "region_code",
            "region",
            "age_band",
            "sex",
            "education_source_code",
            "education_level",
            "labour_market_status",
        ]
    ).agg(pl.col("count").sum(), pl.col("suppressed").any())
    ras209_sampling = _pool_ras209(
        frame=ras209_joint,
        education_pooling=categories.education_pooling,
        release_rows=release_rows,
        minimum_source_count=minimum_source_count,
        minimum_expected_release_count=minimum_expected_release_count,
    )

    ras202 = (
        raw_frames["RAS202"]
        .select(
            pl.col("SOCIO").alias("detailed_status_code"),
            pl.col("SOCIO")
            .replace_strict(labels["RAS202"]["SOCIO"])
            .alias("detailed_status"),
            pl.col("SOCIO").replace_strict(status_map).alias("labour_market_status"),
            pl.col("ALDER").alias("age_key"),
            pl.col("KOEN").replace_strict(categories.sex).alias("sex"),
            pl.col("count"),
            pl.col("suppressed"),
        )
        .with_columns(_ras_age_band_expression().alias("age_band"))
    )
    ras202_threshold = _release_threshold(
        total=float(ras202.get_column("count").sum()),
        release_rows=release_rows,
        minimum_source_count=minimum_source_count,
        minimum_expected_release_count=minimum_expected_release_count,
    )
    ras202_sampling = (
        ras202.group_by(
            [
                "age_band",
                "sex",
                "labour_market_status",
                "detailed_status_code",
                "detailed_status",
            ]
        )
        .agg(pl.col("count").sum(), pl.col("suppressed").any())
        .filter(pl.col("count") >= ras202_threshold)
    )

    befolk = raw_frames["BEFOLK3"].select(
        pl.col("OMRÅDE").alias("municipality_code"),
        pl.col("KØN").replace_strict(categories.sex).alias("sex"),
        pl.col("ALDER").cast(pl.Int16).alias("age"),
        pl.col("count"),
        pl.col("suppressed"),
    )
    befolk = _add_geography(frame=befolk, geography=geography)

    ras210 = raw_frames["RAS210"].select(
        pl.col("BOPKOM").alias("municipality_code"),
        pl.col("SOCIO").alias("status_group_code"),
        pl.col("ALERAMS").alias("age_key"),
        pl.col("KON").replace_strict(categories.sex).alias("sex"),
        pl.col("count"),
        pl.col("suppressed"),
    )
    ras210 = _add_geography(frame=ras210, geography=geography)
    return {
        "folk2_origin_country_marginal": folk2,
        "job_function_sex_marginal": job_function,
        "folk1a_base_unpooled": folk,
        "folk_age_sampling": folk_age_sampling,
        "folk_marital_sampling": folk_marital_sampling,
        "ras209_joint_unpooled": ras209_joint,
        "ras209_sampling": ras209_sampling,
        "ras202_detail_unpooled": ras202,
        "ras202_sampling": ras202_sampling,
        "befolk3_holdout": befolk,
        "ras210_holdout": ras210,
    }


def _add_geography(frame: pl.DataFrame, geography: pl.DataFrame) -> pl.DataFrame:
    """Attach names and parents using the validated official hierarchy.

    Args:
        frame:
            Municipality-keyed source rows.
        geography:
            Validated official geography hierarchy.

    Returns:
        Source rows with official municipality and region fields.

    Raises:
        ValueError:
            If a municipality is duplicated in or absent from the hierarchy.
    """
    lookup = geography.select(
        "municipality_code", "municipality", "region_code", "region"
    )
    if lookup.get_column("municipality_code").n_unique() != lookup.height:
        raise ValueError("Geography hierarchy contains duplicate municipality codes")
    unknown = set(frame.get_column("municipality_code").to_list()) - set(
        lookup.get_column("municipality_code").to_list()
    )
    if unknown:
        raise ValueError(
            f"Municipalities missing from official hierarchy: {sorted(unknown)}"
        )
    return frame.join(lookup, on="municipality_code", how="left", validate="m:1")


def _adjust_young_adult_counts(
    ras209: pl.DataFrame, folk_calibration: pl.DataFrame
) -> pl.DataFrame:
    band = folk_calibration.filter(pl.col("age").is_between(16, 19))
    totals = band.group_by(["municipality_code", "sex"]).agg(
        pl.col("count").sum().alias("band_count")
    )
    adults = (
        band.filter(pl.col("age") >= 18)
        .group_by(["municipality_code", "sex"])
        .agg(pl.col("count").sum().alias("adult_count"))
    )
    factors = totals.join(adults, on=["municipality_code", "sex"]).with_columns(
        (pl.col("adult_count") / pl.col("band_count")).alias("adult_share")
    )
    return (
        ras209.join(
            factors.select("municipality_code", "sex", "adult_share"),
            on=["municipality_code", "sex"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("age_band") == "16-19")
            .then(pl.col("count") * pl.col("adult_share"))
            .otherwise(pl.col("count"))
            .alias("count")
        )
        .drop("adult_share")
    )


def _age_band_expression() -> pl.Expr:
    return (
        pl.when(pl.col("age") <= 29)
        .then(pl.lit("18-29"))
        .when(pl.col("age") <= 49)
        .then(pl.lit("30-49"))
        .when(pl.col("age") <= 66)
        .then(pl.lit("50-66"))
        .otherwise(pl.lit("67+"))
    )


def _invert_status_mapping(categories: CategoryConfig) -> dict[str, str]:
    result = {
        code: status
        for status, codes in categories.labour_market_status.items()
        for code in codes
    }
    duplicates = sum(len(codes) for codes in categories.labour_market_status.values())
    if len(result) != duplicates:
        message = "Labour-market status codes must be unique"
        raise ValueError(message)
    return result


def _job_function_sex_marginal(
    raw_frame: pl.DataFrame,
    official_labels: dict[str, str],
    selected_codes: list[str],
    sex_mapping: dict[str, str],
) -> pl.DataFrame:
    """Validate and prepare the official LONS20 sex marginal.

    Returns:
        Official code, label, sex, and positive count cells.

    Raises:
        ValueError:
            If source coverage, hierarchy, labels, or counts violate the contract.
    """
    _validate_lons20_cells(raw_frame=raw_frame, selected_codes=selected_codes)
    expected_codes = set(DISCO_TWO_DIGIT_CODES)
    labels = {code: official_labels.get(code) for code in expected_codes}
    if any(label is None or not label.strip() for label in labels.values()):
        raise ValueError("LONS20 contains a blank or missing official label")
    unexpected_labels = raw_frame.filter(
        pl.col("ARBF__label")
        != pl.col("ARBF").replace_strict(labels, return_dtype=pl.String)
    )
    if not unexpected_labels.is_empty():
        raise ValueError("LONS20 contains an unexpected official code-label pair")
    prepared = raw_frame.select(
        pl.col("ARBF").alias("job_function_code"),
        pl.col("ARBF").replace_strict(labels).alias("job_function"),
        pl.col("KØN").replace_strict(sex_mapping).alias("sex"),
        pl.col("count"),
    ).sort("sex", "job_function_code")
    if prepared.select("job_function_code", "job_function").unique().height != len(
        DISCO_TWO_DIGIT_CODES
    ):
        raise ValueError("LONS20 contains an unexpected code-label mapping")
    return prepared


def _validate_lons20_cells(raw_frame: pl.DataFrame, selected_codes: list[str]) -> None:
    if selected_codes != list(DISCO_TWO_DIGIT_CODES):
        raise ValueError("LONS20 ARBF selection is not the fixed two-digit partition")
    required = {*LONS20_DIMENSIONS, "ARBF__label", "count", "suppressed"}
    missing = sorted(required - set(raw_frame.columns))
    if missing:
        raise ValueError(f"LONS20 response is missing columns: {missing}")
    if raw_frame.height != len(DISCO_TWO_DIGIT_CODES) * 2:
        raise ValueError("LONS20 must contain one row per job-function code and sex")
    counts = raw_frame.get_column("count")
    integer_types = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    }
    if counts.dtype not in integer_types or bool((counts <= 0).any()):
        raise ValueError("LONS20 counts must be positive integers")
    if bool(raw_frame.get_column("suppressed").any()):
        raise ValueError("LONS20 contains suppressed values")
    _validate_lons20_partition(raw_frame=raw_frame)


def _validate_lons20_partition(raw_frame: pl.DataFrame) -> None:
    for dimension, expected in LONS20_DIMENSIONS.items():
        if dimension not in {"ARBF", "KØN"} and set(
            raw_frame.get_column(dimension).unique()
        ) != set(expected):
            raise ValueError(f"LONS20 has unexpected {dimension} coverage")
    codes = raw_frame.get_column("ARBF")
    if set(codes.unique()) != set(DISCO_TWO_DIGIT_CODES) or any(
        len(str(code)) != 2 or not str(code).isdigit() for code in codes
    ):
        raise ValueError("LONS20 contains totals or mixed DISCO-08 hierarchy levels")
    if raw_frame.select("ARBF", "KØN").n_unique() != raw_frame.height:
        raise ValueError("LONS20 contains duplicate code-sex cells")
    coverage = raw_frame.group_by("ARBF").agg(pl.col("KØN").unique().sort())
    if any(set(sexes) != {"M", "K"} for sexes in coverage.get_column("KØN")):
        raise ValueError("LONS20 is missing a sex distribution")


def _origin_country_marginal(
    raw_frame: pl.DataFrame,
    official_labels: dict[str, str],
    official_labels_da: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Aggregate FOLK2 to the official national origin marginal.

    Args:
        raw_frame:
            Selected FOLK2 cells.
        official_labels:
            Official IELAND code-to-label mapping from table metadata.
        official_labels_da (optional):
            Official Danish IELAND code-to-label mapping.

    Returns:
        One row per official IELAND value, including zero-count categories.

    Raises:
        ValueError: If English and Danish code order differs.
    """
    counts = raw_frame.group_by("IELAND").agg(pl.col("count").sum())
    categories_data: dict[str, list[str]] = {
        "origin_country_code": list(official_labels),
        "origin_country": list(official_labels.values()),
    }
    if official_labels_da is not None:
        if tuple(official_labels) != tuple(official_labels_da):
            raise ValueError("English and Danish FOLK2 label code order differs")
        categories_data["origin_country_da"] = list(official_labels_da.values())
    categories = pl.DataFrame(categories_data)
    return (
        categories.join(
            counts, left_on="origin_country_code", right_on="IELAND", how="left"
        )
        .with_columns(pl.col("count").fill_null(0).cast(pl.Int64))
        .select(
            "origin_country_code",
            "origin_country",
            *(["origin_country_da"] if official_labels_da is not None else []),
            "count",
        )
    )


def _pool_ras209(
    frame: pl.DataFrame,
    education_pooling: dict[str, str],
    release_rows: int,
    minimum_source_count: int,
    minimum_expected_release_count: int,
) -> pl.DataFrame:
    age_pooling = {
        "16-19": "18-29",
        "20-24": "18-29",
        "25-29": "18-29",
        "30-34": "30-49",
        "35-39": "30-49",
        "40-44": "30-49",
        "45-49": "30-49",
        "50-54": "50-66",
        "55-59": "50-66",
        "60-64": "50-66",
        "65-66": "50-66",
        "67-": "67+",
    }
    education_codes = {
        "primary": "H10",
        "secondary_or_vocational": "H20-H35",
        "higher_education": "H40-H80",
        "not_stated": "H90",
    }
    total = float(frame.get_column("count").sum())
    threshold = (
        0
        if "municipality_code" in frame.columns
        else _release_threshold(
            total=total,
            release_rows=release_rows,
            minimum_source_count=minimum_source_count,
            minimum_expected_release_count=minimum_expected_release_count,
        )
    )
    pooled = (
        frame.with_columns(
            pl.col("education_level")
            .replace_strict(education_pooling)
            .alias("education_level"),
            pl.col("age_band").replace_strict(age_pooling).alias("age_band"),
        )
        .with_columns(
            pl.col("education_level")
            .replace_strict(education_codes)
            .alias("education_source_code")
        )
        .group_by(
            [
                "municipality_code",
                "municipality",
                "region_code",
                "region",
                "age_band",
                "sex",
                "education_source_code",
                "education_level",
                "labour_market_status",
            ]
        )
        .agg(pl.col("count").sum(), pl.col("suppressed").any())
        .with_columns(pl.col("count").round(6))
        .filter(pl.col("count") >= threshold)
    )
    coverage = float(pooled.get_column("count").sum()) / total
    if coverage < 0.99:
        message = f"Sparse-cell pooling retains only {coverage:.2%} of RAS209"
        raise ValueError(message)
    return pooled


def _release_threshold(
    total: float,
    release_rows: int,
    minimum_source_count: int,
    minimum_expected_release_count: int,
) -> int:
    expected_threshold = math.ceil(
        minimum_expected_release_count * total / release_rows
    )
    return max(minimum_source_count, expected_threshold)


def _ras_age_band_expression() -> pl.Expr:
    numeric_age = pl.col("age_key").cast(pl.Int16, strict=False)
    return (
        pl.when(numeric_age <= 29)
        .then(pl.lit("18-29"))
        .when(numeric_age <= 49)
        .then(pl.lit("30-49"))
        .when(numeric_age <= 66)
        .then(pl.lit("50-66"))
        .otherwise(pl.lit("67+"))
    )


def _reject_negative_counts(raw_frames: dict[str, pl.DataFrame]) -> None:
    """Reject negative source counts before any source transformation.

    Raises:
        ValueError:
            If any source contains a negative count.
    """
    for table_id, frame in raw_frames.items():
        if bool((frame.get_column("count") < 0).any()):
            raise ValueError(f"{table_id} contains negative counts")


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _origin_country_metrics(
    raw_frame: pl.DataFrame,
    prepared_frame: pl.DataFrame,
    official_labels: dict[str, str],
    selected_codes: list[str],
    expected_zero_codes: list[str] | None = None,
    official_labels_da: dict[str, str] | None = None,
) -> dict[str, object]:
    """Validate the FOLK2 origin marginal and its official partition.

    Args:
        raw_frame:
            Selected FOLK2 cells before aggregation.
        prepared_frame:
            Prepared national marginal.
        official_labels:
            Official IELAND code-to-label mapping from table metadata.
        selected_codes:
            IELAND values frozen in the source lock.
        expected_zero_codes (optional):
            Reviewed selected IELAND values omitted as all-zero BULK partitions.
            Defaults to an empty set.
        official_labels_da (optional):
            Official Danish IELAND code-to-label mapping.

    Returns:
        FOLK2-specific validation metrics.
    """
    observed_codes = set(raw_frame.get_column("IELAND").to_list())
    expected_codes = set(selected_codes)
    approved_zero_codes = set(expected_zero_codes or [])
    missing_raw = sorted(expected_codes - observed_codes)
    unexpected_missing = sorted(set(missing_raw) - approved_zero_codes)
    invalid_approved = sorted(approved_zero_codes - expected_codes)
    expected_zero_not_missing = sorted(approved_zero_codes & observed_codes)
    prepared_code_values = prepared_frame.get_column("origin_country_code").to_list()
    prepared_label_values = prepared_frame.get_column("origin_country").to_list()
    prepared_da_values = (
        prepared_frame.get_column("origin_country_da").to_list()
        if official_labels_da is not None
        else []
    )
    prepared_codes = set(prepared_code_values)
    selected_metadata = {
        code: official_labels[code]
        for code in expected_codes
        if code in official_labels
    }
    prepared_mapping = dict(
        zip(prepared_code_values, prepared_label_values, strict=True)
    )
    prepared_da_mapping = (
        dict(zip(prepared_code_values, prepared_da_values, strict=True))
        if official_labels_da is not None
        else {}
    )
    prepared_counts = dict(
        zip(
            prepared_code_values,
            prepared_frame.get_column("count").to_list(),
            strict=True,
        )
    )
    nonzero_approved = sorted(
        code
        for code in approved_zero_codes
        if code in prepared_counts and prepared_counts[code] != 0
    )
    unhandled_values = sorted(observed_codes - set(official_labels))
    missing_metadata = sorted(expected_codes - set(selected_metadata))
    missing_prepared = sorted(expected_codes - prepared_codes)
    extra_prepared = sorted(prepared_codes - expected_codes)
    mapping_mismatches = sorted(
        code
        for code in expected_codes | prepared_codes
        if prepared_mapping.get(code) != selected_metadata.get(code)
    )
    danish_mapping_mismatches = sorted(
        code
        for code in expected_codes | prepared_codes
        if official_labels_da is not None
        and prepared_da_mapping.get(code) != official_labels_da.get(code)
    )
    suppressed_cells = int(raw_frame.get_column("suppressed").sum())
    total = int(prepared_frame.get_column("count").sum())
    code_unique = prepared_frame.height == len(prepared_codes)
    prepared_label_unique = prepared_frame.height == len(set(prepared_label_values))
    metadata_label_unique = len(selected_metadata) == len(
        set(selected_metadata.values())
    )
    label_unique = prepared_label_unique and metadata_label_unique
    expected_partition = (
        not invalid_approved
        and not unexpected_missing
        and not expected_zero_not_missing
        and not nonzero_approved
        and not missing_prepared
        and not extra_prepared
    )
    metadata_mapping = (
        not missing_metadata
        and not mapping_mismatches
        and not danish_mapping_mismatches
        and metadata_label_unique
        and (
            official_labels_da is None or len(prepared_da_values) == len(prepared_codes)
        )
    )
    zero_suppression = suppressed_cells == 0
    passed = (
        total > 0
        and code_unique
        and label_unique
        and zero_suppression
        and not unhandled_values
        and expected_partition
        and metadata_mapping
    )
    return {
        "passed": passed,
        "positive_total": {"value": total, "passed": total > 0},
        "code_uniqueness": {
            "value": prepared_frame.height,
            "unique": len(prepared_codes),
            "passed": code_unique,
        },
        "label_uniqueness": {
            "value": prepared_frame.height,
            "prepared_unique": len(set(prepared_label_values)),
            "metadata_unique": len(set(selected_metadata.values())),
            "passed": label_unique,
        },
        "metadata_mapping": {
            "expected": selected_metadata,
            "prepared": prepared_mapping,
            "missing_metadata": missing_metadata,
            "mismatches": mapping_mismatches,
            "danish_mismatches": danish_mapping_mismatches,
            "passed": metadata_mapping,
        },
        "zero_suppression": {
            "suppressed_cells": suppressed_cells,
            "zero_count_cells": int((raw_frame.get_column("count") == 0).sum()),
            "passed": zero_suppression,
        },
        "unhandled_values": {
            "values": unhandled_values,
            "passed": not unhandled_values,
        },
        "expected_partition": {
            "expected_codes": len(expected_codes),
            "prepared_codes": len(prepared_codes),
            "missing_raw": missing_raw,
            "approved_zero_codes": sorted(approved_zero_codes),
            "invalid_approved": invalid_approved,
            "unexpected_missing": unexpected_missing,
            "expected_zero_not_missing": expected_zero_not_missing,
            "nonzero_approved": nonzero_approved,
            "missing_prepared": missing_prepared,
            "extra_prepared": extra_prepared,
            "passed": expected_partition,
        },
    }


def _read_source(csv_path: Path, dimension_codes: list[str]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, delimiter=";")
        header = next(reader)
        if len(header) != len(dimension_codes) + 1:
            message = f"Unexpected columns in {csv_path}: {header}"
            raise ValueError(message)
        for raw_row in reader:
            row: dict[str, object] = {}
            for code, value in zip(dimension_codes, raw_row[:-1], strict=True):
                row[code] = value.partition(" ")[0]
                row[f"{code}__label"] = value
            raw_count = raw_row[-1].strip()
            row["suppressed"] = raw_count in {"", "..", "."}
            if row["suppressed"]:
                row["count"] = 0
            else:
                try:
                    parsed_count = Decimal(raw_count)
                except InvalidOperation as error:
                    raise ValueError(
                        f"Malformed count in {csv_path}: {raw_count}"
                    ) from error
                if (
                    not parsed_count.is_finite()
                    or parsed_count != parsed_count.to_integral_value()
                ):
                    raise ValueError(f"Non-integral count in {csv_path}: {raw_count}")
                if parsed_count < 0:
                    raise ValueError(f"Negative count in {csv_path}: {raw_count}")
                row["count"] = int(parsed_count)
            rows.append(row)
    return pl.DataFrame(rows)


def _repository_relative_path(path: Path) -> str:
    """Return a strict repository-relative POSIX path for a contract.

    Raises:
        ValueError: If the path is outside the repository or is an alias.
    """
    root = Path.cwd().resolve()
    if ".." in path.parts or (
        not path.is_absolute() and path.as_posix() != Path(*path.parts).as_posix()
    ):
        raise ValueError("Origin-label contract path is not canonical")
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if candidate.absolute() != resolved:
        raise ValueError("Origin-label contract path is an alias")
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "Origin-label contract must be inside the repository"
        ) from error
    relative_path = relative.as_posix()
    if relative_path != Path(relative_path).as_posix() or ".." in relative.parts:
        raise ValueError("Origin-label contract path is not canonical")
    return relative_path


def _source_metrics(
    frames: dict[str, pl.DataFrame],
    geography_metrics: dict[str, object],
    origin_metrics: dict[str, object],
    lons20_contract_version: int,
    lons20_contract_sha256: str,
    origin_labels_contract: OriginLabelContract,
    origin_labels_contract_path: str,
    origin_labels_contract_sha256: str,
    origin_metadata_en_sha256: str,
    origin_metadata_da_sha256: str,
) -> dict[str, object]:
    """Summarise prepared tables and the geography cross-check.

    Args:
        frames:
            Normalised tables keyed by output name.
        geography_metrics:
            Result of the geography hierarchy cross-check.
        origin_metrics:
            Checks specific to the FOLK2 origin marginal.
        lons20_contract_version:
            Version of the canonical LONS20 contract.
        lons20_contract_sha256:
            SHA-256 checksum of the canonical LONS20 contract.
        origin_labels_contract:
            Validated Danish FOLK2 label contract.
        origin_labels_contract_path:
            Repository-relative contract path.
        origin_labels_contract_sha256:
            Contract byte checksum.
        origin_metadata_en_sha256:
            Archived English metadata byte checksum.
        origin_metadata_da_sha256:
            Archived Danish metadata byte checksum.

    Returns:
        Report payload whose ``passed`` flag gates the prepared bundle.
    """
    table_metrics: dict[str, object] = {}
    passed = True
    for name, frame in frames.items():
        suppressed = (
            int(frame.get_column("suppressed").sum())
            if "suppressed" in frame.columns
            else 0
        )
        total = int(frame.get_column("count").sum())
        valid = frame.height > 0 and total > 0 and suppressed == 0
        passed = passed and valid
        table_metrics[name] = {
            "rows": frame.height,
            "population_total": total,
            "suppressed_cells": suppressed,
            "passed": valid,
        }
    job_function = frames["job_function_sex_marginal"]
    job_function_checks = {
        "two_digit_partition": {
            "passed": set(job_function.get_column("job_function_code"))
            == set(DISCO_TWO_DIGIT_CODES),
            "codes": job_function.get_column("job_function_code").n_unique(),
        },
        "sex_coverage": {
            "passed": job_function.group_by("job_function_code")
            .len()
            .filter(pl.col("len") != 2)
            .is_empty(),
            "sexes": job_function.get_column("sex").n_unique(),
        },
        "positive_counts": {
            "passed": bool((job_function.get_column("count") > 0).all()),
            "total": int(job_function.get_column("count").sum()),
        },
    }
    passed = passed and bool(geography_metrics["passed"])
    passed = passed and bool(origin_metrics["passed"])
    passed = passed and all(
        bool(check["passed"]) for check in job_function_checks.values()
    )
    return {
        "passed": passed,
        "lons20_contract": {
            "version": lons20_contract_version,
            "sha256": lons20_contract_sha256,
        },
        "origin_labels_contract": {
            "path": origin_labels_contract_path,
            "version": origin_labels_contract.version,
            "sha256": origin_labels_contract_sha256,
            "source_metadata_en_sha256": (
                origin_labels_contract.source_metadata_en_sha256
            ),
            "source_metadata_da_sha256": (
                origin_labels_contract.source_metadata_da_sha256
            ),
            "metadata_en_sha256": origin_metadata_en_sha256,
            "metadata_da_sha256": origin_metadata_da_sha256,
        },
        "tables": table_metrics,
        "geography_hierarchy": geography_metrics,
        "origin_country_checks": origin_metrics,
        "job_function_checks": job_function_checks,
    }


def _source_report_markdown(bundle_id: str, metrics: dict[str, object]) -> str:
    tables = metrics["tables"]
    if not isinstance(tables, dict):
        message = "Source metric tables must be a mapping"
        raise TypeError(message)
    lines = [
        "# Source preparation report",
        "",
        f"Bundle: `{bundle_id}`",
        "",
        f"Overall result: **{'PASS' if metrics['passed'] else 'FAIL'}**",
        "",
        "| Prepared table | Rows | Population total | Suppressed | Result |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, raw_metric in tables.items():
        if not isinstance(raw_metric, dict):
            message = f"Invalid source metric for {name}"
            raise TypeError(message)
        result = "PASS" if raw_metric["passed"] else "FAIL"
        lines.append(
            f"| {name} | {raw_metric['rows']:,} | "
            f"{raw_metric['population_total']:,} | "
            f"{raw_metric['suppressed_cells']:,} | {result} |"
        )
    geography = metrics["geography_hierarchy"]
    if not isinstance(geography, dict):
        message = "Geography metrics must be a mapping"
        raise TypeError(message)
    origin = metrics["origin_country_checks"]
    if not isinstance(origin, dict):
        message = "Origin-country metrics must be a mapping"
        raise TypeError(message)
    origin_results = {
        name: _result_text(result["passed"])
        for name, result in origin.items()
        if isinstance(result, dict) and "passed" in result
    }
    contract = metrics["lons20_contract"]
    if not isinstance(contract, dict):
        message = "LONS20 contract metrics must be a mapping"
        raise TypeError(message)
    lines.extend(
        [
            "",
            "## LONS20 canonical contract",
            "",
            f"- Version: **{contract.get('version')}**",
            f"- SHA-256: `{contract.get('sha256')}`",
            "",
            "## FOLK2 origin-country marginal",
            "",
            "This is an independent national marginal, not ethnicity or citizenship.",
            f"- Official IELAND categories: "
            f"{origin['expected_partition']['expected_codes']:,}",
            f"- Positive total: **{origin_results['positive_total']}**",
            f"- Code uniqueness: **{origin_results['code_uniqueness']}**",
            f"- Label uniqueness: **{origin_results['label_uniqueness']}**",
            "- Official code-to-label mapping: "
            f"**{origin_results['metadata_mapping']}**",
            f"- Zero suppression: **{origin_results['zero_suppression']}**",
            f"- Unhandled values: **{origin_results['unhandled_values']}**",
            f"- Expected partition: **{origin_results['expected_partition']}**",
            "",
            "## Geography hierarchy",
            "",
            "Sourced from the official Statistics Denmark classification and "
            "cross-checked against StatBank table metadata.",
            "",
            f"- Municipalities: {geography['municipalities']:,}",
            f"- Landsdele: {geography['landsdele']:,}",
            f"- Regions: {geography['regions']:,}",
            f"- Disagreements with StatBank: "
            f"{len(geography['statbank_disagreements'])}",
            f"- Missing from classification: "
            f"{len(geography['missing_from_classification'])}",
            f"- Result: **{'PASS' if geography['passed'] else 'FAIL'}**",
        ]
    )
    return "\n".join(lines) + "\n"


def _result_text(passed: object) -> str:
    return "PASS" if passed else "FAIL"


def _validate_lons20_source(
    lock: SourceLock, contract_expectations: SourceMetadataExpectations | None = None
) -> LockedSource:
    if contract_expectations is None:
        contract_expectations = lons20_expectations(contract=load_lons20_contract())
    matches = [source for source in lock.sources if source.table_id == "LONS20"]
    if len(matches) != 1:
        raise ValueError("Source lock must contain exactly one LONS20 source")
    source = matches[0]
    if source.role != "job_function_sex_marginal" or source.period != "2024":
        raise ValueError("LONS20 must be the 2024 job-function sex marginal")
    if source.dimensions != LONS20_DIMENSIONS:
        raise ValueError(
            "LONS20 must select only the fixed 42 two-digit DISCO-08 groups and "
            "approved coverage dimensions"
        )
    expectations = source.metadata_expectations
    if expectations is None:
        raise ValueError("LONS20 lock must include versioned metadata expectations")
    if source.unit != expectations.unit:
        raise ValueError("LONS20 lock unit does not match metadata expectations")
    if set(expectations.dimensions) != set(LONS20_DIMENSIONS):
        raise ValueError("LONS20 metadata expectations do not cover every dimension")
    if any(
        set(expectations.values.get(dimension, {})) != set(values)
        for dimension, values in LONS20_DIMENSIONS.items()
    ):
        raise ValueError("LONS20 metadata expectations do not cover selected values")
    if contract_expectations is not None and expectations != contract_expectations:
        raise ValueError("LONS20 lock metadata does not match canonical contract")
    return source


def _validate_origin_metadata(
    *,
    lock_codes: list[str],
    english_metadata: StatBankMetadata,
    danish_metadata: StatBankMetadata,
    contract: OriginLabelContract,
    metadata_en_sha256: str,
    metadata_da_sha256: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Validate the four-way FOLK2 IELAND partition and return both labels.

    Returns:
        English and Danish code-to-label mappings.

    Raises:
        ValueError: If the four code sets or Danish metadata binding differ.
    """
    if (
        len(lock_codes) != ORIGIN_LABEL_COUNT
        or len(set(lock_codes)) != ORIGIN_LABEL_COUNT
    ):
        raise ValueError("FOLK2 lock must contain exactly 241 unique IELAND codes")
    english_labels, danish_labels = bind_origin_triples(
        contract=contract,
        english_metadata=english_metadata,
        danish_metadata=danish_metadata,
        english_metadata_sha256=metadata_en_sha256,
        danish_metadata_sha256=metadata_da_sha256,
    )
    code_sets = {
        "lock": set(lock_codes),
        "English metadata": set(english_labels),
        "Danish metadata": set(danish_labels),
        "origin-label contract English": set(contract.labels_en),
        "origin-label contract Danish": set(contract.labels_da),
    }
    if any(len(codes) != ORIGIN_LABEL_COUNT for codes in code_sets.values()):
        raise ValueError("FOLK2 IELAND metadata and contract must contain 241 codes")
    if len({frozenset(codes) for codes in code_sets.values()}) != 1:
        raise ValueError("FOLK2 IELAND code sets differ across lock and metadata")
    if english_labels != contract.labels_en or danish_labels != contract.labels_da:
        raise ValueError("FOLK2 metadata labels differ from the reviewed triples")
    return english_labels, danish_labels


def _verify_existing_bundle(bundle_dir: Path, manifest_path: Path) -> None:
    del manifest_path
    verify_prepared_bundle(bundle_dir=bundle_dir)


def _verify_ras209_municipality_sets(
    *, locked_codes: set[str], geography: pl.DataFrame, prepared: pl.DataFrame
) -> None:
    """Require one exact RAS209 municipality universe through preparation.

    Args:
        locked_codes:
            Municipality codes selected in the immutable RAS209 lock.
        geography:
            Official hierarchy municipality lookup.
        prepared:
            Prepared municipality-native RAS209 joint.

    Raises:
        ValueError:
            If the locked, hierarchy, and prepared municipality sets differ.
    """
    hierarchy_codes = set(geography.get_column("municipality_code").to_list())
    prepared_codes = set(prepared.get_column("municipality_code").to_list())
    if locked_codes == hierarchy_codes == prepared_codes:
        return
    details = {
        "locked_not_hierarchy": sorted(locked_codes - hierarchy_codes),
        "hierarchy_not_locked": sorted(hierarchy_codes - locked_codes),
        "locked_not_prepared": sorted(locked_codes - prepared_codes),
        "prepared_not_locked": sorted(prepared_codes - locked_codes),
    }
    raise ValueError(f"RAS209 municipality sets differ: {details}")


def read_geography_classification(csv_path: Path) -> pl.DataFrame:
    """Read the official region, landsdel, and municipality classification.

    The attachment is a semicolon-delimited hierarchical listing ordered by
    ``SEKVENS``, where ``NIVEAU`` gives the level of each row. Notes embed
    newlines inside quoted fields, so it must be parsed as CSV rather than
    split by line.

    Args:
        csv_path:
            Classification attachment.

    Returns:
        One row per municipality with its landsdel and region.

    Raises:
        ValueError:
            If a row appears before its parent level.
    """
    rows: list[dict[str, str]] = []
    region: tuple[str, str] | None = None
    landsdel: tuple[str, str] | None = None
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        for record in csv.DictReader(file, delimiter=";"):
            code = (record["KODE"] or "").strip()
            title = (record["TITEL"] or "").strip()
            level = (record["NIVEAU"] or "").strip()
            if level not in {REGION_LEVEL, LANDSDEL_LEVEL, MUNICIPALITY_LEVEL}:
                continue
            if not code or not title:
                message = (
                    "Geography hierarchy contains a blank code or title at "
                    f"level {level or '<blank>'}"
                )
                raise ValueError(message)
            if level == REGION_LEVEL:
                region = (code, title)
                landsdel = None
            elif level == LANDSDEL_LEVEL:
                landsdel = (code, title)
            elif level == MUNICIPALITY_LEVEL:
                if region is None or landsdel is None:
                    message = f"Municipality {code} appears before its parents"
                    raise ValueError(message)
                rows.append(
                    {
                        "municipality_code": code,
                        "municipality": title,
                        "landsdel_code": landsdel[0],
                        "landsdel": landsdel[1],
                        "region_code": region[0],
                        "region": region[1],
                    }
                )
    if not rows:
        message = f"No municipalities found in {csv_path}"
        raise ValueError(message)
    frame = pl.DataFrame(rows)
    duplicate_codes = sorted(
        frame.filter(pl.col("municipality_code").is_duplicated())
        .get_column("municipality_code")
        .unique()
        .to_list()
    )
    if duplicate_codes:
        message = f"Duplicate municipality codes in {csv_path}: {duplicate_codes}"
        raise ValueError(message)
    return frame


def verify_raw_snapshot(
    snapshot_dir: Path,
    snapshot: SnapshotManifest,
    table_id: str,
    role: str,
    period: str,
    expected_query: str,
) -> None:
    """Verify raw files against both their manifest and locked query.

    Args:
        snapshot_dir:
            Content-addressed raw snapshot directory.
        snapshot:
            Parsed raw snapshot manifest.
        table_id:
            Locked table identifier.
        role:
            Locked pipeline role.
        period:
            Locked reference period.
        expected_query:
            Canonical query derived from the source lock.

    Raises:
        ValueError:
            If provenance, a checksum, or query content differs from the lock.
    """
    if (snapshot.table_id, snapshot.role, snapshot.period) != (table_id, role, period):
        message = f"Raw snapshot provenance mismatch: {snapshot_dir}"
        raise ValueError(message)
    expected = {
        "metadata-en.json": snapshot.metadata_sha256,
        "metadata-da.json": snapshot.metadata_da_sha256,
        "query.json": snapshot.query_sha256,
        "data.csv": snapshot.data_sha256,
        "response-headers.json": snapshot.response_headers_sha256,
    }
    verify_checksums(
        base_dir=snapshot_dir,
        expected=expected,
        message="Raw snapshot checksum mismatch",
    )
    query_path = snapshot_dir / "query.json"
    if snapshot.query_sha256 != sha256_text(expected_query):
        message = f"Raw snapshot query does not match source lock: {query_path}"
        raise ValueError(message)
    if query_path.read_text(encoding="utf-8") != expected_query:
        message = f"Raw snapshot query content is not canonical: {query_path}"
        raise ValueError(message)
