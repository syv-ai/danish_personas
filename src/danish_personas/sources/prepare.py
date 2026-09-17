"""Normalise raw official aggregates into an offline sampling bundle."""

import csv
import logging
import math
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ..io import load_yaml_model, sha256_file, sha256_text, verify_checksums, write_json
from ..models import (
    BundleManifest,
    CategoryConfig,
    ClassificationManifest,
    SnapshotManifest,
    SourceLock,
    StatBankMetadata,
)
from .classification import classification_snapshot_dir, verify_classification_snapshot
from .statbank import source_query_content, source_snapshot_dir

LOGGER = logging.getLogger(__name__)
REGION_PREFIX = "Region "
REGION_LEVEL = "1"
LANDSDEL_LEVEL = "2"
MUNICIPALITY_LEVEL = "3"


def prepare_bundle(
    lock_path: Path, categories_path: Path, raw_dir: Path, output_dir: Path
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

    Returns:
        Prepared bundle directory.

    Raises:
        ValueError:
            If the lock has no geography classification.
    """
    lock = load_yaml_model(path=lock_path, model=SourceLock)
    categories = load_yaml_model(path=categories_path, model=CategoryConfig)
    bundle_id = sha256_text(f"{sha256_file(lock_path)}:{sha256_file(categories_path)}")[
        :16
    ]
    bundle_dir = output_dir / bundle_id
    manifest_path = bundle_dir / "bundle-manifest.json"
    if manifest_path.exists():
        _verify_existing_bundle(bundle_dir=bundle_dir, manifest_path=manifest_path)
        LOGGER.info("Reusing prepared bundle %s", bundle_id)
        return bundle_dir

    normalized_dir = bundle_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    snapshots: list[SnapshotManifest] = []
    source_frames: dict[str, pl.DataFrame] = {}
    metadata_by_table: dict[str, StatBankMetadata] = {}
    for source in lock.sources:
        snapshot_dir = source_snapshot_dir(source=source, raw_dir=raw_dir)
        snapshot = SnapshotManifest.model_validate_json(
            (snapshot_dir / "snapshot-manifest.json").read_text()
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
            (snapshot_dir / "metadata-en.json").read_text()
        )
        metadata_by_table[source.table_id] = metadata
        source_frames[source.table_id] = _read_source(
            csv_path=snapshot_dir / "data.csv", dimension_codes=list(source.dimensions)
        )

    classification_snapshots: list[ClassificationManifest] = []
    geography_csv_path: Path | None = None
    for classification in lock.classifications:
        classification_dir = classification_snapshot_dir(
            classification=classification, raw_dir=raw_dir
        )
        classification_snapshot = ClassificationManifest.model_validate_json(
            (classification_dir / "snapshot-manifest.json").read_text()
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
    region_map = _region_map_from_geography(geography=geography)
    frames = _normalise_frames(
        raw_frames=source_frames,
        metadata_by_table=metadata_by_table,
        categories=categories,
        region_map=region_map,
        release_rows=lock.release_rows,
        minimum_source_count=lock.minimum_source_count,
        minimum_expected_release_count=lock.minimum_expected_release_count,
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
    source_metrics = _source_metrics(frames=frames, geography_metrics=geography_metrics)
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
        created_at=_now(),
        source_lock_sha256=sha256_file(lock_path),
        categories_sha256=sha256_file(categories_path),
        source_snapshots=snapshots,
        classification_snapshots=classification_snapshots,
        files=files,
        reference_periods={source.role: source.period for source in lock.sources},
        assumptions=[
            "FOLK1A 2025Q1 is the demographic base nearest RAS November 2024.",
            "FOLK1A ages 16-19 estimate the adult share of RAS209's 16-19 band.",
            "RAS209 jointly supplies broad education and labour-market status.",
            "RAS202 refines detailed status only within the RAS209 broad status.",
            "The RAS209 67+ education band is a proxy for ages 70 and over.",
            "BEFOLK3 and RAS210 are held-out diagnostics, not fitted microdata.",
            "The municipality-region hierarchy comes from the official DST "
            "classification, not from StatBank metadata ordering.",
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


def _geography_metrics(
    geography: pl.DataFrame, statbank_region_map: dict[str, tuple[str, str]]
) -> dict[str, object]:
    classification_codes = dict(
        zip(
            geography.get_column("municipality_code"),
            geography.get_column("region_code"),
            strict=True,
        )
    )
    statbank_codes = {code: region[0] for code, region in statbank_region_map.items()}
    disagreements = sorted(
        code
        for code in classification_codes.keys() & statbank_codes.keys()
        if classification_codes[code] != statbank_codes[code]
    )
    missing = sorted(statbank_codes.keys() - classification_codes.keys())
    complete = geography.null_count().sum_horizontal().item() == 0
    passed = not disagreements and not missing and complete
    return {
        "municipalities": geography.height,
        "regions": geography.get_column("region_code").n_unique(),
        "landsdele": geography.get_column("landsdel_code").n_unique(),
        "statbank_disagreements": disagreements,
        "missing_from_classification": missing,
        "complete": complete,
        "passed": passed,
    }


def _normalise_frames(
    raw_frames: dict[str, pl.DataFrame],
    metadata_by_table: dict[str, StatBankMetadata],
    categories: CategoryConfig,
    region_map: dict[str, tuple[str, str]],
    release_rows: int,
    minimum_source_count: int,
    minimum_expected_release_count: int,
) -> dict[str, pl.DataFrame]:
    labels = {
        table: _metadata_labels(metadata=metadata)
        for table, metadata in metadata_by_table.items()
    }
    status_map = _invert_status_mapping(categories=categories)

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
    folk_calibration = _add_geography(
        frame=folk_calibration,
        region_map=region_map,
        municipality_labels=labels["FOLK1A"]["OMRÅDE"],
    )
    folk = folk_calibration.filter(pl.col("age") >= 18).with_columns(
        _age_band_expression().alias("age_band")
    )
    folk_threshold = _release_threshold(
        total=float(folk.get_column("count").sum()),
        release_rows=release_rows,
        minimum_source_count=minimum_source_count,
        minimum_expected_release_count=minimum_expected_release_count,
    )
    folk_age_sampling = (
        folk.group_by(["age_band", "sex", "age"])
        .agg(pl.col("count").sum(), pl.col("suppressed").any())
        .filter(pl.col("count") >= folk_threshold)
    )
    folk_marital_sampling = (
        folk.group_by(["region_code", "region", "age_band", "sex", "marital_status"])
        .agg(pl.col("count").sum(), pl.col("suppressed").any())
        .filter(pl.col("count") >= folk_threshold)
    )

    ras209 = raw_frames["RAS209"].select(
        pl.col("OMRÅDE").alias("region_code"),
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
    ras209 = ras209.with_columns(
        pl.col("region_code")
        .replace_strict(_region_labels(metadata=metadata_by_table["RAS209"]))
        .alias("region")
    )
    ras209 = _adjust_young_adult_counts(
        ras209=ras209, folk_calibration=folk_calibration
    )
    ras209_joint = ras209.group_by(
        [
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
    befolk = _add_geography(
        frame=befolk,
        region_map=region_map,
        municipality_labels=labels["BEFOLK3"]["OMRÅDE"],
    )

    ras210 = raw_frames["RAS210"].select(
        pl.col("BOPKOM").alias("municipality_code"),
        pl.col("SOCIO").alias("status_group_code"),
        pl.col("ALERAMS").alias("age_key"),
        pl.col("KON").replace_strict(categories.sex).alias("sex"),
        pl.col("count"),
        pl.col("suppressed"),
    )
    ras210 = _add_geography(
        frame=ras210,
        region_map=region_map,
        municipality_labels=labels["RAS210"]["BOPKOM"],
    )
    return {
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


def _add_geography(
    frame: pl.DataFrame,
    region_map: dict[str, tuple[str, str]],
    municipality_labels: dict[str, str],
) -> pl.DataFrame:
    region_codes = {code: region[0] for code, region in region_map.items()}
    region_names = {code: region[1] for code, region in region_map.items()}
    return frame.with_columns(
        pl.col("municipality_code")
        .replace_strict(municipality_labels)
        .alias("municipality"),
        pl.col("municipality_code").replace_strict(region_codes).alias("region_code"),
        pl.col("municipality_code").replace_strict(region_names).alias("region"),
    )


def _adjust_young_adult_counts(
    ras209: pl.DataFrame, folk_calibration: pl.DataFrame
) -> pl.DataFrame:
    band = folk_calibration.filter(pl.col("age").is_between(16, 19))
    totals = band.group_by(["region_code", "sex"]).agg(
        pl.col("count").sum().alias("band_count")
    )
    adults = (
        band.filter(pl.col("age") >= 18)
        .group_by(["region_code", "sex"])
        .agg(pl.col("count").sum().alias("adult_count"))
    )
    factors = totals.join(adults, on=["region_code", "sex"]).with_columns(
        (pl.col("adult_count") / pl.col("band_count")).alias("adult_share")
    )
    return (
        ras209.join(
            factors.select("region_code", "sex", "adult_share"),
            on=["region_code", "sex"],
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


def _metadata_labels(metadata: StatBankMetadata) -> dict[str, dict[str, str]]:
    return {
        variable.id: {value.id: value.text for value in variable.values}
        for variable in metadata.variables
    }


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
    threshold = _release_threshold(
        total=total,
        release_rows=release_rows,
        minimum_source_count=minimum_source_count,
        minimum_expected_release_count=minimum_expected_release_count,
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


def _region_labels(metadata: StatBankMetadata) -> dict[str, str]:
    area = next(variable for variable in metadata.variables if variable.id == "OMRÅDE")
    return {
        value.id: value.text
        for value in area.values
        if value.text.startswith(REGION_PREFIX)
    }


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


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
            raw_count = raw_row[-1].strip()
            row["suppressed"] = raw_count in {"", "..", "."}
            row["count"] = 0 if row["suppressed"] else int(raw_count)
            rows.append(row)
    return pl.DataFrame(rows)


def _region_map_from_geography(geography: pl.DataFrame) -> dict[str, tuple[str, str]]:
    return {
        row["municipality_code"]: (row["region_code"], row["region"])
        for row in geography.iter_rows(named=True)
    }


def _source_metrics(
    frames: dict[str, pl.DataFrame], geography_metrics: dict[str, object]
) -> dict[str, object]:
    table_metrics: dict[str, object] = {}
    passed = True
    for name, frame in frames.items():
        suppressed = int(frame.get_column("suppressed").sum())
        total = int(frame.get_column("count").sum())
        valid = frame.height > 0 and total > 0 and suppressed == 0
        passed = passed and valid
        table_metrics[name] = {
            "rows": frame.height,
            "population_total": total,
            "suppressed_cells": suppressed,
            "passed": valid,
        }
    passed = passed and bool(geography_metrics["passed"])
    return {
        "passed": passed,
        "tables": table_metrics,
        "geography_hierarchy": geography_metrics,
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
    lines.extend(
        [
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


def _verify_existing_bundle(bundle_dir: Path, manifest_path: Path) -> None:
    manifest = BundleManifest.model_validate_json(manifest_path.read_text())
    verify_checksums(
        base_dir=bundle_dir,
        expected=manifest.files,
        message="Prepared bundle verification failed",
    )


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
            if not code:
                continue
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
    return pl.DataFrame(rows)


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
    if query_path.read_text() != expected_query:
        message = f"Raw snapshot query content is not canonical: {query_path}"
        raise ValueError(message)
