"""Mandatory validation gates and reports."""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import ValidationError

from ..io import canonical_json, load_yaml_model, sha256_file, write_json
from ..ladders import MOST_SPECIFIC_RESOLUTION
from ..models import (
    SAMPLER_SCHEMA_VERSION,
    BundleManifest,
    CategoryConfig,
    DemographicRecord,
    MetricResult,
    RunManifest,
    ValidationConfig,
    ValidationReport,
)

LOGGER = logging.getLogger(__name__)
TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)


def validate_demographics(
    run_dir: Path, bundle_dir: Path, validation_config_path: Path, categories_path: Path
) -> ValidationReport:
    """Validate generated demographic and OCEAN records.

    Args:
        run_dir:
            Generated run directory.
        bundle_dir:
            Prepared source bundle used by the run.
        validation_config_path:
            Validation thresholds.
        categories_path:
            Canonical mappings.

    Returns:
        Validation report.
    """
    config = load_yaml_model(path=validation_config_path, model=ValidationConfig)
    categories = load_yaml_model(path=categories_path, model=CategoryConfig)
    manifest_path = run_dir / "run-manifest.json"
    manifest = RunManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    data_path = run_dir / manifest.data_file
    frame = pl.read_parquet(data_path)
    metrics = _provenance_metrics(
        frame=frame, manifest=manifest, data_path=data_path, bundle_dir=bundle_dir
    )
    metrics.extend(
        _structural_metrics(
            frame=frame,
            manifest=manifest,
            categories=categories,
            maximum_backoff_rate=config.maximum_backoff_rate,
        )
    )
    metrics.extend(
        _distribution_metrics(frame=frame, bundle_dir=bundle_dir, config=config)
    )
    metrics.extend(_heldout_metrics(frame=frame, bundle_dir=bundle_dir, config=config))
    metrics.extend(_ocean_metrics(frame=frame, config=config))
    metrics.extend(_origin_mapping_metrics(frame=frame, bundle_dir=bundle_dir))
    metrics.append(
        MetricResult(
            name="llm_calls",
            passed=manifest.llm_calls == 0,
            value=manifest.llm_calls,
            threshold=0,
            details="Phase 2 must not call an LLM.",
        )
    )
    report = ValidationReport(
        kind="demographics",
        passed=all(metric.passed for metric in metrics),
        created_at=_now(),
        subject_id=manifest.run_id,
        metrics=metrics,
    )
    _write_reports(directory=run_dir, report=report)
    return report


def _distribution_metrics(
    frame: pl.DataFrame, bundle_dir: Path, config: ValidationConfig
) -> list[MetricResult]:
    source_dir = bundle_dir / "normalized"
    folk = pl.read_parquet(source_dir / "folk1a_base_unpooled.parquet").with_columns(
        _age_band_expression().alias("age_band")
    )
    ras209 = pl.read_parquet(source_dir / "ras209_sampling.parquet")
    targets = {
        "sex": (folk, ["sex"]),
        "region_code": (folk, ["region_code"]),
        "marital_status": (folk, ["marital_status"]),
        "age_band": (folk, ["age_band"]),
        "education_level": (ras209, ["education_level"]),
        "labour_market_status": (ras209, ["labour_market_status"]),
        "origin_country": (
            pl.read_parquet(source_dir / "folk2_origin_country_marginal.parquet"),
            ["origin_country_code", "origin_country"],
        ),
    }
    metrics: list[MetricResult] = []
    for name in config.mandatory_marginals:
        target_frame, columns = targets[name]
        metrics.extend(
            _compare_distribution(
                name=name,
                generated=frame,
                target=target_frame,
                columns=columns,
                config=config,
                maximum_tv=config.maximum_total_variation["fitted_marginal"],
                smoke_maximum_tv=config.smoke_maximum_total_variation,
            )
        )
    metrics.extend(
        _compare_distribution(
            name="ras209_fitted_joint",
            generated=frame,
            target=ras209,
            columns=[
                "region_code",
                "age_band",
                "sex",
                "education_level",
                "labour_market_status",
            ],
            config=config,
            maximum_tv=config.maximum_total_variation["fitted_marginal"],
            smoke_maximum_tv=config.smoke_maximum_total_variation,
        )
    )
    return metrics


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


def _compare_distribution(
    name: str,
    generated: pl.DataFrame,
    target: pl.DataFrame,
    columns: list[str],
    config: ValidationConfig,
    maximum_tv: float,
    smoke_maximum_tv: float,
) -> list[MetricResult]:
    generated_counts = generated.group_by(columns).len().rename({"len": "observed"})
    target_counts = target.group_by(columns).agg(pl.col("count").sum().alias("target"))
    comparison = target_counts.join(
        generated_counts, on=columns, how="left"
    ).with_columns(pl.col("observed").fill_null(0))
    target_total = float(comparison.get_column("target").sum())
    observed_total = float(comparison.get_column("observed").sum())
    comparison = comparison.with_columns(
        (pl.col("target") / target_total).alias("target_p"),
        (pl.col("observed") / observed_total).alias("observed_p"),
    ).with_columns(
        (pl.col("target_p") * observed_total).alias("expected"),
        (pl.col("observed_p") - pl.col("target_p")).abs().alias("error"),
        pl.max_horizontal(
            pl.lit(config.absolute_proportion_tolerance),
            config.standard_error_multiplier
            * (pl.col("target_p") * (1.0 - pl.col("target_p")) / observed_total).sqrt(),
        ).alias("tolerance"),
    )
    eligible = comparison.filter(pl.col("expected") >= config.minimum_expected_count)
    violations = eligible.filter(pl.col("error") > pl.col("tolerance")).height
    total_variation = float(comparison.get_column("error").sum()) / 2.0
    maximum_tv = maximum_tv if observed_total >= 100_000 else smoke_maximum_tv
    return [
        MetricResult(
            name=f"{name}_cell_tolerance",
            passed=violations == 0,
            value=violations,
            threshold=0,
            details=(
                f"Compared {eligible.height} cells with expected synthetic count >= "
                f"{config.minimum_expected_count:g}."
            ),
        ),
        MetricResult(
            name=f"{name}_total_variation",
            passed=total_variation <= maximum_tv,
            value=total_variation,
            threshold=maximum_tv,
            details="Total variation between generated and fitted marginal.",
        ),
    ]


def _heldout_metrics(
    frame: pl.DataFrame, bundle_dir: Path, config: ValidationConfig
) -> list[MetricResult]:
    source_dir = bundle_dir / "normalized"
    befolk = pl.read_parquet(source_dir / "befolk3_holdout.parquet").with_columns(
        _age_band_expression().alias("age_band")
    )
    generated_status = frame.with_columns(
        pl.when(
            pl.col("detailed_status_code").is_in(
                ["05", "10", "15", "20", "25", "30", "35", "40"]
            )
        )
        .then(pl.lit("00"))
        .when(pl.col("detailed_status_code") == "50")
        .then(pl.lit("50"))
        .otherwise(pl.lit("55"))
        .alias("status_group_code")
    )
    ras210 = pl.read_parquet(source_dir / "ras210_holdout.parquet")
    maximum_tv = config.maximum_total_variation["heldout_marginal"]
    population = _compare_distribution(
        name="heldout_population_joint",
        generated=frame,
        target=befolk,
        columns=["region_code", "sex", "age_band"],
        config=config,
        maximum_tv=maximum_tv,
        smoke_maximum_tv=config.smoke_holdout_maximum_total_variation,
    )
    status = _compare_distribution(
        name="heldout_status_sex",
        generated=generated_status,
        target=ras210,
        columns=["status_group_code", "sex"],
        config=config,
        maximum_tv=maximum_tv,
        smoke_maximum_tv=config.smoke_holdout_maximum_total_variation,
    )
    return [population[-1], status[-1]]


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _ocean_metrics(frame: pl.DataFrame, config: ValidationConfig) -> list[MetricResult]:
    score_columns = [f"{trait}_score" for trait in TRAITS]
    scores = frame.select(score_columns).to_numpy()
    bounds_errors = int(np.count_nonzero((scores < 20.0) | (scores > 80.0)))
    correlation = np.corrcoef(scores, rowvar=False)
    upper = np.abs(correlation[np.triu_indices_from(correlation, k=1)])
    maximum_correlation = float(upper.max())
    threshold = config.maximum_ocean_pairwise_correlation
    correlation_required = frame.height >= 100_000
    return [
        MetricResult(
            name="ocean_score_bounds",
            passed=bounds_errors == 0,
            value=bounds_errors,
            threshold=0,
            details="All OCEAN T-scores are clipped to [20, 80].",
        ),
        MetricResult(
            name="ocean_max_pairwise_correlation",
            passed=(not correlation_required) or maximum_correlation <= threshold,
            value=maximum_correlation,
            threshold=threshold if correlation_required else "informational below 100k",
            details="OCEAN traits are sampled independently of one another.",
        ),
    ]


def _origin_mapping_metrics(
    frame: pl.DataFrame, bundle_dir: Path
) -> list[MetricResult]:
    """Check that generated origin labels exactly match official FOLK2 codes.

    Returns:
        Mapping and positive-weight validation metrics.
    """
    target = pl.read_parquet(
        bundle_dir / "normalized" / "folk2_origin_country_marginal.parquet"
    )
    expected = dict(
        zip(
            target.get_column("origin_country_code").to_list(),
            target.get_column("origin_country").to_list(),
            strict=True,
        )
    )
    observed = {
        row["origin_country_code"]: row["origin_country"]
        for row in frame.select("origin_country_code", "origin_country")
        .unique()
        .iter_rows(named=True)
    }
    mismatches = sum(expected.get(code) != label for code, label in observed.items())
    zero_weight_codes = set(
        target.filter(pl.col("count") <= 0).get_column("origin_country_code").to_list()
    )
    emitted_zero_weight = len(zero_weight_codes & set(observed))
    return [
        MetricResult(
            name="origin_country_mapping",
            passed=mismatches == 0,
            value=mismatches,
            threshold=0,
            details="Generated origin labels must use the official code mapping.",
        ),
        MetricResult(
            name="origin_country_positive_weights",
            passed=emitted_zero_weight == 0,
            value=emitted_zero_weight,
            threshold=0,
            details="Categories with zero official weight must never be emitted.",
        ),
    ]


def _provenance_metrics(
    frame: pl.DataFrame, manifest: RunManifest, data_path: Path, bundle_dir: Path
) -> list[MetricResult]:
    bundle_manifest_path = bundle_dir / "bundle-manifest.json"
    bundle = BundleManifest.model_validate_json(
        bundle_manifest_path.read_text(encoding="utf-8")
    )
    checks = {
        "parquet_checksum": sha256_file(data_path) == manifest.data_sha256,
        "logical_content_checksum": (
            _logical_checksum(frame=frame) == manifest.logical_content_sha256
        ),
        "bundle_identity": bundle.bundle_id == manifest.bundle_id,
        "sampler_schema_version": (
            manifest.sampler_schema_version == SAMPLER_SCHEMA_VERSION
        ),
        "bundle_manifest_checksum": (
            sha256_file(bundle_manifest_path) == manifest.bundle_manifest_sha256
        ),
    }
    return [
        MetricResult(
            name=name,
            passed=passed,
            value="pass" if passed else "fail",
            threshold="pass",
            details="Run provenance and content must match the frozen manifests.",
        )
        for name, passed in checks.items()
    ]


def _logical_checksum(frame: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.iter_rows(named=True):
        digest.update(canonical_json(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _structural_metrics(
    frame: pl.DataFrame,
    manifest: RunManifest,
    categories: CategoryConfig,
    maximum_backoff_rate: float,
) -> list[MetricResult]:
    """Check row counts, identifiers, schema, and sampling provenance.

    Args:
        frame:
            Generated records.
        manifest:
            Manifest the run must agree with.
        categories:
            Canonical category mappings.
        maximum_backoff_rate:
            Largest share of records permitted to come from a coarser cell.

    Returns:
        One metric per structural check.
    """
    invalid_schema = 0
    for row in frame.iter_rows(named=True):
        try:
            DemographicRecord.model_validate(row)
        except ValidationError:
            invalid_schema += 1
    status_lookup = {
        code: status
        for status, codes in categories.labour_market_status.items()
        for code in codes
    }
    status_mismatches = sum(
        status_lookup.get(str(code)) != status
        for code, status in frame.select(
            "detailed_status_code", "labour_market_status"
        ).iter_rows()
    )
    proxy_mismatches = frame.filter(
        (
            (pl.col("age") >= 70)
            & (pl.col("education_resolution") != "ras209_67_plus_proxy")
        )
        | ((pl.col("age") < 70) & (pl.col("education_resolution") != "ras209_age_band"))
    ).height
    unique_ids = frame.get_column("persona_id").n_unique()
    return [
        *_backoff_metrics(frame=frame, maximum_rate=maximum_backoff_rate),
        MetricResult(
            name="row_count",
            passed=frame.height == manifest.rows,
            value=frame.height,
            threshold=manifest.rows,
            details="Parquet row count matches the run manifest.",
        ),
        MetricResult(
            name="unique_ids",
            passed=unique_ids == frame.height,
            value=unique_ids,
            threshold=frame.height,
            details="Every synthetic record has a unique deterministic ID.",
        ),
        MetricResult(
            name="schema_errors",
            passed=invalid_schema == 0,
            value=invalid_schema,
            threshold=0,
            details="Every row validates against the Phase 2 typed schema.",
        ),
        MetricResult(
            name="status_mapping_errors",
            passed=status_mismatches == 0,
            value=status_mismatches,
            threshold=0,
            details="Detailed statuses remain inside their sampled broad status.",
        ),
        MetricResult(
            name="education_proxy_errors",
            passed=proxy_mismatches == 0,
            value=proxy_mismatches,
            threshold=0,
            details="The disclosed RAS209 67+ proxy is labelled for ages 70+ only.",
        ),
    ]


def _backoff_metrics(frame: pl.DataFrame, maximum_rate: float) -> list[MetricResult]:
    """Report how often each ladder fell back to a coarser cell.

    The ladders are independent and of different depths, so each reports its
    own rate; a single combined figure could not say which draw is sparse.

    Args:
        frame:
            Generated records.
        maximum_rate:
            Largest share of records permitted to come from a coarser cell.

    Returns:
        One metric per resolution column.
    """
    metrics: list[MetricResult] = []
    for column, most_specific in MOST_SPECIFIC_RESOLUTION.items():
        counts = frame.get_column(column).value_counts().sort(column)
        backed_off = int(
            counts.filter(pl.col(column) != most_specific).get_column("count").sum()
        )
        rate = backed_off / frame.height if frame.height else 0.0
        breakdown = ", ".join(
            f"{row[column]}: {row['count'] / frame.height:.4%}"
            for row in counts.iter_rows(named=True)
        )
        metrics.append(
            MetricResult(
                name=f"{column}_backoff_rate",
                passed=rate <= maximum_rate,
                value=rate,
                threshold=maximum_rate,
                details=f"Levels used: {breakdown}.",
            )
        )
    return metrics


def _write_reports(directory: Path, report: ValidationReport) -> None:
    json_path = directory / "validation-report.json"
    write_json(path=json_path, payload=report)
    lines = [
        f"# {report.kind.title()} validation report",
        "",
        f"Subject: `{report.subject_id}`",
        "",
        f"Overall result: **{'PASS' if report.passed else 'FAIL'}**",
        "",
        "| Check | Value | Threshold | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    lines.extend(
        f"| {metric.name} | {metric.value} | {metric.threshold} | "
        f"{'PASS' if metric.passed else 'FAIL'} |"
        for metric in report.metrics
    )
    (directory / "validation-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "%s validation %s for %s",
        report.kind,
        "passed" if report.passed else "failed",
        report.subject_id,
    )


def validate_sources(bundle_dir: Path) -> ValidationReport:
    """Validate a prepared source bundle and write reports.

    Args:
        bundle_dir:
            Prepared source bundle.

    Returns:
        Validation report.
    """
    manifest_path = bundle_dir / "bundle-manifest.json"
    manifest = BundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    checksum_failures = [
        relative_path
        for relative_path, checksum in manifest.files.items()
        if not (bundle_dir / relative_path).exists()
        or sha256_file(bundle_dir / relative_path) != checksum
    ]
    source_report_path = bundle_dir / "source-preparation-report.json"
    source_payload = json.loads(source_report_path.read_text(encoding="utf-8"))
    source_passed = source_payload.get("passed") is True
    metrics = [
        MetricResult(
            name="prepared_file_checksums",
            passed=not checksum_failures,
            value=len(checksum_failures),
            threshold=0,
            details=(
                "All prepared files match the bundle manifest."
                if not checksum_failures
                else f"Mismatches: {', '.join(checksum_failures)}"
            ),
        ),
        MetricResult(
            name="source_preparation",
            passed=source_passed,
            value="pass" if source_passed else "fail",
            threshold="pass",
            details="All selected source tables are populated and unsuppressed.",
        ),
    ]
    origin_checks = source_payload.get("origin_country_checks")
    if isinstance(origin_checks, dict):
        for check_name in (
            "positive_total",
            "code_uniqueness",
            "label_uniqueness",
            "metadata_mapping",
            "zero_suppression",
            "unhandled_values",
            "expected_partition",
        ):
            check = origin_checks.get(check_name)
            if isinstance(check, dict) and isinstance(check.get("passed"), bool):
                metrics.append(
                    MetricResult(
                        name=f"folk2_{check_name}",
                        passed=check["passed"],
                        value="pass" if check["passed"] else "fail",
                        threshold="pass",
                        details=f"FOLK2 {check_name.replace('_', ' ')} check.",
                    )
                )
    report = ValidationReport(
        kind="sources",
        passed=all(metric.passed for metric in metrics),
        created_at=_now(),
        subject_id=manifest.bundle_id,
        metrics=metrics,
    )
    _write_reports(directory=bundle_dir, report=report)
    return report
