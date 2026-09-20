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
    ELIGIBLE_JOB_FUNCTION_STATUS_CODES,
    SAMPLER_SCHEMA_VERSION,
    BundleManifest,
    CategoryConfig,
    DemographicRecord,
    MetricResult,
    RunManifest,
    ValidationConfig,
    ValidationReport,
)
from ..origin_labels import load_origin_label_contract
from ..sources.bundle import (
    SOURCE_REPORT,
    _verify_prepared_bundle_capture,
    verify_prepared_bundle,
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
    bundle = verify_prepared_bundle(bundle_dir=bundle_dir)
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
    metrics.extend(_geography_parent_metrics(frame=frame, bundle_dir=bundle_dir))
    metrics.extend(_origin_mapping_metrics(frame=frame, bundle_dir=bundle_dir))
    metrics.extend(
        _job_function_metrics(frame=frame, bundle_dir=bundle_dir, config=config)
    )
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
        origin_labels_contract_path=bundle.origin_labels_contract_path,
        origin_labels_contract_version=bundle.origin_labels_contract_version,
        origin_labels_contract_sha256=bundle.origin_labels_contract_sha256,
        origin_labels_contract_content=bundle.origin_labels_contract_content,
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
        "sex": (ras209, ["sex"]),
        "municipality_code": (ras209, ["municipality_code"]),
        "marital_status": (folk, ["marital_status"]),
        "age_band": (ras209, ["age_band"]),
        "education_level": (ras209, ["education_level"]),
        "labour_market_status": (ras209, ["labour_market_status"]),
        "origin_country": (
            _origin_target(bundle_dir=bundle_dir),
            ["origin_country_code", "origin_country", "origin_country_da"],
        ),
    }
    metrics: list[MetricResult] = []
    for name in config.mandatory_marginals:
        if name == "job_function":
            continue
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
    joint_metrics = _compare_distribution(
        name="ras209_fitted_joint",
        generated=frame,
        target=ras209,
        columns=[
            "municipality_code",
            "age_band",
            "sex",
            "education_level",
            "labour_market_status",
        ],
        config=config,
        maximum_tv=config.maximum_municipality_joint_total_variation,
        # At 2,000 rows the municipality joint has more populated source cells than
        # observations. Its cell and TV gates become statistically meaningful at 100k.
        smoke_maximum_tv=1.0,
    )
    metrics.extend(joint_metrics if frame.height >= 100_000 else joint_metrics[1:])
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
    """Compare all expected and observed categories in one distribution.

    Args:
        name:
            Metric name prefix.
        generated:
            Generated records to count.
        target:
            Official target counts.
        columns:
            Category columns defining each cell.
        config:
            Validation thresholds.
        maximum_tv:
            Statistical-run total-variation threshold.
        smoke_maximum_tv:
            Smoke-run total-variation threshold.

    Returns:
        Cell-tolerance and total-variation metrics.
    """
    generated_counts = generated.group_by(columns).len().rename({"len": "observed"})
    target_counts = target.group_by(columns).agg(pl.col("count").sum().alias("target"))
    comparison = target_counts.join(
        generated_counts, on=columns, how="full", coalesce=True
    ).with_columns(pl.col("target").fill_null(0), pl.col("observed").fill_null(0))
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
    unexpected_rows = int(
        generated_counts.join(
            target_counts.select(columns), on=columns, how="anti", nulls_equal=True
        )
        .get_column("observed")
        .sum()
    )
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
            name=f"{name}_unexpected_categories",
            passed=unexpected_rows == 0,
            value=unexpected_rows,
            threshold=0,
            details="Generated rows must belong to categories in the fitted target.",
        ),
        MetricResult(
            name=f"{name}_total_variation",
            passed=total_variation <= maximum_tv,
            value=total_variation,
            threshold=maximum_tv,
            details="Total variation between generated and fitted marginal.",
        ),
    ]


def _origin_target(*, bundle_dir: Path) -> pl.DataFrame:
    """Return the prepared origin marginal with its complete label triple."""
    target = pl.read_parquet(
        bundle_dir / "normalized" / "folk2_origin_country_marginal.parquet"
    )
    if "origin_country_da" in target.columns:
        return target
    contract = load_origin_label_contract()
    return target.with_columns(
        pl.col("origin_country_code")
        .replace(dict(contract.ordered_labels), default=None)
        .alias("origin_country_da")
    )


def _geography_parent_metrics(
    frame: pl.DataFrame, bundle_dir: Path
) -> list[MetricResult]:
    """Check municipality labels and region parents against the hierarchy.

    Args:
        frame:
            Generated demographic records.
        bundle_dir:
            Verified prepared bundle.

    Returns:
        Official hierarchy consistency metrics.
    """
    columns = ["municipality_code", "municipality", "region_code", "region"]
    observed = frame.select(columns).unique()
    expected = pl.read_parquet(
        bundle_dir / "normalized" / "geography_hierarchy.parquet"
    ).select(columns)
    mismatches = observed.join(
        expected, on=columns, how="anti", nulls_equal=True
    ).height
    municipality_count = observed.get_column("municipality_code").n_unique()
    unique_mapping_count = observed.height
    return [
        MetricResult(
            name="municipality_hierarchy_mapping",
            passed=mismatches == 0,
            value=mismatches,
            threshold=0,
            details=(
                "Municipality names and region parents must match the official "
                "hierarchy."
            ),
        ),
        MetricResult(
            name="municipality_parent_consistency",
            passed=unique_mapping_count == municipality_count,
            value=unique_mapping_count - municipality_count,
            threshold=0,
            details="Each municipality must have exactly one name and region parent.",
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
        columns=["municipality_code", "sex", "age_band"],
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


def _job_function_metrics(
    frame: pl.DataFrame, bundle_dir: Path, config: ValidationConfig
) -> list[MetricResult]:
    """Check LONS20 mapping, eligibility, and sex-conditional allocation.

    Returns:
        Structural and sex-conditional statistical metrics.
    """
    target = pl.read_parquet(
        bundle_dir / "normalized" / "job_function_sex_marginal.parquet"
    )
    eligible = pl.col("detailed_status_code").is_in(
        sorted(ELIGIBLE_JOB_FUNCTION_STATUS_CODES)
    )
    code_present = pl.col("job_function_code").is_not_null() & (
        pl.col("job_function_code").str.strip_chars() != ""
    )
    label_present = pl.col("job_function").is_not_null() & (
        pl.col("job_function").str.strip_chars() != ""
    )
    paired = code_present & label_present
    eligibility_errors = frame.filter(
        (
            eligible
            & (~paired | (pl.col("job_function_resolution") != "lons20_sex_marginal"))
        )
        | (
            ~eligible
            & (
                pl.col("job_function_code").is_not_null()
                | pl.col("job_function").is_not_null()
                | (pl.col("job_function_resolution") != "not_applicable")
            )
        )
    ).height
    columns = ["job_function_code", "job_function"]
    expected_pairs = target.select(columns).unique()
    observed_pairs = frame.filter(eligible).select(columns).unique()
    mapping_errors = observed_pairs.join(
        expected_pairs, on=columns, how="anti", nulls_equal=True
    ).height
    metrics = [
        MetricResult(
            name="job_function_eligibility_errors",
            passed=eligibility_errors == 0,
            value=eligibility_errors,
            threshold=0,
            details="Only approved RAS202 employee statuses receive paired fields.",
        ),
        MetricResult(
            name="job_function_mapping_errors",
            passed=mapping_errors == 0,
            value=mapping_errors,
            threshold=0,
            details=(
                "Generated job functions must retain official LONS20 code-label pairs."
            ),
        ),
    ]
    for sex in ("female", "male"):
        metrics.extend(
            _compare_distribution(
                name=f"job_function_{sex}",
                generated=frame.filter(eligible & (pl.col("sex") == sex)),
                target=target.filter(pl.col("sex") == sex),
                columns=columns,
                config=config,
                maximum_tv=config.maximum_total_variation["fitted_marginal"],
                smoke_maximum_tv=(
                    config.maximum_total_variation["fitted_marginal"]
                    if frame.height >= 100_000
                    else config.smoke_maximum_total_variation
                ),
            )
        )
    return metrics


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
    """Check every generated origin pair against the official FOLK2 mapping.

    Args:
        frame:
            Generated records containing origin codes and labels.
        bundle_dir:
            Prepared source bundle with the official origin marginal.

    Returns:
        Mapping and positive-weight validation metrics.
    """
    target = _origin_target(bundle_dir=bundle_dir)
    columns = ["origin_country_code", "origin_country", "origin_country_da"]
    if not set(columns) <= set(frame.columns):
        return [
            MetricResult(
                name="origin_country_mapping",
                passed=False,
                value="missing origin label columns",
                threshold=0,
                details="Every origin record must contain a complete label triple.",
            )
        ]
    expected_pairs = target.select(columns).unique()
    observed_pairs = frame.select(columns).unique()
    mismatches = observed_pairs.join(
        expected_pairs, on=columns, how="anti", nulls_equal=True
    ).height
    zero_weight_pairs = target.filter(pl.col("count") <= 0).select(columns).unique()
    emitted_zero_weight = observed_pairs.join(
        zero_weight_pairs, on=columns, how="inner", nulls_equal=True
    ).height
    incomplete = frame.select(columns).null_count().row(0)
    incomplete_count = sum(incomplete)
    return [
        MetricResult(
            name="origin_country_complete_labels",
            passed=incomplete_count == 0,
            value=incomplete_count,
            threshold=0,
            details="Origin code, English label, and Danish label are paired.",
        ),
        MetricResult(
            name="origin_country_mapping",
            passed=mismatches == 0,
            value=mismatches,
            threshold=0,
            details="Generated origin code and labels must use the official mapping.",
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
        "origin_contract_binding": (
            manifest.origin_labels_contract_path == bundle.origin_labels_contract_path
            and manifest.origin_labels_contract_version
            == bundle.origin_labels_contract_version
            and manifest.origin_labels_contract_sha256
            == bundle.origin_labels_contract_sha256
            and manifest.origin_labels_contract_content
            == bundle.origin_labels_contract_content
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


def _write_reports(directory: Path, report: ValidationReport) -> dict[str, bytes]:
    json_content = (
        json.dumps(
            report.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
    ).encode("utf-8")
    _atomic_report_write(
        path=directory / "validation-report.json", content=json_content
    )
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
    markdown_content = ("\n".join(lines) + "\n").encode("utf-8")
    _atomic_report_write(
        path=directory / "validation-report.md", content=markdown_content
    )
    LOGGER.info(
        "%s validation %s for %s",
        report.kind,
        "passed" if report.passed else "failed",
        report.subject_id,
    )
    return {
        "validation-report.json": json_content,
        "validation-report.md": markdown_content,
    }


def _atomic_report_write(*, path: Path, content: bytes) -> None:
    """Replace one validation report atomically."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def validate_sources(bundle_dir: Path) -> ValidationReport:
    """Validate a prepared source bundle and write reports.

    Args:
        bundle_dir:
            Prepared source bundle.

    Returns:
        Validation report.

    Raises:
        ValueError:
            If the prepared bundle or an existing bound report is invalid.
    """
    manifest, capture = _verify_prepared_bundle_capture(bundle_dir=bundle_dir)
    source_payload = json.loads(capture.files[SOURCE_REPORT].content.decode("utf-8"))
    metrics = [
        MetricResult(
            name="prepared_bundle_integrity",
            passed=True,
            value="pass",
            threshold="pass",
            details=(
                "Schema, required files, checksums, and source preparation are valid."
            ),
        )
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
    job_function_checks = source_payload.get("job_function_checks")
    if isinstance(job_function_checks, dict):
        for check_name in ("two_digit_partition", "sex_coverage", "positive_counts"):
            check = job_function_checks.get(check_name)
            if isinstance(check, dict) and isinstance(check.get("passed"), bool):
                metrics.append(
                    MetricResult(
                        name=f"lons20_{check_name}",
                        passed=check["passed"],
                        value="pass" if check["passed"] else "fail",
                        threshold="pass",
                        details=f"LONS20 {check_name.replace('_', ' ')} check.",
                    )
                )
    report = ValidationReport(
        kind="sources",
        passed=all(metric.passed for metric in metrics),
        created_at=_now(),
        subject_id=manifest.bundle_id,
        metrics=metrics,
    )
    existing_report = _load_existing_source_report(
        content=(
            capture.files["validation-report.json"].content
            if "validation-report.json" in capture.files
            else None
        )
    )
    if existing_report is not None:
        if _report_semantics(existing_report) != _report_semantics(report):
            raise ValueError(
                "Existing source validation report differs from recomputed validation"
            )
        return existing_report

    report_bytes = _write_reports(directory=bundle_dir, report=report)
    manifest_files = dict(manifest.files)
    for report_name, content in report_bytes.items():
        manifest_files[report_name] = hashlib.sha256(content).hexdigest()
    write_json(
        path=bundle_dir / "bundle-manifest.json",
        payload=manifest.model_copy(update={"files": manifest_files}),
    )
    # Re-capture the rewritten report and manifest so callers never rely on
    # bytes or checksums read before the atomic replacements.
    verify_prepared_bundle(bundle_dir=bundle_dir)
    return report


def _load_existing_source_report(*, content: bytes | None) -> ValidationReport | None:
    """Load an already-bound source report without repairing it implicitly.

    The prepared-bundle verifier has already checked the report checksum when this
    function is called.  Parsing it here ensures a bound report with an invalid
    schema cannot be silently replaced by a fresh validation result.

    Returns:
        The existing report, or ``None`` when validation has not run yet.

    Raises:
        ValueError:
            If the bound report cannot be parsed as a validation report.
    """
    if content is None:
        return None
    try:
        return ValidationReport.model_validate_json(content)
    except (ValueError, TypeError) as error:
        raise ValueError("Bound source validation report is malformed") from error


def _report_semantics(report: ValidationReport) -> dict[str, object]:
    """Return report content excluding its non-semantic creation timestamp."""
    return report.model_dump(mode="json", exclude={"created_at"})
