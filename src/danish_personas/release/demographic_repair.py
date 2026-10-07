"""Offline orchestration of source-backed demographic repairs."""

from pathlib import Path

import polars as pl

from ..models import ELIGIBLE_JOB_FUNCTION_STATUS_CODES
from .source_repair import (
    InfeasibleRepairError,
    _education_pool_code,
    _ras209_age_band,
    assess_release_support,
    repair_from_source,
)

_EDUCATION_LABELS = {
    "H10": "primary",
    "H20-H35": "secondary_or_vocational",
    "H40-H80": "higher_education",
    "H90": "not_stated",
}
_STATUS_CODES = {
    "employed": {"05", "10", "15", "20", "25", "30", "35", "40"},
    "unemployed": {"50", "85", "90", "95"},
    "student": {"130", "154", "156", "158", "160"},
    "retired": {"135", "138", "139", "140", "145", "150", "155"},
    "other": {"80", "100", "105", "110", "115", "120", "125", "128", "170", "175"},
}


def repair_demographics(
    frame: pl.DataFrame, bundle_dir: Path
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Repair source-supported marginals atomically.

    An infeasible stage returns the original frame, never an earlier partial repair.
    RAS202 detail quotas are conditioned on exact age key, sex, and the repaired
    broad status. Rows aged 71 or over share the source's ``71-`` key.

    Returns:
        Repaired frame and diagnostics. Any blocker returns the original frame.
    """
    original = frame.clone()
    current = frame.clone()
    report: dict[str, object] = {"stages": [], "blocking_infeasibility": []}
    id_column = "persona_id" if "persona_id" in frame.columns else None

    def apply_stage(
        name: str,
        filename: str,
        strata: list[str],
        categories: list[str],
        transform: bool = False,
    ) -> None:
        nonlocal current
        stage_report: dict[str, object] = {
            "name": name,
            "changed_persona_ids": [],
            "changed_count": 0,
            "infeasible_strata": 0,
        }
        path = bundle_dir / "normalized" / filename
        required = {*strata, *categories}
        if not path.is_file() or not required.issubset(current.columns):
            _block(
                report=report,
                item=stage_report,
                name=name,
                reason="required source or columns unavailable",
            )
            return
        try:
            source = pl.read_parquet(path)
            if transform and name == "ras209_joint":
                source = source.with_columns(
                    _education_pool_code().alias("education_source_code"),
                    pl.col("age_band").alias("_repair_age_band"),
                )
            if transform and name == "ras202_detail":
                source = source.with_columns(_source_age_key().alias("_repair_age_key"))
            before = current
            current, diagnostics = repair_from_source(
                frame=current, source=source, strata=strata, categories=categories
            )
            positions = _changed_positions(
                before=before, after=current, columns=categories
            )
            stage_report.update(
                changed_count=len(positions),
                diagnostics=diagnostics,
                after_source_support="supported quotas",
            )
            if id_column and positions:
                stage_report["changed_persona_ids"] = (
                    before[id_column].gather(positions).to_list()
                )
        except (InfeasibleRepairError, pl.exceptions.PolarsError, ValueError) as error:
            _block(
                report=report,
                item=stage_report,
                name=name,
                reason="source-backed quota is infeasible",
                count=1,
            )
            stage_report["diagnostic"] = str(error)[:240]
        report["stages"].append(stage_report)

    apply_stage(
        "folk1a_marital",
        "folk1a_base_unpooled.parquet",
        ["age", "municipality_code", "sex"],
        ["marital_status"],
    )

    has_band = "_repair_age_band" in current.columns
    current = current.with_columns(
        _ras209_age_band(pl.col("age")).alias("_repair_age_band")
    )
    apply_stage(
        "ras209_joint",
        "ras209_joint_unpooled.parquet",
        ["_repair_age_band", "municipality_code", "sex"],
        ["education_source_code", "labour_market_status"],
        transform=True,
    )
    if not has_band:
        current = current.drop("_repair_age_band")
    if (
        "education_source_code" in current.columns
        and "education_level" in current.columns
    ):
        current = current.with_columns(
            pl.col("education_source_code")
            .replace_strict(_EDUCATION_LABELS, default=None)
            .alias("education_level")
        )

    if "detailed_status" not in current.columns:
        _missing_stage(report=report, name="ras202_detail")
    else:
        current = _repair_detail(
            current=current, bundle_dir=bundle_dir, report=report, id_column=id_column
        )

    current = _finish_atomic(
        original=original, current=current, report=report, bundle_dir=bundle_dir
    )
    return current, report


def _repair_detail(
    *,
    current: pl.DataFrame,
    bundle_dir: Path,
    report: dict[str, object],
    id_column: str | None,
) -> pl.DataFrame:
    """Repair RAS202 code and label together inside broad-status strata.

    Returns:
        Repaired frame, or the unchanged input when the stage is infeasible.

    """
    name = "ras202_detail"
    item: dict[str, object] = {
        "name": name,
        "changed_persona_ids": [],
        "changed_count": 0,
        "infeasible_strata": 0,
    }
    path = bundle_dir / "normalized" / "ras202_detail_unpooled.parquet"
    required = {
        "age",
        "sex",
        "labour_market_status",
        "detailed_status_code",
        "detailed_status",
    }
    if not path.is_file() or not required.issubset(current.columns):
        _block(
            report=report,
            item=item,
            name=name,
            reason="required source or columns unavailable",
        )
        return current
    try:
        repaired, positions, diagnostics = _compute_detail_repair(
            current=current, path=path
        )
        item.update(changed_count=len(positions), diagnostics=diagnostics)
        if id_column and positions:
            item["changed_persona_ids"] = current[id_column].gather(positions).to_list()
        report["stages"].append(item)
        return repaired
    except (InfeasibleRepairError, pl.exceptions.PolarsError, ValueError) as error:
        _block(
            report=report,
            item=item,
            name=name,
            reason="source-backed detail quota or paired-field update is infeasible",
        )
        item["diagnostic"] = str(error)[:240]
        return current


def _compute_detail_repair(
    *, current: pl.DataFrame, path: Path
) -> tuple[pl.DataFrame, list[int], dict[str, object]]:
    """Apply source-backed RAS202 quotas and align dependent columns.

    Returns:
        Repaired frame, changed row positions, and quota diagnostics.

    Raises:
        ValueError: If source status codes or paired labels are invalid.
    """
    source = pl.read_parquet(path).with_columns(
        _source_age_key().alias("_repair_age_key")
    )
    source = source.filter(~pl.col("suppressed") & (pl.col("count") > 0))
    invalid = source.filter(
        ~pl.struct("labour_market_status", "detailed_status_code").map_elements(
            _valid_status_code, return_dtype=pl.Boolean
        )
    )
    if invalid.height:
        raise ValueError("RAS202 status codes disagree with categories.yaml")
    staged = current.with_columns(
        pl.when(pl.col("age") <= 70)
        .then(pl.col("age").cast(pl.Int16))
        .otherwise(pl.lit(71, dtype=pl.Int16))
        .alias("_repair_age_key")
    )
    repaired, diagnostics = repair_from_source(
        frame=staged,
        source=source,
        strata=["_repair_age_key", "sex", "labour_market_status"],
        categories=["detailed_status_code", "detailed_status"],
    )
    repaired = repaired.drop("_repair_age_key")
    _validate_labels(source=source, repaired=repaired)
    positions = _changed_positions(
        before=current,
        after=repaired,
        columns=["detailed_status_code", "detailed_status"],
    )
    return (
        _update_detail_dependants(
            before=current, repaired=repaired, positions=positions
        ),
        positions,
        diagnostics,
    )


def _update_detail_dependants(
    *, before: pl.DataFrame, repaired: pl.DataFrame, positions: list[int]
) -> pl.DataFrame:
    """Update resolution and job fields for changed detailed categories.

    Returns:
        Frame with dependent categorical fields made consistent.

    Raises:
        ValueError: If a required resolution or paired field is unavailable.
    """
    if not positions:
        return repaired
    if "detailed_status_resolution" not in repaired.columns:
        raise ValueError("detailed-status resolution column is unavailable")
    resolutions = repaired["detailed_status_resolution"].to_list()
    for index in positions:
        resolutions[index] = "age_band_sex_status"
    repaired = repaired.with_columns(
        pl.Series("detailed_status_resolution", resolutions)
    )
    if "job_function_resolution" in repaired.columns:
        return _adjust_job_functions(
            before=before, repaired=repaired, positions=positions
        )
    if {"job_function_code", "job_function"}.issubset(repaired.columns):
        raise ValueError("job-function resolution column is unavailable")
    if "job_title" in repaired.columns:
        titles = repaired["job_title"].to_list()
        for index in positions:
            titles[index] = None
        repaired = repaired.with_columns(pl.Series("job_title", titles))
    return repaired


def _adjust_job_functions(
    *, before: pl.DataFrame, repaired: pl.DataFrame, positions: list[int]
) -> pl.DataFrame:
    """Keep job-function fields paired or fail without source-backed support.

    Returns:
        Frame with paired job-function values and invalidated titles.

    Raises:
        ValueError: If a changed row becomes eligible without valid fields.
    """
    changed = set(positions)
    codes = repaired["detailed_status_code"].to_list()
    old_codes = before["detailed_status_code"].to_list()
    functions = repaired["job_function_code"].to_list()
    labels = repaired["job_function"].to_list()
    resolutions = repaired["job_function_resolution"].to_list()
    titles = (
        repaired["job_title"].to_list() if "job_title" in repaired.columns else None
    )
    for index in changed:
        was_eligible = old_codes[index] in ELIGIBLE_JOB_FUNCTION_STATUS_CODES
        is_eligible = codes[index] in ELIGIBLE_JOB_FUNCTION_STATUS_CODES
        if was_eligible != is_eligible and is_eligible:
            raise ValueError("newly eligible row has no source-backed job function")
        if not is_eligible:
            functions[index] = labels[index] = None
            resolutions[index] = "not_applicable"
        elif not functions[index] or not labels[index]:
            raise ValueError("eligible row lacks paired job-function fields")
        if titles is not None:
            titles[index] = None
    updates: list[pl.Expr] = [
        pl.Series("job_function_code", functions),
        pl.Series("job_function", labels),
        pl.Series("job_function_resolution", resolutions),
    ]
    if titles is not None:
        updates.append(pl.Series("job_title", titles))
    return repaired.with_columns(updates)


def _finish_atomic(
    *,
    original: pl.DataFrame,
    current: pl.DataFrame,
    report: dict[str, object],
    bundle_dir: Path,
) -> pl.DataFrame:
    """Validate invariants and roll back all stages when any blocker exists.

    Returns:
        The original frame on any blocker, otherwise the repaired frame.
    """
    if current.height != original.height or (
        "persona_id" in original.columns
        and current["persona_id"].to_list() != original["persona_id"].to_list()
    ):
        report["blocking_infeasibility"].append(
            {"stage": "frame_integrity", "count": 1}
        )
    if report["blocking_infeasibility"]:
        final = original
    else:
        final = current
    support_paths = [
        bundle_dir / "normalized" / filename
        for filename in (
            "folk1a_base_unpooled.parquet",
            "ras209_joint_unpooled.parquet",
            "ras202_detail_unpooled.parquet",
        )
    ]
    report["after_source_support"] = (
        assess_release_support(frame=final, bundle_dir=bundle_dir)
        if all(path.is_file() for path in support_paths)
        else {"available": False, "reason": "one or more source files unavailable"}
    )
    report["distribution_diagnostics"] = [
        {
            "stage": item["name"],
            "changed_count": item["changed_count"],
            "infeasible_strata": item["infeasible_strata"],
        }
        for item in report["stages"]
    ]
    return final


def _source_age_key() -> pl.Expr:
    """Convert the RAS202 top-code to the numeric release-row key.

    Returns:
        Expression yielding the numeric key.
    """
    return (
        pl.when(pl.col("age_key") == "71-")
        .then(pl.lit(71, dtype=pl.Int16))
        .otherwise(pl.col("age_key").cast(pl.Int16, strict=False))
    )


def _valid_status_code(row: dict[str, object]) -> bool:
    return row["detailed_status_code"] in _STATUS_CODES.get(
        row["labour_market_status"], set()
    )


def _validate_labels(*, source: pl.DataFrame, repaired: pl.DataFrame) -> None:
    """Require every repaired code to have one source-consistent label.

    Raises:
        ValueError: If a code has inconsistent labels or a row's label disagrees.
    """
    labels = source.select("detailed_status_code", "detailed_status").unique()
    if labels.group_by("detailed_status_code").len().filter(pl.col("len") != 1).height:
        raise ValueError("RAS202 codes have non-unique labels")
    expected = labels.rename({"detailed_status": "_expected_label"})
    checked = repaired.join(expected, on="detailed_status_code", how="left")
    if checked.filter(pl.col("detailed_status") != pl.col("_expected_label")).height:
        raise ValueError("detailed status label does not match its code")


def _changed_positions(
    *, before: pl.DataFrame, after: pl.DataFrame, columns: list[str]
) -> list[int]:
    changed = (
        before.select(columns).to_struct().to_series()
        != after.select(columns).to_struct().to_series()
    )
    return [index for index, value in enumerate(changed) if value]


def _block(
    *,
    report: dict[str, object],
    item: dict[str, object],
    name: str,
    reason: str,
    count: int = 1,
) -> None:
    item.update(infeasible_strata=count, blocker=reason)
    report["blocking_infeasibility"].append({"stage": name, "count": count})
    report["stages"].append(item)


def _missing_stage(*, report: dict[str, object], name: str) -> None:
    _block(
        report=report,
        item={
            "name": name,
            "changed_persona_ids": [],
            "changed_count": 0,
            "infeasible_strata": 0,
        },
        name=name,
        reason="detailed status code and label columns unavailable",
    )
