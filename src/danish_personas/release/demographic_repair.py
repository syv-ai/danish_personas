"""Offline orchestration of source-backed demographic repairs."""

from pathlib import Path

import polars as pl

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


def repair_demographics(
    frame: pl.DataFrame, bundle_dir: Path
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Repair supported demographic marginals without modifying protected fields.

    Unsupported strata are reported as blockers. No row-level values are included in
    the report. RAS209's source bands explicitly pool ages 16–19; adult rows 18–19
    remain in that source band. RAS202 ages 71 and above use its ``71-`` top-code.
    """
    current = frame.clone()
    report: dict[str, object] = {"stages": [], "blocking_infeasibility": []}
    id_column = "persona_id" if "persona_id" in frame.columns else None

    def stage(name: str, source_name: str, strata: list[str], categories: list[str],
              source_transform=None) -> None:
        nonlocal current
        item: dict[str, object] = {"name": name, "changed_persona_ids": [],
                                   "changed_count": 0, "infeasible_strata": 0}
        path = bundle_dir / "normalized" / source_name
        if not path.is_file() or not set([*strata, *categories]).issubset(current.columns):
            item["blocker"] = "required source or columns unavailable"
            report["blocking_infeasibility"].append({"stage": name, "count": 1})
            report["stages"].append(item)
            return
        source = pl.read_parquet(path)
        if source_transform is not None:
            source = source_transform(source)
        before = current
        try:
            current, diagnostic = repair_from_source(
                frame=current, source=source, strata=strata, categories=categories
            )
            changed = before.select(categories).to_struct().to_series() != current.select(
                categories
            ).to_struct().to_series()
            positions = [i for i, changed_value in enumerate(changed) if changed_value]
            item.update({"changed_count": len(positions), "diagnostics": diagnostic})
            if id_column and positions:
                item["changed_persona_ids"] = before[id_column].gather(positions).to_list()
            item["after_source_support"] = "supported quotas"
        except InfeasibleRepairError:
            item["infeasible_strata"] = 1
            item["blocker"] = "one or more observed strata lack disclosed positive support"
            report["blocking_infeasibility"].append({"stage": name, "count": 1})
        report["stages"].append(item)

    stage(
        "folk1a_marital", "folk1a_base_unpooled.parquet",
        ["age", "municipality_code", "sex"], ["marital_status"],
    )

    def ras209_source(source: pl.DataFrame) -> pl.DataFrame:
        return source.with_columns(
            _education_pool_code().alias("education_source_code"),
            pl.col("age_band").alias("_repair_age_band"),
        )

    current = current.with_columns(
        _ras209_age_band(pl.col("age")).alias("_repair_age_band")
    )
    # The helper accepts concrete frame columns; retain the computed band only during
    # its operation, then remove it so the public frame schema is unchanged.
    had_band = "_repair_age_band" in frame.columns
    stage("ras209_joint", "ras209_joint_unpooled.parquet",
          ["_repair_age_band", "municipality_code", "sex"],
          ["education_source_code", "labour_market_status"], ras209_source)
    if not had_band:
        current = current.drop("_repair_age_band")
    if "education_source_code" in current.columns and "education_level" in current.columns:
        current = current.with_columns(
            pl.col("education_source_code").replace_strict(
                _EDUCATION_LABELS, default=pl.col("education_level")
            ).alias("education_level")
        )

    def ras202_source(source: pl.DataFrame) -> pl.DataFrame:
        return source.with_columns(
            pl.when(pl.col("age_key") == "71-").then(pl.lit(71)).otherwise(
                pl.col("age_key").cast(pl.Int16, strict=False)
            ).alias("_repair_age_key")
        )

    current = current.with_columns(
        pl.when(pl.col("age") <= 70).then(pl.col("age")).otherwise(pl.lit(71))
        .alias("_repair_age_key")
    )
    had_age_key = "_repair_age_key" in frame.columns
    if {"detailed_status_label", "detailed_status_resolution"}.intersection(
        current.columns
    ):
        report["stages"].append(
            {
                "name": "ras202_detail",
                "changed_persona_ids": [],
                "changed_count": 0,
                "infeasible_strata": 0,
                "blocker": "code-label-resolution repair is not implemented",
            }
        )
        report["blocking_infeasibility"].append(
            {"stage": "ras202_detail", "count": 1}
        )
    else:
        stage("ras202_detail", "ras202_detail_unpooled.parquet",
              ["_repair_age_key", "sex", "labour_market_status"],
              ["detailed_status_code"], ras202_source)
    if not had_age_key:
        current = current.drop("_repair_age_key")

    if {"labour_market_status", "job_function_code"}.issubset(current.columns):
        invalid = current.filter(
            (pl.col("labour_market_status") == "employed")
            & pl.col("job_function_code").is_null()
        ).height
        if invalid:
            report["blocking_infeasibility"].append(
                {"stage": "job_function_eligibility", "count": invalid}
            )
    support_files = [
        bundle_dir / "normalized" / name
        for name in (
            "folk1a_base_unpooled.parquet",
            "ras209_joint_unpooled.parquet",
            "ras202_detail_unpooled.parquet",
        )
    ]
    report["after_source_support"] = (
        assess_release_support(frame=current, bundle_dir=bundle_dir)
        if all(path.is_file() for path in support_files)
        else {"available": False, "reason": "one or more source files unavailable"}
    )
    report["distribution_diagnostics"] = [
        {"stage": item["name"], "changed_count": item["changed_count"],
         "infeasible_strata": item["infeasible_strata"]}
        for item in report["stages"]
    ]
    return current, report
