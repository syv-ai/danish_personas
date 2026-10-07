"""Constrained, source-backed repairs for release frames.

The primitives reconcile categories only inside explicitly supplied strata. They do
not claim a joint that the source does not provide: in particular, origin is never a
repair stratum, and no origin-by-status relationship is inferred. These tools address
local support and marginal mismatch, not full release certification.
"""

from pathlib import Path

import polars as pl


def assess_release_support(
    *, frame: pl.DataFrame, bundle_dir: Path
) -> dict[str, object]:
    """Report source support coverage without exposing missing row-level keys.

    FOLK1A checks exact age/sex/municipality/marital cells. RAS209 checks the
    original age subband and source education codes represented by each pooled
    release category. RAS202 checks exact ages through 70 and its 71+ top-code.

    Args:
        frame: Release rows with source-aligned demographic columns.
        bundle_dir: Prepared bundle root containing normalized Parquets.

    Returns:
        Per-source supported and unsupported row counts. This is diagnostic only;
        it does not validate distributions or disclose missing row-level keys.
    """
    normalized = bundle_dir / "normalized"
    specifications = {
        "folk1a": (
            "folk1a_base_unpooled.parquet",
            ["age", "sex", "municipality_code", "marital_status"],
            ["age", "sex", "municipality_code", "marital_status"],
        ),
        "ras209": (
            "ras209_joint_unpooled.parquet",
            [
                "age_band",
                "municipality_code",
                "sex",
                "education_source_code",
                "labour_market_status",
            ],
            [
                "age_band",
                "municipality_code",
                "sex",
                "education_source_code",
                "labour_market_status",
            ],
        ),
        "ras202": (
            "ras202_detail_unpooled.parquet",
            ["age_key", "sex", "labour_market_status", "detailed_status_code"],
            ["age_key", "sex", "labour_market_status", "detailed_status_code"],
        ),
    }
    results: dict[str, object] = {}
    for name, (filename, source_keys, frame_keys) in specifications.items():
        required_frame_keys = {
            "folk1a": frame_keys,
            "ras209": [
                "age",
                "municipality_code",
                "sex",
                "education_source_code",
                "labour_market_status",
            ],
            "ras202": ["age", "sex", "labour_market_status", "detailed_status_code"],
        }[name]
        missing_columns = set(required_frame_keys).difference(frame.columns)
        if missing_columns:
            results[name] = _missing_column_result(frame=frame, missing=missing_columns)
            continue
        source = pl.read_parquet(normalized / filename)
        _require_columns(source, source_keys)
        source = source.filter((~pl.col("suppressed")) & (pl.col("count") > 0))
        checked_frame = frame
        if name == "ras209":
            missing = {"age", "education_source_code"}.difference(frame.columns)
            if missing:
                results[name] = _missing_column_result(frame=frame, missing=missing)
                continue
            source = source.with_columns(
                _education_pool_code().alias("_release_education_code")
            )
            source_keys = [
                "age_band",
                "municipality_code",
                "sex",
                "_release_education_code",
                "labour_market_status",
            ]
            checked_frame = frame.with_columns(
                _ras209_age_band(pl.col("age")).alias("_release_age_band")
            )
            frame_keys = [
                "_release_age_band",
                "municipality_code",
                "sex",
                "education_source_code",
                "labour_market_status",
            ]
        elif name == "ras202":
            missing = {"age"}.difference(frame.columns)
            if missing:
                results[name] = _missing_column_result(frame=frame, missing=missing)
                continue
            source = source.with_columns(
                pl.when(pl.col("age_key") == "71-")
                .then(pl.lit(71, dtype=pl.Int16))
                .otherwise(pl.col("age_key").cast(pl.Int16, strict=False))
                .alias("_age_key")
            )
            source_keys = [
                "_age_key",
                "sex",
                "labour_market_status",
                "detailed_status_code",
            ]
            checked_frame = frame.with_columns(
                pl.when(pl.col("age") <= 70)
                .then(pl.col("age").cast(pl.Int16))
                .otherwise(pl.lit(71, dtype=pl.Int16))
                .alias("_age_key")
            )
            frame_keys = [
                "_age_key",
                "sex",
                "labour_market_status",
                "detailed_status_code",
            ]
        keys = source.select(source_keys).unique()
        checked = checked_frame.with_row_index("_repair_row").join(
            keys.with_columns(pl.lit(True).alias("_supported")),
            left_on=frame_keys,
            right_on=source_keys,
            how="left",
        )
        unsupported_expression = pl.col("_supported").is_null()
        if name == "ras202":
            unsupported_expression |= pl.col("age") < 18
        unsupported = checked.filter(unsupported_expression)
        results[name] = {
            "supported_rows": checked.height - unsupported.height,
            "unsupported_rows": unsupported.height,
            "missing_columns": [],
            "missing_keys": [],
        }
    return results


def _education_pool_code() -> pl.Expr:
    """Convert an unpooled RAS209 H-code to the release pooling label.

    Returns:
        An expression yielding the pooled release education code.
    """
    code_number = (
        pl.col("education_source_code")
        .cast(pl.String)
        .str.extract(r"^H(\d+)$", 1)
        .cast(pl.Int16, strict=False)
    )
    return (
        pl.when(code_number == 10)
        .then(pl.lit("H10"))
        .when(code_number.is_between(20, 35))
        .then(pl.lit("H20-H35"))
        .when(code_number.is_between(40, 80))
        .then(pl.lit("H40-H80"))
        .when(code_number == 90)
        .then(pl.lit("H90"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )


def _missing_column_result(
    *, frame: pl.DataFrame, missing: set[str]
) -> dict[str, object]:
    """Build a privacy-safe result when the release schema is incomplete.

    Returns:
        An assessment result without row-level keys.
    """
    return {
        "supported_rows": 0,
        "unsupported_rows": frame.height,
        "missing_columns": sorted(missing),
        "missing_keys": [],
    }


def _ras209_age_band(age: pl.Expr) -> pl.Expr:
    """Map exact release ages to the original RAS209 age subbands.

    Returns:
        An expression yielding the source age subband.
    """
    return (
        pl.when(age < 20)
        .then(pl.lit("16-19"))
        .when(age < 25)
        .then(pl.lit("20-24"))
        .when(age < 30)
        .then(pl.lit("25-29"))
        .when(age < 35)
        .then(pl.lit("30-34"))
        .when(age < 40)
        .then(pl.lit("35-39"))
        .when(age < 45)
        .then(pl.lit("40-44"))
        .when(age < 50)
        .then(pl.lit("45-49"))
        .when(age < 55)
        .then(pl.lit("50-54"))
        .when(age < 60)
        .then(pl.lit("55-59"))
        .when(age < 65)
        .then(pl.lit("60-64"))
        .when(age < 67)
        .then(pl.lit("65-66"))
        .otherwise(pl.lit("67-"))
    )


def _require_columns(frame: pl.DataFrame, columns: list[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def repair_from_source(
    *,
    frame: pl.DataFrame,
    source: pl.DataFrame,
    strata: list[str],
    categories: list[str],
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Minimally reassign categories to source proportions within fixed strata.

    Source counts are scaled to each observed release-stratum size using largest
    remainder apportionment. Existing category tuples are retained up to their target
    quota; only surplus rows change, and only to categories with disclosed positive
    counts. Strata with no disclosed source support fail without changing the input.

    Args:
        frame: Release rows. Row order and every non-category column are preserved.
        source: Prepared unpooled source frame containing strata, categories, count
            and suppressed columns.
        strata: Exact columns that must remain unchanged.
        categories: One or more jointly reassigned categorical columns.

    Returns:
        Repaired frame and before/after diagnostics, including changed rows and
        infeasibility details (empty on success).

    Raises:
        InfeasibleRepairError: If an observed stratum lacks disclosed support.
    """
    keys = [*strata, *categories]
    _require_columns(frame, keys)
    _require_columns(source, [*keys, "count", "suppressed"])
    if frame.height == 0:
        return frame.clone(), {"rows": 0, "changed_rows": 0, "strata": []}
    source = source.filter(~pl.col("suppressed") & (pl.col("count") > 0))
    quotas: dict[tuple[object, ...], dict[tuple[object, ...], int]] = {}
    observed = frame.partition_by(strata, as_dict=True, maintain_order=True)
    source_groups = source.partition_by(strata, as_dict=True, maintain_order=True)
    infeasible: list[dict[str, object]] = []
    for raw_group_key, rows in observed.items():
        group_key = _normalise_group(raw_group_key)
        source_rows = source_groups.get(raw_group_key)
        if source_rows is None or source_rows.height == 0:
            infeasible.append({"stratum": group_key, "rows": rows.height})
            continue
        counts = source_rows.group_by(categories).agg(pl.col("count").sum())
        target = _apportion(counts, categories=categories, total=rows.height)
        quotas[group_key] = target
    if infeasible:
        raise InfeasibleRepairError(
            f"No disclosed positive source support for strata: {infeasible[:10]}"
        )

    indexed = frame.with_row_index("_repair_row")
    updates: dict[int, tuple[object, ...]] = {}
    before_mismatch = 0
    for raw_group_key, rows in observed.items():
        group_key = _normalise_group(raw_group_key)
        target = quotas[group_key]
        # Keep earliest source-supported rows up to quota; all deficits receive
        # surplus row indices in stable original order for deterministic minimal edits.
        row_ids = (
            indexed.filter(_group_expression(strata, group_key))
            .get_column("_repair_row")
            .to_list()
        )
        current_by_category: dict[tuple[object, ...], list[int]] = {}
        for row_id, category in zip(
            row_ids, rows.select(categories).iter_rows(), strict=True
        ):
            current_by_category.setdefault(tuple(category), []).append(row_id)
        deficits = dict(target)
        surplus: list[int] = []
        for category, ids in current_by_category.items():
            allowed = min(len(ids), target.get(category, 0))
            deficits[category] = max(0, target.get(category, 0) - allowed)
            surplus.extend(ids[allowed:])
            before_mismatch += len(ids) - allowed
        destinations = [
            category for category, amount in deficits.items() for _ in range(amount)
        ]
        for row_id, destination in zip(surplus, destinations, strict=True):
            updates[row_id] = destination
    repaired = _apply_updates(frame=frame, categories=categories, updates=updates)
    return repaired, {
        "rows": frame.height,
        "changed_rows": len(updates),
        "rows_outside_source_quotas_before": before_mismatch,
        "rows_outside_source_quotas_after": 0,
        "strata": len(observed),
        "preserved_columns": [
            column for column in frame.columns if column not in categories
        ],
        "category_columns": categories,
    }


class InfeasibleRepairError(ValueError):
    """Raised when release strata cannot be reconciled to disclosed source support."""


def _apply_updates(
    *,
    frame: pl.DataFrame,
    categories: list[str],
    updates: dict[int, tuple[object, ...]],
) -> pl.DataFrame:
    if not updates:
        return frame.clone()
    update_frame = pl.DataFrame(
        {
            "_repair_row": list(updates),
            **{
                column: [updates[row][index] for row in updates]
                for index, column in enumerate(categories)
            },
        }
    )
    indexed = frame.with_row_index("_repair_row").join(
        update_frame,
        on="_repair_row",
        how="left",
        suffix="_replacement",
        maintain_order="left",
    )
    return indexed.with_columns(
        [
            pl.coalesce(pl.col(f"{column}_replacement"), pl.col(column)).alias(column)
            for column in categories
        ]
    ).drop(["_repair_row", *[f"{column}_replacement" for column in categories]])


def _apportion(
    counts: pl.DataFrame, *, categories: list[str], total: int
) -> dict[tuple[object, ...], int]:
    values = counts.select([*categories, "count"]).to_dicts()
    denominator = sum(int(row["count"]) for row in values)
    if denominator <= 0:
        raise ValueError("Source stratum has no positive count")
    exact = [int(row["count"]) * total / denominator for row in values]
    allocated = [int(value) for value in exact]
    remainder = total - sum(allocated)
    order = sorted(
        range(len(values)),
        key=lambda index: exact[index] - allocated[index],
        reverse=True,
    )
    for index in order[:remainder]:
        allocated[index] += 1
    return {
        tuple(row[column] for column in categories): allocated[index]
        for index, row in enumerate(values)
    }


def _group_expression(strata: list[str], key: tuple[object, ...]) -> pl.Expr:
    expression = pl.lit(True)
    for column, value in zip(strata, key, strict=True):
        expression &= (
            pl.col(column).is_null() if value is None else pl.col(column) == value
        )
    return expression


def _normalise_group(key: object) -> tuple[object, ...]:
    return key if isinstance(key, tuple) else (key,)
