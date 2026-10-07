"""Constrained, source-backed repairs for release frames.

The primitives reconcile categories only inside explicitly supplied strata. They do
not claim a joint that the source does not provide: in particular, origin is never a
repair stratum, and no origin-by-status relationship is inferred. These tools address
local support and marginal mismatch, not full release certification.
"""

from pathlib import Path

import polars as pl


class InfeasibleRepairError(ValueError):
    """Raised when release strata cannot be reconciled to disclosed source support."""


def assess_release_support(
    *, frame: pl.DataFrame, bundle_dir: Path
) -> dict[str, object]:
    """Report row-level support coverage against prepared FOLK1A, RAS209 and RAS202.

    FOLK1A checks exact age/sex/municipality/marital cells. RAS209 checks the
    unpooled age-band/municipality/sex/education/broad-status joint. RAS202 checks
    age-band/sex/broad-status/detailed-status support. For RAS202, the prepared
    age bands encode the source's 18-70 exact ages and 71+ top-code; this function
    deliberately does not extrapolate those cells to other ages.

    Args:
        frame: Release rows with source-aligned demographic columns.
        bundle_dir: Prepared bundle root containing normalized Parquets.

    Returns:
        Per-source supported and unsupported row counts and representative missing
        source keys. This is diagnostic only; it does not validate distributions.
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
            ["age_band", "sex", "labour_market_status", "detailed_status_code"],
            ["age_band", "sex", "labour_market_status", "detailed_status_code"],
        ),
    }
    results: dict[str, object] = {}
    for name, (filename, source_keys, frame_keys) in specifications.items():
        missing_columns = set(frame_keys).difference(frame.columns)
        if missing_columns:
            results[name] = {
                "supported_rows": 0,
                "unsupported_rows": frame.height,
                "missing_columns": sorted(missing_columns),
                "missing_keys": [],
            }
            continue
        source = pl.read_parquet(normalized / filename).filter(~pl.col("suppressed"))
        _require_columns(source, source_keys)
        keys = source.select(source_keys).unique()
        checked = frame.with_row_index("_repair_row").join(
            keys.with_columns(pl.lit(True).alias("_supported")),
            left_on=frame_keys,
            right_on=source_keys,
            how="left",
        )
        unsupported_expression = pl.col("_supported").is_null()
        if name == "ras202" and "age" in checked.columns:
            unsupported_expression |= pl.col("age") < 18
        unsupported = checked.filter(unsupported_expression)
        results[name] = {
            "supported_rows": checked.height - unsupported.height,
            "unsupported_rows": unsupported.height,
            "missing_columns": [],
            "missing_keys": unsupported.select(frame_keys).unique().head(10).to_dicts(),
        }
    return results


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


def _require_columns(frame: pl.DataFrame, columns: list[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _normalise_group(key: object) -> tuple[object, ...]:
    return key if isinstance(key, tuple) else (key,)


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
