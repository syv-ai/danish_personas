"""Source-backed synthetic allocation of missing education rows for release repair."""

from __future__ import annotations

import hashlib
import json
import typing as t
from pathlib import Path

import polars as pl

from .source_repair import _ras209_age_band

EDUCATION_MISSINGNESS_METHOD = (
    "synthetic_h90_missing_education_allocation_v1_"
    "sha256_seed_persona_id_source_h90_stratum_quotas"
)
SYNTHETIC_ALLOCATION_NOTE = (
    "This is a deterministic synthetic allocation to restore the aggregate "
    "source-backed not-stated education share. It is not observed person-level "
    "education data and must not be described as such."
)
_H90 = "H90"
_NOT_STATED = "not_stated"
_REQUIRED_FRAME_COLUMNS = frozenset(
    {
        "persona_id",
        "age",
        "municipality_code",
        "sex",
        "labour_market_status",
        "education_source_code",
        "education_level",
    }
)
_REQUIRED_SOURCE_COLUMNS = frozenset(
    {
        "age_band",
        "municipality_code",
        "sex",
        "education_source_code",
        "labour_market_status",
        "count",
        "suppressed",
    }
)


class EducationMissingnessReport(t.TypedDict):
    """Aggregate provenance for an offline H90 education allocation."""

    method: str
    seed: int
    synthetic_allocation_note: str
    source_share: float
    source_h90_count: int
    source_age20plus_count: int
    age20plus_rows: int
    source_target_h90: int
    current_h90: int
    after_h90: int
    additional_h90_required: int
    eligible_known_rows: int
    changed_rows: int
    changed_persona_id_sha256: list[str]
    source_support: dict[str, dict[str, int]]
    stratum_allocations: list[dict[str, object]]


def build_education_missingness_candidate(
    frame: pl.DataFrame,
    bundle_dir: Path,
    *,
    seed: int = 0,
    diagnostic_path: Path | None = None,
    official_age20plus_h90_share: float | None = None,
) -> tuple[pl.DataFrame, EducationMissingnessReport]:
    """Build a separate candidate with source-backed H90 education missingness.

    The input frame and source bundle are read only. The returned frame preserves
    row order, persona IDs, prose, and every non-education field. Only existing
    age-20-plus rows with known education can be changed, and only when their
    exact RAS209 municipality, sex, source age subband, and broad labour-market
    status has a disclosed positive H90 source cell.

    Args:
        frame:
            Release candidate frame. It is never mutated.
        bundle_dir:
            Prepared bundle root containing ``normalized/ras209_joint_unpooled``.
        seed (optional):
            Domain salt used with persona ID SHA-256 ranking. Defaults to 0.
        diagnostic_path (optional):
            Optional JSON diagnostic from which to read an official H90 share.
            Defaults to None.
        official_age20plus_h90_share (optional):
            Optional explicit source H90 share. When omitted, the share is read
            from ``diagnostic_path`` if present, otherwise calculated directly
            from the positive unsuppressed RAS209 source rows.

    Returns:
        A separate repaired frame and aggregate provenance report.
    """
    source_path = bundle_dir / "normalized" / "ras209_joint_unpooled.parquet"
    source = pl.read_parquet(source_path)
    return allocate_education_missingness(
        frame=frame,
        ras209_joint_unpooled=source,
        seed=seed,
        diagnostic_path=diagnostic_path,
        official_age20plus_h90_share=official_age20plus_h90_share,
    )


def allocate_education_missingness(
    frame: pl.DataFrame,
    ras209_joint_unpooled: pl.DataFrame,
    *,
    seed: int = 0,
    diagnostic_path: Path | None = None,
    official_age20plus_h90_share: float | None = None,
) -> tuple[pl.DataFrame, EducationMissingnessReport]:
    """Allocate additional H90 rows without changing any live artefact.

    Args:
        frame:
            Release candidate rows.
        ras209_joint_unpooled:
            Offline RAS209 unpooled source rows.
        seed (optional):
            Non-negative deterministic salt. Defaults to 0.
        diagnostic_path (optional):
            Optional JSON diagnostic carrying the official source H90 share.
            Defaults to None.
        official_age20plus_h90_share (optional):
            Explicit source H90 share. Defaults to None.

    Returns:
        A separate frame and an aggregate source/provenance report.

    Raises:
        EducationMissingnessError:
            If the target is below existing H90 rows, the source lacks
            age-20-plus counts, or supported eligible rows are insufficient.
    """
    _validate_inputs(frame=frame, source=ras209_joint_unpooled, seed=seed)
    source = _positive_unsuppressed_source(source=ras209_joint_unpooled)
    source_share, source_h90_count, source_age20plus_count = _source_share(
        source=source,
        diagnostic_path=diagnostic_path,
        explicit_share=official_age20plus_h90_share,
    )
    if source_age20plus_count <= 0:
        raise EducationMissingnessError("RAS209 source has no age-20-plus rows")

    working = frame.with_row_index("_education_missingness_row").with_columns(
        _ras209_age_band(pl.col("age")).alias("_education_missingness_age_band")
    )
    age20plus_rows = working.filter(pl.col("age") >= 20).height
    target_h90 = _rounded_count(source_share * age20plus_rows)
    current_h90 = working.filter((pl.col("age") >= 20) & _is_h90_expr()).height
    additional = target_h90 - current_h90
    if additional < 0:
        raise EducationMissingnessError(
            "Target H90 rows are fewer than existing source-supported H90 rows"
        )

    h90_source = _h90_source_by_stratum(source=source)
    eligible = _eligible_rows(frame=working, h90_source=h90_source)
    if eligible.height < additional:
        raise EducationMissingnessError(
            "Insufficient eligible source-supported known-education rows"
        )

    quotas = _quota_by_stratum(
        h90_source=h90_source, eligible=eligible, total=additional
    )
    changed_row_ids = _select_rows(eligible=eligible, quotas=quotas, seed=seed)
    repaired = _apply_h90(frame=frame, changed_row_ids=changed_row_ids)
    after_h90 = current_h90 + len(changed_row_ids)
    support_before = _h90_support(frame=working, h90_source=h90_source)
    support_after = _h90_support(
        frame=repaired.with_row_index("_education_missingness_row").with_columns(
            _ras209_age_band(pl.col("age")).alias("_education_missingness_age_band")
        ),
        h90_source=h90_source,
    )
    report: EducationMissingnessReport = {
        "method": EDUCATION_MISSINGNESS_METHOD,
        "seed": seed,
        "synthetic_allocation_note": SYNTHETIC_ALLOCATION_NOTE,
        "source_share": source_share,
        "source_h90_count": source_h90_count,
        "source_age20plus_count": source_age20plus_count,
        "age20plus_rows": age20plus_rows,
        "source_target_h90": target_h90,
        "current_h90": current_h90,
        "after_h90": after_h90,
        "additional_h90_required": additional,
        "eligible_known_rows": eligible.height,
        "changed_rows": len(changed_row_ids),
        "changed_persona_id_sha256": _changed_id_hashes(
            frame=working, changed_row_ids=changed_row_ids
        ),
        "source_support": {"before": support_before, "after": support_after},
        "stratum_allocations": _stratum_report(
            h90_source=h90_source, eligible=eligible, quotas=quotas
        ),
    }
    return repaired, report


class EducationMissingnessError(ValueError):
    """Raised when the source-backed H90 allocation is infeasible."""


build_candidate = build_education_missingness_candidate
repair_education_missingness = allocate_education_missingness


def _apply_h90(*, frame: pl.DataFrame, changed_row_ids: set[int]) -> pl.DataFrame:
    if not changed_row_ids:
        return frame.clone()
    row_ids = sorted(changed_row_ids)
    return (
        frame.with_row_index("_education_missingness_row")
        .with_columns(
            pl.when(pl.col("_education_missingness_row").is_in(row_ids))
            .then(pl.lit(_H90))
            .otherwise(pl.col("education_source_code"))
            .alias("education_source_code"),
            pl.when(pl.col("_education_missingness_row").is_in(row_ids))
            .then(pl.lit(_NOT_STATED))
            .otherwise(pl.col("education_level"))
            .alias("education_level"),
        )
        .drop("_education_missingness_row")
    )


def _changed_id_hashes(*, frame: pl.DataFrame, changed_row_ids: set[int]) -> list[str]:
    if not changed_row_ids:
        return []
    identifiers = frame.filter(
        pl.col("_education_missingness_row").is_in(sorted(changed_row_ids))
    ).get_column("persona_id")
    return sorted(
        hashlib.sha256(str(value).encode()).hexdigest() for value in identifiers
    )


def _eligible_rows(*, frame: pl.DataFrame, h90_source: pl.DataFrame) -> pl.DataFrame:
    joined = frame.join(
        h90_source.with_columns(pl.lit(True).alias("_education_missingness_supported")),
        left_on=_frame_stratum_columns(),
        right_on=_stratum_columns(),
        how="left",
    )
    return joined.filter(
        (pl.col("age") >= 20)
        & (~_is_h90_expr())
        & pl.col("_education_missingness_supported").fill_null(False)
    )


def _frame_stratum_columns() -> list[str]:
    return [
        "_education_missingness_age_band",
        "municipality_code",
        "sex",
        "labour_market_status",
    ]


def _is_h90_expr() -> pl.Expr:
    return (pl.col("education_source_code") == _H90) | (
        pl.col("education_level") == _NOT_STATED
    )


def _stratum_columns() -> list[str]:
    return ["age_band", "municipality_code", "sex", "labour_market_status"]


def _h90_source_by_stratum(*, source: pl.DataFrame) -> pl.DataFrame:
    return (
        source.filter(
            _source_age20plus_expr() & (pl.col("education_source_code") == _H90)
        )
        .group_by(_stratum_columns())
        .agg(pl.col("count").sum().alias("source_h90_count"))
        .sort(_stratum_columns())
    )


def _source_age20plus_expr() -> pl.Expr:
    lower = pl.col("age_band").cast(pl.String).str.extract(r"^(\d+)", 1)
    return lower.cast(pl.Int16, strict=False) >= 20


def _h90_support(*, frame: pl.DataFrame, h90_source: pl.DataFrame) -> dict[str, int]:
    h90_rows = frame.filter((pl.col("age") >= 20) & _is_h90_expr())
    if h90_rows.height == 0:
        return {"h90_rows": 0, "supported_h90_rows": 0, "unsupported_h90_rows": 0}
    checked = h90_rows.join(
        h90_source.with_columns(pl.lit(True).alias("_education_missingness_supported")),
        left_on=_frame_stratum_columns(),
        right_on=_stratum_columns(),
        how="left",
    )
    supported = checked.filter(
        pl.col("_education_missingness_supported").fill_null(False)
    ).height
    return {
        "h90_rows": h90_rows.height,
        "supported_h90_rows": supported,
        "unsupported_h90_rows": h90_rows.height - supported,
    }


def _positive_unsuppressed_source(*, source: pl.DataFrame) -> pl.DataFrame:
    return source.filter((~pl.col("suppressed")) & (pl.col("count") > 0))


def _quota_by_stratum(
    *, h90_source: pl.DataFrame, eligible: pl.DataFrame, total: int
) -> dict[tuple[object, ...], int]:
    capacities = _capacity_by_stratum(eligible=eligible)
    weights = _weight_by_stratum(h90_source=h90_source, capacities=capacities)
    if total == 0:
        return {key: 0 for key in weights}
    if sum(capacities.values()) < total:
        raise EducationMissingnessError(
            "Insufficient eligible source-supported known-education rows"
        )
    quotas: dict[tuple[object, ...], int] = {key: 0 for key in weights}
    remaining = _apportion_into_capacities(
        weights=weights, capacities=capacities, quotas=quotas, total=total
    )
    if remaining:
        raise EducationMissingnessError(
            "Unable to satisfy H90 stratum quotas within eligible capacities"
        )
    return quotas


def _apportion_into_capacities(
    *,
    weights: dict[tuple[object, ...], int],
    capacities: dict[tuple[object, ...], int],
    quotas: dict[tuple[object, ...], int],
    total: int,
) -> int:
    remaining = total
    active = set(weights)
    while remaining > 0 and active:
        increments = _largest_remainder_increments(
            active=active, weights=weights, remaining=remaining
        )
        progressed = _apply_quota_increments(
            increments=increments,
            capacities=capacities,
            quotas=quotas,
            active=active,
            remaining=remaining,
        )
        remaining -= progressed
        if progressed == 0:
            break
    return remaining


def _apply_quota_increments(
    *,
    increments: dict[tuple[object, ...], int],
    capacities: dict[tuple[object, ...], int],
    quotas: dict[tuple[object, ...], int],
    active: set[tuple[object, ...]],
    remaining: int,
) -> int:
    progressed = 0
    for key, increment in increments.items():
        if increment <= 0:
            continue
        room = capacities[key] - quotas[key]
        applied = min(room, increment, remaining - progressed)
        quotas[key] += applied
        progressed += applied
        if quotas[key] >= capacities[key]:
            active.remove(key)
    return progressed


def _largest_remainder_increments(
    *,
    active: set[tuple[object, ...]],
    weights: dict[tuple[object, ...], int],
    remaining: int,
) -> dict[tuple[object, ...], int]:
    active_weight = sum(weights[key] for key in active)
    if active_weight <= 0:
        return {}
    shares = [
        (key, remaining * weights[key] / active_weight)
        for key in sorted(active, key=_sort_key)
    ]
    increments = {key: int(share) for key, share in shares}
    remainders = sorted(
        ((share - int(share), _sort_key(key), key) for key, share in shares),
        reverse=True,
    )
    missing = remaining - sum(increments.values())
    for _, _, key in remainders[:missing]:
        increments[key] += 1
    return increments


def _sort_key(key: tuple[object, ...]) -> str:
    return repr(key)


def _capacity_by_stratum(*, eligible: pl.DataFrame) -> dict[tuple[object, ...], int]:
    capacities: dict[tuple[object, ...], int] = {}
    for row in eligible.select(_frame_stratum_columns()).iter_rows():
        key = tuple(row)
        capacities[key] = capacities.get(key, 0) + 1
    return capacities


def _weight_by_stratum(
    *, h90_source: pl.DataFrame, capacities: dict[tuple[object, ...], int]
) -> dict[tuple[object, ...], int]:
    weights: dict[tuple[object, ...], int] = {}
    for row in h90_source.iter_rows(named=True):
        key = tuple(row[column] for column in _stratum_columns())
        if capacities.get(key, 0) > 0:
            weights[key] = int(row["source_h90_count"])
    return weights


def _rounded_count(value: float) -> int:
    return int(value + 0.5)


def _select_rows(
    *, eligible: pl.DataFrame, quotas: dict[tuple[object, ...], int], seed: int
) -> set[int]:
    selected: set[int] = set()
    for key, quota in quotas.items():
        if quota == 0:
            continue
        rows = eligible.filter(_stratum_expr(key=key)).with_columns(
            pl.col("persona_id")
            .map_elements(
                lambda value: _ranking(seed=seed, persona_id=str(value)),
                return_dtype=pl.String,
            )
            .alias("_education_missingness_rank")
        )
        row_ids = (
            rows.sort("_education_missingness_rank")
            .head(quota)
            .get_column("_education_missingness_row")
            .to_list()
        )
        selected.update(int(row_id) for row_id in row_ids)
    return selected


def _ranking(*, seed: int, persona_id: str) -> str:
    payload = f"{EDUCATION_MISSINGNESS_METHOD}:{seed}:{persona_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _stratum_expr(*, key: tuple[object, ...]) -> pl.Expr:
    expression = pl.lit(True)
    for column, value in zip(_frame_stratum_columns(), key, strict=True):
        expression &= pl.col(column).eq(value)
    return expression


def _source_share(
    *, source: pl.DataFrame, diagnostic_path: Path | None, explicit_share: float | None
) -> tuple[float, int, int]:
    age20plus = source.filter(_source_age20plus_expr())
    source_age20plus_count = _sum_count(frame=age20plus)
    source_h90_count = _sum_count(
        frame=age20plus.filter(pl.col("education_source_code") == _H90)
    )
    source_share = (
        explicit_share
        if explicit_share is not None
        else _diagnostic_share(path=diagnostic_path)
    )
    if source_share is None and source_age20plus_count > 0:
        source_share = source_h90_count / source_age20plus_count
    if source_share is None:
        source_share = 0.0
    if not 0 <= source_share <= 1:
        raise ValueError("Official age-20-plus H90 share must be between 0 and 1")
    return source_share, source_h90_count, source_age20plus_count


def _diagnostic_share(*, path: Path | None) -> float | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _find_share(value=payload)


def _find_share(*, value: object) -> float | None:
    parsed = _parse_share_scalar(value=value)
    if parsed is not None:
        return parsed
    if isinstance(value, list):
        return _find_share_in_items(items=value)
    if not isinstance(value, dict):
        return None
    found = _find_preferred_share(mapping=value)
    if found is not None:
        return found
    return _find_named_share(mapping=value)


def _find_named_share(*, mapping: dict[object, object]) -> float | None:
    for key, item in mapping.items():
        if "share" not in str(key).casefold():
            continue
        found = _find_share(value=item)
        if found is not None:
            return found
    return None


def _find_preferred_share(*, mapping: dict[object, object]) -> float | None:
    preferred = (
        "official_age20plus_h90_share",
        "official_age20plus_not_stated_share",
        "source_age20plus_h90_share",
        "source_age20plus_not_stated_share",
        "official_share",
        "source_share",
    )
    for key in preferred:
        if key not in mapping:
            continue
        found = _find_share(value=mapping[key])
        if found is not None:
            return found
    return None


def _find_share_in_items(*, items: list[object]) -> float | None:
    for item in items:
        found = _find_share(value=item)
        if found is not None:
            return found
    return None


def _parse_share_scalar(*, value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
    else:
        return None
    return parsed if 0 <= parsed <= 1 else None


def _sum_count(*, frame: pl.DataFrame) -> int:
    if frame.height == 0:
        return 0
    return int(frame.select(pl.col("count").sum()).item())


def _stratum_report(
    *,
    h90_source: pl.DataFrame,
    eligible: pl.DataFrame,
    quotas: dict[tuple[object, ...], int],
) -> list[dict[str, object]]:
    capacities = _capacity_by_stratum(eligible=eligible)
    source_counts = _source_count_by_stratum(h90_source=h90_source)
    report: list[dict[str, object]] = []
    for key in sorted(
        set(source_counts) | set(capacities) | set(quotas), key=_sort_key
    ):
        item = dict(zip(_stratum_columns(), key, strict=True))
        item.update(
            {
                "source_h90_count": source_counts.get(key, 0),
                "eligible_known_rows": capacities.get(key, 0),
                "changed_rows": quotas.get(key, 0),
            }
        )
        report.append(item)
    return report


def _source_count_by_stratum(
    *, h90_source: pl.DataFrame
) -> dict[tuple[object, ...], int]:
    return {
        tuple(row[column] for column in _stratum_columns()): int(
            row["source_h90_count"]
        )
        for row in h90_source.iter_rows(named=True)
    }


def _validate_inputs(*, frame: pl.DataFrame, source: pl.DataFrame, seed: int) -> None:
    if seed < 0:
        raise ValueError("Seed must be non-negative")
    missing_frame = _REQUIRED_FRAME_COLUMNS.difference(frame.columns)
    if missing_frame:
        raise ValueError(
            f"Missing education-missingness columns: {sorted(missing_frame)}"
        )
    missing_source = _REQUIRED_SOURCE_COLUMNS.difference(source.columns)
    if missing_source:
        raise ValueError(f"Missing RAS209 source columns: {sorted(missing_source)}")
    persona_ids = [
        str(value) if value is not None else "" for value in frame["persona_id"]
    ]
    if any(not value for value in persona_ids):
        raise ValueError("Persona IDs must be non-empty")
    if len(set(persona_ids)) != len(persona_ids):
        raise ValueError("Persona IDs must be unique")
