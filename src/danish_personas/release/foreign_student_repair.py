"""Release-only correction for the v2 foreign-student presentation rule.

The zero Denmark-origin foreign-student rule is a user-requested editorial
presentation rule, not a statistical impossibility inferred from RAS202 or FOLK2.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from ..io import sha256_file

_CODE = "origin_country_code"
_STATUS_CODE = "detailed_status_code"
_STATUS_LABEL = "detailed_status"
_ID = "persona_id"
_HOME_CODE = "5100"
_STUDENT_STATUS = "student"


def repair_published_v2(  # noqa: C901, PLR0912
    *,
    published_v2_path: Path,
    pre_hotfix_v2_path: Path,
    bundle_dir: Path,
    output_path: Path,
    private_manifest_path: Path,
) -> dict[str, object]:
    """Create a minimally changed release repair, failing closed on ambiguity.

    The published and pinned pre-hotfix files must have identical ordered IDs and
    differ only in detailed-status code/label. The exact-age status edits are
    restored from that baseline. Each remaining editorial-rule target exchanges
    its complete origin field group with an age/sex-matched non-Danish candidate
    in another broad status. The private manifest contains row indexes and field
    names only, never IDs or generated text; protect its filesystem permissions.

    Args:
        published_v2_path: Published post-hotfix v2 Parquet.
        pre_hotfix_v2_path: Pinned original v2 Parquet.
        bundle_dir: Prepared bundle root with normalized source tables.
        output_path: New private Parquet output; must not already exist.
        private_manifest_path: New mode-0600 row-index manifest path.

    Returns:
        Aggregate, non-identifying repair summary.

    Raises:
        ValueError: If inputs are ambiguous, unsafe, or fail reconciliation.
    """
    if output_path.exists() or private_manifest_path.exists():
        raise ValueError("Refusing to overwrite repair output")
    hotfix = pl.read_parquet(published_v2_path)
    baseline = pl.read_parquet(pre_hotfix_v2_path)
    _require_columns(
        hotfix,
        (_ID, "age", "sex", "labour_market_status", _CODE, _STATUS_CODE, _STATUS_LABEL),
    )
    _require_columns(baseline, hotfix.columns)
    if hotfix.height != baseline.height or not hotfix[_ID].equals(baseline[_ID]):
        raise ValueError("Published and baseline ordered row identities differ")
    changed_fields = [
        name for name in hotfix.columns if not hotfix[name].equals(baseline[name])
    ]
    if set(changed_fields) != {_STATUS_CODE, _STATUS_LABEL}:
        raise ValueError(
            "Baseline reconciliation is ambiguous or includes unrelated edits"
        )

    rows = hotfix.to_dicts()
    original_rows = baseline.to_dicts()
    restore_indices = [
        i
        for i, row in enumerate(rows)
        if row[_STATUS_CODE] != original_rows[i][_STATUS_CODE]
        or row[_STATUS_LABEL] != original_rows[i][_STATUS_LABEL]
    ]
    if len(restore_indices) != 22 or any(
        original_rows[i]["age"] not in (20, 22) or original_rows[i]["sex"] != "female"
        for i in restore_indices
    ):
        raise ValueError("Expected 11 exact-age swaps do not reconcile uniquely")
    for i in restore_indices:
        rows[i][_STATUS_CODE] = original_rows[i][_STATUS_CODE]
        rows[i][_STATUS_LABEL] = original_rows[i][_STATUS_LABEL]

    origins = [
        name
        for name in hotfix.columns
        if name == _CODE
        or name.startswith("origin_country")
        or name.startswith("origin_")
    ]
    required_origin_fields = {
        "origin_country_code",
        "origin_country",
        "origin_country_da",
    }
    if not required_origin_fields.issubset(origins):
        raise ValueError("Origin field group is incomplete or ambiguous")
    targets = [
        i
        for i, row in enumerate(rows)
        if i in restore_indices
        and row["age"] == 20
        and row["sex"] == "female"
        and row[_STATUS_CODE] == "160"
        and str(row[_CODE]) == _HOME_CODE
    ]
    candidates = [
        i
        for i, row in enumerate(rows)
        if row["age"] == 20
        and row["sex"] == "female"
        and row[_CODE] is not None
        and str(row[_CODE]) != _HOME_CODE
        and row["labour_market_status"] != _STUDENT_STATUS
        and row[_STATUS_CODE] != "160"
    ]
    # Remaining student targets must be the eleven reconciled rows. Any other
    # violation indicates an unsupported or unexpected published schema/state.
    if len(targets) != 11 or any(i not in restore_indices for i in targets):
        raise ValueError("Editorial-rule targets do not reconcile to the baseline")
    if len(candidates) < len(targets):
        raise ValueError("Insufficient same-age, same-sex origin candidates")

    source = pl.read_parquet(
        bundle_dir / "normalized" / "ras202_detail_unpooled.parquet"
    )
    _require_columns(
        source,
        (
            "age_key",
            "sex",
            "labour_market_status",
            "detailed_status_code",
            "detailed_status",
            "count",
            "suppressed",
        ),
    )
    supported_statuses = {
        (
            str(min(_age_key(r["age_key"]), 71)),
            r["sex"],
            r["labour_market_status"],
            str(r["detailed_status_code"]),
            r["detailed_status"],
        )
        for r in source.filter(
            (~pl.col("suppressed")) & (pl.col("count") > 0)
        ).to_dicts()
    }
    origin_source = pl.read_parquet(
        bundle_dir / "normalized" / "folk2_origin_country_marginal.parquet"
    )
    _require_columns(
        origin_source,
        (
            "origin_country_code",
            "origin_country",
            "origin_country_da",
            "eligible_for_sampling",
        ),
    )
    supported_origin_tuples = {
        (row["origin_country_code"], row["origin_country"], row["origin_country_da"])
        for row in origin_source.filter(
            pl.col("eligible_for_sampling").fill_null(False)
        ).to_dicts()
    }
    supported_origins = {item[0] for item in supported_origin_tuples}

    if _HOME_CODE not in supported_origins:
        raise ValueError("Editorial origin category lacks FOLK2 support")
    pairs: list[tuple[int, int]] = []
    unused = set(candidates)
    for target_index in targets:
        target = rows[target_index]
        choices = [
            i
            for i in unused
            if _origin_tuple(rows[i], origins) != _origin_tuple(target, origins)
            and _origin_source_tuple(rows[i]) in supported_origin_tuples
            and _origin_source_tuple(target) in supported_origin_tuples
        ]
        if not choices:
            raise ValueError("No source-supported origin exchange is feasible")
        donor_index = min(
            choices,
            key=lambda i: (
                rows[i]["labour_market_status"] == target["labour_market_status"],
                rows[i][_STATUS_CODE] == target[_STATUS_CODE],
                sum(rows[i][field] != target[field] for field in origins),
                i,
            ),
        )
        unused.remove(donor_index)
        donor = rows[donor_index]
        # Origin values are exchanged as an indivisible schema group.
        target_origin, donor_origin = (
            _origin_tuple(target, origins),
            _origin_tuple(donor, origins),
        )
        for field, target_value, donor_value in zip(
            origins, donor_origin, target_origin, strict=True
        ):
            target[field], donor[field] = target_value, donor_value
        if (
            _origin_source_tuple(target) not in supported_origin_tuples
            or _origin_source_tuple(donor) not in supported_origin_tuples
        ):
            raise ValueError("Changed origin tuple lacks FOLK2 support")
        pairs.append((target_index, donor_index))

    repaired = pl.DataFrame(rows, schema=hotfix.schema)
    if not repaired[_ID].equals(hotfix[_ID]):
        raise ValueError("Repair changed ordered identities")
    if _editorial_violation(repaired):
        raise ValueError("Editorial rule remains violated")
    _verify_changed_fields(hotfix, repaired, origins, restore_indices, pairs)
    changed_status_rows = set(restore_indices)
    for target_index, donor_index in pairs:
        changed_status_rows.update((target_index, donor_index))
    _verify_status_support(
        rows=rows, supported=supported_statuses, indices=changed_status_rows
    )
    before = Counter(zip(hotfix["age"], hotfix["sex"], hotfix[_CODE], strict=True))
    after = Counter(zip(repaired["age"], repaired["sex"], repaired[_CODE], strict=True))
    if before != after:
        raise ValueError("Exact-age/sex origin marginal changed")
    baseline_status = Counter(
        zip(
            baseline["age"],
            baseline["sex"],
            baseline[_STATUS_CODE],
            baseline[_STATUS_LABEL],
            strict=True,
        )
    )
    repaired_status = Counter(
        zip(
            repaired["age"],
            repaired["sex"],
            repaired[_STATUS_CODE],
            repaired[_STATUS_LABEL],
            strict=True,
        )
    )
    if baseline_status != repaired_status:
        raise ValueError(
            "Exact-age/sex detailed-status distribution differs from baseline"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.write_parquet(output_path)
    output_path.chmod(0o600)
    manifest = {
        "published_sha256": sha256_file(published_v2_path),
        "baseline_sha256": sha256_file(pre_hotfix_v2_path),
        "source_status_rows": len(restore_indices),
        "origin_exchange_pairs": len(pairs),
        "changed_row_indexes": sorted(
            {i for pair in pairs for i in pair} | set(restore_indices)
        ),
        "changed_fields": sorted(set(origins) | {_STATUS_CODE, _STATUS_LABEL}),
        "editorial_rule": (
            "zero Denmark-origin foreign students is user-requested, "
            "not official statistical inference"
        ),
        "private": True,
    }
    private_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    private_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    private_manifest_path.chmod(0o600)
    return {
        "output_sha256": sha256_file(output_path),
        "rows": repaired.height,
        "restored_exact_age_rows": len(restore_indices),
        "origin_exchange_pairs": len(pairs),
        "editorial_violations": 0,
    }


def count_editorial_foreign_student_violations(frame: pl.DataFrame) -> int:
    """Count release rows violating the editorial no-Denmark-origin student rule.

    This is a user-requested release presentation gate, not an official statistical
    claim about the joint distribution represented by RAS202 and FOLK2.

    Returns:
        Number of violating release rows.
    """
    _require_columns(frame, ("detailed_status_code", "origin_country_code"))
    return frame.filter(
        (pl.col("detailed_status_code").cast(pl.String) == "160")
        & (pl.col("origin_country_code").cast(pl.String) == _HOME_CODE)
    ).height


def _age_key(value: str | int) -> int:
    """Parse an exact RAS202 age or its official top-code.

    Returns:
        Numeric exact age, using 71 for the top-code.
    """
    return 71 if value == "71-" else int(value)


def _require_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError("Ambiguous schema: required release/source fields are missing")


def _origin_tuple(row: dict[str, Any], fields: list[str]) -> tuple[Any, ...]:
    return tuple(row[field] for field in fields)


def _origin_source_tuple(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (row["origin_country_code"], row["origin_country"], row["origin_country_da"])


def _editorial_violation(frame: pl.DataFrame) -> bool:
    return count_editorial_foreign_student_violations(frame) > 0


def _verify_status_support(
    *,
    rows: list[dict[str, Any]],
    supported: set[tuple[str, Any, Any, str, Any]],
    indices: set[int],
) -> None:
    for index in indices:
        row = rows[index]
        key = (
            str(min(int(row["age"]), 71)),
            row["sex"],
            row["labour_market_status"],
            str(row[_STATUS_CODE]),
            row[_STATUS_LABEL],
        )
        if key not in supported:
            raise ValueError("Changed detailed-status tuple lacks RAS202 support")


def _verify_changed_fields(
    original: pl.DataFrame,
    repaired: pl.DataFrame,
    origin_fields: list[str],
    restored: list[int],
    pairs: list[tuple[int, int]],
) -> None:
    changed_rows = set(restored) | {i for pair in pairs for i in pair}
    changed_columns = set(origin_fields) | {_STATUS_CODE, _STATUS_LABEL}
    for column in original.columns:
        if column not in changed_columns:
            if not repaired[column].equals(original[column]):
                raise ValueError("Repair altered an unapproved field")
        elif column not in origin_fields and column not in {
            _STATUS_CODE,
            _STATUS_LABEL,
        }:
            raise ValueError("Unsafe schema reconciliation")
    for i in range(original.height):
        if i not in changed_rows and any(
            repaired[column][i] != original[column][i] for column in changed_columns
        ):
            raise ValueError("Repair changed an unselected row")
