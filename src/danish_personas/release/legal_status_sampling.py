"""Synthetic fine-detail sampling for ambiguous source legal status."""

from __future__ import annotations

import hashlib
import json
import typing as t

import polars as pl

SOURCE_AMBIGUOUS_MARITAL_STATUS = "married_or_separated"
VALID_LEGAL_STATUS_DETAILS = frozenset({"married", "separated"})
VALID_MARITAL_STATUSES = frozenset(
    {SOURCE_AMBIGUOUS_MARITAL_STATUS, "never_married", "widowed", "divorced"}
)
VALID_RELATIONSHIP_STATUSES = frozenset({"partnered", "not_partnered"})
SYNTHETIC_METHOD = (
    "FOLK1A source code G is Gift/separeret. Preserve the source "
    "marital_status='married_or_separated' for audit support, and fill only "
    "missing or invalid legal_status_detail values on those rows. Rows with "
    "current_relationship_status='not_partnered' are set to separated; partnered "
    "ambiguous rows use a domain-separated SHA-256(seed, persona_id) 50/50 draw "
    "between married and separated. The 50/50 split is an uncalibrated synthetic "
    "assumption, not an official Statistics Denmark proportion."
)


class LegalStatusChange(t.TypedDict):
    """Per-person legal-status detail change for prose review."""

    old_legal_status_detail: str | None
    new_legal_status_detail: t.Literal["married", "separated"]
    current_relationship_status: t.Literal["partnered", "not_partnered"]
    reason: str


class LegalStatusSamplingReport(t.TypedDict):
    """Aggregate provenance and per-ID ledger for synthetic legal detail fills."""

    synthetic_method: str
    seed: int
    row_count: int
    changed: dict[str, LegalStatusChange]
    prose_review_ids: list[str]
    counts: dict[str, object]


def apply_legal_status_sampling(
    frame: pl.DataFrame, *, seed: int = 0
) -> tuple[pl.DataFrame, LegalStatusSamplingReport]:
    """Fill missing ambiguous legal detail without changing source status or prose.

    The input frame is validated fail-closed before a separate repaired frame is
    returned. Existing valid fine detail is preserved. Only rows with source
    ``marital_status == 'married_or_separated'`` and missing or invalid
    ``legal_status_detail`` are filled synthetically.

    Args:
        frame:
            Release frame containing persona IDs, source marital status,
            relationship status, and legal detail.
        seed (optional):
            Non-negative deterministic salt for partnered ambiguous rows.
            Defaults to 0.

    Returns:
        A separate Polars frame with the same column order and row order, and an
        absolute-count provenance report with all changed IDs marked for prose
        review.

    Raises:
        ValueError:
            If required columns, IDs, source categories, relationship statuses,
            or existing fine-detail constraints are invalid.
    """
    _validate_seed(seed=seed)
    required = {
        "persona_id",
        "marital_status",
        "legal_status_detail",
        "current_relationship_status",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing legal-status sampling columns: {sorted(missing)}")

    rows = frame.select(
        "persona_id",
        "marital_status",
        "legal_status_detail",
        "current_relationship_status",
    ).iter_rows(named=True)
    seen: set[str] = set()
    output_details: list[str | None] = []
    changed: dict[str, LegalStatusChange] = {}
    input_marital_statuses: list[object] = []
    input_relationships: list[object] = []
    input_details: list[object] = []
    output_detail_counts_source: list[object] = []
    changed_targets: list[object] = []
    married_or_separated_rows = 0
    preserved_valid_detail_rows = 0
    filled_missing_detail_rows = 0
    filled_invalid_detail_rows = 0

    for row in rows:
        identifier = _identifier(value=row["persona_id"])
        if identifier in seen:
            raise ValueError("Persona IDs must be unique")
        seen.add(identifier)

        marital_status = _marital_status(
            value=row["marital_status"], identifier=identifier
        )
        relationship = _relationship(
            value=row["current_relationship_status"], identifier=identifier
        )
        detail = _legal_status_detail(value=row["legal_status_detail"])

        input_marital_statuses.append(row["marital_status"])
        input_relationships.append(row["current_relationship_status"])
        input_details.append(row["legal_status_detail"])

        if marital_status != SOURCE_AMBIGUOUS_MARITAL_STATUS:
            if detail is not None:
                raise ValueError(
                    "legal_status_detail is only allowed for "
                    f"{SOURCE_AMBIGUOUS_MARITAL_STATUS!r}: {identifier}"
                )
            output_details.append(None)
            output_detail_counts_source.append(None)
            continue

        married_or_separated_rows += 1
        if detail in VALID_LEGAL_STATUS_DETAILS:
            _validate_existing_detail(
                identifier=identifier, detail=detail, relationship=relationship
            )
            preserved_valid_detail_rows += 1
            output_details.append(detail)
            output_detail_counts_source.append(detail)
            continue

        new_detail = _synthetic_detail(
            identifier=identifier, relationship=relationship, seed=seed
        )
        output_details.append(new_detail)
        output_detail_counts_source.append(new_detail)
        changed_targets.append(new_detail)
        if detail is None:
            filled_missing_detail_rows += 1
        else:
            filled_invalid_detail_rows += 1
        changed[identifier] = {
            "old_legal_status_detail": detail,
            "new_legal_status_detail": new_detail,
            "current_relationship_status": relationship,
            "reason": "missing_or_invalid_generated_detail_for_source_G",
        }

    detail_dtype = frame.schema["legal_status_detail"]
    if detail_dtype == pl.Null:
        detail_dtype = pl.String
    repaired = frame.with_columns(
        pl.Series("legal_status_detail", output_details, dtype=detail_dtype)
    )
    return repaired, {
        "synthetic_method": SYNTHETIC_METHOD,
        "seed": seed,
        "row_count": frame.height,
        "changed": changed,
        "prose_review_ids": list(changed),
        "counts": {
            "absolute": {
                "rows": frame.height,
                "married_or_separated_rows": married_or_separated_rows,
                "preserved_valid_detail_rows": preserved_valid_detail_rows,
                "filled_missing_detail_rows": filled_missing_detail_rows,
                "filled_invalid_detail_rows": filled_invalid_detail_rows,
                "changed_rows": len(changed),
                "unchanged_rows": frame.height - len(changed),
            },
            "input": {
                "marital_status": _counts(values=input_marital_statuses),
                "current_relationship_status": _counts(values=input_relationships),
                "legal_status_detail": _counts(values=input_details),
            },
            "output": {
                "legal_status_detail": _counts(values=output_detail_counts_source)
            },
            "changed_to": _counts(values=changed_targets),
        },
    }


sample_legal_status_detail = apply_legal_status_sampling


def _counts(*, values: list[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        label = _count_label(value=value)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _count_label(*, value: object) -> str:
    if value is None:
        return "<null>"
    if value == "":
        return "<empty>"
    return str(value)


def _identifier(*, value: object) -> str:
    if value is None or not str(value):
        raise ValueError("Persona IDs must be non-empty")
    return str(value)


def _legal_status_detail(*, value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Malformed legal_status_detail: {value!r}")
    return value


def _marital_status(*, value: object, identifier: str) -> str:
    if not isinstance(value, str) or value not in VALID_MARITAL_STATUSES:
        raise ValueError(f"Invalid marital_status for {identifier}: {value!r}")
    return value


def _relationship(
    *, value: object, identifier: str
) -> t.Literal["partnered", "not_partnered"]:
    if value == "partnered":
        return "partnered"
    if value == "not_partnered":
        return "not_partnered"
    raise ValueError(f"Invalid current_relationship_status for {identifier}: {value!r}")


def _synthetic_detail(
    *, identifier: str, relationship: t.Literal["partnered", "not_partnered"], seed: int
) -> t.Literal["married", "separated"]:
    if relationship == "not_partnered":
        return "separated"
    if _partnered_draw(identifier=identifier, seed=seed) < 0.5:
        return "married"
    return "separated"


def _partnered_draw(*, identifier: str, seed: int) -> float:
    payload = json.dumps(
        ["legal-status-sampling-v1", seed, identifier],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def _validate_existing_detail(
    *,
    identifier: str,
    detail: str,
    relationship: t.Literal["partnered", "not_partnered"],
) -> None:
    if detail == "married" and relationship == "not_partnered":
        raise ValueError(
            "Existing legal_status_detail='married' contradicts "
            f"current_relationship_status='not_partnered': {identifier}"
        )


def _validate_seed(*, seed: int) -> None:
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Legal-status sampling seed must be a non-negative integer")
