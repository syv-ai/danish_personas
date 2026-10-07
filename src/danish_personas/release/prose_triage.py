"""Conservatively triage attribute repairs for prose review without editing prose."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal

import polars as pl

TriageLabel = Literal[
    "semantic_equivalence_reviewed", "needs_prose_review_or_regeneration"
]

_AUDIT_FIELDS = frozenset(
    {
        "origin_country_code",
        "age_resolution",
        "marital_resolution",
        "detailed_status_resolution",
        "education_resolution",
        "job_function_resolution",
        "origin_resolution",
        "municipality_code",
        "region_code",
        "landsdel_code",
        "source_code",
        "source_id",
        "source_row_id",
        "education_source_code",
    }
)
_LIST_FIELDS = frozenset({"hobbies_and_interests", "skills_and_expertise"})
_LEXICAL_FIELDS = frozenset({"job_title"})
_CODE_TO_VISIBLE_FIELD = {
    "detailed_status_code": "detailed_status",
    "job_function_code": "job_function",
}
_UNSAFE_FIELDS = frozenset(
    {
        "marital_status",
        "legal_status_detail",
        "current_relationship_status",
        "education_level",
        "broad_status",
        "detailed_status",
        "job_function",
        "partner_gender",
        "same_sex_partner_target",
        "gender",
        "sex",
    }
)
_UNSAFE_REASON_TAGS = frozenset(
    {"unresolved_attribute_review", "partner_gender_scenario", "gender_prose_review"}
)
_PROSE_FIELDS = frozenset({"persona_text", "persona_prose", "prose"})


def triage_prose_changes(
    original: pl.DataFrame,
    repaired: pl.DataFrame,
    changed_reasons: Mapping[str, Sequence[str]],
    *,
    persona_id_column: str = "persona_id",
) -> dict[str, object]:
    """Classify changes by whether they can introduce prose conflicts.

    Persona prose itself must never change. The ledger must identify every changed
    persona; recognised unsafe markers may additionally identify unchanged personas.

    Returns:
        Per-persona classifications, reasons, changed fields, and aggregate counts.

    """
    rows, diffs = _collect_diffs(original, repaired, persona_id_column)
    marker_only_ids = _validate_ledger(changed_reasons, set(diffs))
    results: dict[str, dict[str, object]] = {}
    counts = {
        "semantic_equivalence_reviewed": 0,
        "needs_prose_review_or_regeneration": 0,
    }
    for persona_id in sorted(set(diffs) | marker_only_ids):
        fields = diffs.get(persona_id, [])
        before, after = rows[persona_id]
        reasons = list(changed_reasons[persona_id])
        safe = _is_safe_change(fields, before, after, reasons)
        label: TriageLabel = (
            "semantic_equivalence_reviewed"
            if safe
            else "needs_prose_review_or_regeneration"
        )
        counts[label] += 1
        results[persona_id] = {
            "classification": label,
            "changed_fields": fields,
            "reasons": reasons,
        }
    return {"personas": results, "counts": counts}


def _collect_diffs(
    original: pl.DataFrame, repaired: pl.DataFrame, persona_id_column: str
) -> tuple[
    dict[str, tuple[dict[str, object], dict[str, object]]], dict[str, list[str]]
]:
    if original.columns != repaired.columns:
        raise ValueError("Original and repaired frames must have identical columns")
    if persona_id_column not in original.columns:
        raise ValueError(f"Missing persona ID column: {persona_id_column}")
    before_ids = original.get_column(persona_id_column).to_list()
    after_ids = repaired.get_column(persona_id_column).to_list()
    if any(value is None for value in before_ids) or len(set(before_ids)) != len(
        before_ids
    ):
        raise ValueError("Persona IDs must be unique and non-null")
    if before_ids != after_ids:
        raise ValueError("Original and repaired persona IDs must have identical order")

    rows: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    diffs: dict[str, list[str]] = {}
    for before, after in zip(original.to_dicts(), repaired.to_dicts(), strict=True):
        persona_id = str(before[persona_id_column])
        if persona_id in rows:
            raise ValueError("Persona IDs must remain unique after string conversion")
        rows[persona_id] = (before, after)
        fields = [
            field
            for field in original.columns
            if field != persona_id_column and before[field] != after[field]
        ]
        if any(_is_prose_field(field) for field in fields):
            raise ValueError(f"Persona prose must not change: {persona_id}")
        if fields:
            diffs[persona_id] = fields
    return rows, diffs


def _is_prose_field(field: str) -> bool:
    return field in _PROSE_FIELDS or "persona_text" in field or "prose" in field


def _is_safe_change(
    fields: list[str],
    before: dict[str, object],
    after: dict[str, object],
    reasons: list[str],
) -> bool:
    safe_fields = bool(fields) and all(
        _field_is_equivalent(field, before, after) for field in fields
    )
    known_reasons = all(tag in fields or tag in _UNSAFE_REASON_TAGS for tag in reasons)
    return (
        safe_fields
        and known_reasons
        and not any(tag in _UNSAFE_REASON_TAGS for tag in reasons)
    )


def _field_is_equivalent(
    field: str, before: dict[str, object], after: dict[str, object]
) -> bool:
    if field in _AUDIT_FIELDS:
        return True
    if field in _CODE_TO_VISIBLE_FIELD:
        visible_field = _CODE_TO_VISIBLE_FIELD[field]
        return before.get(visible_field) == after.get(visible_field)
    if field in _UNSAFE_FIELDS or _is_prose_field(field):
        return False
    if field in _LIST_FIELDS:
        before_items = _normalised_list(before[field])
        after_items = _normalised_list(after[field])
        return before_items is not None and before_items == after_items
    if field in _LEXICAL_FIELDS:
        old, new = before[field], after[field]
        return (
            isinstance(old, str)
            and isinstance(new, str)
            and bool(_lexical_identity(old))
            and _lexical_identity(old) == _lexical_identity(new)
        )
    return False


def _lexical_identity(value: str) -> str:
    """Normalise case and trailing punctuation without altering word identity.

    Returns:
        A whitespace-normalised lexical identity.
    """
    return " ".join(re.sub(r"[\W_]+$", "", value.casefold(), flags=re.UNICODE).split())


def _normalised_list(value: object) -> set[str] | None:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        return None
    normalised = {_lexical_identity(item) for item in value}
    if not normalised or "" in normalised:
        return None
    return normalised


def _validate_ledger(
    changed_reasons: Mapping[str, Sequence[str]], diff_ids: set[str]
) -> set[str]:
    for persona_id, tags in changed_reasons.items():
        if (
            isinstance(tags, (str, bytes))
            or not tags
            or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
        ):
            raise ValueError(
                f"Changed reasons must be non-empty string sequences: {persona_id}"
            )
    marker_only_ids = {
        persona_id
        for persona_id, tags in changed_reasons.items()
        if persona_id not in diff_ids
        and any(tag in _UNSAFE_REASON_TAGS for tag in tags)
    }
    extra_ids = set(changed_reasons) - diff_ids - marker_only_ids
    missing_ids = diff_ids - set(changed_reasons)
    if missing_ids or extra_ids:
        raise ValueError(
            "Changed-reason ledger does not match frame diff "
            f"(missing={sorted(missing_ids)}, extra={sorted(extra_ids)})"
        )
    return marker_only_ids
