"""Conservatively triage attribute repairs for prose review without editing prose."""

from __future__ import annotations

import re
import string
from collections.abc import Mapping, Sequence
from typing import Literal

import polars as pl

TriageLabel = Literal[
    "semantic_equivalence_reviewed", "needs_prose_review_or_regeneration"
]

# These values are internal provenance, never facts supplied to persona prose.
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
    }
)
_LIST_FIELDS = frozenset({"hobbies_and_interests", "skills_and_expertise"})
_LEXICAL_FIELDS = frozenset({"job_title"})
_UNSAFE_FIELDS = frozenset(
    {
        "marital_status",
        "legal_status_detail",
        "current_relationship_status",
        "education_level",
        "broad_status",
        "detailed_status",
        "detailed_status_code",
        "job_function",
        "job_function_code",
        "partner_gender",
        "same_sex_partner_target",
        "gender",
        "sex",
    }
)
_PROSE_FIELDS = frozenset({"persona_text", "persona_prose", "prose"})
_UNSAFE_REASON_TAGS = frozenset(
    {"unresolved_attribute_review", "partner_gender_scenario", "gender_prose_review"}
)


def triage_prose_changes(
    original: pl.DataFrame,
    repaired: pl.DataFrame,
    changed_reasons: Mapping[str, Sequence[str]],
    *,
    persona_id_column: str = "persona_id",
) -> dict[str, object]:
    """Classify changed personas by whether their edits can introduce new prose conflicts.

    The reason ledger must contain exactly the IDs that differ in the data frames.
    A persona is marked semantically equivalent only if all actual changed fields are
    covered by conservative equivalence rules and no unresolved reason is present.
    This does not certify that existing prose is semantically valid.

    Args:
        original: The unmodified candidate frame.
        repaired: The proposed repaired frame, in identical row and column order.
        changed_reasons: Non-empty reason tags keyed by changed persona ID.
        persona_id_column: Name of the unique identifier column.

    Returns:
        A dictionary with per-persona classifications, reasons, changed fields, and
        aggregate classification counts.

    Raises:
        ValueError: If frame identity/order, schema, or reason-ledger integrity fails.
    """
    if original.columns != repaired.columns:
        raise ValueError("Original and repaired frames must have identical columns")
    if persona_id_column not in original.columns:
        raise ValueError(f"Missing persona ID column: {persona_id_column}")
    original_ids = original.get_column(persona_id_column).to_list()
    repaired_ids = repaired.get_column(persona_id_column).to_list()
    if len(set(original_ids)) != len(original_ids) or any(value is None for value in original_ids):
        raise ValueError("Persona IDs must be unique and non-null")
    if original_ids != repaired_ids:
        raise ValueError("Original and repaired persona IDs must have identical order")

    before_rows = original.to_dicts()
    after_rows = repaired.to_dicts()
    rows_by_id: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    diffs: dict[str, list[str]] = {}
    for before, after in zip(before_rows, after_rows, strict=True):
        persona_id = str(before[persona_id_column])
        if persona_id in rows_by_id:
            raise ValueError("Persona IDs must remain unique after string conversion")
        rows_by_id[persona_id] = (before, after)
        fields = [
            field
            for field in original.columns
            if field != persona_id_column and before[field] != after[field]
        ]
        if fields:
            diffs[persona_id] = fields

    reason_ids = set(changed_reasons)
    diff_ids = set(diffs)
    if reason_ids != diff_ids:
        missing = sorted(diff_ids - reason_ids)
        extra = sorted(reason_ids - diff_ids)
        raise ValueError(
            f"Changed-reason ledger does not match frame diff (missing={missing}, extra={extra})"
        )
    for persona_id, tags in changed_reasons.items():
        if isinstance(tags, (str, bytes)) or not tags or any(
            not isinstance(tag, str) or not tag.strip() for tag in tags
        ):
            raise ValueError(f"Changed reasons must be non-empty string sequences: {persona_id}")

    results: dict[str, dict[str, object]] = {}
    counts = {
        "semantic_equivalence_reviewed": 0,
        "needs_prose_review_or_regeneration": 0,
    }
    for persona_id, fields in diffs.items():
        before, after = rows_by_id[persona_id]
        reasons = list(changed_reasons[persona_id])
        safe = all(_field_is_equivalent(field, before[field], after[field]) for field in fields)
        if any(tag in _UNSAFE_REASON_TAGS for tag in reasons):
            safe = False
        # Unknown reason tags are deliberately not evidence of safe equivalence.
        if any(tag not in _SAFE_REASON_TAGS and tag not in _UNSAFE_REASON_TAGS for tag in reasons):
            safe = False
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


# Recognised reason tags describe safe formatting-only repairs; any other tag fails closed.
_SAFE_REASON_TAGS = frozenset(
    {
        "metadata_only",
        "lexical_formatting_only",
        "list_deduplication_or_formatting",
    }
)


def _field_is_equivalent(field: str, before: object, after: object) -> bool:
    if field in _AUDIT_FIELDS:
        return True
    if field in _UNSAFE_FIELDS:
        return False
    if field in _LIST_FIELDS:
        before_items = _normalised_list(before)
        after_items = _normalised_list(after)
        return before_items is not None and before_items == after_items
    if field in _PROSE_FIELDS or "persona_text" in field or "prose" in field:
        return False
    if field in _LEXICAL_FIELDS and isinstance(before, str) and isinstance(after, str):
        before_identity = _lexical_identity(before)
        return bool(before_identity) and before_identity == _lexical_identity(after)
    return False


def _lexical_identity(value: str) -> str:
    """Normalise case and punctuation while retaining word identity and order."""
    translated = value.translate(str.maketrans("", "", string.punctuation))
    translated = re.sub(r"[^\w\s]", "", translated, flags=re.UNICODE)
    return " ".join(translated.casefold().split())


def _normalised_list(value: object) -> set[str] | None:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        return None
    normalised = {_lexical_identity(item) for item in value}
    if not normalised or "" in normalised:
        return None
    return normalised
