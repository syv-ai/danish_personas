"""Minimal offline repairs of generated attributes, without rewriting prose."""

from __future__ import annotations

import hashlib
import typing as t

import polars as pl

from ..generation.job_titles import JobFunctionTitleMapping
from ..models import ELIGIBLE_JOB_FUNCTION_STATUS_CODES


class GeneratedRepairReport(t.TypedDict):
    """Changed fields and rows still requiring model-generated repair or review."""

    changed: dict[str, list[str]]
    prose_regeneration: dict[str, list[str]]
    unresolved: dict[str, list[str]]


def repair_generated_attributes(
    frame: pl.DataFrame, job_title_mapping: JobFunctionTitleMapping
) -> tuple[pl.DataFrame, GeneratedRepairReport]:
    """Correct unambiguous generated fields, preserving every sampled input.

    Duplicate list items are removed only when the resulting list remains valid under
    the schema. Invalid or undersized lists remain flagged for regeneration. No persona
    text is silently edited.

    Args:
        frame:
            Release candidate with generated fields and original row order.
        job_title_mapping:
            Reviewed allowlists bound to the generation campaign.

    Returns:
        A separate repaired frame and per-person change/unresolved ledger.

    Raises:
        ValueError:
            If fields needed for safe repair are absent or persona IDs repeat.
    """
    required = {
        "persona_id",
        "marital_status",
        "legal_status_detail",
        "current_relationship_status",
        "partner_gender",
        "detailed_status_code",
        "job_function_code",
        "job_title",
        "skills_and_expertise",
        "hobbies_and_interests",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing fields for generated repair: {sorted(missing)}")
    if frame.get_column("persona_id").n_unique() != frame.height:
        raise ValueError("Persona IDs must be unique")

    changed: dict[str, list[str]] = {}
    unresolved: dict[str, list[str]] = {}
    rows = frame.to_dicts()
    for row in rows:
        persona_id = str(row["persona_id"])
        fields: list[str] = []
        pending: list[str] = []
        _repair_relationship(row=row, fields=fields, pending=pending)
        _repair_title(
            row=row,
            mapping=job_title_mapping,
            persona_id=persona_id,
            fields=fields,
            pending=pending,
        )
        _repair_lists(row=row, fields=fields, pending=pending)
        _check_residual_legal_status(row=row, pending=pending)
        if fields:
            changed[persona_id] = sorted(set(fields))
        if pending:
            unresolved[persona_id] = sorted(set(pending))
    report: GeneratedRepairReport = {
        "changed": changed,
        "prose_regeneration": {
            persona_id: fields for persona_id, fields in changed.items()
        },
        "unresolved": unresolved,
    }
    return pl.DataFrame(rows, schema=frame.schema), report


def _check_residual_legal_status(*, row: dict[str, object], pending: list[str]) -> None:
    """Flag invalid legal-status combinations after all deterministic repairs."""
    marital = row["marital_status"]
    detail = row["legal_status_detail"]
    if marital == "married_or_separated" and detail not in {"married", "separated"}:
        reason = (
            "legal_status_detail: missing or invalid for married_or_separated; "
            "marital status cannot determine married versus separated"
        )
        if reason not in pending:
            pending.append(reason)
    elif marital != "married_or_separated" and detail is not None:
        pending.append("legal_status_detail: must be null outside married_or_separated")
    elif detail == "married" and row["current_relationship_status"] != "partnered":
        pending.append("legal_status_detail: married requires partnered relationship")


def _repair_lists(
    *, row: dict[str, object], fields: list[str], pending: list[str]
) -> None:
    """Normalise interests and remove duplicates only when the schema remains valid."""
    for field in ("skills_and_expertise", "hobbies_and_interests"):
        values = row[field]
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            pending.append(field)
            continue
        cleaned = (
            [value.strip().rstrip(".!?;:,… ").lower() for value in values]
            if field == "hobbies_and_interests"
            else values
        )
        unique: list[str] = []
        seen: set[str] = set()
        for value in cleaned:
            key = value.strip().rstrip(".!?;:,… ").casefold()
            if key not in seen:
                unique.append(value)
                seen.add(key)

        # Generation schema requires 3–6 non-empty string items. Do not drop
        # duplicates if doing so would leave too few items.
        valid = 3 <= len(unique) <= 6 and all(
            isinstance(value, str) and value.strip() for value in unique
        )
        if not valid:
            if cleaned != values:
                row[field] = cleaned
                fields.append(field)
            pending.append(field)
            continue
        if unique != values:
            row[field] = unique
            fields.append(field)
        if len({value.casefold() for value in unique}) != len(unique):
            pending.append(field)


def _repair_relationship(
    *, row: dict[str, object], fields: list[str], pending: list[str]
) -> None:
    """Apply source-marital precedence without inventing relationship histories."""
    marital = row["marital_status"]
    detail = row["legal_status_detail"]
    relationship = row["current_relationship_status"]
    if marital != "married_or_separated" and detail is not None:
        row["legal_status_detail"] = None
        fields.append("legal_status_detail")
    elif marital == "married_or_separated":
        if detail not in {"married", "separated"}:
            pending.append(
                "legal_status_detail: missing or invalid for married_or_separated; "
                "marital status cannot determine married versus separated"
            )
        elif detail == "married" and relationship == "not_partnered":
            row["legal_status_detail"] = "separated"
            fields.append("legal_status_detail")
    if relationship == "not_partnered" and row["partner_gender"] is not None:
        row["partner_gender"] = None
        fields.append("partner_gender")
    elif relationship == "partnered" and row["partner_gender"] is None:
        pending.append("partner_gender")


def _repair_title(
    *,
    row: dict[str, object],
    mapping: JobFunctionTitleMapping,
    persona_id: str,
    fields: list[str],
    pending: list[str],
) -> None:
    """Keep reviewed titles and choose a stable title only when eligibility allows."""
    eligible = row["detailed_status_code"] in ELIGIBLE_JOB_FUNCTION_STATUS_CODES
    code = row["job_function_code"]
    title = row["job_title"]
    if not eligible:
        if title is not None:
            row["job_title"] = None
            fields.append("job_title")
        return
    if not isinstance(code, str) or code not in mapping.job_functions:
        pending.append("job_function_code")
        return
    allowed = mapping.job_functions[code].titles
    if isinstance(title, str):
        canonical = next(
            (
                candidate
                for candidate in allowed
                if title.casefold() == candidate.casefold()
            ),
            None,
        )
        if canonical is not None:
            if title != canonical:
                row["job_title"] = canonical
                fields.append("job_title")
            return
    draw = hashlib.sha256(
        f"danish-personas/title-repair\0{persona_id}".encode("utf-8")
    ).digest()
    row["job_title"] = allowed[int.from_bytes(draw[:8], "big") % len(allowed)]
    fields.append("job_title")
