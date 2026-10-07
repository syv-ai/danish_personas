"""Offline, deterministic repairs for objectively repairable generated fields."""

from __future__ import annotations

import hashlib
import typing as t

import polars as pl

from ..generation.job_titles import JobFunctionTitleMapping
from ..models import ELIGIBLE_JOB_FUNCTION_STATUS_CODES

_GENERIC_SKILLS = (
    "samarbejde",
    "kommunikation",
    "planlægning",
    "problemløsning",
    "ansvarlighed",
    "struktur",
    "videndeling",
    "kvalitetssans",
    "fleksibilitet",
    "selvstændighed",
)
_GENERIC_INTERESTS = (
    "gåture",
    "madlavning",
    "musik",
    "læsning",
    "natur",
    "havearbejde",
    "brætspil",
    "fotografering",
    "cykling",
    "svømning",
)


class GeneratedRepairReport(t.TypedDict):
    """Field-level changes and issues intentionally left unresolved."""

    changed: dict[str, list[str]]
    prose_regeneration: dict[str, list[str]]
    unresolved: dict[str, list[str]]


def repair_generated_attributes(
    frame: pl.DataFrame,
    job_title_mapping: JobFunctionTitleMapping,
) -> tuple[pl.DataFrame, GeneratedRepairReport]:
    """Repair generated attributes without changing demographics or persona prose.

    Repairs are deterministic and retain every row. Any changed field that may
    invalidate the prose is separately listed for regeneration. Same-sex target
    conformance is never inferred from observed genders; its IDs remain unresolved.
    """
    required = {
        "persona_id", "marital_status", "legal_status_detail",
        "current_relationship_status", "partner_gender", "job_function_code",
        "job_title", "skills_and_expertise", "hobbies_and_interests",
    }
    absent = required - set(frame.columns)
    if absent:
        message = f"Generated frame is missing required columns: {sorted(absent)}"
        raise ValueError(message)

    rows = frame.to_dicts()
    changed: dict[str, list[str]] = {}
    unresolved: dict[str, list[str]] = {}
    prose_fields = {
        "legal_status_detail", "partner_gender", "job_title",
        "skills_and_expertise", "hobbies_and_interests",
    }
    for row in rows:
        persona_id = str(row["persona_id"])
        fields: list[str] = []
        marital = row["marital_status"]
        detail = row["legal_status_detail"]
        relationship = row["current_relationship_status"]
        if marital != "married_or_separated" and detail is not None:
            row["legal_status_detail"] = None
            fields.append("legal_status_detail")
        elif (
            marital == "married_or_separated"
            and detail == "married"
            and relationship == "not_partnered"
        ):
            row["legal_status_detail"] = "separated"
            fields.append("legal_status_detail")

        if relationship == "not_partnered" and row["partner_gender"] is not None:
            row["partner_gender"] = None
            fields.append("partner_gender")
        elif relationship == "partnered" and expected_gender is None:
            unresolved.setdefault(persona_id, []).append("partner_gender")

        code = row["job_function_code"]
        eligible = (
            code in ELIGIBLE_JOB_FUNCTION_STATUS_CODES
            and code in job_title_mapping.job_functions
        )
        title = row["job_title"]
        if not eligible:
            if title is not None:
                row["job_title"] = None
                fields.append("job_title")
        else:
            allowed = job_title_mapping.job_functions[code].titles
            by_casefold = {value.casefold(): value for value in allowed}
            if isinstance(title, str) and title.casefold() in by_casefold:
                canonical = by_casefold[title.casefold()]
                if title != canonical:
                    row["job_title"] = canonical
                    fields.append("job_title")
            else:
                index = int.from_bytes(
                    hashlib.sha256(persona_id.encode("utf-8")).digest()[:8], "big"
                ) % len(allowed)
                replacement = allowed[index]
                if title != replacement:
                    row["job_title"] = replacement
                    fields.append("job_title")

        _repair_list(
            row, "skills_and_expertise", _GENERIC_SKILLS, fields, unresolved, persona_id
        )
        _repair_list(
            row,
            "hobbies_and_interests",
            _GENERIC_INTERESTS,
            fields,
            unresolved,
            persona_id,
        )
        if fields:
            changed[persona_id] = sorted(set(fields))

    same_sex_ids = [
        str(row["persona_id"])
        for row in rows
        if row.get("current_relationship_status") == "partnered"
        and row.get("same_sex_partner_target") is not None
    ]
    for persona_id in same_sex_ids:
        unresolved.setdefault(persona_id, []).append("same_sex_partner_target")

    report: GeneratedRepairReport = {
        "changed": dict(sorted(changed.items())),
        "prose_regeneration": {
            persona_id: fields
            for persona_id, fields in sorted(changed.items())
            if set(fields) & prose_fields
        },
        "unresolved": {
            persona_id: sorted(set(fields))
            for persona_id, fields in sorted(unresolved.items())
        },
    }
    return pl.DataFrame(rows), report


def _repair_list(
    row: dict[str, object],
    field: str,
    vocabulary: tuple[str, ...],
    changed: list[str],
    unresolved: dict[str, list[str]],
    persona_id: str,
) -> None:
    values = row[field]
    if not isinstance(values, list):
        unresolved.setdefault(persona_id, []).append(field)
        return
    cleaned = [
        value.rstrip(".!?;:… ").lower() if isinstance(value, str) else value
        for value in values
    ]
    seen: set[str] = set()
    duplicates: list[int] = []
    for index, value in enumerate(cleaned):
        key = value.casefold() if isinstance(value, str) else repr(value)
        if key in seen:
            duplicates.append(index)
        seen.add(key)
    used = {value.casefold() for value in cleaned if isinstance(value, str)}
    candidates = [value for value in vocabulary if value.casefold() not in used]
    if len(candidates) < len(duplicates):
        unresolved.setdefault(persona_id, []).append(field)
        return
    for index, replacement in zip(duplicates, candidates, strict=False):
        cleaned[index] = replacement
    if cleaned != values:
        row[field] = cleaned
        changed.append(field)
