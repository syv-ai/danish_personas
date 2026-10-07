"""Deterministic offline completion of generated list attributes."""

from __future__ import annotations

import hashlib
import random
import typing as t

import polars as pl

ListCompletionField: t.TypeAlias = t.Literal[
    "skills_and_expertise", "hobbies_and_interests"
]

BANK_VERSION = "neutral-da-list-bank-v1"
_METHOD = "synthetic_list_completion_v1_seeded_by_persona_id_and_field"
_LIST_FIELDS: tuple[ListCompletionField, ...] = (
    "skills_and_expertise",
    "hobbies_and_interests",
)
_HOBBY_PUNCTUATION_SUFFIX = ".!?;:,"
_BANKS: dict[ListCompletionField, tuple[str, ...]] = {
    "skills_and_expertise": (
        "analyse",
        "dokumentation",
        "formidling",
        "koordinering",
        "kvalitetssikring",
        "organisering",
        "planlægning",
        "problemløsning",
        "procesforståelse",
        "samarbejde",
        "serviceorientering",
        "strukturering",
        "tidsstyring",
        "videndeling",
        "opfølgning",
        "dataoverblik",
    ),
    "hobbies_and_interests": (
        "bagning",
        "brætspil",
        "cykling",
        "film",
        "fotografering",
        "gåture",
        "havearbejde",
        "keramik",
        "krydsord",
        "læsning",
        "madlavning",
        "musik",
        "naturture",
        "podcasts",
        "svømning",
        "tegning",
    ),
}


class ListCompletionChange(t.TypedDict):
    """One synthetic list-field completion."""

    persona_id: str
    field: ListCompletionField
    original: list[str]
    new: list[str]
    seed: int
    method: str
    bank_version: str


class ListCompletionProseReview(t.TypedDict):
    """Persona prose review marker for rows whose generated lists changed."""

    persona_id: str
    fields: list[ListCompletionField]
    reason: str
    method: str


class ListCompletionReport(t.TypedDict):
    """Change ledger and prose-review markers without source sidecars."""

    bank_version: str
    changes: list[ListCompletionChange]
    prose_review: list[ListCompletionProseReview]


def complete_generated_lists(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame, ListCompletionReport]:
    """Complete invalid generated list fields without touching source fields or prose.

    The completion is deliberately conservative: it only inspects ``persona_id`` and
    the two generated list fields, leaves fields without objective generated-check
    failures byte-equivalent, drops only invalid or duplicate items, and fills back to
    the original size with neutral Danish synthetic items from a versioned bank. It
    does not call a provider and does not infer from demographic, geographic, origin,
    religion, or status columns.

    Args:
        frame:
            Release candidate frame with unique persona IDs and generated lists.

    Returns:
        A separate frame with completed lists and a report containing changes plus
        prose-review markers for rows whose lists changed.
    """
    _validate_required_columns(frame=frame)
    _validate_persona_ids(frame=frame)

    rows = frame.to_dicts()
    changes: list[ListCompletionChange] = []
    changed_fields_by_persona: dict[str, list[ListCompletionField]] = {}

    for row in rows:
        persona_id = _read_persona_id(row=row)
        for field in _LIST_FIELDS:
            original = _read_list(row=row, field=field, persona_id=persona_id)
            completed = _complete_list(
                persona_id=persona_id, field=field, values=original
            )
            if completed == original:
                continue
            seed = _seed_for(persona_id=persona_id, field=field)
            row[field] = completed
            changes.append(
                {
                    "persona_id": persona_id,
                    "field": field,
                    "original": original,
                    "new": completed,
                    "seed": seed,
                    "method": _METHOD,
                    "bank_version": BANK_VERSION,
                }
            )
            changed_fields_by_persona.setdefault(persona_id, []).append(field)

    report: ListCompletionReport = {
        "bank_version": BANK_VERSION,
        "changes": changes,
        "prose_review": [
            {
                "persona_id": persona_id,
                "fields": fields,
                "reason": "persona_prose_preserved_after_synthetic_list_completion",
                "method": _METHOD,
            }
            for persona_id, fields in changed_fields_by_persona.items()
        ],
    }
    return pl.DataFrame(rows, schema=frame.schema), report


def _complete_list(
    *, persona_id: str, field: ListCompletionField, values: list[str]
) -> list[str]:
    if not _fails_objective_generated_check(field=field, values=values):
        return values

    retained = _retained_items(field=field, values=values)
    target_size = len(values)
    additions = _synthetic_additions(
        persona_id=persona_id,
        field=field,
        retained=retained,
        count=target_size - len(retained),
    )
    return [*retained, *additions]


def _fails_objective_generated_check(
    *, field: ListCompletionField, values: list[str]
) -> bool:
    if any(not value.strip() for value in values):
        return True
    if len({_item_key(value=value) for value in values}) != len(values):
        return True
    return field == "hobbies_and_interests" and any(
        _fails_hobby_format(value=value) for value in values
    )


def _fails_hobby_format(*, value: str) -> bool:
    return bool(
        value and (value != value.lower() or value[-1] in _HOBBY_PUNCTUATION_SUFFIX)
    )


def _item_key(*, value: str) -> str:
    return value.casefold()


def _retained_items(*, field: ListCompletionField, values: list[str]) -> list[str]:
    retained: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value.strip():
            continue
        retained_value = (
            _repair_hobby_item(value=value)
            if field == "hobbies_and_interests"
            else value
        )
        if not retained_value.strip():
            continue
        key = _item_key(value=retained_value)
        if key in seen:
            continue
        retained.append(retained_value)
        seen.add(key)
    return retained


def _repair_hobby_item(*, value: str) -> str:
    if not _fails_hobby_format(value=value):
        return value
    return value.lower().rstrip(_HOBBY_PUNCTUATION_SUFFIX)


def _synthetic_additions(
    *, persona_id: str, field: ListCompletionField, retained: list[str], count: int
) -> list[str]:
    retained_keys = {_item_key(value=value) for value in retained}
    seed = _seed_for(persona_id=persona_id, field=field)
    candidates = list(_BANKS[field])
    random.Random(seed).shuffle(candidates)
    additions = [
        candidate
        for candidate in candidates
        if _item_key(value=candidate) not in retained_keys
    ]
    if len(additions) < count:
        raise ValueError(f"Synthetic bank {BANK_VERSION} cannot complete {field}")
    return additions[:count]


def _seed_for(*, persona_id: str, field: ListCompletionField) -> int:
    payload = f"{BANK_VERSION}|{persona_id}|{field}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def _read_list(
    *, row: dict[str, object], field: ListCompletionField, persona_id: str
) -> list[str]:
    values = row[field]
    if not isinstance(values, list):
        raise ValueError(f"{field} for {persona_id} must be a list")
    if not 3 <= len(values) <= 6:
        raise ValueError(f"{field} for {persona_id} must contain 3 to 6 items")
    if any(not isinstance(value, str) for value in values):
        raise ValueError(f"{field} for {persona_id} must contain string items")
    return values.copy()


def _read_persona_id(*, row: dict[str, object]) -> str:
    persona_id = row["persona_id"]
    if not isinstance(persona_id, str):
        raise ValueError("Persona IDs must be non-empty strings")
    return persona_id


def _validate_persona_ids(*, frame: pl.DataFrame) -> None:
    persona_ids = frame.get_column("persona_id").to_list()
    invalid_ids = (
        not isinstance(persona_id, str) or not persona_id.strip()
        for persona_id in persona_ids
    )
    if any(invalid_ids):
        raise ValueError("Persona IDs must be non-empty strings")
    if len(set(persona_ids)) != len(persona_ids):
        raise ValueError("Persona IDs must be unique")


def _validate_required_columns(*, frame: pl.DataFrame) -> None:
    required = {"persona_id", *_LIST_FIELDS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing fields for list completion: {sorted(missing)}")
