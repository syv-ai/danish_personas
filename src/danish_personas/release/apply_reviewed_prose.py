"""Build an offline provisional prose-preview frame from reviewed patches."""

from __future__ import annotations

import collections.abc as c
import re
import typing as t

import polars as pl

from ..io import canonical_json, sha256_text
from .prose_patch_verification import (
    ProsePatchVerificationError,
    verify_prose_patch_proposal,
)

_ID_FIELD = "persona_id"
_PROSE_FIELD = "persona"
_PREVIEW_LABEL = "PROVISIONAL PREVIEW"
_PATCHED_STATUS = "patched"
_SKIPPED_STATUSES = frozenset(
    {
        "unchanged_consistent",
        "needs_manual_review",
        "manual_review",
        "manual",
        "reject",
        "rejected",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_SHA_KEY_RE = re.compile(r"(^|_)(sha256|checksum)(_|$)")

JSONScalar: t.TypeAlias = str | int | float | bool | None
JSONValue: t.TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class ReviewedProseReport(t.TypedDict):
    """Aggregate preview report without raw prose or identifiers."""

    label: str
    release_ready: bool
    approves_proposals: bool
    row_count: int
    changed_rows: int
    unchanged_rows: int
    changed_fraction: float
    unresolved_count: int
    skipped_first_pass_rows: int
    skipped_second_pass_rows: int
    accepted_second_pass_rows: int
    source_metadata_sha256: str
    original_published_sha256: str
    v4_structured_sha256: str
    first_pass_records_sha256: str
    second_pass_records_sha256: str
    preview_sha256: str
    changed_id_hashes_sha256: str
    review_status_counts: dict[str, int]
    provisional_notice: str


def apply_reviewed_prose(
    *,
    original_published_frame: pl.DataFrame,
    v4_structured_frame: pl.DataFrame,
    first_pass_records: c.Sequence[c.Mapping[str, object]],
    second_pass_records: c.Sequence[c.Mapping[str, object]],
    source_metadata: c.Mapping[str, object],
    expected_row_count: int | None = None,
) -> tuple[pl.DataFrame, ReviewedProseReport]:
    """Apply independently accepted prose patches to a separate preview frame.

    The function is deliberately offline. It reads no checkpoint tree, calls no
    provider, mutates neither published input frame, and does not approve any
    proposal for release. Only rows with a patched first-pass proposal and an
    accepted second-pass record linked to that exact first checkpoint can replace
    ``persona``; all other fields are copied from ``v4_structured_frame``.

    Args:
        original_published_frame:
            Published baseline frame containing the original Mistral prose.
        v4_structured_frame:
            Immutable v4 structured frame whose non-prose columns are preserved.
        first_pass_records:
            Verified first-pass proposal records. Patched records must include
            ``persona_id`` or ``persona_hash``, ``proposed_text``, ``patches``,
            ``changed_facts``, and ``checkpoint_sha256``.
        second_pass_records:
            Independent verification records. Accepted records must link back via
            ``original_checkpoint_sha256`` and include the review evidence needed
            for local revalidation.
        source_metadata:
            Immutable source and manifest pins. At least one SHA-256/checksum key
            must be present and all such values must be SHA-256 digests.
        expected_row_count (optional):
            Required row count. Defaults to no fixed-size check.

    Returns:
        A separate provisional preview frame and aggregate report.
    """
    _validate_source_metadata(metadata=source_metadata)
    original_rows, ordered_ids = _validated_rows(
        frame=original_published_frame,
        label="original published frame",
        expected_row_count=expected_row_count,
    )
    _validate_v4_frame(
        frame=v4_structured_frame,
        ordered_ids=ordered_ids,
        expected_row_count=expected_row_count,
    )

    first_index, skipped_first, first_status_counts = _index_first_pass_records(
        records=first_pass_records, ordered_ids=ordered_ids
    )
    second_index, skipped_second, second_status_counts = _index_second_pass_records(
        records=second_pass_records, ordered_ids=ordered_ids
    )
    status_counts = _merge_counts(first_status_counts, second_status_counts)

    preview_rows = v4_structured_frame.to_dicts()
    changed_ids: list[str] = []
    for row in preview_rows:
        persona_id = str(row[_ID_FIELD])
        first = first_index.get(persona_id)
        second = second_index.get(persona_id)
        if first is None or second is None:
            row[_PROSE_FIELD] = original_rows[persona_id][_PROSE_FIELD]
            continue
        proposed_text = _accepted_proposed_text(
            persona_id=persona_id,
            first=first,
            second=second,
            original_text=str(original_rows[persona_id][_PROSE_FIELD]),
        )
        row[_PROSE_FIELD] = proposed_text
        changed_ids.append(persona_id)

    preview = pl.DataFrame(preview_rows, schema=v4_structured_frame.schema)
    _assert_only_persona_differs(preview=preview, v4_structured=v4_structured_frame)
    report = _report(
        original=original_published_frame,
        v4_structured=v4_structured_frame,
        first_pass_records=first_pass_records,
        second_pass_records=second_pass_records,
        source_metadata=source_metadata,
        preview=preview,
        row_count=len(ordered_ids),
        changed_ids=changed_ids,
        skipped_first=skipped_first,
        skipped_second=skipped_second,
        status_counts=status_counts,
    )
    return preview, report


def _validated_rows(
    *, frame: pl.DataFrame, label: str, expected_row_count: int | None
) -> tuple[dict[str, dict[str, object]], list[str]]:
    missing = {_ID_FIELD, _PROSE_FIELD} - set(frame.columns)
    if missing:
        raise ValueError(f"{label} lacks required columns: {sorted(missing)}")
    if expected_row_count is not None and frame.height != expected_row_count:
        raise ValueError(f"{label} does not contain the expected row count")
    rows: dict[str, dict[str, object]] = {}
    ordered_ids: list[str] = []
    for row in frame.to_dicts():
        persona_id = row.get(_ID_FIELD)
        persona = row.get(_PROSE_FIELD)
        if not isinstance(persona_id, str) or not persona_id:
            raise ValueError(f"{label} contains a missing persona ID")
        if persona_id in rows:
            raise ValueError(f"{label} contains duplicate persona IDs")
        if not isinstance(persona, str) or not persona.strip():
            raise ValueError(f"{label} contains missing or blank prose")
        rows[persona_id] = row
        ordered_ids.append(persona_id)
    return rows, ordered_ids


def _validate_v4_frame(
    *, frame: pl.DataFrame, ordered_ids: list[str], expected_row_count: int | None
) -> None:
    _, v4_ids = _validated_rows(
        frame=frame, label="v4 structured frame", expected_row_count=expected_row_count
    )
    if v4_ids != ordered_ids:
        raise ValueError("v4 structured frame must preserve ordered published IDs")


def _index_first_pass_records(
    *, records: c.Sequence[c.Mapping[str, object]], ordered_ids: list[str]
) -> tuple[dict[str, c.Mapping[str, object]], int, dict[str, int]]:
    known_ids = set(ordered_ids)
    indexed: dict[str, c.Mapping[str, object]] = {}
    skipped = 0
    status_counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    checkpoint_shas: set[str] = set()
    for record in records:
        persona_id = _record_persona_id(record=record, known_ids=known_ids)
        if persona_id in seen_ids:
            raise ValueError("First-pass records contain duplicate persona IDs")
        seen_ids.add(persona_id)
        _validate_record_hashes(record=record)
        status = _record_status(record=record)
        if status is not None:
            status_counts[status] = status_counts.get(status, 0) + 1
        if status in _SKIPPED_STATUSES:
            skipped += 1
            continue
        if status not in {None, _PATCHED_STATUS}:
            raise ValueError("First-pass record has an unsupported status")
        checkpoint_sha = _required_sha(
            record=record, keys=("checkpoint_sha256", "first_checkpoint_sha256")
        )
        if checkpoint_sha in checkpoint_shas:
            raise ValueError("First-pass records contain duplicate checkpoint hashes")
        checkpoint_shas.add(checkpoint_sha)
        _require_string(record=record, keys=("proposed_text", "proposed_persona"))
        _require_sequence(record=record, key="patches")
        _require_mapping(record=record, key="changed_facts")
        indexed[persona_id] = record
    return indexed, skipped, status_counts


def _index_second_pass_records(
    *, records: c.Sequence[c.Mapping[str, object]], ordered_ids: list[str]
) -> tuple[dict[str, c.Mapping[str, object]], int, dict[str, int]]:
    known_ids = set(ordered_ids)
    indexed: dict[str, c.Mapping[str, object]] = {}
    skipped = 0
    status_counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    link_shas: set[str] = set()
    for record in records:
        persona_id = _record_persona_id(record=record, known_ids=known_ids)
        if persona_id in seen_ids:
            raise ValueError("Second-pass records contain duplicate persona IDs")
        seen_ids.add(persona_id)
        _validate_record_hashes(record=record)
        status = _record_status(record=record)
        if status is not None:
            status_counts[status] = status_counts.get(status, 0) + 1
        accepted = record.get("accepted")
        if accepted is not True:
            skipped += 1
            continue
        link_sha = _required_sha(record=record, keys=("original_checkpoint_sha256",))
        if link_sha in link_shas:
            raise ValueError("Second-pass records contain duplicate checkpoint links")
        link_shas.add(link_sha)
        indexed[persona_id] = record
    return indexed, skipped, status_counts


def _accepted_proposed_text(
    *,
    persona_id: str,
    first: c.Mapping[str, object],
    second: c.Mapping[str, object],
    original_text: str,
) -> str:
    first_sha = _required_sha(
        record=first, keys=("checkpoint_sha256", "first_checkpoint_sha256")
    )
    second_link = _required_sha(record=second, keys=("original_checkpoint_sha256",))
    if first_sha != second_link:
        raise ValueError("Second-pass record links to a different first checkpoint")
    if second.get("accepted") is not True:
        raise ValueError("Second-pass record is not accepted")
    proposed_text = _record_text(
        record=first, keys=("proposed_text", "proposed_persona")
    )
    if not proposed_text.strip():
        raise ValueError("Accepted prose proposal is blank")
    for key in ("proposed_text_sha256", "proposed_persona_sha256"):
        sha_value = first.get(key)
        if isinstance(sha_value, str) and sha_value.lower() != sha256_text(
            proposed_text
        ):
            raise ValueError("First-pass proposed-text hash is stale")
    for key in ("original_persona_sha256", "original_text_sha256"):
        sha_value = first.get(key)
        if isinstance(sha_value, str) and sha_value.lower() != sha256_text(
            original_text
        ):
            raise ValueError("First-pass original-prose hash is stale")
    try:
        result = verify_prose_patch_proposal(
            original_text=original_text,
            changed_facts=_changed_facts(record=first),
            proposed_text=proposed_text,
            patches=_patches(record=first),
            second_review=_second_review(record=second),
            original_checkpoint_sha256=first_sha,
        )
    except ProsePatchVerificationError as exc:
        message = f"Accepted prose patch failed local review: {persona_id}"
        raise ValueError(message) from exc
    if not result.accepted:
        raise ValueError("Accepted second-pass record was not locally accepted")
    return proposed_text


def _record_persona_id(*, record: c.Mapping[str, object], known_ids: set[str]) -> str:
    raw_id = record.get(_ID_FIELD) or record.get("id")
    raw_hash = record.get("persona_hash")
    if isinstance(raw_id, str) and raw_id:
        persona_id = raw_id
    elif isinstance(raw_hash, str):
        matches = [
            persona_id
            for persona_id in known_ids
            if sha256_text(persona_id) == raw_hash
        ]
        if len(matches) != 1:
            raise ValueError("Record persona hash does not identify one published row")
        persona_id = matches[0]
    else:
        raise ValueError("Record is missing a persona ID")
    if persona_id not in known_ids:
        raise ValueError("Record references an unknown persona ID")
    if isinstance(raw_hash, str) and raw_hash.lower() != sha256_text(persona_id):
        raise ValueError("Record persona hash does not match its persona ID")
    return persona_id


def _record_status(*, record: c.Mapping[str, object]) -> str | None:
    status = (
        record.get("disposition") or record.get("status") or record.get("review_status")
    )
    if status is None:
        return None
    if not isinstance(status, str) or not status:
        raise ValueError("Record status must be a non-empty string")
    return status


def _validate_record_hashes(*, record: c.Mapping[str, object]) -> None:
    for key, value in record.items():
        if _SHA_KEY_RE.search(key) and isinstance(value, str):
            _validate_sha(value=value, label=key)


def _required_sha(*, record: c.Mapping[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return _validate_sha(value=value, label=key)
    raise ValueError(f"Record is missing required SHA-256 field: {keys[0]}")


def _validate_sha(*, value: str, label: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value.lower()


def _require_string(*, record: c.Mapping[str, object], keys: tuple[str, ...]) -> None:
    _record_text(record=record, keys=keys)


def _record_text(*, record: c.Mapping[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value
    raise ValueError(f"Record is missing required text field: {keys[0]}")


def _require_sequence(*, record: c.Mapping[str, object], key: str) -> None:
    value = record.get(key)
    if isinstance(value, str) or not isinstance(value, c.Sequence):
        raise ValueError(f"Record is missing required sequence field: {key}")


def _require_mapping(*, record: c.Mapping[str, object], key: str) -> None:
    value = record.get(key)
    if not isinstance(value, c.Mapping):
        raise ValueError(f"Record is missing required mapping field: {key}")


def _changed_facts(
    *, record: c.Mapping[str, object]
) -> c.Mapping[str, c.Mapping[str, object]]:
    value = record.get("changed_facts")
    if not isinstance(value, c.Mapping):
        raise ValueError("First-pass record is missing changed facts")
    changed: dict[str, c.Mapping[str, object]] = {}
    for key, fact in value.items():
        if not isinstance(key, str) or not isinstance(fact, c.Mapping):
            raise ValueError("First-pass changed facts are malformed")
        changed[key] = fact
    return changed


def _patches(*, record: c.Mapping[str, object]) -> list[c.Mapping[str, object]]:
    value = record.get("patches")
    if isinstance(value, str) or not isinstance(value, c.Sequence):
        raise ValueError("First-pass record is missing patches")
    patches: list[c.Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, c.Mapping):
            raise ValueError("First-pass patch is malformed")
        patches.append(item)
    return patches


def _second_review(*, record: c.Mapping[str, object]) -> dict[str, object]:
    verdict = record.get("review_verdict") or record.get("verdict")
    if verdict != "accept":
        raise ValueError("Accepted second-pass record must carry an accept verdict")
    reasons = record.get("reasons")
    evidence = record.get("fact_evidence")
    if not isinstance(reasons, list) or not isinstance(evidence, list):
        raise ValueError("Accepted second-pass record lacks review evidence")
    return {"verdict": verdict, "reasons": reasons, "fact_evidence": evidence}


def _validate_source_metadata(*, metadata: c.Mapping[str, object]) -> None:
    if not metadata:
        raise ValueError("Source metadata must contain immutable SHA-bound pins")
    sha_values = _metadata_sha_values(value=metadata)
    if not sha_values:
        raise ValueError("Source metadata must contain at least one SHA-256 pin")
    for label, value in sha_values:
        _validate_sha(value=value, label=label)


def _metadata_sha_values(
    *, value: object, prefix: str = "metadata"
) -> list[tuple[str, str]]:
    if isinstance(value, c.Mapping):
        pairs: list[tuple[str, str]] = []
        for key, child in value.items():
            key_text = str(key)
            label = f"{prefix}.{key_text}"
            if _SHA_KEY_RE.search(key_text) and isinstance(child, str):
                pairs.append((label, child))
            pairs.extend(_metadata_sha_values(value=child, prefix=label))
        return pairs
    if isinstance(value, list):
        pairs = []
        for index, child in enumerate(value):
            pairs.extend(_metadata_sha_values(value=child, prefix=f"{prefix}[{index}]"))
        return pairs
    return []


def _assert_only_persona_differs(
    *, preview: pl.DataFrame, v4_structured: pl.DataFrame
) -> None:
    for column in v4_structured.columns:
        if column == _PROSE_FIELD:
            continue
        preview_values = preview.get_column(column).to_list()
        v4_values = v4_structured.get_column(column).to_list()
        if preview_values != v4_values:
            raise ValueError("Preview changed a non-prose column")


def _report(
    *,
    original: pl.DataFrame,
    v4_structured: pl.DataFrame,
    first_pass_records: c.Sequence[c.Mapping[str, object]],
    second_pass_records: c.Sequence[c.Mapping[str, object]],
    source_metadata: c.Mapping[str, object],
    preview: pl.DataFrame,
    row_count: int,
    changed_ids: list[str],
    skipped_first: int,
    skipped_second: int,
    status_counts: dict[str, int],
) -> ReviewedProseReport:
    changed_hashes = sorted(sha256_text(persona_id) for persona_id in changed_ids)
    report_status_counts = (
        _metadata_status_counts(metadata=source_metadata) or status_counts
    )
    unresolved_count = _unresolved_count(
        skipped_first=skipped_first,
        skipped_second=skipped_second,
        status_counts=report_status_counts,
    )
    return {
        "label": _PREVIEW_LABEL,
        "release_ready": False,
        "approves_proposals": False,
        "row_count": row_count,
        "changed_rows": len(changed_ids),
        "unchanged_rows": row_count - len(changed_ids),
        "changed_fraction": len(changed_ids) / row_count if row_count else 0.0,
        "unresolved_count": unresolved_count,
        "skipped_first_pass_rows": skipped_first,
        "skipped_second_pass_rows": skipped_second,
        "accepted_second_pass_rows": len(changed_ids),
        "source_metadata_sha256": _hash_json(_jsonable(source_metadata)),
        "original_published_sha256": _frame_hash(frame=original),
        "v4_structured_sha256": _frame_hash(frame=v4_structured),
        "first_pass_records_sha256": _hash_json(_jsonable(first_pass_records)),
        "second_pass_records_sha256": _hash_json(_jsonable(second_pass_records)),
        "preview_sha256": _frame_hash(frame=preview),
        "changed_id_hashes_sha256": _hash_json(changed_hashes),
        "review_status_counts": dict(sorted(report_status_counts.items())),
        "provisional_notice": (
            "Provisional preview only: not release-ready and not an approval of "
            "the reviewed proposals."
        ),
    }


def _metadata_status_counts(*, metadata: c.Mapping[str, object]) -> dict[str, int]:
    raw_status = metadata.get("review_status") or metadata.get("status")
    if not isinstance(raw_status, c.Mapping):
        return {}
    counts: dict[str, int] = {}
    for key, value in raw_status.items():
        if isinstance(key, str) and isinstance(value, int) and value >= 0:
            counts[key] = value
    return counts


def _unresolved_count(
    *, skipped_first: int, skipped_second: int, status_counts: dict[str, int]
) -> int:
    unresolved_statuses = {
        "manual",
        "needs_manual_review",
        "rejected",
        "reject",
        "failed",
        "pending",
        "unchanged_consistent",
    }
    from_status = sum(
        count
        for status, count in status_counts.items()
        if status in unresolved_statuses
    )
    return from_status or skipped_first + skipped_second


def _merge_counts(*counts: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for mapping in counts:
        for key, value in mapping.items():
            merged[key] = merged.get(key, 0) + value
    return merged


def _frame_hash(*, frame: pl.DataFrame) -> str:
    return _hash_json({"columns": frame.columns, "rows": _jsonable(frame.to_dicts())})


def _hash_json(value: object) -> str:
    return sha256_text(canonical_json(value))


def _jsonable(value: object) -> JSONValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, c.Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, c.Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(child) for child in value]
    return str(value)
