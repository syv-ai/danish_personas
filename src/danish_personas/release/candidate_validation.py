"""Offline release-candidate diagnostics against a published baseline.

These diagnostics are deliberately preliminary. They check mechanical invariants,
source support, generated-field contracts, and candidate-vs-original marginal drift;
they do not certify source-distribution quality, prose semantics, or identity privacy.
"""

from __future__ import annotations

import json
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..io import sha256_file, sha256_text
from .foreign_student_repair import count_editorial_foreign_student_violations
from .generated_checks import check_generated_fields
from .source_repair import assess_release_support

EXPECTED_RELEASE_ROWS = 100_000
REPORT_LABEL = "preliminary_release_candidate_diagnostics"
_PRELIMINARY_NOTICE = (
    "Preliminary only until full prose, semantic, and identity privacy gates pass."
)
_NO_SOURCE_DISTRIBUTION_CLAIM = (
    "Candidate-vs-original marginals are diagnostics only; they do not establish "
    "source-distribution pass/fail, and no retrospective thresholds are applied."
)
_ID_FIELD = "persona_id"
_PROSE_FIELD = "persona"
_VERSIONED_FIELDS = (
    "sex",
    "municipality_code",
    "marital_status",
    "age_band",
    "education_level",
    "labour_market_status",
    "job_function",
)
_SOURCE_SUPPORT_SOURCES = ("folk1a", "ras209", "ras202")
_SOURCE_SUPPORT_FIELDS = (
    "age",
    "sex",
    "municipality_code",
    "marital_status",
    "education_source_code",
    "labour_market_status",
    "detailed_status_code",
)
_GENERATED_CHECK_FIELDS = (
    "persona_id",
    "current_relationship_status",
    "partner_gender",
    "legal_status_detail",
    "marital_status",
    "skills_and_expertise",
    "hobbies_and_interests",
    "persona",
)
_BLANK_REPORT_FIELDS = (
    "persona_id",
    "persona",
    "cultural_context",
    "skills_and_expertise",
    "hobbies_and_interests",
    "current_relationship_status",
    "sex",
    "municipality_code",
    "marital_status",
    "age_band",
    "education_level",
    "labour_market_status",
)
_COLLISION_REPORT_FIELDS = (
    "persona",
    "cultural_context",
    "skills_and_expertise",
    "hobbies_and_interests",
)


def _candidate_required_columns() -> frozenset[str]:
    return frozenset(
        (
            _ID_FIELD,
            _PROSE_FIELD,
            *_VERSIONED_FIELDS,
            *_SOURCE_SUPPORT_FIELDS,
            *_GENERATED_CHECK_FIELDS,
        )
    )


def _blank_count(values: pl.Series) -> int:
    total = 0
    for value in values:
        if value is None:
            total += 1
        elif isinstance(value, str) and not value.strip():
            total += 1
        elif isinstance(value, list) and not value:
            total += 1
    return total


@dataclass(frozen=True)
class CollisionMetric:
    """Duplicate nonblank value diagnostic for one aggregate field."""

    field: str
    duplicated_values: int
    rows_in_duplicated_values: int


@dataclass(frozen=True)
class CollisionBlankReport:
    """Aggregate duplicate-ID, repeated-value, and blank-field diagnostics."""

    candidate_duplicate_id_values: int
    original_duplicate_id_values: int
    blank_candidate_rows_by_field: dict[str, int]
    collision_metrics: tuple[CollisionMetric, ...]


def _collision_blank_report(
    *, candidate: pl.DataFrame, original: pl.DataFrame
) -> CollisionBlankReport:
    blank_fields = [
        field for field in _BLANK_REPORT_FIELDS if field in candidate.columns
    ]
    collision_fields = [
        field for field in _COLLISION_REPORT_FIELDS if field in candidate.columns
    ]
    return CollisionBlankReport(
        candidate_duplicate_id_values=_duplicate_value_count(candidate[_ID_FIELD]),
        original_duplicate_id_values=_duplicate_value_count(original[_ID_FIELD]),
        blank_candidate_rows_by_field={
            field: _blank_count(candidate[field]) for field in blank_fields
        },
        collision_metrics=tuple(
            _collision_metric(frame=candidate, field=field)
            for field in collision_fields
        ),
    )


def _collision_metric(*, frame: pl.DataFrame, field: str) -> CollisionMetric:
    counts = Counter(
        _normalise_value(value) for value in frame[field] if not _is_blank(value)
    )
    duplicate_counts = [count for count in counts.values() if count > 1]
    return CollisionMetric(
        field=field,
        duplicated_values=len(duplicate_counts),
        rows_in_duplicated_values=sum(duplicate_counts),
    )


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, pl.Series):
        return value.is_empty()
    return isinstance(value, list) and not value


def _normalise_value(value: object) -> object:
    if isinstance(value, pl.Series):
        return tuple(_normalise_value(item) for item in value.to_list())
    if isinstance(value, list):
        return tuple(_normalise_value(item) for item in value)
    return value


def _duplicate_value_count(values: pl.Series) -> int:
    counts = Counter(_normalise_value(value) for value in values)
    return sum(count > 1 for count in counts.values())


@dataclass(frozen=True)
class GeneratedFieldReport:
    """Generated-field hard failures and advisory prose-review flags."""

    hard_failure_counts: dict[str, int]
    hard_failure_rows: int
    advisory_prose_flag_counts: dict[str, int]
    advisory_prose_flag_rows: int


def _generated_field_report(*, frame: pl.DataFrame) -> GeneratedFieldReport:
    report = check_generated_fields(frame)
    return GeneratedFieldReport(
        hard_failure_counts=report.hard_failure_counts,
        hard_failure_rows=len(report.hard_failure_persona_ids),
        advisory_prose_flag_counts=report.review_flag_counts,
        advisory_prose_flag_rows=len(report.review_flag_persona_ids),
    )


@dataclass(frozen=True)
class IdentityIntegrityReport:
    """Aggregate ID integrity results without exposing row identifiers."""

    row_count: int
    expected_row_count: int
    candidate_unique_ids: int
    original_unique_ids: int
    candidate_blank_id_rows: int
    original_blank_id_rows: int
    candidate_duplicate_id_rows: int
    original_duplicate_id_rows: int
    ordered_ids_match: bool


def _identity_integrity_report(
    *, candidate: pl.DataFrame, original: pl.DataFrame, expected_row_count: int
) -> IdentityIntegrityReport:
    candidate_ids = [
        str(value) if value is not None else "" for value in candidate[_ID_FIELD]
    ]
    original_ids = [
        str(value) if value is not None else "" for value in original[_ID_FIELD]
    ]
    candidate_unique = len(set(candidate_ids))
    original_unique = len(set(original_ids))
    return IdentityIntegrityReport(
        row_count=candidate.height,
        expected_row_count=expected_row_count,
        candidate_unique_ids=candidate_unique,
        original_unique_ids=original_unique,
        candidate_blank_id_rows=sum(not value.strip() for value in candidate_ids),
        original_blank_id_rows=sum(not value.strip() for value in original_ids),
        candidate_duplicate_id_rows=candidate.height - candidate_unique,
        original_duplicate_id_rows=original.height - original_unique,
        ordered_ids_match=candidate_ids == original_ids,
    )


def _load_vetted_patch_manifest(
    *, path: Path | None
) -> tuple[frozenset[str], str | None]:
    if path is None:
        return frozenset(), None
    with path.open(encoding="utf-8") as file:
        payload: object = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Vetted prose patch manifest must be a JSON object")
    if payload.get("version") != 1:
        raise ValueError("Vetted prose patch manifest version must be 1")
    if payload.get("status") != "vetted" and payload.get("vetted") is not True:
        raise ValueError("Vetted prose patch manifest must be explicitly vetted")
    value = payload.get("allowed_persona_id_hashes")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            "Vetted prose patch manifest must list allowed_persona_id_hashes"
        )
    hashes = frozenset(value)
    hex_digits = set(string.hexdigits)
    if any(len(item) != 64 or not set(item) <= hex_digits for item in hashes):
        raise ValueError("Vetted prose patch manifest contains invalid ID hashes")
    return hashes, sha256_file(path)


@dataclass(frozen=True)
class MarginalTotalVariationMetric:
    """One candidate-vs-original marginal total-variation diagnostic."""

    field: str
    total_variation: float
    candidate_distinct_values: int
    original_distinct_values: int


def _marginal_total_variation(
    *, candidate: pl.DataFrame, original: pl.DataFrame, field: str
) -> MarginalTotalVariationMetric:
    candidate_counts = Counter(_normalise_value(value) for value in candidate[field])
    original_counts = Counter(_normalise_value(value) for value in original[field])
    values = set(candidate_counts) | set(original_counts)
    total = candidate.height
    variation = 0.5 * sum(
        abs(candidate_counts[value] / total - original_counts[value] / total)
        for value in values
    )
    return MarginalTotalVariationMetric(
        field=field,
        total_variation=variation,
        candidate_distinct_values=len(candidate_counts),
        original_distinct_values=len(original_counts),
    )


@dataclass(frozen=True)
class ProseChangeReport:
    """Aggregate prose-change results without exposing text or identifiers."""

    changed_rows: int
    blank_candidate_rows: int
    allowed_changed_rows: int
    unvetted_changed_rows: int
    vetted_patch_manifest_sha256: str | None


def _prose_change_report(
    *,
    candidate: pl.DataFrame,
    original: pl.DataFrame,
    allowed_id_hashes: frozenset[str],
    manifest_sha256: str | None,
) -> ProseChangeReport:
    original_by_id = _unique_text_by_id(frame=original)
    changed_hashes: list[str] = []
    blank_rows = 0
    for row in candidate.select([_ID_FIELD, _PROSE_FIELD]).iter_rows(named=True):
        persona_id = str(row[_ID_FIELD]) if row[_ID_FIELD] is not None else ""
        candidate_text = row[_PROSE_FIELD]
        if not isinstance(candidate_text, str) or not candidate_text.strip():
            blank_rows += 1
        original_text = original_by_id.get(persona_id)
        if original_text is not None and candidate_text != original_text:
            changed_hashes.append(sha256_text(persona_id))
    changed = len(changed_hashes)
    unvetted = sum(item not in allowed_id_hashes for item in changed_hashes)
    return ProseChangeReport(
        changed_rows=changed,
        blank_candidate_rows=blank_rows,
        allowed_changed_rows=changed - unvetted,
        unvetted_changed_rows=unvetted,
        vetted_patch_manifest_sha256=manifest_sha256,
    )


def _unique_text_by_id(*, frame: pl.DataFrame) -> dict[str, object]:
    counts = Counter(
        str(value) if value is not None else "" for value in frame[_ID_FIELD]
    )
    texts: dict[str, object] = {}
    for row in frame.select([_ID_FIELD, _PROSE_FIELD]).iter_rows(named=True):
        persona_id = str(row[_ID_FIELD]) if row[_ID_FIELD] is not None else ""
        if counts[persona_id] == 1:
            texts[persona_id] = row[_PROSE_FIELD]
    return texts


def _require_columns(*, frame: pl.DataFrame, columns: frozenset[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Frame is missing required columns: {missing}")


def _require_matching_counts(
    *, candidate: pl.DataFrame, original: pl.DataFrame, expected_row_count: int
) -> None:
    if candidate.height != original.height:
        raise ValueError(
            "Candidate and original row counts differ: "
            f"{candidate.height} != {original.height}"
        )
    if candidate.height != expected_row_count:
        raise ValueError(
            "Candidate row count does not match the expected release size: "
            f"{candidate.height} != {expected_row_count}"
        )


@dataclass(frozen=True)
class SourceSupportReport:
    """Prepared-source support counts for each hard source gate."""

    supported_rows: dict[str, int]
    unsupported_rows: dict[str, int]


@dataclass(frozen=True)
class ReleaseCandidateValidationReport:
    """Typed aggregate release-candidate diagnostic report.

    The report intentionally contains hashes and counts only. It must not include raw
    persona IDs or generated prose.
    """

    label: str
    notice: str
    source_distribution_claim: str
    candidate_sha256: str
    original_sha256: str
    passes_hard_gates: bool
    hard_failure_counts: dict[str, int]
    identity: IdentityIntegrityReport
    prose: ProseChangeReport
    source_support: SourceSupportReport
    generated_fields: GeneratedFieldReport
    marginal_total_variation: tuple[MarginalTotalVariationMetric, ...]
    collision_blank: CollisionBlankReport


def validate_release_candidate(
    *,
    candidate_path: Path,
    original_path: Path,
    bundle_dir: Path,
    vetted_patch_manifest_path: Path | None = None,
    expected_row_count: int = EXPECTED_RELEASE_ROWS,
) -> ReleaseCandidateValidationReport:
    """Validate a release candidate against a published baseline and bundle.

    Args:
        candidate_path:
            Parquet candidate to diagnose.
        original_path:
            Published baseline Parquet for ordered-ID and marginal comparison.
        bundle_dir:
            Prepared source-bundle root containing the normalised source Parquets.
        vetted_patch_manifest_path (optional):
            JSON manifest with explicitly vetted prose-change ID hashes. Defaults to
            no allowed prose changes.
        expected_row_count (optional):
            Required row count. Defaults to the full 100k release size.

    Returns:
        Aggregate report with only counts and file hashes.
    """
    candidate = pl.read_parquet(candidate_path)
    original = pl.read_parquet(original_path)
    _require_matching_counts(
        candidate=candidate, original=original, expected_row_count=expected_row_count
    )
    _require_columns(frame=candidate, columns=_candidate_required_columns())
    _require_columns(
        frame=original, columns=frozenset((_ID_FIELD, _PROSE_FIELD, *_VERSIONED_FIELDS))
    )

    allowed_hashes, manifest_sha256 = _load_vetted_patch_manifest(
        path=vetted_patch_manifest_path
    )
    identity = _identity_integrity_report(
        candidate=candidate, original=original, expected_row_count=expected_row_count
    )
    prose = _prose_change_report(
        candidate=candidate,
        original=original,
        allowed_id_hashes=allowed_hashes,
        manifest_sha256=manifest_sha256,
    )
    source_support = _source_support_report(frame=candidate, bundle_dir=bundle_dir)
    generated_fields = _generated_field_report(frame=candidate)
    marginal_metrics = tuple(
        _marginal_total_variation(candidate=candidate, original=original, field=field)
        for field in _VERSIONED_FIELDS
    )
    collision_blank = _collision_blank_report(candidate=candidate, original=original)
    hard_failure_counts = _hard_failure_counts(
        identity=identity,
        prose=prose,
        source_support=source_support,
        generated_fields=generated_fields,
    )
    editorial_violations = count_editorial_foreign_student_violations(candidate)
    if editorial_violations:
        hard_failure_counts["editorial_foreign_student_violations"] = (
            editorial_violations
        )
    return ReleaseCandidateValidationReport(
        label=REPORT_LABEL,
        notice=_PRELIMINARY_NOTICE,
        source_distribution_claim=_NO_SOURCE_DISTRIBUTION_CLAIM,
        candidate_sha256=sha256_file(candidate_path),
        original_sha256=sha256_file(original_path),
        passes_hard_gates=not hard_failure_counts,
        hard_failure_counts=hard_failure_counts,
        identity=identity,
        prose=prose,
        source_support=source_support,
        generated_fields=generated_fields,
        marginal_total_variation=marginal_metrics,
        collision_blank=collision_blank,
    )


def _hard_failure_counts(
    *,
    identity: IdentityIntegrityReport,
    prose: ProseChangeReport,
    source_support: SourceSupportReport,
    generated_fields: GeneratedFieldReport,
) -> dict[str, int]:
    failures = {
        "candidate_blank_id_rows": identity.candidate_blank_id_rows,
        "original_blank_id_rows": identity.original_blank_id_rows,
        "candidate_duplicate_id_rows": identity.candidate_duplicate_id_rows,
        "original_duplicate_id_rows": identity.original_duplicate_id_rows,
        "ordered_id_mismatch": 0 if identity.ordered_ids_match else 1,
        "blank_prose_rows": prose.blank_candidate_rows,
        "unvetted_prose_changes": prose.unvetted_changed_rows,
        "generated_field_hard_failure_rows": generated_fields.hard_failure_rows,
    }
    for source, unsupported_rows in source_support.unsupported_rows.items():
        failures[f"{source}_unsupported_rows"] = unsupported_rows
    return {key: value for key, value in sorted(failures.items()) if value}


def _source_support_report(
    *, frame: pl.DataFrame, bundle_dir: Path
) -> SourceSupportReport:
    raw_report = assess_release_support(frame=frame, bundle_dir=bundle_dir)
    supported: dict[str, int] = {}
    unsupported: dict[str, int] = {}
    for source in _SOURCE_SUPPORT_SOURCES:
        result = raw_report.get(source)
        if not isinstance(result, dict):
            raise ValueError(f"Source support report missing {source}")
        supported[source] = _integer_result(result=result, key="supported_rows")
        unsupported[source] = _integer_result(result=result, key="unsupported_rows")
        missing_columns = result.get("missing_columns", [])
        if missing_columns:
            raise ValueError(f"Source support check for {source} is missing columns")
    return SourceSupportReport(supported_rows=supported, unsupported_rows=unsupported)


def _integer_result(*, result: dict[object, object], key: str) -> int:
    value = result.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"Source support report has invalid {key!r}")
    return value
