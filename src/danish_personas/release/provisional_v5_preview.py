"""Offline assembly of a merged provisional v5 prose preview."""

from __future__ import annotations

import collections.abc as c
import json
import os
import tempfile
import typing as t
from pathlib import Path

import httpx
import polars as pl

from scripts import build_provisional_prose_candidate as v4_candidate
from scripts import verify_persona_patches as verify

from ..io import canonical_json, sha256_file, sha256_text
from .apply_reviewed_prose import apply_reviewed_prose
from .prose_patch_verification import ProsePatchVerificationResult

DEFAULT_ROOT = Path("/tmp/danish-personas-audit")
DEFAULT_ORIGINAL = DEFAULT_ROOT / "data/train-00000-of-00001.parquet"
DEFAULT_V4_STRUCTURED = DEFAULT_ROOT / "attribute-candidate-v4.parquet"
DEFAULT_V5_H90 = DEFAULT_ROOT / "attribute-candidate-v5-h90-PROVISIONAL.parquet"
DEFAULT_V4_PROSE_PREVIEW = (
    DEFAULT_ROOT / "prose-candidate-v4-reviewed-PROVISIONAL.parquet"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "prose-candidate-v5-merged-PROVISIONAL.parquet"
EXPECTED_RELEASE_ROWS = 100_000
EXPECTED_H90_CHANGED_ROWS = 506
EXPECTED_V4_ACCEPTED_PATCHES = 3_236
EXPECTED_H90_ACCEPTED_PATCHES = 18
H90_CHANGED_FIELDS = frozenset({"education_level", "education_source_code"})
_ID_FIELD = "persona_id"
_PROSE_FIELD = "persona"
_PUBLICATION_PATH_TOKENS = frozenset(
    {"publication", "published", "v1", "tag", "card", "huggingface", "hf"}
)

JSONScalar: t.TypeAlias = str | int | float | bool | None
JSONValue: t.TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def build_provisional_v5_preview(
    *,
    original: Path,
    v4_structured: Path,
    v5_h90: Path,
    v4_prose_preview: Path,
    v4_triage: Path,
    v4_first_prompt: Path,
    verify_prompt: Path,
    registry: Path,
    v4_first_status: Path,
    v4_first_manifest: Path,
    v4_first_checkpoint_root: Path,
    v4_second_status: Path,
    v4_second_manifest: Path,
    v4_second_checkpoint_root: Path,
    h90_triage: Path,
    h90_first_prompt: Path,
    h90_first_status: Path,
    h90_first_manifest: Path,
    h90_first_checkpoint_root: Path,
    h90_second_status: Path,
    h90_second_manifest: Path,
    h90_second_checkpoint_root: Path,
    output: Path,
    write_output: bool,
    expected_row_count: int = EXPECTED_RELEASE_ROWS,
    expected_h90_changed_count: int = EXPECTED_H90_CHANGED_ROWS,
    expected_v4_accepted_count: int = EXPECTED_V4_ACCEPTED_PATCHES,
    expected_h90_accepted_count: int = EXPECTED_H90_ACCEPTED_PATCHES,
    h90_changed_fields: c.Set[str] = H90_CHANGED_FIELDS,
) -> dict[str, JSONValue]:
    """Build or dry-run the merged v5 provisional preview without provider I/O.

    Args:
        original:
            Immutable published v1 Parquet input.
        v4_structured:
            Immutable v4 structured Parquet input.
        v5_h90:
            Immutable provisional v5 H90 structured Parquet input.
        v4_prose_preview:
            Previously built v4 provisional prose preview to revalidate.
        v4_triage:
            V4 first-pass triage file pinned by the v4 manifest.
        v4_first_prompt:
            V4 first-pass prompt pinned by the v4 manifest.
        verify_prompt:
            Shared second-pass verifier prompt.
        registry:
            Pinned model registry.
        v4_first_status:
            V4 first-pass private status.
        v4_first_manifest:
            V4 first-pass private manifest.
        v4_first_checkpoint_root:
            V4 first-pass checkpoint root.
        v4_second_status:
            V4 second-pass private status.
        v4_second_manifest:
            V4 second-pass private manifest.
        v4_second_checkpoint_root:
            V4 second-pass checkpoint root.
        h90_triage:
            H90 first-pass triage file pinned by the H90 manifest.
        h90_first_prompt:
            H90 first-pass prompt pinned by the H90 manifest.
        h90_first_status:
            H90 first-pass private status.
        h90_first_manifest:
            H90 first-pass private manifest.
        h90_first_checkpoint_root:
            H90 first-pass checkpoint root.
        h90_second_status:
            H90 second-pass private status.
        h90_second_manifest:
            H90 second-pass private manifest.
        h90_second_checkpoint_root:
            H90 second-pass checkpoint root.
        output:
            New Parquet output path for ``--write``.
        write_output:
            If true, write the preview and adjacent JSON report.
        expected_row_count (optional):
            Ordered-row contract. Defaults to 100,000.
        expected_h90_changed_count (optional):
            Required H90 changed-row count. Defaults to 506.
        expected_v4_accepted_count (optional):
            Required v4 accepted prose count. Defaults to 3,236.
        expected_h90_accepted_count (optional):
            Required H90 accepted prose count. Defaults to 18.
        h90_changed_fields (optional):
            Exact non-prose field set allowed to change from v4 to v5.

    Returns:
        Aggregate report without raw identifiers or prose.
    """
    if write_output:
        _guard_write_target(output=output)

    original_frame = pl.read_parquet(original)
    v4_frame = pl.read_parquet(v4_structured)
    v5_frame = pl.read_parquet(v5_h90)
    v4_preview_frame = pl.read_parquet(v4_prose_preview)
    _validate_ordered_frames(
        frames=(
            ("original", original_frame),
            ("v4 structured", v4_frame),
            ("v5 H90", v5_frame),
            ("v4 prose preview", v4_preview_frame),
        ),
        expected_row_count=expected_row_count,
    )

    h90_report_path = v5_h90.with_suffix(".report.json")
    h90_report = _load_json_object(path=h90_report_path, label="H90 report")
    h90_hashes = _validate_h90_structured_delta(
        v4_frame=v4_frame,
        v5_frame=v5_frame,
        h90_report=h90_report,
        expected_count=expected_h90_changed_count,
        allowed_fields=h90_changed_fields,
    )
    v4_preview_report = _revalidate_v4_preview(
        original=original,
        v4_structured=v4_structured,
        v4_prose_preview_frame=v4_preview_frame,
        triage=v4_triage,
        first_prompt=v4_first_prompt,
        verify_prompt=verify_prompt,
        registry=registry,
        first_status=v4_first_status,
        first_manifest=v4_first_manifest,
        first_checkpoint_root=v4_first_checkpoint_root,
        second_status=v4_second_status,
        second_manifest=v4_second_manifest,
        second_checkpoint_root=v4_second_checkpoint_root,
        expected_accepted_count=expected_v4_accepted_count,
    )
    h90_first_records, h90_second_records, h90_status = _h90_records(
        v4_structured=v4_structured,
        v5_h90=v5_h90,
        triage=h90_triage,
        first_prompt=h90_first_prompt,
        verify_prompt=verify_prompt,
        registry=registry,
        first_status=h90_first_status,
        first_manifest=h90_first_manifest,
        first_checkpoint_root=h90_first_checkpoint_root,
        second_status=h90_second_status,
        second_manifest=h90_second_manifest,
        second_checkpoint_root=h90_second_checkpoint_root,
        expected_accepted_count=expected_h90_accepted_count,
    )
    preview, merge_report = merge_provisional_v5_preview(
        original_frame=original_frame,
        v4_structured_frame=v4_frame,
        v5_h90_frame=v5_frame,
        v4_prose_preview_frame=v4_preview_frame,
        h90_first_pass_records=h90_first_records,
        h90_second_pass_records=h90_second_records,
        h90_changed_hashes=h90_hashes,
        h90_source_metadata={
            "original_v1_sha256": sha256_file(original),
            "v4_structured_sha256": sha256_file(v4_structured),
            "v5_h90_sha256": sha256_file(v5_h90),
            "v5_h90_report_sha256": sha256_file(h90_report_path),
            "h90_first_manifest_sha256": sha256_file(h90_first_manifest),
            "h90_second_manifest_sha256": sha256_file(h90_second_manifest),
            "h90_second_status_sha256": sha256_file(h90_second_status),
        },
        expected_row_count=expected_row_count,
    )
    report = _summary(
        merge_report=merge_report,
        original=original,
        v4_structured=v4_structured,
        v5_h90=v5_h90,
        v4_prose_preview=v4_prose_preview,
        h90_report_path=h90_report_path,
        v4_preview_report=v4_preview_report,
        h90_report=h90_report,
        h90_status=h90_status,
        output=output,
        write_output=write_output,
    )
    if write_output:
        _write_outputs(output=output, preview=preview, report=report)
    return report


def _guard_write_target(*, output: Path) -> None:
    lowered_parts = {part.lower() for part in output.parts}
    stem_tokens = set(output.stem.lower().replace("-", "_").split("_"))
    if (
        lowered_parts & _PUBLICATION_PATH_TOKENS
        or stem_tokens & _PUBLICATION_PATH_TOKENS
    ):
        raise ProvisionalV5PreviewError("Refusing publication-like output path")
    for path in (output, _report_path(output=output)):
        if path.exists():
            message = f"Refusing to overwrite existing file: {path}"
            raise ProvisionalV5PreviewError(message)


class ProvisionalV5PreviewError(RuntimeError):
    """Raised when the provisional v5 preview cannot be built safely."""


def _report_path(*, output: Path) -> Path:
    return output.with_suffix(".json")


def _h90_records(
    *,
    v4_structured: Path,
    v5_h90: Path,
    triage: Path,
    first_prompt: Path,
    verify_prompt: Path,
    registry: Path,
    first_status: Path,
    first_manifest: Path,
    first_checkpoint_root: Path,
    second_status: Path,
    second_manifest: Path,
    second_checkpoint_root: Path,
    expected_accepted_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    paths = verify.VerifyPaths(
        original=v4_structured,
        candidate=v5_h90,
        triage=triage,
        first_prompt=first_prompt,
        verify_prompt=verify_prompt,
        registry=registry,
        first_status=first_status,
        first_manifest=first_manifest,
        first_checkpoint_root=first_checkpoint_root,
        output_dir=second_checkpoint_root,
        budget_purpose=verify.H90_BUDGET_PURPOSE,
    )
    verify._require_h90_private_inputs(paths=paths)
    loaded = verify.load_first_pass(paths=paths)
    expected_manifest = verify._verification_manifest(paths=paths, loaded=loaded)
    actual_manifest = _load_json_object(
        path=second_manifest, label="H90 second manifest"
    )
    if actual_manifest != expected_manifest:
        raise ProvisionalV5PreviewError("H90 second-pass manifest pins do not match")
    try:
        status = v4_candidate._load_second_status(
            path=second_status,
            manifest=t.cast(dict[str, v4_candidate.JSONValue], actual_manifest),
            first_patched_count=len(loaded.rows),
        )
    except v4_candidate.ProvisionalProseCandidateError as exc:
        raise ProvisionalV5PreviewError(
            "H90 second-pass status validation failed"
        ) from exc
    available = _status_int(status=status, key="available")
    processed = _status_int(status=status, key="processed")
    pending = _status_int(status=status, key="pending")
    accepted = _status_int(status=status, key="accepted")
    if processed != available or pending != 0:
        raise ProvisionalV5PreviewError("H90 second-pass campaign is not terminal")
    if accepted != expected_accepted_count:
        raise ProvisionalV5PreviewError("H90 accepted prose count is not pinned")
    first_records = v4_candidate._first_pass_records(
        loaded=loaded, original_frame=pl.read_parquet(v4_structured)
    )
    second_records = _accepted_second_pass_records(
        paths=paths,
        loaded=loaded,
        status=status,
        second_checkpoint_root=second_checkpoint_root,
        prompt=loaded.verify_prompt,
        verify_prompt=verify_prompt,
        manifest=t.cast(dict[str, JSONValue], actual_manifest),
    )
    return first_records, second_records, status


def _accepted_second_pass_records(
    *,
    paths: verify.VerifyPaths,
    loaded: verify.LoadedFirstPass,
    status: dict[str, object],
    second_checkpoint_root: Path,
    prompt: str,
    verify_prompt: Path,
    manifest: dict[str, JSONValue],
) -> list[dict[str, object]]:
    row_index = {row.persona_hash: row for row in loaded.rows}
    processed_hashes = verify._status_hashes(status)
    if len(processed_hashes) != len(set(processed_hashes)):
        raise ProvisionalV5PreviewError("Second-pass status contains duplicate IDs")
    unknown = sorted(set(processed_hashes) - set(row_index))
    if unknown:
        raise ProvisionalV5PreviewError("Second-pass status contains unknown IDs")
    config = verify._generation_config(prompt_path=verify_prompt)
    budget = _proxy_budget_from_manifest(paths=paths, prompt=prompt, manifest=manifest)
    records: list[dict[str, object]] = []
    rejected_count = 0
    for persona_hash in processed_hashes:
        checkpoint_path = verify._checkpoint_path(
            output_dir=second_checkpoint_root, persona_hash=persona_hash
        )
        try:
            checkpoint_accepted = v4_candidate._checkpoint_is_accepted(
                path=checkpoint_path
            )
        except v4_candidate.ProvisionalProseCandidateError as exc:
            raise ProvisionalV5PreviewError(
                "Second-pass checkpoint validation failed"
            ) from exc
        if not checkpoint_accepted:
            rejected_count += 1
            continue
        row = row_index[persona_hash]
        result = verify.run_proxy_patch_verification(
            row=row.original_row,
            candidate_row=row.candidate_row,
            changed_facts=row.changed_facts,
            proposed_text=row.proposed_text,
            patches=row.patches,
            first_checkpoint_sha256=row.first_checkpoint_sha256,
            prompt=prompt,
            config=config,
            budget=budget,
            checkpoint_path=checkpoint_path,
            transport=_failing_transport(),
        )
        if not result.accepted:
            raise ProvisionalV5PreviewError(
                "Accepted second-pass checkpoint revalidated as rejected"
            )
        records.append(_second_pass_record(row=row, result=result))
    if len(records) != _status_int(status=status, key="accepted"):
        raise ProvisionalV5PreviewError("Second-pass accepted count is stale")
    if rejected_count != _status_int(status=status, key="rejected"):
        raise ProvisionalV5PreviewError("Second-pass rejected count is stale")
    return records


def _failing_transport() -> httpx.MockTransport:
    def respond(_request: httpx.Request) -> httpx.Response:
        raise ProvisionalV5PreviewError("Provider access is disabled offline")

    return httpx.MockTransport(respond)


def _proxy_budget_from_manifest(
    *, paths: verify.VerifyPaths, prompt: str, manifest: dict[str, JSONValue]
) -> verify.ProxyBudget:
    real_paths = verify.VerifyPaths(
        original=paths.original,
        candidate=paths.candidate,
        triage=paths.triage,
        first_prompt=paths.first_prompt,
        verify_prompt=paths.verify_prompt,
        registry=paths.registry,
        first_status=paths.first_status,
        first_manifest=paths.first_manifest,
        first_checkpoint_root=paths.first_checkpoint_root,
        output_dir=paths.output_dir,
        budget_purpose=verify.H90_BUDGET_PURPOSE
        if manifest.get("budget_purpose") == verify.H90_BUDGET_PURPOSE
        else verify.DEFAULT_BUDGET_PURPOSE,
    )
    return verify._proxy_budget(paths=real_paths, prompt=prompt, manifest=manifest)


def _second_pass_record(
    *, row: verify.VerifyRow, result: ProsePatchVerificationResult
) -> dict[str, object]:
    record = t.cast(dict[str, object], result.model_dump(mode="json"))
    raw_id = row.original_row.get(_ID_FIELD)
    if not isinstance(raw_id, str) or sha256_text(raw_id) != row.persona_hash:
        raise ProvisionalV5PreviewError("Second-pass row ID does not match hash")
    if row.candidate_row.get(_ID_FIELD) != raw_id:
        raise ProvisionalV5PreviewError("Second-pass candidate ID mismatch")
    record[_ID_FIELD] = raw_id
    record["persona_hash"] = row.persona_hash
    record["status"] = "accepted" if record.get("accepted") is True else "rejected"
    return record


def _status_int(*, status: dict[str, object], key: str) -> int:
    value = status.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProvisionalV5PreviewError(f"Status count is malformed: {key}")
    return value


def _load_json_object(*, path: Path, label: str) -> dict[str, JSONValue]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvisionalV5PreviewError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ProvisionalV5PreviewError(f"{label} is not a JSON object")
    return t.cast(dict[str, JSONValue], value)


def _revalidate_v4_preview(
    *,
    original: Path,
    v4_structured: Path,
    v4_prose_preview_frame: pl.DataFrame,
    triage: Path,
    first_prompt: Path,
    verify_prompt: Path,
    registry: Path,
    first_status: Path,
    first_manifest: Path,
    first_checkpoint_root: Path,
    second_status: Path,
    second_manifest: Path,
    second_checkpoint_root: Path,
    expected_accepted_count: int,
) -> dict[str, JSONValue]:
    paths = verify.VerifyPaths(
        original=original,
        candidate=v4_structured,
        triage=triage,
        first_prompt=first_prompt,
        verify_prompt=verify_prompt,
        registry=registry,
        first_status=first_status,
        first_manifest=first_manifest,
        first_checkpoint_root=first_checkpoint_root,
        output_dir=second_checkpoint_root,
    )
    loaded = verify.load_first_pass(paths=paths)
    expected_manifest = verify._verification_manifest(paths=paths, loaded=loaded)
    actual_manifest = _load_json_object(
        path=second_manifest, label="v4 second manifest"
    )
    if actual_manifest != expected_manifest:
        raise ProvisionalV5PreviewError("V4 second-pass manifest pins do not match")
    try:
        status = v4_candidate._load_second_status(
            path=second_status,
            manifest=t.cast(dict[str, v4_candidate.JSONValue], actual_manifest),
            first_patched_count=len(loaded.rows),
        )
    except v4_candidate.ProvisionalProseCandidateError as exc:
        raise ProvisionalV5PreviewError(
            "V4 second-pass status validation failed"
        ) from exc
    first_records = v4_candidate._first_pass_records(
        loaded=loaded, original_frame=pl.read_parquet(original)
    )
    second_records = _accepted_second_pass_records(
        paths=paths,
        loaded=loaded,
        status=status,
        second_checkpoint_root=second_checkpoint_root,
        prompt=loaded.verify_prompt,
        verify_prompt=verify_prompt,
        manifest=t.cast(dict[str, JSONValue], actual_manifest),
    )
    rebuilt, report = apply_reviewed_prose(
        original_published_frame=pl.read_parquet(original),
        v4_structured_frame=pl.read_parquet(v4_structured),
        first_pass_records=first_records,
        second_pass_records=second_records,
        source_metadata={
            "original_sha256": sha256_file(original),
            "v4_structured_sha256": sha256_file(v4_structured),
            "first_manifest_sha256": sha256_file(first_manifest),
            "second_manifest_sha256": sha256_file(second_manifest),
            "second_status_sha256": sha256_file(second_status),
        },
        expected_row_count=pl.read_parquet(original).height,
    )
    if _frame_hash(frame=rebuilt) != _frame_hash(frame=v4_prose_preview_frame):
        raise ProvisionalV5PreviewError("V4 prose preview does not match checkpoints")
    accepted = _json_int(
        document=t.cast(dict[str, JSONValue], report),
        key="accepted_second_pass_rows",
        label="v4 preview report",
    )
    if accepted != expected_accepted_count:
        raise ProvisionalV5PreviewError("V4 accepted prose count is not pinned")
    return t.cast(dict[str, JSONValue], report)


def _frame_hash(*, frame: pl.DataFrame) -> str:
    payload = {"columns": frame.columns, "rows": frame.to_dicts()}
    return sha256_text(canonical_json(payload))


def _json_int(*, document: dict[str, JSONValue], key: str, label: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProvisionalV5PreviewError(f"{label} {key} is malformed")
    return value


def _summary(
    *,
    merge_report: dict[str, JSONValue],
    original: Path,
    v4_structured: Path,
    v5_h90: Path,
    v4_prose_preview: Path,
    h90_report_path: Path,
    v4_preview_report: dict[str, JSONValue],
    h90_report: dict[str, JSONValue],
    h90_status: dict[str, object],
    output: Path,
    write_output: bool,
) -> dict[str, JSONValue]:
    blockers = _h90_blockers(status=h90_status)
    return {
        **merge_report,
        "dry_run": not write_output,
        "written": write_output,
        "candidate_path": str(output),
        "report_path": str(_report_path(output=output)),
        "release_ready": False,
        "not_release_ready": True,
        "publication_allowed": False,
        "approves_proposals": False,
        "source_hashes": {
            "original_v1_sha256": sha256_file(original),
            "v4_structured_sha256": sha256_file(v4_structured),
            "v5_h90_sha256": sha256_file(v5_h90),
            "v5_h90_report_sha256": sha256_file(h90_report_path),
            "v4_prose_preview_sha256": sha256_file(v4_prose_preview),
        },
        "v4_preview_sha256": _json_string(
            document=v4_preview_report, key="preview_sha256", label="v4 preview report"
        ),
        "h90_blockers": blockers,
        "h90_blocker_total": sum(blockers.values()),
        "h90_generation_counts": _h90_generation_counts(report=h90_report),
        "provisional_notice": (
            "PROVISIONAL only: not release-ready, publication_allowed=false, "
            "and stale v4 prose patches are excluded from all H90 changed rows."
        ),
    }


def _h90_blockers(*, status: dict[str, object]) -> dict[str, int]:
    return {
        "needs_manual_review": _status_int(status=status, key="manual"),
        "fact_not_stated_or_unchanged_consistent": _status_int(
            status=status, key="unchanged_consistent"
        ),
        "second_pass_rejected": _status_int(status=status, key="rejected"),
        "second_pass_pending": _status_int(status=status, key="pending"),
        "second_pass_failed": _status_int(status=status, key="failed"),
    }


def _h90_generation_counts(*, report: dict[str, JSONValue]) -> dict[str, JSONValue]:
    keys = (
        "source_share",
        "source_h90_count",
        "source_age20plus_count",
        "age20plus_rows",
        "source_target_h90",
        "current_h90",
        "after_h90",
        "additional_h90_required",
        "changed_rows",
        "source_support",
        "total_variation",
        "tv_diagnostic",
    )
    return {key: report[key] for key in keys if key in report}


def _json_string(*, document: dict[str, JSONValue], key: str, label: str) -> str:
    value = document.get(key)
    if not isinstance(value, str):
        raise ProvisionalV5PreviewError(f"{label} {key} is malformed")
    return value


def _validate_h90_structured_delta(
    *,
    v4_frame: pl.DataFrame,
    v5_frame: pl.DataFrame,
    h90_report: dict[str, JSONValue],
    expected_count: int,
    allowed_fields: c.Set[str],
) -> list[str]:
    report_hashes = _report_hashes(report=h90_report)
    actual_hashes: list[str] = []
    for v4_row, v5_row in zip(v4_frame.to_dicts(), v5_frame.to_dicts(), strict=True):
        persona_id = _row_id(row=v4_row)
        changed = {
            field
            for field in set(v4_row) & set(v5_row)
            if field != _PROSE_FIELD and v4_row[field] != v5_row[field]
        }
        if not changed:
            continue
        if changed != set(allowed_fields):
            raise ProvisionalV5PreviewError("V5 H90 changed non-H90 fields")
        actual_hashes.append(sha256_text(persona_id))
    if sorted(actual_hashes) != report_hashes:
        raise ProvisionalV5PreviewError("V5 H90 changed IDs do not match report")
    if len(actual_hashes) != expected_count:
        raise ProvisionalV5PreviewError("V5 H90 changed-row count is not pinned")
    return sorted(actual_hashes)


def _report_hashes(*, report: dict[str, JSONValue]) -> list[str]:
    value = report.get("changed_persona_id_sha256")
    if not isinstance(value, list):
        raise ProvisionalV5PreviewError("H90 report lacks changed ID hashes")
    return sorted(_hash_string(value=item, label="H90 report hash") for item in value)


def _hash_string(*, value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ProvisionalV5PreviewError(f"{label} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ProvisionalV5PreviewError(f"{label} is not a SHA-256 digest") from exc
    return value.lower()


def _row_id(*, row: c.Mapping[str, object]) -> str:
    persona_id = row.get(_ID_FIELD)
    if not isinstance(persona_id, str) or not persona_id:
        raise ProvisionalV5PreviewError("Frame contains a missing persona ID")
    return persona_id


def _validate_ordered_frames(
    *, frames: c.Sequence[tuple[str, pl.DataFrame]], expected_row_count: int | None
) -> list[str]:
    if not frames:
        raise ProvisionalV5PreviewError("No frames supplied")
    ordered: list[str] | None = None
    for label, frame in frames:
        if {_ID_FIELD, _PROSE_FIELD} - set(frame.columns):
            raise ProvisionalV5PreviewError(f"{label} lacks required columns")
        if expected_row_count is not None and frame.height != expected_row_count:
            raise ProvisionalV5PreviewError(f"{label} row count is not pinned")
        ids = [_row_id(row=row) for row in frame.to_dicts()]
        if len(ids) != len(set(ids)):
            raise ProvisionalV5PreviewError(f"{label} contains duplicate IDs")
        if ordered is None:
            ordered = ids
        elif ids != ordered:
            raise ProvisionalV5PreviewError(f"{label} ordered IDs do not match")
        for row in frame.to_dicts():
            persona = row.get(_PROSE_FIELD)
            if not isinstance(persona, str) or not persona.strip():
                raise ProvisionalV5PreviewError(f"{label} contains blank prose")
    return ordered or []


def _write_outputs(
    *, output: Path, preview: pl.DataFrame, report: dict[str, JSONValue]
) -> None:
    output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(output.parent, 0o700)
    _write_new_parquet(path=output, frame=preview)
    _write_new_json(path=_report_path(output=output), value=report)


def _write_new_json(*, path: Path, value: dict[str, JSONValue]) -> None:
    if path.exists():
        raise ProvisionalV5PreviewError(f"Refusing to overwrite existing file: {path}")
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_parquet(*, path: Path, frame: pl.DataFrame) -> None:
    if path.exists():
        raise ProvisionalV5PreviewError(f"Refusing to overwrite existing file: {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.close(fd)
        frame.write_parquet(temporary)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def merge_provisional_v5_preview(
    *,
    original_frame: pl.DataFrame,
    v4_structured_frame: pl.DataFrame,
    v5_h90_frame: pl.DataFrame,
    v4_prose_preview_frame: pl.DataFrame,
    h90_first_pass_records: c.Sequence[c.Mapping[str, object]],
    h90_second_pass_records: c.Sequence[c.Mapping[str, object]],
    h90_changed_hashes: c.Sequence[str],
    h90_source_metadata: c.Mapping[str, object],
    expected_row_count: int | None = None,
) -> tuple[pl.DataFrame, dict[str, JSONValue]]:
    """Merge v4 accepted prose and H90 accepted patches into v5 structure.

    Args:
        original_frame:
            Published baseline frame with original prose.
        v4_structured_frame:
            Structured v4 frame used by the v4 prose preview.
        v5_h90_frame:
            Structured v5 H90 frame that supplies all non-prose columns.
        v4_prose_preview_frame:
            Revalidated v4 provisional prose preview.
        h90_first_pass_records:
            Locally verified H90 first-pass records.
        h90_second_pass_records:
            Locally verified accepted H90 second-pass records.
        h90_changed_hashes:
            SHA-256 persona ID hashes for every H90 structured change.
        h90_source_metadata:
            Immutable H90 source pins for the H90 prose application step.
        expected_row_count (optional):
            Required row count. Defaults to no fixed-size check.

    Returns:
        Preview frame and aggregate report without raw identifiers or prose.

    Raises:
        ProvisionalV5PreviewError:
            If frame order, identifiers, hashes, or prose boundaries are unsafe.
    """
    ordered_ids = _validate_ordered_frames(
        frames=(
            ("original", original_frame),
            ("v4 structured", v4_structured_frame),
            ("v5 H90", v5_h90_frame),
            ("v4 prose preview", v4_prose_preview_frame),
        ),
        expected_row_count=expected_row_count,
    )
    _assert_only_persona_differs(
        left=v4_prose_preview_frame, right=v4_structured_frame, label="v4 prose preview"
    )
    h90_hash_set = _validate_hashes(hashes=h90_changed_hashes, label="H90 hashes")
    ids_by_hash = {sha256_text(persona_id): persona_id for persona_id in ordered_ids}
    if not h90_hash_set <= set(ids_by_hash):
        raise ProvisionalV5PreviewError("H90 changed hashes include unknown rows")

    h90_preview, h90_apply_report = apply_reviewed_prose(
        original_published_frame=original_frame,
        v4_structured_frame=v5_h90_frame,
        first_pass_records=h90_first_pass_records,
        second_pass_records=h90_second_pass_records,
        source_metadata=h90_source_metadata,
        expected_row_count=expected_row_count,
    )
    original_persona = _persona_by_id(frame=original_frame)
    v4_preview_persona = _persona_by_id(frame=v4_prose_preview_frame)
    merged_rows = h90_preview.to_dicts()
    v4_accepted_hashes: list[str] = []
    retained_v4_hashes: list[str] = []
    excluded_overlap_hashes: list[str] = []
    for row in merged_rows:
        persona_id = _row_id(row=row)
        persona_hash = sha256_text(persona_id)
        if v4_preview_persona[persona_id] == original_persona[persona_id]:
            continue
        v4_accepted_hashes.append(persona_hash)
        if persona_hash in h90_hash_set:
            excluded_overlap_hashes.append(persona_hash)
            continue
        row[_PROSE_FIELD] = v4_preview_persona[persona_id]
        retained_v4_hashes.append(persona_hash)

    preview = pl.DataFrame(merged_rows, schema=v5_h90_frame.schema)
    _assert_only_persona_differs(
        left=preview, right=v5_h90_frame, label="merged v5 preview"
    )
    final_changed_hashes = sorted(
        sha256_text(persona_id)
        for persona_id, persona in _persona_by_id(frame=preview).items()
        if persona != original_persona[persona_id]
    )
    h90_accepted_hashes = sorted(
        persona_hash
        for persona_hash in h90_hash_set
        if persona_hash in final_changed_hashes
    )
    report: dict[str, JSONValue] = {
        "row_count": len(ordered_ids),
        "v4_accepted_prose_count": len(v4_accepted_hashes),
        "v4_retained_non_h90_count": len(retained_v4_hashes),
        "v4_excluded_overlap_count": len(excluded_overlap_hashes),
        "h90_changed_count": len(h90_hash_set),
        "h90_accepted_count": _json_int(
            document=t.cast(dict[str, JSONValue], h90_apply_report),
            key="accepted_second_pass_rows",
            label="H90 apply report",
        ),
        "final_changed_prose_count": len(final_changed_hashes),
        "h90_changed_persona_hashes": sorted(h90_hash_set),
        "h90_accepted_persona_hashes": h90_accepted_hashes,
        "v4_retained_persona_hashes_sha256": sha256_text(
            canonical_json(sorted(retained_v4_hashes))
        ),
        "v4_excluded_overlap_hashes": sorted(excluded_overlap_hashes),
        "final_changed_persona_hashes_sha256": sha256_text(
            canonical_json(final_changed_hashes)
        ),
        "preview_sha256": _frame_hash(frame=preview),
        "asserted_non_prose_columns_match_v5": True,
    }
    return preview, report


def _assert_only_persona_differs(
    *, left: pl.DataFrame, right: pl.DataFrame, label: str
) -> None:
    if left.columns != right.columns:
        raise ProvisionalV5PreviewError(f"{label} columns do not match")
    for column in right.columns:
        if column == _PROSE_FIELD:
            continue
        if left.get_column(column).to_list() != right.get_column(column).to_list():
            raise ProvisionalV5PreviewError(f"{label} changed non-prose columns")


def _persona_by_id(*, frame: pl.DataFrame) -> dict[str, str]:
    personas: dict[str, str] = {}
    for row in frame.to_dicts():
        persona_id = _row_id(row=row)
        persona = row.get(_PROSE_FIELD)
        if not isinstance(persona, str):
            raise ProvisionalV5PreviewError("Frame contains non-string prose")
        personas[persona_id] = persona
    return personas


def _validate_hashes(*, hashes: c.Sequence[str], label: str) -> set[str]:
    return {_hash_string(value=value, label=label) for value in hashes}
