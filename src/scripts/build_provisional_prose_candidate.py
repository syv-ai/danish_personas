"""Build an offline provisional v5 prose-candidate preview."""

from __future__ import annotations

import json
import os
import tempfile
import typing as t
from pathlib import Path

import click
import httpx
import polars as pl

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.environment import load_repository_environment
from danish_personas.generation.proxy_budget import ProxyBudgetError
from danish_personas.generation.proxy_patch_verifier import (
    ProxyPatchVerificationError,
    run_proxy_patch_verification,
)
from danish_personas.io import canonical_json, sha256_file, sha256_text
from danish_personas.release.apply_reviewed_prose import apply_reviewed_prose
from danish_personas.release.prose_patch_verification import (
    ProsePatchVerificationResult,
)
from scripts import verify_persona_patches as verify
from scripts.build_prose_review_v4_dashboard import ProseReviewV4DashboardError

DEFAULT_ROOT = Path("/tmp/danish-personas-audit")
DEFAULT_OUTPUT = DEFAULT_ROOT / "prose-candidate-v5-PROVISIONAL.parquet"
DEFAULT_PUBLICATION_V1 = DEFAULT_ROOT / "prose-publication-v1.parquet"

JSONScalar: t.TypeAlias = str | int | float | bool | None
JSONValue: t.TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@click.command()
@click.option(
    "--original", type=click.Path(path_type=Path), default=verify.DEFAULT_ORIGINAL
)
@click.option(
    "--v4-structured", type=click.Path(path_type=Path), default=verify.DEFAULT_CANDIDATE
)
@click.option(
    "--triage", type=click.Path(path_type=Path), default=verify.DEFAULT_TRIAGE
)
@click.option(
    "--first-prompt",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_FIRST_PASS_PROMPT,
)
@click.option(
    "--verify-prompt",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_VERIFY_PROMPT,
)
@click.option(
    "--registry", type=click.Path(path_type=Path), default=verify.DEFAULT_REGISTRY
)
@click.option(
    "--first-status",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_FIRST_PASS_STATUS,
)
@click.option(
    "--first-manifest",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_FIRST_PASS_MANIFEST,
)
@click.option(
    "--first-checkpoint-root",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_FIRST_PASS_DIR,
)
@click.option(
    "--second-status",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_OUTPUT_DIR / "status.json",
)
@click.option(
    "--second-manifest",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_OUTPUT_DIR / "manifest.json",
)
@click.option(
    "--second-checkpoint-root",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_OUTPUT_DIR,
)
@click.option("--output", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT)
@click.option(
    "--publication-v1", type=click.Path(path_type=Path), default=DEFAULT_PUBLICATION_V1
)
@click.option("--write", "write_output", is_flag=True, default=False)
def main(
    original: Path,
    v4_structured: Path,
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
    output: Path,
    publication_v1: Path,
    write_output: bool,
) -> None:
    """Build or dry-run the provisional candidate without provider access.

    Raises:
        click.ClickException: If an input, checkpoint, or write target is unsafe.
    """
    configure_cli_logging()
    try:
        summary = build_provisional_prose_candidate(
            original=original,
            v4_structured=v4_structured,
            triage=triage,
            first_prompt=first_prompt,
            verify_prompt=verify_prompt,
            registry=registry,
            first_status=first_status,
            first_manifest=first_manifest,
            first_checkpoint_root=first_checkpoint_root,
            second_status=second_status,
            second_manifest=second_manifest,
            second_checkpoint_root=second_checkpoint_root,
            output=output,
            publication_v1=publication_v1,
            write_output=write_output,
        )
    except (
        ProvisionalProseCandidateError,
        ProseReviewV4DashboardError,
        verify.PatchVerificationCampaignError,
        ProxyBudgetError,
        ProxyPatchVerificationError,
        ValueError,
        OSError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def build_provisional_prose_candidate(
    *,
    original: Path,
    v4_structured: Path,
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
    output: Path,
    publication_v1: Path,
    write_output: bool,
) -> dict[str, JSONValue]:
    """Build a provisional preview from already accepted private reviews.

    Args:
        original: Published 100k Parquet input with original prose.
        v4_structured: Immutable structured v4 Parquet input.
        triage: First-pass triage JSON path pinned by the first manifest.
        first_prompt: First-pass prompt pinned by the first manifest.
        verify_prompt: Second-pass prompt pinned by the second manifest.
        registry: Pinned model registry path.
        first_status: First-pass private status JSON.
        first_manifest: First-pass private manifest JSON.
        first_checkpoint_root: First-pass private checkpoint root.
        second_status: Second-pass private status JSON.
        second_manifest: Second-pass private manifest JSON.
        second_checkpoint_root: Second-pass private checkpoint root.
        output: Provisional candidate Parquet path.
        publication_v1: Publication-v1 path that must not already exist.
        write_output: If true, write the candidate and adjacent report.

    Returns:
        Aggregate report without raw prose or raw identifiers.

    Raises:
        ProvisionalProseCandidateError: If inputs, status, checkpoints, or write
            targets are unsafe.
    """
    if write_output:
        _guard_write_targets(output=output, publication_v1=publication_v1)
    _require_private_files(
        files={
            "first status": first_status,
            "first manifest": first_manifest,
            "second status": second_status,
            "second manifest": second_manifest,
        }
    )

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
    actual_manifest = _load_json_object(path=second_manifest, label="second manifest")
    if actual_manifest != expected_manifest:
        raise ProvisionalProseCandidateError("Second-pass manifest pins do not match")
    status = _load_second_status(
        path=second_status,
        manifest=t.cast(dict[str, JSONValue], actual_manifest),
        available=len(loaded.rows),
    )
    row_index = {row.persona_hash: row for row in loaded.rows}
    processed_hashes = verify._status_hashes(status)
    unknown = sorted(set(processed_hashes) - set(row_index))
    if unknown:
        raise ProvisionalProseCandidateError("Second-pass status contains unknown IDs")

    config = verify._generation_config(prompt_path=verify_prompt)
    budget = verify._proxy_budget(
        paths=paths,
        prompt=loaded.verify_prompt,
        manifest=t.cast(dict[str, JSONValue], actual_manifest),
    )
    accepted_records: list[dict[str, object]] = []
    accepted_hashes: list[str] = []
    rejected_count = 0
    for persona_hash in processed_hashes:
        checkpoint_path = verify._checkpoint_path(
            output_dir=second_checkpoint_root, persona_hash=persona_hash
        )
        accepted = _checkpoint_is_accepted(path=checkpoint_path)
        if not accepted:
            rejected_count += 1
            continue
        row = row_index[persona_hash]
        result = run_proxy_patch_verification(
            row=row.original_row,
            candidate_row=row.candidate_row,
            changed_facts=row.changed_facts,
            proposed_text=row.proposed_text,
            patches=row.patches,
            first_checkpoint_sha256=row.first_checkpoint_sha256,
            prompt=loaded.verify_prompt,
            config=config,
            budget=budget,
            checkpoint_path=checkpoint_path,
            transport=_failing_transport(),
        )
        if not result.accepted:
            raise ProvisionalProseCandidateError(
                "Accepted second-pass checkpoint revalidated as rejected"
            )
        accepted_hashes.append(persona_hash)
        accepted_records.append(_second_pass_record(row=row, result=result))

    if (
        len(accepted_records) != status["accepted"]
        or rejected_count != status["rejected"]
    ):
        raise ProvisionalProseCandidateError("Second-pass status counts are stale")

    original_frame = pl.read_parquet(original)
    v4_frame = pl.read_parquet(v4_structured)
    first_records = _first_pass_records(loaded=loaded, original_frame=original_frame)
    source_metadata = _source_metadata(
        original=original,
        v4_structured=v4_structured,
        first_status=first_status,
        first_manifest=first_manifest,
        second_status=second_status,
        second_manifest=second_manifest,
        status=status,
        loaded=loaded,
    )
    preview, report = apply_reviewed_prose(
        original_published_frame=original_frame,
        v4_structured_frame=v4_frame,
        first_pass_records=first_records,
        second_pass_records=accepted_records,
        source_metadata=source_metadata,
        expected_row_count=original_frame.height,
    )
    summary = _summary(
        report=t.cast(dict[str, JSONValue], report),
        source_metadata=source_metadata,
        status=status,
        loaded=loaded,
        accepted_hashes=accepted_hashes,
        output=output,
        write_output=write_output,
    )
    if write_output:
        _write_outputs(output=output, preview=preview, report=summary)
    return summary


class ProvisionalProseCandidateError(RuntimeError):
    """Raised when a provisional preview cannot be built safely."""


def _require_private_files(*, files: dict[str, Path]) -> None:
    for label, path in files.items():
        if path.stat().st_mode & 0o777 != 0o600:
            raise ProvisionalProseCandidateError(f"{label} must be private (0600)")


def _load_json_object(*, path: Path, label: str) -> dict[str, JSONValue]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvisionalProseCandidateError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ProvisionalProseCandidateError(f"{label} is not a JSON object")
    return t.cast(dict[str, JSONValue], value)


def _load_second_status(
    *, path: Path, manifest: dict[str, JSONValue], available: int
) -> dict[str, object]:
    if path.stat().st_mode & 0o777 != 0o600:
        raise ProvisionalProseCandidateError(
            "Second-pass status must be private (0600)"
        )
    document = _load_json_object(path=path, label="second status")
    status = verify._validate_status(document)
    if status.get("manifest") != manifest:
        raise ProvisionalProseCandidateError(
            "Second-pass status does not match manifest"
        )
    if status["available"] != available:
        raise ProvisionalProseCandidateError("Second-pass available count is stale")
    if status["processed"] != status["accepted"] + status["rejected"]:
        raise ProvisionalProseCandidateError("Second-pass processed count is stale")
    if status["pending"] != status["available"] - status["processed"]:
        raise ProvisionalProseCandidateError("Second-pass pending count is stale")
    if len(verify._status_hashes(status)) != status["processed"]:
        raise ProvisionalProseCandidateError("Second-pass processed IDs are stale")
    return status


def _checkpoint_is_accepted(*, path: Path) -> bool:
    if not path.exists():
        raise ProvisionalProseCandidateError("Second-pass checkpoint is missing")
    if path.stat().st_mode & 0o777 != 0o600:
        raise ProvisionalProseCandidateError(
            "Second-pass checkpoint must be private (0600)"
        )
    checkpoint = _load_json_object(path=path, label="second checkpoint")
    digest = checkpoint.get("checkpoint_sha256")
    unsigned = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    if not isinstance(digest, str) or digest.lower() != sha256_text(
        canonical_json(unsigned)
    ):
        raise ProvisionalProseCandidateError(
            "Second-pass checkpoint checksum is invalid"
        )
    accepted = checkpoint.get("accepted")
    if not isinstance(accepted, bool):
        raise ProvisionalProseCandidateError(
            "Second-pass checkpoint lacks accepted flag"
        )
    return accepted


def _second_pass_record(
    *, row: verify.VerifyRow, result: ProsePatchVerificationResult
) -> dict[str, object]:
    record = t.cast(dict[str, object], result.model_dump(mode="json"))
    persona_id = _persona_id(row=row)
    record["persona_id"] = persona_id
    record["persona_hash"] = row.persona_hash
    record["status"] = "accepted" if record.get("accepted") is True else "rejected"
    return record


def _first_pass_records(
    *, loaded: verify.LoadedFirstPass, original_frame: pl.DataFrame
) -> list[dict[str, object]]:
    ids_by_hash = _ids_by_hash(frame=original_frame)
    patched_by_hash = {row.persona_hash: row for row in loaded.rows}
    records: list[dict[str, object]] = []
    for decision in loaded.decisions:
        persona_id = ids_by_hash.get(decision.persona_hash)
        if persona_id is None:
            raise ProvisionalProseCandidateError("First-pass decision has unknown ID")
        row = patched_by_hash.get(decision.persona_hash)
        record: dict[str, object] = {
            "persona_id": persona_id,
            "persona_hash": decision.persona_hash,
            "disposition": decision.disposition,
        }
        if row is not None:
            record.update(
                {
                    "checkpoint_sha256": row.first_checkpoint_sha256,
                    "original_persona_sha256": sha256_text(
                        str(row.original_row["persona"])
                    ),
                    "proposed_text_sha256": sha256_text(row.proposed_text),
                    "proposed_text": row.proposed_text,
                    "patches": row.patches,
                    "changed_facts": row.changed_facts,
                }
            )
        records.append(record)
    return records


def _source_metadata(
    *,
    original: Path,
    v4_structured: Path,
    first_status: Path,
    first_manifest: Path,
    second_status: Path,
    second_manifest: Path,
    status: dict[str, object],
    loaded: verify.LoadedFirstPass,
) -> dict[str, JSONValue]:
    return {
        "original_published_sha256": sha256_file(original),
        "v4_structured_sha256": sha256_file(v4_structured),
        "first_status_sha256": sha256_file(first_status),
        "first_manifest_sha256": sha256_file(first_manifest),
        "first_manifest_content_sha256": sha256_text(canonical_json(loaded.manifest)),
        "second_status_sha256": sha256_file(second_status),
        "second_manifest_sha256": sha256_file(second_manifest),
        "review_status": _review_status_counts(status=status, loaded=loaded),
    }


def _review_status_counts(
    *, status: dict[str, object], loaded: verify.LoadedFirstPass
) -> dict[str, int]:
    return {
        "accepted": int(status["accepted"]),
        "rejected": int(status["rejected"]),
        "pending": int(status["pending"]),
        "failed": int(status["failed"]),
        "needs_manual_review": loaded.manual,
        "unchanged_consistent": loaded.unchanged_consistent,
    }


def _summary(
    *,
    report: dict[str, JSONValue],
    source_metadata: dict[str, JSONValue],
    status: dict[str, object],
    loaded: verify.LoadedFirstPass,
    accepted_hashes: list[str],
    output: Path,
    write_output: bool,
) -> dict[str, JSONValue]:
    report_path = _report_path(output=output)
    first_counts = _first_counts(loaded=loaded)
    second_counts = _review_status_counts(status=status, loaded=loaded)
    return {
        **report,
        "dry_run": not write_output,
        "candidate_path": str(output),
        "report_path": str(report_path),
        "written": write_output,
        "release_ready": False,
        "not_release_ready": True,
        "accepted_patch_count": len(accepted_hashes),
        "manual_count": loaded.manual,
        "pending_count": int(status["pending"]),
        "unresolved_count": int(report["unresolved_count"]),
        "first_pass_counts": first_counts,
        "second_pass_counts": second_counts,
        "source_hashes": _source_hashes(metadata=source_metadata),
        "accepted_patch_hashes_sha256": sha256_text(
            canonical_json(sorted(accepted_hashes))
        ),
        "provisional_notice": (
            "PROVISIONAL only: this candidate is not release-ready, does not approve "
            "the reviewed proposals, and must not be published as v1."
        ),
    }


def _source_hashes(*, metadata: dict[str, JSONValue]) -> dict[str, str]:
    return {
        key: value
        for key, value in metadata.items()
        if key.endswith("sha256") and isinstance(value, str)
    }


def _first_counts(*, loaded: verify.LoadedFirstPass) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in loaded.decisions:
        counts[decision.disposition] = counts.get(decision.disposition, 0) + 1
    return dict(sorted(counts.items()))


def _ids_by_hash(*, frame: pl.DataFrame) -> dict[str, str]:
    if "persona_id" not in frame.columns:
        raise ProvisionalProseCandidateError("Published frame is missing persona_id")
    ids: dict[str, str] = {}
    for raw_id in frame.get_column("persona_id").to_list():
        if not isinstance(raw_id, str) or not raw_id:
            raise ProvisionalProseCandidateError(
                "Published frame has invalid persona_id"
            )
        persona_hash = sha256_text(raw_id)
        if persona_hash in ids:
            raise ProvisionalProseCandidateError("Published frame has duplicate IDs")
        ids[persona_hash] = raw_id
    return ids


def _persona_id(*, row: verify.VerifyRow) -> str:
    raw_id = row.original_row.get("persona_id")
    if not isinstance(raw_id, str) or sha256_text(raw_id) != row.persona_hash:
        raise ProvisionalProseCandidateError("First-pass row ID does not match hash")
    candidate_id = row.candidate_row.get("persona_id")
    if candidate_id != raw_id:
        raise ProvisionalProseCandidateError("Original and v4 row IDs do not match")
    return raw_id


def _guard_write_targets(*, output: Path, publication_v1: Path) -> None:
    report = _report_path(output=output)
    for path in (output, report, publication_v1):
        if path.exists():
            message = f"Refusing to overwrite existing file: {path}"
            raise ProvisionalProseCandidateError(message)


def _write_outputs(
    *, output: Path, preview: pl.DataFrame, report: dict[str, JSONValue]
) -> None:
    output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(output.parent, 0o700)
    report_path = _report_path(output=output)
    _write_new_parquet(path=output, frame=preview)
    _write_new_json(path=report_path, value=report)


def _write_new_parquet(*, path: Path, frame: pl.DataFrame) -> None:
    if path.exists():
        message = f"Refusing to overwrite existing file: {path}"
        raise ProvisionalProseCandidateError(message)
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


def _write_new_json(*, path: Path, value: dict[str, JSONValue]) -> None:
    if path.exists():
        message = f"Refusing to overwrite existing file: {path}"
        raise ProvisionalProseCandidateError(message)
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


def _report_path(*, output: Path) -> Path:
    return output.with_suffix(".json")


def _failing_transport() -> httpx.MockTransport:
    def respond(_request: httpx.Request) -> httpx.Response:
        raise ProvisionalProseCandidateError("Provider access is disabled offline")

    return httpx.MockTransport(respond)


if __name__ == "__main__":
    load_repository_environment()
    main()
