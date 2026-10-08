"""Compose the private minimal v2 release candidate from completed Sol results."""

from __future__ import annotations

import hashlib
import io
import json
import os
import typing as t
from pathlib import Path

import click
import polars as pl

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.environment import load_repository_environment
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.proxy_budget import ProxyBudget, ProxyBudgetError
from danish_personas.generation.sol_adjudication import (
    SolAdjudicationError,
    SolAdjudicationResult,
    run_sol_adjudication,
)
from danish_personas.io import canonical_json, sha256_file, sha256_text
from scripts import adjudicate_h90_release as h90
from scripts import adjudicate_persona_release as first_pass

DEFAULT_ROOT = first_pass.DEFAULT_ROOT
DEFAULT_ORIGINAL = first_pass.DEFAULT_ORIGINAL
DEFAULT_CANDIDATE = first_pass.DEFAULT_CANDIDATE
DEFAULT_CANDIDATE_REPORT = first_pass.DEFAULT_REPORT
DEFAULT_H90_STATUS = h90.DEFAULT_H90_STATUS
DEFAULT_H90_OUTPUT_DIR = h90.DEFAULT_OUTPUT_DIR
DEFAULT_PROMPT = first_pass.DEFAULT_PROMPT
DEFAULT_REGISTRY = first_pass.DEFAULT_REGISTRY
DEFAULT_SOURCE_BUNDLE = Path("data/processed/6e27b5c08fbeae79")
DEFAULT_OUTPUT = DEFAULT_ROOT / "final-v2-candidate.parquet"
DEFAULT_MANIFEST_OUTPUT = DEFAULT_ROOT / "final-v2-candidate.manifest.json"
DEFAULT_REPORT_OUTPUT = DEFAULT_ROOT / "final-v2-candidate.report.json"
EXPECTED_V5_CHANGED_PROSE_ROWS = 3_239

JSONScalar: t.TypeAlias = str | int | float | bool | None
JSONDocument: t.TypeAlias = (
    JSONScalar | list["JSONDocument"] | dict[str, "JSONDocument"]
)


@click.command()
@click.option("--original", type=click.Path(path_type=Path), default=DEFAULT_ORIGINAL)
@click.option("--candidate", type=click.Path(path_type=Path), default=DEFAULT_CANDIDATE)
@click.option(
    "--candidate-report",
    type=click.Path(path_type=Path),
    default=DEFAULT_CANDIDATE_REPORT,
)
@click.option(
    "--h90-status", type=click.Path(path_type=Path), default=DEFAULT_H90_STATUS
)
@click.option(
    "--h90-output-dir", type=click.Path(path_type=Path), default=DEFAULT_H90_OUTPUT_DIR
)
@click.option("--prompt", type=click.Path(path_type=Path), default=DEFAULT_PROMPT)
@click.option("--registry", type=click.Path(path_type=Path), default=DEFAULT_REGISTRY)
@click.option(
    "--source-bundle", type=click.Path(path_type=Path), default=DEFAULT_SOURCE_BUNDLE
)
@click.option("--output", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT)
@click.option(
    "--manifest-output",
    type=click.Path(path_type=Path),
    default=DEFAULT_MANIFEST_OUTPUT,
)
@click.option(
    "--report-output", type=click.Path(path_type=Path), default=DEFAULT_REPORT_OUTPUT
)
def main(
    original: Path,
    candidate: Path,
    candidate_report: Path,
    h90_status: Path,
    h90_output_dir: Path,
    prompt: Path,
    registry: Path,
    source_bundle: Path,
    output: Path,
    manifest_output: Path,
    report_output: Path,
) -> None:
    """Compose the final private candidate without provider or upload access."""
    configure_cli_logging()
    try:
        summary = compose_minimal_v2_candidate(
            original=original,
            candidate=candidate,
            candidate_report=candidate_report,
            h90_status=h90_status,
            h90_output_dir=h90_output_dir,
            prompt=prompt,
            registry=registry,
            source_bundle=source_bundle,
            output=output,
            manifest_output=manifest_output,
            report_output=report_output,
        )
    except (
        MinimalV2CandidateError,
        h90.H90AdjudicationError,
        first_pass.PersonaReleaseAdjudicationError,
        ProxyBudgetError,
        SolAdjudicationError,
        OSError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(summary, ensure_ascii=False, sort_keys=True))


class MinimalV2CandidateError(RuntimeError):
    """Raised when the minimal v2 candidate cannot be composed safely."""


def compose_minimal_v2_candidate(
    *,
    original: Path,
    candidate: Path,
    candidate_report: Path,
    h90_status: Path,
    h90_output_dir: Path,
    prompt: Path,
    registry: Path,
    source_bundle: Path,
    output: Path,
    manifest_output: Path,
    report_output: Path,
    expected_row_count: int = first_pass.EXPECTED_RELEASE_ROWS,
    expected_original_sha256: str = first_pass.EXPECTED_ORIGINAL_SHA256,
    expected_h90_count: int = h90.EXPECTED_H90_ROWS,
    expected_v5_changed_prose_rows: int = EXPECTED_V5_CHANGED_PROSE_ROWS,
) -> dict[str, JSONDocument]:
    """Compose a private candidate from an already completed H90 campaign.

    Args:
        original:
            SHA-pinned published v1 Parquet.
        candidate:
            Current merged private v5 candidate.
        candidate_report:
            Report adjacent to the current candidate.
        h90_status:
            Completed H90 scope status used to select rows.
        h90_output_dir:
            Completed final H90 Sol campaign directory.
        prompt:
            Prompt path pinned by the H90 campaign.
        registry:
            Registry path pinned by the H90 campaign.
        source_bundle:
            Prepared source-bundle path recorded for later validation.
        output:
            Private final candidate Parquet output.
        manifest_output:
            Private vetted-patch manifest output.
        report_output:
            Private aggregate report output.
        expected_row_count (optional):
            Required ordered release row count.
        expected_original_sha256 (optional):
            Required SHA-256 of the original Parquet.
        expected_h90_count (optional):
            Required H90 selected-row count.
        expected_v5_changed_prose_rows (optional):
            Required current-v5 prose delta count.

    Returns:
        Summary containing only paths, counts, and checksums.
    """
    _guard_output_paths(paths=(output, manifest_output, report_output))
    campaign = _load_completed_h90_campaign(
        original=original,
        candidate=candidate,
        candidate_report=candidate_report,
        h90_status=h90_status,
        h90_output_dir=h90_output_dir,
        prompt=prompt,
        registry=registry,
        expected_original_sha256=expected_original_sha256,
        expected_row_count=expected_row_count,
        expected_h90_count=expected_h90_count,
    )
    original_frame = pl.read_parquet(original)
    candidate_frame = pl.read_parquet(candidate)
    _validate_ordered_ids(
        original_frame=original_frame,
        candidate_frame=candidate_frame,
        expected_row_count=expected_row_count,
    )
    original_persona_by_hash = _persona_by_hash(frame=original_frame)
    candidate_changed_hashes = _changed_prose_hashes(
        frame=candidate_frame, original_persona_by_hash=original_persona_by_hash
    )
    if len(candidate_changed_hashes) != expected_v5_changed_prose_rows:
        raise MinimalV2CandidateError("Current v5 changed-prose count mismatch")

    final_frame, compose_report = _apply_h90_results(
        candidate_frame=candidate_frame,
        original_persona_by_hash=original_persona_by_hash,
        campaign=campaign,
    )
    _assert_only_persona_differs(left=final_frame, right=candidate_frame)
    allowed_hashes = _changed_prose_hashes(
        frame=final_frame, original_persona_by_hash=original_persona_by_hash
    )
    parquet_bytes = _parquet_bytes(frame=final_frame)
    candidate_sha256 = hashlib.sha256(parquet_bytes).hexdigest()
    manifest = _vetted_manifest(
        allowed_hashes=allowed_hashes,
        original=original,
        candidate=candidate,
        candidate_report=candidate_report,
        h90_status=h90_status,
        h90_output_dir=h90_output_dir,
        source_bundle=source_bundle,
        candidate_sha256=candidate_sha256,
        campaign=campaign,
        compose_report=compose_report,
    )
    report = _private_report(
        original=original,
        candidate=candidate,
        candidate_report=candidate_report,
        h90_status=h90_status,
        h90_output_dir=h90_output_dir,
        source_bundle=source_bundle,
        output=output,
        manifest_output=manifest_output,
        candidate_sha256=candidate_sha256,
        allowed_hashes=allowed_hashes,
        candidate_changed_hashes=candidate_changed_hashes,
        campaign=campaign,
        compose_report=compose_report,
    )
    _write_private_bytes(path=output, content=parquet_bytes)
    _write_private_json_checked(path=manifest_output, payload=manifest)
    _write_private_json_checked(path=report_output, payload=report)
    return {
        "candidate_sha256": candidate_sha256,
        "final_changed_prose_rows": len(allowed_hashes),
        "h90_patched_applied": compose_report["h90_patched_applied"],
        "h90_preserved_unresolved_or_privacy_blocked": compose_report[
            "h90_preserved_unresolved_or_privacy_blocked"
        ],
        "manifest_output": manifest_output.as_posix(),
        "output": output.as_posix(),
        "report_output": report_output.as_posix(),
    }


def _load_completed_h90_campaign(
    *,
    original: Path,
    candidate: Path,
    candidate_report: Path,
    h90_status: Path,
    h90_output_dir: Path,
    prompt: Path,
    registry: Path,
    expected_original_sha256: str,
    expected_row_count: int,
    expected_h90_count: int,
) -> dict[str, object]:
    paths = h90.H90Paths(
        original=original,
        candidate=candidate,
        report=candidate_report,
        h90_status=h90_status,
        prompt=prompt,
        output_dir=h90_output_dir,
        registry=registry,
    )
    state = h90._load_h90_state(
        paths=paths,
        expected_original_sha256=expected_original_sha256,
        expected_row_count=expected_row_count,
        expected_h90_count=expected_h90_count,
    )
    selection = h90._preflight_h90_selection(
        rows=h90._select_h90_rows(state=state), state=state
    )
    expected_manifest = h90._campaign_manifest(
        paths=paths, state=state, selection=selection
    )
    manifest_path = h90_output_dir / "manifest.json"
    first_pass._require_private_file(path=manifest_path, label="H90 manifest")
    saved_manifest = first_pass._load_json_object(
        path=manifest_path, label="H90 manifest"
    )
    if saved_manifest != expected_manifest:
        raise MinimalV2CandidateError("H90 manifest binding does not match")

    status_path = h90_output_dir / "status.json"
    first_pass._require_private_file(path=status_path, label="H90 status")
    status = first_pass._load_json_object(path=status_path, label="H90 status")
    if (
        status.get("version") != h90.STATUS_VERSION
        or status.get("campaign") != h90.CAMPAIGN
    ):
        raise MinimalV2CandidateError("H90 status version does not match")
    if status.get("manifest_sha256") != first_pass._hash_json(expected_manifest):
        raise MinimalV2CandidateError("H90 status manifest binding does not match")
    h90._validate_status(status=status, output_dir=h90_output_dir)
    summary = h90._status_summary(status=status, selection=selection, workers=1)
    if summary["pending"] != 0:
        raise MinimalV2CandidateError("H90 campaign is not complete")

    config = first_pass._generation_config(prompt_path=prompt)
    budget = h90._proxy_budget(paths=paths, state=state, selection=selection)
    results = _resume_h90_results(
        selection=selection,
        status=status,
        output_dir=h90_output_dir,
        prompt=state.inputs.prompt,
        config=config,
        budget=budget,
    )
    processed_hashes = {
        record["persona_hash"] for record in h90._processed_records(status=status)
    }
    if processed_hashes != set(selection.selected_hashes):
        raise MinimalV2CandidateError("H90 status does not cover the exact selection")
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "selection": selection,
        "state": state,
        "status": status,
        "status_path": status_path,
        "status_sha256": sha256_file(status_path),
        "summary": summary,
        "results": results,
    }


def _resume_h90_results(
    *,
    selection: h90.H90Selection,
    status: dict[str, object],
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
) -> dict[str, SolAdjudicationResult | None]:
    rows_by_hash = {row.persona_hash: row for row in selection.provider_rows}
    privacy_hashes = set(selection.privacy_blocked_hashes)
    results: dict[str, SolAdjudicationResult | None] = {}
    transport = first_pass._failing_transport()
    try:
        for record in h90._processed_records(status=status):
            persona_hash = record["persona_hash"]
            disposition = record["disposition"]
            if disposition == "privacy_blocked":
                if persona_hash not in privacy_hashes:
                    raise MinimalV2CandidateError(
                        "H90 privacy-blocked row is outside the selection"
                    )
                results[persona_hash] = None
                continue
            row = rows_by_hash.get(persona_hash)
            if row is None:
                raise MinimalV2CandidateError("H90 status references an unknown row")
            checkpoint_path = first_pass._checkpoint_path(
                output_dir=output_dir, persona_hash=row.persona_hash
            )
            result = run_sol_adjudication(
                original_persona=str(row.candidate_row[first_pass.PERSONA_FIELD]),
                candidate_row=row.candidate_row,
                prompt=prompt,
                config=config,
                budget=budget,
                checkpoint_path=checkpoint_path,
                transport=transport,
                changed_fact_hints=row.changed_facts,
                original_row=row.original_row,
            )
            if result.disposition != disposition:
                raise MinimalV2CandidateError(
                    "H90 checkpoint disposition does not match status"
                )
            results[persona_hash] = result
    finally:
        transport.close()
    return results


def _apply_h90_results(
    *,
    candidate_frame: pl.DataFrame,
    original_persona_by_hash: dict[str, str],
    campaign: dict[str, object],
) -> tuple[pl.DataFrame, dict[str, JSONDocument]]:
    selection = t.cast(h90.H90Selection, campaign["selection"])
    status = t.cast(dict[str, object], campaign["status"])
    results = t.cast(dict[str, SolAdjudicationResult | None], campaign["results"])
    rows = candidate_frame.to_dicts()
    selected_by_hash = {row.persona_hash: row for row in selection.rows}
    dispositions = _disposition_counts(status=status)
    patched_applied = 0
    patched_changed_from_candidate = 0
    for persona_hash, result in results.items():
        if result is None or result.disposition != "patched":
            continue
        row = selected_by_hash[persona_hash]
        if rows[row.index] != row.candidate_row:
            raise MinimalV2CandidateError(
                "Candidate row binding changed during compose"
            )
        current = rows[row.index][first_pass.PERSONA_FIELD]
        first_pass._check_restricted_text(
            value=result.proposed_text, label="patched persona prose"
        )
        rows[row.index][first_pass.PERSONA_FIELD] = result.proposed_text
        patched_applied += 1
        if result.proposed_text != current:
            patched_changed_from_candidate += 1
    final_frame = pl.DataFrame(rows, schema=candidate_frame.schema)
    final_changed_hashes = _changed_prose_hashes(
        frame=final_frame, original_persona_by_hash=original_persona_by_hash
    )
    preserved = dispositions["unresolved"] + dispositions["privacy_blocked"]
    return final_frame, {
        "h90_consistent_preserved": dispositions["consistent"],
        "h90_patched_applied": patched_applied,
        "h90_patched_changed_from_candidate": patched_changed_from_candidate,
        "h90_unresolved_preserved": dispositions["unresolved"],
        "h90_privacy_blocked_preserved": dispositions["privacy_blocked"],
        "h90_preserved_unresolved_or_privacy_blocked": preserved,
        "final_changed_prose_rows": len(final_changed_hashes),
        "final_changed_persona_hashes_sha256": sha256_text(
            canonical_json(final_changed_hashes)
        ),
    }


def _vetted_manifest(
    *,
    allowed_hashes: list[str],
    original: Path,
    candidate: Path,
    candidate_report: Path,
    h90_status: Path,
    h90_output_dir: Path,
    source_bundle: Path,
    candidate_sha256: str,
    campaign: dict[str, object],
    compose_report: dict[str, JSONDocument],
) -> dict[str, JSONDocument]:
    return {
        "version": 1,
        "status": "vetted",
        "review_basis": (
            "automated_source_and_exact_patch_checks_no_human_semantic_certification"
        ),
        "allowed_persona_id_hashes": allowed_hashes,
        "counts": {
            "allowed_changed_prose_rows": len(allowed_hashes),
            "h90_patched_applied": compose_report["h90_patched_applied"],
            "h90_unresolved_preserved": compose_report["h90_unresolved_preserved"],
            "h90_privacy_blocked_preserved": compose_report[
                "h90_privacy_blocked_preserved"
            ],
        },
        "outputs": {"candidate_sha256": candidate_sha256},
        "source_hashes": {
            "candidate_report_sha256": sha256_file(candidate_report),
            "current_v5_candidate_sha256": sha256_file(candidate),
            "h90_campaign_manifest_sha256": str(campaign["manifest_sha256"]),
            "h90_campaign_status_sha256": str(campaign["status_sha256"]),
            "h90_scope_status_sha256": sha256_file(h90_status),
            "original_v1_sha256": sha256_file(original),
        },
        "source_paths": {
            "h90_output_dir": h90_output_dir.as_posix(),
            "source_bundle": source_bundle.as_posix(),
        },
    }


def _private_report(
    *,
    original: Path,
    candidate: Path,
    candidate_report: Path,
    h90_status: Path,
    h90_output_dir: Path,
    source_bundle: Path,
    output: Path,
    manifest_output: Path,
    candidate_sha256: str,
    allowed_hashes: list[str],
    candidate_changed_hashes: list[str],
    campaign: dict[str, object],
    compose_report: dict[str, JSONDocument],
) -> dict[str, JSONDocument]:
    summary = t.cast(dict[str, object], campaign["summary"])
    unresolved = compose_report["h90_unresolved_preserved"]
    privacy_blocked = compose_report["h90_privacy_blocked_preserved"]
    return {
        "version": 1,
        "status": "vetted",
        "review_basis": (
            "automated_source_and_exact_patch_checks_no_human_semantic_certification"
        ),
        "candidate_sha256": candidate_sha256,
        "candidate_output": output.as_posix(),
        "vetted_patch_manifest_output": manifest_output.as_posix(),
        "source_bundle_for_later_validation": source_bundle.as_posix(),
        "h90_unresolved_or_privacy_blocked_preserved_count": (
            t.cast(int, unresolved) + t.cast(int, privacy_blocked)
        ),
        "h90_unresolved_preserved_count": unresolved,
        "h90_privacy_blocked_preserved_count": privacy_blocked,
        "counts": {
            "rows": summary["total"],
            "existing_v5_changed_prose_rows": len(candidate_changed_hashes),
            "final_changed_prose_rows": len(allowed_hashes),
            "h90_consistent_preserved": compose_report[
                "h90_consistent_preserved"
            ],
            "h90_patched_applied": compose_report["h90_patched_applied"],
            "h90_patched_changed_from_candidate": compose_report[
                "h90_patched_changed_from_candidate"
            ],
            "h90_unresolved_preserved": unresolved,
            "h90_privacy_blocked_preserved": privacy_blocked,
        },
        "hashes": {
            "allowed_persona_hashes_sha256": sha256_text(
                canonical_json(allowed_hashes)
            ),
            "candidate_report_sha256": sha256_file(candidate_report),
            "current_v5_candidate_sha256": sha256_file(candidate),
            "existing_v5_changed_hashes_sha256": sha256_text(
                canonical_json(candidate_changed_hashes)
            ),
            "final_changed_hashes_sha256": sha256_text(canonical_json(allowed_hashes)),
            "h90_campaign_manifest_sha256": str(campaign["manifest_sha256"]),
            "h90_campaign_status_sha256": str(campaign["status_sha256"]),
            "h90_scope_status_sha256": sha256_file(h90_status),
            "original_v1_sha256": sha256_file(original),
        },
        "h90_output_dir": h90_output_dir.as_posix(),
        "validation_note": (
            "Source validation and full distribution validation were not run by this "
            "composer."
        ),
    }


def _validate_ordered_ids(
    *,
    original_frame: pl.DataFrame,
    candidate_frame: pl.DataFrame,
    expected_row_count: int,
) -> None:
    if (
        original_frame.height != expected_row_count
        or candidate_frame.height != expected_row_count
    ):
        raise MinimalV2CandidateError("Release row count mismatch")
    original_ids = _ordered_ids(frame=original_frame)
    candidate_ids = _ordered_ids(frame=candidate_frame)
    if original_ids != candidate_ids:
        raise MinimalV2CandidateError("Ordered release IDs do not match")
    if len(set(original_ids)) != expected_row_count:
        raise MinimalV2CandidateError("Original release IDs are not unique")


def _ordered_ids(*, frame: pl.DataFrame) -> list[str]:
    if first_pass.ID_FIELD not in frame.columns:
        raise MinimalV2CandidateError("Frame is missing persona IDs")
    return [
        str(value) if value is not None else ""
        for value in frame[first_pass.ID_FIELD]
    ]


def _persona_by_hash(*, frame: pl.DataFrame) -> dict[str, str]:
    values: dict[str, str] = {}
    for row in frame.select([first_pass.ID_FIELD, first_pass.PERSONA_FIELD]).iter_rows(
        named=True
    ):
        persona_id = (
            str(row[first_pass.ID_FIELD])
            if row[first_pass.ID_FIELD] is not None
            else ""
        )
        persona = row[first_pass.PERSONA_FIELD]
        if not persona_id or not isinstance(persona, str) or not persona.strip():
            raise MinimalV2CandidateError("Frame contains blank ID or prose")
        persona_hash = sha256_text(persona_id)
        if persona_hash in values:
            raise MinimalV2CandidateError("Frame contains duplicate persona IDs")
        values[persona_hash] = persona
    return values


def _changed_prose_hashes(
    *, frame: pl.DataFrame, original_persona_by_hash: dict[str, str]
) -> list[str]:
    changed: list[str] = []
    for row in frame.select([first_pass.ID_FIELD, first_pass.PERSONA_FIELD]).iter_rows(
        named=True
    ):
        persona_id = (
            str(row[first_pass.ID_FIELD])
            if row[first_pass.ID_FIELD] is not None
            else ""
        )
        persona_hash = sha256_text(persona_id)
        original_persona = original_persona_by_hash.get(persona_hash)
        if original_persona is None:
            raise MinimalV2CandidateError("Candidate contains an unknown persona ID")
        if row[first_pass.PERSONA_FIELD] != original_persona:
            changed.append(persona_hash)
    return sorted(changed)


def _assert_only_persona_differs(*, left: pl.DataFrame, right: pl.DataFrame) -> None:
    if left.schema != right.schema or left.height != right.height:
        raise MinimalV2CandidateError("Final candidate schema changed")
    for column in left.columns:
        if column == first_pass.PERSONA_FIELD:
            continue
        if left.get_column(column).to_list() != right.get_column(column).to_list():
            raise MinimalV2CandidateError("Final candidate changed non-prose fields")


def _disposition_counts(*, status: dict[str, object]) -> dict[str, int]:
    counts = {"consistent": 0, "patched": 0, "privacy_blocked": 0, "unresolved": 0}
    for record in h90._processed_records(status=status):
        counts[record["disposition"]] += 1
    return counts


def _guard_output_paths(*, paths: tuple[Path, ...]) -> None:
    if len(set(paths)) != len(paths):
        raise MinimalV2CandidateError("Output paths must be distinct")
    for path in paths:
        if path.is_symlink():
            raise MinimalV2CandidateError("Output path must not be a symlink")
        if path.exists() and not path.is_file():
            raise MinimalV2CandidateError("Output path must be a regular file")


def _parquet_bytes(*, frame: pl.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


def _write_private_json_checked(
    *, path: Path, payload: dict[str, JSONDocument]
) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    _write_private_bytes(path=path, content=(content + "\n").encode("utf-8"))


def _write_private_bytes(*, path: Path, content: bytes) -> None:
    digest = hashlib.sha256(content).hexdigest()
    if path.exists():
        if sha256_file(path) != digest:
            raise MinimalV2CandidateError("Existing output checksum mismatch")
        os.chmod(path, 0o600)
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    load_repository_environment()
    main()
