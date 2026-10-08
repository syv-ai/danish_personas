"""Run the private H90 Sol adjudication release scope."""

from __future__ import annotations

import collections.abc as c
import concurrent.futures as futures
import json
import re
import typing as t
from dataclasses import dataclass
from pathlib import Path

import click
import httpx
import polars as pl

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.environment import load_repository_environment
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.proxy_budget import (
    ADJUDICATION_MODEL_ENV,
    BASE_URL,
    SOL_ADJUDICATION_MODEL,
    SOL_ADJUDICATION_PURPOSE,
    ProxyBudget,
    ProxyBudgetError,
    require_runtime_model,
)
from danish_personas.generation.sol_adjudication import (
    SOL_ALLOWED_FACT_FIELDS,
    SOL_MAX_OUTPUT_TOKENS,
    SolAdjudicationError,
    SolAdjudicationResponse,
    SolAdjudicationResult,
    preflight_sol_adjudication_payload,
    run_sol_adjudication,
)
from danish_personas.io import canonical_json, sha256_file, sha256_text
from scripts import adjudicate_persona_release as first_pass

DEFAULT_H90_STATUS = first_pass.DEFAULT_ROOT / "persona-review-h90-v5" / "status.json"
DEFAULT_OUTPUT_DIR = first_pass.DEFAULT_ROOT / "sol-h90-final"
CAMPAIGN = "persona-sol-adjudication-h90-final"
EXPECTED_H90_ROWS = 506
MANIFEST_VERSION = 1
STATUS_VERSION = 1

JSONScalar: t.TypeAlias = str | int | float | bool | None
JSONDocument: t.TypeAlias = (
    JSONScalar | list["JSONDocument"] | dict[str, "JSONDocument"]
)
SolRunner: t.TypeAlias = c.Callable[
    [
        str,
        dict[str, object],
        str,
        GenerationConfig,
        ProxyBudget,
        Path,
        httpx.BaseTransport,
        dict[str, dict[str, object]],
        dict[str, object],
    ],
    SolAdjudicationResult,
]
Disposition: t.TypeAlias = t.Literal[
    "consistent", "patched", "unresolved", "privacy_blocked", "validation_failed"
]
LOCAL_VALIDATION_FAILURE = "Sol response failed bounded local validation retries"
ROW_ATTEMPT_LIFETIME_EXHAUSTED = "Per-row proxy attempt lifetime exhausted"
MAX_ROW_ATTEMPTS = 10
LOCAL_VALIDATION_COMPLETIONS = 3
MAX_LOCAL_VALIDATION_BATCHES = (
    MAX_ROW_ATTEMPTS + LOCAL_VALIDATION_COMPLETIONS - 1
) // LOCAL_VALIDATION_COMPLETIONS


@click.command()
@click.option(
    "--original", type=click.Path(path_type=Path), default=first_pass.DEFAULT_ORIGINAL
)
@click.option(
    "--candidate", type=click.Path(path_type=Path), default=first_pass.DEFAULT_CANDIDATE
)
@click.option(
    "--report", type=click.Path(path_type=Path), default=first_pass.DEFAULT_REPORT
)
@click.option(
    "--h90-status", type=click.Path(path_type=Path), default=DEFAULT_H90_STATUS
)
@click.option(
    "--prompt", type=click.Path(path_type=Path), default=first_pass.DEFAULT_PROMPT
)
@click.option(
    "--output-dir", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT_DIR
)
@click.option(
    "--registry", type=click.Path(path_type=Path), default=first_pass.DEFAULT_REGISTRY
)
@click.option(
    "--workers", type=click.IntRange(min=1, max=4), default=1, show_default=True
)
@click.option("--run", "execute", is_flag=True, default=False)
def main(
    original: Path,
    candidate: Path,
    report: Path,
    h90_status: Path,
    prompt: Path,
    output_dir: Path,
    registry: Path,
    workers: int,
    execute: bool,
) -> None:
    """Run or dry-run the H90 final Sol campaign.

    Raises:
        click.ClickException: If an input, resume, budget, or proxy boundary is unsafe.
    """
    configure_cli_logging()
    paths = H90Paths(
        original=original,
        candidate=candidate,
        report=report,
        h90_status=h90_status,
        prompt=prompt,
        output_dir=output_dir,
        registry=registry,
    )
    try:
        summary = run_h90_release_adjudication(
            paths=paths,
            execute=execute,
            workers=workers,
            expected_original_sha256=first_pass.EXPECTED_ORIGINAL_SHA256,
            expected_row_count=first_pass.EXPECTED_RELEASE_ROWS,
            expected_h90_count=EXPECTED_H90_ROWS,
            sol_runner=_run_sol_adjudication_adapter,
        )
    except (
        H90AdjudicationError,
        first_pass.PersonaReleaseAdjudicationError,
        ProxyBudgetError,
        SolAdjudicationError,
        OSError,
    ) as exc:
        raise click.ClickException(_safe_error_message(exc=exc)) from exc
    click.echo(json.dumps(summary, ensure_ascii=False, sort_keys=True))


@dataclass(frozen=True)
class H90Paths:
    """Private H90 campaign inputs and output paths."""

    original: Path
    candidate: Path
    report: Path
    h90_status: Path
    prompt: Path
    output_dir: Path
    registry: Path


@dataclass(frozen=True)
class H90State:
    """Loaded H90 scope and release inputs."""

    inputs: first_pass.ReleaseInputs
    status: dict[str, object]
    status_sha256: str
    status_json_sha256: str
    source_hash: str
    processed_hashes: list[str]
    processed_hashes_sha256: str


@dataclass(frozen=True)
class H90Selection:
    """Candidate-ordered H90 Sol selection."""

    rows: list[first_pass.ReleaseRow]
    provider_rows: list[first_pass.ReleaseRow]
    privacy_blocked_hashes: list[str]
    selected_hashes: list[str]
    selected_hashes_sha256: str
    ordered_input_hashes: list[str]
    ordered_id_sha256: str
    ordered_id_hashes_sha256: str
    candidate_preview_sha256: str


@dataclass(frozen=True)
class H90RowResult:
    """Sanitised row result for H90 status."""

    persona_hash: str
    disposition: Disposition
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None
    response_sha256: str | None = None


class H90AdjudicationError(Exception):
    """Raised when H90 adjudication must fail closed."""


def run_h90_release_adjudication(
    *,
    paths: H90Paths,
    execute: bool,
    workers: int,
    expected_original_sha256: str,
    expected_row_count: int,
    expected_h90_count: int,
    sol_runner: SolRunner,
) -> dict[str, object]:
    """Run or dry-run the H90 final Sol adjudication campaign.

    Args:
        paths: Immutable inputs and private output path.
        execute: If true, provider I/O is allowed. Defaults to dry-run in the CLI.
        workers: Worker count from one to four.
        expected_original_sha256: Required SHA-256 of the frozen baseline parquet.
        expected_row_count: Required ordered release row count.
        expected_h90_count: Required number of H90 hashes in the completed status.
        sol_runner: Injectable per-row Sol runner for offline callers.

    Returns:
        Machine-readable progress summary without raw persona IDs or prose.

    Raises:
        H90AdjudicationError: If the run cannot continue safely.
    """
    first_pass._require_worker_count(workers=workers)
    state = _load_h90_state(
        paths=paths,
        expected_original_sha256=expected_original_sha256,
        expected_row_count=expected_row_count,
        expected_h90_count=expected_h90_count,
    )
    raw_selection = _select_h90_rows(state=state)
    selection = _preflight_h90_selection(rows=raw_selection, state=state)
    manifest = _campaign_manifest(paths=paths, state=state, selection=selection)
    if not execute:
        return _dry_run_summary(manifest=manifest, selection=selection, workers=workers)

    first_pass._prepare_private_output(output_dir=paths.output_dir)
    first_pass._write_or_check_manifest(
        path=paths.output_dir / "manifest.json", manifest=manifest
    )
    status_path = paths.output_dir / "status.json"
    status = _load_or_create_status(
        status_path=status_path, output_dir=paths.output_dir, manifest=manifest
    )
    _record_privacy_blocks(status=status, status_path=status_path, selection=selection)
    config = first_pass._generation_config(prompt_path=paths.prompt)
    budget = _proxy_budget(paths=paths, state=state, selection=selection)
    _validate_processed_checkpoints(
        selection=selection,
        status=status,
        output_dir=paths.output_dir,
        prompt=state.inputs.prompt,
        config=config,
        budget=budget,
    )
    pending = _pending_rows(selection=selection, status=status)
    if pending:
        _process_pending(
            rows=pending,
            status=status,
            status_path=status_path,
            output_dir=paths.output_dir,
            prompt=state.inputs.prompt,
            config=config,
            budget=budget,
            workers=workers,
            sol_runner=sol_runner,
        )
    summary = _status_summary(status=status, selection=selection, workers=workers)
    if summary["pending"] != 0:
        raise H90AdjudicationError("H90 Sol campaign ended with pending rows")
    return summary


def _safe_error_message(*, exc: Exception) -> str:
    if isinstance(exc, H90AdjudicationError):
        return _scrub_private_text(value=str(exc))
    return first_pass._safe_error_message(exc=exc)


def _scrub_private_text(*, value: str) -> str:
    scrubbed = re.sub(r"[0-9a-f]{64}", "[hash]", value)
    return re.sub(r"raw[-_][A-Za-z0-9_.:-]+", "[id]", scrubbed)


def _load_h90_state(
    *,
    paths: H90Paths,
    expected_original_sha256: str,
    expected_row_count: int,
    expected_h90_count: int,
) -> H90State:
    first_pass._require_private_file(path=paths.h90_status, label="H90 status")
    inputs = _load_release_inputs_without_global_prose_check(
        paths=paths,
        expected_original_sha256=expected_original_sha256,
        expected_row_count=expected_row_count,
    )
    status = first_pass._load_json_object(path=paths.h90_status, label="H90 status")
    processed_hashes = _h90_processed_hashes(
        status=status, expected_h90_count=expected_h90_count
    )
    h90_status_sha256 = sha256_file(paths.h90_status)
    h90_status_json_sha256 = first_pass._hash_json(status)
    processed_hashes_sha256 = sha256_text(canonical_json(processed_hashes))
    source_hash = sha256_text(
        canonical_json(
            {
                "candidate_preview_sha256": inputs.candidate_preview_sha256,
                "h90_status_json_sha256": h90_status_json_sha256,
                "h90_status_sha256": h90_status_sha256,
                "ordered_id_hashes_sha256": inputs.ordered_id_hashes_sha256,
                "ordered_id_sha256": inputs.ordered_id_sha256,
                "processed_hashes_sha256": processed_hashes_sha256,
                "report_sha256": inputs.report_sha256,
            }
        )
    )
    return H90State(
        inputs=inputs,
        status=status,
        status_sha256=h90_status_sha256,
        status_json_sha256=h90_status_json_sha256,
        source_hash=source_hash,
        processed_hashes=processed_hashes,
        processed_hashes_sha256=processed_hashes_sha256,
    )


def _load_release_inputs_without_global_prose_check(
    *, paths: H90Paths, expected_original_sha256: str, expected_row_count: int
) -> first_pass.ReleaseInputs:
    original_sha256 = sha256_file(paths.original)
    if original_sha256 != expected_original_sha256:
        raise H90AdjudicationError("Frozen original parquet SHA-256 mismatch")
    original = pl.read_parquet(paths.original)
    candidate = pl.read_parquet(paths.candidate)
    first_pass._require_columns(frame=original, label="original")
    first_pass._require_columns(frame=candidate, label="candidate")
    first_pass._require_row_count(
        frame=original, expected=expected_row_count, label="original"
    )
    first_pass._require_row_count(
        frame=candidate, expected=expected_row_count, label="candidate"
    )
    original_ids = first_pass._ordered_ids(frame=original, label="original")
    candidate_ids = first_pass._ordered_ids(frame=candidate, label="candidate")
    if original_ids != candidate_ids:
        raise H90AdjudicationError("Ordered release IDs do not match")

    report = first_pass._load_json_object(path=paths.report, label="candidate report")
    preview_sha256 = first_pass._frame_hash(frame=candidate)
    if (
        first_pass._json_string(
            document=report, key="preview_sha256", label="candidate report"
        )
        != preview_sha256
    ):
        raise H90AdjudicationError("Candidate report preview SHA-256 mismatch")
    reported_count = report.get("row_count")
    if isinstance(reported_count, int) and reported_count != expected_row_count:
        raise H90AdjudicationError("Candidate report row count mismatch")
    source_hashes = report.get("source_hashes")
    if isinstance(source_hashes, dict):
        reported_original = source_hashes.get("original_v1_sha256")
        if isinstance(reported_original, str) and reported_original != original_sha256:
            raise H90AdjudicationError("Candidate report original SHA mismatch")

    prompt = paths.prompt.read_text(encoding="utf-8")
    first_pass._check_restricted_text(value=prompt, label="prompt")
    original_rows = original.to_dicts()
    candidate_rows = candidate.to_dicts()
    id_hashes = [sha256_text(persona_id) for persona_id in original_ids]
    return first_pass.ReleaseInputs(
        original_rows=original_rows,
        candidate_rows=candidate_rows,
        ordered_id_hashes=id_hashes,
        ordered_id_sha256=sha256_text(canonical_json(original_ids)),
        ordered_id_hashes_sha256=sha256_text(canonical_json(id_hashes)),
        candidate_preview_sha256=preview_sha256,
        report_sha256=sha256_file(paths.report),
        prompt=prompt,
    )


def _h90_processed_hashes(
    *, status: dict[str, object], expected_h90_count: int
) -> list[str]:
    hashes = status.get("processed_persona_hashes")
    if not isinstance(hashes, list):
        raise H90AdjudicationError("H90 status is missing processed hashes")
    processed: list[str] = []
    seen: set[str] = set()
    for value in hashes:
        if not isinstance(value, str) or not _is_sha256(value=value):
            raise H90AdjudicationError("H90 status contains an invalid hash")
        if value in seen:
            raise H90AdjudicationError("H90 status contains duplicate hashes")
        seen.add(value)
        processed.append(value)
    if len(processed) != expected_h90_count:
        raise H90AdjudicationError("H90 status hash count mismatch")
    processed_count = status.get("processed")
    pending = status.get("pending")
    if processed_count != expected_h90_count or pending != 0:
        raise H90AdjudicationError("H90 status is not a completed exact scope")
    return processed


def _select_h90_rows(*, state: H90State) -> list[first_pass.ReleaseRow]:
    h90_hashes = set(state.processed_hashes)
    rows: list[first_pass.ReleaseRow] = []
    for index, (original, candidate, persona_hash) in enumerate(
        zip(
            state.inputs.original_rows,
            state.inputs.candidate_rows,
            state.inputs.ordered_id_hashes,
            strict=True,
        )
    ):
        if persona_hash not in h90_hashes:
            continue
        rows.append(
            first_pass.ReleaseRow(
                index=index,
                persona_hash=persona_hash,
                original_row=original,
                candidate_row=candidate,
                changed_facts=first_pass._changed_facts(
                    original=original, candidate=candidate
                ),
            )
        )
    if len(rows) != len(state.processed_hashes):
        raise H90AdjudicationError("H90 hashes are not all present in the candidate")
    selected_hashes = [row.persona_hash for row in rows]
    if set(selected_hashes) != h90_hashes:
        raise H90AdjudicationError("H90 candidate selection does not match status")
    return rows


def _preflight_h90_selection(
    *, rows: list[first_pass.ReleaseRow], state: H90State
) -> H90Selection:
    provider_rows: list[first_pass.ReleaseRow] = []
    privacy_blocked_hashes: list[str] = []
    for row in rows:
        try:
            preflight_sol_adjudication_payload(
                original_persona=str(row.candidate_row[first_pass.PERSONA_FIELD]),
                candidate_row=row.candidate_row,
                prompt=state.inputs.prompt,
                changed_fact_hints=row.changed_facts,
                original_row=row.original_row,
            )
        except SolAdjudicationError:
            privacy_blocked_hashes.append(row.persona_hash)
        else:
            provider_rows.append(row)
    selected_hashes = [row.persona_hash for row in rows]
    return H90Selection(
        rows=rows,
        provider_rows=provider_rows,
        privacy_blocked_hashes=privacy_blocked_hashes,
        selected_hashes=selected_hashes,
        selected_hashes_sha256=sha256_text(canonical_json(selected_hashes)),
        ordered_input_hashes=list(state.inputs.ordered_id_hashes),
        ordered_id_sha256=state.inputs.ordered_id_sha256,
        ordered_id_hashes_sha256=state.inputs.ordered_id_hashes_sha256,
        candidate_preview_sha256=state.inputs.candidate_preview_sha256,
    )


def _campaign_manifest(
    *, paths: H90Paths, state: H90State, selection: H90Selection
) -> dict[str, JSONDocument]:
    schema_hash = sha256_text(
        canonical_json(SolAdjudicationResponse.provider_json_schema())
    )
    return {
        "version": MANIFEST_VERSION,
        "campaign": CAMPAIGN,
        "inputs": {
            "original": sha256_file(paths.original),
            "candidate": sha256_file(paths.candidate),
            "candidate_preview": state.inputs.candidate_preview_sha256,
            "candidate_report": state.inputs.report_sha256,
            "h90_status": state.status_sha256,
            "h90_status_json": state.status_json_sha256,
            "prompt": sha256_file(paths.prompt),
            "registry": sha256_file(paths.registry),
            "schema": schema_hash,
            "source": state.source_hash,
        },
        "model": SOL_ADJUDICATION_MODEL,
        "base_url": BASE_URL,
        "reasoning_effort": "none",
        "max_tokens": SOL_MAX_OUTPUT_TOKENS,
        "maximum_http_attempts": 5,
        "budget_purpose": SOL_ADJUDICATION_PURPOSE,
        "allowed_facts": sorted(SOL_ALLOWED_FACT_FIELDS),
        "selection": {
            "ordered_release_rows": len(selection.ordered_input_hashes),
            "ordered_id_sha256": selection.ordered_id_sha256,
            "ordered_id_hashes_sha256": selection.ordered_id_hashes_sha256,
            "ordered_input_hashes_sha256": sha256_text(
                canonical_json(selection.ordered_input_hashes)
            ),
            "h90_status_processed_hashes_sha256": (state.processed_hashes_sha256),
            "selected_rows": len(selection.rows),
            "selected_hashes": selection.selected_hashes,
            "selected_hashes_sha256": selection.selected_hashes_sha256,
            "provider_rows": len(selection.provider_rows),
            "privacy_blocked": len(selection.privacy_blocked_hashes),
            "privacy_blocked_hashes_sha256": sha256_text(
                canonical_json(selection.privacy_blocked_hashes)
            ),
        },
    }


def _proxy_budget(
    *, paths: H90Paths, state: H90State, selection: H90Selection
) -> ProxyBudget:
    del selection
    # The Sol ledger is immutable and shared with the first-pass campaign;
    # the H90 selection is pinned separately by its private manifest.
    return first_pass._proxy_budget(
        paths=first_pass.ReleasePaths(
            original=paths.original,
            candidate=paths.candidate,
            report=paths.report,
            prompt=paths.prompt,
            output_dir=paths.output_dir,
            registry=paths.registry,
        ),
        inputs=state.inputs,
        manifest={},
    )


def _dry_run_summary(
    *, manifest: dict[str, JSONDocument], selection: H90Selection, workers: int
) -> dict[str, object]:
    return {
        "dry_run": True,
        "run_required": True,
        "campaign": CAMPAIGN,
        "manifest_sha256": first_pass._hash_json(manifest),
        "selected": len(selection.rows),
        "provider_selected": len(selection.provider_rows),
        "total": len(selection.rows),
        "pending": len(selection.provider_rows),
        "processed": 0,
        "consistent": 0,
        "patched": 0,
        "unresolved": 0,
        "privacy_blocked": len(selection.privacy_blocked_hashes),
        "validation_failed": 0,
        "workers": workers,
    }


def _load_or_create_status(
    *, status_path: Path, output_dir: Path, manifest: dict[str, JSONDocument]
) -> dict[str, object]:
    manifest_sha256 = first_pass._hash_json(manifest)
    if not status_path.exists():
        status: dict[str, object] = {
            "version": STATUS_VERSION,
            "campaign": CAMPAIGN,
            "manifest_sha256": manifest_sha256,
            "counts": _zero_counts(),
            "processed": [],
        }
        first_pass._write_status(path=status_path, status=status)
        return status
    first_pass._require_private_file(path=status_path, label="status")
    status = first_pass._load_json_object(path=status_path, label="status")
    if status.get("version") != STATUS_VERSION or status.get("campaign") != CAMPAIGN:
        raise H90AdjudicationError("Status version does not match")
    if status.get("manifest_sha256") != manifest_sha256:
        raise H90AdjudicationError("Status manifest binding does not match")
    _validate_status(status=status, output_dir=output_dir)
    return status


def _validate_status(*, status: dict[str, object], output_dir: Path) -> None:
    processed = _processed_records(status=status)
    seen: set[str] = set()
    for record in processed:
        persona_hash = record["persona_hash"]
        if persona_hash in seen:
            raise H90AdjudicationError("Status contains duplicate rows")
        seen.add(persona_hash)
        if record["disposition"] == "privacy_blocked":
            continue
        checkpoint = first_pass._checkpoint_path(
            output_dir=output_dir, persona_hash=persona_hash
        )
        if record["disposition"] == "validation_failed":
            if checkpoint.exists():
                raise H90AdjudicationError(
                    "Status validation-failed row unexpectedly has a checkpoint"
                )
            continue
        if record.get("checkpoint") != first_pass._checkpoint_reference(
            output_dir=output_dir, path=checkpoint
        ):
            raise H90AdjudicationError("Status checkpoint reference does not match")
        if not checkpoint.exists():
            raise H90AdjudicationError("Status references missing checkpoint")
        first_pass._require_private_file(path=checkpoint, label="checkpoint")
        if sha256_file(checkpoint) != record.get("checkpoint_sha256"):
            raise H90AdjudicationError("Status checkpoint hash does not match")
        if first_pass._checkpoint_response_hash(path=checkpoint) != record.get(
            "response_sha256"
        ):
            raise H90AdjudicationError("Status response hash does not match")
    if not _status_counts_match(
        observed=status.get("counts"),
        expected=_counts_from_processed(processed=processed),
    ):
        raise H90AdjudicationError("Status counts are inconsistent")


def _record_privacy_blocks(
    *, status: dict[str, object], status_path: Path, selection: H90Selection
) -> None:
    changed = False
    processed = _processed_records(status=status)
    existing = {record["persona_hash"]: record for record in processed}
    privacy_hashes = set(selection.privacy_blocked_hashes)
    for persona_hash, record in existing.items():
        disposition = record["disposition"]
        if disposition == "privacy_blocked" and persona_hash not in privacy_hashes:
            raise H90AdjudicationError("Status privacy block no longer matches inputs")
        if disposition != "privacy_blocked" and persona_hash in privacy_hashes:
            raise H90AdjudicationError("Status sent a privacy-blocked row")
    for persona_hash in selection.privacy_blocked_hashes:
        if persona_hash in existing:
            continue
        processed.append(
            {"persona_hash": persona_hash, "disposition": "privacy_blocked"}
        )
        changed = True
    if changed:
        status["processed"] = processed
        status["counts"] = _counts_from_processed(processed=processed)
        status["selected_total"] = len(selection.rows)
        first_pass._write_status(path=status_path, status=status)


def _validate_processed_checkpoints(
    *,
    selection: H90Selection,
    status: dict[str, object],
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
) -> None:
    rows_by_hash = {row.persona_hash: row for row in selection.provider_rows}
    privacy_hashes = set(selection.privacy_blocked_hashes)
    transport = first_pass._failing_transport()
    try:
        for record in _processed_records(status=status):
            persona_hash = record["persona_hash"]
            if record["disposition"] == "privacy_blocked":
                if persona_hash not in privacy_hashes:
                    raise H90AdjudicationError(
                        "Status references a row outside the privacy blocks"
                    )
                continue
            row = rows_by_hash.get(persona_hash)
            if row is None:
                raise H90AdjudicationError(
                    "Status references a row outside the provider selection"
                )
            if record["disposition"] == "validation_failed":
                checkpoint_path = first_pass._checkpoint_path(
                    output_dir=output_dir, persona_hash=row.persona_hash
                )
                if checkpoint_path.exists():
                    raise H90AdjudicationError(
                        "Validation-failed Sol row unexpectedly has a checkpoint"
                    )
                continue
            checkpoint_path = first_pass._checkpoint_path(
                output_dir=output_dir, persona_hash=row.persona_hash
            )
            try:
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
            except (RuntimeError, SolAdjudicationError) as exc:
                raise H90AdjudicationError(
                    "Processed Sol checkpoint does not match current inputs"
                ) from exc
            if result.disposition != record["disposition"]:
                raise H90AdjudicationError(
                    "Processed Sol checkpoint disposition does not match status"
                )
    finally:
        transport.close()


def _pending_rows(
    *, selection: H90Selection, status: dict[str, object]
) -> list[first_pass.ReleaseRow]:
    processed = {record["persona_hash"] for record in _processed_records(status=status)}
    status["selected_total"] = len(selection.rows)
    return [row for row in selection.provider_rows if row.persona_hash not in processed]


def _process_pending(
    *,
    rows: list[first_pass.ReleaseRow],
    status: dict[str, object],
    status_path: Path,
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    workers: int,
    sol_runner: SolRunner,
) -> None:
    row_iter = iter(rows)
    future_map: dict[futures.Future[H90RowResult], first_pass.ReleaseRow] = {}
    stop_exc: Exception | None = None
    executor = futures.ThreadPoolExecutor(max_workers=workers)
    try:
        _submit_futures(
            row_iter=row_iter,
            future_map=future_map,
            executor=executor,
            limit=workers,
            output_dir=output_dir,
            prompt=prompt,
            config=config,
            budget=budget,
            sol_runner=sol_runner,
        )
        while future_map:
            done, _ = futures.wait(future_map, return_when=futures.FIRST_COMPLETED)
            future = next(iter(done))
            row = future_map.pop(future)
            if future.cancelled():
                continue
            completed_exc = _record_completed_future(
                future=future, row=row, status=status, status_path=status_path
            )
            if completed_exc is not None:
                if stop_exc is None:
                    stop_exc = completed_exc
                first_pass._cancel_not_started(future_map=future_map)
                continue
            if stop_exc is None and not first_pass._has_completed_future(
                future_map=future_map
            ):
                _submit_futures(
                    row_iter=row_iter,
                    future_map=future_map,
                    executor=executor,
                    limit=workers,
                    output_dir=output_dir,
                    prompt=prompt,
                    config=config,
                    budget=budget,
                    sol_runner=sol_runner,
                )
    finally:
        if stop_exc is not None:
            first_pass._cancel_not_started(future_map=future_map)
        executor.shutdown(wait=stop_exc is None, cancel_futures=stop_exc is not None)
    if stop_exc is not None:
        raise H90AdjudicationError(
            "Sol adjudication stopped before all selected rows were resolved"
        ) from stop_exc


def _submit_futures(
    *,
    row_iter: c.Iterator[first_pass.ReleaseRow],
    future_map: dict[futures.Future[H90RowResult], first_pass.ReleaseRow],
    executor: futures.ThreadPoolExecutor,
    limit: int,
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    sol_runner: SolRunner,
) -> None:
    while len(future_map) < limit:
        try:
            row = next(row_iter)
        except StopIteration:
            return
        future_map[
            executor.submit(
                _run_one_row,
                row=row,
                output_dir=output_dir,
                prompt=prompt,
                config=config,
                budget=budget,
                sol_runner=sol_runner,
            )
        ] = row


def _run_one_row(
    *,
    row: first_pass.ReleaseRow,
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    sol_runner: SolRunner,
) -> H90RowResult:
    transport = httpx.HTTPTransport()
    checkpoint_path = first_pass._checkpoint_path(
        output_dir=output_dir, persona_hash=row.persona_hash
    )
    try:
        for batch in range(MAX_LOCAL_VALIDATION_BATCHES):
            try:
                result = sol_runner(
                    str(row.candidate_row[first_pass.PERSONA_FIELD]),
                    row.candidate_row,
                    prompt,
                    config,
                    budget,
                    checkpoint_path,
                    transport,
                    row.changed_facts,
                    row.original_row,
                )
            except SolAdjudicationError as exc:
                if not _is_local_validation_failure(exc=exc):
                    raise
                if batch + 1 == MAX_LOCAL_VALIDATION_BATCHES:
                    return _validation_failed_result(
                        persona_hash=row.persona_hash, checkpoint_path=checkpoint_path
                    )
                continue
            except ProxyBudgetError as exc:
                if _is_row_attempt_lifetime_exhausted(exc=exc):
                    return _validation_failed_result(
                        persona_hash=row.persona_hash, checkpoint_path=checkpoint_path
                    )
                raise
            return H90RowResult(
                persona_hash=row.persona_hash,
                disposition=result.disposition,
                checkpoint=first_pass._checkpoint_reference(
                    output_dir=output_dir, path=checkpoint_path
                ),
                checkpoint_sha256=sha256_file(checkpoint_path),
                response_sha256=first_pass._checkpoint_response_hash(
                    path=checkpoint_path
                ),
            )
    finally:
        transport.close()
    return _validation_failed_result(
        persona_hash=row.persona_hash, checkpoint_path=checkpoint_path
    )


def _record_completed_future(
    *,
    future: futures.Future[H90RowResult],
    row: first_pass.ReleaseRow,
    status: dict[str, object],
    status_path: Path,
) -> Exception | None:
    try:
        result = future.result()
    except Exception as exc:
        return exc
    if result.persona_hash != row.persona_hash:
        return H90AdjudicationError("Completed row binding does not match")
    _record_status_result(status=status, result=result)
    first_pass._write_status(path=status_path, status=status)
    return None


def _record_status_result(*, status: dict[str, object], result: H90RowResult) -> None:
    processed = _processed_records(status=status)
    if any(record["persona_hash"] == result.persona_hash for record in processed):
        return
    record: dict[str, str] = {
        "persona_hash": result.persona_hash,
        "disposition": result.disposition,
    }
    if result.disposition not in {"privacy_blocked", "validation_failed"}:
        if (
            result.checkpoint is None
            or result.checkpoint_sha256 is None
            or result.response_sha256 is None
        ):
            raise H90AdjudicationError("Completed provider row lacks a checkpoint")
        record.update(
            {
                "checkpoint": result.checkpoint,
                "checkpoint_sha256": result.checkpoint_sha256,
                "response_sha256": result.response_sha256,
            }
        )
    processed.append(record)
    status["processed"] = processed
    status["counts"] = _counts_from_processed(processed=processed)


def _status_summary(
    *, status: dict[str, object], selection: H90Selection, workers: int
) -> dict[str, object]:
    processed = _processed_records(status=status)
    counts = _counts_from_processed(processed=processed)
    selected = len(selection.rows)
    return {
        "dry_run": False,
        "campaign": CAMPAIGN,
        "manifest_sha256": status["manifest_sha256"],
        "selected": selected,
        "provider_selected": len(selection.provider_rows),
        "total": selected,
        "pending": max(selected - len(processed), 0),
        "processed": min(len(processed), selected),
        "consistent": counts["consistent"],
        "patched": counts["patched"],
        "unresolved": counts["unresolved"],
        "privacy_blocked": counts["privacy_blocked"],
        "validation_failed": counts["validation_failed"],
        "workers": workers,
        "budget": None,
    }


def _processed_records(*, status: dict[str, object]) -> list[dict[str, str]]:
    processed = status.get("processed")
    if not isinstance(processed, list):
        raise H90AdjudicationError("Status processed rows are invalid")
    records: list[dict[str, str]] = []
    for item in processed:
        if not isinstance(item, dict):
            raise H90AdjudicationError("Status processed rows are invalid")
        persona_hash = item.get("persona_hash")
        disposition = item.get("disposition")
        if not isinstance(persona_hash, str) or not _is_sha256(value=persona_hash):
            raise H90AdjudicationError("Status processed rows are invalid")
        if disposition not in {
            "consistent",
            "patched",
            "unresolved",
            "privacy_blocked",
            "validation_failed",
        }:
            raise H90AdjudicationError("Status processed rows are invalid")
        record = {"persona_hash": persona_hash, "disposition": str(disposition)}
        if disposition in {"privacy_blocked", "validation_failed"}:
            if disposition == "validation_failed" and any(
                key in item
                for key in ("checkpoint", "checkpoint_sha256", "response_sha256")
            ):
                raise H90AdjudicationError("Status processed rows are invalid")
            records.append(record)
            continue
        checkpoint = item.get("checkpoint")
        checkpoint_sha256 = item.get("checkpoint_sha256")
        response_sha256 = item.get("response_sha256")
        if (
            not isinstance(checkpoint, str)
            or not _is_sha256(value=checkpoint_sha256)
            or not _is_sha256(value=response_sha256)
        ):
            raise H90AdjudicationError("Status processed rows are invalid")
        record.update(
            {
                "checkpoint": checkpoint,
                "checkpoint_sha256": str(checkpoint_sha256),
                "response_sha256": str(response_sha256),
            }
        )
        records.append(record)
    return records


def _counts_from_processed(*, processed: list[dict[str, str]]) -> dict[str, int]:
    counts = _zero_counts()
    for record in processed:
        disposition = record["disposition"]
        counts[disposition] += 1
    return counts


def _zero_counts() -> dict[str, int]:
    return {
        "consistent": 0,
        "patched": 0,
        "unresolved": 0,
        "privacy_blocked": 0,
        "validation_failed": 0,
    }


def _status_counts_match(*, observed: object, expected: dict[str, int]) -> bool:
    if observed == expected:
        return True
    if expected["validation_failed"] != 0 or not isinstance(observed, dict):
        return False
    legacy_expected = dict(expected)
    legacy_expected.pop("validation_failed")
    return observed == legacy_expected


def _is_local_validation_failure(*, exc: SolAdjudicationError) -> bool:
    return str(exc) == LOCAL_VALIDATION_FAILURE


def _is_row_attempt_lifetime_exhausted(*, exc: ProxyBudgetError) -> bool:
    return str(exc) == ROW_ATTEMPT_LIFETIME_EXHAUSTED


def _validation_failed_result(
    *, persona_hash: str, checkpoint_path: Path
) -> H90RowResult:
    if checkpoint_path.exists():
        raise H90AdjudicationError(
            "Validation-failed Sol row unexpectedly has a checkpoint"
        )
    return H90RowResult(persona_hash=persona_hash, disposition="validation_failed")


def _is_sha256(*, value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _run_sol_adjudication_adapter(
    original_persona: str,
    candidate_row: dict[str, object],
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    checkpoint_path: Path,
    transport: httpx.BaseTransport,
    changed_fact_hints: dict[str, dict[str, object]],
    original_row: dict[str, object],
) -> SolAdjudicationResult:
    require_runtime_model(ADJUDICATION_MODEL_ENV)
    return run_sol_adjudication(
        original_persona=original_persona,
        candidate_row=candidate_row,
        prompt=prompt,
        config=config,
        budget=budget,
        checkpoint_path=checkpoint_path,
        transport=transport,
        changed_fact_hints=changed_fact_hints,
        original_row=original_row,
    )


if __name__ == "__main__":
    load_repository_environment()
    main()
