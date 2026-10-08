"""Run the private Sol follow-up pass for unresolved release rows."""

from __future__ import annotations

import collections.abc as c
import json
import re
import typing as t
from dataclasses import dataclass
from pathlib import Path

import click
import httpx

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
from scripts import adjudicate_persona_release as first_pass

DEFAULT_PARENT_OUTPUT_DIR = first_pass.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_DIR = first_pass.DEFAULT_ROOT / "sol-followup-pilot32"
DEFAULT_MAX_ROWS = 9
CAMPAIGN = "persona-sol-adjudication-v2-unresolved-followup"
MANIFEST_VERSION = 1
STATUS_VERSION = 1
FOLLOWUP_MODE: t.Literal["unresolved_followup"] = "unresolved_followup"

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
    "--prompt", type=click.Path(path_type=Path), default=first_pass.DEFAULT_PROMPT
)
@click.option(
    "--parent-output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_PARENT_OUTPUT_DIR,
)
@click.option(
    "--output-dir", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT_DIR
)
@click.option(
    "--registry", type=click.Path(path_type=Path), default=first_pass.DEFAULT_REGISTRY
)
@click.option("--max-rows", type=click.IntRange(min=1), default=DEFAULT_MAX_ROWS)
@click.option("--all-unresolved", is_flag=True, default=False)
@click.option(
    "--workers", type=click.IntRange(min=1, max=2), default=1, show_default=True
)
@click.option("--run", "execute", is_flag=True, default=False)
def main(
    original: Path,
    candidate: Path,
    report: Path,
    prompt: Path,
    parent_output_dir: Path,
    output_dir: Path,
    registry: Path,
    max_rows: int | None,
    all_unresolved: bool,
    workers: int,
    execute: bool,
) -> None:
    """Run or dry-run the unresolved-row Sol follow-up campaign.

    Raises:
        click.ClickException: If an input, resume, budget, or proxy boundary is unsafe.
    """
    configure_cli_logging()
    paths = FollowupPaths(
        original=original,
        candidate=candidate,
        report=report,
        prompt=prompt,
        parent_output_dir=parent_output_dir,
        output_dir=output_dir,
        registry=registry,
    )
    try:
        summary = run_unresolved_followup(
            paths=paths,
            execute=execute,
            max_rows=None if all_unresolved else max_rows,
            workers=workers,
            expected_original_sha256=first_pass.EXPECTED_ORIGINAL_SHA256,
            expected_row_count=first_pass.EXPECTED_RELEASE_ROWS,
            sol_runner=_run_sol_followup_adapter,
        )
    except (
        FollowupAdjudicationError,
        first_pass.PersonaReleaseAdjudicationError,
        ProxyBudgetError,
        SolAdjudicationError,
        OSError,
    ) as exc:
        raise click.ClickException(_safe_error_message(exc=exc)) from exc
    click.echo(json.dumps(summary, ensure_ascii=False, sort_keys=True))


@dataclass(frozen=True)
class FollowupPaths:
    """Private follow-up campaign inputs and output paths."""

    original: Path
    candidate: Path
    report: Path
    prompt: Path
    parent_output_dir: Path
    output_dir: Path
    registry: Path


def _safe_error_message(*, exc: Exception) -> str:
    if isinstance(exc, FollowupAdjudicationError):
        return _scrub_private_text(value=str(exc))
    return first_pass._safe_error_message(exc=exc)


def _scrub_private_text(*, value: str) -> str:
    scrubbed = re.sub(r"[0-9a-f]{64}", "[hash]", value)
    scrubbed = re.sub(r"raw[-_][A-Za-z0-9_.:-]+", "[id]", scrubbed)
    return scrubbed


def run_unresolved_followup(
    *,
    paths: FollowupPaths,
    execute: bool,
    max_rows: int | None,
    workers: int,
    expected_original_sha256: str,
    expected_row_count: int,
    sol_runner: SolRunner,
) -> dict[str, object]:
    """Run or dry-run the unresolved follow-up campaign.

    Args:
        paths: Immutable inputs, parent artefacts, and private output path.
        execute: If true, provider I/O is allowed. Defaults to dry-run in the CLI.
        max_rows: Optional ordered prefix bound for pilot processing.
        workers: Worker count from one to two.
        expected_original_sha256: Required SHA-256 of the frozen baseline parquet.
        expected_row_count: Required ordered release row count.
        sol_runner: Injectable per-row Sol runner for offline tests.

    Returns:
        Machine-readable progress summary without raw persona IDs or prose.

    Raises:
        FollowupAdjudicationError: If the run cannot continue safely.
    """
    _require_worker_count(workers=workers)
    try:
        parent = _load_validated_parent_state(
            paths=paths,
            expected_original_sha256=expected_original_sha256,
            expected_row_count=expected_row_count,
        )
        selection = _select_unresolved_rows(parent=parent)
        manifest = _followup_manifest(paths=paths, parent=parent, selection=selection)
        scoped_rows = _scoped_rows(selection=selection, max_rows=max_rows)
        first_pass._preflight_selected_rows(
            rows=scoped_rows, prompt=parent.inputs.prompt
        )
    except first_pass.PersonaReleaseAdjudicationError as exc:
        raise FollowupAdjudicationError(str(exc)) from exc
    if not execute:
        return _dry_run_summary(
            manifest=manifest, selection=selection, max_rows=max_rows, workers=workers
        )

    first_pass._prepare_private_output(output_dir=paths.output_dir)
    first_pass._write_or_check_manifest(
        path=paths.output_dir / "manifest.json", manifest=manifest
    )
    status_path = paths.output_dir / "status.json"
    status = _load_or_create_status(
        status_path=status_path, output_dir=paths.output_dir, manifest=manifest
    )
    config = first_pass._generation_config(prompt_path=paths.prompt)
    budget = first_pass._proxy_budget(
        paths=_release_paths(paths=paths, output_dir=paths.output_dir),
        inputs=parent.inputs,
        manifest=manifest,
    )
    _validate_processed_followup_checkpoints(
        selection=selection,
        status=status,
        output_dir=paths.output_dir,
        prompt=parent.inputs.prompt,
        config=config,
        budget=budget,
    )
    pending = _pending_rows(selection=selection, status=status, max_rows=max_rows)
    if pending:
        try:
            first_pass._process_pending(
                rows=pending,
                status=status,
                status_path=status_path,
                output_dir=paths.output_dir,
                prompt=parent.inputs.prompt,
                config=config,
                budget=budget,
                workers=workers,
                sol_runner=sol_runner,
            )
        except first_pass.PersonaReleaseAdjudicationError as exc:
            _record_failure_status(
                status=status,
                status_path=status_path,
                selected=len(_scoped_rows(selection=selection, max_rows=max_rows)),
                message=str(exc),
            )
            raise FollowupAdjudicationError(str(exc)) from exc
    summary = _status_summary(status=status, max_rows=max_rows, workers=workers)
    if max_rows is None and summary["pending"] != 0:
        raise FollowupAdjudicationError("Sol follow-up ended with pending rows")
    return summary


class FollowupAdjudicationError(Exception):
    """Raised when follow-up adjudication must fail closed."""


def _source_hash(*, inputs: first_pass.ReleaseInputs) -> str:
    return sha256_text(
        canonical_json(
            {
                "candidate_preview_sha256": inputs.candidate_preview_sha256,
                "ordered_id_sha256": inputs.ordered_id_sha256,
                "ordered_id_hashes_sha256": inputs.ordered_id_hashes_sha256,
                "report_sha256": inputs.report_sha256,
            }
        )
    )


def _load_or_create_status(
    *, status_path: Path, output_dir: Path, manifest: dict[str, JSONDocument]
) -> dict[str, object]:
    manifest_sha256 = first_pass._hash_json(manifest)
    if not status_path.exists():
        status: dict[str, object] = {
            "version": STATUS_VERSION,
            "campaign": CAMPAIGN,
            "manifest_sha256": manifest_sha256,
            "counts": {"consistent": 0, "patched": 0, "unresolved": 0},
            "processed": [],
        }
        first_pass._write_status(path=status_path, status=status)
        return status
    first_pass._require_private_file(path=status_path, label="status")
    status = first_pass._load_json_object(path=status_path, label="status")
    if status.get("version") != STATUS_VERSION or status.get("campaign") != CAMPAIGN:
        raise FollowupAdjudicationError("Status version does not match")
    if status.get("manifest_sha256") != manifest_sha256:
        raise FollowupAdjudicationError("Status manifest binding does not match")
    first_pass._validate_status(status=status, output_dir=output_dir)
    return status


@dataclass(frozen=True)
class ParentState:
    """Validated first-pass artefact state."""

    inputs: first_pass.ReleaseInputs
    selection: first_pass.ReleaseSelection
    manifest: dict[str, object]
    manifest_sha256: str
    status: dict[str, object]
    status_sha256: str
    processed: list[dict[str, str]]


def _load_validated_parent_state(
    *, paths: FollowupPaths, expected_original_sha256: str, expected_row_count: int
) -> ParentState:
    inputs = first_pass.load_release_inputs(
        paths=_release_paths(paths=paths, output_dir=paths.parent_output_dir),
        expected_original_sha256=expected_original_sha256,
        expected_row_count=expected_row_count,
    )
    selection = first_pass.select_release_rows(inputs=inputs)
    expected_manifest = first_pass._campaign_manifest(
        paths=_release_paths(paths=paths, output_dir=paths.parent_output_dir),
        inputs=inputs,
        selection=selection,
    )
    parent_manifest_path = paths.parent_output_dir / "manifest.json"
    parent_status_path = paths.parent_output_dir / "status.json"
    first_pass._require_private_file(path=parent_manifest_path, label="parent manifest")
    first_pass._require_private_file(path=parent_status_path, label="parent status")
    parent_manifest = first_pass._load_json_object(
        path=parent_manifest_path, label="parent manifest"
    )
    if parent_manifest != expected_manifest:
        raise FollowupAdjudicationError("Parent manifest binding does not match inputs")
    parent_status = first_pass._load_json_object(
        path=parent_status_path, label="parent status"
    )
    if parent_status.get("manifest_sha256") != first_pass._hash_json(parent_manifest):
        raise FollowupAdjudicationError("Parent status manifest binding does not match")
    first_pass._validate_status(
        status=parent_status, output_dir=paths.parent_output_dir
    )
    config = first_pass._generation_config(prompt_path=paths.prompt)
    budget = first_pass._proxy_budget(
        paths=_release_paths(paths=paths, output_dir=paths.parent_output_dir),
        inputs=inputs,
        manifest=parent_manifest,
    )
    first_pass._validate_processed_checkpoints(
        selection=selection,
        status=parent_status,
        output_dir=paths.parent_output_dir,
        prompt=inputs.prompt,
        config=config,
        budget=budget,
    )
    return ParentState(
        inputs=inputs,
        selection=selection,
        manifest=parent_manifest,
        manifest_sha256=sha256_file(parent_manifest_path),
        status=parent_status,
        status_sha256=sha256_file(parent_status_path),
        processed=first_pass._processed_records(status=parent_status),
    )


def _release_paths(
    *, paths: FollowupPaths, output_dir: Path
) -> first_pass.ReleasePaths:
    return first_pass.ReleasePaths(
        original=paths.original,
        candidate=paths.candidate,
        report=paths.report,
        prompt=paths.prompt,
        output_dir=output_dir,
        registry=paths.registry,
    )


def _record_failure_status(
    *, status: dict[str, object], status_path: Path, selected: int, message: str
) -> None:
    processed = len(first_pass._processed_records(status=status))
    status["last_failure"] = {
        "message": _scrub_private_text(value=message),
        "processed": min(processed, selected),
        "pending": max(selected - min(processed, selected), 0),
    }
    first_pass._write_status(path=status_path, status=status)


def _require_worker_count(*, workers: int) -> None:
    if workers not in {1, 2}:
        raise FollowupAdjudicationError("Worker count must be one or two")


@dataclass(frozen=True)
class FollowupSelection:
    """Unresolved parent rows selected for follow-up adjudication."""

    rows: list[first_pass.ReleaseRow]
    row_bindings: list[dict[str, str | int]]
    ordered_input_hashes: list[str]
    ordered_id_sha256: str
    ordered_id_hashes_sha256: str


def _dry_run_summary(
    *,
    selection: FollowupSelection,
    manifest: dict[str, JSONDocument],
    max_rows: int | None,
    workers: int,
) -> dict[str, object]:
    selected = first_pass._bounded_total(total=len(selection.rows), max_rows=max_rows)
    return {
        "dry_run": True,
        "run_required": True,
        "campaign": CAMPAIGN,
        "manifest_sha256": first_pass._hash_json(manifest),
        "selected": selected,
        "total": len(selection.rows),
        "pending": selected,
        "consistent": 0,
        "patched": 0,
        "unresolved": 0,
        "processed": 0,
        "max_rows": max_rows,
        "workers": workers,
    }


def _followup_manifest(
    *, paths: FollowupPaths, parent: ParentState, selection: FollowupSelection
) -> dict[str, JSONDocument]:
    first_manifest_inputs = parent.manifest.get("inputs")
    if not isinstance(first_manifest_inputs, dict):
        raise FollowupAdjudicationError("Parent manifest inputs are invalid")
    return {
        "version": MANIFEST_VERSION,
        "campaign": CAMPAIGN,
        "parent": {
            "campaign": first_pass.CAMPAIGN,
            "manifest_sha256": parent.manifest_sha256,
            "manifest_json_sha256": first_pass._hash_json(parent.manifest),
            "status_sha256": parent.status_sha256,
            "status_json_sha256": first_pass._hash_json(parent.status),
            "processed": len(parent.processed),
        },
        "inputs": {
            "original": str(first_manifest_inputs["original"]),
            "candidate": str(first_manifest_inputs["candidate"]),
            "candidate_report": str(first_manifest_inputs["candidate_report"]),
            "candidate_preview": str(first_manifest_inputs["candidate_preview"]),
            "prompt": sha256_file(paths.prompt),
            "schema": str(first_manifest_inputs["schema"]),
            "registry": sha256_file(paths.registry),
            "source": _source_hash(inputs=parent.inputs),
        },
        "model": str(parent.manifest["model"]),
        "base_url": str(parent.manifest["base_url"]),
        "reasoning_effort": str(parent.manifest["reasoning_effort"]),
        "max_tokens": t.cast(int, parent.manifest["max_tokens"]),
        "maximum_http_attempts": t.cast(int, parent.manifest["maximum_http_attempts"]),
        "budget_purpose": str(parent.manifest["budget_purpose"]),
        "allowed_facts": t.cast(list[JSONDocument], parent.manifest["allowed_facts"]),
        "adjudication_mode": FOLLOWUP_MODE,
        "selection": {
            "unresolved_rows": len(selection.rows),
            "ordered_id_sha256": selection.ordered_id_sha256,
            "ordered_id_hashes_sha256": selection.ordered_id_hashes_sha256,
            "ordered_input_hashes": selection.ordered_input_hashes,
            "row_bindings": selection.row_bindings,
        },
    }


def _pending_rows(
    *, selection: FollowupSelection, status: dict[str, object], max_rows: int | None
) -> list[first_pass.ReleaseRow]:
    processed = {
        record["persona_hash"]
        for record in first_pass._processed_records(status=status)
    }
    rows = _scoped_rows(selection=selection, max_rows=max_rows)
    status["selected_total"] = len(selection.rows)
    return [row for row in rows if row.persona_hash not in processed]


def _scoped_rows(
    *, selection: FollowupSelection, max_rows: int | None
) -> list[first_pass.ReleaseRow]:
    return selection.rows[
        : first_pass._bounded_total(total=len(selection.rows), max_rows=max_rows)
    ]


def _select_unresolved_rows(*, parent: ParentState) -> FollowupSelection:
    parent_rows = {row.persona_hash: row for row in parent.selection.rows}
    rows: list[first_pass.ReleaseRow] = []
    bindings: list[dict[str, str | int]] = []
    for record in parent.processed:
        if record["disposition"] != "unresolved":
            continue
        parent_row = parent_rows.get(record["persona_hash"])
        if parent_row is None:
            raise FollowupAdjudicationError(
                "Parent status references a row outside the release inputs"
            )
        followup_id = _followup_checkpoint_id(parent_record=record)
        rows.append(
            first_pass.ReleaseRow(
                index=parent_row.index,
                persona_hash=followup_id,
                original_row=parent_row.original_row,
                candidate_row=parent_row.candidate_row,
                changed_facts=parent_row.changed_facts,
            )
        )
        bindings.append(
            {
                "row_index": parent_row.index,
                "row_hash": record["persona_hash"],
                "first_pass_checkpoint_sha256": record["checkpoint_sha256"],
                "first_pass_response_sha256": record["response_sha256"],
                "followup_checkpoint_id": followup_id,
            }
        )
    return FollowupSelection(
        rows=rows,
        row_bindings=bindings,
        ordered_input_hashes=list(parent.inputs.ordered_id_hashes),
        ordered_id_sha256=parent.inputs.ordered_id_sha256,
        ordered_id_hashes_sha256=parent.inputs.ordered_id_hashes_sha256,
    )


def _followup_checkpoint_id(*, parent_record: dict[str, str]) -> str:
    return sha256_text(
        canonical_json(
            {
                "campaign": CAMPAIGN,
                "mode": FOLLOWUP_MODE,
                "row_hash": parent_record["persona_hash"],
                "first_pass_checkpoint_sha256": parent_record["checkpoint_sha256"],
                "first_pass_response_sha256": parent_record["response_sha256"],
            }
        )
    )


def _status_summary(
    *, status: dict[str, object], max_rows: int | None, workers: int
) -> dict[str, object]:
    summary = first_pass._status_summary(
        status=status, max_rows=max_rows, workers=workers
    )
    summary["campaign"] = CAMPAIGN
    return summary


def _validate_processed_followup_checkpoints(
    *,
    selection: FollowupSelection,
    status: dict[str, object],
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
) -> None:
    rows_by_hash = {row.persona_hash: row for row in selection.rows}
    transport = first_pass._failing_transport()
    try:
        for record in first_pass._processed_records(status=status):
            row = rows_by_hash.get(record["persona_hash"])
            if row is None:
                raise FollowupAdjudicationError(
                    "Status references a row outside the follow-up selection"
                )
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
                    adjudication_mode=FOLLOWUP_MODE,
                )
            except (RuntimeError, SolAdjudicationError) as exc:
                raise FollowupAdjudicationError(
                    "Processed Sol follow-up checkpoint does not match current inputs"
                ) from exc
            if result.disposition != record["disposition"]:
                raise FollowupAdjudicationError(
                    "Processed Sol follow-up disposition does not match status"
                )
    finally:
        transport.close()


def _run_sol_followup_adapter(
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
        adjudication_mode=FOLLOWUP_MODE,
    )


load_repository_environment()


if __name__ == "__main__":
    main()
