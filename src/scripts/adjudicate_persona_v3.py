"""Run privacy-safe, resumable private Sol review for the v3 candidate."""

from __future__ import annotations

import concurrent.futures as futures
import json
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeAlias, cast

import click
import httpx
import polars as pl

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.environment import load_repository_environment
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.proxy_budget import (
    ADJUDICATION_MODEL_ENV,
    BASE_URL,
    SOL_ADJUDICATION_LEDGER_MAX_TOKENS,
    SOL_ADJUDICATION_MODEL,
    V3_EXTENDED_ADJUDICATION_PURPOSE,
    ProxyBudget,
    ProxyBudgetError,
    require_runtime_model,
)
from danish_personas.generation.sol_adjudication import (
    SOL_ALLOWED_FACT_FIELDS,
    SolAdjudicationError,
    SolAdjudicationResponse,
    preflight_sol_adjudication_payload,
    run_sol_adjudication,
)
from danish_personas.io import canonical_json, sha256_file, sha256_text
from danish_personas.release.candidate_validation import validate_release_candidate
from danish_personas.release.generated_checks import check_generated_fields
from scripts.adjudicate_persona_release import (
    _check_restricted_text,
    _check_rows_private,
    _prepare_private_output,
    _write_private_json,
)

JSONScalar: TypeAlias = str | int | float | bool | None
JSONDocument: TypeAlias = JSONScalar | list["JSONDocument"] | dict[str, "JSONDocument"]

ROOT = Path("/tmp/danish-personas-audit")
ORIGINAL = ROOT / "hf-v2-foreign-hotfix-remote/data/train-00000-of-00001.parquet"
CANDIDATE = ROOT / "v3-private/repaired-final.parquet"
REPAIR = ROOT / "v3-private/repair-manifest-final.json"
REVIEW = ROOT / "persona-review-v4/checkpoints"
H90 = ROOT / "sol-h90-final/status.json"
FOLLOWUP = ROOT / "sol-followup-pilot32/status.json"
PROMPT = Path("config/persona-sol-adjudication-da.md")
OUTPUT = ROOT / "v3-private/campaign-extended"
REGISTRY = Path.home() / ".pi/agent/models-store.json"
EXPECTED_ROWS = 100_000
ID_FIELD = "persona_id"
TEXT_FIELD = "persona"
CAMPAIGN = "persona-sol-adjudication-v3-extended"
VERSION = 1
MAX_ROW_ATTEMPTS = 10
LOCAL_VALIDATION_COMPLETIONS = 3
MAX_LOCAL_VALIDATION_BATCHES = (
    MAX_ROW_ATTEMPTS + LOCAL_VALIDATION_COMPLETIONS - 1
) // LOCAL_VALIDATION_COMPLETIONS
LOCAL_VALIDATION_FAILURE = "Sol response failed bounded local validation retries"
ROW_ATTEMPT_LIFETIME_EXHAUSTED = "Per-row proxy attempt lifetime exhausted"


class ReviewError(Exception):
    """A safe-to-report v3 campaign failure."""


@click.command()
@click.option("--original", type=click.Path(path_type=Path), default=ORIGINAL)
@click.option("--candidate", type=click.Path(path_type=Path), default=CANDIDATE)
@click.option("--repair-manifest", type=click.Path(path_type=Path), default=REPAIR)
@click.option("--review-checkpoints", type=click.Path(path_type=Path), default=REVIEW)
@click.option("--h90-status", type=click.Path(path_type=Path), default=H90)
@click.option("--followup-status", type=click.Path(path_type=Path), default=FOLLOWUP)
@click.option("--prompt", type=click.Path(path_type=Path), default=PROMPT)
@click.option("--output-dir", type=click.Path(path_type=Path), default=OUTPUT)
@click.option("--registry", type=click.Path(path_type=Path), default=REGISTRY)
@click.option("--workers", type=click.IntRange(min=1, max=20), default=4)
@click.option("--run", "execute", is_flag=True, default=False)
def main(
    original: Path,
    candidate: Path,
    repair_manifest: Path,
    review_checkpoints: Path,
    h90_status: Path,
    followup_status: Path,
    prompt: Path,
    output_dir: Path,
    registry: Path,
    workers: int,
    execute: bool,
) -> None:
    """Dry-run by default; ``--run`` enables calls to the configured local proxy.

    Raises:
        click.ClickException: If an input or private campaign boundary is invalid.
    """
    configure_cli_logging()
    load_repository_environment()
    try:
        summary = run_campaign(
            original=original,
            candidate=candidate,
            repair_path=repair_manifest,
            review_path=review_checkpoints,
            h90_path=h90_status,
            followup_path=followup_status,
            prompt_path=prompt,
            output_dir=output_dir,
            registry=registry,
            workers=workers,
            execute=execute,
        )
    except (
        ReviewError,
        OSError,
        ValueError,
        ProxyBudgetError,
        SolAdjudicationError,
    ) as exc:
        raise click.ClickException(_safe_error(exc)) from exc
    click.echo(json.dumps(summary, sort_keys=True))


def run_campaign(  # noqa: C901, PLR0912, PLR0915
    *,
    original: Path,
    candidate: Path,
    repair_path: Path,
    review_path: Path,
    h90_path: Path,
    followup_path: Path,
    prompt_path: Path,
    output_dir: Path,
    registry: Path,
    workers: int,
    execute: bool,
) -> dict[str, object]:
    """Validate pinned inputs, build the sorted queue and optionally run.

    Returns:
        Aggregate campaign progress without row identifiers or persona text.

    Raises:
        ReviewError: If pinned inputs or durable resume state are invalid.
        SolAdjudicationError: If local request validation or adjudication fails.
    """
    if workers not in range(1, 21):
        raise ReviewError("Worker count is outside the supported range")
    base = pl.read_parquet(original)
    frame = pl.read_parquet(candidate)
    if base.height != EXPECTED_ROWS or frame.height != EXPECTED_ROWS:
        raise ReviewError("Input row count mismatch")
    if ID_FIELD not in base.columns or ID_FIELD not in frame.columns:
        raise ReviewError("Input identifier column missing")
    ids = base[ID_FIELD].to_list()
    candidate_ids = frame[ID_FIELD].to_list()
    if ids != candidate_ids or any(not isinstance(value, str) for value in ids):
        raise ReviewError("Ordered input identifiers do not match")
    id_hashes = [sha256_text(value) for value in ids]
    ordered_hash = sha256_text(canonical_json(id_hashes))
    base_rows, candidate_rows = base.to_dicts(), frame.to_dicts()
    _check_rows_private(original_rows=base_rows, candidate_rows=candidate_rows)
    repair = _json_object(repair_path)
    if (
        repair.get("private") is not True
        or repair.get("published_sha256") != sha256_file(original)
        or repair.get("candidate_sha256") != sha256_file(candidate)
    ):
        raise ReviewError(
            "Repair manifest is not bound to the immutable published input"
        )
    changed_indexes = repair.get("changed_row_indexes")
    if not isinstance(changed_indexes, list) or any(
        not isinstance(i, int) or i < 0 or i >= EXPECTED_ROWS for i in changed_indexes
    ):
        raise ReviewError("Repair row index list is invalid")
    actual_changed = [
        index
        for index, (before, after) in enumerate(
            zip(base_rows, candidate_rows, strict=True)
        )
        if before != after
    ]
    if sorted(changed_indexes) != actual_changed:
        raise ReviewError("Repair row indexes do not match the candidate changes")
    reasons: dict[int, set[str]] = {}
    for index in changed_indexes:
        reasons.setdefault(index, set()).add("v3_demographic_repair")
    checkpoint_by_hash: dict[str, dict[str, Any]] = {}
    for path in sorted(review_path.glob("*/*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            digest = path.stem
            if len(digest) == 64 and doc.get("disposition") == "needs_manual_review":
                checkpoint_by_hash[digest] = doc
        except OSError, json.JSONDecodeError:
            raise ReviewError("Prior review checkpoint is unreadable") from None
    positions = {digest: i for i, digest in enumerate(id_hashes)}
    for digest in checkpoint_by_hash:
        if digest in positions:
            reasons.setdefault(positions[digest], set()).add("prior_manual_review")
    generated = check_generated_fields(frame)
    unsafe_ids = {
        finding.persona_id
        for finding in generated.review_flags
        if finding.check == "unsafe_or_forbidden_pattern"
    }
    for persona_id in unsafe_ids:
        digest = sha256_text(persona_id)
        if digest in positions:
            reasons.setdefault(positions[digest], set()).add(
                "unsafe_or_forbidden_pattern"
            )
    unmatched_status_records = {
        "h90": _unmatched_status_count(
            h90_path, positions, {"unresolved", "validation_failed"}
        ),
        "followup": _followup_unmatched_count(followup_path, positions),
    }
    _add_status_reasons(
        h90_path,
        positions,
        reasons,
        "h90_unresolved",
        {"unresolved", "validation_failed"},
    )
    _add_followup_reasons(followup_path, positions, reasons)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    _check_restricted_text(value=prompt_text, label="prompt")
    selected = sorted(reasons)
    reason_counts: dict[str, int] = {}
    for labels in reasons.values():
        for label in labels:
            reason_counts[label] = reason_counts.get(label, 0) + 1

    schema_hash = sha256_text(
        canonical_json(SolAdjudicationResponse.provider_json_schema())
    )
    config_hash = sha256_text(
        canonical_json(
            {
                "base_url": BASE_URL,
                "model": SOL_ADJUDICATION_MODEL,
                "api_key_env": None,
                "timeout_seconds": 360.0,
                "maximum_http_attempts": 5,
                "maximum_total_requests": None,
                "retry_backoff_seconds": 1.0,
                "maximum_rows_per_shard": 1,
                "max_tokens": None,
                "enable_thinking": None,
                "reasoning_effort": "none",
                "origin_label_contract_sha256": sha256_file(
                    Path("config/folk2-ieland-labels-da.yaml")
                ),
            }
        )
    )
    input_hashes = {
        "baseline_sha256": sha256_file(original),
        "candidate_sha256": sha256_file(candidate),
        "repair_sha256": sha256_file(repair_path),
        "h90_status_sha256": sha256_file(h90_path),
        "followup_status_sha256": sha256_file(followup_path),
        "ordered_id_hashes_sha256": ordered_hash,
        "prompt_sha256": sha256_file(prompt_path),
        "schema_sha256": schema_hash,
        "config_sha256": config_hash,
        "queue_sha256": sha256_text(canonical_json(selected)),
    }
    manifest = {
        "campaign": CAMPAIGN,
        "version": VERSION,
        "input_hashes": input_hashes,
        "selected_total": len(selected),
        "reason_counts": reason_counts,
    }
    if not execute:
        return {
            "campaign": CAMPAIGN,
            "dry_run": True,
            "selected_total": len(selected),
            "reason_counts": reason_counts,
            "unmatched_prior_status_records": unmatched_status_records,
            "ordered_id_hashes_sha256": ordered_hash,
        }
    if any(unmatched_status_records.values()):
        raise ReviewError(
            "Prior unresolved status contains rows not bound to this release"
        )
    validation = validate_release_candidate(
        candidate_path=candidate,
        original_path=original,
        bundle_dir=Path("data/processed/6e27b5c08fbeae79"),
    )
    if not validation.passes_hard_gates:
        raise ReviewError("Candidate failed mandatory offline release validation")
    manifest["candidate_validation_sha256"] = validation.candidate_sha256
    if manifest["candidate_validation_sha256"] != input_hashes["candidate_sha256"]:
        raise ReviewError("Validated candidate checksum changed")

    require_runtime_model(ADJUDICATION_MODEL_ENV)
    _prepare_private_output(output_dir=output_dir)
    manifest_path = output_dir / "manifest.json"
    _write_or_verify(manifest_path, manifest)
    prompt_hash = input_hashes["prompt_sha256"]
    source_hash = sha256_text(canonical_json(input_hashes))
    budget = ProxyBudget(
        ledger_path=output_dir / "ignored-sol-budget.jsonl",
        registry_path=registry,
        campaign=CAMPAIGN,
        source_hash=source_hash,
        prompt_hash=prompt_hash,
        schema_hash=schema_hash,
        model=SOL_ADJUDICATION_MODEL,
        input_usd_per_million="0.1",
        output_usd_per_million="0.5",
        max_tokens=SOL_ADJUDICATION_LEDGER_MAX_TOKENS,
        cap_usd=Decimal("1"),
        uncapped=True,
        uncapped_purpose=V3_EXTENDED_ADJUDICATION_PURPOSE,
    )
    config = GenerationConfig.model_validate(
        {
            "base_url": BASE_URL,
            "model": SOL_ADJUDICATION_MODEL,
            "api_key_env": None,
            "timeout_seconds": 360.0,
            "maximum_http_attempts": 5,
            "maximum_total_requests": None,
            "retry_backoff_seconds": 1.0,
            "maximum_rows_per_shard": 1,
            "max_tokens": None,
            "enable_thinking": None,
            "reasoning_effort": "none",
            "prompt": prompt_path,
            "origin_label_contract": Path("config/folk2-ieland-labels-da.yaml"),
        }
    )
    status_path = output_dir / "status.json"
    status = _load_status(status_path, input_hashes, len(selected))
    selected_hashes = {id_hashes[index] for index in selected}
    done: set[str] = set()
    for item in status["processed"]:
        digest = item.get("persona_hash")
        disposition = item.get("disposition")
        if (
            not isinstance(digest, str)
            or digest not in selected_hashes
            or digest in done
            or disposition
            not in {
                "consistent",
                "patched",
                "unresolved",
                "privacy_blocked",
                "validation_failed",
            }
            or item.get("result_sha256") != sha256_text(f"{digest}:{disposition}")
        ):
            raise ReviewError("Existing status contains an invalid row disposition")
        if disposition == "validation_failed":
            _verify_validation_failure(
                output_dir=output_dir, digest=digest, input_hashes=input_hashes
            )
        elif disposition != "privacy_blocked":
            checkpoint = item.get("checkpoint")
            checkpoint_hash = item.get("checkpoint_sha256")
            response_hash = item.get("response_sha256")
            if (
                not isinstance(checkpoint, str)
                or not isinstance(checkpoint_hash, str)
                or not isinstance(response_hash, str)
            ):
                raise ReviewError("Processed row lacks its checkpoint binding")
            checkpoint_path = output_dir / checkpoint
            if sha256_file(checkpoint_path) != checkpoint_hash:
                raise ReviewError("Processed checkpoint checksum mismatch")
            document = _json_object(checkpoint_path)
            if document.get("response_sha256") != response_hash:
                raise ReviewError("Processed response binding mismatch")
            index = positions[digest]
            row, old = candidate_rows[index], base_rows[index]
            hints = {
                field: {"old": old[field], "new": row[field]}
                for field in sorted((set(old) & set(row)) & SOL_ALLOWED_FACT_FIELDS)
                if old[field] != row[field]
            }
            # Existing checkpoint only: the runner revalidates schema, response,
            # prompt, source and row bindings and cannot fall through to HTTP.
            verified = run_sol_adjudication(
                original_persona=row[TEXT_FIELD],
                candidate_row=row,
                prompt=prompt_text,
                config=config,
                budget=budget,
                checkpoint_path=checkpoint_path,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(500, request=request)
                ),
                changed_fact_hints=hints,
                original_row=old,
            )
            if verified.disposition != disposition:
                raise ReviewError("Processed checkpoint disposition mismatch")
        done.add(digest)
    pending = [index for index in selected if id_hashes[index] not in done]
    recovered: list[dict[str, Any]] = []
    for index in pending:
        digest = id_hashes[index]
        evidence_path = _validation_evidence_path(output_dir=output_dir, digest=digest)
        if evidence_path.exists():
            _verify_validation_failure(
                output_dir=output_dir, digest=digest, input_hashes=input_hashes
            )
            recovered.append(
                {
                    "persona_hash": digest,
                    "disposition": "validation_failed",
                    "result_sha256": sha256_text(f"{digest}:validation_failed"),
                }
            )
    if recovered:
        status["processed"].extend(recovered)
        status["processed"].sort(key=lambda item: item["persona_hash"])
        status["counts"] = _count_dispositions(status["processed"])
        status["progress"] = {
            "completed": len(status["processed"]),
            "total": len(selected),
            "pending": len(selected) - len(status["processed"]),
        }
        _write_private_json(path=status_path, value=status)
        done.update(item["persona_hash"] for item in recovered)
    pending = [index for index in pending if id_hashes[index] not in done]
    # Complete privacy preflight for every queued row before opening a client or
    # allowing any worker to issue a request.
    blocked: set[int] = set()
    for index in pending:
        row, old = candidate_rows[index], base_rows[index]
        hints = {
            field: {"old": old[field], "new": row[field]}
            for field in sorted((set(old) & set(row)) & SOL_ALLOWED_FACT_FIELDS)
            if old[field] != row[field]
        }
        try:
            preflight_sol_adjudication_payload(
                original_persona=row[TEXT_FIELD],
                candidate_row=row,
                prompt=prompt_text,
                changed_fact_hints=hints,
                original_row=old,
            )
        except SolAdjudicationError as exc:
            if "contains a restricted identity term" not in str(exc):
                raise
            blocked.add(index)
    for index in sorted(blocked):
        digest = id_hashes[index]
        status["processed"].append(
            {
                "persona_hash": digest,
                "disposition": "privacy_blocked",
                "result_sha256": sha256_text(f"{digest}:privacy_blocked"),
            }
        )
    status["processed"].sort(key=lambda item: item["persona_hash"])
    status["counts"] = _count_dispositions(status["processed"])
    status["progress"] = {
        "completed": len(status["processed"]),
        "total": len(selected),
        "pending": len(selected) - len(status["processed"]),
    }
    _write_private_json(path=status_path, value=status)
    pending = [index for index in pending if index not in blocked]
    with httpx.Client(base_url=BASE_URL, timeout=360) as client:
        transport = client._transport
        row_iter = iter(pending)
        future_map: dict[futures.Future[dict[str, Any]], int] = {}
        pool = futures.ThreadPoolExecutor(max_workers=workers)
        try:
            _submit_rows(
                row_iter=row_iter,
                future_map=future_map,
                pool=pool,
                workers=workers,
                candidate_rows=candidate_rows,
                base_rows=base_rows,
                reasons=reasons,
                id_hashes=id_hashes,
                prompt=prompt_text,
                prompt_path=prompt_path,
                config=config,
                budget=budget,
                output_dir=output_dir,
                transport=transport,
                input_hashes=input_hashes,
            )
            while future_map:
                completed, _ = futures.wait(
                    future_map, return_when=futures.FIRST_COMPLETED
                )
                future = next(iter(completed))
                future_map.pop(future)
                result = future.result()
                status["processed"].append(result)
                status["processed"].sort(key=lambda item: item["persona_hash"])
                status["counts"] = _count_dispositions(status["processed"])
                status["progress"] = {
                    "completed": len(status["processed"]),
                    "total": len(selected),
                    "pending": len(selected) - len(status["processed"]),
                }
                _write_private_json(path=status_path, value=status)
                _submit_rows(
                    row_iter=row_iter,
                    future_map=future_map,
                    pool=pool,
                    workers=workers,
                    candidate_rows=candidate_rows,
                    base_rows=base_rows,
                    reasons=reasons,
                    id_hashes=id_hashes,
                    prompt=prompt_text,
                    prompt_path=prompt_path,
                    config=config,
                    budget=budget,
                    output_dir=output_dir,
                    transport=transport,
                    input_hashes=input_hashes,
                )
        except BaseException:
            for future in future_map:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
    return {
        "campaign": CAMPAIGN,
        "dry_run": False,
        "selected_total": len(selected),
        "counts": status["counts"],
        "progress": status["progress"],
    }


def _submit_rows(  # noqa: PLR0913
    *,
    row_iter: Iterator[int],
    future_map: dict[futures.Future[dict[str, Any]], int],
    pool: futures.ThreadPoolExecutor,
    workers: int,
    candidate_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    reasons: dict[int, set[str]],
    id_hashes: list[str],
    prompt: str,
    prompt_path: Path,
    config: GenerationConfig,
    budget: ProxyBudget,
    output_dir: Path,
    transport: httpx.BaseTransport,
    input_hashes: dict[str, str],
) -> None:
    while len(future_map) < workers:
        try:
            index = next(row_iter)
        except StopIteration:
            return
        future_map[
            pool.submit(
                _process_row,
                index,
                candidate_rows,
                base_rows,
                reasons,
                id_hashes,
                prompt,
                prompt_path,
                config,
                budget,
                output_dir,
                transport,
                input_hashes,
            )
        ] = index


def _process_row(
    index: int,
    candidate_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    reasons: dict[int, set[str]],
    id_hashes: list[str],
    prompt: str,
    prompt_path: Path,
    config: GenerationConfig,
    budget: ProxyBudget,
    output_dir: Path,
    transport: httpx.BaseTransport,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    row = candidate_rows[index]
    old = base_rows[index]
    hints = {
        field: {"old": old[field], "new": row[field]}
        for field in sorted((set(old) & set(row)) & SOL_ALLOWED_FACT_FIELDS)
        if old[field] != row[field]
    }
    # Existing prose is the baseline for these adjudications; facts are the v3 row.
    digest = id_hashes[index]
    checkpoint_path = output_dir / "checkpoints" / digest[:2] / f"{digest}.json"
    for batch in range(MAX_LOCAL_VALIDATION_BATCHES):
        try:
            result = run_sol_adjudication(
                original_persona=row[TEXT_FIELD],
                candidate_row=row,
                prompt=prompt,
                config=config,
                budget=budget,
                checkpoint_path=checkpoint_path,
                transport=transport,
                changed_fact_hints=hints,
                original_row=old,
            )
        except SolAdjudicationError as exc:
            if str(exc) != LOCAL_VALIDATION_FAILURE:
                raise
            if batch + 1 == MAX_LOCAL_VALIDATION_BATCHES:
                return _record_validation_failure(
                    output_dir=output_dir,
                    digest=digest,
                    checkpoint_path=checkpoint_path,
                    input_hashes=input_hashes,
                    failure=LOCAL_VALIDATION_FAILURE,
                    batches=batch + 1,
                )
            continue
        except ProxyBudgetError as exc:
            if str(exc) != ROW_ATTEMPT_LIFETIME_EXHAUSTED:
                raise
            return _record_validation_failure(
                output_dir=output_dir,
                digest=digest,
                checkpoint_path=checkpoint_path,
                input_hashes=input_hashes,
                failure=ROW_ATTEMPT_LIFETIME_EXHAUSTED,
                batches=batch + 1,
            )
        break
    document = _json_object(checkpoint_path)
    terminal = result.disposition
    return {
        "persona_hash": id_hashes[index],
        "disposition": terminal,
        "result_sha256": sha256_text(f"{id_hashes[index]}:{terminal}"),
        "checkpoint": str(checkpoint_path.relative_to(output_dir)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "response_sha256": document.get("response_sha256"),
    }


def _unmatched_status_count(
    path: Path, positions: dict[str, int], dispositions: set[str]
) -> int:
    """Count prior terminal rows that cannot be bound to this ordered release.

    Returns:
        Number of unmatched terminal rows, without exposing their identifiers.

    Raises:
        ReviewError: If the prior status document has an invalid shape.
    """
    processed = _json_object(path).get("processed")
    if not isinstance(processed, list):
        raise ReviewError("Prior status shape is invalid")
    return sum(
        1
        for item in processed
        if isinstance(item, dict)
        and item.get("disposition") in dispositions
        and item.get("persona_hash") not in positions
    )


def _followup_bindings(path: Path, positions: dict[str, int]) -> dict[str, int]:
    """Validate the follow-up selection manifest and return checkpoint bindings.

    Returns:
        Mapping from follow-up checkpoint identifiers to release row indexes.

    Raises:
        ReviewError: If manifest inputs cannot be bound to the release.
    """
    manifest = _json_object(path.parent / "manifest.json")
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise ReviewError("Follow-up selection manifest is invalid")
    ordered = selection.get("ordered_input_hashes")
    bindings = selection.get("row_bindings")
    if (
        not isinstance(ordered, list)
        or not isinstance(bindings, list)
        or ordered
        != [digest for digest, _ in sorted(positions.items(), key=lambda item: item[1])]
    ):
        raise ReviewError("Follow-up ordered input hashes do not match the release")
    result: dict[str, int] = {}
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ReviewError("Follow-up selection bindings are invalid")
        checkpoint = binding.get("followup_checkpoint_id")
        row_index = binding.get("row_index")
        row_hash = binding.get("row_hash")
        if (
            not isinstance(checkpoint, str)
            or not isinstance(row_index, int)
            or row_index < 0
            or row_index >= len(ordered)
            or checkpoint in result
            or row_hash != ordered[row_index]
        ):
            raise ReviewError("Follow-up selection bindings are invalid")
        # The manifest hashes the v2 persona identifier at the exact selected row.
        if (
            ordered[row_index] not in positions
            or positions[ordered[row_index]] != row_index
        ):
            raise ReviewError("Follow-up selection row does not bind to this release")
        result[checkpoint] = row_index
    expected_digest = sha256_text(canonical_json(ordered))
    expected = selection.get("ordered_id_hashes_sha256")
    if not isinstance(expected, str) or expected != expected_digest:
        raise ReviewError("Follow-up ordered-input digest mismatch")
    return result


def _followup_unmatched_count(path: Path, positions: dict[str, int]) -> int:
    bindings = _followup_bindings(path, positions)
    processed = _followup_processed(path)
    return sum(
        1
        for item in processed
        if isinstance(item, dict)
        and item.get("disposition") == "unresolved"
        and item.get("persona_hash") not in bindings
    )


def _followup_processed(path: Path) -> list[dict[str, Any]]:
    """Load status only after binding it to the exact selection manifest.

    Returns:
        Validated processed status rows.

    Raises:
        ReviewError: If the manifest binding or status integrity is invalid.
    """
    status = _json_object(path)
    manifest = _json_object(path.parent / "manifest.json")
    if status.get("manifest_sha256") != sha256_text(canonical_json(manifest)):
        raise ReviewError("Follow-up status is not bound to its selection manifest")
    processed = status.get("processed")
    if not isinstance(processed, list) or any(
        not isinstance(item, dict) for item in processed
    ):
        raise ReviewError("Prior status shape is invalid")
    actual_counts = {
        key: value for key, value in _count_dispositions(processed).items() if value > 0
    }
    if status.get("counts") != actual_counts:
        raise ReviewError("Follow-up status counts do not match processed rows")
    selection = manifest.get("selection")
    bindings = selection.get("row_bindings") if isinstance(selection, dict) else None
    if not isinstance(bindings, list):
        raise ReviewError("Follow-up manifest bindings are invalid")
    known = {
        binding.get("followup_checkpoint_id")
        for binding in bindings
        if isinstance(binding, dict)
    }
    seen: set[str] = set()
    for item in processed:
        digest = item.get("persona_hash")
        if (
            not isinstance(digest, str)
            or digest not in known
            or digest in seen
            or item.get("disposition")
            not in {
                "consistent",
                "patched",
                "unresolved",
                "privacy_blocked",
                "validation_failed",
            }
        ):
            raise ReviewError("Follow-up processed rows do not match manifest order")
        seen.add(digest)
    return processed


def _add_followup_reasons(
    path: Path, positions: dict[str, int], reasons: dict[int, set[str]]
) -> None:
    bindings = _followup_bindings(path, positions)
    processed = _followup_processed(path)
    for item in processed:
        if isinstance(item, dict) and item.get("disposition") == "unresolved":
            checkpoint = item.get("persona_hash")
            if checkpoint in bindings:
                row_index = bindings[checkpoint]
                reasons.setdefault(row_index, set()).add("followup_unresolved")


def _add_status_reasons(
    path: Path,
    positions: dict[str, int],
    reasons: dict[int, set[str]],
    reason: str,
    dispositions: set[str],
) -> None:
    doc = _json_object(path)
    processed = doc.get("processed")
    if not isinstance(processed, list):
        raise ReviewError("Prior status shape is invalid")
    for item in processed:
        if isinstance(item, dict) and item.get("disposition") in dispositions:
            digest = item.get("persona_hash")
            if isinstance(digest, str) and digest in positions:
                reasons.setdefault(positions[digest], set()).add(reason)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        raise ReviewError("Private input JSON is unreadable") from None
    if not isinstance(value, dict):
        raise ReviewError("Private input JSON has invalid shape")
    return value


def _write_or_verify(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if _json_object(path) != value:
            raise ReviewError("Existing campaign manifest does not match inputs")
    else:
        _write_private_json(path=path, value=value)


def _validation_evidence_path(*, output_dir: Path, digest: str) -> Path:
    return output_dir / "validation-failed" / digest[:2] / f"{digest}.json"


def _record_validation_failure(
    *,
    output_dir: Path,
    digest: str,
    checkpoint_path: Path,
    input_hashes: dict[str, str],
    failure: str,
    batches: int,
) -> dict[str, Any]:
    if checkpoint_path.exists():
        raise ReviewError("Validation-failed row unexpectedly has a checkpoint")
    evidence = {
        "campaign": CAMPAIGN,
        "persona_hash": digest,
        "input_hashes": input_hashes,
        "failure": failure,
        "batches": batches,
    }
    path = _validation_evidence_path(output_dir=output_dir, digest=digest)
    _write_private_json(path=path, value=cast(dict[str, JSONDocument], evidence))
    return {
        "persona_hash": digest,
        "disposition": "validation_failed",
        "result_sha256": sha256_text(f"{digest}:validation_failed"),
    }


def _verify_validation_failure(
    *, output_dir: Path, digest: str, input_hashes: dict[str, str]
) -> None:
    checkpoint = output_dir / "checkpoints" / digest[:2] / f"{digest}.json"
    if checkpoint.exists():
        raise ReviewError("Validation-failed row unexpectedly has a checkpoint")
    evidence = _json_object(
        _validation_evidence_path(output_dir=output_dir, digest=digest)
    )
    if (
        set(evidence)
        != {"campaign", "persona_hash", "input_hashes", "failure", "batches"}
        or evidence.get("campaign") != CAMPAIGN
        or evidence.get("persona_hash") != digest
        or evidence.get("input_hashes") != input_hashes
        or evidence.get("failure")
        not in {LOCAL_VALIDATION_FAILURE, ROW_ATTEMPT_LIFETIME_EXHAUSTED}
        or not isinstance(evidence.get("batches"), int)
        or not 1 <= evidence["batches"] <= MAX_LOCAL_VALIDATION_BATCHES
    ):
        raise ReviewError("Validation-failed evidence is not bound to this campaign")


def _load_status(path: Path, hashes: dict[str, str], total: int) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": VERSION,
            "campaign": CAMPAIGN,
            "input_hashes": hashes,
            "selected_total": total,
            "processed": [],
            "counts": {},
            "progress": {"completed": 0, "total": total, "pending": total},
        }
    status = _json_object(path)
    if status.get("input_hashes") != hashes or status.get("selected_total") != total:
        raise ReviewError("Existing status is bound to different inputs")
    processed = status.get("processed")
    if not isinstance(processed, list):
        raise ReviewError("Existing status shape is invalid")
    counts = _count_dispositions(processed)
    if status.get("counts") != counts:
        raise ReviewError("Existing status disposition counts are invalid")
    progress = status.get("progress")
    expected_progress = {
        "completed": len(processed),
        "total": total,
        "pending": total - len(processed),
    }
    if progress != expected_progress or len(processed) > total:
        raise ReviewError("Existing status progress is invalid")
    return status


def _count_dispositions(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "consistent": 0,
        "patched": 0,
        "unresolved": 0,
        "privacy_blocked": 0,
        "validation_failed": 0,
    }
    for item in items:
        disposition = item.get("disposition")
        if isinstance(disposition, str):
            counts[disposition] = counts.get(disposition, 0) + 1
    return counts


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, (ProxyBudgetError, SolAdjudicationError)):
        return (
            "Private adjudication stopped safely; inspect local configuration "
            "and private logs."
        )
    if isinstance(exc, ReviewError):
        return str(exc)
    return "Private adjudication stopped safely due to an input or filesystem error."


if __name__ == "__main__":
    load_repository_environment()
    main()
