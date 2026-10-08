"""Run a separate resumable follow-up for non-terminal v3 campaign rows."""

from __future__ import annotations

import concurrent.futures
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

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
    V3_TARGETED_FOLLOWUP_PURPOSE,
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
from scripts import adjudicate_persona_v3 as base_campaign
from scripts.adjudicate_persona_release import (
    _check_restricted_text,
    _write_private_json,
)

DEFAULT_BASE = Path("/tmp/danish-personas-audit/v3-private/campaign-long")
DEFAULT_CANDIDATE = base_campaign.CANDIDATE
DEFAULT_ORIGINAL = Path(
    "/tmp/danish-personas-audit/hf-v2-foreign-hotfix-remote/data/train-00000-of-00001.parquet"
)
DEFAULT_OUTPUT = Path("/tmp/danish-personas-audit/v3-private/campaign-followup")
FOLLOWUP_CAMPAIGN = "persona-sol-adjudication-v3-targeted-followup"


@click.command()
@click.option("--base-campaign", type=click.Path(path_type=Path), default=DEFAULT_BASE)
@click.option("--candidate", type=click.Path(path_type=Path), default=DEFAULT_CANDIDATE)
@click.option("--original", type=click.Path(path_type=Path), default=DEFAULT_ORIGINAL)
@click.option("--output-dir", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT)
@click.option(
    "--prompt",
    "prompt_path",
    type=click.Path(path_type=Path),
    default=base_campaign.PROMPT,
)
@click.option(
    "--registry", type=click.Path(path_type=Path), default=base_campaign.REGISTRY
)
@click.option("--workers", type=click.IntRange(1, 20), default=20)
@click.option("--run", "execute", is_flag=True)
def main(
    base_campaign: Path,
    candidate: Path,
    original: Path,
    output_dir: Path,
    prompt_path: Path,
    registry: Path,
    workers: int,
    execute: bool,
) -> None:
    """Dry-run by default; live calls require ``--run`` and runtime model env.

    Raises:
        click.ClickException: If campaign inputs or resume state are invalid.
    """
    configure_cli_logging()
    try:
        result = followup(
            base_dir=base_campaign,
            candidate=candidate,
            original=original,
            output_dir=output_dir,
            prompt_path=prompt_path,
            registry=registry,
            workers=workers,
            execute=execute,
        )
    except (OSError, ValueError, KeyError, RuntimeError, SolAdjudicationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, sort_keys=True))


def followup(  # noqa: C901, PLR0912
    *,
    base_dir: Path,
    candidate: Path,
    original: Path,
    output_dir: Path,
    prompt_path: Path,
    registry: Path,
    workers: int,
    execute: bool,
) -> dict[str, object]:
    """Select unresolved base dispositions and execute/resume their second pass.

    Returns:
        Aggregate progress without identifiers or persona text.

    Raises:
        RuntimeError: If campaign bindings or resumable state are invalid.
        SolAdjudicationError: If local payload or checkpoint verification fails.
        ProxyBudgetError: If budget integrity or authorisation fails.
    """
    base_manifest = _json(base_dir / "manifest.json")
    base_status = _json(base_dir / "status.json")
    if (
        base_manifest.get("campaign") != base_campaign.CAMPAIGN
        or base_status.get("campaign") != base_campaign.CAMPAIGN
        or base_status.get("input_hashes") != base_manifest.get("input_hashes")
        or base_status.get("progress", {}).get("pending") != 0
        or base_status.get("progress", {}).get("completed")
        != base_manifest.get("selected_total")
    ):
        raise RuntimeError("Base campaign must be complete and checksum-bound")
    frame, old_frame = pl.read_parquet(candidate), pl.read_parquet(original)
    if frame.height != base_campaign.EXPECTED_ROWS or old_frame.height != frame.height:
        raise RuntimeError("Input row count mismatch")
    ids = frame[base_campaign.ID_FIELD].to_list()
    if ids != old_frame[base_campaign.ID_FIELD].to_list():
        raise RuntimeError("Ordered identifiers do not match")
    id_hashes = [sha256_text(value) for value in ids]
    base_inputs = base_manifest.get("input_hashes")
    expected_base_inputs = {
        "baseline_sha256": sha256_file(original),
        "candidate_sha256": sha256_file(candidate),
        "ordered_id_hashes_sha256": sha256_text(canonical_json(id_hashes)),
    }
    if not isinstance(base_inputs, dict) or any(
        base_inputs.get(key) != value for key, value in expected_base_inputs.items()
    ):
        raise RuntimeError("Follow-up files do not match the completed campaign")
    positions = {digest: index for index, digest in enumerate(id_hashes)}
    processed = base_status.get("processed")
    if not isinstance(processed, list):
        raise RuntimeError("Base campaign records are invalid")
    selected_items = [
        item
        for item in processed
        if isinstance(item, dict)
        and item.get("disposition")
        in {"unresolved", "validation_failed", "privacy_blocked"}
    ]
    selected_hashes = [item.get("persona_hash") for item in selected_items]
    if any(digest not in positions for digest in selected_hashes) or len(
        set(selected_hashes)
    ) != len(selected_hashes):
        raise RuntimeError("Base selected identities are not bound to source rows")
    selection = sorted(positions[digest] for digest in selected_hashes)
    prompt = prompt_path.read_text(encoding="utf-8")
    _check_restricted_text(value=prompt, label="prompt")
    input_hashes = {
        "base_manifest_sha256": sha256_file(base_dir / "manifest.json"),
        "base_status_sha256": sha256_file(base_dir / "status.json"),
        "candidate_sha256": sha256_file(candidate),
        "original_sha256": sha256_file(original),
        "ordered_id_hashes_sha256": sha256_text(canonical_json(id_hashes)),
        "prompt_sha256": sha256_file(prompt_path),
        "queue_sha256": sha256_text(canonical_json(selection)),
    }
    manifest = {
        "campaign": FOLLOWUP_CAMPAIGN,
        "version": 1,
        "input_hashes": input_hashes,
        "selected_total": len(selection),
    }
    if not execute:
        return {
            "campaign": FOLLOWUP_CAMPAIGN,
            "dry_run": True,
            "selected_total": len(selection),
        }
    require_runtime_model(ADJUDICATION_MODEL_ENV)
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest_path = output_dir / "manifest.json"
    _write_bound(manifest_path, manifest)
    schema_hash = sha256_text(
        canonical_json(SolAdjudicationResponse.provider_json_schema())
    )
    source_hash = sha256_text(canonical_json(input_hashes))
    budget = ProxyBudget(
        registry_path=registry,
        campaign=FOLLOWUP_CAMPAIGN,
        source_hash=source_hash,
        prompt_hash=input_hashes["prompt_sha256"],
        schema_hash=schema_hash,
        model=SOL_ADJUDICATION_MODEL,
        input_usd_per_million="0.1",
        output_usd_per_million="0.5",
        max_tokens=SOL_ADJUDICATION_LEDGER_MAX_TOKENS,
        cap_usd=Decimal("1"),
        uncapped=True,
        uncapped_purpose=V3_TARGETED_FOLLOWUP_PURPOSE,
    )
    config = GenerationConfig.model_validate(
        {
            "base_url": BASE_URL,
            "model": SOL_ADJUDICATION_MODEL,
            "api_key_env": None,
            "timeout_seconds": 3600.0,
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
    status = _resume_status(status_path, manifest, len(selection))
    done = {item["persona_hash"] for item in status["processed"]}
    rows, old_rows = frame.to_dicts(), old_frame.to_dicts()
    # Privacy-blocked rows are never exposed; retain their local evidence disposition.
    for source_item in selected_items:
        digest = source_item["persona_hash"]
        if source_item["disposition"] == "privacy_blocked" and digest not in done:
            _append(
                status_path,
                status,
                {
                    "persona_hash": digest,
                    "disposition": "privacy_blocked_local_only",
                    "result_sha256": sha256_text(
                        f"{digest}:privacy_blocked_local_only"
                    ),
                    "base_result_sha256": source_item.get("result_sha256"),
                },
            )
            done.add(digest)
    pending = [
        i
        for i in selection
        if id_hashes[i] not in done
        and next(x for x in selected_items if x["persona_hash"] == id_hashes[i])[
            "disposition"
        ]
        != "privacy_blocked"
    ]
    with (
        httpx.Client(base_url=BASE_URL, timeout=3600.0) as client,
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        tasks = {}
        for index in pending:
            row, prior = rows[index], old_rows[index]
            hints = {
                field: {"old": prior[field], "new": row[field]}
                for field in sorted((set(prior) & set(row)) & SOL_ALLOWED_FACT_FIELDS)
                if prior[field] != row[field]
            }
            try:
                preflight_sol_adjudication_payload(
                    original_persona=row[base_campaign.TEXT_FIELD],
                    candidate_row=row,
                    prompt=prompt,
                    changed_fact_hints=hints,
                    original_row=prior,
                )
            except SolAdjudicationError as exc:
                digest = id_hashes[index]
                _append(
                    status_path,
                    status,
                    {
                        "persona_hash": digest,
                        "disposition": "preflight_failed",
                        "result_sha256": sha256_text(f"{digest}:preflight_failed"),
                        "base_result_sha256": next(
                            x["result_sha256"]
                            for x in selected_items
                            if x["persona_hash"] == digest
                        ),
                        "evidence_sha256": sha256_text(str(exc)),
                    },
                )
                done.add(digest)
                continue
            digest = id_hashes[index]
            cp = output_dir / "checkpoints" / digest[:2] / f"{digest}.json"
            tasks[
                pool.submit(
                    run_sol_adjudication,
                    original_persona=row[base_campaign.TEXT_FIELD],
                    candidate_row=row,
                    prompt=prompt,
                    config=config,
                    budget=budget,
                    checkpoint_path=cp,
                    transport=client._transport,
                    changed_fact_hints=hints,
                    original_row=prior,
                    adjudication_mode="unresolved_followup",
                )
            ] = (index, cp)
        for future in concurrent.futures.as_completed(tasks):
            index, cp = tasks[future]
            digest = id_hashes[index]
            try:
                result = future.result()
                doc = _json(cp)
            except httpx.TransportError:
                # Keep transport failures pending so a later run can retry them
                # against the same verified campaign and durable attempt ledger.
                continue
            except SolAdjudicationError as exc:
                if str(exc) != "Sol response failed bounded local validation retries":
                    raise
                # No valid provider result exists. Leave this row pending rather
                # than manufacturing a successful editorial verdict.
                continue
            except ProxyBudgetError as exc:
                if str(exc) != "Per-row proxy attempt lifetime exhausted":
                    raise
                # Exhaustion remains a release blocker, not a verified review.
                continue
            _append(
                status_path,
                status,
                {
                    "persona_hash": digest,
                    "disposition": result.disposition,
                    "result_sha256": sha256_text(f"{digest}:{result.disposition}"),
                    "checkpoint": str(cp.relative_to(output_dir)),
                    "checkpoint_sha256": sha256_file(cp),
                    "response_sha256": doc.get("response_sha256"),
                    "base_result_sha256": next(
                        x["result_sha256"]
                        for x in selected_items
                        if x["persona_hash"] == digest
                    ),
                },
            )
    if status["progress"]["pending"] != 0 or status["progress"]["completed"] != len(
        selection
    ):
        raise RuntimeError("Follow-up has pending rows; refusing successful completion")
    expected_counts: dict[str, int] = {}
    for item in status["processed"]:
        expected_counts[item["disposition"]] = (
            expected_counts.get(item["disposition"], 0) + 1
        )
    if status.get("counts") != expected_counts:
        raise RuntimeError("Follow-up disposition totals mismatch")
    return {
        "campaign": FOLLOWUP_CAMPAIGN,
        "dry_run": False,
        "selected_total": len(selection),
        "progress": status["progress"],
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Campaign document is not an object")
    return value


def _write_bound(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if _json(path) != value:
            raise RuntimeError("Existing follow-up manifest has different bindings")
        return
    _write_private_json(path=path, value=value)


def _resume_status(  # noqa: C901, PLR0912
    path: Path, manifest: dict[str, Any], total: int
) -> dict[str, Any]:
    if path.exists():
        status = _json(path)
        if status.get("input_hashes") != manifest["input_hashes"]:
            raise RuntimeError("Follow-up status input binding mismatch")
        processed = status.get("processed")
        if not isinstance(processed, list):
            raise RuntimeError("Follow-up status is invalid")
        if (
            status.get("campaign") != FOLLOWUP_CAMPAIGN
            or status.get("version") != 1
            or status.get("selected_total") != total
            or status.get("progress")
            != {
                "completed": len(processed),
                "total": total,
                "pending": total - len(processed),
            }
            or len({row.get("persona_hash") for row in processed}) != len(processed)
        ):
            raise RuntimeError(
                "Follow-up status progress or identity binding is invalid"
            )
        counts: dict[str, int] = {}
        for item in processed:
            if item.get("result_sha256") != sha256_text(
                f"{item.get('persona_hash')}:{item.get('disposition')}"
            ):
                raise RuntimeError("Follow-up result digest mismatch")
            if item.get("disposition") in {"patched", "consistent", "unresolved"}:
                checkpoint = item.get("checkpoint")
                cp = (
                    path.parent / checkpoint
                    if isinstance(checkpoint, str)
                    else Path("/")
                )
                if (
                    not cp.is_relative_to(path.parent)
                    or not cp.is_file()
                    or sha256_file(cp) != item.get("checkpoint_sha256")
                ):
                    raise RuntimeError("Follow-up checkpoint binding mismatch")
                doc = _json(cp)
                if doc.get("response_sha256") != item.get("response_sha256"):
                    raise RuntimeError("Follow-up response binding mismatch")
            counts[item["disposition"]] = counts.get(item["disposition"], 0) + 1
        if status.get("counts") != counts:
            raise RuntimeError("Follow-up disposition totals mismatch")
        return status
    return {
        "campaign": FOLLOWUP_CAMPAIGN,
        "version": 1,
        "input_hashes": manifest["input_hashes"],
        "selected_total": total,
        "processed": [],
        "counts": {},
        "progress": {"completed": 0, "total": total, "pending": total},
    }


def _append(path: Path, status: dict[str, Any], item: dict[str, Any]) -> None:
    if any(
        row.get("persona_hash") == item["persona_hash"] for row in status["processed"]
    ):
        return
    status["processed"].append(item)
    status["processed"].sort(key=lambda row: row["persona_hash"])
    counts: dict[str, int] = {}
    for row in status["processed"]:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    status["counts"] = counts
    status["progress"] = {
        "completed": len(status["processed"]),
        "total": status["selected_total"],
        "pending": status["selected_total"] - len(status["processed"]),
    }
    _write_private_json(path=path, value=status)


if __name__ == "__main__":
    load_repository_environment()
    main()
