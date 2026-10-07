"""Run private second-pass verification for v4 persona prose patches."""

from __future__ import annotations

import collections.abc as c
import concurrent.futures as futures
import datetime as dt
import email.utils
import json
import logging
import os
import time
import typing as t
from dataclasses import dataclass
from pathlib import Path

import click
import httpx

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.environment import load_repository_environment
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.proxy_budget import (
    BASE_URL,
    MODEL,
    PATCH_VERIFICATION_PURPOSE,
    ProxyBudget,
    ProxyBudgetError,
)
from danish_personas.generation.proxy_patch_runner import _ALLOWED_FACTS
from danish_personas.generation.proxy_patch_verifier import (
    ProxyPatchVerificationError,
    run_proxy_patch_verification,
)
from danish_personas.io import canonical_json, sha256_file, sha256_text
from danish_personas.release.prose_patch_verification import (
    ProsePatchSecondReview,
    ProsePatchVerificationResult,
)
from scripts.build_prose_review_v4_dashboard import (
    DEFAULT_CANDIDATE,
    DEFAULT_ORIGINAL,
    DEFAULT_REGISTRY,
    DEFAULT_TRIAGE,
    DashboardPaths,
    ProseReviewV4DashboardError,
    ReviewDecision,
    _changed_facts_mapping,
    _load_complete_checkpoint,
    _rows_by_hash,
    _verify_checkpoint_decision,
    _verify_decision_counts,
    _verify_manifest_sources,
    _verify_status_counts,
    _verify_status_manifest,
)
from scripts.build_prose_review_v4_dashboard import (
    DEFAULT_OUTPUT_DIR as DEFAULT_FIRST_PASS_DIR,
)
from scripts.build_prose_review_v4_dashboard import (
    DEFAULT_PROMPT as DEFAULT_FIRST_PASS_PROMPT,
)
from scripts.build_prose_review_v4_dashboard import (
    DEFAULT_STATUS as DEFAULT_FIRST_PASS_STATUS,
)
from scripts.build_prose_review_v4_dashboard import (
    _checkpoint_path as _first_checkpoint_path,
)
from scripts.build_prose_review_v4_dashboard import (
    _status_hashes as _first_status_hashes,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_ROOT = Path("/tmp/danish-personas-audit")
DEFAULT_VERIFY_PROMPT = Path("config/persona-verify-da.md")
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "persona-verify-v4"
DEFAULT_FIRST_PASS_MANIFEST = DEFAULT_FIRST_PASS_DIR / "manifest.json"
CAMPAIGN = "persona-patch-verify-v4"
STATUS_VERSION = 1
MANIFEST_VERSION = 1
TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_HTTP_ATTEMPTS_PER_ROW = 5
MAX_RETRY_DELAY_SECONDS = 120.0
MIN_RETRY_DELAY_SECONDS = 1.0
PROVISIONAL_NOTICE = (
    "accepted is provisional and is not a final release decision; a later release "
    "gate is still required"
)

JSONScalar: t.TypeAlias = str | int | float | bool | None
JSONValue: t.TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class VerifyAttemptResult:
    """One logical verification result and transient retry count."""

    result: ProsePatchVerificationResult
    transient_retries: int


@dataclass(frozen=True)
class VerifyPaths:
    """Filesystem inputs and private output location."""

    original: Path
    candidate: Path
    triage: Path
    first_prompt: Path
    verify_prompt: Path
    registry: Path
    first_status: Path
    first_manifest: Path
    first_checkpoint_root: Path
    output_dir: Path


@dataclass(frozen=True)
class VerifyRow:
    """One first-pass patched row available for second-pass verification."""

    persona_hash: str
    original_row: dict[str, JSONValue]
    candidate_row: dict[str, JSONValue]
    changed_facts: dict[str, dict[str, object]]
    proposed_text: str
    patches: list[dict[str, str]]
    first_checkpoint_sha256: str


@dataclass(frozen=True)
class LoadedFirstPass:
    """Locally verified first-pass decisions and source pins."""

    decisions: list[ReviewDecision]
    rows: list[VerifyRow]
    manual: int
    unchanged_consistent: int
    manifest: dict[str, JSONValue]
    verify_prompt: str


class VerifyRunner(t.Protocol):
    """Callable contract for the injectable proxy patch verifier."""

    def __call__(
        self,
        *,
        row: dict[str, object],
        candidate_row: dict[str, object],
        changed_facts: dict[str, dict[str, object]],
        proposed_text: str,
        patches: list[dict[str, str]],
        first_checkpoint_sha256: str,
        prompt: str,
        config: GenerationConfig,
        budget: ProxyBudget,
        checkpoint_path: Path,
        transport: httpx.BaseTransport,
    ) -> ProsePatchVerificationResult:
        """Run or resume one second-pass patch verification."""


VerifyFutureMap: t.TypeAlias = dict[futures.Future[VerifyAttemptResult], VerifyRow]


class TransientVerificationAttemptsExhausted(Exception):
    """Raised when a row exhausts bounded transient provider retries."""

    def __init__(self, *, transient_retries: int) -> None:
        """Initialise the exhausted-retry marker.

        Args:
            transient_retries: Number of transient retries already consumed.
        """
        super().__init__("Transient provider failures exhausted for one row")
        self.transient_retries = transient_retries


@click.command()
@click.option("--original", type=click.Path(path_type=Path), default=DEFAULT_ORIGINAL)
@click.option("--candidate", type=click.Path(path_type=Path), default=DEFAULT_CANDIDATE)
@click.option("--triage", type=click.Path(path_type=Path), default=DEFAULT_TRIAGE)
@click.option(
    "--first-prompt", type=click.Path(path_type=Path), default=DEFAULT_FIRST_PASS_PROMPT
)
@click.option(
    "--verify-prompt", type=click.Path(path_type=Path), default=DEFAULT_VERIFY_PROMPT
)
@click.option("--registry", type=click.Path(path_type=Path), default=DEFAULT_REGISTRY)
@click.option(
    "--first-status", type=click.Path(path_type=Path), default=DEFAULT_FIRST_PASS_STATUS
)
@click.option(
    "--first-manifest",
    type=click.Path(path_type=Path),
    default=DEFAULT_FIRST_PASS_MANIFEST,
)
@click.option(
    "--first-checkpoint-root",
    type=click.Path(path_type=Path),
    default=DEFAULT_FIRST_PASS_DIR,
)
@click.option(
    "--output-dir", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT_DIR
)
@click.option("--max-rows", type=click.IntRange(min=1), default=None)
@click.option(
    "--workers", type=click.IntRange(min=1, max=4), default=1, show_default=True
)
@click.option("--run", "execute", is_flag=True, default=False)
def main(
    original: Path,
    candidate: Path,
    triage: Path,
    first_prompt: Path,
    verify_prompt: Path,
    registry: Path,
    first_status: Path,
    first_manifest: Path,
    first_checkpoint_root: Path,
    output_dir: Path,
    max_rows: int | None,
    workers: int,
    execute: bool,
) -> None:
    """Run or dry-run the private v4 patch-verification campaign.

    Raises:
        click.ClickException: If source, checkpoint, resume, or proxy state is unsafe.
    """
    configure_cli_logging()
    paths = VerifyPaths(
        original=original,
        candidate=candidate,
        triage=triage,
        first_prompt=first_prompt,
        verify_prompt=verify_prompt,
        registry=registry,
        first_status=first_status,
        first_manifest=first_manifest,
        first_checkpoint_root=first_checkpoint_root,
        output_dir=output_dir,
    )
    try:
        summary = run_patch_verification_campaign(
            paths=paths,
            execute=execute,
            max_rows=max_rows,
            workers=workers,
            verify_runner=_run_proxy_patch_verification_adapter,
        )
    except (
        PatchVerificationCampaignError,
        ProseReviewV4DashboardError,
        ProxyBudgetError,
        ProxyPatchVerificationError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def run_patch_verification_campaign(
    *,
    paths: VerifyPaths,
    execute: bool,
    max_rows: int | None,
    workers: int,
    verify_runner: VerifyRunner,
) -> dict[str, object]:
    """Run or dry-run the second-pass patch-verification campaign.

    Args:
        paths: Input and private output paths.
        execute: If true, provider I/O is allowed. The CLI defaults to dry-run.
        max_rows: Optional prefix bound for pilot processing.
        workers: Worker count from one to four.
        verify_runner: Injectable proxy verifier for offline tests.

    Returns:
        Machine-readable progress summary without raw persona IDs or prose.

    Raises:
        PatchVerificationCampaignError: If inputs, resume state, or workers are unsafe.
    """
    _require_worker_count(workers=workers)
    loaded = load_first_pass(paths=paths)
    manifest = _verification_manifest(paths=paths, loaded=loaded)
    if not execute:
        return _dry_run_summary(
            rows=loaded.rows,
            manual=loaded.manual,
            unchanged_consistent=loaded.unchanged_consistent,
            max_rows=max_rows,
            workers=workers,
        )

    _prepare_private_output(paths.output_dir)
    _write_or_check_manifest(path=paths.output_dir / "manifest.json", manifest=manifest)
    status_path = paths.output_dir / "status.json"
    status = _load_or_create_status(
        status_path=status_path,
        manifest=manifest,
        available=len(loaded.rows),
        manual=loaded.manual,
        unchanged_consistent=loaded.unchanged_consistent,
    )
    _refresh_available_counts(
        status=status,
        available=len(loaded.rows),
        manual=loaded.manual,
        unchanged_consistent=loaded.unchanged_consistent,
    )
    pending = _pending_rows(rows=loaded.rows, status=status, max_rows=max_rows)
    if not pending:
        _write_status(path=status_path, status=status)
        return _public_status_summary(
            status=status, status_path=status_path, max_rows=max_rows, workers=workers
        )

    config = _generation_config(prompt_path=paths.verify_prompt)
    budget = _proxy_budget(paths=paths, prompt=loaded.verify_prompt, manifest=manifest)
    try:
        _process_pending(
            rows=pending,
            status=status,
            status_path=status_path,
            output_dir=paths.output_dir,
            prompt=loaded.verify_prompt,
            config=config,
            budget=budget,
            workers=workers,
            verify_runner=verify_runner,
        )
    finally:
        _write_status(path=status_path, status=status)
    summary = _public_status_summary(
        status=status, status_path=status_path, max_rows=max_rows, workers=workers
    )
    if max_rows is None and summary["pending"] != 0:
        raise PatchVerificationCampaignError(
            "Full patch verification run ended with pending rows"
        )
    return summary


class PatchVerificationCampaignError(Exception):
    """Raised when the patch-verification CLI must fail closed."""


def load_first_pass(*, paths: VerifyPaths) -> LoadedFirstPass:
    """Load and locally revalidate completed first-pass checkpoints.

    Args:
        paths: Input paths for first-pass sources and verification prompt.

    Returns:
        Verified first-pass decisions and second-pass patched-row inputs.

    Raises:
        PatchVerificationCampaignError: If a source row or checkpoint is missing.
    """
    first_manifest = _load_json_object(path=paths.first_manifest, label="manifest.json")
    first_status = _load_json_object(path=paths.first_status, label="status.json")
    dashboard_paths = DashboardPaths(
        original=paths.original,
        candidate=paths.candidate,
        triage=paths.triage,
        prompt=paths.first_prompt,
        registry=paths.registry,
        status=paths.first_status,
        manifest=paths.first_manifest,
        checkpoint_root=paths.first_checkpoint_root,
        output=paths.output_dir / "unused-dashboard.html",
    )
    _verify_status_manifest(status=first_status, manifest=first_manifest)
    first_prompt = _verify_manifest_sources(
        manifest=first_manifest, paths=dashboard_paths
    )
    hashes = _first_status_hashes(status=first_status)
    _verify_status_counts(status=first_status, hashes=hashes)
    original_rows = _rows_by_hash(path=paths.original, label="original")
    candidate_rows = _rows_by_hash(path=paths.candidate, label="candidate")

    decisions: list[ReviewDecision] = []
    verify_rows: list[VerifyRow] = []
    for persona_hash in sorted(hashes):
        checkpoint_path = _first_checkpoint_path(
            checkpoint_root=paths.first_checkpoint_root, persona_hash=persona_hash
        )
        checkpoint = _load_complete_checkpoint(path=checkpoint_path)
        original = original_rows.get(persona_hash)
        candidate = candidate_rows.get(persona_hash)
        if original is None:
            raise PatchVerificationCampaignError("First-pass original row is missing")
        if candidate is None:
            raise PatchVerificationCampaignError("First-pass candidate row is missing")
        decision = _verify_checkpoint_decision(
            checkpoint=checkpoint,
            persona_hash=persona_hash,
            original=original,
            candidate=candidate,
            prompt=first_prompt,
            manifest=first_manifest,
        )
        decisions.append(decision)
        if decision.disposition != "patched":
            continue
        first_sha = checkpoint.get("checkpoint_sha256")
        if not isinstance(first_sha, str):
            raise PatchVerificationCampaignError("First-pass checkpoint SHA is missing")
        changed_facts = _changed_facts_mapping(original=original, candidate=candidate)
        patches = [
            {"old_excerpt": old_excerpt, "new_excerpt": new_excerpt}
            for old_excerpt, new_excerpt in decision.patches
        ]
        verify_rows.append(
            VerifyRow(
                persona_hash=persona_hash,
                original_row=original,
                candidate_row=candidate,
                changed_facts=changed_facts,
                proposed_text=decision.proposed_text,
                patches=patches,
                first_checkpoint_sha256=first_sha,
            )
        )
    _verify_decision_counts(status=first_status, decisions=decisions)
    return LoadedFirstPass(
        decisions=decisions,
        rows=verify_rows,
        manual=sum(
            decision.disposition == "needs_manual_review" for decision in decisions
        ),
        unchanged_consistent=sum(
            decision.disposition == "unchanged_consistent" for decision in decisions
        ),
        manifest=first_manifest,
        verify_prompt=paths.verify_prompt.read_text(encoding="utf-8"),
    )


def _verification_manifest(
    *, paths: VerifyPaths, loaded: LoadedFirstPass
) -> dict[str, JSONValue]:
    schema_hash = sha256_text(
        canonical_json(ProsePatchSecondReview.provider_json_schema())
    )
    return {
        "version": MANIFEST_VERSION,
        "campaign": CAMPAIGN,
        "inputs": {
            "original": sha256_file(paths.original),
            "candidate_v4": sha256_file(paths.candidate),
            "triage": sha256_file(paths.triage),
            "first_pass_manifest": sha256_file(paths.first_manifest),
            "first_pass_manifest_content": sha256_text(canonical_json(loaded.manifest)),
            "verify_prompt": sha256_file(paths.verify_prompt),
            "registry": sha256_file(paths.registry),
            "schema": schema_hash,
        },
        "first_pass_campaign": loaded.manifest.get("campaign"),
        "model": MODEL,
        "base_url": BASE_URL,
        "reasoning_effort": "none",
        "max_tokens": None,
        "maximum_http_attempts": 1,
        "allowed_facts": sorted(_ALLOWED_FACTS),
        "provisional_notice": PROVISIONAL_NOTICE,
    }


def _generation_config(*, prompt_path: Path) -> GenerationConfig:
    return GenerationConfig(
        base_url=BASE_URL,
        model=MODEL,
        api_key_env=None,
        timeout_seconds=120.0,
        maximum_http_attempts=1,
        maximum_total_requests=1,
        retry_backoff_seconds=0.0,
        maximum_rows_per_shard=1,
        max_tokens=None,
        enable_thinking=None,
        reasoning_effort="none",
        prompt=prompt_path,
        origin_label_contract=Path("config/folk2-ieland-labels-da.yaml"),
    )


def _proxy_budget(
    *, paths: VerifyPaths, prompt: str, manifest: dict[str, JSONValue]
) -> ProxyBudget:
    inputs = manifest["inputs"]
    if not isinstance(inputs, dict) or not isinstance(inputs.get("schema"), str):
        raise PatchVerificationCampaignError("Manifest schema hash is malformed")
    return ProxyBudget(
        registry_path=paths.registry,
        campaign=CAMPAIGN,
        source_hash=sha256_text(canonical_json(manifest)),
        prompt_hash=sha256_text(prompt),
        schema_hash=inputs["schema"],
        uncapped=True,
        uncapped_purpose=PATCH_VERIFICATION_PURPOSE,
    )


def _dry_run_summary(
    *,
    rows: list[VerifyRow],
    manual: int,
    unchanged_consistent: int,
    max_rows: int | None,
    workers: int,
) -> dict[str, object]:
    would_process = len(rows) if max_rows is None else min(max_rows, len(rows))
    return {
        "dry_run": True,
        "campaign": CAMPAIGN,
        "available": len(rows),
        "would_process": would_process,
        "accepted": 0,
        "rejected": 0,
        "manual": manual,
        "unchanged_consistent": unchanged_consistent,
        "processed": 0,
        "pending": len(rows),
        "max_rows": max_rows,
        "workers": workers,
        "accepted_is_provisional": True,
        "provisional_notice": PROVISIONAL_NOTICE,
    }


def _prepare_private_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    if output_dir.stat().st_mode & 0o077:
        raise PatchVerificationCampaignError("Output directory must be private (0700)")


def _write_or_check_manifest(*, path: Path, manifest: dict[str, JSONValue]) -> None:
    if path.exists():
        existing = _load_json_object(path=path, label="verification manifest")
        if existing != manifest:
            raise PatchVerificationCampaignError(
                "manifest.json pins do not match current inputs"
            )
        if path.stat().st_mode & 0o777 != 0o600:
            raise PatchVerificationCampaignError("manifest.json must be private (0600)")
        return
    _write_json(path=path, value=manifest)


def _load_or_create_status(
    *,
    status_path: Path,
    manifest: dict[str, JSONValue],
    available: int,
    manual: int,
    unchanged_consistent: int,
) -> dict[str, object]:
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("manifest") != manifest:
            raise PatchVerificationCampaignError(
                "status.json pins do not match current inputs"
            )
        if status_path.stat().st_mode & 0o777 != 0o600:
            raise PatchVerificationCampaignError("status.json must be private (0600)")
        return _validate_status(status)
    status: dict[str, object] = {
        "version": STATUS_VERSION,
        "manifest": manifest,
        "available": available,
        "accepted": 0,
        "rejected": 0,
        "manual": manual,
        "unchanged_consistent": unchanged_consistent,
        "failed": 0,
        "attempted": 0,
        "transient_retries": 0,
        "processed": 0,
        "pending": available,
        "processed_persona_hashes": [],
        "accepted_is_provisional": True,
        "provisional_notice": PROVISIONAL_NOTICE,
    }
    _write_status(path=status_path, status=status)
    return status


def _validate_status(status: object) -> dict[str, object]:
    if not isinstance(status, dict) or status.get("version") != STATUS_VERSION:
        raise PatchVerificationCampaignError("status.json is malformed")
    for key in {
        "available",
        "accepted",
        "rejected",
        "manual",
        "unchanged_consistent",
        "failed",
        "attempted",
        "transient_retries",
        "processed",
        "pending",
    }:
        if not isinstance(status.get(key), int) or int(status[key]) < 0:
            raise PatchVerificationCampaignError("status.json counters are malformed")
    _status_hashes(status)
    return status


def _refresh_available_counts(
    *, status: dict[str, object], available: int, manual: int, unchanged_consistent: int
) -> None:
    status["available"] = available
    status["manual"] = manual
    status["unchanged_consistent"] = unchanged_consistent
    status["pending"] = _pending_count(status=status)
    status["accepted_is_provisional"] = True
    status["provisional_notice"] = PROVISIONAL_NOTICE


def _pending_rows(
    *, rows: list[VerifyRow], status: dict[str, object], max_rows: int | None
) -> list[VerifyRow]:
    processed = set(_status_hashes(status))
    pending = [row for row in rows if row.persona_hash not in processed]
    if max_rows is not None:
        return pending[:max_rows]
    return pending


def _process_pending(
    *,
    rows: list[VerifyRow],
    status: dict[str, object],
    status_path: Path,
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    workers: int,
    verify_runner: VerifyRunner,
) -> None:
    row_iter = iter(rows)
    future_map: VerifyFutureMap = {}
    stop_exc: Exception | None = None
    executor = futures.ThreadPoolExecutor(max_workers=workers)
    try:
        _submit_verification_futures(
            row_iter=row_iter,
            future_map=future_map,
            executor=executor,
            limit=workers,
            output_dir=output_dir,
            prompt=prompt,
            config=config,
            budget=budget,
            verify_runner=verify_runner,
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
                _cancel_not_started(future_map=future_map)
                continue
            if stop_exc is None and not _has_completed_future(future_map=future_map):
                _submit_verification_futures(
                    row_iter=row_iter,
                    future_map=future_map,
                    executor=executor,
                    limit=workers,
                    output_dir=output_dir,
                    prompt=prompt,
                    config=config,
                    budget=budget,
                    verify_runner=verify_runner,
                )
    finally:
        if stop_exc is not None:
            _cancel_not_started(future_map=future_map)
        executor.shutdown(wait=stop_exc is None, cancel_futures=stop_exc is not None)
    if stop_exc is not None:
        raise PatchVerificationCampaignError(
            "Patch verification stopped before all rows were resolved"
        ) from stop_exc


def _submit_verification_futures(
    *,
    row_iter: c.Iterator[VerifyRow],
    future_map: VerifyFutureMap,
    executor: futures.ThreadPoolExecutor,
    limit: int,
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    verify_runner: VerifyRunner,
) -> None:
    while len(future_map) < limit:
        try:
            row = next(row_iter)
        except StopIteration:
            return
        future_map[
            executor.submit(
                _run_one_verification,
                row=row,
                output_dir=output_dir,
                prompt=prompt,
                config=config,
                budget=budget,
                verify_runner=verify_runner,
            )
        ] = row


def _record_completed_future(
    *,
    future: futures.Future[VerifyAttemptResult],
    row: VerifyRow,
    status: dict[str, object],
    status_path: Path,
) -> Exception | None:
    status["attempted"] = _status_int(status, "attempted") + 1
    try:
        attempt_result = future.result()
    except Exception as exc:
        _record_transient_retries(status=status, exc=exc)
        status["failed"] = _status_int(status, "failed") + 1
        status["pending"] = _pending_count(status=status)
        _write_status(path=status_path, status=status)
        return exc
    status["transient_retries"] = (
        _status_int(status, "transient_retries") + attempt_result.transient_retries
    )
    _record_result(status=status, row=row, result=attempt_result.result)
    _write_status(path=status_path, status=status)
    return None


def _record_result(
    *, status: dict[str, object], row: VerifyRow, result: ProsePatchVerificationResult
) -> None:
    if result.accepted:
        status["accepted"] = _status_int(status, "accepted") + 1
    else:
        status["rejected"] = _status_int(status, "rejected") + 1
    status["processed"] = _status_int(status, "processed") + 1
    hashes = _status_hashes(status)
    if row.persona_hash not in hashes:
        hashes.append(row.persona_hash)
    status["processed_persona_hashes"] = hashes
    status["pending"] = _pending_count(status=status)


def _record_transient_retries(*, status: dict[str, object], exc: Exception) -> None:
    if isinstance(exc, TransientVerificationAttemptsExhausted):
        status["transient_retries"] = (
            _status_int(status, "transient_retries") + exc.transient_retries
        )


def _cancel_not_started(*, future_map: VerifyFutureMap) -> None:
    for future in future_map:
        future.cancel()


def _has_completed_future(*, future_map: VerifyFutureMap) -> bool:
    return any(future.done() for future in future_map)


def _run_one_verification(
    *,
    row: VerifyRow,
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    verify_runner: VerifyRunner,
) -> VerifyAttemptResult:
    transient_retries = 0
    for attempt in range(1, MAX_HTTP_ATTEMPTS_PER_ROW + 1):
        transport = httpx.HTTPTransport()
        try:
            result = verify_runner(
                row=row.original_row,
                candidate_row=row.candidate_row,
                changed_facts=row.changed_facts,
                proposed_text=row.proposed_text,
                patches=row.patches,
                first_checkpoint_sha256=row.first_checkpoint_sha256,
                prompt=prompt,
                config=config,
                budget=budget,
                checkpoint_path=_checkpoint_path(
                    output_dir=output_dir, persona_hash=row.persona_hash
                ),
                transport=transport,
            )
        except Exception as exc:
            transport.close()
            if not _is_retryable_transient_error(exc):
                raise
            if attempt == MAX_HTTP_ATTEMPTS_PER_ROW:
                raise TransientVerificationAttemptsExhausted(
                    transient_retries=transient_retries
                ) from exc
            transient_retries += 1
            time.sleep(_retry_delay_seconds(exc=exc, attempt=attempt))
            continue
        transport.close()
        return VerifyAttemptResult(result=result, transient_retries=transient_retries)
    raise PatchVerificationCampaignError("Patch verification retry loop ended")


def _is_retryable_transient_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in TRANSIENT_STATUS_CODES
    return isinstance(exc, httpx.TransportError)


def _retry_delay_seconds(*, exc: Exception, attempt: int) -> float:
    retry_after = _retry_after_seconds(exc=exc)
    delay = 2 ** (attempt - 1) if retry_after is None else retry_after
    return min(MAX_RETRY_DELAY_SECONDS, max(MIN_RETRY_DELAY_SECONDS, float(delay)))


def _retry_after_seconds(*, exc: Exception) -> float | None:
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    value = exc.response.headers.get("Retry-After")
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.isdecimal():
        return float(stripped)
    try:
        retry_at = email.utils.parsedate_to_datetime(stripped)
    except TypeError, ValueError:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=dt.UTC)
    return (retry_at - dt.datetime.now(tz=dt.UTC)).total_seconds()


def _checkpoint_path(*, output_dir: Path, persona_hash: str) -> Path:
    return output_dir / "checkpoints" / persona_hash[:2] / f"{persona_hash}.json"


def _public_status_summary(
    *, status: dict[str, object], status_path: Path, max_rows: int | None, workers: int
) -> dict[str, object]:
    status["pending"] = _pending_count(status=status)
    return {
        "dry_run": False,
        "campaign": CAMPAIGN,
        "status_path": str(status_path),
        "available": status["available"],
        "accepted": status["accepted"],
        "rejected": status["rejected"],
        "manual": status["manual"],
        "unchanged_consistent": status["unchanged_consistent"],
        "failed": status["failed"],
        "attempted": status["attempted"],
        "transient_retries": status["transient_retries"],
        "processed": status["processed"],
        "pending": status["pending"],
        "max_rows": max_rows,
        "workers": workers,
        "accepted_is_provisional": True,
        "provisional_notice": PROVISIONAL_NOTICE,
    }


def _pending_count(*, status: dict[str, object]) -> int:
    return max(0, _status_int(status, "available") - len(_status_hashes(status)))


def _status_hashes(status: dict[str, object]) -> list[str]:
    value = status.get("processed_persona_hashes")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PatchVerificationCampaignError(
            "status.json processed hashes are malformed"
        )
    if len(value) != len(set(value)):
        raise PatchVerificationCampaignError("status.json processed hashes are invalid")
    return list(value)


def _status_int(status: dict[str, object], key: str) -> int:
    value = status.get(key)
    if not isinstance(value, int) or value < 0:
        raise PatchVerificationCampaignError(f"status.json counter is malformed: {key}")
    return value


def _require_worker_count(*, workers: int) -> None:
    if workers < 1 or workers > 4:
        raise PatchVerificationCampaignError("workers must be between one and four")


def _load_json_object(*, path: Path, label: str) -> dict[str, JSONValue]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PatchVerificationCampaignError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise PatchVerificationCampaignError(f"{label} is not a JSON object")
    return t.cast(dict[str, JSONValue], value)


def _write_status(*, path: Path, status: dict[str, object]) -> None:
    _write_json(path=path, value=t.cast(dict[str, JSONValue], status))


def _write_json(*, path: Path, value: dict[str, JSONValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _run_proxy_patch_verification_adapter(
    *,
    row: dict[str, object],
    candidate_row: dict[str, object],
    changed_facts: dict[str, dict[str, object]],
    proposed_text: str,
    patches: list[dict[str, str]],
    first_checkpoint_sha256: str,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    checkpoint_path: Path,
    transport: httpx.BaseTransport,
) -> ProsePatchVerificationResult:
    return run_proxy_patch_verification(
        row=row,
        candidate_row=candidate_row,
        changed_facts=changed_facts,
        proposed_text=proposed_text,
        patches=patches,
        first_checkpoint_sha256=first_checkpoint_sha256,
        prompt=prompt,
        config=config,
        budget=budget,
        checkpoint_path=checkpoint_path,
        transport=transport,
    )


if __name__ == "__main__":
    load_repository_environment()
    main()
