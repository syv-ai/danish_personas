"""Run private persona prose review campaigns."""

from __future__ import annotations

import collections.abc as c
import concurrent.futures as futures
import datetime as dt
import email.utils
import json
import logging
import os
import re
import tempfile
import time
import typing as t
from dataclasses import dataclass
from pathlib import Path

import click
import httpx
import polars as pl

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.environment import load_repository_environment
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.prose_review import (
    ProseReviewResponse,
    ProseReviewResult,
)
from danish_personas.generation.proxy_budget import (
    BASE_URL,
    EDUCATION_REVIEW_PURPOSE,
    MODEL,
    REVIEW_MODEL_ENV,
    ProxyBudget,
    ProxyBudgetError,
    require_runtime_model,
)
from danish_personas.generation.proxy_patch_runner import _ALLOWED_FACTS
from danish_personas.generation.proxy_review_runner import (
    ProxyReviewError,
    run_proxy_review,
)
from danish_personas.io import canonical_json, sha256_file, sha256_text

LOGGER = logging.getLogger(__name__)

DEFAULT_ROOT = Path("/tmp/danish-personas-audit")
DEFAULT_ORIGINAL = DEFAULT_ROOT / "data/train-00000-of-00001.parquet"
DEFAULT_CANDIDATE = DEFAULT_ROOT / "attribute-candidate-v4.parquet"
DEFAULT_H90_CANDIDATE = DEFAULT_ROOT / "attribute-candidate-v5-h90-PROVISIONAL.parquet"
DEFAULT_TRIAGE = DEFAULT_ROOT / "prose-triage-v1.json"
DEFAULT_PROMPT = Path("config/persona-review-da.md")
DEFAULT_H90_PROMPT = Path("config/persona-review-h90-da.md")
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "persona-review-v4"
DEFAULT_H90_OUTPUT_DIR = DEFAULT_ROOT / "persona-review-h90-v5"
DEFAULT_REGISTRY = Path.home() / ".pi" / "agent" / "models-store.json"
EXPECTED_ORIGINAL_SHA256 = (
    "c178e63d40046274bcc559bdd9322f32336656da980c1809f794d68449d6250f"
)
EXPECTED_CANDIDATE_SHA256 = (
    "8e7f770f8163b51134f2f9eb9e1dd74bfc247dbdf8287559065966de597ea8cf"
)
CAMPAIGN = "persona-prose-review-v4"
H90_CAMPAIGN = "persona-prose-review-h90-v5"
DEFAULT_BUDGET_PURPOSE = "v4"
H90_BUDGET_PURPOSE = EDUCATION_REVIEW_PURPOSE
H90_CHANGED_ROWS = 506
H90_CHANGED_FIELDS = frozenset({"education_level", "education_source_code"})
ID_FIELD = "persona_id"
PERSONA_FIELD = "persona"
TRIAGE_CLASSIFICATION = "needs_prose_review_or_regeneration"
STATUS_VERSION = 1
MANIFEST_VERSION = 1
TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_HTTP_ATTEMPTS_PER_ROW = 5
MAX_RETRY_DELAY_SECONDS = 120.0
MIN_RETRY_DELAY_SECONDS = 1.0

RESTRICTED_FIELDS = frozenset(
    {
        "origin_country",
        "origin_country_code",
        "origin_country_resolution",
        "origin_contract_sha256",
        "sexual_orientation",
        "partner_sexual_orientation",
        "transgender",
        "partner_transgender",
        "variation_in_sex_characteristics",
        "partner_variation_in_sex_characteristics",
        "same_sex_partner_target",
        "partner_same_sex_target",
    }
)
SENSITIVE_TEXT = re.compile(
    r"\b(?:sexual\s+orientation|seksuel\s+orientering|homoseksuel|"
    r"biseksuel|heteroseksuel|lesbisk|queer|transkønnet|transgender|"
    r"transseksuel|interkønnet|intersex|sex\s+characteristics)\b",
    re.IGNORECASE,
)

JSONScalar: t.TypeAlias = str | int | float | bool | None
JSONValue: t.TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
BudgetPurpose: t.TypeAlias = t.Literal["v4", "h90_v5"]


@dataclass(frozen=True)
class ReviewAttemptResult:
    """One logical row result and the transient retries it needed."""

    result: ProseReviewResult
    transient_retries: int


@dataclass(frozen=True)
class ReviewInputs:
    """Loaded prose review inputs keyed by private persona ID in memory only."""

    original_rows: dict[str, dict[str, object]]
    candidate_rows: dict[str, dict[str, object]]
    triage_personas: dict[str, dict[str, object]]
    prompt: str
    h90_report: dict[str, object] | None


@dataclass(frozen=True)
class ReviewPaths:
    """Filesystem inputs and private output location."""

    original: Path
    candidate: Path
    triage: Path
    prompt: Path
    output_dir: Path
    registry: Path


@dataclass(frozen=True)
class ReviewRow:
    """One row that may be resolved by the proxy review campaign."""

    persona_id: str
    persona_hash: str
    original_row: dict[str, object]
    candidate_row: dict[str, object]
    changed_facts: dict[str, dict[str, object]]


class ReviewRunner(t.Protocol):
    """Callable contract for the injectable proxy review runner."""

    def __call__(
        self,
        *,
        row: dict[str, object],
        candidate_row: dict[str, object],
        changed_facts: dict[str, dict[str, object]],
        prompt: str,
        config: GenerationConfig,
        budget: ProxyBudget,
        checkpoint_path: Path,
        transport: httpx.BaseTransport,
    ) -> ProseReviewResult:
        """Run or resume one prose review."""


class TransientReviewAttemptsExhausted(RuntimeError):
    """Raised when a row exhausts bounded transient provider retries."""

    def __init__(self, *, transient_retries: int) -> None:
        """Initialise with the number of retries already consumed."""
        super().__init__("Transient prose review provider failures exhausted")
        self.transient_retries = transient_retries


ReviewFutureMap: t.TypeAlias = dict[futures.Future[ReviewAttemptResult], ReviewRow]


@click.command()
@click.option("--original", type=click.Path(path_type=Path), default=DEFAULT_ORIGINAL)
@click.option("--candidate", type=click.Path(path_type=Path), default=DEFAULT_CANDIDATE)
@click.option("--triage", type=click.Path(path_type=Path), default=DEFAULT_TRIAGE)
@click.option("--prompt", type=click.Path(path_type=Path), default=DEFAULT_PROMPT)
@click.option(
    "--output-dir", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT_DIR
)
@click.option("--registry", type=click.Path(path_type=Path), default=DEFAULT_REGISTRY)
@click.option("--max-rows", type=click.IntRange(min=1), default=None)
@click.option(
    "--workers", type=click.IntRange(min=1, max=4), default=1, show_default=True
)
@click.option(
    "--budget-purpose",
    type=click.Choice([DEFAULT_BUDGET_PURPOSE, H90_BUDGET_PURPOSE]),
    default=DEFAULT_BUDGET_PURPOSE,
    show_default=True,
)
@click.option("--run", "execute", is_flag=True, default=False)
def main(
    original: Path,
    candidate: Path,
    triage: Path,
    prompt: Path,
    output_dir: Path,
    registry: Path,
    max_rows: int | None,
    workers: int,
    budget_purpose: BudgetPurpose,
    execute: bool,
) -> None:
    """Run or dry-run a private persona prose review campaign.

    Raises:
        click.ClickException: If input, resume, budget, or proxy boundaries are unsafe.
    """
    configure_cli_logging()
    if budget_purpose == H90_BUDGET_PURPOSE:
        if original == DEFAULT_ORIGINAL:
            original = DEFAULT_CANDIDATE
        if candidate == DEFAULT_CANDIDATE:
            candidate = DEFAULT_H90_CANDIDATE
        if prompt == DEFAULT_PROMPT:
            prompt = DEFAULT_H90_PROMPT
        if output_dir == DEFAULT_OUTPUT_DIR:
            output_dir = DEFAULT_H90_OUTPUT_DIR
    paths = ReviewPaths(
        original=original,
        candidate=candidate,
        triage=triage,
        prompt=prompt,
        output_dir=output_dir,
        registry=registry,
    )
    try:
        expected_original_sha256 = EXPECTED_ORIGINAL_SHA256
        expected_candidate_sha256 = EXPECTED_CANDIDATE_SHA256
        if budget_purpose == H90_BUDGET_PURPOSE:
            _require_h90_private_inputs(paths=paths)
            expected_original_sha256 = EXPECTED_CANDIDATE_SHA256
            expected_candidate_sha256 = _h90_report_candidate_sha256(
                candidate=paths.candidate
            )
        summary = run_review_campaign(
            paths=paths,
            execute=execute,
            max_rows=max_rows,
            workers=workers,
            expected_original_sha256=expected_original_sha256,
            expected_candidate_sha256=expected_candidate_sha256,
            review_runner=_run_proxy_review_adapter,
            budget_purpose=budget_purpose,
        )
    except (PersonaProseReviewError, ProxyBudgetError, ProxyReviewError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def _h90_report_candidate_sha256(*, candidate: Path) -> str:
    report_path = _h90_report_path(candidate=candidate)
    document = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PersonaProseReviewError("H90 v5 report JSON must be an object")
    for key in (
        "candidate_sha256",
        "candidate_v5_sha256",
        "candidate_h90_v5_sha256",
        "output_sha256",
        "parquet_sha256",
    ):
        value = document.get(key)
        if isinstance(value, str):
            if not _is_sha256(value):
                raise PersonaProseReviewError(
                    "H90 v5 report candidate SHA-256 malformed"
                )
            actual = sha256_file(candidate)
            if value != actual:
                raise PersonaProseReviewError(
                    "H90 v5 report candidate SHA-256 mismatch"
                )
            return value
    raise PersonaProseReviewError("H90 v5 report lacks candidate SHA-256")


class PersonaProseReviewError(Exception):
    """Raised when the prose review CLI must fail closed."""


def _h90_report_path(*, candidate: Path) -> Path:
    return candidate.with_suffix(".report.json")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _require_h90_private_inputs(*, paths: ReviewPaths) -> None:
    for path, label in (
        (paths.original, "H90 original v4 parquet"),
        (paths.candidate, "H90 candidate parquet"),
        (paths.triage, "H90 triage JSON"),
        (_h90_report_path(candidate=paths.candidate), "H90 report JSON"),
    ):
        _require_private_file(path=path, label=label)


def _require_private_file(*, path: Path, label: str) -> None:
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise PersonaProseReviewError(f"{label} is missing or unreadable") from exc
    if mode & 0o077:
        raise PersonaProseReviewError(f"{label} must be private (mode 0600)")


def run_review_campaign(
    *,
    paths: ReviewPaths,
    execute: bool,
    max_rows: int | None,
    workers: int,
    expected_original_sha256: str,
    expected_candidate_sha256: str,
    review_runner: ReviewRunner,
    budget_purpose: BudgetPurpose = DEFAULT_BUDGET_PURPOSE,
) -> dict[str, object]:
    """Run or dry-run a prose review campaign.

    Args:
        paths: Input and private output paths.
        execute: If true, provider I/O is allowed. Defaults to dry-run in the CLI.
        max_rows: Optional prefix bound for pilot processing.
        workers: Worker count from one to four.
        expected_original_sha256: Required SHA-256 of the frozen baseline parquet.
        expected_candidate_sha256: Required SHA-256 of the selected candidate parquet.
        review_runner: Injectable proxy review runner for offline tests.
        budget_purpose: Explicit local ledger purpose. Defaults to the pinned v4
            prose review campaign.

    Returns:
        Machine-readable progress summary without raw persona IDs or prose.

    Raises:
        PersonaProseReviewError: If source, resume, or worker inputs are unsafe.
    """
    _require_worker_count(workers=workers)
    _require_budget_purpose(budget_purpose=budget_purpose)
    h90_report = None
    if budget_purpose == H90_BUDGET_PURPOSE:
        _require_h90_private_inputs(paths=paths)
        h90_report = _load_h90_report(candidate=paths.candidate)
        if h90_report["candidate_sha256"] != expected_candidate_sha256:
            raise PersonaProseReviewError("H90 v5 report candidate SHA-256 mismatch")
    base_manifest = _base_manifest(
        paths=paths,
        expected_original_sha256=expected_original_sha256,
        expected_candidate_sha256=expected_candidate_sha256,
        budget_purpose=budget_purpose,
        h90_report=h90_report,
    )
    loaded = load_review_inputs(paths=paths, h90_report=h90_report)
    selection = select_review_rows(inputs=loaded)
    manifest = _campaign_manifest(base_manifest=base_manifest, selection=selection)
    if not execute:
        return _dry_run_summary(
            manifest=manifest, selection=selection, max_rows=max_rows, workers=workers
        )

    _prepare_private_output(paths.output_dir)
    _write_or_check_manifest(path=paths.output_dir / "manifest.json", manifest=manifest)
    status_path = paths.output_dir / "status.json"
    status = _load_or_create_status(
        status_path=status_path, manifest=manifest, selection=selection
    )
    pending = _pending_rows(selection=selection, status=status, max_rows=max_rows)
    if not pending:
        return _public_status_summary(
            status=status, status_path=status_path, max_rows=max_rows, workers=workers
        )

    config = _generation_config(prompt_path=paths.prompt)
    budget = _proxy_budget(
        paths=paths,
        prompt=loaded.prompt,
        manifest=manifest,
        budget_purpose=budget_purpose,
    )
    try:
        _process_pending(
            rows=pending,
            status=status,
            status_path=status_path,
            output_dir=paths.output_dir,
            prompt=loaded.prompt,
            config=config,
            budget=budget,
            workers=workers,
            review_runner=review_runner,
        )
    finally:
        _write_status(path=status_path, status=status)
    summary = _public_status_summary(
        status=status, status_path=status_path, max_rows=max_rows, workers=workers
    )
    if max_rows is None and summary["pending"] != 0:
        raise PersonaProseReviewError("Full prose review run ended with pending rows")
    return summary


def _base_manifest(
    *,
    paths: ReviewPaths,
    expected_original_sha256: str,
    expected_candidate_sha256: str,
    budget_purpose: BudgetPurpose,
    h90_report: dict[str, object] | None,
) -> dict[str, JSONValue]:
    original_sha256 = sha256_file(paths.original)
    if original_sha256 != expected_original_sha256:
        raise PersonaProseReviewError("Frozen original parquet SHA-256 does not match")
    candidate_sha256 = sha256_file(paths.candidate)
    if candidate_sha256 != expected_candidate_sha256:
        raise PersonaProseReviewError("candidate parquet SHA-256 does not match")
    schema_hash = sha256_text(
        canonical_json(ProseReviewResponse.provider_json_schema())
    )
    inputs: dict[str, JSONValue] = {
        "triage": sha256_file(paths.triage),
        "prompt": sha256_file(paths.prompt),
        "registry": sha256_file(paths.registry),
        "schema": schema_hash,
    }
    campaign = CAMPAIGN
    if budget_purpose == H90_BUDGET_PURPOSE:
        if h90_report is None:
            raise PersonaProseReviewError("H90 v5 report is required")
        campaign = H90_CAMPAIGN
        inputs.update(
            {
                "original_v4": original_sha256,
                "candidate_h90_v5": candidate_sha256,
                "h90_report": sha256_file(_h90_report_path(candidate=paths.candidate)),
            }
        )
    else:
        inputs.update({"original": original_sha256, "candidate_v4": candidate_sha256})
    manifest: dict[str, JSONValue] = {
        "version": MANIFEST_VERSION,
        "campaign": campaign,
        "inputs": inputs,
        "model": MODEL,
        "base_url": BASE_URL,
        "reasoning_effort": "none",
        "max_tokens": None,
        "maximum_http_attempts": 1,
        "triage_classification": TRIAGE_CLASSIFICATION,
        "allowed_facts": sorted(_ALLOWED_FACTS),
    }
    if budget_purpose == H90_BUDGET_PURPOSE:
        manifest["budget_purpose"] = budget_purpose
    return manifest


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


def _load_h90_report(*, candidate: Path) -> dict[str, object]:
    report_path = _h90_report_path(candidate=candidate)
    document = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PersonaProseReviewError("H90 v5 report JSON must be an object")
    candidate_sha256 = _h90_report_candidate_sha256(candidate=candidate)
    changed_hashes = document.get("changed_persona_id_sha256")
    if not isinstance(changed_hashes, list) or not all(
        _is_sha256(value) for value in changed_hashes
    ):
        raise PersonaProseReviewError("H90 v5 report changed ID hashes are malformed")
    if len(changed_hashes) != H90_CHANGED_ROWS or len(set(changed_hashes)) != len(
        changed_hashes
    ):
        raise PersonaProseReviewError("H90 v5 report must contain exactly 506 IDs")
    changed_rows = document.get("changed_rows")
    if changed_rows is not None and changed_rows != H90_CHANGED_ROWS:
        raise PersonaProseReviewError("H90 v5 report changed row count mismatch")
    return {
        "candidate_sha256": candidate_sha256,
        "changed_persona_id_sha256": sorted(changed_hashes),
    }


def _validate_status(status: object) -> dict[str, object]:
    if not isinstance(status, dict):
        raise PersonaProseReviewError("status.json must be an object")
    status.setdefault("transient_retries", 0)
    for key in (
        "total_triage_selected",
        "reviewable",
        "no_changed_fact",
        "privacy_skipped",
        "patched",
        "unchanged_consistent",
        "needs_manual_review",
        "failed",
        "attempted",
        "transient_retries",
        "processed",
        "pending",
    ):
        _status_int(status, key)
    _status_hashes(status)
    return status


def _status_hashes(status: dict[str, object]) -> list[str]:
    value = status.get("processed_persona_hashes")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PersonaProseReviewError("status.json processed hashes are invalid")
    if len(value) != len(set(value)):
        raise PersonaProseReviewError("status.json contains duplicate processed rows")
    return list(value)


def _status_int(status: dict[str, object], key: str) -> int:
    value = status.get(key)
    if not isinstance(value, int) or value < 0:
        raise PersonaProseReviewError("status.json progress counters are invalid")
    return value


def _write_status(*, path: Path, status: dict[str, object]) -> None:
    _write_json(path=path, value=t.cast(dict[str, JSONValue], status))


def _write_json(*, path: Path, value: dict[str, JSONValue]) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _prepare_private_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    if output_dir.stat().st_mode & 0o077:
        raise PersonaProseReviewError("Output directory must be private (mode 0700)")


def _process_pending(
    *,
    rows: list[ReviewRow],
    status: dict[str, object],
    status_path: Path,
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    workers: int,
    review_runner: ReviewRunner,
) -> None:
    row_iter = iter(rows)
    future_map: ReviewFutureMap = {}
    stop_exc: Exception | None = None
    executor = futures.ThreadPoolExecutor(max_workers=workers)
    try:
        _submit_review_futures(
            row_iter=row_iter,
            future_map=future_map,
            executor=executor,
            limit=workers,
            output_dir=output_dir,
            prompt=prompt,
            config=config,
            budget=budget,
            review_runner=review_runner,
        )
        while future_map:
            done, _ = futures.wait(future_map, return_when=futures.FIRST_COMPLETED)
            future = next(iter(done))
            row = future_map.pop(future)
            if future.cancelled():
                continue
            completed_exc = _record_completed_review_future(
                future=future, row=row, status=status, status_path=status_path
            )
            if completed_exc is not None:
                if stop_exc is None:
                    stop_exc = completed_exc
                _cancel_not_started(future_map=future_map)
                continue
            if stop_exc is None and not _has_completed_future(future_map=future_map):
                _submit_review_futures(
                    row_iter=row_iter,
                    future_map=future_map,
                    executor=executor,
                    limit=workers,
                    output_dir=output_dir,
                    prompt=prompt,
                    config=config,
                    budget=budget,
                    review_runner=review_runner,
                )
    finally:
        if stop_exc is not None:
            _cancel_not_started(future_map=future_map)
        executor.shutdown(wait=stop_exc is None, cancel_futures=stop_exc is not None)
    if stop_exc is not None:
        raise PersonaProseReviewError(
            "Prose review stopped before all rows were resolved"
        ) from stop_exc


def _cancel_not_started(*, future_map: ReviewFutureMap) -> None:
    for future in future_map:
        future.cancel()


def _has_completed_future(*, future_map: ReviewFutureMap) -> bool:
    return any(future.done() for future in future_map)


def _record_completed_review_future(
    *,
    future: futures.Future[ReviewAttemptResult],
    row: ReviewRow,
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
        if _must_stop(exc):
            return exc
        LOGGER.warning("Persona prose review failed for one selected row")
        return None
    status["transient_retries"] = (
        _status_int(status, "transient_retries") + attempt_result.transient_retries
    )
    _record_result(status=status, row=row, result=attempt_result.result)
    _write_status(path=status_path, status=status)
    return None


def _must_stop(exc: Exception) -> bool:
    if isinstance(exc, TransientReviewAttemptsExhausted):
        return True
    if isinstance(exc, (ProxyBudgetError, ProxyReviewError)):
        return True
    return isinstance(exc, httpx.HTTPStatusError)


def _pending_count(*, status: dict[str, object]) -> int:
    return max(0, _status_int(status, "reviewable") - _status_int(status, "processed"))


def _record_result(
    *, status: dict[str, object], row: ReviewRow, result: ProseReviewResult
) -> None:
    if result.disposition not in {
        "patched",
        "unchanged_consistent",
        "needs_manual_review",
    }:
        raise PersonaProseReviewError("Prose review returned an unknown disposition")
    status[result.disposition] = _status_int(status, result.disposition) + 1
    status["processed"] = _status_int(status, "processed") + 1
    hashes = _status_hashes(status)
    if row.persona_hash not in hashes:
        hashes.append(row.persona_hash)
    status["processed_persona_hashes"] = hashes
    status["pending"] = _pending_count(status=status)


def _record_transient_retries(*, status: dict[str, object], exc: Exception) -> None:
    if isinstance(exc, TransientReviewAttemptsExhausted):
        status["transient_retries"] = (
            _status_int(status, "transient_retries") + exc.transient_retries
        )


def _submit_review_futures(
    *,
    row_iter: c.Iterator[ReviewRow],
    future_map: ReviewFutureMap,
    executor: futures.ThreadPoolExecutor,
    limit: int,
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    review_runner: ReviewRunner,
) -> None:
    while len(future_map) < limit:
        try:
            row = next(row_iter)
        except StopIteration:
            return
        future_map[
            executor.submit(
                _run_one_review,
                row=row,
                output_dir=output_dir,
                prompt=prompt,
                config=config,
                budget=budget,
                review_runner=review_runner,
            )
        ] = row


def _proxy_budget(
    *,
    paths: ReviewPaths,
    prompt: str,
    manifest: dict[str, JSONValue],
    budget_purpose: BudgetPurpose,
) -> ProxyBudget:
    inputs = manifest["inputs"]
    if not isinstance(inputs, dict) or not isinstance(inputs.get("schema"), str):
        raise PersonaProseReviewError("Manifest schema hash is malformed")
    uncapped_purpose = None
    if budget_purpose == H90_BUDGET_PURPOSE:
        uncapped_purpose = H90_BUDGET_PURPOSE
    return ProxyBudget(
        registry_path=paths.registry,
        campaign=str(manifest["campaign"]),
        source_hash=sha256_text(canonical_json(manifest)),
        prompt_hash=sha256_text(prompt),
        schema_hash=inputs["schema"],
        uncapped=True,
        uncapped_purpose=uncapped_purpose,
    )


def _public_status_summary(
    *, status: dict[str, object], status_path: Path, max_rows: int | None, workers: int
) -> dict[str, object]:
    status["pending"] = _pending_count(status=status)
    manifest = status.get("manifest", {})
    campaign = manifest.get("campaign") if isinstance(manifest, dict) else CAMPAIGN
    return {
        "dry_run": False,
        "campaign": campaign,
        "status_path": str(status_path),
        "total_triage_selected": status["total_triage_selected"],
        "reviewable": status["reviewable"],
        "patched": status["patched"],
        "unchanged_consistent": status["unchanged_consistent"],
        "needs_manual_review": status["needs_manual_review"],
        "privacy_skipped": status["privacy_skipped"],
        "no_changed_fact": status["no_changed_fact"],
        "failed": status["failed"],
        "attempted": status["attempted"],
        "transient_retries": status["transient_retries"],
        "processed": status["processed"],
        "pending": status["pending"],
        "max_rows": max_rows,
        "workers": workers,
    }


def _require_budget_purpose(*, budget_purpose: BudgetPurpose) -> None:
    if budget_purpose not in {DEFAULT_BUDGET_PURPOSE, H90_BUDGET_PURPOSE}:
        raise PersonaProseReviewError("Unsupported budget purpose")


def _require_worker_count(*, workers: int) -> None:
    if workers < 1 or workers > 4:
        raise PersonaProseReviewError("workers must be between 1 and 4")


def _write_or_check_manifest(*, path: Path, manifest: dict[str, JSONValue]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise PersonaProseReviewError(
                "manifest.json pins do not match current inputs"
            )
        if path.stat().st_mode & 0o777 != 0o600:
            raise PersonaProseReviewError("manifest.json must be private (mode 0600)")
        return
    _write_json(path=path, value=manifest)


def load_review_inputs(
    *, paths: ReviewPaths, h90_report: dict[str, object] | None = None
) -> ReviewInputs:
    """Load only the inputs needed to compute real allowlisted fact changes.

    Returns:
        Original rows, candidate rows, triage metadata, and prompt text.
    """
    original = pl.read_parquet(paths.original)
    candidate = pl.read_parquet(paths.candidate)
    _require_columns(
        frame=original, columns={ID_FIELD, PERSONA_FIELD}, label="original"
    )
    _require_columns(
        frame=candidate, columns={ID_FIELD, PERSONA_FIELD}, label="candidate"
    )
    return ReviewInputs(
        original_rows=_rows_by_id(original, label="original"),
        candidate_rows=_rows_by_id(candidate, label="candidate"),
        triage_personas=_load_triage_personas(paths.triage),
        prompt=paths.prompt.read_text(encoding="utf-8"),
        h90_report=h90_report,
    )


def _load_triage_personas(path: Path) -> dict[str, dict[str, object]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not {"personas", "counts"}.issubset(document):
        raise PersonaProseReviewError("Triage JSON must contain personas and counts")
    personas = document["personas"]
    if not isinstance(personas, dict):
        raise PersonaProseReviewError("Triage personas must be keyed by persona ID")
    parsed: dict[str, dict[str, object]] = {}
    for persona_id, entry in personas.items():
        if not isinstance(persona_id, str) or not isinstance(entry, dict):
            raise PersonaProseReviewError("Triage persona entries are malformed")
        parsed[persona_id] = entry
    return parsed


def _require_columns(*, frame: pl.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise PersonaProseReviewError(
            f"{label} parquet is missing required columns: {sorted(missing)}"
        )


def _rows_by_id(frame: pl.DataFrame, *, label: str) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for row in frame.to_dicts():
        persona_id = row.get(ID_FIELD)
        if not isinstance(persona_id, str) or not persona_id:
            raise PersonaProseReviewError(f"{label} row has a malformed persona_id")
        if persona_id in rows:
            raise PersonaProseReviewError(f"{label} contains duplicate persona_id")
        rows[persona_id] = dict(row)
    return rows


@dataclass(frozen=True)
class Selection:
    """Deterministic full-campaign selection and static skip counts."""

    reviewable: list[ReviewRow]
    total_triage_selected: int
    no_changed_fact: int
    privacy_skipped: int


def _campaign_manifest(
    *, base_manifest: dict[str, JSONValue], selection: Selection
) -> dict[str, JSONValue]:
    return {
        **base_manifest,
        "selection": {
            "total_triage_selected": selection.total_triage_selected,
            "reviewable": len(selection.reviewable),
            "no_changed_fact": selection.no_changed_fact,
            "privacy_skipped": selection.privacy_skipped,
        },
    }


def _dry_run_summary(
    *,
    manifest: dict[str, JSONValue],
    selection: Selection,
    max_rows: int | None,
    workers: int,
) -> dict[str, object]:
    reviewable = len(selection.reviewable)
    would_process = min(reviewable, max_rows) if max_rows is not None else reviewable
    return {
        "dry_run": True,
        "campaign": manifest["campaign"],
        "manifest_sha256": sha256_text(canonical_json(manifest)),
        "total_triage_selected": selection.total_triage_selected,
        "reviewable": reviewable,
        "would_process": would_process,
        "no_changed_fact": selection.no_changed_fact,
        "privacy_skipped": selection.privacy_skipped,
        "workers": workers,
    }


def _load_or_create_status(
    *, status_path: Path, manifest: dict[str, JSONValue], selection: Selection
) -> dict[str, object]:
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("manifest") != manifest:
            raise PersonaProseReviewError(
                "status.json pins do not match current inputs"
            )
        return _validate_status(status)
    status: dict[str, object] = {
        "version": STATUS_VERSION,
        "manifest": manifest,
        "total_triage_selected": selection.total_triage_selected,
        "reviewable": len(selection.reviewable),
        "no_changed_fact": selection.no_changed_fact,
        "privacy_skipped": selection.privacy_skipped,
        "patched": 0,
        "unchanged_consistent": 0,
        "needs_manual_review": 0,
        "failed": 0,
        "attempted": 0,
        "transient_retries": 0,
        "processed": 0,
        "pending": len(selection.reviewable),
        "processed_persona_hashes": [],
    }
    _write_status(path=status_path, status=status)
    return status


def _pending_rows(
    *, selection: Selection, status: dict[str, object], max_rows: int | None
) -> list[ReviewRow]:
    processed = set(_status_hashes(status))
    pending = [row for row in selection.reviewable if row.persona_hash not in processed]
    if max_rows is not None:
        return pending[:max_rows]
    return pending


def select_review_rows(*, inputs: ReviewInputs) -> Selection:
    """Select all triage review IDs and label rows with no allowed fact changes.

    Returns:
        Full-campaign selection and static skip counts.

    Raises:
        PersonaProseReviewError: If the opt-in H90 v5 selection is unsafe.
    """
    if inputs.h90_report is not None:
        _validate_h90_inputs(inputs=inputs)
    reviewable: list[ReviewRow] = []
    total = 0
    no_changed_fact = 0
    privacy_skipped = 0
    for persona_id, entry in sorted(inputs.triage_personas.items()):
        if entry.get("classification") != TRIAGE_CLASSIFICATION:
            continue
        total += 1
        row = _selected_row(persona_id=persona_id, inputs=inputs)
        if row is None:
            if inputs.h90_report is not None:
                raise PersonaProseReviewError("H90 v5 selected row has no H90 change")
            no_changed_fact += 1
            continue
        if _has_privacy_risk(row=row):
            if inputs.h90_report is not None:
                raise PersonaProseReviewError("H90 v5 selected row is privacy-unsafe")
            privacy_skipped += 1
            continue
        reviewable.append(row)
    if inputs.h90_report is not None and len(reviewable) != H90_CHANGED_ROWS:
        raise PersonaProseReviewError("H90 v5 selection must contain exactly 506 rows")
    return Selection(
        reviewable=reviewable,
        total_triage_selected=total,
        no_changed_fact=no_changed_fact,
        privacy_skipped=privacy_skipped,
    )


def _has_privacy_risk(*, row: ReviewRow) -> bool:
    changed_fields = _actual_changed_fields(
        original=row.original_row, candidate=row.candidate_row
    )
    if changed_fields & RESTRICTED_FIELDS:
        return True
    text_values = [
        str(value) for pair in row.changed_facts.values() for value in pair.values()
    ]
    text_values.append(str(row.original_row.get(PERSONA_FIELD, "")))
    return any(SENSITIVE_TEXT.search(value) is not None for value in text_values)


def _actual_changed_fields(
    *, original: dict[str, object], candidate: dict[str, object]
) -> set[str]:
    fields = (set(original) & set(candidate)) - {PERSONA_FIELD}
    return {field for field in fields if original[field] != candidate[field]}


def _selected_row(*, persona_id: str, inputs: ReviewInputs) -> ReviewRow | None:
    original = inputs.original_rows.get(persona_id)
    candidate = inputs.candidate_rows.get(persona_id)
    if original is None or candidate is None:
        return None
    if original.get(PERSONA_FIELD) != candidate.get(PERSONA_FIELD):
        raise PersonaProseReviewError("Candidate prose must match original prose")
    changed_facts = _changed_facts(original=original, candidate=candidate)
    if not changed_facts:
        return None
    return ReviewRow(
        persona_id=persona_id,
        persona_hash=sha256_text(persona_id),
        original_row=original,
        candidate_row=candidate,
        changed_facts=changed_facts,
    )


def _changed_facts(
    *, original: dict[str, object], candidate: dict[str, object]
) -> dict[str, dict[str, object]]:
    facts: dict[str, dict[str, object]] = {}
    for field in sorted((set(original) & set(candidate)) & _ALLOWED_FACTS):
        old = original[field]
        new = candidate[field]
        if old != new:
            facts[field] = {"old": old, "new": new}
    return facts


def _validate_h90_inputs(*, inputs: ReviewInputs) -> None:
    if inputs.h90_report is None:
        raise PersonaProseReviewError("H90 v5 report is required")
    report_hashes = _h90_report_hashes(report=inputs.h90_report)
    selected_hashes = sorted(
        sha256_text(persona_id)
        for persona_id, entry in inputs.triage_personas.items()
        if entry.get("classification") == TRIAGE_CLASSIFICATION
    )
    if selected_hashes != report_hashes:
        raise PersonaProseReviewError("H90 v5 triage IDs do not match report")
    actual_hashes: list[str] = []
    for persona_id, original in sorted(inputs.original_rows.items()):
        candidate = inputs.candidate_rows.get(persona_id)
        if candidate is None:
            raise PersonaProseReviewError("H90 v5 candidate is missing a v4 row")
        changed_fields = _actual_changed_fields(original=original, candidate=candidate)
        if not changed_fields:
            continue
        if changed_fields != H90_CHANGED_FIELDS:
            raise PersonaProseReviewError("H90 v5 candidate changed non-H90 fields")
        if (
            candidate.get("education_source_code") != "H90"
            or candidate.get("education_level") != "not_stated"
        ):
            raise PersonaProseReviewError("H90 v5 candidate has invalid H90 values")
        actual_hashes.append(sha256_text(persona_id))
    if set(inputs.candidate_rows) != set(inputs.original_rows):
        raise PersonaProseReviewError("H90 v5 candidate row IDs do not match v4")
    if sorted(actual_hashes) != report_hashes:
        raise PersonaProseReviewError(
            "H90 v5 candidate changed IDs do not match report"
        )


def _h90_report_hashes(*, report: dict[str, object]) -> list[str]:
    hashes = report.get("changed_persona_id_sha256")
    if not isinstance(hashes, list) or not all(_is_sha256(value) for value in hashes):
        raise PersonaProseReviewError("H90 v5 report changed ID hashes are malformed")
    return sorted(str(value) for value in hashes)


def _run_one_review(
    *,
    row: ReviewRow,
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    review_runner: ReviewRunner,
) -> ReviewAttemptResult:
    transient_retries = 0
    for attempt in range(1, MAX_HTTP_ATTEMPTS_PER_ROW + 1):
        transport = httpx.HTTPTransport()
        try:
            result = review_runner(
                row=row.original_row,
                candidate_row=row.candidate_row,
                changed_facts=row.changed_facts,
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
                raise TransientReviewAttemptsExhausted(
                    transient_retries=transient_retries
                ) from exc
            transient_retries += 1
            time.sleep(_retry_delay_seconds(exc=exc, attempt=attempt))
            continue
        transport.close()
        return ReviewAttemptResult(result=result, transient_retries=transient_retries)
    raise PersonaProseReviewError("Prose review retry loop ended unexpectedly")


def _checkpoint_path(*, output_dir: Path, persona_hash: str) -> Path:
    return output_dir / "checkpoints" / persona_hash[:2] / f"{persona_hash}.json"


def _is_retryable_transient_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in TRANSIENT_STATUS_CODES
    return isinstance(exc, httpx.TransportError)


def _retry_delay_seconds(*, exc: Exception, attempt: int) -> float:
    retry_after = _retry_after_seconds(exc=exc)
    if retry_after is None:
        delay = 2 ** (attempt - 1)
    else:
        delay = retry_after
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


def _run_proxy_review_adapter(
    *,
    row: dict[str, object],
    candidate_row: dict[str, object],
    changed_facts: dict[str, dict[str, object]],
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    checkpoint_path: Path,
    transport: httpx.BaseTransport,
) -> ProseReviewResult:
    require_runtime_model(REVIEW_MODEL_ENV)
    return run_proxy_review(
        row=row,
        candidate_row=candidate_row,
        changed_facts=changed_facts,
        prompt=prompt,
        config=config,
        budget=budget,
        checkpoint_path=checkpoint_path,
        transport=transport,
    )


if __name__ == "__main__":
    load_repository_environment()
    main()
